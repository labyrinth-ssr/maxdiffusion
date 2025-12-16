# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight loading utilities for Wan2.1-T2V-1.3B model."""

import gc
import os
import re
from enum import Enum

import jax
import jax.numpy as jnp
import safetensors
from etils import epath
from huggingface_hub import snapshot_download
from flax import nnx

from . import umt5 as t5_lib


def cast_with_exclusion(path, x, dtype_to_cast):
    """
    Casts arrays to dtype_to_cast, but keeps params from any 'norm' layer in float32.
    """

    exclusion_keywords = [
        "norm",  # For all LayerNorm/GroupNorm layers
        "time_embed",  # The entire time conditioning module
        "text_proj",  # The entire text conditioning module
        "scale_shift_table",  # Catches both the final and the AdaLN tables
    ]

    path_str = ".".join(str(k.key) if isinstance(k, jax.tree_util.DictKey) else str(k) for k in path)

    if any(keyword in path_str.lower() for keyword in exclusion_keywords):
        # Keep LayerNorm/GroupNorm weights and biases in full precision
        return x.astype(jnp.float32)
    else:
        # Cast everything else to dtype_to_cast
        return x.astype(dtype_to_cast)

def _torch_key_to_jax_key(mapping, source_key):
    """Convert a PyTorch/Diffusers key to JAX key with transform info."""
    subs = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) == 0:
        # Key not found in mapping, might be OK (e.g., VAE weights)
        return None, None
    if len(subs) > 1:
        raise ValueError(f"Multiple patterns matched for key {source_key}: {subs}")
    return subs[0]

def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict=None):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None and transform.value is not None:
            permute, reshape = transform.value
            if reshape is not None:
                tensor = tensor.reshape(reshape)
            if permute:
                tensor = tensor.transpose(permute)

        if key not in state_dict:
            raise KeyError(f"Key {key} not found in state_dict. Available keys: {list(state_dict.keys())[:10]}...")

        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}")

        # Assign with or without sharding
        if sharding_dict is not None and key in sharding_dict:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None and key in sharding_dict else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)

def _stoi(s):
    """Convert string to int if possible, otherwise return string."""
    try:
        return int(s)
    except ValueError:
        return s

def _get_t5_key_mapping():
    """Define mapping from HuggingFace UMT5 keys to JAX UMT5 keys."""

    class Transform(Enum):
        """Transformations for UMT5 parameters"""

        NONE = None
        TRANSPOSE = ((1, 0), None)  # For linear layers: (out, in) -> (in, out)

    # UMT5/UMT5 uses standard HuggingFace naming
    mapping = {
        # Shared token embeddings
        r"shared\.weight": ("encoder.token_embedding.embedding", Transform.NONE),
        r"encoder\.embed_tokens\.weight": ("encoder.token_embedding.embedding", Transform.NONE),
        # Encoder blocks - Self attention
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.q\.weight": (
            r"encoder.blocks.\1.attn.q.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.k\.weight": (
            r"encoder.blocks.\1.attn.k.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.v\.weight": (
            r"encoder.blocks.\1.attn.v.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.o\.weight": (
            r"encoder.blocks.\1.attn.o.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.relative_attention_bias\.weight": (
            r"encoder.blocks.\1.pos_embedding.embedding.embedding",
            Transform.NONE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.layer_norm\.weight": (r"encoder.blocks.\1.norm1.weight", Transform.NONE),
        # Encoder blocks - Feed forward
        r"encoder\.block\.([0-9]+)\.layer\.1\.DenseReluDense\.wi_0\.weight": (
            r"encoder.blocks.\1.ffn.gate.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.1\.DenseReluDense\.wi_1\.weight": (
            r"encoder.blocks.\1.ffn.fc1.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.1\.DenseReluDense\.wo\.weight": (
            r"encoder.blocks.\1.ffn.fc2.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.1\.layer_norm\.weight": (r"encoder.blocks.\1.norm2.weight", Transform.NONE),
        # Final layer norm
        r"encoder\.final_layer_norm\.weight": ("encoder.norm.weight", Transform.NONE),
    }

    return mapping

def create_t5_encoder_from_safe_tensors(
    file_dir: str,
    mesh: jax.sharding.Mesh | None = None,
    is_sf: bool = True,
    config: t5_lib.T5Config | None = None,
) -> t5_lib.T5EncoderModel:
    """
    Load UMT5 encoder from safetensors checkpoint.

    Args:
        file_dir: Directory containing .safetensors files or path to text_encoder directory
        mesh: Optional JAX mesh for sharding
        is_sf: Whether to load from safetensors (True) or PyTorch checkpoint (False)
        config: T5Config to use. If None, defaults to UMT5-XXL

    Returns:
        T5EncoderModel with loaded weights
    """
    # Use provided config or default to UMT5-XXL
    if config is None:
        config = t5_lib.T5Config.umt5_xxl()

    t5_encoder = nnx.eval_shape(lambda: t5_lib.T5EncoderModel(config, rngs=nnx.Rngs(params=0, dropout=0)))
    graph_def, abs_state = nnx.split(t5_encoder)
    state_dict = abs_state.to_pure_dict()

    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_t5_key_mapping()
    conversion_errors = []
    loaded_keys = []
    skipped_keys = []

    # Check if input is local directory or HuggingFace model ID
    if os.path.isdir(file_dir):
        # Local directory: use directly
        print(f"Loading VAE from local directory: {file_dir}")
        local_dir = file_dir
    else:
        # HuggingFace model ID: download entire repo
        print(f"Downloading VAE from HuggingFace: {file_dir}")
        local_dir = snapshot_download(
            repo_id=file_dir,
            allow_patterns=["text_encoder/*.safetensors"],  # Only download VAE weights
            cache_dir=None,  # Use default cache: ~/.cache/huggingface/hub/
        )
        print(f"Downloaded to: {local_dir}")


    # Check if file_dir is the model root or text_encoder subdirectory
    file_path = epath.Path(local_dir).expanduser()
    text_encoder_path = file_path / "text_encoder"

    def load_pytorch_weights(file_dir):
        from transformers import UMT5ForConditionalGeneration

        model = UMT5ForConditionalGeneration.from_pretrained(file_dir)
        encoder_state = {k: v for k, v in model.state_dict().items() if k.startswith("encoder.")}
        return encoder_state

    if is_sf:
        if text_encoder_path.exists():
            files = sorted(list(text_encoder_path.glob("model-*.safetensors")))
        else:
            files = sorted(list(file_path.glob("*.safetensors")))
        if not files:
            raise ValueError(f"No safetensors found in {file_dir} or {file_dir}/text_encoder")
        print(f"Found {len(files)} UMT5 encoder safetensors file(s)")

        for f in files:
            print(f"Loading UMT5 weights from {f.name}...")
            with safetensors.safe_open(f, framework="numpy") as sf:
                for torch_key in sf.keys():
                    tensor = sf.get_tensor(torch_key)

                    jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)

                    if jax_key is None:
                        # Skip keys not in our mapping
                        skipped_keys.append(torch_key)
                        # print(f"{torch_key} is not mapped")
                        continue

                    keys = [_stoi(k) for k in jax_key.split(".")]
                    try:
                        _assign_weights(keys, tensor, state_dict, torch_key, transform, sharding)
                        loaded_keys.append(torch_key)
                    except Exception as e:
                        full_jax_key = ".".join([str(k) for k in keys])
                        conversion_errors.append(
                            f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                        )
            gc.collect()
    else:
        print(f"Loading UMT5 weights from PyTorch checkpoint in {file_dir}...")
        pt_state = load_pytorch_weights(file_dir)
        for torch_key, tensor in pt_state.items():
            jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)

            if jax_key is None:
                # Skip keys not in our mapping
                skipped_keys.append(torch_key)
                # print(f"{torch_key} is not mapped")
                continue

            keys = [_stoi(k) for k in jax_key.split(".")]
            try:
                _assign_weights(keys, tensor.numpy(), state_dict, torch_key, transform, sharding)
                loaded_keys.append(torch_key)
            except Exception as e:
                full_jax_key = ".".join([str(k) for k in keys])
                conversion_errors.append(f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}")
        gc.collect()

    print(f"Loaded {len(loaded_keys)} UMT5 weight tensors")
    print(f"Skipped {len(skipped_keys)} weight tensors")

    if conversion_errors:
        print(f"\nWarning: {len(conversion_errors)} conversion errors occurred:")
        for err in conversion_errors:  # Show first 10 errors
            print(f"  {err}")
        # if len(conversion_errors) > 10:
        #     print(f"  ... and {len(conversion_errors) - 10} more")

    if len(loaded_keys) == 0:
        raise ValueError("No UMT5 weights were loaded! Check the checkpoint structure and key mapping.")

    gc.collect()
    return nnx.merge(graph_def, state_dict)

__all__ = [
    "create_t5_encoder_from_safe_tensors",
]
