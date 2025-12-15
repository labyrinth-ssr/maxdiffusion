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
import re
from enum import Enum

import jax
import jax.numpy as jnp
import safetensors
from etils import epath
from flax import nnx

from . import my_transformer_wan as model_lib

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


def _get_dit_mapping(cfg: model_lib.TransformerWanModelConfig):
    class Transform(Enum):
        """Transformations for model parameters"""

        NONE = None
        TRANSPOSE = ((1, 0), None)  # For linear layers: (out, in) -> (in, out)
        TRANSPOSE_CONV = ((2, 3, 4, 1, 0), None)  # For 3D conv: (out, in, t, h, w) -> (t, h, w, in, out)

    mapping = {
        # Patch embedding (input projection)
        r"patch_embedding\.weight": ("patch_embed.kernel", Transform.TRANSPOSE_CONV),
        r"patch_embedding\.bias": ("patch_embed.bias", Transform.NONE),
        # Time embedder - Sequential uses integer indices (0, 1, 2), not layers_0
        r"condition_embedder\.time_embedder\.linear_1\.weight": (
            "time_embed.time_embedding.layers.0.kernel",
            Transform.TRANSPOSE,
        ),
        r"condition_embedder\.time_embedder\.linear_1\.bias": (
            "time_embed.time_embedding.layers.0.bias",
            Transform.NONE,
        ),
        r"condition_embedder\.time_embedder\.linear_2\.weight": (
            "time_embed.time_embedding.layers.2.kernel",
            Transform.TRANSPOSE,
        ),
        r"condition_embedder\.time_embedder\.linear_2\.bias": (
            "time_embed.time_embedding.layers.2.bias",
            Transform.NONE,
        ),
        r"condition_embedder\.time_proj\.weight": ("time_embed.time_projection.layers.1.kernel", Transform.TRANSPOSE),
        r"condition_embedder\.time_proj\.bias": ("time_embed.time_projection.layers.1.bias", Transform.NONE),
        # Text embedder (projects UMT5 embeddings to hidden dim)
        r"condition_embedder\.text_embedder\.linear_1\.weight": ("text_proj.layers.0.kernel", Transform.TRANSPOSE),
        r"condition_embedder\.text_embedder\.linear_1\.bias": ("text_proj.layers.0.bias", Transform.NONE),
        r"condition_embedder\.text_embedder\.linear_2\.weight": ("text_proj.layers.2.kernel", Transform.TRANSPOSE),
        r"condition_embedder\.text_embedder\.linear_2\.bias": ("text_proj.layers.2.bias", Transform.NONE),
        # Transformer blocks - Self attention (attn1)
        r"blocks\.([0-9]+)\.attn1\.norm_q\.weight": (r"blocks.\1.self_attn.q_norm.scale", Transform.NONE),
        r"blocks\.([0-9]+)\.attn1\.norm_k\.weight": (r"blocks.\1.self_attn.k_norm.scale", Transform.NONE),
        r"blocks\.([0-9]+)\.attn1\.to_q\.weight": (r"blocks.\1.self_attn.q_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn1\.to_q\.bias": (r"blocks.\1.self_attn.q_proj.bias", Transform.NONE),
        r"blocks\.([0-9]+)\.attn1\.to_k\.weight": (r"blocks.\1.self_attn.k_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn1\.to_k\.bias": (r"blocks.\1.self_attn.k_proj.bias", Transform.NONE),
        r"blocks\.([0-9]+)\.attn1\.to_v\.weight": (r"blocks.\1.self_attn.v_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn1\.to_v\.bias": (r"blocks.\1.self_attn.v_proj.bias", Transform.NONE),
        r"blocks\.([0-9]+)\.attn1\.to_out\.0\.weight": (r"blocks.\1.self_attn.out_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn1\.to_out\.0\.bias": (r"blocks.\1.self_attn.out_proj.bias", Transform.NONE),
        # Transformer blocks - Cross attention (attn2)
        # Note: CrossAttention only has q_norm, not k_norm; norm_k is skipped
        r"blocks\.([0-9]+)\.attn2\.norm_q\.weight": (r"blocks.\1.cross_attn.q_norm.scale", Transform.NONE),
        r"blocks\.([0-9]+)\.attn2\.norm_k\.weight": (r"blocks.\1.cross_attn.k_norm.scale", Transform.NONE),
        r"blocks\.([0-9]+)\.attn2\.to_q\.weight": (r"blocks.\1.cross_attn.q_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn2\.to_q\.bias": (r"blocks.\1.cross_attn.q_proj.bias", Transform.NONE),
        # Note: to_k and to_v need special handling - they're fused into kv_proj in JAX
        # See _load_fused_kv_weights() below
        r"blocks\.([0-9]+)\.attn2\.to_out\.0\.weight": (r"blocks.\1.cross_attn.out_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn2\.to_out\.0\.bias": (r"blocks.\1.cross_attn.out_proj.bias", Transform.NONE),
        # Transformer blocks - Feed forward (Sequential creates 'layers' dict with 0, 2 keys)
        r"blocks\.([0-9]+)\.ffn\.net\.0\.proj\.weight": (r"blocks.\1.mlp.layers.0.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.ffn\.net\.0\.proj\.bias": (r"blocks.\1.mlp.layers.0.bias", Transform.NONE),
        r"blocks\.([0-9]+)\.ffn\.net\.2\.weight": (r"blocks.\1.mlp.layers.2.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.ffn\.net\.2\.bias": (r"blocks.\1.mlp.layers.2.bias", Transform.NONE),
        # Transformer blocks - Norm and modulation
        r"blocks\.([0-9]+)\.norm2\.weight": (r"blocks.\1.norm2.scale", Transform.NONE),
        r"blocks\.([0-9]+)\.norm2\.bias": (r"blocks.\1.norm2.bias", Transform.NONE),
        r"blocks\.([0-9]+)\.scale_shift_table": (r"blocks.\1.scale_shift_table", Transform.NONE),
        # Output projection
        r"scale_shift_table": ("final_layer.scale_shift_table", Transform.NONE),
        r"proj_out\.weight": ("final_layer.linear.kernel", Transform.TRANSPOSE),
        r"proj_out\.bias": ("final_layer.linear.bias", Transform.NONE),
        r"norm_out\.weight": ("final_layer.norm.scale", Transform.NONE),
    }

    return mapping

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

def create_model_from_safe_tensors(
    file_dir: str,
    cfg: model_lib.TransformerWanModelConfig,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.Wan2DiT:
    """
    Load Wan2.1-T2V-1.3B DiT model from safetensors checkpoint.

    Args:
        file_dir: Directory containing .safetensors files or path to transformer directory
        cfg: Model configuration
        mesh: Optional JAX mesh for sharding
        load_transformer_only: If True, only load transformer weights (not VAE/text encoder)

    Returns:
        Wan2DiT model with loaded weights
    """
    # Check if file_dir is the model root or transformer subdirectory
    file_path = epath.Path(file_dir).expanduser()
    transformer_path = file_path / "transformer"

    if transformer_path.exists():
        # Look in transformer subdirectory
        files = sorted(list(transformer_path.glob("diffusion_pytorch_model-*.safetensors")))
    else:
        # Look in provided directory
        files = sorted(list(file_path.glob("diffusion_pytorch_model-*.safetensors")))
        if not files:
            files = sorted(list(file_path.glob("*.safetensors")))

    if not files:
        raise ValueError(f"No safetensors found in {file_dir} or {file_dir}/transformer")

    print(f"Found {len(files)} DiT transformer safetensors file(s)")

    # Create model structure
    wan2_dit = nnx.eval_shape(lambda: model_lib.Wan2DiT(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(wan2_dit)
    state_dict = abs_state.to_pure_dict()

    # Setup sharding if mesh provided
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_dit_mapping(cfg)
    conversion_errors = []
    loaded_keys = []
    skipped_keys = []

    # Collect K/V weights for fusion into kv_proj
    kv_weights = {}  # {block_idx: {'k_weight': ..., 'k_bias': ..., 'v_weight': ..., 'v_bias': ...}}

    for f in files:
        print(f"Loading weights from {f.name}...")
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                # Special handling for cross-attention K/V fusion
                kv_match = re.match(r"blocks\.([0-9]+)\.attn2\.to_([kv])\.(weight|bias)", torch_key)
                if kv_match:
                    block_idx = int(kv_match.group(1))
                    kv_type = kv_match.group(2)  # 'k' or 'v'
                    param_type = kv_match.group(3)  # 'weight' or 'bias'

                    if block_idx not in kv_weights:
                        kv_weights[block_idx] = {}
                    kv_weights[block_idx][f"{kv_type}_{param_type}"] = tensor
                    loaded_keys.append(torch_key)
                    continue

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)

                if jax_key is None:
                    # Skip keys not in our mapping (e.g., VAE, text encoder, attn2.norm_k)
                    skipped_keys.append(torch_key)
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

    # Fuse collected K/V weights into kv_proj
    import jax.numpy as jnp

    for block_idx, weights in kv_weights.items():
        if all(k in weights for k in ["k_weight", "k_bias", "v_weight", "v_bias"]):
            # Transpose and concatenate: (out, in) -> (in, out) then concat -> (in, 2*out)
            k_weight = weights["k_weight"].T  # (in, out)
            v_weight = weights["v_weight"].T  # (in, out)
            kv_kernel = jnp.concatenate([k_weight, v_weight], axis=1)  # (in, 2*out)

            kv_bias = jnp.concatenate([weights["k_bias"], weights["v_bias"]])  # (2*out,)

            # Assign to state dict
            state_dict["blocks"][block_idx]["cross_attn"]["kv_proj"]["kernel"] = jax.device_put(kv_kernel)
            state_dict["blocks"][block_idx]["cross_attn"]["kv_proj"]["bias"] = jax.device_put(kv_bias)

    print(f"Loaded {len(loaded_keys)} weight tensors")
    print(f"Skipped {len(skipped_keys)} weight tensors (VAE/text encoder/attn2.norm_k)")

    state_dict = jax.tree_util.tree_map_with_path(
        lambda path, x: cast_with_exclusion(path, x, dtype_to_cast=cfg.weights_dtype), state_dict
    )

    if conversion_errors:
        print(f"\n Warning: {len(conversion_errors)} conversion errors occurred:")
        for err in conversion_errors:  # Show first 5 errors
            print(f"  {err}")
        # if len(conversion_errors) > 5:
        #     print(f"  ... and {len(conversion_errors) - 5} more")

    gc.collect()
    return nnx.merge(graph_def, state_dict)

__all__ = [
    "create_model_from_safe_tensors",
]
