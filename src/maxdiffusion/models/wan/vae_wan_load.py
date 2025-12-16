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

from . import vae_wan as vae_lib


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

def _get_vae_key_mapping():
    """Define mapping from PyTorch VAE keys to JAX VAE keys."""

    class Transform(Enum):
        """Transformations for VAE parameters"""

        NONE = None
        TRANSPOSE_2D_CONV = ((2, 3, 1, 0), None)  # For 2D conv: (out, in, h, w) -> (h, w, in, out)
        TRANSPOSE_3D = ((2, 3, 4, 1, 0), None)  # For 3D conv: (out, in, t, h, w) -> (t, h, w, in, out)
        SQUEEZE = (None, (-1,))  # Squeeze to 1D: (C, 1, 1, 1) -> (C,)

    # PyTorch format: (out_channels, in_channels, kernel_size...)
    # JAX format: (kernel_size..., in_channels, out_channels)
    mapping = {
        # Post-quantization conv: 1x1x1 conv
        r"post_quant_conv\.weight": ("conv2.conv.kernel", Transform.TRANSPOSE_3D),
        r"post_quant_conv\.bias": ("conv2.conv.bias", Transform.NONE),
        # Decoder input conv
        r"decoder\.conv_in\.weight": ("decoder.conv_in.conv.kernel", Transform.TRANSPOSE_3D),
        r"decoder\.conv_in\.bias": ("decoder.conv_in.conv.bias", Transform.NONE),
        # Mid block resnets
        r"decoder\.mid_block\.resnets\.0\.norm1\.gamma": ("decoder.mid_block1.norm1.scale", Transform.SQUEEZE),
        r"decoder\.mid_block\.resnets\.0\.conv1\.weight": (
            "decoder.mid_block1.conv1.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.mid_block\.resnets\.0\.conv1\.bias": ("decoder.mid_block1.conv1.conv.bias", Transform.NONE),
        r"decoder\.mid_block\.resnets\.0\.norm2\.gamma": ("decoder.mid_block1.norm2.scale", Transform.SQUEEZE),
        r"decoder\.mid_block\.resnets\.0\.norm2\.bias": ("decoder.mid_block1.norm2.scale", Transform.NONE),
        r"decoder\.mid_block\.resnets\.0\.conv2\.weight": (
            "decoder.mid_block1.conv2.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.mid_block\.resnets\.0\.conv2\.bias": ("decoder.mid_block1.conv2.conv.bias", Transform.NONE),
        r"decoder\.mid_block\.resnets\.1\.norm1\.gamma": ("decoder.mid_block2.norm1.scale", Transform.SQUEEZE),
        r"decoder\.mid_block\.resnets\.1\.conv1\.weight": (
            "decoder.mid_block2.conv1.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.mid_block\.resnets\.1\.conv1\.bias": ("decoder.mid_block2.conv1.conv.bias", Transform.NONE),
        r"decoder\.mid_block\.resnets\.1\.norm2\.gamma": ("decoder.mid_block2.norm2.scale", Transform.SQUEEZE),
        r"decoder\.mid_block\.resnets\.1\.conv2\.weight": (
            "decoder.mid_block2.conv2.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.mid_block\.resnets\.1\.conv2\.bias": ("decoder.mid_block2.conv2.conv.bias", Transform.NONE),
        # Mid attention block
        r"decoder\.mid_block\.attentions\.0\.norm\.gamma": ("decoder.mid_attn.norm.scale", Transform.SQUEEZE),
        r"decoder\.mid_block\.attentions\.0\.to_qkv\.weight": (
            "decoder.mid_attn.qkv.kernel",
            Transform.TRANSPOSE_2D_CONV,
        ),
        r"decoder\.mid_block\.attentions\.0\.to_qkv\.bias": ("decoder.mid_attn.qkv.bias", Transform.NONE),
        r"decoder\.mid_block\.attentions\.0\.proj\.weight": (
            "decoder.mid_attn.proj.kernel",
            Transform.TRANSPOSE_2D_CONV,
        ),
        r"decoder\.mid_block\.attentions\.0\.proj\.bias": ("decoder.mid_attn.proj.bias", Transform.NONE),
        # Up blocks - resnets (pattern for all 4 stages, 3 resnets each)
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.norm1\.gamma": (
            r"decoder.up_blocks_\1.\2.norm1.scale",
            Transform.SQUEEZE,
        ),
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.conv1\.weight": (
            r"decoder.up_blocks_\1.\2.conv1.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.conv1\.bias": (
            r"decoder.up_blocks_\1.\2.conv1.conv.bias",
            Transform.NONE,
        ),
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.norm2\.gamma": (
            r"decoder.up_blocks_\1.\2.norm2.scale",
            Transform.SQUEEZE,
        ),
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.conv2\.weight": (
            r"decoder.up_blocks_\1.\2.conv2.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.conv2\.bias": (
            r"decoder.up_blocks_\1.\2.conv2.conv.bias",
            Transform.NONE,
        ),
        # Skip connections (only in block 1, resnet 0)
        r"decoder\.up_blocks\.1\.resnets\.0\.conv_shortcut\.weight": (
            "decoder.up_blocks_1.0.skip_conv.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.up_blocks\.1\.resnets\.0\.conv_shortcut\.bias": (
            "decoder.up_blocks_1.0.skip_conv.conv.bias",
            Transform.NONE,
        ),
        # Upsamplers for blocks 0, 1, 2 (block 3 has no upsampler)
        # Block 0: Upsample3D (time_conv + spatial_conv)
        r"decoder\.up_blocks\.0\.upsamplers\.0\.time_conv\.weight": (
            "decoder.up_sample_0.time_conv.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.up_blocks\.0\.upsamplers\.0\.time_conv\.bias": (
            "decoder.up_sample_0.time_conv.conv.bias",
            Transform.NONE,
        ),
        r"decoder\.up_blocks\.0\.upsamplers\.0\.resample\.1\.weight": (
            "decoder.up_sample_0.spatial_conv.kernel",
            Transform.TRANSPOSE_2D_CONV,
        ),
        r"decoder\.up_blocks\.0\.upsamplers\.0\.resample\.1\.bias": (
            "decoder.up_sample_0.spatial_conv.bias",
            Transform.NONE,
        ),
        # Block 1: Upsample3D (time_conv + spatial_conv)
        r"decoder\.up_blocks\.1\.upsamplers\.0\.time_conv\.weight": (
            "decoder.up_sample_1.time_conv.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.up_blocks\.1\.upsamplers\.0\.time_conv\.bias": (
            "decoder.up_sample_1.time_conv.conv.bias",
            Transform.NONE,
        ),
        r"decoder\.up_blocks\.1\.upsamplers\.0\.resample\.1\.weight": (
            "decoder.up_sample_1.spatial_conv.kernel",
            Transform.TRANSPOSE_2D_CONV,
        ),
        r"decoder\.up_blocks\.1\.upsamplers\.0\.resample\.1\.bias": (
            "decoder.up_sample_1.spatial_conv.bias",
            Transform.NONE,
        ),
        # Block 2: Upsample2D (conv only, no time_conv)
        r"decoder\.up_blocks\.2\.upsamplers\.0\.resample\.1\.weight": (
            "decoder.up_sample_2.conv.kernel",
            Transform.TRANSPOSE_2D_CONV,
        ),
        r"decoder\.up_blocks\.2\.upsamplers\.0\.resample\.1\.bias": ("decoder.up_sample_2.conv.bias", Transform.NONE),
        # Output layers
        r"decoder\.norm_out\.gamma": ("decoder.norm_out.scale", Transform.SQUEEZE),
        r"decoder\.conv_out\.weight": ("decoder.conv_out.conv.kernel", Transform.TRANSPOSE_3D),
        r"decoder\.conv_out\.bias": ("decoder.conv_out.conv.bias", Transform.NONE),
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


def create_vae_decoder_from_safe_tensors(
    file_dir: str,
    mesh: jax.sharding.Mesh | None = None,
) -> vae_lib.WanVAEDecoder:
    """
    Load Wan-VAE decoder from safetensors checkpoint.

    Args:
        file_dir: Directory containing .safetensors files or path to VAE directory
        mesh: Optional JAX mesh for sharding

    Returns:
        WanVAEDecoder with loaded weights
    """
    # Check if file_dir is the model root or VAE subdirectory
    file_path = epath.Path(file_dir).expanduser()
    vae_path = file_path / "vae"

    if vae_path.exists():
        # Look in vae subdirectory
        files = list(vae_path.glob("*.safetensors"))
    else:
        # Look in provided directory
        files = list(file_path.glob("*.safetensors"))

    if not files:
        raise ValueError(f"No safetensors found in {file_dir} or {file_dir}/vae")

    print(f"Found {len(files)} VAE safetensors file(s)")

    # Create VAE decoder structure
    vae_decoder = nnx.eval_shape(lambda: vae_lib.WanVAEDecoder(rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(vae_decoder)
    state_dict = abs_state.to_pure_dict()

    # Setup sharding if mesh provided
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_vae_key_mapping()
    conversion_errors = []
    loaded_keys = []
    skipped_keys = []

    for f in files:
        print(f"Loading VAE weights from {f.name}...")
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)

                if jax_key is None:
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

    print(f"Loaded {len(loaded_keys)} VAE weight tensors")
    print(f"Skipped {len(skipped_keys)} weight tensors")

    if conversion_errors:
        print(f"\nWarning: {len(conversion_errors)} conversion errors occurred:")
        for err in conversion_errors:  # Show first 10 errors
            print(f"  {err}")
        # if len(conversion_errors) > 10:
        #     print(f"  ... and {len(conversion_errors) - 10} more")

    if len(loaded_keys) == 0:
        raise ValueError("No VAE weights were loaded! Check the checkpoint structure and key mapping.")

    gc.collect()
    return nnx.merge(graph_def, state_dict)

__all__ = [
    "create_vae_decoder_from_safe_tensors",
]
