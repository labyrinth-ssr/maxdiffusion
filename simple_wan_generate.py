"""
Simple WAN 2.1-T2V-1.3B Video Generation Script
Uses custom transformer, UMT5 encoder, and VAE
No config files - all parameters hardcoded
"""

import argparse
import time
import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer
import numpy as np

# Import custom models
from src.maxdiffusion.models.wan.transformers import my_transformer_wan_load, my_transformer_wan
from src.maxdiffusion.models.wan import umt5_load, umt5
from src.maxdiffusion.models.wan import vae_wan_load, vae_wan
from src.maxdiffusion.schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler


def encode_text(prompt, negative_prompt, tokenizer, text_encoder, max_length=512):
    """Encode text prompts to embeddings."""
    print(f"Encoding prompt: {prompt}")

    # Tokenize
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )

    # Encode with custom UMT5 (on CPU to save memory)
    with jax.default_device(jax.devices('cpu')[0]):
        input_ids = jnp.array(text_inputs.input_ids)
        attention_mask = jnp.array(text_inputs.attention_mask)

        # Get embeddings
        prompt_embeds = text_encoder(input_ids, attention_mask, deterministic=True).last_hidden_state
        prompt_embeds = jnp.array(prompt_embeds)

    # Encode negative prompt
    neg_inputs = tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )

    with jax.default_device(jax.devices('cpu')[0]):
        neg_ids = jnp.array(neg_inputs.input_ids)
        neg_mask = jnp.array(neg_inputs.attention_mask)

        negative_embeds = text_encoder(neg_ids, neg_mask, deterministic=True).last_hidden_state
        negative_embeds = jnp.array(negative_embeds)

    return prompt_embeds, negative_embeds


def generate_latents(
    transformer,
    prompt_embeds,
    negative_embeds,
    scheduler,
    scheduler_state,
    num_frames=81,
    height=480,
    width=832,
    num_steps=30,
    guidance_scale=5.0,
    seed=42,
):
    """Generate video latents using transformer."""
    print(f"\nGenerating latents: {num_frames} frames @ {height}x{width}")
    print(f"Steps: {num_steps}, Guidance: {guidance_scale}")

    # Calculate latent dimensions
    vae_temporal_factor = 4  # 2^2 from [True, True]
    vae_spatial_factor = 8   # 2^3 from [False, True, True]

    num_latent_frames = (num_frames - 1) // vae_temporal_factor + 1
    latent_h = height // vae_spatial_factor
    latent_w = width // vae_spatial_factor

    # Initialize random latents (channel-first: B, C, T, H, W)
    rng = jax.random.key(seed)
    latents = jax.random.normal(
        rng,
        shape=(1, 16, num_latent_frames, latent_h, latent_w),
        dtype=jnp.float32
    )

    print(f"Initial latents shape: {latents.shape}")

    # Set timesteps
    scheduler_state = scheduler.set_timesteps(
        scheduler_state,
        num_inference_steps=num_steps,
        shape=latents.shape
    )

    # Concatenate prompts for CFG
    prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_embeds], axis=0)

    # Denoising loop
    graphdef, state, rest_of_state = nnx.split(transformer, nnx.Param, ...)

    for step in range(num_steps):
        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]

        # Concatenate latents for CFG
        latents_combined = jnp.concatenate([latents, latents], axis=0)
        timestep = jnp.broadcast_to(t, (latents_combined.shape[0],))

        # Forward pass
        transformer_merged = nnx.merge(graphdef, state, rest_of_state)
        noise_pred = transformer_merged(
            hidden_states=latents_combined,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds_combined,
        )

        # Apply CFG
        noise_cond, noise_uncond = jnp.split(noise_pred, 2, axis=0)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Scheduler step
        latents, scheduler_state = scheduler.step(
            scheduler_state,
            noise_pred,
            t,
            latents
        ).to_tuple()

        if step % 10 == 0:
            print(f"Step {step}/{num_steps}")

    print(f"Final latents shape: {latents.shape}")
    return latents


def decode_video(latents, vae_decoder):
    """Decode latents to video using VAE."""
    print("\nDecoding latents to video...")

    # Denormalize latents (CRITICAL for quality!)
    latents_mean = jnp.array(vae_decoder.latents_mean).reshape(1, 16, 1, 1, 1)
    latents_std = 1.0 / jnp.array(vae_decoder.latents_std).reshape(1, 16, 1, 1, 1)
    latents = latents / latents_std + latents_mean
    latents = latents.astype(jnp.float32)

    print(f"Denormalized latents: min={latents.min():.4f}, max={latents.max():.4f}")

    # Decode (VAE expects channel-last: B, T, H, W, C)
    latents_channel_last = jnp.transpose(latents, (0, 2, 3, 4, 1))
    video = vae_decoder.decode(latents_channel_last)[0]

    print(f"Decoded video shape: {video.shape}")
    return video


def postprocess_video(video):
    """Convert video to uint8 format."""
    # Ensure channel-last: (B, T, H, W, C)
    if video.shape[1] == 3:
        video = jnp.transpose(video, (0, 2, 3, 4, 1))

    # Normalize to [0, 255]
    video = (video + 1.0) / 2.0
    video = jnp.clip(video, 0.0, 1.0)
    video = (video * 255.0).astype(jnp.uint8)

    return video


def save_video(video, output_path="output.mp4", fps=24):
    """Save video to file."""
    try:
        import imageio
        print(f"\nSaving video to {output_path}...")

        # video shape: (1, T, H, W, 3)
        video_frames = np.array(video[0])  # Remove batch dim

        # Save with imageio
        imageio.mimwrite(output_path, video_frames, fps=fps, codec='libx264', quality=8)
        print(f"✓ Video saved successfully!")
    except ImportError:
        print("Warning: imageio not installed. Saving as numpy array instead.")
        np.save(output_path.replace('.mp4', '.npy'), video)


def main():
    parser = argparse.ArgumentParser(description="Simple WAN Video Generation")
    parser.add_argument("--prompt", type=str, default="A cat playing piano", help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality", help="Negative prompt")
    parser.add_argument("--num_frames", type=int, default=41, help="Number of frames")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=720, help="Video width")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=118445, help="Random seed")
    parser.add_argument("--model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        help="Path to pretrained model")
    parser.add_argument("--output", type=str, default="generated_video.mp4", help="Output video path")
    args = parser.parse_args()

    print("=" * 70)
    print("WAN 2.1-T2V-1.3B Video Generation")
    print("=" * 70)

    start_time = time.time()

    # 1. Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        subfolder="tokenizer",
    )

    # 2. Load custom UMT5 encoder (on CPU)
    print("\n[2/5] Loading custom UMT5 encoder...")
    with jax.default_device(jax.devices('cpu')[0]):
        text_encoder = umt5_load.create_t5_encoder_from_safe_tensors(
            args.model_path,
            mesh=None,
        )
    print(f"✓ UMT5 encoder loaded on CPU")

    # 3. Load custom transformer
    print("\n[3/5] Loading custom WAN transformer...")
    config = my_transformer_wan.TransformerWanModelConfig()
    transformer = my_transformer_wan_load.create_model_from_safe_tensors(
        args.model_path,
        config,
        mesh=None,
    )
    # Wrap in adapter
    transformer = my_transformer_wan.WanModelAdapter(pretrained_model=transformer)
    print(f"✓ Transformer loaded: {config.num_layers} layers, {config.hidden_dim} dim")

    # 4. Load custom VAE
    print("\n[4/5] Loading custom VAE decoder...")
    vae_decoder = vae_wan_load.create_vae_decoder_from_safe_tensors(
        args.model_path,
        mesh=None,
    )
    wan_vae = vae_wan.WanVAEAdapter(vae_decoder=vae_decoder)
    print(f"✓ VAE decoder loaded")

    # 5. Load scheduler
    print("\n[5/5] Loading scheduler...")
    scheduler, scheduler_state = FlaxUniPCMultistepScheduler.from_pretrained(
        args.model_path,
        subfolder="scheduler",
        flow_shift=3.0,  # 3.0 for 480p, 5.0 for 720p
    )
    print(f"✓ Scheduler loaded")

    print(f"\n{'='*70}")
    print("Starting Generation")
    print(f"{'='*70}")

    # Encode text
    prompt_embeds, negative_embeds = encode_text(
        args.prompt,
        args.negative_prompt,
        tokenizer,
        text_encoder,
        max_length=512,
    )

    # Generate latents
    latents = generate_latents(
        transformer,
        prompt_embeds,
        negative_embeds,
        scheduler,
        scheduler_state,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

    # Decode to video
    video = decode_video(latents, wan_vae)

    # Postprocess
    video = postprocess_video(video)

    # Save
    save_video(video, args.output, fps=24)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("✓ Generation Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.2f}s")
    print(f"FPS: {args.num_frames / total_time:.2f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()