"""
Simplified WAN Pipeline for Text-to-Video Generation
Minimal implementation for wan2.1-t2v-1.3b
"""

from typing import List, Optional, Union
import jax
import jax.numpy as jnp
from flax import nnx
from transformers import T5Tokenizer, FlaxT5EncoderModel

from ...schedulers.flow_match_euler_discrete_scheduler_flax import FlowMatchEulerDiscreteSchedulerFlax


class SimpleWanPipeline:
    """
    Simplified pipeline for WAN text-to-video generation.

    Usage:
        # Initialize with your models
        pipeline = SimpleWanPipeline(
            transformer=your_transformer,
            vae=your_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler
        )

        # Generate video
        video = pipeline(
            prompt="A cat playing piano",
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=50
        )
    """

    def __init__(
        self,
        transformer,  # Your WAN transformer model
        vae,  # Your VAE model (with decode method)
        text_encoder,  # T5 encoder
        tokenizer,  # T5 tokenizer
        scheduler,  # Flow matching scheduler
        seed: int = 0,
    ):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.seed = seed

        # Get VAE scale factors
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        transformer,
        vae,
        seed: int = 0,
    ):
        """
        Load tokenizer, text encoder, and scheduler from pretrained checkpoint.
        Use your own transformer and VAE.
        """
        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_path,
            subfolder="tokenizer",
        )

        # Load text encoder
        text_encoder = FlaxT5EncoderModel.from_pretrained(
            pretrained_model_path,
            subfolder="text_encoder",
            dtype=jnp.bfloat16,
        )

        # Load scheduler
        scheduler = FlowMatchEulerDiscreteSchedulerFlax.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler",
        )

        return cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            seed=seed,
        )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_sequence_length: int = 512,
    ):
        """Encode text prompt to embeddings."""
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="np",
        )

        # Get prompt embeddings
        prompt_embeds = self.text_encoder(
            input_ids=jnp.array(text_inputs.input_ids),
            attention_mask=jnp.array(text_inputs.attention_mask),
        )[0]

        # Handle negative prompt
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            negative_text_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="np",
            )

            negative_prompt_embeds = self.text_encoder(
                input_ids=jnp.array(negative_text_inputs.input_ids),
                attention_mask=jnp.array(negative_text_inputs.attention_mask),
            )[0]
        else:
            # Use empty prompt as negative
            negative_prompt_embeds = jnp.zeros_like(prompt_embeds)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        seed: Optional[int] = None,
    ):
        """Initialize random latents for generation."""
        if seed is None:
            seed = self.seed

        rng = jax.random.key(seed)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (
            batch_size,
            16,  # num_channels_latents for WAN
            num_latent_frames,
            latent_height,
            latent_width,
        )

        latents = jax.random.normal(rng, shape=shape, dtype=jnp.float32)
        return latents

    def denoise_latents(
        self,
        latents: jnp.ndarray,
        prompt_embeds: jnp.ndarray,
        negative_prompt_embeds: jnp.ndarray,
        num_inference_steps: int,
        guidance_scale: float,
    ):
        """Run denoising loop with transformer."""
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Prepare for classifier-free guidance
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            latents = jnp.concatenate([latents, latents], axis=0)
            prompt_embeds_combined = jnp.concatenate([negative_prompt_embeds, prompt_embeds], axis=0)
        else:
            prompt_embeds_combined = prompt_embeds

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Predict noise
            timestep = jnp.array([t] * latents.shape[0], dtype=jnp.int32)

            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds_combined,
            )

            # Apply classifier-free guidance
            if do_cfg:
                noise_uncond, noise_cond = jnp.split(noise_pred, 2, axis=0)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                latents_single = jnp.split(latents, 2, axis=0)[0]
            else:
                latents_single = latents

            # Scheduler step
            latents_single = self.scheduler.step(
                noise_pred,
                t,
                latents_single,
            ).prev_sample

            # Update latents for next iteration
            if do_cfg:
                latents = jnp.concatenate([latents_single, latents_single], axis=0)
            else:
                latents = latents_single

        # Return final latents (without duplication)
        if do_cfg:
            latents = jnp.split(latents, 2, axis=0)[0]

        return latents

    def decode_latents(self, latents: jnp.ndarray):
        """Decode latents to video frames using VAE."""
        # Denormalize latents
        latents_mean = jnp.array(self.vae.latents_mean).reshape(1, self.vae.z_dim, 1, 1, 1)
        latents_std = jnp.array(self.vae.latents_std).reshape(1, self.vae.z_dim, 1, 1, 1)
        latents = latents / latents_std + latents_mean

        # Decode with VAE
        if hasattr(self.vae, 'decode'):
            video = self.vae.decode(latents, cache=None)[0]
        else:
            video = self.vae(latents)

        return video

    def postprocess_video(self, video: jnp.ndarray):
        """Convert video from [-1, 1] to [0, 255] uint8."""
        # video shape: (batch, time, height, width, channels) or (batch, channels, time, height, width)

        # Ensure channel-last format
        if video.shape[1] == 3:  # (B, C, T, H, W) -> (B, T, H, W, C)
            video = jnp.transpose(video, (0, 2, 3, 4, 1))

        # Normalize to [0, 1]
        video = (video + 1.0) / 2.0
        video = jnp.clip(video, 0.0, 1.0)

        # Convert to uint8
        video = (video * 255.0).astype(jnp.uint8)

        return video

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        max_sequence_length: int = 512,
        seed: Optional[int] = None,
        return_latents: bool = False,
    ):
        """
        Generate video from text prompt.

        Args:
            prompt: Text prompt(s) for generation
            negative_prompt: Negative text prompt(s)
            height: Video height (default: 480)
            width: Video width (default: 832)
            num_frames: Number of frames (default: 81)
            num_inference_steps: Number of denoising steps (default: 50)
            guidance_scale: Classifier-free guidance scale (default: 5.0)
            max_sequence_length: Max tokens for text encoding (default: 512)
            seed: Random seed (if None, uses pipeline seed)
            return_latents: If True, return latents instead of decoded video

        Returns:
            video: Generated video as numpy array (B, T, H, W, 3) uint8, or latents if return_latents=True
        """
        # Validate num_frames
        if (num_frames - 1) % self.vae_scale_factor_temporal != 0:
            num_frames = (num_frames // self.vae_scale_factor_temporal) * self.vae_scale_factor_temporal + 1
            print(f"Adjusted num_frames to {num_frames}")

        # Determine batch size
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        # 1. Encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
        )

        # 2. Prepare latents
        latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
        )

        # 3. Denoise latents
        latents = self.denoise_latents(
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        if return_latents:
            return latents

        # 4. Decode latents
        video = self.decode_latents(latents)

        # 5. Postprocess
        video = self.postprocess_video(video)

        return video


# Example usage
"""
from simple_wan_pipeline import SimpleWanPipeline

# Load your models
transformer = ...  # Your WAN transformer
vae = ...  # Your VAE

# Create pipeline
pipeline = SimpleWanPipeline.from_pretrained(
    pretrained_model_path="path/to/wan2.1-t2v-1.3b",
    transformer=transformer,
    vae=vae,
)

# Generate video
video = pipeline(
    prompt="A cat playing piano",
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=5.0,
)

# video is numpy array (1, 81, 480, 832, 3) uint8
# Save with your preferred video writer (imageio, opencv, etc.)
"""