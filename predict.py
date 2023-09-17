# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import torch
from diffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.pipeline = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", cache_dir="model_cache", torch_dtype=torch.float16).to("cuda")

        self.prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
            "warp-ai/wuerstchen-prior",
            cache_dir="model_cache",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
            "warp-ai/wuerstchen", cache_dir="model_cache", torch_dtype=torch.float16
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Anthropomorphic cat dressed as a firefighter",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image.",
            default=1536,
        ),
        height: int = Input(
            description="Height of output image.",
            default=1024,
        ),
        num_images_per_prompt: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        prior_num_inference_steps: int = Input(
            description="Number of prior denoising steps.", ge=1, le=500, default=60
        ),
        prior_guidance_scale: float = Input(
            description="Scale for classifier-free guidance in prior.",
            ge=1,
            le=20,
            default=4.0,
        ),
        decoder_num_inference_steps: int = Input(
            description="Number of prior denoising steps.", ge=1, le=500, default=12
        ),
        decoder_guidance_scale: float = Input(
            description="Scale for classifier-free guidance in decoder.",
            ge=0,
            le=20,
            default=0.0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> list[Path]:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        prior_output = self.prior_pipeline(
            prompt=prompt,
            height=height,
            width=width,
            timesteps=DEFAULT_STAGE_C_TIMESTEPS,
            negative_prompt=negative_prompt,
            guidance_scale=prior_guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        )
        prior_output = next(prior_output)
        decoder_output = self.decoder_pipeline(
            image_embeddings=prior_output.image_embeddings,
            prompt=prompt,
            num_inference_steps=decoder_num_inference_steps,
            guidance_scale=decoder_guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            output_type="pil",
        ).images

        output_paths = []
        for i, sample in enumerate(decoder_output):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
