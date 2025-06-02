"""
Notebook to explore generations from the stable diffusion pipeline
and compare the results of different schedulers.
"""

# %%
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import make_image_grid
from IPython.display import Image, display
import torch
# %%


def run(prompt: str, pipeline: StableDiffusionPipeline, seed: int = None):
    print(prompt)
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    else:
        generator = None
    image = pipeline(prompt, generator=generator)
    display(image.images[0])


def run_grid(
    prompts: list[str],
    cols: int,
    pipeline1: StableDiffusionPipeline,
    pipeline2: StableDiffusionPipeline | None = None,
    seed: int = 42,
):
    """
    Generate a grid of images from multiple prompts using batch processing.
    If pipeline2 is provided, results are displayed side by side for comparison.

    Args:
        prompts: List of prompts to generate images from
        cols: Number of images to generate per prompt
        pipeline1: First StableDiffusionPipeline to use for generation
        pipeline2: Optional second StableDiffusionPipeline to use for generation
        seed: Random seed for deterministic generation
    """
    rows = len(prompts)

    if pipeline2:
        print(
            f"Generating {cols} images for each of {rows} prompts with both pipelines"
        )
    else:
        print(f"Generating {cols} images for each of {rows} prompts with pipeline1")

    # Generate images with first pipeline
    generator1 = torch.Generator(device=pipeline1.device).manual_seed(seed)
    results1 = pipeline1(prompts, num_images_per_prompt=cols, generator=generator1)
    images1 = results1.images

    if pipeline2:
        # Generate images with second pipeline
        generator2 = torch.Generator(device=pipeline2.device).manual_seed(seed)
        results2 = pipeline2(prompts, num_images_per_prompt=cols, generator=generator2)
        images2 = results2.images

        # Combine images from both pipelines
        combined_images = []
        for i in range(0, len(images1), cols):
            combined_images.extend(images1[i : i + cols])
            combined_images.extend(images2[i : i + cols])

        # Create and display the grid with double the number of columns
        grid = make_image_grid(combined_images, rows=rows, cols=cols * 2)
    else:
        # Create and display the grid with images from pipeline1 only
        grid = make_image_grid(images1, rows=rows, cols=cols)

    display(grid)


# %%

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
device = "cuda"
pipe.to(device)

# %%
run("a portrait of a wise old warrior", pipe)
