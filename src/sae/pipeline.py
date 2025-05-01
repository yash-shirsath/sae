# %%
from diffusers import StableDiffusionPipeline

# %%


def run(prompt: str, pipeline: StableDiffusionPipeline):
    print(prompt)


if __name__ == "__main__":
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    run("a photo of an astronaut riding a horse on mars", pipeline)

# %%
