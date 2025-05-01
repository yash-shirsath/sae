# %%
from diffusers import StableDiffusionPipeline
from IPython.display import Image, display
import torch
# %%


def run(prompt: str, pipeline: StableDiffusionPipeline):
    print(prompt)
    image = pipeline(prompt)
    display(image.images[0])


if __name__ == "__main__":
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    device = "cuda"
    pipeline.to(device)
    run("a portrait of a wise old warrior", pipeline)

# %%
print(torch.cuda.is_available())
print(torch.cuda.get_arch_list())

# %%
