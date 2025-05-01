# %%
from diffusers import StableDiffusionPipeline
from IPython.display import Image, display
import torch
# %%


def run(prompt: str, pipeline: StableDiffusionPipeline):
    print(prompt)
    image = pipeline(prompt)
    display(image.images[0])


#%%

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
device = "cuda"
pipe.to(device)

#%%
run("a portrait of a wise old warrior", pipe)

# %%

def custom_pipeline(prompt: str, pipe: StableDiffusionPipeline):
    pass
    # prompt_embeds, negative_prompt_embeds = pipe.encode_prompt()
    # prompt_embeds.shape = b, s, h



# %%
