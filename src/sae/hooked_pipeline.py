from collections import defaultdict
from typing import List, Optional, Union
from diffusers import DDIMScheduler, StableDiffusionPipeline # type: ignore
from torch import t  
from einops import rearrange  

class HookedDiffusionPipeline:
    def __init__(self, pipe: StableDiffusionPipeline) -> None:
        self.pipe = pipe

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def run_with_cache(
        self,
        prompt: Union[str, List[str]] = "",
        positions_to_cache: Optional[List[str]] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator: Optional[Union[t.Generator, List[t.Generator]]] = None,
    ) -> dict:
        if positions_to_cache is None:
            positions_to_cache = []
        position_activation_map = defaultdict(list)

        hook_handles = [
            self._register_cache_hook(position, position_activation_map)
            for position in positions_to_cache
        ]
        hook_handles = [h for h in hook_handles if h]

        self._run_generation_pipeline(
            prompt, guidance_scale, num_inference_steps, generator
        )

        for handle in hook_handles:
            handle.remove()

        cache_output = {}
        # Stack all tensors after hooks are removed
        for position in position_activation_map:
            # stacks list of tensors into a step dimension
            activations = t.stack(position_activation_map[position], dim=0)
            s, p, c, h, w = activations.shape
            assert (s, c, h, w) == (num_inference_steps, 1280, 16, 16), (
                f"unexpected activation shape: {activations.shape}"
            )
            activations = rearrange(activations, "steps p c h w -> (steps p) (h w) c")
            cache_output[position] = activations

        return cache_output

    def _run_generation_pipeline(
        self,
        prompt: Union[str, List[str]],
        guidance_scale: float,
        num_inference_steps: int,
        generator: Optional[Union[t.Generator, List[t.Generator]]],
    ):
        self.pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

    def _register_cache_hook(self, position: str, cache_output: dict):
        def hook_fn(module, input, output):
            assert isinstance(output, tuple) and len(output) == 1, (
                "unexpected output from hook"
            )
            activations = output[0].to(self.cfg.activation_dtype)
            _, c, h, w = activations.shape
            assert (c, h, w) == (1280, 16, 16), "unexpected activation shape"
            neg_conditioned, text_conditioned = activations.chunk(2, dim=0)
            cache_output[position].append(text_conditioned)

        block: t.nn.Module = self._locate_block(position)
        return block.register_forward_hook(hook_fn)

    def _locate_block(self, position: str) -> t.nn.Module:
        block = self.pipe
        for step in position.split("."):
            if step.isdigit():
                step = int(step)
                block = block[step]  # type: ignore
            else:
                block = getattr(block, step)  # type: ignore
        return block  # type: ignore

    def to(self, device: str):
        self.pipe.to(device)

    @classmethod
    def from_pretrained(cls, model_name: str, activation_dtype: t.dtype) -> "HookedDiffusionPipeline":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=activation_dtype,
            safety_checker=None,
        )
        assert isinstance(pipe, StableDiffusionPipeline)
        return cls(pipe)

