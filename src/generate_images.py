"""
Generate images for concepts across a range of hyperparameters.
"""

import os
import pickle
import sys

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from packaging import version
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import fire
from diffusers.utils.import_utils import is_xformers_available

import sae.hooks as hooks
from data.activation_capture_prompts.definitions import concepts, styles
from sae.feature_importance import compute_feature_importance
from sae.hooked_pipeline import HookedDiffusionPipeline
from sae.model import Sae

"""
optimizations from https://pytorch.org/blog/accelerating-generative-ai-3/
"""
torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True  # type: ignore
torch._inductor.config.coordinate_descent_tuning = True  # type: ignore
torch._inductor.config.epilogue_fusion = False  # type: ignore
torch._inductor.config.coordinate_descent_check_all_directions = True  # type: ignore


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_sae(sae_checkpoint, hookpoint, device):
    sae = Sae.load_from_disk(
        os.path.join(sae_checkpoint, hookpoint), device=device
    ).eval()
    sae = sae.to(dtype=torch.float16)
    return sae


def main(
    pipe_checkpoint,
    hookpoint,
    concept_latents_path,
    sae_checkpoint,
    seed=42,
    steps=100,
    percentiles=[99.99, 99.995, 99.999],
    multipliers=[-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0],
    guidance_scale=9.0,
    output_dir="generated_imgs",
    num_concepts=2,
    prompts_per_concept=80,
    styles_per_prompt=9,
    batch_size=30,
):
    accelerator = Accelerator()
    device = accelerator.device

    model = HookedDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        activation_dtype=torch.float16,
    )
    model = model.to(device)

    if is_xformers_available():
        import xformers  # type: ignore

        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if accelerator.is_main_process:
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        model.enable_xformers_memory_efficient_attention()  # type: ignore

    seed_everything(seed)

    sae = load_sae(sae_checkpoint, hookpoint, device)
    with open(
        concept_latents_path,
        "rb",
    ) as f:
        concept_latents_dict = pickle.load(f)

    subset_concepts = concepts[:num_concepts]
    used_concept_latents_dict = {
        concept_: concept_latents_dict[concept_] for concept_ in subset_concepts
    }
    concept_prompt_dict = {concept_: [] for concept_ in subset_concepts}
    for concept_to_unlearn in subset_concepts:
        with open(
            os.path.join(
                "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                f"sd_prompt_{concept_to_unlearn}.txt",
            ),
            "r",
        ) as prompt_file:
            prompts = prompt_file.readlines()
            prompt = [p.strip() for p in prompts]
            prompt = prompt[:prompts_per_concept]
            for p in prompt:
                for i, style in enumerate(styles):
                    if i >= styles_per_prompt:
                        break
                    if p.endswith("."):
                        p = p[:-1]
                    suffix = (
                        f" in {style.replace('_', ' ')} style." if style != "_" else ""
                    )
                    prompt = f"{p}{suffix}"
                    concept_prompt_dict[concept_to_unlearn].append(prompt)

    progress_bar = tqdm(
        total=len(multipliers) * len(subset_concepts) * len(percentiles),
        disable=not accelerator.is_main_process,
    )
    for multiplier in multipliers:
        for percentile in percentiles:
            for concept_to_unlearn in subset_concepts:
                if accelerator.is_main_process:
                    progress_bar.set_description(
                        f"Multiplier: {multiplier} Percentile: {percentile} Concept: {concept_to_unlearn}"
                    )
                output_path = os.path.join(
                    output_dir,
                    f"percentile_{percentile}_multiplier_{multiplier}/{concept_to_unlearn}",
                )
                os.makedirs(output_path, exist_ok=True)
                all_prompts = [
                    (concept_name, prompt)
                    for concept_name, prompts in concept_prompt_dict.items()
                    for prompt in prompts
                ]
                input_concepts = []
                with accelerator.split_between_processes(all_prompts) as local_tuples:
                    local_prompts = [prompt.strip() for _, prompt in local_tuples]  # type: ignore
                    local_concepts = [concept_name for concept_name, _ in local_tuples]
                    all_images = []
                    for i in range(0, len(local_prompts), batch_size):
                        batch_prompts = local_prompts[i : i + batch_size]
                        steering_hooks = {}
                        steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                            concept_to_unlearn=[concept_to_unlearn],
                            percentile=percentile,
                            multiplier=multiplier,
                            feature_importance_fn=compute_feature_importance,
                            concept_latents_dict=used_concept_latents_dict,
                            sae=sae,
                            steps=steps,
                            preserve_error=True,
                        )

                        # Create a new generator for each batch to ensure consistent results
                        batch_generator = torch.Generator(device="cpu").manual_seed(
                            seed + i
                        )
                        with torch.no_grad():
                            batch_images = model.run_with_hooks(
                                prompt=batch_prompts,
                                generator=batch_generator,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                position_hook_dict=steering_hooks,
                            )
                        all_images.extend(batch_images)
                    images = all_images
                    input_concepts.extend(local_concepts)
                accelerator.wait_for_everyone()
                images = gather_object(images)
                input_concepts = gather_object(input_concepts)
                if accelerator.is_main_process:
                    for i, (img, object_concept) in enumerate(
                        zip(images, input_concepts)
                    ):
                        img.save(
                            os.path.join(
                                output_path,
                                f"{object_concept}_seed{seed}_{i}.jpg",
                            )
                        )
                if accelerator.is_main_process:
                    progress_bar.update(1)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)
