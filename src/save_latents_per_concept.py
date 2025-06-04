"""
Map Concepts to SAE Latents and Save to Disk. 
These latents will be used to calculate feature importances during steering.
"""

import os
import sys
import fire
import torch
from diffusers.utils.import_utils import is_xformers_available

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sae.hooked_pipeline import HookedDiffusionPipeline
from sae.model import Sae
from data.activation_capture_prompts.definitions import concepts, styles

import pickle

import tqdm

"""
optimizations from https://pytorch.org/blog/accelerating-generative-ai-3/
"""
torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True  # type: ignore
torch._inductor.config.coordinate_descent_tuning = True  # type: ignore
torch._inductor.config.epilogue_fusion = False  # type: ignore
torch._inductor.config.coordinate_descent_check_all_directions = True  # type: ignore



def main(
    checkpoint_path,
    hookpoint,
    pipe_path,
    save_dir,
    steps=100,
    num_concepts=20,
    prompts_per_concept=80,
    themes_per_prompt=9,
    batch_size=30,
    seed=188,
):
    subset_concepts = concepts[:num_concepts]
    concept_prompts_dict = {concept_avail: [] for concept_avail in subset_concepts}
    for concept_avail in subset_concepts:
        with open(
            os.path.join(
                "data/activation_capture_prompts/anchor_prompts/finetune_prompts",
                f"sd_prompt_{concept_avail}.txt",
            ),
            "r",
        ) as prompt_file:
            prompts = prompt_file.readlines()
            # Renamed 'prompt' to 'current_prompts' to avoid conflict with outer 'prompt' in loop
            current_prompts = [p.strip() for p in prompts]
            current_prompts = current_prompts[:prompts_per_concept] 
            for p_text in current_prompts: # Renamed 'p' to 'p_text'
                for i, theme in enumerate(styles):
                    if i >= themes_per_prompt:
                        break
                    original_p_text = p_text # Store original p_text to avoid modification in loop
                    if original_p_text.endswith('.'):
                        original_p_text = original_p_text[:-1]
                    suffix = f" in {theme.replace('_', ' ')} style." if theme != "_" else ""
                    final_prompt_text = f"{original_p_text}{suffix}"
                    concept_prompts_dict[concept_avail].append(final_prompt_text)

    sae = Sae.load_from_disk(
        os.path.join(checkpoint_path, hookpoint), device="cuda"
    ).eval()

    sae = sae.to(dtype=torch.float16)

    pipe = HookedDiffusionPipeline.from_pretrained(
        pipe_path,
        activation_dtype=torch.float16,
    )
    pipe.to("cuda")
    
    if is_xformers_available():
        print("Enabling xFormers memory efficient attention")
        pipe.pipe.unet.enable_xformers_memory_efficient_attention()

    concept_latents_dict = {}

    progress_bar = tqdm.tqdm(list(concept_prompts_dict.keys()), total=len(concept_prompts_dict))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for concept_avail in progress_bar:
        progress_bar.set_description(f"Processing concept: {concept_avail}")
        all_prompts_for_concept = concept_prompts_dict[concept_avail]
        
        concept_sae_latents_list = []

        if not all_prompts_for_concept:
            d_sae_val = sae.num_latents
            concept_latents_dict[concept_avail] = torch.empty((0, steps, d_sae_val), dtype=torch.float16)
            continue

        for batch_start_idx in tqdm.tqdm(range(0, len(all_prompts_for_concept), batch_size), total=len(all_prompts_for_concept) // batch_size if batch_size > 0 else 0 ): # Added check for batch_size > 0
            batch_end_idx = min(batch_start_idx + batch_size, len(all_prompts_for_concept))
            prompts_batch = all_prompts_for_concept[batch_start_idx:batch_end_idx]

            if not prompts_batch:
                continue

            _, acts_cache = pipe.run_with_cache(
                prompt=prompts_batch,
                generator=generator,
                num_inference_steps=steps,
                positions_to_cache=[hookpoint],
                guidance_scale=9.0,
            )
            activations = acts_cache["output"][hookpoint].cpu()
            assert activations.shape[0] == len(prompts_batch)
            assert activations.shape[1] == steps
            
            with torch.no_grad():
                for i in range(len(prompts_batch)):
                    single_prompt_activations = activations[i]
                    sae_in = single_prompt_activations.reshape(steps, -1, sae.d_in)
                    
                    top_acts, top_indices = sae.encode(sae_in.to(sae.device))
                    
                    sae_out_sparse = torch.zeros(
                        (top_acts.shape[0], sae.num_latents),
                        device=sae.device,
                        dtype=top_acts.dtype,
                    ).scatter_(-1, top_indices, top_acts)
                    
                    sae_out_reconstructed = sae_out_sparse.reshape(steps, -1, sae.num_latents).cpu()
                    
                    mean_sae_features = sae_out_reconstructed.mean(1).to(dtype=torch.float16)
                    concept_sae_latents_list.append(mean_sae_features)

        if concept_sae_latents_list: # Corrected class_sae_latents_list to concept_sae_latents_list
            concept_latents_dict[concept_avail] = torch.stack(concept_sae_latents_list)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"concept_latents_dict_{hookpoint}.pkl"), "wb") as f:
        pickle.dump(concept_latents_dict, f)
    print(f"Saved to {save_dir}/concept_latents_dict_{hookpoint}.pkl")


if __name__ == "__main__":
    fire.Fire(main)
