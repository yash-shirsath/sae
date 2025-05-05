import os
from pathlib import Path

import torch

from data.activation_capture_prompts.definitions import objects, styles


def read_prompts_file(object_name):
    """Read prompts from the corresponding object file."""
    file_path = Path(__file__).parent / f"sd_prompt_{object_name}.txt"
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def generate_styled_prompts():
    """Generate prompts by combining object prompts with style modifiers."""
    all_prompts = []
    prompt_metadata = []  #  (object, style) pairs for each prompt

    for obj in objects:
        base_prompts = read_prompts_file(obj)

        for prompt in base_prompts:
            # Add the base prompt without style
            all_prompts.append(prompt)
            prompt_metadata.append((obj, "None"))

            for style in styles:
                styled_prompt = f"{prompt}, {style} style"
                all_prompts.append(styled_prompt)
                prompt_metadata.append((obj, style))

    return all_prompts, prompt_metadata


def save_prompts_pytorch(output_dir=None, filename="prompts"):
    """Save prompts as PyTorch tensors."""
    if output_dir is None:
        output_dir = Path(__file__).parent

    prompts, metadata = generate_styled_prompts()

    # Create dictionaries for object and style mapping
    object_to_idx = {obj: i for i, obj in enumerate(objects)}
    style_to_idx = {style: i for i, style in enumerate(styles + ["None"])}

    # Convert metadata to indices
    object_indices = [object_to_idx[m[0]] for m in metadata]
    style_indices = [style_to_idx[m[1]] for m in metadata]

    assert len(prompts) == len(object_indices) == len(style_indices)

    data_dict = {
        "prompts": prompts,
        "object_indices": torch.tensor(object_indices),
        "style_indices": torch.tensor(style_indices),
        "object_names": objects,
        "style_names": styles + ["None"],
    }

    test = [
        data_dict["style_names"][si].lower() in p.lower()
        for p, si in zip(
            data_dict["prompts"],
            data_dict["style_indices"],
        )
        if data_dict["style_names"][si] != "None"
    ]
    assert all(test)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.pt")

    torch.save(data_dict, output_path)

    print(f"Saved {len(prompts)} prompts to {output_path}")
    return output_path


if __name__ == "__main__":
    pt_path = save_prompts_pytorch()

    print("\nTo load the data in your training script:")
    print(f"data = torch.load('{pt_path}')")
