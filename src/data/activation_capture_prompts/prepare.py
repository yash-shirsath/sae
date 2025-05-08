import os
from pathlib import Path

import pandas as pd

from data.activation_capture_prompts.definitions import objects, styles


def read_prompts_file(object_name):
    """Read prompts from the corresponding object file."""
    file_path = Path(__file__).parent / f"sd_prompt_{object_name}.txt"
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def generate_styled_prompts() -> pd.DataFrame:
    """Generate prompts by combining object prompts with style modifiers."""
    records = []

    for obj in objects:
        base_prompts = read_prompts_file(obj)

        for prompt in base_prompts:
            # Add the base prompt without style
            records.append({"object": obj, "style": "None", "prompt": prompt})

            for style in styles:
                styled_prompt = f"{prompt}, {style} style"
                records.append({"object": obj, "style": style, "prompt": styled_prompt})

    return pd.DataFrame(records)


def save_prompts_dataframe(output_dir=None, filename="all_prompts.parquet"):
    """Save prompts DataFrame to a Parquet file."""
    if output_dir is None:
        output_dir = Path(__file__).parent

    prompts_df = generate_styled_prompts()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    prompts_df.to_parquet(output_path, index=False)

    print(f"Saved prompts DataFrame to {output_path}")
    return output_path


def load_generated_prompts(
    path="/workspace/sae/src/data/activation_capture_prompts/all_prompts.parquet",
) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    parquet_path = save_prompts_dataframe()

    print("\nTo load the data in your script:")
    print(f"import pandas as pd")
    print(f"prompts_df = pd.read_parquet('{parquet_path}')")
