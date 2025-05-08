import os
from pathlib import Path

import pandas as pd

from data.activation_capture_prompts.definitions import concepts, styles


def read_prompts_file(concept_name):
    """Read prompts from the corresponding object file."""
    file_path = Path(__file__).parent / f"sd_prompt_{concept_name}.txt"
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def generate_styled_prompts() -> pd.DataFrame:
    """Generate prompts by combining concept prompts with style modifiers."""
    records = []

    for concept in concepts:
        base_prompts = read_prompts_file(concept)

        for prompt in base_prompts:
            # Add the base prompt without style
            records.append({"concept": concept, "style": "None", "prompt": prompt})

            for style in styles:
                styled_prompt = f"{prompt}, {style} style"
                records.append(
                    {"concept": concept, "style": style, "prompt": styled_prompt}
                )

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


def load_generated_prompts(filename="all_prompts.parquet") -> pd.DataFrame:
    path = os.path.join(Path(__file__).parent, filename)
    return pd.read_parquet(path)


if __name__ == "__main__":
    parquet_path = save_prompts_dataframe()

    print("\nTo load the data in your script:")
    print(f"import pandas as pd")
    print(f"prompts_df = pd.read_parquet('{parquet_path}')")
