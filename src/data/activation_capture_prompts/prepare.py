import os
from pathlib import Path
from typing import Optional

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


def balance_concepts_styles(
    df: pd.DataFrame, main_concept: str, random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Subset the prompts DataFrame to create a balanced dataset with a main concept and other concepts.
    Ensures even distribution of both concepts and styles in the non-main concept portion.

    Args:
        df: Input DataFrame containing prompts with 'concept' and 'style' columns
        main_concept: The main concept to include all prompts for
        random_state: Optional random seed for reproducibility

    Returns:
        DataFrame containing:
        - All prompts for the main concept
        - An even distribution of other concepts and their styles
    """
    # Get all prompts for the main concept
    main_concept_df = df[df["concept"] == main_concept].copy()
    main_concept_count = len(main_concept_df)

    # Get all other concepts
    other_concepts_df = df[df["concept"] != main_concept].copy()

    # Get unique concepts and styles
    unique_concepts = other_concepts_df["concept"].unique()
    unique_styles = other_concepts_df["style"].unique()
    n_concepts = len(unique_concepts)
    n_styles = len(unique_styles)

    # Calculate how many prompts we need per concept to achieve even distribution
    prompts_per_concept = main_concept_count // n_concepts
    remaining_prompts = main_concept_count % n_concepts

    # Initialize list to store sampled prompts
    sampled_prompts = []

    # For each concept, sample prompts while trying to maintain even style distribution
    for i, concept in enumerate(unique_concepts):
        # Add one extra prompt to the first 'remaining_prompts' concepts to account for division remainder
        n_samples = prompts_per_concept + (1 if i < remaining_prompts else 0)

        # Get all prompts for this concept
        concept_df = other_concepts_df[other_concepts_df["concept"] == concept]

        # Calculate how many prompts we want per style for this object
        prompts_per_style = n_samples // n_styles
        remaining_style_prompts = n_samples % n_styles

        # Sample prompts for each style
        style_samples = []
        for j, style in enumerate(unique_styles):
            # Add one extra prompt to the first 'remaining_style_prompts' styles
            style_n_samples = prompts_per_style + (
                1 if j < remaining_style_prompts else 0
            )

            # Get prompts for this style within this concept
            style_df = concept_df[concept_df["style"] == style]

            # If we have more prompts than needed, sample randomly
            if len(style_df) > style_n_samples:
                style_df = style_df.sample(n=style_n_samples, random_state=random_state)
            elif len(style_df) < style_n_samples:
                # If we don't have enough prompts for this style, take all of them
                pass

            style_samples.append(style_df)

        # Combine all style samples for this concept
        concept_sampled = pd.concat(style_samples, ignore_index=True)

        # If we still need more prompts to reach n_samples, sample from remaining prompts
        if len(concept_sampled) < n_samples:
            remaining_df = concept_df[~concept_df.index.isin(concept_sampled.index)]
            if len(remaining_df) > 0:
                additional_samples = remaining_df.sample(
                    n=min(n_samples - len(concept_sampled), len(remaining_df)),
                    random_state=random_state,
                )
                concept_sampled = pd.concat(
                    [concept_sampled, additional_samples], ignore_index=True
                )

        sampled_prompts.append(concept_sampled)

    # Combine main concept and sampled other concepts
    result_df = pd.concat([main_concept_df] + sampled_prompts, ignore_index=True)

    return result_df


if __name__ == "__main__":
    parquet_path = save_prompts_dataframe()

    print("\nTo load the data in your script:")
    print(f"import pandas as pd")
    print(f"prompts_df = pd.read_parquet('{parquet_path}')")
