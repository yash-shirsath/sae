"""
Tools to understand which latent space directions correspond to each concept.
"""

import torch as t


def compute_feature_importance(
    concept_latents_dict, target_concept, timestep, epsilon=1e-8
):
    if target_concept not in concept_latents_dict:
        raise ValueError(f"target_concept '{target_concept}' not found.")

    # Mean activation for the target style (shape: [num_features])
    latents_x = concept_latents_dict[target_concept][:, timestep, :].float()
    mean_x = latents_x.mean(dim=0)

    # All other styles
    other_concepts = [c for c in concept_latents_dict if c != target_concept]
    if not other_concepts:
        # If there's only one style, can't compare.
        return mean_x  # or t.zeros_like(mean_x), depending on your needs

    # Mean activation for the combined "others"
    latents_others = t.cat(
        [concept_latents_dict[c][:, timestep, :].float() for c in other_concepts], dim=0
    )
    mean_others = latents_others.mean(dim=0)

    # Denominators: total activation across all features
    total_x = mean_x.sum() + epsilon
    total_others = mean_others.sum() + epsilon

    # Proportions
    p_x = mean_x / total_x
    p_others = mean_others / total_others

    # Difference-based score
    scores = p_x - p_others

    return scores


def get_percentile_threshold(scores, percentile=95):
    """
    Returns the threshold for the given percentile.

    Args:
        scores (t.Tensor): 1D tensor of unnormalized scores, shape [num_features].
        percentile (float):    Percentile in [0,100].

    Returns:
        threshold (float): The score value at the given percentile.
    """
    # Convert percentile from 0–100 to a fraction (0–1)
    fraction = percentile / 100.0

    # Use PyTorch's built-in quantile function (available in PyTorch 1.7+)
    threshold = t.quantile(scores, fraction)

    return threshold
