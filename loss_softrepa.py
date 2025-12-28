from typing import Literal
import math

import torch
import torch.nn.functional as F


# Asks "How aligned are the directions of the prediction and target vectors?" This is a measure of cosine similarity.
def text_weighted_infonce_loss(
    model_pred,
    target,
    text_embeddings,
    is_dropped,
    temperature=0.07,
    hard_negative_weight=2.0,
    use_text_similarity=True,
):
    """
    InfoNCE with text-aware weighting:
    - Harder penalties for similar captions (confusable)
    - Can use text embeddings for semantic similarity

    Args:
        model_pred: [B, C, H, W]
        target: [B, C, H, W]
        text_embeddings: [B, T, D] pre-computed text embeddings (optional)
        is_dropped: [B] boolean mask
        temperature: scaling factor
        hard_negative_weight: extra weight for similar captions
        use_text_similarity: whether to weight by caption similarity

    Returns:
        per_sample_loss: [B] contrastive losses
    """
    B = model_pred.shape[0]
    device = model_pred.device

    # Identify valid samples
    valid_mask = ~is_dropped

    num_valid = valid_mask.sum()
    if num_valid < 2:
        return torch.zeros(B, device=device)

    # Extract valid samples
    pred_valid = model_pred[valid_mask]
    target_valid = target[valid_mask]

    # Flatten and normalize
    pred_flat = pred_valid.reshape(num_valid, -1)
    target_flat = target_valid.reshape(num_valid, -1)

    pred_norm = F.normalize(pred_flat, p=2, dim=1)
    target_norm = F.normalize(target_flat, p=2, dim=1)

    # Compute similarity matrix
    sim_matrix = (
        torch.mm(pred_norm, target_norm.t()) / temperature
    )  # [num_valid, num_valid]

    # Text-based weighting
    if use_text_similarity:
        text_sim = text_embeddings_to_similarity(text_embeddings)

        # Weight negatives by text similarity
        # Similar captions = harder negatives = higher weight
        # Create weight matrix: 1.0 for diagonal, hard_negative_weight for similar off-diagonal
        weight_matrix = torch.ones_like(sim_matrix)

        # Apply extra weight to similar captions (hard negatives)
        for i in range(num_valid):
            for j in range(num_valid):
                if i != j:
                    # Scale weight by text similarity
                    # High text_sim → hard negative → higher weight
                    weight_matrix[i, j] = (
                        1.0 + (hard_negative_weight - 1.0) * text_sim[i, j]
                    )

        # Apply weights to similarity matrix (for negatives only)
        mask = torch.eye(num_valid, device=device, dtype=torch.bool)
        sim_matrix = torch.where(mask, sim_matrix, sim_matrix * weight_matrix)

    # Standard InfoNCE, normalized by sqrt(batch_size) for stability
    labels = torch.arange(num_valid, device=device)
    loss_per_sample = F.cross_entropy(sim_matrix, labels, reduction="none") / math.sqrt(
        num_valid
    )

    # Map back to full batch size with zeros for dropped samples
    full_losses = torch.zeros(B, device=device)
    full_losses[valid_mask] = loss_per_sample

    return full_losses


def text_weighted_infonce_loss_with_softrepa(
    model_pred,
    target,
    text_embeddings,
    is_dropped,
    temperature=0.07,
    sigma=1.0,
    hard_negative_weight=2.0,
    similarity_method: Literal["cosine", "softrepa"] = "softrepa",
):
    """
    InfoNCE with text-aware weighting, using SoftREPA's error-based similarity.

    Sigma for SoftREPA controls the sharpness of the similarity measure. A smaller sigma will make the logits more spread out (e.g., -10 vs -1000). This leads to a "sharper" softmax distribution, forcing the model to be more confident in its positive pair. A larger sigma will shrink the logits closer together, creating a "softer" distribution. You will need to tune sigma as a hyperparameter. A good starting point is 1.0.

    Args:
        model_pred: [B, C, H, W] (this is v_pred)
        target: [B, C, H, W] (this is v_target)
        text_embeddings: [B, T, D]
        is_dropped: [B]
        sigma: scaling factor for SoftREPA error-based logits
        temperature: scaling factor for normalized cosine distance
        hard_negative_weight: extra weight for similar captions
        use_text_similarity: whether to weight by caption similarity

    Returns:
        per_sample_loss: [B] contrastive losses
    """
    B = model_pred.shape[0]
    device = model_pred.device

    # Identify valid samples
    valid_mask = ~is_dropped

    num_valid = valid_mask.sum()
    if num_valid < 2:
        return torch.zeros(B, device=device)

    # Extract valid samples
    pred_valid = model_pred[valid_mask]
    target_valid = target[valid_mask]

    # Flatten (but DO NOT normalize)
    pred_flat = pred_valid.reshape(num_valid, -1)
    target_flat = target_valid.reshape(num_valid, -1)

    if similarity_method == "softrepa":
        # Error-based logits as per SoftREPA
        # Asks "How small is the error between the prediction for image i and the target for image j?" It directly uses the model's primary objective (minimizing prediction error) as the basis for contrastive learning.
        sim_matrix = calculate_softrepa_error_based_logits(
            pred_flat, target_flat, sigma=sigma
        )
        # The output `sim_matrix` is a [num_valid, num_valid] matrix of logits.
    elif similarity_method == "cosine":
        # cosine similarity between velocities
        # Asks "How aligned are the directions of the prediction and target vectors?"
        pred_norm = F.normalize(pred_flat, p=2, dim=1)
        target_norm = F.normalize(target_flat, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = (
            torch.mm(pred_norm, target_norm.t()) / temperature
        )  # [num_valid, num_valid]
    else:
        raise ValueError(f"Unknown similarity_method: {similarity_method}")

    text_sim = text_embeddings_to_similarity(text_embeddings)

    # Weight negatives by text similarity
    # Similar captions = harder negatives = higher weight
    # Create weight matrix: 1.0 for diagonal, hard_negative_weight for similar off-diagonal
    weight_matrix = torch.ones_like(sim_matrix)

    # Apply extra weight to similar captions (hard negatives)
    for i in range(num_valid):
        for j in range(num_valid):
            if i != j:
                # Scale weight by text similarity
                # High text_sim → hard negative → higher weight
                weight_matrix[i, j] = (
                    1.0 + (hard_negative_weight - 1.0) * text_sim[i, j]
                )

    # Apply weights to similarity matrix (for negatives only)
    mask = torch.eye(num_valid, device=device, dtype=torch.bool)
    sim_matrix = torch.where(mask, sim_matrix, sim_matrix * weight_matrix)

    # Standard InfoNCE (THIS PART REMAINS EXACTLY THE SAME)
    labels = torch.arange(num_valid, device=device)
    loss_per_sample = F.cross_entropy(sim_matrix, labels, reduction="none") / math.sqrt(
        num_valid
    )

    # Map back to full batch size with zeros for dropped samples
    full_losses = torch.zeros(B, device=device)
    full_losses[valid_mask] = loss_per_sample

    return full_losses


def text_embeddings_to_similarity(text_embeddings):
    """
    Convert [B, T, D] text embeddings to [B, B] similarity matrix.

    Args:
        text_embeddings: [B, 77, D] CLIP penultimate hidden states

    Returns:
        sim_matrix: [B, B] pairwise similarity
    """
    B, T, D = text_embeddings.shape

    # Option 1: Mean pool over sequence dimension
    pooled = text_embeddings.mean(dim=1)  # [B, D]

    # Option 2: CLS token (first token) - often used with CLIP
    # pooled = text_embeddings[:, 0, :]  # [B, D]

    # Option 3: Max pool (take most salient features)
    # pooled = text_embeddings.max(dim=1)[0]  # [B, D]

    # Compute cosine similarity
    pooled_norm = F.normalize(pooled, p=2, dim=1)
    sim_matrix = torch.mm(pooled_norm, pooled_norm.t())

    return sim_matrix


def calculate_softrepa_error_based_logits(pred_flat, target_flat, sigma=1.0):
    """
    Calculates a similarity matrix based on negative squared error, as in SoftREPA.
    The "similarity" is actually the logit for the contrastive loss.

    Args:
        pred_flat: [N, D] tensor of flattened model predictions (v_pred).
        target_flat: [N, D] tensor of flattened targets (v_target).
        sigma: A scaling parameter, similar to temperature. Controls the sharpness.

    Returns:
        logit_matrix: [N, N] matrix where entry (i, j) is -||pred_i - target_j||^2 / sigma.
    """
    # Use broadcasting to compute the pairwise squared error efficiently
    # pred_flat:       [N, 1, D]
    # target_flat:     [1, N, D]
    # diff:            [N, N, D]
    diff = pred_flat.unsqueeze(1) - target_flat.unsqueeze(0)

    # Sum the squares over the feature dimension (D)
    # This computes ||v_pred_i - v_target_j||^2 for all i, j
    squared_error_matrix = torch.sum(diff**2, dim=2)  # Shape: [N, N]

    D = pred_flat.shape[-1]
    normalized_error = squared_error_matrix / D

    # The logit is the negative squared error, scaled by sigma
    # A small error -> large logit (good match)
    # A large error -> small (very negative) logit (bad match)
    logit_matrix = -normalized_error / sigma

    return logit_matrix

