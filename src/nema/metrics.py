"""Metrics used in Tables 1--3 of the NGA paper."""

from __future__ import annotations

import numpy as np
import torch


def johnson_similarity(
    common_nodes: int,
    common_edges: int,
    left_size: int,
    right_size: int,
) -> float:
    if left_size <= 0 or right_size <= 0:
        return 0.0
    return float((common_nodes + common_edges) ** 2 / (left_size * right_size))


def retrieval_metrics(
    predicted: list[float] | np.ndarray,
    target: list[float] | np.ndarray,
    group_size: int = 100,
    relevance_threshold: float = 0.5,
) -> dict[str, float]:
    """Paper-compatible MRR, P@10 and MAP over contiguous query groups."""

    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    if predicted.shape != target.shape or predicted.ndim != 1:
        raise ValueError("predicted and target must be equal-length vectors")
    if predicted.size % group_size:
        raise ValueError("number of pairs must be divisible by group_size")

    mrr_values, p10_values, ap_values, overlap_values, top10_ap_values = [], [], [], [], []
    ndcg10_values = []
    relevant_counts = []
    for start in range(0, predicted.size, group_size):
        pred = predicted[start : start + group_size]
        truth = target[start : start + group_size]
        # Preserve the released evaluator's exact NumPy descending-order idiom
        # (including its tie behavior), rather than substituting a stable sort.
        order = np.argsort(pred)[::-1]

        # The released NGA evaluator designates the first maximally similar item.
        chosen = int(np.flatnonzero(truth == truth.max())[0])
        rank = int(np.flatnonzero(order == chosen)[0]) + 1
        mrr_values.append(1.0 / rank)

        relevant = truth > relevance_threshold
        relevant_counts.append(int(relevant.sum()))
        p10_values.append(float(relevant[order[:10]].sum() / 10.0))
        truth_order = np.argsort(truth)[::-1]
        top_k = min(10, group_size)
        overlap_values.append(
            len(set(order[:top_k].tolist()) & set(truth_order[:top_k].tolist())) / top_k
        )
        top10_relevant = np.zeros(group_size, dtype=bool)
        top10_relevant[truth_order[:top_k]] = True
        ranked_top10_relevant = top10_relevant[order]
        ranked_precision = np.cumsum(ranked_top10_relevant) / np.arange(1, group_size + 1)
        top10_ap_values.append(
            float((ranked_precision * ranked_top10_relevant).sum() / top_k)
        )
        discounts = np.log2(np.arange(2, top_k + 2))
        gain = np.exp2(truth[order[:top_k]]) - 1.0
        ideal_gain = np.exp2(truth[truth_order[:top_k]]) - 1.0
        ideal_dcg = float((ideal_gain / discounts).sum())
        ndcg10_values.append(float((gain / discounts).sum() / ideal_dcg) if ideal_dcg else 1.0)
        # This is a direct vectorized equivalent of evaluate.py's
        # calculate_map/calculate_ap implementation at the pinned commit.
        pred_tensor = torch.tensor(pred.tolist()).reshape(1, -1)
        relevant_tensor = (torch.tensor(truth.tolist()) > relevance_threshold).reshape(1, -1)
        precisions = torch.zeros_like(pred_tensor)
        for k in range(1, group_size + 1):
            top_k = torch.topk(pred_tensor, k=k, dim=1).indices
            top_k_relevant = torch.gather(relevant_tensor, 1, top_k)
            precisions[:, k - 1] = top_k_relevant.sum(dim=1) / k
        prediction_order = torch.sort(pred_tensor, dim=1, descending=True).indices
        sorted_relevant = torch.gather(relevant_tensor, 1, prediction_order)
        relevant_count = relevant_tensor.sum(dim=1).clamp_min(1)
        ap_values.append(float(((precisions * sorted_relevant).sum(dim=1) / relevant_count)[0]))

    return {
        "MRR": float(np.mean(mrr_values)),
        "P@10": float(np.mean(p10_values)),
        "MAP": float(np.mean(ap_values)),
        "Top10-overlap": float(np.mean(overlap_values)),
        "Top10-AP": float(np.mean(top10_ap_values)),
        "NDCG@10": float(np.mean(ndcg10_values)),
        "mean_relevant_per_query": float(np.mean(relevant_counts)),
        "label_limited_max_P@10": float(
            np.mean([min(count, 10) / 10.0 for count in relevant_counts])
        ),
        "queries": len(mrr_values),
    }
