"""Metric bounds for retrieval labels left ambiguous by oracle timeouts.

The point estimators intentionally reproduce :func:`nema.metrics.retrieval_metrics`:

* relevance is the strict predicate ``true_similarity > threshold``;
* MRR ranks the first (original candidate order) maximally similar target with
  NumPy's released descending-order idiom for the predicted scores;
* P@10 uses the same NumPy order; and
* AP uses the released evaluator's PyTorch ``topk``/``sort`` implementation.

For AP, a polynomial dynamic program is exact when PyTorch's top-k sets are
the prefixes of its full sorted order.  At non-prefix score ties, an exact
label-relaxation dynamic program operates on equal-score blocks while using
the released evaluator's precomputed PyTorch ``topk`` and ``sort`` sets.  A
conservative unit interval remains as an explicit complexity fallback.
"""

from __future__ import annotations

import json
import hashlib
from itertools import product
from math import inf
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def validate_relative_code_provenance(
    code_sha256: dict[str, str], paths: Sequence[Path]
) -> None:
    """Validate hashes keyed by the exact repository-relative provenance path."""

    for path in paths:
        if path.is_absolute():
            raise ValueError(f"code provenance path must be repository-relative: {path}")
        expected = code_sha256.get(path.as_posix())
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if expected != actual:
            raise ValueError(f"metric-bounds code provenance mismatch: {path}")


def read_jsonl_unique_records(path: Path) -> list[dict]:
    """Read JSONL while refusing blank/invalid/duplicate-key records."""

    records: list[dict] = []
    seen: set[int] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if "key" not in record:
            raise ValueError(f"JSONL record lacks key at {path}:{line_number}")
        key = int(record["key"])
        if key in seen:
            raise ValueError(f"duplicate JSONL key {key} at {path}:{line_number}")
        seen.add(key)
        records.append(record)
    if not records:
        raise ValueError(f"empty JSONL input: {path}")
    return records


def validate_oracle_grid(
    records: Sequence[dict],
    *,
    expected_queries: int | None = None,
    expected_candidates: int | None = None,
) -> tuple[int, int]:
    """Validate exact contiguous query/candidate/key coverage for an oracle."""

    by_query: dict[int, set[int]] = {}
    pairs: set[tuple[int, int]] = set()
    keys: set[int] = set()
    for record in records:
        key = int(record["key"])
        query = int(record["query_index"])
        candidate = int(record["candidate_index"])
        if key in keys:
            raise ValueError(f"duplicate oracle key: {key}")
        if (query, candidate) in pairs:
            raise ValueError(f"duplicate oracle query/candidate pair: {(query, candidate)}")
        if query < 0 or candidate < 0:
            raise ValueError("oracle query/candidate indices must be non-negative")
        keys.add(key)
        pairs.add((query, candidate))
        by_query.setdefault(query, set()).add(candidate)
    queries = sorted(by_query)
    if queries != list(range(len(queries))):
        raise ValueError(f"oracle query indices are not contiguous: {queries}")
    candidate_counts = {len(indices) for indices in by_query.values()}
    if len(candidate_counts) != 1:
        raise ValueError(f"oracle candidate counts differ by query: {candidate_counts}")
    candidates = candidate_counts.pop()
    expected_candidate_indices = set(range(candidates))
    for query, indices in by_query.items():
        if indices != expected_candidate_indices:
            missing = sorted(expected_candidate_indices - indices)
            extra = sorted(indices - expected_candidate_indices)
            raise ValueError(
                f"oracle candidate coverage mismatch for query {query}: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
    expected_keys = set(range(1, len(records) + 1))
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        extra = sorted(keys - expected_keys)
        raise ValueError(
            f"oracle keys are not exact 1..N coverage: missing={missing[:10]} "
            f"extra={extra[:10]}"
        )
    for record in records:
        expected_key = int(record["query_index"]) * candidates + int(
            record["candidate_index"]
        ) + 1
        if int(record["key"]) != expected_key:
            raise ValueError(
                f"oracle key/query/candidate identity mismatch: key={record['key']} "
                f"expected={expected_key}"
            )
    query_count = len(queries)
    if expected_queries is not None and query_count != expected_queries:
        raise ValueError(
            f"oracle query count {query_count} differs from manifest {expected_queries}"
        )
    if expected_candidates is not None and candidates != expected_candidates:
        raise ValueError(
            f"oracle candidate count {candidates} differs from manifest {expected_candidates}"
        )
    return query_count, candidates


def released_point_metrics(
    predicted: Sequence[float],
    truth: Sequence[float],
    *,
    relevance_threshold: float = 0.5,
) -> dict[str, float]:
    """Return one-query MRR/P@10/AP with the released evaluator semantics."""

    pred = np.asarray(predicted, dtype=float)
    target = np.asarray(truth, dtype=float)
    if pred.ndim != 1 or pred.shape != target.shape or pred.size == 0:
        raise ValueError("predicted/truth must be non-empty equal-length vectors")

    numpy_order = np.argsort(pred)[::-1]
    chosen = int(np.flatnonzero(target == target.max())[0])
    rank = int(np.flatnonzero(numpy_order == chosen)[0]) + 1
    relevant = target > relevance_threshold
    p10 = float(relevant[numpy_order[: min(10, pred.size)]].sum() / 10.0)
    return {
        "MRR": 1.0 / rank,
        "P@10": p10,
        "MAP": released_average_precision(pred, relevant),
    }


def aggregate_released_point_metrics(
    predicted: Sequence[float],
    truth: Sequence[float],
    *,
    group_size: int,
    relevance_threshold: float = 0.5,
) -> dict[str, float]:
    """Aggregate the one-query implementation over contiguous query groups."""

    if group_size < 1 or len(predicted) != len(truth) or len(predicted) % group_size:
        raise ValueError("invalid aggregate retrieval shapes/group_size")
    groups = [
        released_point_metrics(
            predicted[start : start + group_size],
            truth[start : start + group_size],
            relevance_threshold=relevance_threshold,
        )
        for start in range(0, len(predicted), group_size)
    ]
    return {
        metric: float(np.mean([group[metric] for group in groups]))
        for metric in ("MRR", "P@10", "MAP")
    }


def released_average_precision(
    predicted: Sequence[float], relevant: Sequence[bool]
) -> float:
    """Reproduce the released PyTorch AP implementation for one query."""

    pred_tensor = torch.tensor(np.asarray(predicted, dtype=float).tolist()).reshape(1, -1)
    relevant_tensor = torch.tensor(
        np.asarray(relevant, dtype=bool).tolist(), dtype=torch.bool
    ).reshape(1, -1)
    group_size = pred_tensor.shape[1]
    precisions = torch.zeros_like(pred_tensor)
    for k in range(1, group_size + 1):
        top_k = torch.topk(pred_tensor, k=k, dim=1).indices
        top_k_relevant = torch.gather(relevant_tensor, 1, top_k)
        precisions[:, k - 1] = top_k_relevant.sum(dim=1) / k
    prediction_order = torch.sort(pred_tensor, dim=1, descending=True).indices
    sorted_relevant = torch.gather(relevant_tensor, 1, prediction_order)
    relevant_count = relevant_tensor.sum(dim=1).clamp_min(1)
    return float(((precisions * sorted_relevant).sum(dim=1) / relevant_count)[0])


def _torch_order_and_prefix_compatibility(
    predicted: Sequence[float],
) -> tuple[list[int], bool]:
    order, _, _, compatible = _torch_released_ap_structure(predicted)
    return order, compatible


def _torch_released_ap_structure(
    predicted: Sequence[float],
) -> tuple[list[int], list[set[int]], list[list[int]], bool]:
    """Precompute the exact Torch order, top-k sets, and equal-score blocks.

    Tensor construction intentionally matches :func:`released_average_precision`.
    A block is defined by equality after that construction (normally float32),
    not by equality of the input's NumPy float64 values.
    """

    pred = torch.tensor(np.asarray(predicted, dtype=float).tolist())
    if pred.ndim != 1 or pred.numel() == 0:
        raise ValueError("predicted must be a non-empty vector")
    if not bool(torch.isfinite(pred).all()):
        raise ValueError("predicted scores must be finite")
    order = torch.sort(pred, descending=True).indices.tolist()
    order = [int(index) for index in order]
    top_sets: list[set[int]] = []
    prefix: set[int] = set()
    compatible = True
    for k in range(1, pred.numel() + 1):
        prefix.add(int(order[k - 1]))
        top_set = {int(index) for index in torch.topk(pred, k=k).indices.tolist()}
        top_sets.append(top_set)
        if top_set != prefix:
            compatible = False

    blocks: list[list[int]] = []
    for index in order:
        if not blocks or bool(pred[index] != pred[blocks[-1][0]]):
            blocks.append([index])
        else:
            blocks[-1].append(index)

    higher: set[int] = set()
    first_rank = 1
    for block in blocks:
        block_set = set(block)
        allowed = higher | block_set
        for rank in range(first_rank, first_rank + len(block)):
            top_set = top_sets[rank - 1]
            if (
                len(top_set) != rank
                or not higher <= top_set
                or not top_set <= allowed
                or len(top_set & block_set) != rank - len(higher)
            ):
                raise AssertionError(
                    "Torch top-k set is inconsistent with its equal-score blocks"
                )
        higher.update(block_set)
        first_rank += len(block)
    return order, top_sets, blocks, compatible


def _validate_relaxed_labels(labels: Sequence[int | None]) -> None:
    if any(label not in (0, 1, None) for label in labels):
        raise ValueError("labels must contain only 0, 1, or None")


def _ap_bounds_prefix_dp_from_order(
    order: Sequence[int], labels: Sequence[int | None]
) -> tuple[float, float]:
    """Prefix-DP implementation using an already frozen Torch sort order."""

    _validate_relaxed_labels(labels)
    minimum: dict[int, float] = {0: 0.0}
    maximum: dict[int, float] = {0: 0.0}
    fixed_seen = 0
    fixed_total = sum(label == 1 for label in labels)
    for rank, index in enumerate(order, start=1):
        label = labels[index]
        if label == 0:
            continue
        if label == 1:
            for selected in list(minimum):
                contribution = (fixed_seen + selected + 1) / rank
                minimum[selected] += contribution
                maximum[selected] += contribution
            fixed_seen += 1
            continue

        next_minimum: dict[int, float] = {}
        next_maximum: dict[int, float] = {}
        for selected, value in minimum.items():
            next_minimum[selected] = min(next_minimum.get(selected, inf), value)
            chosen = value + (fixed_seen + selected + 1) / rank
            next_minimum[selected + 1] = min(
                next_minimum.get(selected + 1, inf), chosen
            )
        for selected, value in maximum.items():
            next_maximum[selected] = max(next_maximum.get(selected, -inf), value)
            chosen = value + (fixed_seen + selected + 1) / rank
            next_maximum[selected + 1] = max(
                next_maximum.get(selected + 1, -inf), chosen
            )
        minimum, maximum = next_minimum, next_maximum

    lower = inf
    upper = -inf
    for selected in minimum:
        denominator = max(fixed_total + selected, 1)
        lower = min(lower, minimum[selected] / denominator)
        upper = max(upper, maximum[selected] / denominator)
    return float(lower), float(upper)


def ap_bounds_prefix_dp(
    predicted: Sequence[float], labels: Sequence[int | None]
) -> tuple[float, float]:
    """Exact AP extrema by DP when top-k sets equal sorted-order prefixes.

    ``labels`` contains 0/1 for certified labels and ``None`` for ambiguous
    labels.  The state is the number of ambiguous positives selected so far;
    for each state we retain the minimum and maximum AP numerator.
    """

    if len(predicted) != len(labels) or not labels:
        raise ValueError("predicted/labels must be non-empty and equal length")
    order, compatible = _torch_order_and_prefix_compatibility(predicted)
    if not compatible:
        raise ValueError("PyTorch top-k tie behavior is not prefix-compatible")
    return _ap_bounds_prefix_dp_from_order(order, labels)


def _tie_block_transition_estimate(
    blocks: Sequence[Sequence[int]], labels: Sequence[int | None]
) -> tuple[int, int]:
    """Return ``(max ambiguous labels/block, assignment-state scans)``."""

    ambiguous_before = 0
    maximum_in_block = 0
    transition_scans = 0
    for block in blocks:
        ambiguous = sum(labels[index] is None for index in block)
        maximum_in_block = max(maximum_in_block, ambiguous)
        transition_scans += (ambiguous_before + 1) * (1 << ambiguous)
        ambiguous_before += ambiguous
    return maximum_in_block, transition_scans


def _block_assignment_terms(
    block: Sequence[int],
    *,
    first_rank: int,
    top_sets: Sequence[set[int]],
    labels: Sequence[int | None],
) -> list[tuple[int, float, float]]:
    """Enumerate ``(selected ambiguous, A, B)`` terms for one tie block.

    For sorted rank ``r`` in the block, let ``s_r`` be the candidate emitted
    by ``torch.sort`` and ``T_r`` the candidate set emitted by
    ``torch.topk(..., k=r)``.  For a local label assignment ``x``:

    ``A(x) = sum_r y[s_r] / r`` and
    ``B(x) = sum_r y[s_r] * sum_{j in T_r intersect block} y[j] / r``.

    If ``C`` relevant candidates occur in higher-score blocks, the block's AP
    numerator contribution is exactly ``C * A(x) + B(x)``.
    """

    position = {index: offset for offset, index in enumerate(block)}
    known_relevant_mask = 0
    uncertain_positions: list[int] = []
    for offset, index in enumerate(block):
        label = labels[index]
        if label == 1:
            known_relevant_mask |= 1 << offset
        elif label is None:
            uncertain_positions.append(offset)

    ranked_positions = [position[index] for index in block]
    top_masks: list[int] = []
    block_set = set(block)
    for rank in range(first_rank, first_rank + len(block)):
        mask = 0
        for index in top_sets[rank - 1] & block_set:
            mask |= 1 << position[index]
        top_masks.append(mask)

    terms: list[tuple[int, float, float]] = []
    for choice in range(1 << len(uncertain_positions)):
        relevant_mask = known_relevant_mask
        for bit, offset in enumerate(uncertain_positions):
            if choice & (1 << bit):
                relevant_mask |= 1 << offset
        a_term = 0.0
        b_term = 0.0
        for local_rank, sorted_position in enumerate(ranked_positions):
            if not relevant_mask & (1 << sorted_position):
                continue
            rank = first_rank + local_rank
            a_term += 1.0 / rank
            b_term += (relevant_mask & top_masks[local_rank]).bit_count() / rank
        terms.append((choice.bit_count(), a_term, b_term))
    return terms


def _ap_bounds_tie_block_dp_from_structure(
    labels: Sequence[int | None],
    *,
    top_sets: Sequence[set[int]],
    blocks: Sequence[Sequence[int]],
) -> tuple[float, float]:
    """Exact block DP after the released Torch tie structure is frozen."""

    _validate_relaxed_labels(labels)
    minimum: dict[int, float] = {0: 0.0}
    maximum: dict[int, float] = {0: 0.0}
    fixed_prior = 0
    first_rank = 1
    for block in blocks:
        terms = _block_assignment_terms(
            block,
            first_rank=first_rank,
            top_sets=top_sets,
            labels=labels,
        )
        next_minimum: dict[int, float] = {}
        next_maximum: dict[int, float] = {}
        for selected_prior in minimum:
            relevant_prior = fixed_prior + selected_prior
            local_minimum: dict[int, float] = {}
            local_maximum: dict[int, float] = {}
            for selected_local, a_term, b_term in terms:
                contribution = relevant_prior * a_term + b_term
                local_minimum[selected_local] = min(
                    local_minimum.get(selected_local, inf), contribution
                )
                local_maximum[selected_local] = max(
                    local_maximum.get(selected_local, -inf), contribution
                )
            for selected_local in local_minimum:
                selected = selected_prior + selected_local
                next_minimum[selected] = min(
                    next_minimum.get(selected, inf),
                    minimum[selected_prior] + local_minimum[selected_local],
                )
                next_maximum[selected] = max(
                    next_maximum.get(selected, -inf),
                    maximum[selected_prior] + local_maximum[selected_local],
                )
        minimum, maximum = next_minimum, next_maximum
        fixed_prior += sum(labels[index] == 1 for index in block)
        first_rank += len(block)

    fixed_total = sum(label == 1 for label in labels)
    lower = inf
    upper = -inf
    for selected in minimum:
        denominator = max(fixed_total + selected, 1)
        lower = min(lower, minimum[selected] / denominator)
        upper = max(upper, maximum[selected] / denominator)
    return float(lower), float(upper)


def ap_bounds_tie_block_dp(
    predicted: Sequence[float],
    labels: Sequence[int | None],
    *,
    max_uncertain_per_block: int = 20,
    max_transition_scans: int = 50_000_000,
) -> tuple[float, float]:
    """Exact AP extrema under released Torch top-k/sort tie semantics.

    The cross-block state is the number of ambiguous positives selected in
    higher-score blocks.  Work is ``O(sum_b 2**u_b * U_before_b)`` after the
    evaluator's ``O(n**2)`` top-k set precomputation; ``u_b`` is the number of
    ambiguous labels in block ``b``.  Limits are explicit so callers can use a
    rigorous conservative fallback instead of an unbounded computation.
    """

    if len(predicted) != len(labels) or not labels:
        raise ValueError("predicted/labels must be non-empty and equal length")
    if max_uncertain_per_block < 0 or max_transition_scans < 1:
        raise ValueError("tie-block complexity limits must be positive")
    _validate_relaxed_labels(labels)
    _, top_sets, blocks, _ = _torch_released_ap_structure(predicted)
    maximum_in_block, transition_scans = _tie_block_transition_estimate(
        blocks, labels
    )
    if maximum_in_block > max_uncertain_per_block:
        raise ValueError(
            "ambiguous labels in one Torch tie block exceed exact-DP limit: "
            f"{maximum_in_block} > {max_uncertain_per_block}"
        )
    if transition_scans > max_transition_scans:
        raise ValueError(
            "tie-block exact-DP transition budget exceeded: "
            f"{transition_scans} > {max_transition_scans}"
        )
    return _ap_bounds_tie_block_dp_from_structure(
        labels, top_sets=top_sets, blocks=blocks
    )


def exhaustive_label_metric_bounds(
    predicted: Sequence[float], labels: Sequence[int | None]
) -> dict[str, tuple[float, float]]:
    """Exact released P@10/AP bounds by enumerating ambiguous labels."""

    uncertain = [index for index, label in enumerate(labels) if label is None]
    if len(uncertain) > 20:
        raise ValueError("exhaustive released-evaluator audit is limited to 20 labels")
    pred = np.asarray(predicted, dtype=float)
    numpy_order = np.argsort(pred)[::-1]
    p10_values: list[float] = []
    ap_values: list[float] = []
    for choices in product((0, 1), repeat=len(uncertain)):
        concrete = np.asarray(
            [0 if label is None else int(label) for label in labels], dtype=bool
        )
        for index, choice in zip(uncertain, choices, strict=True):
            concrete[index] = bool(choice)
        p10_values.append(
            float(concrete[numpy_order[: min(10, pred.size)]].sum() / 10.0)
        )
        ap_values.append(released_average_precision(pred, concrete))
    return {
        "P@10": (min(p10_values), max(p10_values)),
        "MAP": (min(ap_values), max(ap_values)),
    }


def label_metric_bounds(
    predicted: Sequence[float], labels: Sequence[int | None]
) -> tuple[dict[str, tuple[float, float]], str]:
    """Return exact P@10/AP bounds and the AP proof method used."""

    pred = np.asarray(predicted, dtype=float)
    if pred.size != len(labels) or pred.size == 0:
        raise ValueError("predicted/labels must be non-empty and equal length")
    numpy_order = np.argsort(pred)[::-1]
    top = numpy_order[: min(10, pred.size)]
    fixed = sum(labels[int(index)] == 1 for index in top)
    uncertain_top = sum(labels[int(index)] is None for index in top)
    p10_bounds = (fixed / 10.0, (fixed + uncertain_top) / 10.0)

    order, top_sets, blocks, compatible = _torch_released_ap_structure(pred)
    if compatible:
        ap_bounds = _ap_bounds_prefix_dp_from_order(order, labels)
        proof = "exact_prefix_dynamic_program"
    else:
        maximum_in_block, transition_scans = _tie_block_transition_estimate(
            blocks, labels
        )
        if maximum_in_block > 20:
            ap_bounds = (0.0, 1.0)
            proof = "conservative_unit_interval_due_to_oversized_tie_block"
        elif transition_scans > 50_000_000:
            ap_bounds = (0.0, 1.0)
            proof = "conservative_unit_interval_due_to_tie_block_transition_budget"
        else:
            ap_bounds = _ap_bounds_tie_block_dp_from_structure(
                labels, top_sets=top_sets, blocks=blocks
            )
            proof = "exact_tie_block_dynamic_program_released_topk"
    return {"P@10": p10_bounds, "MAP": ap_bounds}, proof


def possible_true_max_indices(
    lower: Sequence[float], upper: Sequence[float]
) -> list[int]:
    """Candidates that can be the evaluator's first true-similarity maximum.

    Original-order tie breaking is respected: an earlier candidate must be
    able to fall strictly below the candidate, while a later candidate need
    only be able to fall to an equal value.
    """

    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if lo.ndim != 1 or lo.shape != hi.shape or lo.size == 0:
        raise ValueError("lower/upper must be non-empty equal-length vectors")
    if np.any(lo > hi):
        raise ValueError("lower truth bound exceeds upper truth bound")
    possible: list[int] = []
    for index in range(lo.size):
        candidate_upper = hi[index]
        if np.any(lo[:index] >= candidate_upper):
            continue
        if np.any(lo[index + 1 :] > candidate_upper):
            continue
        possible.append(index)
    if not possible:
        raise AssertionError("at least one candidate must be a possible maximum")
    return possible


def mrr_bounds(
    predicted: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> tuple[float, float]:
    """Conservative-exact interval-relaxation MRR bounds for one query."""

    pred = np.asarray(predicted, dtype=float)
    if pred.size != len(lower):
        raise ValueError("predicted and truth bounds differ in length")
    order = np.argsort(pred)[::-1]
    ranks = np.empty(pred.size, dtype=int)
    ranks[order] = np.arange(1, pred.size + 1)
    reciprocals = [1.0 / int(ranks[index]) for index in possible_true_max_indices(lower, upper)]
    return min(reciprocals), max(reciprocals)
