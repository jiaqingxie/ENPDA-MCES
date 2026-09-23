from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from nema.retrieval_bounds import (
    aggregate_released_point_metrics,
    ap_bounds_prefix_dp,
    ap_bounds_tie_block_dp,
    exhaustive_label_metric_bounds,
    label_metric_bounds,
    mrr_bounds,
    possible_true_max_indices,
    read_jsonl_unique_records,
    released_average_precision,
    validate_oracle_grid,
    validate_relative_code_provenance,
)
from nema.metrics import retrieval_metrics


def brute_ap(predicted: list[float], labels: list[int | None]) -> tuple[float, float]:
    uncertain = [index for index, label in enumerate(labels) if label is None]
    values = []
    for choices in product((0, 1), repeat=len(uncertain)):
        concrete = [0 if label is None else label for label in labels]
        for index, choice in zip(uncertain, choices, strict=True):
            concrete[index] = choice
        values.append(released_average_precision(predicted, concrete))
    return min(values), max(values)


@pytest.mark.parametrize("seed", range(12))
def test_ap_prefix_dp_matches_small_exhaustive_released_evaluator(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # Unique scores make torch.topk prefixes agree with torch.sort while still
    # permuting the prediction order across test cases.
    predicted = (rng.permutation(7) + np.arange(7) * 1e-4).tolist()
    labels: list[int | None] = [1, None, 0, None, 1, None, 0]
    assert ap_bounds_prefix_dp(predicted, labels) == pytest.approx(
        brute_ap(predicted, labels), abs=2e-6
    )


def test_tied_ap_uses_exact_released_topk_tie_block_dp() -> None:
    predicted = [0.0, 3.0, 1.0, 0.0, 3.0]
    labels: list[int | None] = [None, 1, 0, None, 1]
    bounds, proof = label_metric_bounds(predicted, labels)
    exhaustive = exhaustive_label_metric_bounds(predicted, labels)
    assert proof == "exact_tie_block_dynamic_program_released_topk"
    assert bounds["MAP"] == pytest.approx(exhaustive["MAP"], abs=2e-6)
    assert bounds["P@10"] == pytest.approx(exhaustive["P@10"])


@pytest.mark.parametrize("seed", range(24))
def test_tie_block_dp_matches_exhaustive_with_multiple_mixed_ties(seed: int) -> None:
    rng = np.random.default_rng(10_000 + seed)
    size = int(rng.integers(4, 10))
    # Integer-valued scores deliberately create several exact Torch tie blocks.
    predicted = rng.integers(-1, 3, size=size).astype(float).tolist()
    raw_labels = rng.choice(np.asarray([0, 1, 2]), size=size, replace=True)
    labels: list[int | None] = [
        None if int(value) == 2 else int(value) for value in raw_labels
    ]
    # Exercise known and unknown labels in every generated family.
    labels[0] = None
    labels[1] = 1
    labels[2] = 0
    expected = exhaustive_label_metric_bounds(predicted, labels)["MAP"]
    assert ap_bounds_tie_block_dp(predicted, labels) == pytest.approx(
        expected, abs=2e-6
    )
    # Also compare every one of the 2**U concrete label assignments, not only
    # their extrema, so a non-extremal tie-block contribution cannot hide.
    uncertain = [index for index, label in enumerate(labels) if label is None]
    for choices in product((0, 1), repeat=len(uncertain)):
        concrete = [0 if label is None else label for label in labels]
        for index, choice in zip(uncertain, choices, strict=True):
            concrete[index] = choice
        point = released_average_precision(predicted, concrete)
        lower, upper = ap_bounds_tie_block_dp(predicted, concrete)
        assert lower == pytest.approx(point, abs=2e-6)
        assert upper == pytest.approx(point, abs=2e-6)


def test_tie_block_dp_matches_exhaustive_when_all_labels_are_unknown() -> None:
    predicted = [2.0, 2.0, 1.0, 1.0, 1.0, 0.0]
    labels: list[int | None] = [None] * len(predicted)
    assert ap_bounds_tie_block_dp(predicted, labels) == pytest.approx(
        exhaustive_label_metric_bounds(predicted, labels)["MAP"], abs=2e-6
    )


def test_tie_block_dp_complexity_limits_are_explicit() -> None:
    predicted = [1.0, 1.0, 1.0, 0.0]
    labels: list[int | None] = [None, None, None, 1]
    with pytest.raises(ValueError, match="tie block exceed"):
        ap_bounds_tie_block_dp(
            predicted, labels, max_uncertain_per_block=2
        )
    with pytest.raises(ValueError, match="transition budget exceeded"):
        ap_bounds_tie_block_dp(predicted, labels, max_transition_scans=1)


def test_label_metric_bounds_names_oversized_tie_block_fallback() -> None:
    predicted = [1.0] * 21 + [0.0]
    labels: list[int | None] = [None] * 21 + [0]
    bounds, proof = label_metric_bounds(predicted, labels)
    assert proof == "conservative_unit_interval_due_to_oversized_tie_block"
    assert bounds["MAP"] == (0.0, 1.0)


def test_p10_bounds_count_only_ambiguous_top10_labels() -> None:
    predicted = list(range(12, 0, -1))
    labels: list[int | None] = [1, None, 0, None, 1, 0, 0, 1, None, 0, None, None]
    bounds, _ = label_metric_bounds(predicted, labels)
    assert bounds["P@10"] == pytest.approx((0.3, 0.6))
    assert bounds["P@10"] == pytest.approx(
        exhaustive_label_metric_bounds(predicted, labels)["P@10"]
    )


def test_mrr_bounds_respect_original_truth_tie_and_prediction_rank() -> None:
    predicted = [0.1, 0.9, 0.5]
    lower = [0.8, 0.7, 0.1]
    upper = [0.8, 0.9, 0.2]
    assert possible_true_max_indices(lower, upper) == [0, 1]
    # Candidate 1 is predicted rank 1; candidate 0 is rank 3.
    assert mrr_bounds(predicted, lower, upper) == pytest.approx((1 / 3, 1.0))

    # Candidate 1 cannot become the first maximum when its upper bound merely
    # ties the earlier exact candidate 0.
    upper[1] = 0.8
    assert possible_true_max_indices(lower, upper) == [0]
    assert mrr_bounds(predicted, lower, upper) == pytest.approx((1 / 3, 1 / 3))


def test_zero_ambiguity_aggregate_matches_retrieval_metrics() -> None:
    rng = np.random.default_rng(20260823)
    predicted = rng.normal(size=24).tolist()
    # Include exact truth ties so the first-max MRR behavior is exercised.
    truth = rng.uniform(size=24)
    truth[1] = truth[4] = 0.95
    truth[13] = truth[17] = 0.91
    truth = truth.tolist()
    ours = aggregate_released_point_metrics(predicted, truth, group_size=12)
    reference = retrieval_metrics(predicted, truth, group_size=12)
    for metric in ("MRR", "P@10", "MAP"):
        assert ours[metric] == pytest.approx(reference[metric], abs=2e-6)


def test_jsonl_reader_rejects_duplicate_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        json.dumps({"key": 1}) + "\n" + json.dumps({"key": "1"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSONL key"):
        read_jsonl_unique_records(path)


def test_oracle_grid_rejects_missing_or_duplicate_candidate_coverage() -> None:
    complete = [
        {"key": query * 2 + candidate + 1, "query_index": query, "candidate_index": candidate}
        for query in range(2)
        for candidate in range(2)
    ]
    assert validate_oracle_grid(complete) == (2, 2)
    with pytest.raises(ValueError, match="differs from manifest"):
        validate_oracle_grid(complete, expected_queries=20, expected_candidates=500)
    broken = [dict(record) for record in complete]
    broken[-1]["candidate_index"] = 0
    with pytest.raises(ValueError, match="duplicate oracle query/candidate pair"):
        validate_oracle_grid(broken)


def test_code_provenance_uses_repository_relative_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    absolute_dunder_file = (tmp_path / "scripts" / "audit.py").resolve()
    absolute_dunder_file.parent.mkdir()
    absolute_dunder_file.write_text("print('ok')\n", encoding="utf-8")
    digest = hashlib.sha256(absolute_dunder_file.read_bytes()).hexdigest()
    # Simulate H100 Python exposing an absolute __file__, then deliberately
    # canonicalize it to the repository-relative key written by provenance.
    canonical = absolute_dunder_file.relative_to(tmp_path)
    assert canonical == Path("scripts/audit.py")
    validate_relative_code_provenance({"scripts/audit.py": digest}, [canonical])
    with pytest.raises(ValueError, match="repository-relative"):
        validate_relative_code_provenance(
            {str(absolute_dunder_file): digest}, [absolute_dunder_file]
        )
