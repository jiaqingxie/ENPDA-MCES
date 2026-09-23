"""Independent hard-score checks for the protein OOD experiment."""

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("protein_ood", SCRIPTS / "run_enpda_protein_ood.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_known_planted_optimum_by_exhaustive_matching():
    # Path with one edge removed and its vertices independently relabeled.
    pair = {"num_nodes": 4, "source_edges": [[0, 1], [1, 2], [2, 3]],
            "target_edges": [[0, 3], [1, 2]]}
    scores = [module.score_mapping(pair, list(p)) for p in itertools.permutations(range(4))]
    assert max(scores) == len(pair["target_edges"]) == 2
    assert min(scores) == 0


def test_partial_mapping_and_isolated_vertices():
    pair = {"num_nodes": 4, "source_edges": [[0, 1], [1, 2]], "target_edges": [[0, 1]]}
    assert module.score_mapping(pair, [1, 0, -1, 3]) == 1
    assert module.score_mapping(pair, [-1, -1, -1, -1]) == 0


def test_invalid_maps_are_rejected():
    pair = {"num_nodes": 3, "source_edges": [[0, 1]], "target_edges": [[0, 1]]}
    for mapping in ([0, 0, 1], [0, 1], [0, 1, 3], [0, 1, -2]):
        with pytest.raises(ValueError):
            module.score_mapping(pair, mapping)
