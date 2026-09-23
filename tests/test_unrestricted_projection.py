import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from nema.unrestricted_projection import UnrestrictedProjection


def pair():
    graph = {'nodes': [6] * 6,
             'edges': [[i, (i + 1) % 6, 12] for i in range(6)]}
    return {'left': graph, 'right': graph}


def test_partial_aromatic_ring_is_retained_and_native_witness_not_augmented():
    policy = UnrestrictedProjection(pair())
    result = policy.project([-1, 1, 2, 3, 4, 5])
    assert result['common_edges'] == 4 and policy.verify(result)
    native = policy.project(list(range(6)), {(0, 1), (1, 2)})
    assert native['common_edges'] == 2 and policy.verify(native)


def test_invalid_labels_and_duplicate_targets_or_witnesses():
    p = pair()
    p['right'] = {**p['right'], 'nodes': [7, 6, 6, 6, 6, 6]}
    policy = UnrestrictedProjection(p)
    result = policy.project(list(range(6)))
    assert result['mapping'][0] == -1 and result['common_edges'] == 4
    with pytest.raises(AssertionError):
        policy.project([1, 1, 2, 3, 4, 5])
    result['bond_witness'].append(result['bond_witness'][0])
    with pytest.raises(AssertionError):
        policy.verify(result)


def test_unrestricted_journal_preserves_early_partial_and_rejects_late_better(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/run_aromatic_ring_trial.py'
    spec = importlib.util.spec_from_file_location('unrestricted_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = tmp_path / 'events.jsonl'
    journal = module.RingJournal(pair(), trace, 0, 60, 'unrestricted')
    with patch.object(module.time, 'perf_counter', return_value=59.9):
        assert journal.publish([-1, 1, 2, 3, 4, 5], 'early')
    with patch.object(module.time, 'perf_counter', return_value=60.1):
        assert not journal.publish(list(range(6)), 'late')
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['common_edges'] == 4
    assert journal.late_candidates == 1
