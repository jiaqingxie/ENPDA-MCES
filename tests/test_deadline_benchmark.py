import json
from pathlib import Path

import torch

from nema import deadline_benchmark as adapters
from nema.graph import GraphPair, LabeledGraph


def pair():
    graph = LabeledGraph(torch.tensor([6, 8, 6]),
        torch.tensor([[0, 1], [1, 2]]), torch.tensor([1, 1]))
    return GraphPair(graph, graph, key='toy')


def test_early_partial_survives_late_better_answer(tmp_path, monkeypatch):
    now = [1.]
    monkeypatch.setattr(adapters.time, 'perf_counter', lambda: now[0])
    journal = adapters.Journal(pair(), tmp_path / 'trace.jsonl', 0., 60.)
    assert journal.publish([0, 1, -1], 'early_partial')
    now[0] = 60.001
    assert not journal.publish([0, 1, 2], 'late_optimum')
    rows = [json.loads(s) for s in (tmp_path / 'trace.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['common_edges'] == 1
    assert rows[0]['source'] == 'early_partial'


def test_mapping_completed_at_deadline_is_eligible(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters.time, 'perf_counter', lambda: 60.)
    journal = adapters.Journal(pair(), tmp_path / 'trace.jsonl', 0., 60.)
    assert journal.publish([0, 1, 2], 'at_deadline')
    assert journal.best == (2, 3)


def test_classical_callbacks_produce_valid_incumbents(tmp_path):
    import time
    for method in ('fmcs', 'mcsplit'):
        path = tmp_path / (method + '.jsonl')
        journal = adapters.Journal(pair(), path, time.perf_counter(), 1.)
        if method == 'fmcs':
            adapters.fmcs(pair(), journal)
        else:
            root = Path(__file__).resolve().parents[1]
            adapters.mcsplit(pair(), journal,
                root / 'artifacts/strict60_20260919/mcsplit/libmcsplit_trace.so')
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        assert rows[-1]['common_edges'] == 2
        assert rows[-1]['completed_at_seconds'] < 1.
