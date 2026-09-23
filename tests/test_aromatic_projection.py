from rdkit import Chem
from nema.aromatic_projection import AromaticProjection


def graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {'smiles': smiles, 'nodes': [a.GetAtomicNum() for a in mol.GetAtoms()],
            'edges': [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondType())]
                      for b in mol.GetBonds()]}


def test_ring_complete_and_partial():
    g = graph('c1ccccc1CC')
    p = AromaticProjection({'left': g, 'right': g})
    full = p.project(list(range(8)))
    assert full['common_edges'] == 8 and p.verify(full)
    mapping = list(range(8)); mapping[0] = -1
    partial = p.project(mapping)
    assert partial['common_edges'] == 2 and partial['removed_aromatic_edges'] == 4
    assert p.verify(partial)


def test_heterocycle_and_native_subset():
    p = AromaticProjection({'left': graph('c1ccccc1'), 'right': graph('c1ccncc1')})
    assert p.project(list(range(6)))['common_edges'] == 0
    g = graph('c1ccccc1')
    p = AromaticProjection({'left': g, 'right': g})
    partial = p.project(list(range(6)), allowed_edges={(0, 1), (1, 2)})
    assert partial['common_edges'] == 0  # must not invent missing native ring bonds


def test_fused_and_nonaromatic():
    for smiles in ('c1ccc2ccccc2c1', 'CC(=O)NCC', 'c1ccc2[nH]ccc2c1'):
        g = graph(smiles); p = AromaticProjection({'left': g, 'right': g})
        result = p.project(list(range(len(g['nodes']))))
        assert result['common_edges'] == len(g['edges']) and p.verify(result)


def test_verifier_rejects_partial_ring():
    import pytest
    g = graph('c1ccccc1'); p = AromaticProjection({'left': g, 'right': g})
    result = {'mapping': list(range(6)), 'bond_witness': [[0, 1, 0, 1]],
              'common_edges': 1, 'common_nodes': 2}
    with pytest.raises(AssertionError):
        p.verify(result)


def test_projection_finishing_after_deadline_cannot_replace_early_incumbent(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from unittest.mock import patch
    path = Path(__file__).resolve().parents[1] / 'scripts/run_aromatic_ring_trial.py'
    spec = importlib.util.spec_from_file_location('ring_trial_deadline_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    g = graph('c1ccccc1')
    trace = tmp_path / 'events.jsonl'
    journal = module.RingJournal({'left': g, 'right': g}, trace, 0., 60.)
    with patch.object(module.time, 'perf_counter', return_value=59.9):
        assert journal.publish([-1] * 6, 'early_empty')
    with patch.object(module.time, 'perf_counter', return_value=60.1):
        assert not journal.publish(list(range(6)), 'late_complete_ring')
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(events) == 1 and events[0]['common_edges'] == 0
    assert journal.late_candidates == 1
