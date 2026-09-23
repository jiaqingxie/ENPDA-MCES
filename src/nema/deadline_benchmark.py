"""Adapters which publish immutable feasible maps before an external deadline.

These are evaluation adapters, not new model definitions. The supervisor owns the
clock and terminates a worker at the deadline; its journal survives termination.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from rdkit.Chem import rdFMCS
from torch import nn
from torch_geometric.utils import from_networkx, to_dense_adj, to_networkx

from nema.association import AssociationGraph
from nema.classical_followups import _fmcs_molecule
from nema.enpda_anytime import search_trace
from nema.official_nga import _official_modules
from nema.sinkhorn import sample_gumbel_like


class Journal:
    """Count compatible edges independently and serialize only timely incumbents."""

    def __init__(self, pair, path: Path, started: float, budget: float):
        self.pair, self.started, self.budget = pair, started, budget
        self.stream = path.open('a', buffering=1)
        self.best = (-1, -1)
        self.events = 0
        self.late_candidates = 0
        self.left_labels = pair.left.node_labels.tolist()
        self.right_labels = pair.right.node_labels.tolist()
        self.edges = [(u, v, int(label)) for (u, v), label in zip(
            pair.left.edge_index.t().tolist(), pair.left.edge_labels.tolist())]
        self.target = {tuple(sorted((u, v))): int(label) for (u, v), label in zip(
            pair.right.edge_index.t().tolist(), pair.right.edge_labels.tolist())}

    def elapsed(self):
        return time.perf_counter() - self.started

    def expired(self):
        return self.elapsed() >= self.budget

    def publish(self, mapping, source: str):
        if isinstance(mapping, torch.Tensor):
            mapping = mapping.detach().cpu().long().tolist()
        mapping = [int(v) for v in mapping]
        assert len(mapping) == len(self.left_labels)
        used = set()
        for u, v in enumerate(mapping):
            assert -1 <= v < len(self.right_labels)
            if v >= 0:
                assert v not in used
                used.add(v)
                # Incompatible isolated assignments preserve no admissible edge.
                # Explicitly omit them to serialize a labelled partial injection.
                if self.left_labels[u] != self.right_labels[v]:
                    mapping[u] = -1
        witness = [(u, v) for u, v, label in self.edges
                   if mapping[u] >= 0 and mapping[v] >= 0
                   and self.target.get(tuple(sorted((mapping[u], mapping[v])))) == label]
        stats = (len(witness), len({v for edge in witness for v in edge}))
        completed = self.elapsed()
        if completed > self.budget:
            self.late_candidates += 1
            return False
        if stats > self.best:
            self.stream.write(json.dumps({'mapping': mapping, 'common_edges': stats[0],
                'common_nodes': stats[1], 'completed_at_seconds': completed,
                'source': source}) + '\n')
            self.stream.flush()
            self.best = stats
            self.events += 1
        return True


def nga(data_s, data_t, journal: Journal, seed: int, *, epochs=200, samples=10,
        decode_every_epoch=False, vendor_root=None):
    """Pinned MCS2 optimization; late Hungarian outputs are never credited.

    The primary arm keeps the released epoch>=100 decoding schedule. The separate
    NGA-anytime arm can decode from epoch one and is explicitly an adaptation.
    """
    model_module, acg_module, utils_module = _official_modules(vendor_root)
    torch.manual_seed(seed)
    device = 'cuda'
    data_s, data_t = data_s.to(device), data_t.to(device)
    data_s.edge_label, data_t.edge_label = data_s.edge_attr, data_t.edge_attr
    gs = to_networkx(data_s, to_undirected=True, node_attrs=['x'], edge_attrs=['edge_label'])
    gt = to_networkx(data_t, to_undirected=True, node_attrs=['x'], edge_attrs=['edge_label'])
    gst, _ = acg_module.association_common_graph(gs, gt)
    for node in gst:
        gst.nodes[node]['label'] = node
    dst = from_networkx(gst).to(device)
    labels = [(torch.as_tensor(gs.nodes[u]['x']).reshape(-1)[0].item(),
               torch.as_tensor(gt.nodes[v]['x']).reshape(-1)[0].item()) for u, v in gst]
    dst.x = torch.tensor(labels, dtype=data_s.x.dtype, device=device)
    model = model_module.MCS2(emb_dim=32, hidden_dim=32, num_layers=8,
        Ns=gs.number_of_nodes(), Nt=gt.number_of_nodes(), sample_num=samples).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    adj = to_dense_adj(dst.edge_index, max_num_nodes=data_s.num_nodes * data_t.num_nodes)[0]
    completed_epochs = 0
    for epoch in range(1, epochs + 1):
        if journal.expired():
            break
        optimizer.zero_grad(set_to_none=True)
        loss, assignment = model(data_s, data_t, dst, adj)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        completed_epochs = epoch
        if journal.expired():
            break
        if decode_every_epoch or epoch >= epochs / 2:
            with torch.no_grad():
                hard = utils_module.hungarian(assignment,
                    torch.tensor([assignment.shape[1]] * samples, device=device),
                    torch.tensor([assignment.shape[2]] * samples, device=device))
                # Transfer and map decoding are included in the deadline.
                hard_cpu = hard.detach().cpu()
                for index, chosen in enumerate(hard_cpu):
                    mapping = chosen.argmax(dim=-1)
                    mapping[chosen.sum(dim=-1) == 0] = -1
                    if not journal.publish(mapping, f'epoch_{epoch}:sample_{index}'):
                        break
    return {'completed_epochs': completed_epochs, 'samples': samples,
            'decode_every_epoch': decode_every_epoch}


def enpda(pair, model, journal: Journal, seed: int, method: str, rotation_seed=0):
    left, right, swapped = pair.oriented()
    association = AssociationGraph.build(left, right)

    def publish(mapping, source):
        mapping = mapping.detach().cpu().long()
        if swapped:
            inverse = torch.full((pair.left.num_nodes,), -1, dtype=torch.long)
            for u, v in enumerate(mapping.tolist()):
                if v >= 0:
                    inverse[v] = u
            mapping = inverse
        return journal.publish(mapping, source)

    streams = []
    generator = torch.Generator(device='cuda').manual_seed(seed)
    core_only = method in ('enpda_core', 'analytic_core')
    analytic = method.startswith('analytic')
    with torch.inference_mode():
        for index in range(1 if core_only else 4):
            if journal.expired():
                break
            mode = 'analytic' if analytic or index == 3 else 'learned'
            noise = None
            if index in (1, 2):
                noise = .5 * sample_gumbel_like(torch.empty(association.shape, device='cuda'), generator)
            output = model(association, mode=mode, initial_noise=noise)
            mapping = output.hard_mapping().detach().cpu()
            source = f'{mode}_{index}'
            publish(mapping, source + ':core')
            scores = output.assignment.detach().cpu()
            streams.append((source, scores, mapping))
    if core_only:
        return {'streams': len(streams)}
    rotation = rotation_seed % len(streams) if streams else 0
    streams = streams[rotation:] + streams[:rotation]
    audits = []
    for index, (source, scores, mapping) in enumerate(streams):
        remaining = journal.budget - journal.elapsed()
        if remaining <= 0:
            break
        allocation = remaining / (len(streams) - index)
        _, audit = search_trace(association, scores, mapping, (allocation,),
            restarts=64, max_passes=30, anneal_steps=2500, lns_steps=250,
            seed=seed + 1009 * index,
            on_candidate=lambda candidate, stage: publish(candidate, source + ':' + stage))
        audits.append(audit)
    return {'streams': len(streams), 'search_audits': audits}


def fmcs(pair, journal: Journal):
    class Accept(rdFMCS.MCSAcceptance):
        def __call__(self, mol1, mol2, atoms, bonds, params):
            if journal.expired():
                return False
            mapping = [-1] * pair.left.num_nodes
            swap = mol1.GetIntProp('_enpda_side') != 0
            for u, v in atoms:
                if swap:
                    u, v = v, u
                mapping[u] = v
            return journal.publish(mapping, 'accepted_common_subgraph')

    class Progress(rdFMCS.MCSProgress):
        def __call__(self, stat, params):
            return not journal.expired()

    params = rdFMCS.MCSParameters()
    params.MaximizeBonds = True
    params.Timeout = 60
    params.AtomTyper = rdFMCS.AtomCompare.CompareIsotopes
    params.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    params.ShouldAcceptMCS, params.ProgressCallback = Accept(), Progress()
    mols = [_fmcs_molecule(pair.left, 0), _fmcs_molecule(pair.right, 1)]
    if journal.expired():
        return {'canceled': True}
    result = rdFMCS.FindMCS(mols, params)
    return {'canceled': bool(result.canceled), 'objective': 'connected common bonds'}


def mcsplit(pair, journal: Journal, library: Path):
    import ctypes
    lib = ctypes.CDLL(str(library.resolve()))
    ptr = ctypes.POINTER(ctypes.c_int)
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ptr)
    lib.run_mcsplit_trace.argtypes = [ctypes.c_int, ptr, ctypes.c_int, ptr,
        ctypes.c_int, ptr, ctypes.c_int, ptr, ctypes.c_double, callback_type]
    lib.run_mcsplit_trace.restype = ctypes.c_int
    arrays = []
    for graph in (pair.left, pair.right):
        labels = np.asarray(graph.node_labels.tolist(), dtype=np.int32)
        edges = np.asarray([(u, v, int(label) + 1) for (u, v), label in zip(
            graph.edge_index.t().tolist(), graph.edge_labels.tolist())], dtype=np.int32).reshape(-1, 3)
        arrays.extend((labels, edges))
    errors = []

    def accepted(n, values):
        try:
            journal.publish([values[i] for i in range(n)], 'native_vertex_incumbent')
        except Exception as exc:
            errors.append(repr(exc))

    callback = callback_type(accepted)
    remaining = journal.budget - journal.elapsed()
    if remaining <= 0:
        return {'canceled': True}
    pointer = lambda a: a.ctypes.data_as(ptr)
    canceled = lib.run_mcsplit_trace(pair.left.num_nodes, pointer(arrays[0]), pair.left.num_edges,
        pointer(arrays[1]), pair.right.num_nodes, pointer(arrays[2]), pair.right.num_edges,
        pointer(arrays[3]), remaining, callback)
    if errors:
        raise RuntimeError(errors)
    return {'canceled': bool(canceled), 'objective': 'induced common vertices'}
