"""Classical reference adapters; their original objectives are explicitly preserved.

FMCS maximizes connected common bonds. McSplit maximizes induced common vertices.
Both return legal partial maps evaluated by the unrestricted MCES edge objective.
Timeouts are cooperative, and full measured latency is always recorded.
"""
from __future__ import annotations
import ctypes
import time
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from nema.graph import GraphPair

def score_mapping(pair: GraphPair, mapping: list[int]) -> int:
    assert len(mapping) == pair.left.num_nodes
    targets = [v for v in mapping if v >= 0]
    assert len(targets) == len(set(targets))
    for u, v in enumerate(mapping):
        assert -1 <= v < pair.right.num_nodes
        if v >= 0:
            assert int(pair.left.node_labels[u]) == int(pair.right.node_labels[v])
    target_edges = {tuple(sorted((u,v))): int(label) for (u,v),label in zip(pair.right.edge_index.t().tolist(),pair.right.edge_labels.tolist())}
    return sum(mapping[u] >= 0 and mapping[v] >= 0 and target_edges.get(tuple(sorted((mapping[u],mapping[v])))) == label
               for (u,v),label in zip(pair.left.edge_index.t().tolist(),pair.left.edge_labels.tolist()))

def _fmcs_molecule(graph, side):
    editable = Chem.RWMol()
    for label in graph.node_labels.tolist():
        atom = Chem.Atom(6)
        atom.SetIsotope(int(label)+1)
        atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    # Native edge labels are RDKit bond-type integers; universal nonmolecular labels use single bonds.
    for (u,v),label in zip(graph.edge_index.t().tolist(),graph.edge_labels.tolist()):
        editable.AddBond(u,v,Chem.BondType.values[int(label)] if label else Chem.BondType.SINGLE)
    mol = editable.GetMol()
    mol.SetIntProp('_enpda_side', side)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol

def fmcs(pair: GraphPair, budget: float) -> dict:
    start=time.perf_counter(); deadline=start+budget
    best_map=[-1]*pair.left.num_nodes; best_score=0; best_time=0.0
    class Accept(rdFMCS.MCSAcceptance):
        def __call__(self,mol1,mol2,atoms,bonds,params):
            nonlocal best_map,best_score,best_time
            if time.perf_counter() >= deadline: return False
            mapping=[-1]*pair.left.num_nodes
            swap=mol1.GetIntProp('_enpda_side') != 0
            for u,v in atoms:
                if swap: u,v=v,u
                mapping[u]=v
            score=score_mapping(pair,mapping)
            now=time.perf_counter()
            if now < deadline and score > best_score:
                best_map,best_score,best_time=mapping,score,now-start
            return now < deadline
    class Progress(rdFMCS.MCSProgress):
        def __call__(self,stat,params): return time.perf_counter() < deadline
    p=rdFMCS.MCSParameters(); p.MaximizeBonds=True; p.Timeout=60
    p.AtomTyper=rdFMCS.AtomCompare.CompareIsotopes
    p.BondTyper=rdFMCS.BondCompare.CompareOrderExact
    p.ShouldAcceptMCS=Accept(); p.ProgressCallback=Progress()
    molecules=[_fmcs_molecule(pair.left,0),_fmcs_molecule(pair.right,1)]
    result=None
    if time.perf_counter() < deadline: result=rdFMCS.FindMCS(molecules,p)
    return {'mapping':best_map,'common_edges':best_score,'runtime_seconds':time.perf_counter()-start,
            'incumbent_seconds':best_time,'timed_out':result is None or bool(result.canceled),
            'objective':'connected maximum common bonds','budget_seconds':budget}

class McSplit:
    def __init__(self, library: Path):
        self.lib=ctypes.CDLL(str(library.resolve()))
        ptr=ctypes.POINTER(ctypes.c_int)
        self.lib.run_mcsplit.argtypes=[ctypes.c_int,ptr,ctypes.c_int,ptr,ctypes.c_int,ptr,ctypes.c_int,ptr,ctypes.c_double,ptr]
        self.lib.run_mcsplit.restype=ctypes.c_int
    def __call__(self,pair: GraphPair,budget: float) -> dict:
        start=time.perf_counter()
        arrays=[]
        for g in (pair.left,pair.right):
            labels=np.asarray(g.node_labels.tolist(),dtype=np.int32)
            edges=np.asarray([(u,v,label+1) for (u,v),label in zip(g.edge_index.t().tolist(),g.edge_labels.tolist())],dtype=np.int32).reshape(-1,3)
            arrays.extend([labels,edges])
        ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        out=np.full(pair.left.num_nodes,-1,dtype=np.int32)
        remaining=budget-(time.perf_counter()-start)
        timed_out=True
        if remaining>0:
            timed_out=bool(self.lib.run_mcsplit(pair.left.num_nodes,ptr(arrays[0]),pair.left.num_edges,ptr(arrays[1]),pair.right.num_nodes,ptr(arrays[2]),pair.right.num_edges,ptr(arrays[3]),remaining,ptr(out)))
        mapping=out.tolist(); score=score_mapping(pair,mapping)
        return {'mapping':mapping,'common_edges':score,'runtime_seconds':time.perf_counter()-start,
                'timed_out':timed_out,'objective':'maximum common induced vertices','budget_seconds':budget}
