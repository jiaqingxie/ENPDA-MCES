"""Check adapter maps against independent tiny-graph definitions."""
import itertools
from pathlib import Path
import torch
import pytest
from nema.graph import GraphPair, LabeledGraph
from nema.classical_followups import McSplit, fmcs, score_mapping

def graph(labels,edges):
    return LabeledGraph(torch.tensor(labels),torch.tensor([(u,v) for u,v,_ in edges]).reshape(-1,2).t().long(),torch.tensor([l for _,_,l in edges]).long())

def test_fmcs_exact_labels_and_reversed_molecule_order():
    a=graph([6,7,6,8],[(0,1,1),(1,2,2),(2,3,1)])
    b=graph([7,6,6],[(0,1,2),(1,2,1)])
    for pair in (GraphPair(a,b),GraphPair(b,a)):
        result=fmcs(pair,1)
        assert result['common_edges']==1
        assert score_mapping(pair,result['mapping'])==1
        assert result['incumbent_seconds']<1

def test_mcsplit_preserves_induced_vertex_objective_and_labels():
    lib=Path('artifacts/mcsplit-adapter/libmcsplit.so')
    if not lib.exists():pytest.skip('build pinned McSplit adapter first')
    solver=McSplit(lib)
    examples=[
        (graph([1]*4,[(0,1,0),(1,2,0),(2,3,0)]),graph([1]*4,[(0,1,0),(1,2,0),(2,3,0),(0,3,0)])),
        (graph([1,2,1],[(0,1,1),(1,2,2)]),graph([1,2,1],[(0,1,2),(1,2,1)]))]
    for a,b in examples:
        ae={tuple(sorted((u,v))):l for (u,v),l in zip(a.edge_index.t().tolist(),a.edge_labels.tolist())}
        be={tuple(sorted((u,v))):l for (u,v),l in zip(b.edge_index.t().tolist(),b.edge_labels.tolist())}
        optimum=0
        for mapping in itertools.product(range(-1,b.num_nodes),repeat=a.num_nodes):
            selected=[u for u,v in enumerate(mapping) if v>=0]
            if len({mapping[u] for u in selected})!=len(selected):continue
            if any(a.node_labels[u]!=b.node_labels[mapping[u]] for u in selected):continue
            if any(ae.get((u,v))!=be.get(tuple(sorted((mapping[u],mapping[v])))) for u,v in itertools.combinations(selected,2)):continue
            optimum=max(optimum,len(selected))
        r=solver(GraphPair(a,b),1)
        assert sum(v>=0 for v in r['mapping'])==optimum
        assert r['common_edges']==score_mapping(GraphPair(a,b),r['mapping'])

def test_zero_budget_gives_empty_feasible_map():
    pair=GraphPair(graph([1,1],[(0,1,0)]),graph([1,1],[(0,1,0)]))
    result=fmcs(pair,0)
    assert result['common_edges']==0
    assert result['mapping']==[-1,-1]
