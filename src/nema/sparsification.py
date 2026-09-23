"""Equivariant learned candidate pruning and sparse ACG construction."""

from __future__ import annotations

import math

import torch
from torch import nn

from nema.association import AssociationGraph
from nema.graph import LabeledGraph


def scalable_node_statistics(graph: LabeledGraph) -> torch.Tensor:
    """Linear-time node statistics suitable for graphs with hundreds of nodes."""
    n = graph.num_nodes
    adjacency = [[] for _ in range(n)]
    for k in range(graph.num_edges):
        u, v = int(graph.edge_index[0, k]), int(graph.edge_index[1, k])
        adjacency[u].append(v); adjacency[v].append(u)
    degree = torch.tensor([len(x) for x in adjacency], dtype=torch.float32)
    scale = max(float(n - 1), 1.0)
    mean = torch.zeros(n); std = torch.zeros(n); maximum = torch.zeros(n)
    for i, neighbors in enumerate(adjacency):
        if neighbors:
            values = degree[torch.tensor(neighbors)] / scale
            mean[i] = values.mean(); std[i] = values.std(unbiased=False); maximum[i] = values.max()
    return torch.stack((degree / scale, mean, std, maximum), dim=-1)


def scalable_candidate_features(left: LabeledGraph, right: LabeledGraph) -> torch.Tensor:
    a, b = scalable_node_statistics(left), scalable_node_statistics(right)
    differences = (a[:, None, :] - b[None, :, :]).abs()
    compatibility = left.node_labels[:, None].eq(right.node_labels[None, :]).float().unsqueeze(-1)
    closeness = torch.exp(-8.0 * differences)
    degree_ratio = (
        torch.minimum(a[:, None, :1], b[None, :, :1])
        / torch.maximum(a[:, None, :1], b[None, :, :1]).clamp_min(1e-4)
    )
    return torch.cat((compatibility, closeness, degree_ratio), dim=-1)


class EquivariantCandidateScorer(nn.Module):
    """Positive-order-independent candidate logits with row/column/global context."""
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(4*hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h=self.local(features); row=h.mean(-2,keepdim=True).expand_as(h); col=h.mean(-3,keepdim=True).expand_as(h); glob=h.mean((-3,-2),keepdim=True).expand_as(h)
        return self.output(torch.cat((h,row,col,glob),-1)).squeeze(-1)


def topk_candidate_mask(logits: torch.Tensor, compatibility: torch.Tensor, k: int) -> torch.Tensor:
    """Permutation-equivariant row/column top-k union with compatible support."""
    if k < 1: raise ValueError("k must be positive")
    masked = logits.masked_fill(~compatibility, -torch.inf)
    n, m = masked.shape; result = torch.zeros_like(compatibility)
    row_k=min(k,m); row_index=torch.topk(masked,row_k,dim=1).indices; result.scatter_(1,row_index,True)
    col_k=min(k,n); col_index=torch.topk(masked,col_k,dim=0).indices
    columns=torch.arange(m,device=masked.device).expand(col_k,m); result[col_index,columns]=True
    result &= compatibility
    # Degenerate label vocabularies may leave a row empty; retain its best finite candidate.
    empty=~result.any(dim=1)
    if bool(empty.any()): result[empty,torch.argmax(masked[empty],dim=1)]=True
    return result


def build_pruned_association(left: LabeledGraph, right: LabeledGraph, mask: torch.Tensor) -> AssociationGraph:
    """Build only ACG edges whose endpoint candidates survive ``mask``."""
    expected=(left.num_nodes,right.num_nodes)
    if tuple(mask.shape)!=expected: raise ValueError(f"expected mask {expected}, found {tuple(mask.shape)}")
    mask=mask.detach().cpu().bool() & left.node_labels[:,None].eq(right.node_labels[None,:])
    right_adjacency: dict[int,list[list[int]]] = {}
    for k in range(right.num_edges):
        u,v,label=int(right.edge_index[0,k]),int(right.edge_index[1,k]),int(right.edge_labels[k])
        rows=right_adjacency.setdefault(label,[[] for _ in range(right.num_nodes)])
        rows[u].append(v); rows[v].append(u)
    edge_u=[]; edge_v=[]; m=right.num_nodes
    candidate_columns=[torch.nonzero(mask[i],as_tuple=True)[0].tolist() for i in range(left.num_nodes)]
    candidate_sets=[set(values) for values in candidate_columns]
    for k in range(left.num_edges):
        u,v,label=int(left.edge_index[0,k]),int(left.edge_index[1,k]),int(left.edge_labels[k])
        adjacency=right_adjacency.get(label)
        if adjacency is None: continue
        allowed_v=candidate_sets[v]
        for x in candidate_columns[u]:
            for y in adjacency[x]:
                if y in allowed_v:
                    a,b=u*m+x,v*m+y
                    edge_u.append(min(a,b)); edge_v.append(max(a,b))
    if edge_u:
        # A source edge is stored once and the target adjacency is directed;
        # consequently every compatible oriented edge yields one unique ACG
        # edge and no global Python set/sort is needed.
        tensor_u=torch.tensor(edge_u,dtype=torch.long); tensor_v=torch.tensor(edge_v,dtype=torch.long)
    else:
        tensor_u=torch.empty(0,dtype=torch.long); tensor_v=torch.empty(0,dtype=torch.long)
    return AssociationGraph(left,right,mask,tensor_u,tensor_v)


def certificate_gap(association: AssociationGraph, mapping: torch.Tensor) -> tuple[int,int,float]:
    lower=association.hard_statistics(mapping)[0]
    upper=min(association.left.num_edges,association.right.num_edges)
    gap=max(upper-lower,0)/max(upper,1)
    return lower,upper,gap
