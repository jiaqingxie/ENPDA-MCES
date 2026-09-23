"""Protocol-matched neural retrieval baselines.

The classes in this module are clean PyTorch implementations of the defining
mechanisms of SimGNN, Graph Matching Networks (GMN), and NeuroMatch.  They use
the same categorical molecular graph input and return a scalar similarity in
``[0, 1]`` so that all three can be trained and ranked under one MCES protocol.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nema.graph import LabeledGraph


def size_upper_bound_similarity(left: LabeledGraph, right: LabeledGraph) -> float:
    """Size-only upper-bound baseline retained for the public retrieval protocol."""

    left_size = left.num_nodes + left.num_edges
    right_size = right.num_nodes + right.num_edges
    if not left_size or not right_size:
        return 0.0
    return min(left_size, right_size) / max(left_size, right_size)


def wl_bag(graph: LabeledGraph, iterations: int = 3) -> Counter[str]:
    """Deterministic Weisfeiler--Lehman multiset fingerprint."""

    neighbors: list[list[tuple[int, int]]] = [[] for _ in range(graph.num_nodes)]
    for (source, target), edge_label in zip(
        graph.edge_index.t().tolist(), graph.edge_labels.tolist(), strict=True
    ):
        neighbors[source].append((target, int(edge_label)))
        neighbors[target].append((source, int(edge_label)))
    labels = [str(int(label)) for label in graph.node_labels.tolist()]
    features: Counter[str] = Counter(f"0:{label}" for label in labels)
    for iteration in range(1, iterations + 1):
        updated = []
        for node in range(graph.num_nodes):
            context = ",".join(
                sorted(f"{edge_label}:{labels[neighbor]}" for neighbor, edge_label in neighbors[node])
            )
            digest = hashlib.sha256(f"{labels[node]}|{context}".encode()).hexdigest()[:20]
            updated.append(digest)
            features[f"{iteration}:{digest}"] += 1
        labels = updated
    return features


def cosine_counter(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def wl_cosine_similarity(left: LabeledGraph, right: LabeledGraph) -> float:
    return cosine_counter(wl_bag(left), wl_bag(right))


@dataclass
class EncodedGraph:
    """Device-resident categorical graph used by the retrieval networks."""

    node_labels: torch.Tensor
    edge_index: torch.Tensor
    edge_labels: torch.Tensor

    @classmethod
    def from_graph(cls, graph: LabeledGraph, device: torch.device) -> "EncodedGraph":
        if graph.num_edges:
            reverse = graph.edge_index.flip(0)
            edge_index = torch.cat((graph.edge_index, reverse), dim=1)
            edge_labels = torch.cat((graph.edge_labels, graph.edge_labels), dim=0)
        else:
            edge_index = graph.edge_index
            edge_labels = graph.edge_labels
        return cls(
            node_labels=graph.node_labels.to(device=device, dtype=torch.long),
            edge_index=edge_index.to(device=device, dtype=torch.long),
            edge_labels=edge_labels.to(device=device, dtype=torch.long),
        )


class EdgeMessageLayer(nn.Module):
    """Permutation-equivariant message layer with categorical edge features."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(nodes)
        if edge_index.numel():
            source, target = edge_index
            messages = self.message(torch.cat((nodes[source], edge_features), dim=-1))
            aggregate.index_add_(0, target, messages)
        update = self.update(torch.cat((nodes, aggregate), dim=-1))
        return self.norm(nodes + update)


class MolecularGraphEncoder(nn.Module):
    """Shared edge-aware GNN encoder used by all protocol-matched baselines."""

    def __init__(
        self,
        hidden_dim: int = 64,
        layers: int = 3,
        node_vocab: int = 128,
        edge_vocab: int = 16,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(node_vocab, hidden_dim)
        self.edge_embedding = nn.Embedding(edge_vocab, hidden_dim)
        self.layers = nn.ModuleList(EdgeMessageLayer(hidden_dim) for _ in range(layers))

    def initialize(self, graph: EncodedGraph) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.node_embedding(graph.node_labels.clamp(0, self.node_embedding.num_embeddings - 1))
        edges = self.edge_embedding(graph.edge_labels.clamp(0, self.edge_embedding.num_embeddings - 1))
        return nodes, edges

    def forward(self, graph: EncodedGraph) -> torch.Tensor:
        nodes, edges = self.initialize(graph)
        for layer in self.layers:
            nodes = layer(nodes, graph.edge_index, edges)
        return nodes


class AttentionPool(nn.Module):
    """SimGNN-style context-aware graph pooling."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.context = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        context = torch.tanh(self.context(nodes.mean(dim=0, keepdim=True)))
        weights = torch.sigmoid(nodes @ context.transpose(0, 1))
        return (weights * nodes).sum(dim=0) / nodes.shape[0] ** 0.5


class NeuralTensorSimilarity(nn.Module):
    """Bilinear tensor comparison used by SimGNN."""

    def __init__(self, hidden_dim: int, tensor_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(tensor_dim, hidden_dim, hidden_dim))
        self.block = nn.Linear(2 * hidden_dim, tensor_dim)
        self.bias = nn.Parameter(torch.zeros(tensor_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        bilinear = torch.einsum("d,kde,e->k", left, self.weight, right)
        return F.relu(bilinear + self.block(torch.cat((left, right))) + self.bias)


class SimGNNBaseline(nn.Module):
    """SimGNN: independent GNNs, attention pooling, tensor and node comparison."""

    def __init__(self, hidden_dim: int = 64, layers: int = 3, bins: int = 16) -> None:
        super().__init__()
        self.encoder = MolecularGraphEncoder(hidden_dim, layers)
        self.pool = AttentionPool(hidden_dim)
        self.tensor = NeuralTensorSimilarity(hidden_dim, hidden_dim // 2)
        self.bins = bins
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + bins, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, left: EncodedGraph, right: EncodedGraph) -> torch.Tensor:
        left_nodes = self.encoder(left)
        right_nodes = self.encoder(right)
        left_graph = self.pool(left_nodes)
        right_graph = self.pool(right_nodes)
        tensor_score = self.tensor(left_graph, right_graph)
        cosine = F.normalize(left_nodes, dim=-1) @ F.normalize(right_nodes, dim=-1).t()
        # The histogram is an auxiliary, non-differentiable comparison exactly as
        # in SimGNN; the graph/tensor path remains fully differentiable.
        histogram = torch.histc(cosine, bins=self.bins, min=-1.0, max=1.0)
        histogram = histogram / histogram.sum().clamp_min(1.0)
        return torch.sigmoid(self.head(torch.cat((tensor_score, histogram))).squeeze(-1))


class GraphMatchingNetworkBaseline(nn.Module):
    """GMN with cross-graph attention inside each message-passing layer."""

    def __init__(self, hidden_dim: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.encoder = MolecularGraphEncoder(hidden_dim, 0)
        self.local_layers = nn.ModuleList(EdgeMessageLayer(hidden_dim) for _ in range(layers))
        self.cross_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(layers)
        )
        self.pool = AttentionPool(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, left: EncodedGraph, right: EncodedGraph) -> torch.Tensor:
        left_nodes, left_edges = self.encoder.initialize(left)
        right_nodes, right_edges = self.encoder.initialize(right)
        scale = math.sqrt(left_nodes.shape[-1])
        for local, cross in zip(self.local_layers, self.cross_updates, strict=True):
            left_local = local(left_nodes, left.edge_index, left_edges)
            right_local = local(right_nodes, right.edge_index, right_edges)
            affinity = left_local @ right_local.t() / scale
            left_match = affinity.softmax(dim=1) @ right_local
            right_match = affinity.t().softmax(dim=1) @ left_local
            left_nodes = cross(torch.cat((left_local, left_local - left_match), dim=-1))
            right_nodes = cross(torch.cat((right_local, right_local - right_match), dim=-1))
        left_graph = self.pool(left_nodes)
        right_graph = self.pool(right_nodes)
        comparison = torch.cat((torch.abs(left_graph - right_graph), left_graph * right_graph))
        return torch.sigmoid(self.head(comparison).squeeze(-1))


class NeuroMatchBaseline(nn.Module):
    """NeuroMatch-style non-negative order embeddings adapted to similarity ranking."""

    def __init__(self, hidden_dim: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.encoder = MolecularGraphEncoder(hidden_dim, layers)
        self.pool = AttentionPool(hidden_dim)
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.calibration = nn.Linear(3, 1)

    def graph_embedding(self, graph: EncodedGraph) -> torch.Tensor:
        return self.project(self.pool(self.encoder(graph)))

    def forward(self, left: EncodedGraph, right: EncodedGraph) -> torch.Tensor:
        left_embedding = self.graph_embedding(left)
        right_embedding = self.graph_embedding(right)
        left_in_right = torch.square(F.relu(left_embedding - right_embedding)).mean()
        right_in_left = torch.square(F.relu(right_embedding - left_embedding)).mean()
        symmetric = torch.square(left_embedding - right_embedding).mean()
        features = torch.stack((-left_in_right, -right_in_left, -symmetric))
        return torch.sigmoid(self.calibration(features).squeeze(-1))


def build_retrieval_baseline(name: str, hidden_dim: int = 64, layers: int = 3) -> nn.Module:
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized == "simgnn":
        return SimGNNBaseline(hidden_dim=hidden_dim, layers=layers)
    if normalized == "gmn":
        return GraphMatchingNetworkBaseline(hidden_dim=hidden_dim, layers=layers)
    if normalized == "neuromatch":
        return NeuroMatchBaseline(hidden_dim=hidden_dim, layers=layers)
    raise ValueError(f"unknown retrieval baseline: {name}")
