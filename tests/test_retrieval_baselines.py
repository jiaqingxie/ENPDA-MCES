import torch
import pytest

from nema.graph import LabeledGraph
from nema.retrieval_baselines import (
    EncodedGraph,
    build_retrieval_baseline,
    size_upper_bound_similarity,
    wl_cosine_similarity,
)


def toy_graph() -> LabeledGraph:
    return LabeledGraph(
        node_labels=torch.tensor([6, 7, 8]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_labels=torch.tensor([1, 2]),
    )


def test_retrieval_baselines_are_symmetric_and_differentiable() -> None:
    graph = toy_graph()
    encoded = EncodedGraph.from_graph(graph, torch.device("cpu"))
    for name in ("simgnn", "gmn", "neuromatch"):
        model = build_retrieval_baseline(name, hidden_dim=16, layers=2)
        model.eval()
        forward = model(encoded, encoded)
        reverse = model(encoded, encoded)
        assert forward.shape == ()
        assert 0 <= float(forward) <= 1
        assert torch.allclose(forward, reverse)
        forward.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_existing_nonlearned_retrieval_interfaces_are_preserved() -> None:
    graph = toy_graph()
    assert size_upper_bound_similarity(graph, graph) == 1.0
    assert wl_cosine_similarity(graph, graph) == pytest.approx(1.0)
