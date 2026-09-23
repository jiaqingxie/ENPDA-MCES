"""Thin adapter around the commit-pinned official NGA implementation."""

from __future__ import annotations

import importlib
import pickle
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch_geometric.utils import from_networkx, to_dense_adj, to_networkx

from nema.data import load_pair, recover_empty_molecular_data
from nema.solvers import SolveResult


OFFICIAL_COMMIT = "e4a8f1f9ec9e31f79f3fbd648717dfbb9fe113fc"


def _official_modules(vendor_root: str | Path | None = None):
    if vendor_root is None:
        vendor_root = Path(__file__).resolve().parents[2] / "vendor" / "nga-official"
    vendor_root = Path(vendor_root).resolve()
    if not (vendor_root / "model.py").exists():
        raise FileNotFoundError(
            f"official NGA checkout not found at {vendor_root}; run git submodule update --init"
        )
    path = str(vendor_root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return (
        importlib.import_module("model"),
        importlib.import_module("ACG"),
        importlib.import_module("utils"),
    )


def solve_official_nga_path(
    path: str | Path,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    samples: int = 10,
    time_budget: float = 60.0,
    seed: int = 0,
    device: str = "cpu",
    vendor_root: str | Path | None = None,
) -> SolveResult:
    """Run the official ``MCS2`` model and its exact training/evaluation loop."""

    started = time.perf_counter()
    torch.manual_seed(seed)
    model_module, acg_module, utils_module = _official_modules(vendor_root)
    pair = load_pair(path)
    with Path(path).open("rb") as stream:
        left_list, right_list = pickle.load(stream)
    if len(left_list) != 1 or len(right_list) != 1:
        raise ValueError("official evaluation expects one graph pair per file")
    data_s, source_recovered = recover_empty_molecular_data(left_list[0])
    data_t, target_recovered = recover_empty_molecular_data(right_list[0])
    data_s = data_s.to(device)
    data_t = data_t.to(device)
    data_s.edge_label = data_s.edge_attr
    data_t.edge_label = data_t.edge_attr
    graph_s = to_networkx(
        data_s, to_undirected=True, node_attrs=["x"], edge_attrs=["edge_label"]
    )
    graph_t = to_networkx(
        data_t, to_undirected=True, node_attrs=["x"], edge_attrs=["edge_label"]
    )
    graph_st, _ = acg_module.association_common_graph(graph_s, graph_t)
    for node in graph_st.nodes():
        graph_st.nodes[node]["label"] = node
    data_st = from_networkx(graph_st).to(device)
    # ``MCS2`` uses ``data_st.x`` only to construct the node-label compatibility
    # mask.  PyG's NetworkX converter can leave this field unset when one of the
    # product attributes is not stackable (this occurs in a few released pairs).
    # Reconstruct the exact intended tensor from the product-node order instead
    # of changing any part of the official model.
    product_labels = []
    for source_node, target_node in graph_st.nodes():
        source_label = torch.as_tensor(graph_s.nodes[source_node]["x"]).reshape(-1)[0]
        target_label = torch.as_tensor(graph_t.nodes[target_node]["x"]).reshape(-1)[0]
        product_labels.append((source_label.item(), target_label.item()))
    data_st.x = torch.tensor(product_labels, dtype=data_s.x.dtype, device=device)

    model = model_module.MCS2(
        emb_dim=32,
        hidden_dim=32,
        num_layers=8,
        Ns=graph_s.number_of_nodes(),
        Nt=graph_t.number_of_nodes(),
        sample_num=samples,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    adjacency_st = to_dense_adj(
        data_st.edge_index, max_num_nodes=data_s.num_nodes * data_t.num_nodes
    )[0]

    best_edges = -1
    best_nodes = 0
    best_similarity = 0.0
    best_mapping: list[int] = []
    best_soft = None
    completed_epochs = 0
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, assignment = model(data_s, data_t, data_st, adjacency_st)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_epochs = epoch

        elapsed = time.perf_counter() - started
        budget_exhausted = time_budget > 0 and elapsed >= time_budget
        # The released loop decodes during the second half of its 200-epoch
        # upper bound.  The paper additionally imposes 60 seconds per
        # unsupervised instance.  If a large graph reaches that wall-clock
        # limit earlier, decode the current official assignment once so the
        # time budget remains real rather than silently running to epoch 100.
        if epoch >= epochs / 2 or budget_exhausted or epoch == epochs:
            with torch.no_grad():
                hard = utils_module.hungarian(
                    assignment,
                    torch.tensor([assignment.shape[1]] * assignment.shape[0], device=device),
                    torch.tensor([assignment.shape[2]] * assignment.shape[0], device=device),
                )
                flat = hard.reshape(samples, -1, 1)
                mask = flat * flat.transpose(-1, -2)
                predicted = adjacency_st.unsqueeze(0) * mask
                edge_counts = torch.sum(predicted > 0, dim=(1, 2)) // 2
                sample_index = int(torch.argmax(edge_counts))
                edge_count = int(edge_counts[sample_index])
                node_count = int((predicted[sample_index].sum(dim=-1) > 0).sum())
                similarity = float(
                    (edge_count + node_count) ** 2
                    / (
                        (data_s.num_nodes + data_s.num_edges / 2)
                        * (data_t.num_nodes + data_t.num_edges / 2)
                    )
                )
                if edge_count >= best_edges:
                    best_edges = edge_count
                    best_nodes = node_count
                    best_similarity = similarity
                    best_soft = float(-loss.detach())
                    chosen = hard[sample_index]
                    selected_columns = torch.argmax(chosen, dim=-1)
                    # pygmtools pads the smaller side of a rectangular
                    # assignment with all-zero rows.  A raw argmax would encode
                    # every such unmatched row as column zero and make the
                    # serialized mapping look non-injective.  Preserve the
                    # official hard matrix exactly while representing those
                    # rows explicitly as unmatched.
                    selected_columns[chosen.sum(dim=-1) == 0] = -1
                    best_mapping = selected_columns.cpu().tolist()
        if budget_exhausted:
            break

    if best_edges < 0:
        raise RuntimeError("official NGA stopped before producing an evaluated assignment")
    accuracy = best_edges / pair.true_edges if pair.true_edges else None
    squared_error = (
        (best_similarity - pair.true_similarity) ** 2
        if pair.true_similarity is not None
        else None
    )
    return SolveResult(
        method="NGA-official",
        key=pair.key,
        common_edges=best_edges,
        common_nodes=best_nodes,
        similarity=best_similarity,
        runtime_seconds=time.perf_counter() - started,
        mapping=best_mapping,
        true_edges=pair.true_edges,
        true_nodes=pair.true_nodes,
        true_similarity=pair.true_similarity,
        accuracy=accuracy,
        similarity_squared_error=squared_error,
        soft_objective=best_soft,
        metadata={
            "official_commit": OFFICIAL_COMMIT,
            "official_model": "MCS2",
            "epochs": completed_epochs,
            "samples": samples,
            "seed": seed,
            "time_budget_seconds": time_budget,
            "budget_exhausted": completed_epochs < epochs,
            "mapping_encoding": "source_to_target; -1 denotes an unmatched source row",
            "input_provenance": pair.metadata,
            "smiles_recovery_applied": source_recovered or target_recovered,
        },
    )
