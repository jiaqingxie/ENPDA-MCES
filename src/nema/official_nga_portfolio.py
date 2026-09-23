"""Extract soft proposals from the commit-pinned official NGA training loop."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import torch
from torch import nn
from torch_geometric.utils import from_networkx, to_dense_adj, to_networkx

from nema.association import AssociationGraph
from nema.data import load_pair, recover_empty_molecular_data
from nema.official_nga import OFFICIAL_COMMIT, _official_modules
from nema.rounding import hungarian_mapping


def train_official_nga_proposal(
    path: str | Path,
    *,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    samples: int = 10,
    time_budget: float = 60.0,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """Run official per-pair NGA and retain its best observable soft sample.

    Sample and epoch selection uses the directly observed preserved-edge
    objective, exactly as the released evaluator; oracle labels are never used.
    The returned matrix is transposed when needed to match AEMA's convention of
    putting the smaller graph on the row side.
    """

    started = time.perf_counter()
    torch.manual_seed(seed)
    model_module, acg_module, utils_module = _official_modules()
    pair = load_pair(path)
    with Path(path).open("rb") as stream:
        left_list, right_list = pickle.load(stream)
    if len(left_list) != 1 or len(right_list) != 1:
        raise ValueError("official NGA proposal extraction expects one graph pair per file")
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
    best_score: torch.Tensor | None = None
    best_epoch = 0
    best_sample = -1
    best_loss = None
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
                if edge_count >= best_edges:
                    best_edges = edge_count
                    best_score = assignment[sample_index].detach().cpu().float()
                    best_epoch = epoch
                    best_sample = sample_index
                    best_loss = float(loss.detach())
        if budget_exhausted:
            break

    if best_score is None:
        raise RuntimeError("official NGA stopped before producing a soft proposal")
    left, right, swapped = pair.oriented()
    if swapped:
        best_score = best_score.transpose(0, 1).contiguous()
    association = AssociationGraph.build(left, right)
    if tuple(best_score.shape) != association.shape:
        raise RuntimeError(
            f"oriented official score shape {tuple(best_score.shape)} != {association.shape}"
        )
    best_score = best_score.clamp_min(1e-12)
    proposal_stats = association.hard_statistics(hungarian_mapping(best_score))
    return best_score, {
        "proposal_source": "official_pair_trained_NGA",
        "official_commit": OFFICIAL_COMMIT,
        "official_model": "MCS2",
        "training_seed": seed,
        "completed_epochs": completed_epochs,
        "selected_epoch": best_epoch,
        "selected_sample": best_sample,
        "official_selected_edge_count": best_edges,
        "oriented_hungarian_edges": proposal_stats[0],
        "oriented_hungarian_nodes": proposal_stats[1],
        "negative_training_loss": -best_loss if best_loss is not None else None,
        "training_runtime_seconds": time.perf_counter() - started,
        "time_budget_seconds": time_budget,
        "budget_exhausted": completed_epochs < epochs,
        "smiles_recovery_applied": source_recovered or target_recovered,
        "score_orientation_swapped": swapped,
    }
