"""Train-once/test-many adaptation of the official NGA ``MCS2`` model.

The neural architecture is unchanged.  The only methodological change is that
one parameter vector is optimized across training graph pairs and then frozen
at evaluation time, instead of constructing and optimizing fresh parameters
for every test pair.  This module intentionally keeps that adaptation separate
from NEMA so it can serve as an auditable amortization control.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch_geometric.data import Data

from nema.association import AssociationGraph
from nema.graph import GraphPair, LabeledGraph
from nema.official_nga import OFFICIAL_COMMIT, _official_modules
from nema.solvers import SolveResult, _finalize


@dataclass(frozen=True)
class PreparedAmortizedNGAPair:
    """A graph pair in the tensor layout consumed by official NGA ``MCS2``."""

    pair: GraphPair
    association: AssociationGraph
    source: Data
    target: Data
    product: Data
    product_adjacency: torch.Tensor

    @property
    def normalizer(self) -> int:
        return max(
            2 * min(self.association.left.num_edges, self.association.right.num_edges),
            1,
        )

    def to(self, device: str | torch.device) -> "PreparedAmortizedNGAPair":
        return replace(
            self,
            association=self.association.to(device),
            source=self.source.to(device),
            target=self.target.to(device),
            product=self.product.to(device),
            product_adjacency=self.product_adjacency.to(device),
        )


def _to_bidirected_pyg(graph: LabeledGraph) -> Data:
    if graph.num_edges:
        reverse = graph.edge_index.flip(0)
        edge_index = torch.cat((graph.edge_index, reverse), dim=1)
        edge_attr = torch.cat((graph.edge_labels, graph.edge_labels), dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(0, dtype=torch.long)
    return Data(
        x=graph.node_labels.clone(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=graph.num_nodes,
    )


def prepare_amortized_nga_pair(
    pair: GraphPair,
    device: str | torch.device = "cpu",
) -> PreparedAmortizedNGAPair:
    """Construct the full product-node grid and sparse official-NGA ACG."""

    left, right = pair.left, pair.right
    association = AssociationGraph.build(left, right)
    if association.num_edges:
        edge_index = torch.stack(
            (
                torch.cat((association.edge_u, association.edge_v)),
                torch.cat((association.edge_v, association.edge_u)),
            )
        )
        values = torch.ones(edge_index.shape[1], dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty(0, dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        edge_index,
        values,
        (association.num_candidates, association.num_candidates),
    ).coalesce()
    product_labels = torch.stack(
        (
            left.node_labels[:, None].expand(-1, right.num_nodes).reshape(-1),
            right.node_labels[None, :].expand(left.num_nodes, -1).reshape(-1),
        ),
        dim=1,
    )
    prepared = PreparedAmortizedNGAPair(
        pair=pair,
        association=association,
        source=_to_bidirected_pyg(left),
        target=_to_bidirected_pyg(right),
        product=Data(
            x=product_labels,
            edge_index=edge_index,
            num_nodes=association.num_candidates,
        ),
        product_adjacency=adjacency,
    )
    return prepared.to(device)


def build_official_mcs2(
    samples: int = 1,
    device: str = "cpu",
    vendor_root: str | None = None,
) -> nn.Module:
    """Instantiate the exact released ``MCS2`` parameterization."""

    model_module, _, _ = _official_modules(vendor_root)
    return model_module.MCS2(
        emb_dim=32,
        hidden_dim=32,
        num_layers=8,
        Ns=1,
        Nt=1,
        sample_num=samples,
    ).to(device)


def _set_pair_shape(model: nn.Module, prepared: PreparedAmortizedNGAPair, samples: int) -> None:
    model.Ns = prepared.association.left.num_nodes
    model.Nt = prepared.association.right.num_nodes
    model.sample_num = samples


def train_amortized_nga(
    model: nn.Module,
    pairs: Sequence[GraphPair],
    epochs: int = 20,
    learning_rate: float = 1e-3,
    accumulation: int = 8,
    samples: int = 1,
    seed: int = 0,
    device: str = "cpu",
    initial_history: Sequence[dict[str, float]] | None = None,
    optimizer_state: dict | None = None,
    initial_training_seconds: float = 0.0,
    on_epoch: Callable[
        [list[dict[str, float]], torch.optim.Optimizer, float], None
    ]
    | None = None,
) -> tuple[list[dict[str, float]], float]:
    """Fit one official NGA network across pairs without correspondence labels.

    The raw official objective is divided by a pair-size upper bound before
    averaging.  This gives the shared model a favorable, size-balanced training
    signal while preserving the exact per-pair ACG objective.
    """

    if not pairs:
        raise ValueError("at least one training pair is required")
    if accumulation < 1 or samples < 1:
        raise ValueError("accumulation and samples must be positive")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    history = [dict(record) for record in (initial_history or ())]
    if len(history) >= epochs:
        return history[:epochs], initial_training_seconds

    print(f"precomputing {len(pairs)} sparse official-NGA graph pairs", flush=True)
    prepared = []
    for position, pair in enumerate(pairs, 1):
        prepared.append(prepare_amortized_nga_pair(pair, device=device))
        if position % 100 == 0 or position == len(pairs):
            print(f"prepared {position}/{len(pairs)}", flush=True)

    indices = list(range(len(prepared)))
    for _ in range(len(history)):
        rng.shuffle(indices)
    training_started = time.perf_counter()
    for epoch in range(len(history) + 1, epochs + 1):
        rng.shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        total_objective = 0.0
        total_loss = 0.0
        for position, index in enumerate(indices, 1):
            item = prepared[index]
            _set_pair_shape(model, item, samples)
            loss, _ = model(
                item.source,
                item.target,
                item.product,
                item.product_adjacency,
            )
            normalized_loss = loss.reshape(()) / item.normalizer
            (normalized_loss / accumulation).backward()
            total_loss += float(normalized_loss.detach())
            total_objective -= float(normalized_loss.detach())
            if position % accumulation == 0 or position == len(indices):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        elapsed = initial_training_seconds + (time.perf_counter() - training_started)
        record = {
            "epoch": float(epoch),
            "loss": total_loss / len(indices),
            "soft_objective": total_objective / len(indices),
            "cumulative_training_seconds": elapsed,
        }
        history.append(record)
        print(
            f"epoch {epoch}/{epochs} loss={record['loss']:.6f} "
            f"soft_objective={record['soft_objective']:.6f} "
            f"training_seconds={elapsed:.1f}",
            flush=True,
        )
        if on_epoch is not None:
            on_epoch(history, optimizer, elapsed)
    elapsed = initial_training_seconds + (time.perf_counter() - training_started)
    return history, elapsed


class AmortizedNGASolver:
    """Frozen shared official-NGA inference with no per-pair gradient updates."""

    def __init__(
        self,
        model: nn.Module,
        samples: int = 10,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.samples = samples
        self.seed = seed
        self.device = device

    def solve(self, pair: GraphPair) -> SolveResult:
        started = time.perf_counter()
        torch.manual_seed(self.seed)
        item = prepare_amortized_nga_pair(pair, device=self.device)
        _set_pair_shape(self.model, item, self.samples)
        _, _, utils_module = _official_modules()
        with torch.no_grad():
            loss, assignment = self.model(
                item.source,
                item.target,
                item.product,
                item.product_adjacency,
            )
            hard = utils_module.hungarian(
                assignment,
                torch.full(
                    (assignment.shape[0],),
                    assignment.shape[1],
                    dtype=torch.long,
                    device=self.device,
                ),
                torch.full(
                    (assignment.shape[0],),
                    assignment.shape[2],
                    dtype=torch.long,
                    device=self.device,
                ),
            )
            best_mapping = None
            best_stats = (-1, -1)
            best_soft = None
            objectives = item.association.objective(assignment)
            for sample_index in range(assignment.shape[0]):
                chosen = hard[sample_index]
                mapping = torch.argmax(chosen, dim=-1).long()
                mapping[chosen.sum(dim=-1) == 0] = -1
                stats = item.association.hard_statistics(mapping)
                if stats > best_stats:
                    best_mapping = mapping
                    best_stats = stats
                    best_soft = float(objectives[sample_index])
        if best_mapping is None:
            raise RuntimeError("amortized NGA produced no assignment")
        return _finalize(
            "NGA-Amortized",
            pair,
            item.association,
            best_mapping,
            started,
            soft_objective=best_soft,
            metadata={
                "official_commit": OFFICIAL_COMMIT,
                "official_model": "MCS2",
                "samples": self.samples,
                "seed": self.seed,
                "per_pair_gradient_updates": 0,
                "training_protocol": "shared across graph pairs; normalized label-free ACG objective",
                "raw_model_loss": float(loss),
            },
        )
