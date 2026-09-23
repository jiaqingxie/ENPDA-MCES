"""Unsupervised amortized training for NEMA."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import torch

from nema.association import AssociationGraph
from nema.features import structural_features
from nema.graph import GraphPair
from nema.models.nema import NEMAModel


def train_nema(
    model: NEMAModel,
    pairs: Sequence[GraphPair],
    epochs: int = 20,
    learning_rate: float = 1e-3,
    accumulation: int = 8,
    seed: int = 0,
    device: str = "cpu",
    initial_history: Sequence[dict[str, float]] | None = None,
    optimizer_state: dict | None = None,
    on_epoch: Callable[[list[dict[str, float]], torch.optim.Optimizer], None] | None = None,
) -> list[dict[str, float]]:
    """Maximize the label-free normalized ACG objective across graph pairs."""

    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    history = [dict(record) for record in (initial_history or ())]
    if len(history) >= epochs:
        return history[:epochs]

    print(f"precomputing {len(pairs)} association graphs and equivariant features", flush=True)
    prepared: list[tuple[AssociationGraph, torch.Tensor]] = []
    for position, pair in enumerate(pairs, 1):
        left, right, _ = pair.oriented()
        association = AssociationGraph.build(left, right)
        fixed_features = structural_features(association).to(device)
        prepared.append((association.to(device), fixed_features))
        if position % 100 == 0 or position == len(pairs):
            print(f"prepared {position}/{len(pairs)}", flush=True)

    indices = list(range(len(pairs)))
    for _ in range(len(history)):
        rng.shuffle(indices)

    for epoch in range(len(history) + 1, epochs + 1):
        rng.shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        total_objective = 0.0
        total_loss = 0.0
        for position, index in enumerate(indices, 1):
            association, fixed_features = prepared[index]
            output = model(association, fixed_features=fixed_features, line_search=True)
            normalizer = max(
                2 * min(association.left.num_edges, association.right.num_edges), 1
            )
            normalized = output.objectives[-1] / normalizer
            # Intermediate supervision makes every mirror step useful while staying unsupervised.
            intermediate = output.objectives[1:].mean() / normalizer
            loss = -(0.75 * normalized + 0.25 * intermediate) / accumulation
            loss.backward()
            total_loss += float(loss.detach()) * accumulation
            total_objective += float(normalized.detach())
            if position % accumulation == 0 or position == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        record = {
            "epoch": float(epoch),
            "loss": total_loss / max(len(indices), 1),
            "soft_objective": total_objective / max(len(indices), 1),
        }
        history.append(record)
        print(
            f"epoch {epoch}/{epochs} loss={record['loss']:.6f} "
            f"soft_objective={record['soft_objective']:.6f}",
            flush=True,
        )
        if on_epoch is not None:
            on_epoch(history, optimizer)
    return history
