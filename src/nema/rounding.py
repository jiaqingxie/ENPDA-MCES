"""Hungarian rounding and objective-preserving discrete refinement."""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Callable
from scipy.optimize import linear_sum_assignment

from nema.association import AssociationGraph


def hungarian_mapping(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be a matrix")
    row, col = linear_sum_assignment(-scores.detach().cpu().double().numpy())
    mapping = torch.full((scores.shape[0],), -1, dtype=torch.long)
    mapping[torch.from_numpy(row)] = torch.from_numpy(col)
    return mapping


class HardObjective:
    """Fast exact scorer for hard mappings, equivalent to selected ACG edges."""

    def __init__(self, association: AssociationGraph):
        self.association = association
        self.left_edges = [
            (
                int(association.left.edge_index[0, k]),
                int(association.left.edge_index[1, k]),
                int(association.left.edge_labels[k]),
            )
            for k in range(association.left.num_edges)
        ]
        self.right_edges = {
            (
                min(
                    int(association.right.edge_index[0, k]),
                    int(association.right.edge_index[1, k]),
                ),
                max(
                    int(association.right.edge_index[0, k]),
                    int(association.right.edge_index[1, k]),
                ),
            ): int(association.right.edge_labels[k])
            for k in range(association.right.num_edges)
        }
        self.mask = association.candidate_mask.cpu().numpy()
        if self.left_edges:
            edge_array = np.asarray(self.left_edges, dtype=np.int64)
            self.left_u = edge_array[:, 0]
            self.left_v = edge_array[:, 1]
            self.left_labels = edge_array[:, 2]
        else:
            self.left_u = np.empty(0, dtype=np.int64)
            self.left_v = np.empty(0, dtype=np.int64)
            self.left_labels = np.empty(0, dtype=np.int64)
        self.right_label_matrix = np.full(
            (association.right.num_nodes, association.right.num_nodes),
            np.iinfo(np.int64).min,
            dtype=np.int64,
        )
        for (u, v), label in self.right_edges.items():
            self.right_label_matrix[u, v] = label
            self.right_label_matrix[v, u] = label

    def __call__(self, mapping: torch.Tensor | np.ndarray | list[int]) -> tuple[int, int]:
        if isinstance(mapping, torch.Tensor):
            values = mapping.tolist()
        elif isinstance(mapping, np.ndarray):
            values = mapping.astype(np.int64, copy=False)
        else:
            values = np.asarray(mapping, dtype=np.int64)
        if isinstance(values, list):
            values = np.asarray(values, dtype=np.int64)
        if values.shape != (self.association.left.num_nodes,):
            raise ValueError("mapping has the wrong number of rows")
        if not self.left_u.size or self.association.right.num_nodes == 0:
            return 0, 0

        mapped_u = values[self.left_u]
        mapped_v = values[self.left_v]
        valid = (
            (mapped_u >= 0)
            & (mapped_u < self.association.right.num_nodes)
            & (mapped_v >= 0)
            & (mapped_v < self.association.right.num_nodes)
        )
        safe_u = np.clip(mapped_u, 0, self.association.right.num_nodes - 1)
        safe_v = np.clip(mapped_v, 0, self.association.right.num_nodes - 1)
        valid &= self.mask[self.left_u, safe_u] & self.mask[self.left_v, safe_v]
        active = valid & (self.right_label_matrix[safe_u, safe_v] == self.left_labels)
        edge_count = int(active.sum())
        if not edge_count:
            return 0, 0
        incident = np.zeros(self.association.left.num_nodes, dtype=bool)
        incident[self.left_u[active]] = True
        incident[self.left_v[active]] = True
        return edge_count, int(incident.sum())


def constructive_mapping(
    association: AssociationGraph,
    scores: torch.Tensor,
    rng: np.random.Generator,
    on_candidate: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Grow an injective common subgraph directly from sparse ACG edges."""

    rows, cols = association.shape
    candidate_count = rows * cols
    adjacency = [set() for _ in range(candidate_count)]
    edge_u = association.edge_u.tolist()
    edge_v = association.edge_v.tolist()
    for u, v in zip(edge_u, edge_v, strict=True):
        adjacency[u].add(v)
        adjacency[v].add(u)
    if not edge_u:
        return hungarian_mapping(scores)

    flat_score = scores.detach().cpu().reshape(-1).numpy()
    edge_quality = np.asarray([flat_score[u] + flat_score[v] for u, v in zip(edge_u, edge_v)])
    top = np.argsort(-edge_quality)[: min(64, len(edge_u))]
    # Mostly favor strong neural seeds while retaining structural diversity.
    seed_position = int(rng.choice(top)) if rng.random() < 0.75 else int(rng.integers(len(edge_u)))
    selected = {edge_u[seed_position], edge_v[seed_position]}
    used_rows = {node // cols for node in selected}
    used_cols = {node % cols for node in selected}

    def emit_partial():
        if on_candidate is not None:
            partial = torch.full((rows,), -1, dtype=torch.long)
            for node in selected:
                partial[node // cols] = node % cols
            on_candidate(partial)

    emit_partial()

    while True:
        connected: list[tuple[int, float, int]] = []
        for node in range(candidate_count):
            row, col = divmod(node, cols)
            if (
                row in used_rows
                or col in used_cols
                or not bool(association.candidate_mask[row, col])
            ):
                continue
            gain = len(adjacency[node] & selected)
            if gain:
                connected.append((gain, float(flat_score[node]), node))
        if connected:
            connected.sort(reverse=True)
            maximum_gain = connected[0][0]
            shortlist = [item for item in connected if item[0] == maximum_gain][:4]
            _, _, chosen = shortlist[int(rng.integers(len(shortlist)))]
            selected.add(chosen)
            used_rows.add(chosen // cols)
            used_cols.add(chosen % cols)
            emit_partial()
            continue

        # Start another connected component if it adds an edge without conflicts.
        disconnected: list[tuple[float, int, int]] = []
        for u, v in zip(edge_u, edge_v, strict=True):
            ru, cu = divmod(u, cols)
            rv, cv = divmod(v, cols)
            if (
                ru not in used_rows
                and rv not in used_rows
                and cu not in used_cols
                and cv not in used_cols
            ):
                disconnected.append((float(flat_score[u] + flat_score[v]), u, v))
        if not disconnected:
            break
        disconnected.sort(reverse=True)
        shortlist = disconnected[: min(8, len(disconnected))]
        _, u, v = shortlist[int(rng.integers(len(shortlist)))]
        selected.update((u, v))
        used_rows.update((u // cols, v // cols))
        used_cols.update((u % cols, v % cols))
        emit_partial()

    mapping = torch.full((rows,), -1, dtype=torch.long)
    for node in selected:
        mapping[node // cols] = node % cols
    missing_rows = torch.nonzero(mapping < 0, as_tuple=True)[0]
    remaining_cols = torch.tensor([column for column in range(cols) if column not in used_cols])
    if missing_rows.numel():
        sub_scores = scores.detach().cpu()[missing_rows][:, remaining_cols]
        sub_mapping = hungarian_mapping(sub_scores)
        mapping[missing_rows] = remaining_cols[sub_mapping]
    return mapping


def refine_mapping(
    association: AssociationGraph,
    mapping: torch.Tensor,
    max_passes: int = 20,
    on_candidate: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Steepest-ascent 2-exchange/reassignment search on the exact hard objective."""

    current = mapping.detach().cpu().long().clone()
    rows, cols = association.shape
    if current.numel() != rows or len(set(current.tolist())) != rows:
        raise ValueError("mapping must be an injective assignment")
    objective = HardObjective(association)
    best_score = objective(current)
    if on_candidate is not None:
        on_candidate(current)

    for _ in range(max_passes):
        candidate_best = best_score
        candidate_mapping: torch.Tensor | None = None

        for i in range(rows):
            for j in range(i + 1, rows):
                proposal = current.clone()
                proposal[i], proposal[j] = current[j], current[i]
                score = objective(proposal)
                if score > candidate_best:
                    candidate_best, candidate_mapping = score, proposal
                    if on_candidate is not None:
                        on_candidate(proposal)

        used = set(current.tolist())
        unused = [column for column in range(cols) if column not in used]
        for i in range(rows):
            for column in unused:
                proposal = current.clone()
                proposal[i] = column
                score = objective(proposal)
                if score > candidate_best:
                    candidate_best, candidate_mapping = score, proposal
                    if on_candidate is not None:
                        on_candidate(proposal)

        if candidate_mapping is None:
            break
        current, best_score = candidate_mapping, candidate_best
    return current


def anneal_mapping(
    association: AssociationGraph,
    mapping: torch.Tensor,
    steps: int,
    rng: np.random.Generator,
    start_temperature: float = 1.25,
    end_temperature: float = 0.03,
    on_candidate: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Escape 2-opt basins with reversible swaps/reassignments."""

    if steps <= 0:
        return mapping
    objective = HardObjective(association)
    rows, cols = association.shape
    current = mapping.detach().cpu().long().clone()
    current_score = objective(current)
    best, best_score = current.clone(), current_score
    if on_candidate is not None:
        on_candidate(best)
    for step in range(steps):
        proposal = current.clone()
        if cols > rows and rng.random() < 0.25:
            used = set(current.tolist())
            unused = [column for column in range(cols) if column not in used]
            row = int(rng.integers(rows))
            proposal[row] = int(rng.choice(unused))
        else:
            i, j = rng.choice(rows, size=2, replace=False)
            proposal[i], proposal[j] = current[j], current[i]
        proposal_score = objective(proposal)
        # Edge count is primary; incident nodes only break edge-count ties.
        delta = (proposal_score[0] - current_score[0]) + (
            proposal_score[1] - current_score[1]
        ) / (rows + 1)
        fraction = step / max(steps - 1, 1)
        temperature = start_temperature * (end_temperature / start_temperature) ** fraction
        if delta >= 0 or rng.random() < np.exp(delta / temperature):
            current, current_score = proposal, proposal_score
            if current_score > best_score:
                best, best_score = current.clone(), current_score
                if on_candidate is not None:
                    on_candidate(best)
    return best


def large_neighborhood_refine(
    association: AssociationGraph,
    mapping: torch.Tensor,
    scores: torch.Tensor,
    steps: int,
    rng: np.random.Generator,
    on_candidate: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Destroy and optimally rematch small neighborhoods against the fixed boundary."""

    if steps <= 0:
        return mapping
    objective = HardObjective(association)
    rows, cols = association.shape
    current = mapping.detach().cpu().long().clone()
    current_score = objective(current)
    best, best_score = current.clone(), current_score
    if on_candidate is not None:
        on_candidate(best)
    node_adjacency: list[list[tuple[int, int]]] = [[] for _ in range(rows)]
    for u, v, label in objective.left_edges:
        node_adjacency[u].append((v, label))
        node_adjacency[v].append((u, label))
    score_matrix = scores.detach().cpu().numpy()

    for step in range(steps):
        destroy_size = int(rng.integers(3, min(12, rows) + 1))
        if rng.random() < 0.65:
            center = int(rng.integers(rows))
            destroyed = {center}
            frontier = [center]
            while frontier and len(destroyed) < destroy_size:
                node = frontier.pop(0)
                neighbors = [neighbor for neighbor, _ in node_adjacency[node]]
                rng.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor not in destroyed:
                        destroyed.add(neighbor)
                        frontier.append(neighbor)
                        if len(destroyed) == destroy_size:
                            break
            if len(destroyed) < destroy_size:
                choices = [row for row in range(rows) if row not in destroyed]
                extra = rng.choice(
                    choices,
                    destroy_size - len(destroyed),
                    replace=False,
                )
                destroyed.update(extra.tolist())
        else:
            destroyed = set(rng.choice(rows, destroy_size, replace=False).tolist())
        destroyed_rows = sorted(destroyed)
        fixed_rows = [row for row in range(rows) if row not in destroyed]
        fixed_columns = {int(current[row]) for row in fixed_rows}
        available_columns = [column for column in range(cols) if column not in fixed_columns]

        weights = np.full((len(destroyed_rows), len(available_columns)), -1e3, dtype=float)
        for ii, row in enumerate(destroyed_rows):
            for jj, column in enumerate(available_columns):
                if not objective.mask[row, column]:
                    continue
                boundary_gain = 0.0
                for neighbor, label in node_adjacency[row]:
                    if neighbor in destroyed:
                        continue
                    target_neighbor = int(current[neighbor])
                    if objective.right_edges.get(
                        (min(column, target_neighbor), max(column, target_neighbor))
                    ) == label:
                        boundary_gain += 1.0
                weights[ii, jj] = boundary_gain + 0.05 * score_matrix[row, column]
        assigned_rows, assigned_columns = linear_sum_assignment(-weights)
        proposal = current.clone()
        for ii, jj in zip(assigned_rows, assigned_columns, strict=True):
            proposal[destroyed_rows[ii]] = available_columns[jj]
        proposal_score = objective(proposal)
        delta = (proposal_score[0] - current_score[0]) + (
            proposal_score[1] - current_score[1]
        ) / (rows + 1)
        temperature = 0.5 * (0.02 / 0.5) ** (step / max(steps - 1, 1))
        if delta >= 0 or rng.random() < np.exp(delta / temperature):
            current, current_score = proposal, proposal_score
            if current_score > best_score:
                best, best_score = current.clone(), current_score
                if on_candidate is not None:
                    on_candidate(best)
    return best


def perturb_and_refine(
    association: AssociationGraph,
    scores: torch.Tensor,
    restarts: int = 8,
    max_passes: int = 20,
    anneal_steps: int = 1000,
    lns_steps: int = 100,
    seed: int = 0,
) -> torch.Tensor:
    """Deterministic seeded perturb-and-improve around a soft assignment."""

    rng = np.random.default_rng(seed)
    objective = HardObjective(association)
    best = refine_mapping(association, hungarian_mapping(scores), max_passes=max_passes)
    best_stats = objective(best)
    for restart in range(1, restarts):
        mode = restart % 3
        if mode == 0:
            noise_scale = 0.25 + 1.25 * (restart / max(restarts - 1, 1))
            noise = torch.from_numpy(rng.gumbel(size=scores.shape)).to(scores) * noise_scale
            proposal = hungarian_mapping(scores.clamp_min(1e-12).log() + noise)
        elif mode == 1:
            proposal = constructive_mapping(association, scores, rng)
        else:
            proposal = best.clone()
            swaps = 2 + restart % max(2, min(8, association.left.num_nodes // 3))
            for _ in range(swaps):
                i, j = rng.choice(association.left.num_nodes, size=2, replace=False)
                old_i = proposal[i].clone()
                proposal[i] = proposal[j]
                proposal[j] = old_i
        proposal = anneal_mapping(association, proposal, anneal_steps, rng)
        proposal = refine_mapping(association, proposal, max_passes=max_passes)
        stats = objective(proposal)
        if stats > best_stats:
            best, best_stats = proposal, stats

    # Keep a second, independent constructive stream.  Interleaving
    # constructive proposals with long annealing chains makes their random
    # states depend on the annealing budget; a larger budget can then explore
    # different seeds rather than a strict superset.  This portfolio restores
    # the anytime property: increasing ``restarts`` retains every earlier
    # constructive seed and can never discard its best hard MCES solution.
    constructive_rng = np.random.default_rng(seed)
    for _ in range(restarts):
        proposal = constructive_mapping(association, scores, constructive_rng)
        proposal = refine_mapping(association, proposal, max_passes=max_passes)
        stats = objective(proposal)
        if stats > best_stats:
            best, best_stats = proposal, stats

    best = large_neighborhood_refine(association, best, scores, lns_steps, rng)
    return refine_mapping(association, best, max_passes=max_passes)
