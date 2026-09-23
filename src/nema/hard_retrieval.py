"""Method-independent construction of a hard molecular retrieval benchmark.

The protocol deliberately separates candidate construction from relevance:
controlled edits and size/WL-nearest held-out graphs only create the candidate
pool, while RDKit RASCAL is the sole source of binary and graded truth.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem, rdBase
from scipy.optimize import linear_sum_assignment

from nema.graph import LabeledGraph
from nema.oracle import to_rdkit_molecule
from nema.reconstructed import BankGraph, graph_fingerprint
from nema.retrieval_baselines import (
    cosine_counter,
    size_upper_bound_similarity,
    wl_bag,
)


CONTROLLED_KINDS = (
    "edge_deletion",
    "atom_substitution",
    "bond_relabel",
    "edge_rewire",
    "mixed",
)
FORMAL_SEEDS = (20260823, 20260824, 20260825)

# Conservative same-group substitutions. Every proposal is still required to
# pass a full RDKit sanitization, so this table is a proposal mechanism rather
# than an assertion of chemical validity.
ATOM_SUBSTITUTIONS: dict[int, tuple[int, ...]] = {
    6: (14,),
    14: (6,),
    7: (15,),
    15: (7,),
    8: (16,),
    16: (8,),
    9: (17, 35, 53),
    17: (9, 35, 53),
    35: (9, 17, 53),
    53: (9, 17, 35),
}


@dataclass
class CandidateGraph:
    """A materialized candidate plus its complete construction provenance."""

    candidate_id: str
    graph: BankGraph
    candidate_kind: str
    parent_fingerprint: str
    audit: dict[str, object]

    def manifest_record(self) -> dict[str, object]:
        record = self.graph.manifest_record()
        record.update(
            {
                "candidate_id": self.candidate_id,
                "candidate_kind": self.candidate_kind,
                "parent_fingerprint": self.parent_fingerprint,
                "audit": self.audit,
            }
        )
        return record


@dataclass
class HardPoolEntry:
    """One candidate in one query pool, including method-independent hardness."""

    candidate: CandidateGraph
    size_similarity: float
    wl_similarity: float
    hard_score: float

    def manifest_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "candidate_kind": self.candidate.candidate_kind,
            "size_similarity": self.size_similarity,
            "wl_similarity": self.wl_similarity,
            "hard_score": self.hard_score,
        }


def _strict_molecule(graph: LabeledGraph) -> Chem.Mol | None:
    molecule = to_rdkit_molecule(graph)
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(molecule)
    except (ValueError, RuntimeError):
        return None
    return molecule


def is_strictly_valid_molecule(data: object) -> bool:
    """Return whether the graph is a simple graph and fully sanitizes in RDKit."""

    try:
        graph = LabeledGraph.from_pyg(data)
    except (TypeError, ValueError, RuntimeError):
        return False
    if graph.num_nodes == 0 or graph.num_edges == 0:
        return False
    edges = [tuple(edge) for edge in graph.edge_index.t().tolist()]
    if len(edges) != len(set(edges)) or any(source == target for source, target in edges):
        return False
    return _strict_molecule(graph) is not None


def mces_similarity_upper_bound(left: LabeledGraph, right: LabeledGraph) -> float:
    """A cheap, auditable upper bound for classifying timed-out RASCAL pairs.

    Every preserved edge must agree on its bond label and both endpoint atom
    labels. A second relaxation matches equal-label atoms with a weight equal
    to the overlap of their incident ``(bond label, neighbor atom label)``
    multisets. Each common edge contributes to two such incident overlaps.
    Both relaxations ignore global consistency and therefore remain upper
    bounds; taking their minimum is also a valid upper bound.
    """

    left_size = left.num_nodes + left.num_edges
    right_size = right.num_nodes + right.num_edges
    if not left_size or not right_size:
        return 0.0
    left_nodes = Counter(int(label) for label in left.node_labels.tolist())
    right_nodes = Counter(int(label) for label in right.node_labels.tolist())
    node_cap = sum((left_nodes & right_nodes).values())

    def edge_signatures(graph: LabeledGraph) -> Counter[tuple[int, int, int]]:
        signatures: Counter[tuple[int, int, int]] = Counter()
        for (source, target), edge_label in zip(
            graph.edge_index.t().tolist(), graph.edge_labels.tolist(), strict=True
        ):
            endpoints = sorted(
                (int(graph.node_labels[source]), int(graph.node_labels[target]))
            )
            signatures[(endpoints[0], endpoints[1], int(edge_label))] += 1
        return signatures

    typed_edge_cap = sum(
        (edge_signatures(left) & edge_signatures(right)).values()
    )
    typed_objective_cap = node_cap + typed_edge_cap

    def incident_signatures(graph: LabeledGraph) -> list[Counter[tuple[int, int]]]:
        signatures = [Counter() for _ in range(graph.num_nodes)]
        for (source, target), edge_label in zip(
            graph.edge_index.t().tolist(), graph.edge_labels.tolist(), strict=True
        ):
            signatures[source][(int(edge_label), int(graph.node_labels[target]))] += 1
            signatures[target][(int(edge_label), int(graph.node_labels[source]))] += 1
        return signatures

    left_incident = incident_signatures(left)
    right_incident = incident_signatures(right)
    incident_cap = 0.0
    common_labels = set(left_nodes) & set(right_nodes)
    for label in common_labels:
        left_indices = [
            index for index, value in enumerate(left.node_labels.tolist()) if int(value) == label
        ]
        right_indices = [
            index for index, value in enumerate(right.node_labels.tolist()) if int(value) == label
        ]
        weights = np.asarray(
            [
                [
                    1.0
                    + 0.5
                    * sum(
                        (left_incident[left_index] & right_incident[right_index]).values()
                    )
                    for right_index in right_indices
                ]
                for left_index in left_indices
            ],
            dtype=float,
        )
        rows, columns = linear_sum_assignment(weights, maximize=True)
        incident_cap += float(weights[rows, columns].sum())
    objective_cap = min(
        float(min(left_size, right_size)),
        float(typed_objective_cap),
        incident_cap,
    )
    return float(objective_cap**2 / (left_size * right_size))


def rdkit_to_pyg(molecule: Chem.Mol, template: object) -> object:
    """Convert a sanitized RDKit molecule to the release's bidirected PyG format."""

    candidate = template.clone()
    node_labels = torch.tensor(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=torch.long
    )
    directed_edges: list[tuple[int, int]] = []
    directed_labels: list[int] = []
    for bond in molecule.GetBonds():
        source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        label = int(bond.GetBondType())
        directed_edges.extend(((source, target), (target, source)))
        directed_labels.extend((label, label))
    order = sorted(range(len(directed_edges)), key=lambda index: directed_edges[index])
    candidate.x = node_labels
    candidate.edge_index = torch.tensor(
        [directed_edges[index] for index in order], dtype=torch.long
    ).t().contiguous()
    candidate.edge_attr = torch.tensor(
        [directed_labels[index] for index in order], dtype=torch.long
    )
    for attribute in ("y", "smiles"):
        if hasattr(candidate, attribute):
            delattr(candidate, attribute)
    return candidate


def _sanitize_copy(editable: Chem.RWMol) -> Chem.Mol | None:
    molecule = editable.GetMol()
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(molecule)
    except (ValueError, RuntimeError):
        return None
    return molecule


def _delete_edges(
    molecule: Chem.Mol, count: int, generator: random.Random
) -> tuple[Chem.Mol, list[dict[str, object]]] | None:
    eligible = [
        bond
        for bond in molecule.GetBonds()
        if not bond.GetIsAromatic() and bond.GetBondType() == Chem.BondType.SINGLE
    ]
    if len(eligible) < count:
        eligible = [bond for bond in molecule.GetBonds() if not bond.GetIsAromatic()]
    if len(eligible) < count:
        return None
    selected = generator.sample(eligible, count)
    operations = [
        {
            "operation": "delete_edge",
            "source": bond.GetBeginAtomIdx(),
            "target": bond.GetEndAtomIdx(),
            "bond_label_before": int(bond.GetBondType()),
        }
        for bond in selected
    ]
    editable = Chem.RWMol(molecule)
    for operation in operations:
        editable.RemoveBond(int(operation["source"]), int(operation["target"]))
    sanitized = _sanitize_copy(editable)
    return (sanitized, operations) if sanitized is not None else None


def _substitute_atoms(
    molecule: Chem.Mol, count: int, generator: random.Random
) -> tuple[Chem.Mol, list[dict[str, object]]] | None:
    eligible = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if not atom.GetIsAromatic() and atom.GetAtomicNum() in ATOM_SUBSTITUTIONS
    ]
    if len(eligible) < count:
        return None
    selected = generator.sample(eligible, count)
    editable = Chem.RWMol(molecule)
    operations: list[dict[str, object]] = []
    for atom_index in selected:
        atom = editable.GetAtomWithIdx(atom_index)
        before = atom.GetAtomicNum()
        after = generator.choice(ATOM_SUBSTITUTIONS[before])
        atom.SetAtomicNum(after)
        operations.append(
            {
                "operation": "substitute_atom_label",
                "atom": atom_index,
                "atomic_number_before": before,
                "atomic_number_after": after,
            }
        )
    sanitized = _sanitize_copy(editable)
    return (sanitized, operations) if sanitized is not None else None


def _relabel_bonds(
    molecule: Chem.Mol, count: int, generator: random.Random
) -> tuple[Chem.Mol, list[dict[str, object]]] | None:
    eligible = [
        bond.GetIdx()
        for bond in molecule.GetBonds()
        if not bond.GetIsAromatic()
        and bond.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE)
    ]
    if len(eligible) < count:
        return None
    selected = generator.sample(eligible, count)
    editable = Chem.RWMol(molecule)
    operations: list[dict[str, object]] = []
    for bond_index in selected:
        bond = editable.GetBondWithIdx(bond_index)
        before = bond.GetBondType()
        after = Chem.BondType.DOUBLE if before == Chem.BondType.SINGLE else Chem.BondType.SINGLE
        bond.SetBondType(after)
        operations.append(
            {
                "operation": "relabel_edge",
                "source": bond.GetBeginAtomIdx(),
                "target": bond.GetEndAtomIdx(),
                "bond_label_before": int(before),
                "bond_label_after": int(after),
            }
        )
    sanitized = _sanitize_copy(editable)
    return (sanitized, operations) if sanitized is not None else None


def _rewire_edges(
    molecule: Chem.Mol, count: int, generator: random.Random
) -> tuple[Chem.Mol, list[dict[str, object]]] | None:
    editable = Chem.RWMol(molecule)
    operations: list[dict[str, object]] = []
    for _ in range(count):
        bonds = [
            bond
            for bond in editable.GetBonds()
            if not bond.GetIsAromatic() and bond.GetBondType() == Chem.BondType.SINGLE
        ]
        proposals = [(first, second) for i, first in enumerate(bonds) for second in bonds[i + 1 :]]
        generator.shuffle(proposals)
        applied = False
        for first, second in proposals[:256]:
            a, b = first.GetBeginAtomIdx(), first.GetEndAtomIdx()
            c, d = second.GetBeginAtomIdx(), second.GetEndAtomIdx()
            if len({a, b, c, d}) != 4:
                continue
            alternatives = ((a, c, b, d), (a, d, b, c))
            alternatives = list(alternatives)
            generator.shuffle(alternatives)
            for first_u, first_v, second_u, second_v in alternatives:
                if (
                    editable.GetBondBetweenAtoms(first_u, first_v) is not None
                    or editable.GetBondBetweenAtoms(second_u, second_v) is not None
                ):
                    continue
                editable.RemoveBond(a, b)
                editable.RemoveBond(c, d)
                editable.AddBond(first_u, first_v, Chem.BondType.SINGLE)
                editable.AddBond(second_u, second_v, Chem.BondType.SINGLE)
                operations.append(
                    {
                        "operation": "degree_preserving_edge_rewire",
                        "removed": [[a, b], [c, d]],
                        "added": [[first_u, first_v], [second_u, second_v]],
                        "bond_label": int(Chem.BondType.SINGLE),
                    }
                )
                applied = True
                break
            if applied:
                break
        if not applied:
            return None
    sanitized = _sanitize_copy(editable)
    return (sanitized, operations) if sanitized is not None else None


def mutate_molecular_graph(
    data: object,
    kind: str,
    edit_count: int,
    seed: int,
    max_attempts: int = 128,
) -> tuple[object, dict[str, object]]:
    """Apply an auditable deterministic edit and require strict chemical validity.

    ``edit_count`` is the number of primitive changes. A rewire counts as one
    degree-preserving two-edge switch. Mixed edits split the budget between
    edge deletions and atom-label substitutions.
    """

    if kind not in CONTROLLED_KINDS:
        raise ValueError(f"unsupported controlled edit kind: {kind}")
    if edit_count <= 0:
        raise ValueError("edit_count must be positive")
    source = LabeledGraph.from_pyg(data)
    molecule = _strict_molecule(source)
    if molecule is None:
        raise ValueError("source graph does not pass strict RDKit sanitization")

    source_fingerprint = graph_fingerprint(data)
    result: tuple[Chem.Mol, list[dict[str, object]]] | None = None
    used_attempt = -1
    for attempt in range(max_attempts):
        generator = random.Random(seed + attempt * 1_000_003)
        if kind == "edge_deletion":
            result = _delete_edges(molecule, edit_count, generator)
        elif kind == "atom_substitution":
            result = _substitute_atoms(molecule, edit_count, generator)
        elif kind == "bond_relabel":
            result = _relabel_bonds(molecule, edit_count, generator)
        elif kind == "edge_rewire":
            result = _rewire_edges(molecule, edit_count, generator)
        else:
            deletion_count = max(1, math.ceil(edit_count / 2))
            substitution_count = max(1, edit_count - deletion_count)
            deleted = _delete_edges(molecule, deletion_count, generator)
            result = None
            if deleted is not None:
                substituted = _substitute_atoms(deleted[0], substitution_count, generator)
                if substituted is not None:
                    result = (substituted[0], deleted[1] + substituted[1])
        if result is not None:
            # Multiple degree-preserving switches can occasionally undo one
            # another. The release tensor also omits RDKit charges and other
            # properties, so validate the exact serialized round trip rather
            # than only the richer in-memory molecule.
            proposed = rdkit_to_pyg(result[0], data)
            if (
                graph_fingerprint(proposed) == source_fingerprint
                or not is_strictly_valid_molecule(proposed)
            ):
                result = None
                continue
            used_attempt = attempt
            break
    if result is None:
        raise ValueError(
            f"could not produce a valid {kind} mutation with {edit_count} edits "
            f"after {max_attempts} attempts"
        )

    mutated = rdkit_to_pyg(result[0], data)
    if not is_strictly_valid_molecule(mutated):
        raise AssertionError("internal error: accepted mutation is not strictly valid")
    before_fingerprint = source_fingerprint
    after_fingerprint = graph_fingerprint(mutated)
    if before_fingerprint == after_fingerprint:
        raise AssertionError("controlled mutation did not change the graph fingerprint")
    audit: dict[str, object] = {
        "mutation_kind": kind,
        "requested_edit_count": edit_count,
        "primitive_operation_count": len(result[1]),
        "seed": seed,
        "accepted_attempt": used_attempt,
        "operations": result[1],
        "strict_rdkit_sanitized": True,
        "source_fingerprint": before_fingerprint,
        "result_fingerprint": after_fingerprint,
        "canonical_smiles": Chem.MolToSmiles(result[0], canonical=True),
    }
    return mutated, audit


def select_protocol_parents(
    bank: list[BankGraph],
    queries: int,
    seed: int,
    parent_pool_size: int = 650,
) -> tuple[list[BankGraph], list[BankGraph], dict[str, int]]:
    """Select disjoint, train-excluded and strictly valid parent graphs."""

    valid = [entry for entry in bank if is_strictly_valid_molecule(entry.data)]
    if len(valid) <= queries:
        raise ValueError(f"only {len(valid)} strictly valid held-out graphs for {queries} queries")
    ordered = sorted(
        valid,
        key=lambda entry: (
            LabeledGraph.from_pyg(entry.data).num_edges,
            LabeledGraph.from_pyg(entry.data).num_nodes,
            entry.fingerprint,
        ),
    )[: min(parent_pool_size, len(valid))]
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(ordered), generator=generator).tolist()
    selected = [ordered[index] for index in permutation]
    query_parents = selected[:queries]
    candidate_parents = selected[queries:]
    if {entry.fingerprint for entry in query_parents} & {
        entry.fingerprint for entry in candidate_parents
    }:
        raise AssertionError("query and candidate parent fingerprints overlap")
    return query_parents, candidate_parents, {
        "held_out_bank_count": len(bank),
        "strict_valid_bank_count": len(valid),
        "selection_pool_count": len(ordered),
        "query_parent_count": len(query_parents),
        "candidate_parent_count": len(candidate_parents),
    }


def _as_candidate(
    parent: BankGraph,
    data: object,
    candidate_id: str,
    kind: str,
    audit: dict[str, object],
) -> CandidateGraph:
    fingerprint = graph_fingerprint(data)
    graph = BankGraph(
        fingerprint=fingerprint,
        data=data,
        source_path=parent.source_path,
        source_side=kind,
        recovered_from_smiles=parent.recovered_from_smiles,
    )
    return CandidateGraph(candidate_id, graph, kind, parent.fingerprint, audit)


def build_candidate_library(
    parents: list[BankGraph],
    training_fingerprints: set[str],
    forbidden_fingerprints: set[str],
    seed: int,
    derived_per_parent: int = 2,
) -> list[CandidateGraph]:
    """Build a reusable held-out candidate library without model outputs."""

    library: list[CandidateGraph] = []
    seen = set(training_fingerprints) | set(forbidden_fingerprints)
    for parent_index, parent in enumerate(parents):
        if parent.fingerprint not in seen:
            base_audit = {
                "mutation_kind": "none",
                "strict_rdkit_sanitized": True,
                "source_fingerprint": parent.fingerprint,
                "result_fingerprint": parent.fingerprint,
            }
            library.append(
                _as_candidate(
                    parent,
                    parent.data.clone(),
                    f"parent-{parent_index}",
                    "held_out_parent",
                    base_audit,
                )
            )
            seen.add(parent.fingerprint)

        derived = 0
        # Label and structure variants enlarge the reusable library enough for
        # 1000-way evaluation even when a released dataset has <1000 parents.
        recipes = (
            ("atom_substitution", 1),
            ("edge_deletion", 2),
            ("edge_rewire", 1),
            ("mixed", 2),
            ("bond_relabel", 1),
        )
        for recipe_index, (kind, edit_count) in enumerate(recipes):
            if derived >= derived_per_parent:
                break
            mutation_seed = seed + (parent_index + 1) * 100_003 + recipe_index * 997
            try:
                data, audit = mutate_molecular_graph(
                    parent.data, kind, edit_count, mutation_seed
                )
            except ValueError:
                continue
            fingerprint = graph_fingerprint(data)
            if fingerprint in seen:
                continue
            library.append(
                _as_candidate(
                    parent,
                    data,
                    f"parent-{parent_index}-{kind}-{derived}",
                    f"held_out_parent_{kind}",
                    audit,
                )
            )
            seen.add(fingerprint)
            derived += 1
    return library


def build_controlled_candidates(
    query: BankGraph,
    query_index: int,
    count: int,
    seed: int,
    training_fingerprints: set[str],
) -> list[CandidateGraph]:
    """Create multi-type, nontrivial query variants with no fixed top-10 quota."""

    if count < len(CONTROLLED_KINDS):
        raise ValueError(f"controlled count must be at least {len(CONTROLLED_KINDS)}")
    candidates: list[CandidateGraph] = []
    seen = set(training_fingerprints) | {query.fingerprint}
    level = 0
    failures: list[str] = []
    while len(candidates) < count and level < count * 5:
        kind = CONTROLLED_KINDS[level % len(CONTROLLED_KINDS)]
        # No identity or one-edit positives: every controlled candidate has at
        # least two primitive edits, with difficulty increasing by rounds.
        edit_count = 2 + level // len(CONTROLLED_KINDS)
        mutation_seed = seed + (query_index + 1) * 1_000_003 + level * 10_007
        try:
            data, audit = mutate_molecular_graph(
                query.data, kind, edit_count, mutation_seed
            )
        except ValueError as error:
            failures.append(f"{kind}:{edit_count}:{error}")
            level += 1
            continue
        fingerprint = graph_fingerprint(data)
        if fingerprint in seen:
            level += 1
            continue
        candidate_id = f"query-{query_index}-{kind}-{edit_count}-{level}"
        candidates.append(
            _as_candidate(
                query,
                data,
                candidate_id,
                f"controlled_{kind}",
                audit,
            )
        )
        seen.add(fingerprint)
        level += 1
    if len(candidates) != count:
        tail = "; ".join(failures[-3:])
        raise ValueError(
            f"query {query_index}: generated only {len(candidates)}/{count} controlled "
            f"candidates; last failures: {tail}"
        )
    return candidates


def _hardness(
    query: LabeledGraph,
    candidate: LabeledGraph,
    query_wl: object | None = None,
    candidate_wl: object | None = None,
) -> tuple[float, float, float]:
    size = size_upper_bound_similarity(query, candidate)
    wl = cosine_counter(query_wl or wl_bag(query), candidate_wl or wl_bag(candidate))
    # WL is the stronger non-learned structural baseline; size breaks ties and
    # discourages trivially separable graph-size negatives.
    return size, wl, 0.35 * size + 0.65 * wl


def build_query_pool(
    query: BankGraph,
    controlled: list[CandidateGraph],
    library: list[CandidateGraph],
    candidates: int,
    seed: int,
    query_index: int,
    library_wl: dict[str, object] | None = None,
) -> list[HardPoolEntry]:
    """Select and shuffle a hard pool using only size and WL similarities."""

    hard_count = candidates - len(controlled)
    if hard_count <= 0:
        raise ValueError("candidate count must exceed controlled candidate count")
    query_graph = LabeledGraph.from_pyg(query.data)
    query_wl = wl_bag(query_graph)
    scored: list[HardPoolEntry] = []
    controlled_fingerprints = {entry.graph.fingerprint for entry in controlled}
    for candidate in library:
        if candidate.graph.fingerprint in controlled_fingerprints:
            continue
        size, wl, score = _hardness(
            query_graph,
            LabeledGraph.from_pyg(candidate.graph.data),
            query_wl,
            library_wl.get(candidate.graph.fingerprint) if library_wl else None,
        )
        scored.append(HardPoolEntry(candidate, size, wl, score))
    scored.sort(
        key=lambda entry: (
            -entry.hard_score,
            -entry.wl_similarity,
            -entry.size_similarity,
            entry.candidate.graph.fingerprint,
        )
    )
    if len(scored) < hard_count:
        raise ValueError(f"need {hard_count} hard candidates but library has {len(scored)}")
    selected = scored[:hard_count]
    for candidate in controlled:
        size, wl, score = _hardness(
            query_graph,
            LabeledGraph.from_pyg(candidate.graph.data),
            query_wl,
        )
        selected.append(HardPoolEntry(candidate, size, wl, score))
    generator = torch.Generator().manual_seed(seed + (query_index + 1) * 10_000_019)
    permutation = torch.randperm(len(selected), generator=generator).tolist()
    return [selected[index] for index in permutation]


def build_hard_protocol(
    bank: list[BankGraph],
    training_fingerprints: set[str],
    queries: int,
    candidates: int,
    controlled_count: int,
    seed: int,
    parent_pool_size: int = 650,
    derived_per_parent: int = 2,
) -> tuple[list[BankGraph], list[list[HardPoolEntry]], list[CandidateGraph], dict[str, int]]:
    """Materialize every deterministic selection needed by the oracle builder."""

    query_parents, candidate_parents, audit = select_protocol_parents(
        bank, queries, seed, parent_pool_size
    )
    query_fingerprints = {entry.fingerprint for entry in query_parents}
    library = build_candidate_library(
        candidate_parents,
        training_fingerprints,
        query_fingerprints,
        seed,
        derived_per_parent,
    )
    library_wl = {
        entry.graph.fingerprint: wl_bag(LabeledGraph.from_pyg(entry.graph.data))
        for entry in library
    }
    pools: list[list[HardPoolEntry]] = []
    for query_index, query in enumerate(query_parents):
        controlled = build_controlled_candidates(
            query,
            query_index,
            controlled_count,
            seed,
            training_fingerprints,
        )
        pools.append(
            build_query_pool(
                query,
                controlled,
                library,
                candidates,
                seed,
                query_index,
                library_wl,
            )
        )
    query_fingerprints = {entry.fingerprint for entry in query_parents}
    candidate_parent_fingerprints = {
        entry.parent_fingerprint for entry in library
    }
    candidate_fingerprints = {
        entry.candidate.graph.fingerprint for pool in pools for entry in pool
    }
    leakage_audit = {
        "query_parent_vs_mces_train_overlap": len(
            query_fingerprints & training_fingerprints
        ),
        "candidate_parent_vs_mces_train_overlap": len(
            candidate_parent_fingerprints & training_fingerprints
        ),
        "materialized_candidate_vs_mces_train_overlap": len(
            candidate_fingerprints & training_fingerprints
        ),
        "query_vs_candidate_parent_overlap": len(
            query_fingerprints & candidate_parent_fingerprints
        ),
        "invalid_materialized_candidate_count": sum(
            not is_strictly_valid_molecule(entry.candidate.graph.data)
            for pool in pools
            for entry in pool
        ),
        "duplicate_candidates_within_query_max": max(
            candidates
            - len({entry.candidate.graph.fingerprint for entry in pool})
            for pool in pools
        ),
    }
    if any(leakage_audit.values()):
        raise AssertionError(f"hard retrieval audit failed: {leakage_audit}")
    audit.update(
        {
            "candidate_library_count": len(library),
            "controlled_candidates_per_query": controlled_count,
            "hard_candidates_per_query": candidates - controlled_count,
            **leakage_audit,
        }
    )
    return query_parents, pools, library, audit
