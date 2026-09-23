import json

import torch
from torch_geometric.data import Data

import nema.benchmark as benchmark
from nema.data import recover_empty_molecular_data
from nema.graph import GraphPair, LabeledGraph
from nema.hard_retrieval import (
    CONTROLLED_KINDS,
    build_candidate_library,
    build_controlled_candidates,
    build_query_pool,
    is_strictly_valid_molecule,
    mces_similarity_upper_bound,
    mutate_molecular_graph,
)
from nema.oracle import rascal_mces_truth
from nema.reconstructed import (
    BankGraph,
    controlled_candidate_specs,
    delete_undirected_edges,
    graph_fingerprint,
)
from nema.solvers import SolveResult, source_to_target_mapping


def _record(key: int) -> dict:
    return {
        "key": str(key),
        "accuracy": 1.0,
        "similarity_squared_error": 0.0,
        "runtime_seconds": 1.0,
        "common_edges": 2,
        "true_edges": 2,
        "similarity": float(key),
        "true_similarity": float(key),
    }


def test_empty_molecular_graph_is_recovered_without_sanitization():
    empty = Data(
        x=torch.empty(0, dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty(0, dtype=torch.long),
        smiles="CC",
    )

    recovered, changed = recover_empty_molecular_data(empty)

    assert changed is True
    assert recovered.x.tolist() == [6, 6]
    assert recovered.edge_index.tolist() == [[0, 1], [1, 0]]
    assert recovered.edge_attr.tolist() == [1, 1]


def test_recovered_graph_round_trips_through_rascal_oracle():
    graph = LabeledGraph(
        node_labels=torch.tensor([6, 6, 6]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_labels=torch.tensor([1, 1]),
    )

    truth = rascal_mces_truth(graph, graph)

    assert truth["common_edges"] == 2
    assert truth["common_nodes"] == 3
    assert truth["similarity"] == 1.0
    assert truth["timed_out"] is False


def test_swapped_mapping_is_serialized_in_original_source_direction():
    source = LabeledGraph(
        node_labels=torch.ones(4, dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_labels=torch.ones(3, dtype=torch.long),
    )
    target = LabeledGraph(
        node_labels=torch.ones(2, dtype=torch.long),
        edge_index=torch.tensor([[0], [1]]),
        edge_labels=torch.ones(1, dtype=torch.long),
    )
    pair = GraphPair(source, target)

    mapping = source_to_target_mapping(pair, torch.tensor([3, 1]), swapped=True)

    assert mapping.tolist() == [-1, 1, -1, 0]


def test_graph_fingerprint_normalizes_edge_direction_and_order():
    first = Data(
        x=torch.tensor([6, 7, 8]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.tensor([1, 1, 2, 2]),
    )
    second = Data(
        x=first.x.clone(),
        edge_index=torch.tensor([[2, 1, 0, 1], [1, 2, 1, 0]]),
        edge_attr=torch.tensor([2, 2, 1, 1]),
    )

    assert graph_fingerprint(first) == graph_fingerprint(second)


def test_controlled_edge_deletion_removes_both_directions_deterministically():
    source = Data(
        x=torch.tensor([6, 6, 6, 6]),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
        ),
        edge_attr=torch.ones(8, dtype=torch.long),
    )

    first = delete_undirected_edges(source, count=2, seed=9)
    second = delete_undirected_edges(source, count=2, seed=9)

    assert LabeledGraph.from_pyg(first).num_edges == 2
    assert graph_fingerprint(first) == graph_fingerprint(second)
    assert LabeledGraph.from_pyg(source).num_edges == 4


def test_controlled_candidate_pools_have_shuffled_positive_budget():
    pools = controlled_candidate_specs(
        query_edge_counts=[30, 31, 32],
        negative_pool=12,
        candidates=10,
        positives=2,
        seed=17,
    )

    assert [len(pool) for pool in pools] == [10, 10, 10]
    assert all(
        sum(spec["kind"] == "controlled_edge_deletion" for spec in pool) == 2
        for pool in pools
    )
    assert pools == controlled_candidate_specs([30, 31, 32], 12, 10, 2, 17)


def _pyg_from_smiles(smiles: str) -> Data:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    edges = []
    labels = []
    for bond in molecule.GetBonds():
        source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((source, target), (target, source)))
        labels.extend((int(bond.GetBondType()), int(bond.GetBondType())))
    return Data(
        x=torch.tensor([atom.GetAtomicNum() for atom in molecule.GetAtoms()]),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(labels, dtype=torch.long),
    )


def test_hard_mutations_are_deterministic_auditable_and_strictly_valid():
    source = _pyg_from_smiles("CCOC(=O)NCCc1ccccc1")

    successful_kinds = set()
    for kind in CONTROLLED_KINDS:
        first, first_audit = mutate_molecular_graph(source, kind, edit_count=2, seed=91)
        second, second_audit = mutate_molecular_graph(source, kind, edit_count=2, seed=91)
        assert is_strictly_valid_molecule(first)
        assert graph_fingerprint(first) == graph_fingerprint(second)
        assert first_audit == second_audit
        assert first_audit["strict_rdkit_sanitized"] is True
        assert first_audit["source_fingerprint"] != first_audit["result_fingerprint"]
        assert first_audit["primitive_operation_count"] >= 1
        successful_kinds.add(kind)

    assert successful_kinds == set(CONTROLLED_KINDS)


def test_hard_pool_uses_only_size_wl_and_has_no_fixed_ten_easy_variants():
    parents = []
    for index, smiles in enumerate(
        ("CCCCCC", "CCCCCO", "CCCCCN", "CCCOCC", "CCNCCC", "CCSCCC")
    ):
        data = _pyg_from_smiles(smiles)
        parents.append(
            BankGraph(graph_fingerprint(data), data, f"graph-{index}", "left", False)
        )
    query = parents[0]
    controlled = build_controlled_candidates(query, 0, 5, 123, set())
    library = build_candidate_library(parents[1:], set(), {query.fingerprint}, 123, 1)
    pool = build_query_pool(query, controlled, library, 10, 123, 0)

    assert len(pool) == 10
    assert len(controlled) == len(CONTROLLED_KINDS)
    assert all(entry.candidate.audit.get("requested_edit_count", 2) >= 2 for entry in pool
               if entry.candidate.candidate_kind.startswith("controlled_"))
    assert all(
        abs(entry.hard_score - (0.35 * entry.size_similarity + 0.65 * entry.wl_similarity))
        < 1e-12
        for entry in pool
    )


def test_hard_timeout_relaxation_is_a_valid_similarity_upper_bound():
    left = _pyg_from_smiles("CCOC(=O)NCC")
    right = _pyg_from_smiles("CCNC(=O)OCC")
    left_graph = LabeledGraph.from_pyg(left)
    right_graph = LabeledGraph.from_pyg(right)
    exact = rascal_mces_truth(left_graph, right_graph)["similarity"]

    upper = mces_similarity_upper_bound(left_graph, right_graph)

    assert exact <= upper + 1e-12
    assert 0.0 <= upper <= 1.0








def test_retrieval_summary_accepts_threshold_censored_zero_truth(tmp_path):
    result = tmp_path / "retrieval.jsonl"
    records = []
    for index in range(100):
        record = _record(index + 1)
        record["accuracy"] = None
        record["true_edges"] = 0
        record["true_similarity"] = 1.0 if index < 10 else 0.0
        record["similarity"] = record["true_similarity"]
        records.append(record)
    result.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = benchmark.summarize_results(result, retrieval=True)

    assert summary["accuracy_percent"] is None
    assert summary["optimal_solution_percent"] is None
    assert summary["P@10"] == 1.0


def test_official_best_of_three_matches_released_evaluator(monkeypatch):
    graph = LabeledGraph(
        torch.ones(2, dtype=torch.long),
        torch.tensor([[0], [1]]),
        torch.ones(1, dtype=torch.long),
    )
    monkeypatch.setattr(
        benchmark,
        "load_pair",
        lambda _: GraphPair(graph, graph, true_edges=4, true_nodes=2, true_similarity=1.0),
    )
    candidates = iter(
        [
            SolveResult("NGA-official", "1", 3, 2, 0.6, 1.0, [0, 1], similarity_squared_error=0.4),
            # Equal edge accuracy is ignored even when nodes/error improve.
            SolveResult("NGA-official", "1", 3, 3, 0.9, 1.0, [0, 1], similarity_squared_error=0.1),
            # Strictly better accuracy is selected, but evaluate.py retains the
            # lower Table-2 error observed before that improvement.
            SolveResult("NGA-official", "1", 4, 2, 0.5, 1.0, [0, 1], similarity_squared_error=0.5),
        ]
    )
    monkeypatch.setattr(
        benchmark,
        "solve_official_nga_path",
        lambda *args, **kwargs: next(candidates),
    )

    record = benchmark._solve_path(
        {
            "path": "unused.pkl",
            "method": "nga",
            "torch_threads": 1,
            "nga_runs": 3,
            "epochs": 200,
            "learning_rate": 1e-3,
            "samples": 10,
            "time_budget": 60,
            "seed": 7,
            "device": "cpu",
        }
    )

    assert record["common_edges"] == 4
    assert record["similarity"] == 0.5
    assert record["similarity_squared_error"] == 0.4
    assert len(record["metadata"]["run_summaries"]) == 3
