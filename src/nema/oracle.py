"""Independent RDKit RASCAL oracle used to audit recovered molecular graphs."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdRascalMCES

from nema.graph import LabeledGraph


def to_rdkit_molecule(graph: LabeledGraph) -> Chem.Mol:
    """Convert the project's integer-labelled graph representation to RDKit."""

    editable = Chem.RWMol()
    for label in graph.node_labels.tolist():
        editable.AddAtom(Chem.Atom(int(label)))
    for (source, target), label in zip(
        graph.edge_index.t().tolist(),
        graph.edge_labels.tolist(),
        strict=True,
    ):
        editable.AddBond(int(source), int(target), Chem.BondType.values[int(label)])
    molecule = editable.GetMol()
    # Unsanitized recovery is intentional for the molecules that failed the
    # release's kekulization path. RASCAL still needs cached valence/ring data.
    molecule.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(molecule)
    return molecule


def rascal_mces_truth(
    left: LabeledGraph,
    right: LabeledGraph,
    similarity_threshold: float = 0.0,
    timeout_seconds: int = 20000,
) -> dict[str, float | int | bool]:
    """Recompute MCES counts and similarity with the release's RASCAL options.

    ``similarity_threshold`` is left at the release-compatible value of zero
    by default.  Retrieval reconstruction can set it to the paper's 0.5
    relevance cutoff: RASCAL then proves whether a pair is relevant without
    spending exponential time resolving the exact ordering of irrelevant
    candidates.
    """

    options = rdRascalMCES.RascalOptions()
    options.similarityThreshold = float(similarity_threshold)
    options.completeAromaticRings = False
    options.maxBondMatchPairs = 100000
    options.timeout = int(timeout_seconds)
    matches = rdRascalMCES.FindMCES(
        to_rdkit_molecule(left),
        to_rdkit_molecule(right),
        options,
    )
    if not matches:
        return {
            "common_edges": 0,
            "common_nodes": 0,
            "similarity": 0.0,
            "timed_out": False,
        }
    match = matches[0]
    return {
        "common_edges": len(match.bondMatches()),
        "common_nodes": len(match.atomMatches()),
        "similarity": float(match.similarity),
        "timed_out": bool(match.timedOut),
    }
