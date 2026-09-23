"""Acquisition and loading of the public NGA evaluation pairs."""

from __future__ import annotations

import hashlib
import pickle
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

import torch

from nema.graph import GraphPair, LabeledGraph

OFFICIAL_DATA_URL = "https://raw.githubusercontent.com/LOGO-CUHKSZ/NGA/e4a8f1f9ec9e31f79f3fbd648717dfbb9fe113fc/data.zip"
OFFICIAL_DATA_SHA256 = "2eba0cca01584ef6cda6cb685a1f2274d98233d70ed3813979aef36827d2634a"


def download_official_data(root: str | Path, force: bool = False) -> Path:
    """Download the exact public archive at the pinned official NGA revision."""

    root = Path(root)
    marker = root / "MCES" / "AIDS-test" / "raw" / "graphs_1.pkl"
    if marker.exists() and not force:
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "nga_official_data.zip"
    urllib.request.urlretrieve(OFFICIAL_DATA_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != OFFICIAL_DATA_SHA256:
        raise RuntimeError(f"official data checksum mismatch: {digest}")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(root.parent)
    extracted = root.parent / "data"
    if root != extracted:
        if root.exists():
            raise FileExistsError(f"refusing to replace existing directory {root}")
        extracted.rename(root)
    return root


def _numeric_key(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def recover_empty_molecular_data(data: object) -> tuple[object, bool]:
    """Recover a graph emptied by RDKit kekulization from its stored SMILES.

    The released MOLHIV pickle occasionally contains a non-empty ``smiles``
    field but empty graph tensors.  Parsing without sanitization preserves the
    atom order, atomic-number labels, bond topology, and integer RDKit bond
    labels used by the release.  The original object is returned unchanged
    unless this exact failure mode is present.
    """

    if int(data.num_nodes) > 0:
        return data, False
    smiles = getattr(data, "smiles", None)
    if not isinstance(smiles, str) or not smiles:
        return data, False

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return data, False
    node_labels = torch.tensor(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        dtype=torch.long,
    )
    edge_pairs: list[tuple[int, int]] = []
    edge_labels: list[int] = []
    for bond in molecule.GetBonds():
        source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        label = int(bond.GetBondType())
        edge_pairs.extend(((source, target), (target, source)))
        edge_labels.extend((label, label))
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_labels, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(0, dtype=torch.long)
    recovered = data.clone()
    recovered.x = node_labels
    recovered.edge_index = edge_index
    recovered.edge_attr = edge_attr
    return recovered, True


def pair_paths(
    root: str | Path,
    dataset: str,
    split: str = "test",
    retrieval: bool = False,
    limit: int | None = None,
) -> list[Path]:
    root = Path(root)
    if retrieval:
        raw = root / "retrieval" / dataset / "raw" / split
    else:
        raw = root / "MCES" / f"{dataset}-{split}" / "raw"
    paths = sorted(raw.glob("graphs_*.pkl"), key=_numeric_key)
    # Predeclared native-test sanitation: these released pickles have empty
    # molecular tensors. Keep recovery available to other protocols, but never
    # silently reintroduce these seven inputs into the 291-pair native benchmark.
    if not retrieval and dataset == "MOLHIV" and split == "test":
        excluded = {23, 46, 48, 54, 61, 64, 76}
        paths = [p for p in paths if _numeric_key(p) not in excluded]
    return paths[:limit] if limit is not None else paths


def load_pairs(path: str | Path) -> list[GraphPair]:
    """Load an official PyG pickle.

    Pickle is intentionally used only for the pinned, checksummed archive above.
    Do not call this function on untrusted files.
    """

    path = Path(path)
    with path.open("rb") as stream:
        left_list, right_list = pickle.load(stream)
    if len(left_list) != len(right_list):
        raise ValueError(f"unpaired graph lists in {path}")
    stem_key = path.stem.rsplit("_", 1)[-1]
    pairs = []
    for index, (left_data, right_data) in enumerate(zip(left_list, right_list, strict=True)):
        left_data, left_recovered = recover_empty_molecular_data(left_data)
        right_data, right_recovered = recover_empty_molecular_data(right_data)
        truth = left_data.y.reshape(-1).detach().cpu().tolist()
        key = stem_key if len(left_list) == 1 else f"{stem_key}:{index}"
        recovered_sides = [
            side
            for side, recovered in (("left", left_recovered), ("right", right_recovered))
            if recovered
        ]
        pairs.append(
            GraphPair(
                left=LabeledGraph.from_pyg(left_data),
                right=LabeledGraph.from_pyg(right_data),
                true_edges=int(round(truth[0])),
                true_nodes=int(round(truth[1])),
                true_similarity=float(truth[2]),
                key=key,
                metadata={
                    "input_recovery": "unsanitized_smiles",
                    "recovered_sides": recovered_sides,
                }
                if recovered_sides
                else None,
            )
        )
    return pairs


def load_pair(path: str | Path) -> GraphPair:
    pairs = load_pairs(path)
    if len(pairs) != 1:
        raise ValueError(f"expected one pair in {path}, got {len(pairs)}")
    return pairs[0]


def iter_pairs(paths: list[Path]) -> Iterator[GraphPair]:
    for path in paths:
        yield from load_pairs(path)
