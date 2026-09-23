"""Reaction adapters and reference metrics, separate from ENPDA inference.

Atom-map numbers are evaluator-only metadata. ENPDA sees atomic numbers and
bond types, with independently shuffled atom orders on the two sides.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import networkx as nx
from rdkit import Chem
import torch

from nema.graph import GraphPair, LabeledGraph


def molecule_graph(molecule: Chem.Mol) -> LabeledGraph:
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds()]
    return LabeledGraph(
        torch.tensor([a.GetAtomicNum() for a in molecule.GetAtoms()], dtype=torch.long),
        torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
        torch.tensor(
            [int(round(2 * b.GetBondTypeAsDouble())) for b in molecule.GetBonds()],
            dtype=torch.long,
        ),
    )


@dataclass
class ReactionExample:
    key: str
    reactants: Chem.Mol
    products: Chem.Mol
    gold_product_to_reactant: list[int]
    pair: GraphPair


def prepare_reaction(mapped_smiles: str, key: str, seed: int) -> ReactionExample:
    """Build heavy-atom graphs without agents, coordinates, or map-ID features."""
    parts = mapped_smiles.split(">")
    if len(parts) != 3:
        raise ValueError("expected reactants>agents>products")
    rng = random.Random(seed)
    molecules, maps = [], []
    for smiles in (parts[0], parts[2]):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("invalid SMILES")
        # Remove explicit H even if the source assigns H a map number.
        remove = Chem.RemoveHsParameters()
        remove.removeMapped = True
        remove.removeDegreeZero = True
        remove.removeHydrides = True
        remove.removeIsotopes = True
        remove.removeOnlyHNeighbors = True
        molecule = Chem.RemoveHs(molecule, remove)
        if not molecule.GetNumAtoms():
            raise ValueError("empty heavy-atom graph")
        if any(a.GetAtomicNum() <= 1 for a in molecule.GetAtoms()):
            raise ValueError("wildcard or unremoved explicit hydrogen")
        order = list(range(molecule.GetNumAtoms()))
        rng.shuffle(order)
        molecule = Chem.RenumberAtoms(molecule, order)
        ids = [a.GetAtomMapNum() for a in molecule.GetAtoms()]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ValueError("missing or duplicate reference atom-map IDs")
        maps.append(ids)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        molecules.append(molecule)
    reactants, products = molecules
    lookup = {value: index for index, value in enumerate(maps[0])}
    gold = [lookup.get(value, -1) for value in maps[1]]
    if not any(value >= 0 for value in gold):
        raise ValueError("no reference heavy-atom correspondence")
    for product, reactant in enumerate(gold):
        if reactant >= 0 and (
            products.GetAtomWithIdx(product).GetAtomicNum()
            != reactants.GetAtomWithIdx(reactant).GetAtomicNum()
        ):
            raise ValueError("reference changes chemical element")
    return ReactionExample(
        key, reactants, products, gold,
        GraphPair(molecule_graph(reactants), molecule_graph(products), key=key),
    )


def _atom_state(atom: Chem.Atom) -> tuple:
    return (
        atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
        atom.GetIsAromatic(), atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
    )


def condensed_graph(example: ReactionExample, product_to_reactant: list[int]) -> nx.Graph:
    """Encode both molecular sides jointly; ignore stereochemistry explicitly.

    Isomorphism allows *consistent* automorphisms on either side. It cannot
    independently forgive each atom in a symmetry class, which would overcount.
    """
    graph = nx.Graph()
    for atom in example.reactants.GetAtoms():
        graph.add_node(atom.GetIdx(), state=(_atom_state(atom), None))
    next_node = example.reactants.GetNumAtoms()
    product_nodes = []
    for atom, reactant in zip(example.products.GetAtoms(), product_to_reactant, strict=True):
        node = reactant
        if node < 0:
            node = next_node
            next_node += 1
            graph.add_node(node, state=(None, None))
        before, _ = graph.nodes[node]["state"]
        graph.nodes[node]["state"] = (before, _atom_state(atom))
        product_nodes.append(node)
    for side, molecule in enumerate((example.reactants, example.products)):
        for bond in molecule.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if side:
                u, v = product_nodes[u], product_nodes[v]
            state = list(graph.edges[u, v]["state"]) if graph.has_edge(u, v) else [0, 0]
            state[side] = int(round(2 * bond.GetBondTypeAsDouble()))
            graph.add_edge(u, v, state=tuple(state))
    return graph


def score_mapping(example: ReactionExample, reactant_to_product: list[int]) -> dict:
    """Strict reference agreement and symmetry-aware whole-reaction correctness."""
    if len(reactant_to_product) != example.reactants.GetNumAtoms():
        raise ValueError("wrong mapping length")
    predicted = [-1] * example.products.GetNumAtoms()
    element_compatible = True
    for reactant, product in enumerate(reactant_to_product):
        if product == -1:
            continue
        if product < 0 or product >= len(predicted) or predicted[product] >= 0:
            raise ValueError("out-of-range or non-injective mapping")
        predicted[product] = reactant
        element_compatible &= (
            example.reactants.GetAtomWithIdx(reactant).GetAtomicNum()
            == example.products.GetAtomWithIdx(product).GetAtomicNum()
        )
    gold = example.gold_product_to_reactant
    known = sum(value >= 0 for value in gold)
    correct = sum(g >= 0 and g == p for g, p in zip(gold, predicted, strict=True))
    strict = predicted == gold
    if strict:
        symmetric = True
    elif element_compatible:
        reference_graph = condensed_graph(example, gold)
        predicted_graph = condensed_graph(example, predicted)
        symmetric = nx.is_isomorphic(
            reference_graph, predicted_graph,
            node_match=lambda a, b: a["state"] == b["state"],
            edge_match=lambda a, b: a["state"] == b["state"],
        )
    else:
        symmetric = False
    return {
        "reference_product_atoms": known,
        "correct_product_atoms_strict": correct,
        "atom_accuracy_strict": correct / known,
        "reaction_exact_strict": strict,
        "reaction_exact_symmetry_no_stereo": symmetric,
        "product_coverage": sum(p >= 0 for p in predicted) / len(predicted),
        "element_compatible": element_compatible,
        "product_to_reactant": predicted,
    }
