"""Explicit complete-aromatic-cycle projection for an inference-only pilot.

This is a constrained bond-witness projection, not the paper's original MCES
objective and not a claim that RDKit's similarly named option enforces it.
An aromatic bond is retained only as part of a complete RDKit symmetric-SSSR
aromatic cycle mapped to a complete aromatic cycle in the other molecule.
Fused systems may retain a constituent cycle; preserving every functional group
or an entire fused system is deliberately not claimed.
"""
from __future__ import annotations

from rdkit import Chem


def edge(u, v):
    return (min(u, v), max(u, v))


class AromaticProjection:
    def __init__(self, pair):
        self.pair = pair
        self.mols = []
        self.bonds = []
        self.aromatic = []
        self.rings = []
        for name in ('left', 'right'):
            graph = pair[name]
            mol = Chem.MolFromSmiles(graph['smiles'])
            assert mol is not None
            assert [a.GetAtomicNum() for a in mol.GetAtoms()] == graph['nodes']
            bonds = {edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx()): int(b.GetBondType())
                     for b in mol.GetBonds()}
            assert bonds == {edge(u, v): label for u, v, label in graph['edges']}
            aromatic = {edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                        for b in mol.GetBonds() if b.GetIsAromatic()}
            Chem.GetSymmSSSR(mol)
            rings = []
            for indices in mol.GetRingInfo().BondRings():
                ring = frozenset(edge(mol.GetBondWithIdx(i).GetBeginAtomIdx(),
                                      mol.GetBondWithIdx(i).GetEndAtomIdx()) for i in indices)
                if ring <= aromatic:
                    rings.append(ring)
            self.mols.append(mol)
            self.bonds.append(bonds)
            self.aromatic.append(aromatic)
            self.rings.append(rings)
        self.target_rings = set(self.rings[1])

    def clean_mapping(self, mapping):
        mapping = list(map(int, mapping))
        assert len(mapping) == len(self.pair['left']['nodes'])
        used = set()
        for u, v in enumerate(mapping):
            assert -1 <= v < len(self.pair['right']['nodes'])
            if v >= 0:
                assert v not in used
                used.add(v)
                if self.pair['left']['nodes'][u] != self.pair['right']['nodes'][v]:
                    mapping[u] = -1
        return mapping

    def project(self, mapping, allowed_edges=None):
        """Keep all complete mapped cycles and all compatible nonaromatic bonds.

        allowed_edges restricts native solver bond witnesses: projection must
        never add a bond absent from the supplied native candidate.
        """
        mapping = self.clean_mapping(mapping)
        compatible = {}
        for (u, v), label in self.bonds[0].items():
            if mapping[u] < 0 or mapping[v] < 0:
                continue
            if allowed_edges is not None and (u, v) not in allowed_edges:
                continue
            mapped = edge(mapping[u], mapping[v])
            if self.bonds[1].get(mapped) == label:
                compatible[(u, v)] = mapped
        retained = {e for e, f in compatible.items()
                    if e not in self.aromatic[0] and f not in self.aromatic[1]}
        matched_rings = []
        for ring in self.rings[0]:
            if not ring <= compatible.keys():
                continue
            mapped_ring = frozenset(compatible[e] for e in ring)
            if mapped_ring in self.target_rings:
                retained.update(ring)
                matched_rings.append({'left': sorted(ring), 'right': sorted(mapped_ring)})
        witness = [[u, v, mapping[u], mapping[v]] for u, v in sorted(retained)]
        return {'mapping': mapping, 'bond_witness': witness,
                'common_edges': len(witness),
                'common_nodes': len({u for e in retained for u in e}),
                'raw_common_edges': len(compatible),
                'removed_aromatic_edges': len(compatible) - len(witness),
                'complete_ring_witnesses': matched_rings}

    def verify(self, result):
        """Validate the submitted edge subset, independently of its selection."""
        mapping = result['mapping']
        assert self.clean_mapping(mapping) == mapping
        source, target = set(), set()
        for u, v, a, b in result['bond_witness']:
            assert mapping[u] == a and mapping[v] == b
            e, f = edge(u, v), edge(a, b)
            assert e not in source and f not in target
            assert e in self.bonds[0] and self.bonds[0][e] == self.bonds[1].get(f)
            source.add(e); target.add(f)
        # Reconstruct full-cycle coverage from the witness, not stored ring claims.
        covered_left, covered_right = set(), set()
        for ring in self.rings[0]:
            if ring <= source:
                mapped = frozenset(edge(mapping[u], mapping[v]) for u, v in ring)
                if mapped in self.target_rings and mapped <= target:
                    covered_left.update(ring); covered_right.update(mapped)
        assert source & self.aromatic[0] <= covered_left
        assert target & self.aromatic[1] <= covered_right
        assert result['common_edges'] == len(source)
        assert result['common_nodes'] == len({u for e in source for u in e})
        return True
