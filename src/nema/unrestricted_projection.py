"""Score explicit labeled MCES witnesses without a complete-cycle restriction."""


def edge(u, v):
    return (min(u, v), max(u, v))


class UnrestrictedProjection:
    def __init__(self, pair):
        self.pair = pair
        self.bonds = [{edge(u, v): label for u, v, label in pair[side]['edges']}
                      for side in ('left', 'right')]

    def clean_mapping(self, mapping):
        mapping = list(map(int, mapping))
        left, right = self.pair['left']['nodes'], self.pair['right']['nodes']
        assert len(mapping) == len(left)
        used = set()
        for u, v in enumerate(mapping):
            assert -1 <= v < len(right)
            if v >= 0:
                assert v not in used
                used.add(v)
                if left[u] != right[v]:
                    mapping[u] = -1
        return mapping

    def project(self, mapping, allowed_edges=None):
        mapping = self.clean_mapping(mapping)
        witness = []
        for (u, v), label in sorted(self.bonds[0].items()):
            if mapping[u] < 0 or mapping[v] < 0:
                continue
            if allowed_edges is not None and (u, v) not in allowed_edges:
                continue
            if self.bonds[1].get(edge(mapping[u], mapping[v])) == label:
                witness.append([u, v, mapping[u], mapping[v]])
        return {'mapping': mapping, 'bond_witness': witness,
                'common_edges': len(witness),
                'common_nodes': len({u for e in witness for u in e[:2]})}

    def verify(self, result):
        mapping = result['mapping']
        assert self.clean_mapping(mapping) == mapping
        source, target = set(), set()
        for u, v, a, b in result['bond_witness']:
            assert mapping[u] == a and mapping[v] == b and a >= 0 and b >= 0
            e, f = edge(u, v), edge(a, b)
            assert e not in source and f not in target
            assert e in self.bonds[0] and self.bonds[0][e] == self.bonds[1].get(f)
            source.add(e)
            target.add(f)
        assert result['common_edges'] == len(source)
        assert result['common_nodes'] == len({u for e in source for u in e})
        return True
