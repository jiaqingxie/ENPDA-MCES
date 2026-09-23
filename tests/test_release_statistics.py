import numpy as np

from scripts.summarize_release import connected_pair_blocks, paired_interval


def test_shared_graphs_are_grouped_transitively():
    pairs = [{"identities": x} for x in ((1, 2), (3, 4), (2, 3), (5, 6), (7, 7))]
    groups = {frozenset(g) for g in connected_pair_blocks(pairs)}
    assert groups == {frozenset((0, 1, 2)), frozenset((3,)), frozenset((4,))}


def test_paired_bootstrap_preserves_constant_difference():
    result = paired_interval(np.full((3, 4), 2.5), [[0, 1, 2], [3]], draws=50)
    assert result == [2.5, 2.5]
