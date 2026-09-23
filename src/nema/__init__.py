"""Equivariant Neural Primal-Dual Assignment for labeled MCES.

The ``nema`` namespace is retained for checkpoint and import compatibility.
"""

from nema.association import AssociationGraph
from nema.graph import GraphPair, LabeledGraph
from nema.models.nema import NEMAModel
from nema.models.enpda import ENPDAModel
from nema.solvers import NGASolver, NEMASolver, SolveResult

__all__ = [
    "AssociationGraph",
    "GraphPair",
    "LabeledGraph",
    "NEMAModel",
    "ENPDAModel",
    "NEMASolver",
    "NGASolver",
    "SolveResult",
]

__version__ = "0.1.0"

