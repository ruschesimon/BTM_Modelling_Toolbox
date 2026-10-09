# -------------------- Public Imports --------------------

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from uuid import UUID


# -------------------- Local Imports --------------------

from ..core.entity import Entity
from ..core.port import Port
from ..core.vertex import Vertex
from ..utils.numeric import normalize_numeric_input, ScalarValue, NumericValue

# -------------------- Class Definition --------------------

@dataclass(kw_only=True)
class Component(Vertex, ABC):
    """
    Abstrakte Basisklasse für Komponenten eines Energiesystems.

    Eine Komponente besitzt eine Menge von Ports, über die sie
    mit anderen Komponenten verbunden werden kann.
    """

    variable_cost: NumericValue = 0.0
    fixed_cost: ScalarValue = 0.0


    def __post_init__(self) -> None:
        super().__post_init__()

        self._normalize_costs()


    def _normalize_costs(self) -> None:
        self.variable_cost = normalize_numeric_input(
            self.variable_cost,
            name="variable_cost",
        )

        self.fixed_cost = normalize_numeric_input(
            self.fixed_cost,
            name="fixed_cost",
            allow_negative=False,
            allow_series=False,
        )

    '''
    def get_variableCost(self, timeIndex):
        timeSeries = build_time_series(self.variable_cost, "variable_cost", timeIndex)
        return timeSeries

    
    def get_fixedCost(self):
        return self.fixed_cost
    '''