# -------------------- Public Imports --------------------

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field

import pandas as pd


# -------------------- Local Imports --------------------

from ..core.vertex import Vertex
from ..network.system import EnergySystem
from ..utils.numeric import normalize_numeric_input, build_time_series, ScalarValue, NumericValue

# -------------------- Class Definition --------------------

@dataclass(kw_only=True, init=False)
class Component(Vertex, ABC):
    """
    Abstrakte Basisklasse für Komponenten eines Energiesystems.

    Eine Komponente besitzt eine Menge von Ports, über die sie
    mit anderen Komponenten verbunden werden kann.
    """

    _variable_cost: NumericValue = field(init=False, repr=False)
    fixed_cost: ScalarValue = 0.0

    def __init__(
        self,
        *,
        variable_cost: NumericValue = 0.0,
        fixed_cost: ScalarValue = 0.0,
        label: str | None = None,
    ) -> None:
        self._variable_cost = variable_cost
        self.fixed_cost = fixed_cost
        super().__init__(label=label)

    @property
    def variable_cost(self) -> pd.Series:
        """Return costs on the assigned energy system's RangeIndex.

        Scalar inputs are broadcast; Series are aligned by index label.
        Negative costs are allowed. Access without a system raises
        RuntimeError; invalid system/index types raise TypeError.
        """
        return self._parameter_series(self._variable_cost, "variable_cost")


    def __post_init__(self) -> None:
        super().__post_init__()

        self._normalize_costs()


    def _normalize_costs(self) -> None:
        self._variable_cost = normalize_numeric_input(
            self._variable_cost,
            name="variable_cost",
        )

        self.fixed_cost = normalize_numeric_input(
            self.fixed_cost,
            name="fixed_cost",
            allow_negative=False,
            allow_series=False,
        )

    def _parameter_series(self, value: NumericValue, name: str) -> pd.Series:
        system = self.energy_system
        if not isinstance(system, EnergySystem):
            raise TypeError("energy_system must be an instance of EnergySystem.")
        if not isinstance(system.time_index, pd.RangeIndex):
            raise TypeError("energy_system.time_index must be a pandas RangeIndex.")
        return build_time_series(value, name, system.time_index)
