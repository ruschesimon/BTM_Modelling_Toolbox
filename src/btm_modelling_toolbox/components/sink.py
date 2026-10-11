"""A terminal consumer of an energy carrier."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import pandas as pd

from .component import Component
from ..core.carrier import Carrier
from ..core.port import Port, PortDirection
from ..network.system import EnergySystem
from ..utils.numeric import NumericValue, build_time_series, normalize_numeric_input


@dataclass(kw_only=True, init=False)
class Sink(Component):
    """Consume a carrier with scalar or time-dependent capacity bounds.

    Capacity properties require an EnergySystem with a pandas RangeIndex.
    Initialization without an energy system is allowed; reading capacities
    before assignment raises RuntimeError. Series are aligned by index label.
    """

    carrier: Carrier
    _min_capacity: NumericValue = field(init=False, repr=False)
    _max_capacity: NumericValue = field(init=False, repr=False)

    def __init__(
        self,
        *,
        carrier: Carrier,
        min_capacity: NumericValue = 0.0,
        max_capacity: NumericValue = float("inf"),
        energy_system: EnergySystem | None = None,
        label: str | None = None,
        variable_cost: NumericValue = 0.0,
        fixed_cost: Real = 0.0,
    ) -> None:
        self.carrier = carrier
        self._min_capacity = normalize_numeric_input(
            min_capacity, name="min_capacity", allow_negative=False,
        )
        self._max_capacity = normalize_numeric_input(
            max_capacity, name="max_capacity", allow_negative=False,
        )
        super().__init__(
            label=label, variable_cost=variable_cost, fixed_cost=fixed_cost,
        )
        if energy_system is not None:
            self._set_energy_system(energy_system)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.carrier, Carrier):
            raise TypeError("carrier must be an instance of Carrier.")
        self._validate_capacities()
        self.add_port(Port(
            label=f"{self.label}_{self.carrier.label}_input",
            vertex=self,
            carrier=self.carrier,
            direction=PortDirection.INPUT,
        ))

    @property
    def min_capacity(self) -> pd.Series:
        return self._parameter_series(self._min_capacity, "min_capacity")

    @property
    def max_capacity(self) -> pd.Series:
        return self._parameter_series(self._max_capacity, "max_capacity")

    def _validate_capacities(self) -> None:
        minimum, maximum = self._min_capacity, self._max_capacity
        if isinstance(minimum, pd.Series) and isinstance(maximum, pd.Series):
            if not minimum.index.equals(maximum.index):
                raise ValueError("min_capacity and max_capacity must have identical indices.")
        invalid = maximum < minimum
        if isinstance(invalid, pd.Series):
            invalid = invalid.any()
        if invalid:
            raise ValueError("max_capacity must be greater than or equal to min_capacity.")