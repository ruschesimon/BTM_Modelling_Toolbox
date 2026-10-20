"""A single-carrier distribution junction with a throughput capacity."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .bus import Bus
from .system import EnergySystem
from ..core.carrier import Carrier
from ..utils.numeric import NumericValue, build_time_series, normalize_numeric_input


@dataclass(kw_only=True, init=False)
class Distribution(Bus):
    """Extend Bus with a nonnegative scalar or time-dependent capacity.

    Capacity defaults to infinity (unlimited) and uses the carrier's flow unit.
    The property returns a Series aligned by label to energy_system.time_index,
    which must be a pandas RangeIndex. Assignment and ports are inherited from
    Vertex/Bus. This class declares capacity without generating flow constraints.
    """

    _capacity: NumericValue = field(init=False, repr=False)

    def __init__(
        self, *, carrier: Carrier, capacity: NumericValue = float('inf'),
        energy_system: EnergySystem | None = None, label: str | None = None,
    ) -> None:
        self._capacity = normalize_numeric_input(capacity, name='capacity', allow_negative=False)
        super().__init__(carrier=carrier, label=label)
        if energy_system is not None:
            self._set_energy_system(energy_system)

    @property
    def capacity(self) -> pd.Series:
        system = self.energy_system
        if not isinstance(system, EnergySystem):
            raise TypeError('energy_system must be an instance of EnergySystem.')
        if not isinstance(system.time_index, pd.RangeIndex):
            raise TypeError('energy_system.time_index must be a pandas RangeIndex.')
        return build_time_series(self._capacity, 'capacity', system.time_index)
