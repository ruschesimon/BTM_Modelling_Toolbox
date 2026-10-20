"""A four-port transformer with parameters for bidirectional transfer."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from ..converter import Converter
from ...core.carrier import CurrentType, Electricity
from ...core.port import Port
from ...network.system import EnergySystem
from ...utils.numeric import NumericValue
from ._thermal import matching_indices, numeric


class Transformer(Converter):
    """Provide AC input and output ports on each of two voltage levels.

    Both port pairs remain available. This class stores capacity (sending-end
    MW), efficiency in (0, 1], and no_load_loss (MW), but selects no operating
    direction and imposes no mutual exclusion. Each input has capacity as its
    upper bound; each output has efficiency*capacity as an upper bound.

    Conversion factors are 1 on both inputs and efficiency on both outputs.
    A separate transformer formulation must route the directional flows and
    apply the shared capacity and energized-state loss. no_load_loss is kept
    as a device parameter, not subtracted independently at both outputs.
    No reactive power, impedance, tap changing or AC load flow is modeled.
    """

    _allow_shared_carriers = True

    def __init__(
        self, *, primary_carrier: Electricity, secondary_carrier: Electricity,
        capacity: NumericValue, efficiency: NumericValue = 1., no_load_loss: NumericValue = 0.,
        energy_system: EnergySystem | None = None, label: str | None = None,
        variable_cost: NumericValue = 0., fixed_cost: Real = 0.,
    ) -> None:
        for name, value in (('primary_carrier', primary_carrier), ('secondary_carrier', secondary_carrier)):
            if not isinstance(value, Electricity):
                raise TypeError(f'{name} must be an Electricity instance.')
            if value.current_type is not CurrentType.AC:
                raise ValueError(f'{name} must use alternating current (AC).')
            # Recheck characteristics because carrier configuration is mutable.
            numeric(value.voltage_level, f'{name}.voltage_level', scalar=True, nonnegative=True)
            if value.voltage_level <= 0:
                raise ValueError('Both voltages must be known and greater than zero.')
        if primary_carrier.voltage_level == secondary_carrier.voltage_level:
            raise ValueError('Primary and secondary voltage levels must differ.')
        if primary_carrier.id == secondary_carrier.id:
            raise ValueError('Primary and secondary carriers must have distinct IDs.')
        self.primary_carrier, self.secondary_carrier = primary_carrier, secondary_carrier
        self._capacity = numeric(capacity, 'capacity', nonnegative=True)
        self._efficiency = numeric(efficiency, 'efficiency', nonnegative=True)
        self._no_load_loss = numeric(no_load_loss, 'no_load_loss', nonnegative=True)
        matching_indices(self._capacity, self._efficiency, self._no_load_loss)
        if np.any(self._efficiency <= 0) or np.any(self._efficiency > 1):
            raise ValueError('efficiency must be in (0, 1].')
        carriers = (primary_carrier, secondary_carrier)
        super().__init__(
            input_carriers=carriers, output_carriers=carriers,
            input_conversion={c.id: 1. for c in carriers},
            output_conversion={c.id: self._efficiency for c in carriers},
            input_max={c.id: self._capacity for c in carriers},
            output_max={c.id: self._efficiency * self._capacity for c in carriers},
            energy_system=energy_system, label=label,
            variable_cost=variable_cost, fixed_cost=fixed_cost,
        )

    @property
    def capacity(self) -> pd.Series:
        return self._parameter_series(self._capacity, 'capacity')

    @property
    def efficiency(self) -> pd.Series:
        return self._parameter_series(self._efficiency, 'efficiency')

    @property
    def no_load_loss(self) -> pd.Series:
        return self._parameter_series(self._no_load_loss, 'no_load_loss')

    def _side_port(self, carrier: Electricity, *, input_: bool) -> Port:
        ports = self.input_ports if input_ else self.output_ports
        return next(port for port in ports if port.carrier is carrier)

    @property
    def primary_input(self) -> Port:
        return self._side_port(self.primary_carrier, input_=True)

    @property
    def primary_output(self) -> Port:
        return self._side_port(self.primary_carrier, input_=False)

    @property
    def secondary_input(self) -> Port:
        return self._side_port(self.secondary_carrier, input_=True)

    @property
    def secondary_output(self) -> Port:
        return self._side_port(self.secondary_carrier, input_=False)
