"""Fuel boiler with the reference model's affine part-load curve."""

from __future__ import annotations

from numbers import Real

import pandas as pd

from ..converter import Converter
from ...core.carrier import Carrier, CarrierCategory
from ...network.system import EnergySystem
from ...utils.numeric import NumericValue
from ._thermal import affine_fuel_curve, carrier, efficiency, numeric, operation_parameters


class HeatBoiler(Converter):
    """Provide heat with fuel = A * operating + B * heat.

    power_min/max_capacity are the positive finite reference output powers
    used to fit A and B, not additional fuel-input bounds. heat_min/max_capacity
    are the independent delivered-heat bounds from the legacy interface.
    Efficiencies are dimensionless scalars at those two reference powers.
    All power quantities must use the same energy/time unit (e.g. MW).
    A has power units and B is dimensionless; coefficients are valid on the
    reference load interval. No on/off, ramp, or minimum-time solver is added.
    """

    _allow_signed_conversion = True

    def __init__(
        self, *, input_carrier: Carrier, output_carrier: Carrier,
        power_min_efficiency: Real, power_max_efficiency: Real,
        power_min_capacity: NumericValue, power_max_capacity: NumericValue,
        heat_min_capacity: NumericValue = 0., heat_max_capacity: NumericValue = float('inf'),
        min_gradient: Real = -float('inf'), max_gradient: Real = float('inf'),
        min_down_time: int = 0, min_up_time: int = 0, initial_operation: bool = False,
        energy_system: EnergySystem | None = None, label: str | None = None,
        variable_cost: NumericValue = 0., fixed_cost: Real = 0.,
    ) -> None:
        self.input_carrier = carrier(input_carrier, 'input_carrier',
                                     (CarrierCategory.GAS, CarrierCategory.LIQUID_FUEL))
        self.output_carrier = carrier(output_carrier, 'output_carrier', (CarrierCategory.HEAT,))
        self.power_min_efficiency = efficiency(power_min_efficiency, 'power_min_efficiency')
        self.power_max_efficiency = efficiency(power_max_efficiency, 'power_max_efficiency')
        self._power_min_capacity = numeric(power_min_capacity, 'power_min_capacity', nonnegative=True)
        self._power_max_capacity = numeric(power_max_capacity, 'power_max_capacity', nonnegative=True)
        intercept, slope = affine_fuel_curve(
            self._power_min_capacity, self._power_max_capacity,
            self.power_min_efficiency, self.power_max_efficiency,
        )
        operation_parameters(self, min_gradient=min_gradient, max_gradient=max_gradient,
                             min_down_time=min_down_time, min_up_time=min_up_time,
                             initial_operation=initial_operation)
        super().__init__(
            input_carriers=input_carrier, output_carriers=output_carrier,
            input_conversion=slope, input_offset=intercept,
            output_conversion=1., output_min=heat_min_capacity, output_max=heat_max_capacity,
            energy_system=energy_system, label=label, variable_cost=variable_cost, fixed_cost=fixed_cost,
        )

    @property
    def efficiency_absolute(self) -> pd.Series:
        return self.input_offset[self.input_carrier.id].rename('efficiency_absolute')

    @property
    def efficiency_relative(self) -> pd.Series:
        return self.input_conversion[self.input_carrier.id].rename('efficiency_relative')

    @property
    def power_min_capacity(self) -> pd.Series:
        return self._parameter_series(self._power_min_capacity, 'power_min_capacity')

    @property
    def power_max_capacity(self) -> pd.Series:
        return self._parameter_series(self._power_max_capacity, 'power_max_capacity')

    @property
    def heat_min_capacity(self) -> pd.Series:
        return self.output_min[self.output_carrier.id].rename('heat_min_capacity')

    @property
    def heat_max_capacity(self) -> pd.Series:
        return self.output_max[self.output_carrier.id].rename('heat_max_capacity')
