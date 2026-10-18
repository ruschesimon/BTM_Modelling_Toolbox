"""Combined heat and power with affine fuel and heat coupling."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

import numpy as np
import pandas as pd

from ..converter import Converter
from ...core.carrier import Carrier, CarrierCategory
from ...network.system import EnergySystem
from ...utils.numeric import NumericValue
from ._thermal import affine_fuel_curve, carrier, efficiency, numeric, operation_parameters


class CombinedHeatPower(Converter):
    """Use electrical power P as activity: F=A*u+B*P; Q=eta_total*F-P.

    A and B reproduce the electrical efficiencies at the two reference
    electrical powers. u is the operating indicator. This retains the legacy
    affine heat/power coupling, rather than assuming a constant ratio.
    Fuel, power and heat use the same power unit (e.g. MW); A has power units,
    B and efficiencies are dimensionless. Signed slopes/offsets are allowed:
    nonnegative physical flows are required only on the rated load interval.
    Operating metadata is retained without generating solver constraints.
    """

    _allow_signed_conversion = True

    def __init__(
        self, *, input_carrier: Carrier, output_carriers: Iterable[Carrier],
        power_min_efficiency: Real, power_max_efficiency: Real, chp_total_efficiency: Real,
        power_min_capacity: NumericValue, power_max_capacity: NumericValue,
        heat_min_capacity: NumericValue = 0., heat_max_capacity: NumericValue = float('inf'),
        min_gradient: Real = -float('inf'), max_gradient: Real = float('inf'),
        min_down_time: int = 0, min_up_time: int = 0, initial_operation: bool = False,
        energy_system: EnergySystem | None = None, label: str | None = None,
        variable_cost: NumericValue = 0., fixed_cost: Real = 0.,
    ) -> None:
        self.input_carrier = carrier(input_carrier, 'input_carrier',
                                     (CarrierCategory.GAS, CarrierCategory.LIQUID_FUEL))
        outputs = self._normalize_carriers(output_carriers, 'output_carriers')
        electricity = [c for c in outputs if c.category == CarrierCategory.ELECTRICITY]
        heat = [c for c in outputs if c.category == CarrierCategory.HEAT]
        if len(outputs) != 2 or len(electricity) != 1 or len(heat) != 1:
            raise ValueError('output_carriers must contain one electricity and one heat carrier.')
        self.electricity_carrier, self.heat_carrier = electricity[0], heat[0]
        self.power_min_efficiency = efficiency(power_min_efficiency, 'power_min_efficiency')
        self.power_max_efficiency = efficiency(power_max_efficiency, 'power_max_efficiency')
        self.chp_total_efficiency = efficiency(chp_total_efficiency, 'chp_total_efficiency')
        if self.chp_total_efficiency < max(self.power_min_efficiency, self.power_max_efficiency):
            raise ValueError('chp_total_efficiency must not be below the electrical efficiencies.')
        lower = numeric(power_min_capacity, 'power_min_capacity', nonnegative=True)
        upper = numeric(power_max_capacity, 'power_max_capacity', nonnegative=True)
        intercept, slope = affine_fuel_curve(
            lower, upper, self.power_min_efficiency, self.power_max_efficiency,
        )
        coupling_absolute = self.chp_total_efficiency * intercept
        coupling_relative = self.chp_total_efficiency * slope - 1
        if (not np.isfinite(coupling_absolute).all()
                or not np.isfinite(coupling_relative).all()):
            raise ValueError('Reference data produce non-finite heat coupling coefficients.')
        operation_parameters(self, min_gradient=min_gradient, max_gradient=max_gradient,
                             min_down_time=min_down_time, min_up_time=min_up_time,
                             initial_operation=initial_operation)
        super().__init__(
            input_carriers=input_carrier, output_carriers=outputs,
            input_conversion=slope, input_offset=intercept,
            output_conversion={self.electricity_carrier.id: 1., self.heat_carrier.id: coupling_relative},
            output_offset={self.electricity_carrier.id: 0., self.heat_carrier.id: coupling_absolute},
            output_min={self.electricity_carrier.id: lower, self.heat_carrier.id: heat_min_capacity},
            output_max={self.electricity_carrier.id: upper, self.heat_carrier.id: heat_max_capacity},
            energy_system=energy_system, label=label, variable_cost=variable_cost, fixed_cost=fixed_cost,
        )

    @property
    def efficiency_absolute(self) -> pd.Series:
        return self.input_offset[self.input_carrier.id].rename('efficiency_absolute')

    @property
    def efficiency_relative(self) -> pd.Series:
        return self.input_conversion[self.input_carrier.id].rename('efficiency_relative')

    @property
    def coupling_absolute(self) -> pd.Series:
        return self.output_offset[self.heat_carrier.id].rename('coupling_absolute')

    @property
    def coupling_relative(self) -> pd.Series:
        return self.output_conversion[self.heat_carrier.id].rename('coupling_relative')

    @property
    def power_min_capacity(self) -> pd.Series:
        return self.output_min[self.electricity_carrier.id].rename('power_min_capacity')

    @property
    def power_max_capacity(self) -> pd.Series:
        return self.output_max[self.electricity_carrier.id].rename('power_max_capacity')

    @property
    def heat_min_capacity(self) -> pd.Series:
        return self.output_min[self.heat_carrier.id].rename('heat_min_capacity')

    @property
    def heat_max_capacity(self) -> pd.Series:
        return self.output_max[self.heat_carrier.id].rename('heat_max_capacity')
