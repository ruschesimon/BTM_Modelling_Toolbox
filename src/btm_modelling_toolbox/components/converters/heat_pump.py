"""Heat pump with a calibrated temperature-dependent Carnot COP."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

import numpy as np
import pandas as pd

from ..converter import Converter
from ...core.carrier import Carrier, CarrierCategory
from ...network.system import EnergySystem
from ...utils.numeric import NumericValue, NormalizedNumeric
from ._thermal import carrier, kelvin, matching_indices, numeric, operation_parameters


class HeatPump(Converter):
    """Transfer source heat plus electrical input into delivered heat.

    Activity is electrical power W. Q_out=COP*W, Q_source=(COP-1)*W.
    Both the electricity and environmental-heat input ports from the reference
    are retained. Source and delivered heat must use distinct carrier IDs.

    COP=k*(T_supply+dT)/(T_supply-T_source+2*dT), with k calibrated to
    norm_cop at the nominal temperatures. Absolute temperatures are converted
    to Kelvin; temperature_spread is a Kelvin difference even for Fahrenheit
    input (matching the reference). supply_temperature defaults to the nominal
    supply temperature; source_temperature is required. Explicit compute_cop
    overrides use temperature_unit and do not modify configured coefficients.
    COP >= 1 and positive effective temperature lift are required for this
    nonnegative two-input heating model. No defrost/cooling model is implied.
    Capacities use a common power unit (e.g. MW); COP is dimensionless.
    """

    def __init__(
        self, *, input_carriers: Iterable[Carrier], output_carrier: Carrier,
        norm_cop: Real, norm_supply_temperature: Real, norm_source_temperature: Real,
        source_temperature: NumericValue, temperature_unit: str,
        heat_min_capacity: NumericValue, heat_max_capacity: NumericValue,
        supply_temperature: NumericValue | None = None, temperature_spread: Real = 5.,
        power_min_capacity: NumericValue = 0., power_max_capacity: NumericValue = float('inf'),
        min_down_time: int = 0, min_up_time: int = 0, initial_operation: bool = False,
        energy_system: EnergySystem | None = None, label: str | None = None,
        variable_cost: NumericValue = 0., fixed_cost: Real = 0.,
    ) -> None:
        inputs = self._normalize_carriers(input_carriers, 'input_carriers')
        electricity = [c for c in inputs if c.category == CarrierCategory.ELECTRICITY]
        heat = [c for c in inputs if c.category == CarrierCategory.HEAT]
        if len(inputs) != 2 or len(electricity) != 1 or len(heat) != 1:
            raise ValueError('input_carriers must contain one electricity and one source heat carrier.')
        self.electricity_carrier, self.source_heat_carrier = electricity[0], heat[0]
        self.output_carrier = carrier(output_carrier, 'output_carrier', (CarrierCategory.HEAT,))
        self.norm_cop = numeric(norm_cop, 'norm_cop', scalar=True, nonnegative=True)
        if self.norm_cop < 1:
            raise ValueError('norm_cop must be at least one for the two-input heating model.')
        self.temperature_unit = temperature_unit
        self.temperature_spread = numeric(temperature_spread, 'temperature_spread',
                                          scalar=True, nonnegative=True)
        self.norm_supply_temperature = kelvin(
            numeric(norm_supply_temperature, 'norm_supply_temperature', scalar=True),
            temperature_unit, 'norm_supply_temperature',
        )
        self.norm_source_temperature = kelvin(
            numeric(norm_source_temperature, 'norm_source_temperature', scalar=True),
            temperature_unit, 'norm_source_temperature',
        )
        nominal_carnot = self._carnot(self.norm_supply_temperature, self.norm_source_temperature)
        self._cop_factor = self.norm_cop / nominal_carnot
        self._source_temperature = kelvin(source_temperature, temperature_unit, 'source_temperature')
        self._supply_temperature = (self.norm_supply_temperature if supply_temperature is None
                                    else kelvin(supply_temperature, temperature_unit, 'supply_temperature'))
        cop = self._raw_cop(self._supply_temperature, self._source_temperature)
        operation_parameters(self, min_down_time=min_down_time, min_up_time=min_up_time,
                             initial_operation=initial_operation)
        super().__init__(
            input_carriers=inputs, output_carriers=output_carrier,
            input_conversion={self.electricity_carrier.id: 1., self.source_heat_carrier.id: cop - 1},
            output_conversion=cop,
            input_min={self.electricity_carrier.id: power_min_capacity, self.source_heat_carrier.id: 0.},
            input_max={self.electricity_carrier.id: power_max_capacity, self.source_heat_carrier.id: float('inf')},
            output_min=heat_min_capacity, output_max=heat_max_capacity,
            energy_system=energy_system, label=label, variable_cost=variable_cost, fixed_cost=fixed_cost,
        )

    def _carnot(self, supply: NumericValue, source: NumericValue) -> NormalizedNumeric:
        matching_indices(supply, source)
        hot, cold = supply + self.temperature_spread, source - self.temperature_spread
        if np.any(cold <= 0) or np.any(hot <= cold):
            raise ValueError('Effective temperatures must satisfy 0 < T_cold < T_hot in Kelvin.')
        return hot / (hot - cold)

    def _raw_cop(self, supply: NumericValue, source: NumericValue) -> NormalizedNumeric:
        result = self._cop_factor * self._carnot(supply, source)
        if not np.isfinite(result).all() or np.any(result < 1):
            raise ValueError('Operating temperatures must produce finite COP values >= 1.')
        return result

    def compute_cop(self, supply_temperature: NumericValue | None = None,
                    source_temperature: NumericValue | None = None) -> pd.Series:
        """Evaluate the reference COP equation without changing configuration.

        Explicit temperatures use temperature_unit; omitted values use the
        configured Kelvin temperatures. The result uses energy_system.time_index.
        """
        supply = (self._supply_temperature if supply_temperature is None else
                  kelvin(supply_temperature, self.temperature_unit, 'supply_temperature'))
        source = (self._source_temperature if source_temperature is None else
                  kelvin(source_temperature, self.temperature_unit, 'source_temperature'))
        return self._parameter_series(self._raw_cop(supply, source), 'cop')

    @property
    def cop(self) -> pd.Series:
        return self.output_conversion[self.output_carrier.id].rename('cop')

    @property
    def source_temperature(self) -> pd.Series:
        """Configured source temperature in Kelvin."""
        return self._parameter_series(self._source_temperature, 'source_temperature')

    @property
    def supply_temperature(self) -> pd.Series:
        """Configured supply temperature in Kelvin."""
        return self._parameter_series(self._supply_temperature, 'supply_temperature')

    @property
    def power_min_capacity(self) -> pd.Series:
        return self.input_min[self.electricity_carrier.id].rename('power_min_capacity')

    @property
    def power_max_capacity(self) -> pd.Series:
        return self.input_max[self.electricity_carrier.id].rename('power_max_capacity')

    @property
    def heat_min_capacity(self) -> pd.Series:
        return self.output_min[self.output_carrier.id].rename('heat_min_capacity')

    @property
    def heat_max_capacity(self) -> pd.Series:
        return self.output_max[self.output_carrier.id].rename('heat_max_capacity')
