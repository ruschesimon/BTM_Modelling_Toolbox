"""Storage parameters and graph ports for a single energy carrier."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

from .component import Component
from ..core.carrier import Carrier
from ..core.port import Port, PortDirection
from ..network.system import EnergySystem
from ..utils.numeric import NumericValue, build_time_series, normalize_numeric_input


StateValue = Real | Literal["min", "mid", "max"]


@dataclass(kw_only=True, init=False)
class Storage(Component):
    """Store a carrier through separate charge and discharge ports.

    ``energy_capacity`` and ``min_energy_content`` are finite nonnegative
    scalars in the carrier's energy unit. Charge/discharge capacities default
    to energy_capacity (legacy convention; specify power limits explicitly
    when time steps are not one hour). Efficiencies default to 1 and losses
    to 0. Relative losses and loss_rate are fractions in [0, 1]; efficiencies
    are in (0, 1]. No balance equations or time-step scaling are applied here.

    Numeric parameter properties return Series indexed by the assigned
    EnergySystem's RangeIndex. Operating parameters accept scalars or Series;
    Series are aligned by label, with missing/duplicate labels rejected on
    access. Initial/final states are scalar energy contents, defaulting to
    the midpoint; 'min', 'mid', and 'max' resolve without an energy system.
    System assignment uses the inherited Vertex methods.
    """

    carrier: Carrier
    _energy_capacity: Real = field(init=False, repr=False)
    _min_energy_content: Real = field(init=False, repr=False)
    _charge_capacity: NumericValue = field(init=False, repr=False)
    _discharge_capacity: NumericValue = field(init=False, repr=False)
    _charge_efficiency: NumericValue = field(init=False, repr=False)
    _discharge_efficiency: NumericValue = field(init=False, repr=False)
    _fixed_loss_absolute: NumericValue = field(init=False, repr=False)
    _fixed_loss_relative: NumericValue = field(init=False, repr=False)
    _loss_rate: NumericValue = field(init=False, repr=False)
    _initial_state: Real = field(init=False, repr=False)
    _final_state: Real = field(init=False, repr=False)

    def __init__(
        self,
        *,
        carrier: Carrier,
        energy_capacity: Real,
        min_energy_content: Real = 0.0,
        charge_capacity: NumericValue | None = None,
        discharge_capacity: NumericValue | None = None,
        initial_state: StateValue = "mid",
        final_state: StateValue = "mid",
        charge_efficiency: NumericValue = 1.0,
        discharge_efficiency: NumericValue = 1.0,
        fixed_loss_absolute: NumericValue = 0.0,
        fixed_loss_relative: NumericValue = 0.0,
        loss_rate: NumericValue = 0.0,
        energy_system: EnergySystem | None = None,
        label: str | None = None,
        variable_cost: NumericValue = 0.0,
        fixed_cost: Real = 0.0,
    ) -> None:
        self.carrier = carrier
        for name, value in (("energy_capacity", energy_capacity),
                            ("min_energy_content", min_energy_content)):
            setattr(self, f"_{name}", self._normalize_parameter(
                value, name, allow_series=False,
            ))
        if self._min_energy_content > self._energy_capacity:
            raise ValueError("min_energy_content must not exceed energy_capacity.")
        parameters = {
            "charge_capacity": energy_capacity if charge_capacity is None else charge_capacity,
            "discharge_capacity": energy_capacity if discharge_capacity is None else discharge_capacity,
            "charge_efficiency": charge_efficiency,
            "discharge_efficiency": discharge_efficiency,
            "fixed_loss_absolute": fixed_loss_absolute,
            "fixed_loss_relative": fixed_loss_relative,
            "loss_rate": loss_rate,
        }
        for name, value in parameters.items():
            normalized = self._normalize_parameter(value, name)
            if name in ("charge_efficiency", "discharge_efficiency"):
                self._check_fraction(normalized, name, strictly_positive=True)
            elif name in ("fixed_loss_relative", "loss_rate"):
                self._check_fraction(normalized, name)
            setattr(self, f"_{name}", normalized)
        self._initial_state = self._resolve_state(initial_state, "initial_state")
        self._final_state = self._resolve_state(final_state, "final_state")
        super().__init__(label=label, variable_cost=variable_cost, fixed_cost=fixed_cost)
        if energy_system is not None:
            self._set_energy_system(energy_system)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.carrier, Carrier):
            raise TypeError("carrier must be an instance of Carrier.")
        for suffix, direction in (("input", PortDirection.INPUT),
                                  ("output", PortDirection.OUTPUT)):
            self.add_port(Port(
                label=f"{self.label}_{self.carrier.label}_{suffix}",
                vertex=self, carrier=self.carrier, direction=direction,
            ))

    @staticmethod
    def _normalize_parameter(value, name, *, allow_series=True):
        normalized = normalize_numeric_input(
            value, name=name, allow_negative=False, allow_series=allow_series,
        )
        if not np.isfinite(normalized).all():
            raise ValueError(f"{name} must contain only finite values.")
        return normalized

    @staticmethod
    def _check_fraction(value, name, *, strictly_positive=False) -> None:
        invalid = (value > 1) | (value <= 0 if strictly_positive else value < 0)
        if np.any(invalid):
            interval = "(0, 1]" if strictly_positive else "[0, 1]"
            raise ValueError(f"{name} must be in {interval}.")

    def _resolve_state(self, value: StateValue, name: str) -> Real:
        if isinstance(value, str):
            states = {
                "min": self._min_energy_content,
                "mid": self._min_energy_content / 2 + self._energy_capacity / 2,
                "max": self._energy_capacity,
            }
            if value not in states:
                raise ValueError(f"{name} must be 'min', 'mid', 'max', or a scalar.")
            value = states[value]
        normalized = self._normalize_parameter(value, name, allow_series=False)
        if not self._min_energy_content <= normalized <= self._energy_capacity:
            raise ValueError(f"{name} must lie between min_energy_content and energy_capacity.")
        return normalized

    @property
    def energy_capacity(self) -> pd.Series:
        return self._parameter_series(self._energy_capacity, "energy_capacity")

    @property
    def min_energy_content(self) -> pd.Series:
        return self._parameter_series(self._min_energy_content, "min_energy_content")

    @property
    def charge_capacity(self) -> pd.Series:
        return self._parameter_series(self._charge_capacity, "charge_capacity")

    @property
    def discharge_capacity(self) -> pd.Series:
        return self._parameter_series(self._discharge_capacity, "discharge_capacity")

    @property
    def charge_efficiency(self) -> pd.Series:
        return self._parameter_series(self._charge_efficiency, "charge_efficiency")

    @property
    def discharge_efficiency(self) -> pd.Series:
        return self._parameter_series(self._discharge_efficiency, "discharge_efficiency")

    @property
    def fixed_loss_absolute(self) -> pd.Series:
        return self._parameter_series(self._fixed_loss_absolute, "fixed_loss_absolute")

    @property
    def fixed_loss_relative(self) -> pd.Series:
        return self._parameter_series(self._fixed_loss_relative, "fixed_loss_relative")

    @property
    def loss_rate(self) -> pd.Series:
        return self._parameter_series(self._loss_rate, "loss_rate")

    @property
    def initial_state(self) -> Real:
        return self._initial_state

    @property
    def final_state(self) -> Real:
        return self._final_state
