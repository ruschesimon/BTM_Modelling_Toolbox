"""Multi-carrier conversion parameters relative to a shared activity."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from uuid import UUID

import numpy as np
import pandas as pd

from .component import Component
from ..core.carrier import Carrier
from ..core.port import Port, PortDirection
from ..network.system import EnergySystem
from ..utils.numeric import NumericValue, NormalizedNumeric, normalize_numeric_input


CarrierParameter = NumericValue | Mapping[UUID, NumericValue]


@dataclass(kw_only=True, init=False)
class Converter(Component):
    """Convert between one or more input and output carriers.

    Each carrier gets a directional port. Conversion factors multiply a
    common activity variable: input/output flow = factor * activity. Factors
    are finite and nonnegative (zero is allowed); they need not be <= 1.
    This class declares parameters, without constructing balance equations.

    For one carrier, parameters accept a scalar or Series directly. For
    multiple carriers, provide a complete mapping keyed by carrier.id (UUID).
    Parameter properties return read-only UUID-to-Series mappings, aligned
    to energy_system.time_index through Component._parameter_series.
    Bounds default to zero/infinity. A carrier cannot appear on both sides;
    different carrier instances may have the same label.
    """

    _input_carriers: tuple[Carrier, ...] = field(init=False, repr=False)
    _output_carriers: tuple[Carrier, ...] = field(init=False, repr=False)
    _input_conversion: dict[UUID, NormalizedNumeric] = field(init=False, repr=False)
    _output_conversion: dict[UUID, NormalizedNumeric] = field(init=False, repr=False)
    _input_min: dict[UUID, NormalizedNumeric] = field(init=False, repr=False)
    _input_max: dict[UUID, NormalizedNumeric] = field(init=False, repr=False)
    _output_min: dict[UUID, NormalizedNumeric] = field(init=False, repr=False)
    _output_max: dict[UUID, NormalizedNumeric] = field(init=False, repr=False)

    def __init__(
        self,
        *,
        input_carriers: Carrier | Iterable[Carrier],
        output_carriers: Carrier | Iterable[Carrier],
        input_conversion: CarrierParameter,
        output_conversion: CarrierParameter,
        input_min: CarrierParameter | None = None,
        input_max: CarrierParameter | None = None,
        output_min: CarrierParameter | None = None,
        output_max: CarrierParameter | None = None,
        energy_system: EnergySystem | None = None,
        label: str | None = None,
        variable_cost: NumericValue = 0.0,
        fixed_cost: Real = 0.0,
    ) -> None:
        self._input_carriers = self._normalize_carriers(input_carriers, 'input_carriers')
        self._output_carriers = self._normalize_carriers(output_carriers, 'output_carriers')
        if {c.id for c in self.inputs} & {c.id for c in self.outputs}:
            raise ValueError('Input and output carriers must be disjoint.')
        for side, carriers, conversion, minimum, maximum in (
            ('input', self.inputs, input_conversion, input_min, input_max),
            ('output', self.outputs, output_conversion, output_min, output_max),
        ):
            for suffix, value, default in (
                ('conversion', conversion, None), ('min', minimum, 0.),
                ('max', maximum, float('inf')),
            ):
                name = f'{side}_{suffix}'
                if value is None and default is not None:
                    value = {c.id: default for c in carriers}
                normalized = self._normalize_parameters(value, carriers, name)
                if suffix == 'conversion' and any(
                    not np.isfinite(v).all() for v in normalized.values()
                ):
                    raise ValueError(f'{name} must contain only finite values.')
                setattr(self, f'_{name}', normalized)
            self._validate_bounds(getattr(self, f'_{side}_min'),
                                  getattr(self, f'_{side}_max'), side)
        super().__init__(label=label, variable_cost=variable_cost, fixed_cost=fixed_cost)
        if energy_system is not None:
            self._set_energy_system(energy_system)

    def __post_init__(self) -> None:
        super().__post_init__()
        for carriers, direction in ((self.inputs, PortDirection.INPUT),
                                    (self.outputs, PortDirection.OUTPUT)):
            for carrier in carriers:
                # UUID makes labels unique even for carriers with equal labels.
                self.add_port(Port(
                    label=f'{self.label}_{carrier.label}_{carrier.id}_{direction.value}',
                    vertex=self, carrier=carrier, direction=direction,
                ))

    @staticmethod
    def _normalize_carriers(value, name: str) -> tuple[Carrier, ...]:
        if isinstance(value, Carrier):
            carriers = (value,)
        else:
            if isinstance(value, (str, bytes)):
                raise TypeError(f'{name} must contain Carrier instances.')
            try:
                carriers = tuple(value)
            except TypeError as exc:
                raise TypeError(f'{name} must be a Carrier or iterable of Carriers.') from exc
        if not carriers:
            raise ValueError(f'{name} must contain at least one carrier.')
        if not all(isinstance(c, Carrier) for c in carriers):
            raise TypeError(f'{name} must contain only Carrier instances.')
        if len({c.id for c in carriers}) != len(carriers):
            raise ValueError(f'{name} must not contain duplicate carriers.')
        return carriers

    @staticmethod
    def _normalize_parameters(value, carriers, name) -> dict[UUID, NormalizedNumeric]:
        if isinstance(value, Mapping):
            if set(value) != {c.id for c in carriers}:
                raise ValueError(f'{name} must have exactly the selected carrier.id keys.')
        else:
            if len(carriers) != 1:
                raise TypeError(f'{name} must be a carrier.id mapping for multiple carriers.')
            value = {carriers[0].id: value}
        return {
            c.id: normalize_numeric_input(value[c.id], name=f'{name}[{c.id}]', allow_negative=False)
            for c in carriers
        }

    @staticmethod
    def _validate_bounds(minimum, maximum, side: str) -> None:
        for carrier_id, lower in minimum.items():
            upper = maximum[carrier_id]
            if isinstance(lower, pd.Series) and isinstance(upper, pd.Series):
                if not lower.index.equals(upper.index):
                    raise ValueError(f'{side}_min and {side}_max must have identical indices.')
            if np.any(lower > upper):
                raise ValueError(f'{side}_min must not exceed {side}_max for {carrier_id}.')

    def _parameter_mapping(self, values, name: str) -> Mapping[UUID, pd.Series]:
        return MappingProxyType({
            carrier_id: self._parameter_series(value, f'{name}[{carrier_id}]')
            for carrier_id, value in values.items()
        })

    @property
    def input_carriers(self) -> tuple[Carrier, ...]:
        return self._input_carriers

    @property
    def output_carriers(self) -> tuple[Carrier, ...]:
        return self._output_carriers

    @property
    def inputs(self) -> tuple[Carrier, ...]:
        return self.input_carriers

    @property
    def outputs(self) -> tuple[Carrier, ...]:
        return self.output_carriers

    @property
    def input_conversion(self) -> Mapping[UUID, pd.Series]:
        return self._parameter_mapping(self._input_conversion, 'input_conversion')

    @property
    def output_conversion(self) -> Mapping[UUID, pd.Series]:
        return self._parameter_mapping(self._output_conversion, 'output_conversion')

    @property
    def input_min(self) -> Mapping[UUID, pd.Series]:
        return self._parameter_mapping(self._input_min, 'input_min')

    @property
    def input_max(self) -> Mapping[UUID, pd.Series]:
        return self._parameter_mapping(self._input_max, 'input_max')

    @property
    def output_min(self) -> Mapping[UUID, pd.Series]:
        return self._parameter_mapping(self._output_min, 'output_min')

    @property
    def output_max(self) -> Mapping[UUID, pd.Series]:
        return self._parameter_mapping(self._output_max, 'output_max')
