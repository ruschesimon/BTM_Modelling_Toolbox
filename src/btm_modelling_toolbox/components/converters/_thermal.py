"""Shared validation and reference curves for the thermal converters."""

from numbers import Integral

import numpy as np
import pandas as pd

from ...core.carrier import Carrier, CarrierCategory
from ...utils.numeric import NumericValue, NormalizedNumeric, normalize_numeric_input


def numeric(value: NumericValue, name: str, *, scalar: bool = False,
            nonnegative: bool = False, finite: bool = True) -> NormalizedNumeric:
    result = normalize_numeric_input(value, name=name, allow_negative=not nonnegative,
                                     allow_series=not scalar)
    if finite and not np.isfinite(result).all():
        raise ValueError(f'{name} must contain only finite values.')
    if isinstance(result, pd.Series) and result.index.has_duplicates:
        raise ValueError(f'{name} must not contain duplicate index labels.')
    return result


def matching_indices(*values: NormalizedNumeric) -> None:
    series = [v for v in values if isinstance(v, pd.Series)]
    if series and any(not v.index.equals(series[0].index) for v in series[1:]):
        raise ValueError('Time-dependent parameters must have identical indices.')


def carrier(value: Carrier, name: str, categories: tuple[CarrierCategory, ...]) -> Carrier:
    if not isinstance(value, Carrier):
        raise TypeError(f'{name} must be a Carrier instance.')
    if value.category not in categories:
        raise ValueError(f'{name} must have a category in {categories}.')
    return value


def efficiency(value: NumericValue, name: str) -> NormalizedNumeric:
    result = numeric(value, name, scalar=True, nonnegative=True)
    if not 0 < result <= 1:
        raise ValueError(f'{name} must be in (0, 1].')
    return result


def affine_fuel_curve(lower: NumericValue, upper: NumericValue,
                      eta_min: float, eta_max: float) -> tuple[NormalizedNumeric, NormalizedNumeric]:
    """Return A, B for fuel=A+B*power through both reference load points."""
    matching_indices(lower, upper)
    if np.any(lower <= 0) or np.any(upper <= lower):
        raise ValueError('Reference capacities must satisfy 0 < power_min_capacity < power_max_capacity.')
    # Algebraically identical to the legacy reciprocal expressions, but avoids
    # reciprocal powers and the associated unnecessary divisions.
    fuel_min, fuel_max = lower / eta_min, upper / eta_max
    slope = (fuel_max - fuel_min) / (upper - lower)
    intercept = fuel_min - slope * lower
    if not np.isfinite(intercept).all() or not np.isfinite(slope).all():
        raise ValueError('Reference capacities produce non-finite fuel coefficients.')
    return intercept, slope


def operation_parameters(component, *, min_gradient=-float('inf'), max_gradient=float('inf'),
                         min_down_time=0, min_up_time=0, initial_operation=False) -> None:
    """Retain legacy operating metadata; no unit-commitment solver is implied."""
    lower = numeric(min_gradient, 'min_gradient', scalar=True, finite=False)
    upper = numeric(max_gradient, 'max_gradient', scalar=True, finite=False)
    if lower > upper:
        raise ValueError('min_gradient must not exceed max_gradient.')
    component.min_gradient, component.max_gradient = lower, upper
    for name, value in (('min_down_time', min_down_time), ('min_up_time', min_up_time)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f'{name} must be an integer.')
        if value < 0:
            raise ValueError(f'{name} must be nonnegative.')
        setattr(component, name, int(value))
    if not isinstance(initial_operation, bool):
        raise TypeError('initial_operation must be a bool.')
    component.initial_operation = initial_operation


def kelvin(value: NumericValue, unit: str, name: str) -> NormalizedNumeric:
    value = numeric(value, name)
    if not isinstance(unit, str):
        raise TypeError('temperature_unit must be a string.')
    key = unit.lower()
    if key in ('kelvin', 'k'):
        result = value
    elif key in ('celsius', '°c', 'c'):
        result = value + 273.15
    elif key in ('fahrenheit', '°f', 'f'):
        result = (value - 32) * (5 / 9) + 273.15
    else:
        raise ValueError('temperature_unit must be Kelvin, Celsius, or Fahrenheit.')
    if np.any(result <= 0) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be above absolute zero and finite.')
    return result
