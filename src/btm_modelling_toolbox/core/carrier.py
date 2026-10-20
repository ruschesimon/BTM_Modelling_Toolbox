# -------------------- Public Imports --------------------
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Hashable
from enum import Enum
from math import isfinite
from numbers import Real


# -------------------- Local Imports --------------------
from .entity import Entity


# -------------------- Enum Definition --------------------

class CarrierCategory(str, Enum):
    ELECTRICITY = "electricity"
    HEAT = "heat"
    GAS = "gas"
    LIQUID_FUEL = "liquid_fuel"
    OTHER = "other"


class CurrentType(str, Enum):
    ALTERNATING_CURRENT = "alternating_current"
    DIRECT_CURRENT = "direct_current"

    AC = ALTERNATING_CURRENT
    DC = DIRECT_CURRENT


# -------------------- Class Definition --------------------

@dataclass(slots=True, kw_only=True)
class Carrier(Entity, ABC):

    """
    Abstrakte Basisklasse für Energie- und Stoffträger.

    Jede konkrete Unterklasse muss mindestens einen Namen,
    eine Kategorie, eine Standardeinheit und einen
    Kompatibilitätsschlüssel bereitstellen.
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """Maschinenlesbarer Label des Carriers."""
        raise NotImplementedError

    @property
    @abstractmethod
    def category(self) -> CarrierCategory:
        """Übergeordnete Kategorie des Carriers."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_unit(self) -> str:
        """Standardeinheit für Flüsse dieses Carriers."""
        raise NotImplementedError

    @property
    @abstractmethod
    def compatibility_key(self) -> tuple[Hashable, ...]:
        """
        Eigenschaften, die für eine direkte Verbindung
        übereinstimmen müssen.
        """
        raise NotImplementedError

    def same_type_as(self, other: object) -> bool:
        """Prüft, ob zwei Objekte denselben Carrier-Typ besitzen."""
        return type(self) is type(other)

    def is_compatible_with(self, other: object) -> bool:
        """
        Prüft, ob zwei Carrier direkt miteinander verbunden
        werden können.
        """
        if not isinstance(other, Carrier):
            return False

        return (
            type(self) is type(other)
            and self.compatibility_key == other.compatibility_key
        )


# -------------------- Class Definition --------------------

@dataclass(slots=True, kw_only=True)
class Electricity(Carrier):
    """Electrical carrier characterized by current type and nominal voltage.

    current_type defaults to AC. voltage_level is a positive finite scalar in
    volts, or None for an unspecified voltage. Compatibility requires identical
    current types and voltages, independent of UUID/label. None matches only
    None; it is not a wildcard for a known voltage. Direction remains a Port
    characteristic. Voltage conversion or AC/DC conversion requires a Converter.
    """

    current_type: CurrentType = CurrentType.ALTERNATING_CURRENT
    voltage_level: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.current_type, CurrentType):
            raise TypeError(
                "current_type must be an instance of CurrentType."
            )

        if self.voltage_level is not None:
            if isinstance(self.voltage_level, bool) or not isinstance(self.voltage_level, Real):
                raise TypeError("voltage_level must be a real scalar in volts or None.")
            if not isfinite(self.voltage_level) or self.voltage_level <= 0:
                raise ValueError("voltage_level must be finite and greater than zero.")
            self.voltage_level = float(self.voltage_level)

    @property
    def label(self) -> str:
        return "electricity"

    @property
    def category(self) -> CarrierCategory:
        return CarrierCategory.ELECTRICITY

    @property
    def default_unit(self) -> str:
        return "MW"

    @property
    def compatibility_key(self) -> tuple[Hashable, ...]:
        return (
            self.current_type,
            self.voltage_level,
        )

# -------------------- Class Definition --------------------

@dataclass(slots=True, kw_only=True)
class Heat(Carrier):
    """Thermal carrier characterized by temperature level and transfer medium.

    temperature_level is a finite scalar in degrees Celsius, strictly above
    absolute zero, or None when unspecified. medium is a nonempty text label
    (e.g. 'water', 'steam', 'air'), or None. Medium labels are stripped and
    case-folded; aliases/translations and mixture compositions are not inferred.

    Compatibility requires identical temperature levels and normalized media,
    independent of UUID/label. None matches only None, never a specified value.
    Heat transfer/temperature changes belong to components, not direct edges.
    The flow unit is MW of thermal power; no thermodynamic state is computed.
    """

    temperature_level: float | None = None
    medium: str | None = None

    def __post_init__(self) -> None:
        Entity.__post_init__(self)
        if self.temperature_level is not None:
            if isinstance(self.temperature_level, bool) or not isinstance(self.temperature_level, Real):
                raise TypeError('temperature_level must be a real scalar in degrees Celsius or None.')
            if not isfinite(self.temperature_level) or self.temperature_level <= -273.15:
                raise ValueError('temperature_level must be finite and above absolute zero (-273.15 °C).')
            self.temperature_level = float(self.temperature_level)
        if self.medium is not None:
            if not isinstance(self.medium, str):
                raise TypeError('medium must be a string or None.')
            self.medium = self.medium.strip().casefold()
            if not self.medium:
                raise ValueError('medium must not be empty.')

    @property
    def label(self) -> str:
        return 'heat'

    @property
    def category(self) -> CarrierCategory:
        return CarrierCategory.HEAT

    @property
    def default_unit(self) -> str:
        return 'MW'

    @property
    def compatibility_key(self) -> tuple[Hashable, ...]:
        return self.temperature_level, self.medium
