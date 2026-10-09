# -------------------- Public Imports --------------------

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import pandas as pd


# -------------------- Local Imports --------------------

from .component import Component
from ..core.carrier import Carrier
from ..core.port import Port, FlowDirection
from ..utils.numeric import NumericValue, normalize_numeric_input




@dataclass(kw_only=True)
class Sink(Component):
    """A component that consumes an energy carrier.

    A Sink represents a terminal consumer of an energy carrier,
    such as an electrical load, heat demand, or gas demand.

    Parameters
    ----------
    carrier : Carrier
        Energy carrier consumed by the sink.

    min_capacity : Real | pd.Series | None
        Minimum consumption capacity.

    max_capacity : Real | pd.Series | None
        Maximum consumption capacity.
    """

    carrier: Carrier

    min_capacity: NumericValue = 0.0
    max_capacity: NumericValue = float('inf')


    def __post_init__(self) -> None:
        super().__post_init__()

        self._validate_carrier()
        self._validate_capacity_definition()
        self._create_port()


    def _validate_carrier(self) -> None:
        if not isinstance(self.carrier, Carrier):
            raise TypeError("carrier must be an instance of Carrier.")


    def _validate_capacities(self) -> None:
        """ Validate minimum and maximum capacities."""

        self.min_capacity = normalize_numeric_input(
            self.min_capacity,
            name="min_capacity",
            allow_negative=False,
        )
        self.max_capacity = normalize_numeric_input(
            self.max_capacity,
            name="max_capacity",
            allow_negative=False,
        )

        if isinstance(self.min_capacity, Real) and isinstance(self.max_capacity, Real):
            if self.max_capacity < self.min_capacity:
                raise ValueError(
                    f"Sink {self.name!r}: max_capacity must be greater than "
                    "or equal to min_capacity."
                )
            return

        if isinstance(self.min_capacity, pd.Series) and isinstance(self.max_capacity, pd.Series):
            if not self.min_capacity.index.equals(self.max_capacity.index):
                raise ValueError(
                    f"Sink {self.name!r}: min_capacity and max_capacity "
                    "must have identical indices."
                )
        if isinstance(self.min_capacity, pd.Series) or isinstance(self.max_capacity, pd.Series):
            if (self.max_capacity < self.min_capacity).any():
                raise ValueError(
                    f"Sink {self.name!r}: max_capacity must be greater than "
                    "or equal to min_capacity at every time step."
                )


    def _create_port(self) -> None:
        self.add_port(
            Port(
                label=f"{self.label}_{self.carrier.label}_input",
                vertex=self,
                carrier=self.carrier,
                direction=FlowDirection.INPUT,
            )
        )



    def _validate_capacity_definition(self) -> None:
        """Validate the capacity specification of the sink."""

        if self.fixed_capacity is not None:
            if self.min_capacity is not None or self.max_capacity is not None:
                raise ValueError(
                    f"Sink {self.name!r}: fixed_capacity cannot be combined "
                    "with min_capacity or max_capacity."
                )

            self._validate_capacity_value(
                self.fixed_capacity,
                "fixed_capacity",
            )

            return

        if self.min_capacity is not None:
            self._validate_capacity_value(
                self.min_capacity,
                "min_capacity",
            )

        if self.max_capacity is not None:
            self._validate_capacity_value(
                self.max_capacity,
                "max_capacity",
            )

        if (
            isinstance(self.min_capacity, Real)
            and isinstance(self.max_capacity, Real)
            and self.min_capacity > self.max_capacity
        ):
            raise ValueError(
                f"Sink {self.name!r}: min_capacity must not be greater "
                "than max_capacity."
            )
