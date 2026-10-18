# -------------------- Public Imports --------------------

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


# -------------------- Local Imports --------------------

from .entity import Entity
from .port import Port
from ..network.system import EnergySystem
from ..utils.numeric import NumericValue, build_time_series, normalize_numeric_input


# -------------------- Type Checking Imports --------------------

if TYPE_CHECKING:
    from .carrier import Carrier
    from .vertex import Vertex


# -------------------- Class Definition --------------------

@dataclass(slots=True, kw_only=True, init=False)
class Edge(Entity):
    """Connect two ports with scalar or time-dependent variable costs.

    Costs default to zero and may be negative. Reading variable_cost requires
    an explicitly assigned EnergySystem with a pandas RangeIndex. Series
    inputs are aligned by index label; missing/duplicate labels raise errors.
    Capacity is a nonnegative sending-end limit (default infinity). Efficiency
    is a scalar or Series in [0, 1] (default 1). Received flow equals sending
    flow times efficiency; losses are proportional, with no fixed offset.
    """

    source_port: Port
    sink_port: Port
    energy_system: EnergySystem | None = field(default=None, repr=False)
    _variable_cost: NumericValue = field(init=False, repr=False)
    _capacity: NumericValue = field(init=False, repr=False)
    _efficiency: NumericValue = field(init=False, repr=False)

    def __init__(
        self,
        *,
        source_port: Port,
        sink_port: Port,
        variable_cost: NumericValue = 0.0,
        capacity: NumericValue = float("inf"),
        efficiency: NumericValue = 1.0,
        energy_system: EnergySystem | None = None,
        label: str | None = None,
    ) -> None:
        self.source_port = source_port
        self.sink_port = sink_port
        self.energy_system = energy_system
        self._variable_cost = normalize_numeric_input(
            variable_cost, name="variable_cost",
        )
        self._capacity = normalize_numeric_input(
            capacity, name="capacity", allow_negative=False,
        )
        self._efficiency = normalize_numeric_input(
            efficiency, name="efficiency", allow_negative=False,
        )
        if np.any(self._efficiency > 1):
            raise ValueError("efficiency must be between zero and one.")
        Entity.__init__(self, label=label)

    @property
    def variable_cost(self) -> pd.Series:
        return self._parameter_series(self._variable_cost, "variable_cost")

    @property
    def capacity(self) -> pd.Series:
        """Maximum sending-end flow; defaults to an unlimited capacity."""
        return self._parameter_series(self._capacity, "capacity")

    @property
    def efficiency(self) -> pd.Series:
        """Received/sent flow ratio in [0, 1]; defaults to lossless transfer."""
        return self._parameter_series(self._efficiency, "efficiency")

    def __post_init__(self) -> None:
        Entity.__post_init__(self)

        self._validate_ports()
        self._connect_ports()

    def _validate_ports(self) -> None:
        if not isinstance(self.source_port, Port):
            raise TypeError("source_port must be an instance of Port.")

        if not isinstance(self.sink_port, Port):
            raise TypeError("sink_port must be an instance of Port.")

        if not self.source_port.is_output:
            raise ValueError(
                f"Source port {self.source_port.label!r} must be an output port."
            )

        if not self.sink_port.is_input:
            raise ValueError(
                f"Sink port {self.sink_port.label!r} must be an input port."
            )

        if self.source_port.carrier != self.sink_port.carrier:
            raise ValueError(
                "Source and sink port must use the same carrier."
            )

        if self.source_port.is_connected:
            raise ValueError(
                f"Source port {self.source_port.label!r} is already connected."
            )

        if self.sink_port.is_connected:
            raise ValueError(
                f"Sink port {self.sink_port.label!r} is already connected."
            )

        if self.source_port.vertex is self.sink_port.vertex:
            raise ValueError(
                "An edge cannot connect ports of the same vertex."
            )

    def _connect_ports(self) -> None:
        self.source_port._connect(self)
        self.sink_port._connect(self)

    def _parameter_series(self, value: NumericValue, name: str) -> pd.Series:
        system = self.energy_system
        if system is None:
            raise RuntimeError("Edge is not assigned to an EnergySystem.")
        if not isinstance(system, EnergySystem):
            raise TypeError("energy_system must be an instance of EnergySystem.")
        if not isinstance(system.time_index, pd.RangeIndex):
            raise TypeError("energy_system.time_index must be a pandas RangeIndex.")
        return build_time_series(value, name, system.time_index)

    @property
    def source_vertex(self) -> Vertex:
        return self.source_port.vertex

    @property
    def sink_vertex(self) -> Vertex:
        return self.sink_port.vertex

    @property
    def carrier(self) -> Carrier:
        return self.source_port.carrier


# -------------------- Class Definition --------------------

@dataclass(slots=True, kw_only=True)
class BidirectionalEdge(Entity):
    forward_source_port: Port
    forward_sink_port: Port
    backward_source_port: Port
    backward_sink_port: Port

    forward_edge: Edge = field(
        init=False,
        repr=True,
        compare=True,
    )
    backward_edge: Edge = field(
        init=False,
        repr=True,
        compare=True,
    )

    def __post_init__(self) -> None:
        Entity().__post_init__()

        self._validate_ports()

        self.forward_edge = Edge(
            source_port=self.forward_source_port,
            sink_port=self.forward_sink_port,
        )

        self.backward_edge = Edge(
            source_port=self.backward_source_port,
            sink_port=self.backward_sink_port,
        )

    def _validate_ports(self) -> None:
        ports = (
            self.forward_source_port,
            self.forward_sink_port,
            self.backward_source_port,
            self.backward_sink_port,
        )

        if not all(isinstance(port, Port) for port in ports):
            raise TypeError("All ports must be instances of Port.")

        if self.forward_source_port.vertex is not self.backward_sink_port.vertex:
            raise ValueError(
                "forward_source_port and backward_sink_port "
                "must belong to the same vertex."
            )

        if self.forward_sink_port.vertex is not self.backward_source_port.vertex:
            raise ValueError(
                "forward_sink_port and backward_source_port "
                "must belong to the same vertex."
            )

        if self.forward_source_port.carrier != self.backward_sink_port.carrier:
            raise ValueError(
                "Ports of the first vertex must use the same carrier."
            )

        if self.forward_sink_port.carrier != self.backward_source_port.carrier:
            raise ValueError(
                "Ports of the second vertex must use the same carrier."
            )

        if self.forward_source_port.carrier != self.forward_sink_port.carrier:
            raise ValueError(
                "All ports of a bidirectional edge must use the same carrier."
            )

    @property
    def edges(self) -> tuple[Edge, Edge]:
        return self.forward_edge, self.backward_edge