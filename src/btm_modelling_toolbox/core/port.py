# -------------------- Public Imports --------------------

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING


# -------------------- Local Imports --------------------

from .carrier import Carrier
from .entity import Entity


# -------------------- Type Checking Imports --------------------

if TYPE_CHECKING:
    from .edge import Edge
    from .vertex import Vertex


# -------------------- Enum Definition --------------------

class PortDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"


# -------------------- Class Definition --------------------

@dataclass(slots=True, kw_only=True)
class Port(Entity):
    vertex: Vertex
    carrier: Carrier
    direction: PortDirection

    edge: Edge | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        Entity().__post_init__()

        if not isinstance(self.carrier, Carrier):
            raise TypeError("carrier must be an instance of Carrier.")

        if not isinstance(self.direction, PortDirection):
            raise TypeError(
                "direction must be an instance of PortDirection."
            )

    @property
    def is_connected(self) -> bool:
        return self.edge is not None

    @property
    def is_input(self) -> bool:
        return self.direction is PortDirection.INPUT

    @property
    def is_output(self) -> bool:
        return self.direction is PortDirection.OUTPUT

    def _connect(self, edge: Edge) -> None:
        if self.edge is not None:
            raise ValueError(
                f"Port {self.label!r} is already connected."
            )

        self.edge = edge

    def _disconnect(self) -> None:
        self.edge = None