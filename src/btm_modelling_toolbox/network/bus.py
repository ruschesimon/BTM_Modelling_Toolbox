# -------------------- Public Imports --------------------

from __future__ import annotations
from dataclasses import dataclass


# -------------------- Local Imports --------------------

from ..core.carrier import Carrier
from ..core.port import PortDirection, Port
from ..core.vertex import Vertex


# -------------------- Class Definition --------------------

@dataclass(slots=True, kw_only=True)
class Bus(Vertex):
    """A bus represents a junction for a single energy carrier.

    A Bus connects multiple vertices that exchange the same energy carrier.
    It does not transform, generate, consume, or store energy itself.

    Parameters
    ----------
    carrier : Carrier
        Energy carrier transported through the bus.
    """

    carrier: Carrier

    def __post_init__(self) -> None:
        Vertex.__post_init__(self)

        self._validate_carrier()
        self._create_ports()



    def _validate_carrier(self) -> None:
        if not isinstance(self.carrier, Carrier):
            raise TypeError("carrier must be an instance of Carrier.")


    def _create_ports(self) -> None:
        self.add_ports((
            Port(
                label=f"{self.label}_{self.carrier.label}_input",
                vertex=self,
                carrier=self.carrier,
                direction=PortDirection.INPUT,
            ),
            Port(
                label=f"{self.label}_{self.carrier.label}_output",
                vertex=self,
                carrier=self.carrier,
                direction=PortDirection.OUTPUT,
            ),

        ))
