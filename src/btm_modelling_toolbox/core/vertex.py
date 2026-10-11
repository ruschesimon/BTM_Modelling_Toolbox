# -------------------- Public Imports --------------------

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping
from unicodedata import name
from uuid import UUID


# -------------------- Local Imports --------------------

from .entity import Entity
from .port import Port


# -------------------- Class Definition --------------------

@dataclass(kw_only=True)
class Vertex(Entity, ABC):
    """
    Basisklasse aller Knoten des Energiesystemgraphen.

    Sowohl Komponenten als auch Busse sind Vertices.
    Jeder Vertex besitzt Ports, über die Connections
    mit anderen Vertices hergestellt werden können.
    """

    _ports: dict[UUID, Port] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    _energy_system: EnergySystem | None = field(
        default=None,
        init=False,
        repr=False,
    )

    

    def __post_init__(self) -> None:
        super().__post_init__()
        

    @property
    def energy_system(self) -> EnergySystem:
        if self._energy_system is None:
            raise RuntimeError(f"Vertex {self.name!r} is not assigned to an EnergySystem.")

        return self._energy_system

    @property
    def ports(self) -> Mapping[UUID, Port]:
        """Schreibgeschützte Sicht auf die Ports."""
        return MappingProxyType(self._ports)

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return tuple(
            port
            for port in self._ports.values()
            if port.direction.allows_input
        )

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return tuple(
            port
            for port in self._ports.values()
            if port.direction.allows_output
        )

    def add_port(
            self, 
            port: Port,
        ) -> None:
            """Fügt dem Vertex einen Port hinzu."""
            if not isinstance(port, Port):
                raise TypeError("port must be an instance of Port.")
    
            if port.id in self._ports:
                raise ValueError(f"Port with ID {port.id} already exists on component {self.label!r}.")
    
            if self.has_port_label(port.label):
                raise ValueError(f"Component {self.label!r} already has a port named {port.label!r}.")
    
            self._ports[port.id] = port


    def add_ports(
        self,
        ports: Iterable[Port],
    ) -> None:
        """
        Fügt mehrere Ports atomar hinzu.

        Zunächst werden alle Ports geprüft. Erst danach
        wird das interne Dictionary verändert.
        """
        ports = tuple(ports)

        incoming_ids = [port.id for port in ports]
        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("The provided ports contain duplicate UUIDs.")

        incoming_names = [port.name for port in ports]
        if len(incoming_names) != len(set(incoming_names)):
            raise ValueError("The provided ports contain duplicate names.")

        for port in ports:
            self.add_port(port)


    def remove_port(
        self,
        port_id: UUID,
    ) -> Port:
        try:
            return self._ports.pop(port_id)
        except KeyError as exc:
            raise KeyError(f"Vertex {self.label!r} has no port with ID {port_id}.") from exc
        

    def get_port(
        self,
        port_id: UUID,
    ) -> Port:
        if not isinstance(port_id, UUID):
            raise TypeError("port_id must be an instance of UUID.")

        try:
            return self._ports[port_id]
        except KeyError as exc:
            raise KeyError(f"Vertex {self.label!r} has no port with ID {port_id}.") from exc


    def get_port_by_label(
        self,
        label: str,
    ) -> Port:
        if not isinstance(label, str):
            raise TypeError("label must be a string.")

        for port in self._ports.values():
            if port.label == label:
                return port

        raise KeyError(f"Vertex {self.label!r} has no port named {label!r}.")


    def has_port(
        self,
        port_id: UUID,
    ) -> bool:
        return port_id in self._ports


    def has_port_label(
        self,
        label: str,
    ) -> bool:
        return any(port.label == label for port in self._ports.values())


    def __len__(self) -> int:
        return len(self._ports)

    def __contains__(self, port: object) -> bool:
        return (
            isinstance(port, Port)
            and port.id in self._ports
        )

    @property
    def has_energy_system(self) -> bool:
        return self._energy_system is not None


    def _set_energy_system(self, energy_system: EnergySystem) -> None:
        if not self.has_energy_system:
            raise ValueError(f"Vertex {self.name!r} is already assigned to an EnergySystem.")

        self._energy_system = energy_system


    def _remove_energy_system(self) -> None:
        self._energy_system = None