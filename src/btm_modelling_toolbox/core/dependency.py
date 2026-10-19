"""Declarative relationships between directed edge flows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from uuid import UUID

import numpy as np

from .edge import Edge
from .entity import Entity
from ..network.system import EnergySystem
from ..utils.numeric import normalize_numeric_input


@dataclass(kw_only=True)
class Dependency(Entity):
    """Base for declarative dependencies; identity and label come from Entity.

    Dependencies describe constraints but do not generate or solve them.
    There is no intermediate FlowDependency class or location concept.
    """

    _energy_system: EnergySystem | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def has_energy_system(self) -> bool:
        return self._energy_system is not None

    @property
    def energy_system(self) -> EnergySystem:
        if self._energy_system is None:
            raise RuntimeError(f'Dependency {self.label!r} is not assigned to an EnergySystem.')
        return self._energy_system

    def _set_energy_system(self, energy_system: EnergySystem) -> None:
        if not isinstance(energy_system, EnergySystem):
            raise TypeError('energy_system must be an instance of EnergySystem.')
        if self.has_energy_system:
            raise ValueError('Dependency is already assigned to an EnergySystem.')
        self._energy_system = energy_system

    def _remove_energy_system(self) -> None:
        self._energy_system = None

    @property
    def referenced_edges(self) -> tuple[Edge, ...]:
        """Directed edges used by this dependency (possibly on both sides)."""
        return self._referenced_flows()

    @staticmethod
    def _normalize_flows(value: Edge | Iterable[Edge], name: str) -> tuple[Edge, ...]:
        if isinstance(value, Edge):
            flows = (value,)
        else:
            if isinstance(value, (str, bytes, Mapping)):
                raise TypeError(f'{name} must be an Edge or an iterable of Edges.')
            try:
                flows = tuple(value)
            except TypeError as exc:
                raise TypeError(f'{name} must be an Edge or an iterable of Edges.') from exc
        if not all(isinstance(flow, Edge) for flow in flows):
            raise TypeError(f'{name} must contain only Edge instances.')
        if len({flow.id for flow in flows}) != len(flows):
            raise ValueError(f'{name} must not contain duplicate edge IDs.')
        return flows

    def _referenced_flows(self) -> tuple[Edge, ...]:
        return ()

    def check_system_compatibility(self, energy_system: EnergySystem) -> bool:
        """Check that every referenced Edge is registered in the given system.

        Membership is checked by identity as well as UUID, including ownership.
        Returns False for missing/foreign edges; invalid argument types raise
        TypeError. Does not assign objects or change system membership.
        A plain Dependency has no references and is therefore compatible.
        """
        if not isinstance(energy_system, EnergySystem):
            raise TypeError('energy_system must be an instance of EnergySystem.')
        return all(energy_system.edges.get(flow.id) is flow
                   and flow.energy_system is energy_system
                   for flow in self._referenced_flows())


@dataclass(kw_only=True, init=False)
class MutuallyExclusiveFlowDependency(Dependency):
    """At most max_count of the referenced flows may be active per time step.

    This is an upper bound, not an exactly-one condition: all flows may be off.
    max_count defaults to 1; 0 disables all flows. Values above the group size
    are allowed (a redundant constraint). A future solver must define activity
    and connect binary indicators to flow bounds; this class stores no big-M
    approximation or arbitrary positive-flow tolerance.
    """

    _flows: tuple[Edge, ...] = field(init=False, repr=False)
    _max_count: int = field(init=False)

    def __init__(self, *, flows: Edge | Iterable[Edge], max_count: int = 1,
                 label: str | None = None) -> None:
        self._flows = self._normalize_flows(flows, 'flows')
        if not self._flows:
            raise ValueError('flows must contain at least one Edge.')
        if isinstance(max_count, (bool, np.bool_)) or not isinstance(max_count, Integral):
            raise TypeError('max_count must be an integer.')
        if max_count < 0:
            raise ValueError('max_count must be nonnegative.')
        self._max_count = int(max_count)
        Entity.__init__(self, label=label)

    @property
    def flows(self) -> tuple[Edge, ...]:
        return self._flows

    @property
    def max_count(self) -> int:
        return self._max_count

    def _referenced_flows(self) -> tuple[Edge, ...]:
        return self.flows


@dataclass(kw_only=True, init=False)
class LinearFlowDependency(Dependency):
    """Declare sum(a_i * flow_i) on the left equal to the right at each step.

    Each side accepts an Edge or iterable of Edges. Coefficients are complete
    UUID-to-real-scalar mappings, signed, finite, and dimensionally consistent
    with the chosen flows. This class performs no automatic unit conversion.
    Edge flow means sending-end flow; reception-side relations must account
    for transmission efficiencies explicitly in their coefficients.

    A side may be empty (with an empty coefficient mapping) to represent zero;
    both sides empty is rejected. An edge may appear on both sides, but never
    twice within one side. Zero coefficients are allowed. No offset, inequality,
    time-dependent coefficients or solver-specific constraint is introduced.
    """

    _flows_lhs: tuple[Edge, ...] = field(init=False, repr=False)
    _flows_rhs: tuple[Edge, ...] = field(init=False, repr=False)
    _coefficients_lhs: dict[UUID, Real] = field(init=False, repr=False)
    _coefficients_rhs: dict[UUID, Real] = field(init=False, repr=False)

    def __init__(self, *, flows_lhs: Edge | Iterable[Edge], flows_rhs: Edge | Iterable[Edge],
                 coefficients_lhs: Mapping[UUID, Real], coefficients_rhs: Mapping[UUID, Real],
                 label: str | None = None) -> None:
        self._flows_lhs = self._normalize_flows(flows_lhs, 'flows_lhs')
        self._flows_rhs = self._normalize_flows(flows_rhs, 'flows_rhs')
        if not self._flows_lhs and not self._flows_rhs:
            raise ValueError('At least one side must contain an Edge.')
        self._coefficients_lhs = self._normalize_coefficients(
            coefficients_lhs, self._flows_lhs, 'coefficients_lhs',
        )
        self._coefficients_rhs = self._normalize_coefficients(
            coefficients_rhs, self._flows_rhs, 'coefficients_rhs',
        )
        Entity.__init__(self, label=label)

    @staticmethod
    def _normalize_coefficients(value: Mapping[UUID, Real], flows: tuple[Edge, ...],
                                name: str) -> dict[UUID, Real]:
        if not isinstance(value, Mapping):
            raise TypeError(f'{name} must be a mapping keyed by edge UUIDs.')
        if not all(isinstance(key, UUID) for key in value):
            raise TypeError(f'{name} keys must be UUIDs.')
        if set(value) != {flow.id for flow in flows}:
            raise ValueError(f'{name} keys must exactly match the referenced edge IDs.')
        result = {}
        for key, coefficient in value.items():
            normalized = normalize_numeric_input(coefficient, name=f'{name}[{key}]', allow_series=False)
            if not np.isfinite(normalized):
                raise ValueError(f'{name} values must be finite.')
            result[key] = normalized
        return result

    @property
    def flows_lhs(self) -> tuple[Edge, ...]:
        return self._flows_lhs

    @property
    def flows_rhs(self) -> tuple[Edge, ...]:
        return self._flows_rhs

    @property
    def flows_lhs_ids(self) -> frozenset[UUID]:
        return frozenset(flow.id for flow in self.flows_lhs)

    @property
    def flows_rhs_ids(self) -> frozenset[UUID]:
        return frozenset(flow.id for flow in self.flows_rhs)

    @property
    def coefficients_lhs(self) -> Mapping[UUID, Real]:
        return MappingProxyType(self._coefficients_lhs)

    @property
    def coefficients_rhs(self) -> Mapping[UUID, Real]:
        return MappingProxyType(self._coefficients_rhs)

    def _referenced_flows(self) -> tuple[Edge, ...]:
        return self.flows_lhs + self.flows_rhs
