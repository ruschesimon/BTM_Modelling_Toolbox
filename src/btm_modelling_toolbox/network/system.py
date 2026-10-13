"""Time context and ownership of an energy system's directed graph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
import pandas as pd

from ..core.entity import Entity

if TYPE_CHECKING:
    from ..components.component import Component
    from ..core.edge import Edge, BidirectionalEdge
    from ..core.vertex import Vertex
    from .bus import Bus


@dataclass(kw_only=True, init=False, eq=False)
class EnergySystem(Entity):
    """Own vertices and edges, without a location hierarchy.

    Register vertices before edges. Each object belongs to at most one system;
    constructor-time assignment to this system is accepted on registration.
    Batch additions validate every object before changing membership. Removing
    an edge disconnects its ports. Removing connected vertices requires cascade.

    time_index is a nonempty increasing RangeIndex, or None until configured.
    time_increment is the duration of one step in seconds, independent of the
    index labels. set_temporal_context uses integer Unix seconds (end excluded);
    naive datetimes are interpreted as UTC. No solver equations are generated.
    """

    _vertices: dict[UUID, Vertex] = field(default_factory=dict, init=False, repr=False)
    _edges: dict[UUID, Edge] = field(default_factory=dict, init=False, repr=False)
    _time_index: pd.RangeIndex | None = field(default=None, init=False, repr=False)
    _time_increment: _time_increment: timedelta | None = field(default=None, init=False, repr=False)

    def __init__(self, *, label: str | None = None,
                 time_index: pd.RangeIndex | None = None,
                 time_increment: timedelta | None = None) -> None:
        self._vertices = {}
        self._edges = {}
        self.time_index = time_index
        self.time_increment = time_increment
        Entity.__init__(self, label=label)

    @property
    def time_index(self) -> pd.RangeIndex | None:
        return self._time_index

    @time_index.setter
    def time_index(self, value: pd.RangeIndex | None) -> None:
        if value is not None:
            if not isinstance(value, pd.RangeIndex):
                raise TypeError('time_index must be a pandas RangeIndex or None.')
            if len(value) == 0 or value.step <= 0:
                raise ValueError('time_index must be nonempty and increasing.')
        self._time_index = value

    @property
    def time_increment(self) -> timedelta | None:
        return self._time_increment

    @time_increment.setter
    def time_increment(self, value: timedelta | None) -> None:
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, timedelta):
                raise TypeError('time_increment must be a timedelta or None.')
            if not np.isfinite(value.total_seconds()) or value <= timedelta(0):
                raise ValueError('time_increment must be finite and greater than zero.')
        self._time_increment = value

    @property
    def tau(self) -> float:
        """Duration of a time step in hours (legacy tao)."""
        if self.time_increment is None:
            raise RuntimeError('time_increment is not configured.')
        return self.time_increment / timedelta(hours=1)

    def set_temporal_context(self, start: datetime, end: datetime, step: timedelta) -> None:
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise TypeError('start and end must be datetime objects.')
        if not isinstance(step, timedelta):
            raise TypeError('step must be a timedelta.')
        seconds = step.total_seconds()
        if seconds <= 0 or not seconds.is_integer():
            raise ValueError('step must be a positive whole number of seconds.')
        timestamps = []
        for value in (start, end):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            stamp = value.timestamp()
            if not stamp.is_integer():
                raise ValueError('start and end must lie on whole seconds.')
            timestamps.append(int(stamp))
        if timestamps[1] <= timestamps[0]:
            raise ValueError('end must be later than start.')
        index = pd.RangeIndex(timestamps[0], timestamps[1], int(seconds))
        self.time_index = index
        self.time_increment = step

    @property
    def vertices(self) -> Mapping[UUID, Vertex]:
        return MappingProxyType(self._vertices)

    @property
    def edges(self) -> Mapping[UUID, Edge]:
        return MappingProxyType(self._edges)

    @property
    def components(self) -> Mapping[UUID, Component]:
        from ..components.component import Component
        return MappingProxyType({key: v for key, v in self._vertices.items() if isinstance(v, Component)})

    @property
    def buses(self) -> Mapping[UUID, Bus]:
        from .bus import Bus
        return MappingProxyType({key: v for key, v in self._vertices.items() if isinstance(v, Bus)})

    @staticmethod
    def _require_uuid(value) -> None:
        if not isinstance(value, UUID):
            raise TypeError('Element ID must be a UUID.')

    def get_element(self, element_id: UUID) -> Vertex | Edge:
        self._require_uuid(element_id)
        if element_id in self._vertices:
            return self._vertices[element_id]
        if element_id in self._edges:
            return self._edges[element_id]
        raise KeyError(f'Element {element_id} is not registered in this EnergySystem.')

    def get_vertex(self, vertex_id: UUID) -> Vertex:
        self._require_uuid(vertex_id)
        return self._vertices[vertex_id]

    def get_edge(self, edge_id: UUID) -> Edge:
        self._require_uuid(edge_id)
        return self._edges[edge_id]

    def add_vertex(self, vertex: Vertex) -> None:
        self.add_vertices((vertex,))

    def add_vertices(self, vertices: Iterable[Vertex]) -> None:
        from ..core.vertex import Vertex
        vertices = tuple(vertices)
        seen = set(self._vertices) | set(self._edges)
        for vertex in vertices:
            if not isinstance(vertex, Vertex):
                raise TypeError('vertices must contain Vertex instances.')
            if vertex.id in seen:
                raise ValueError(f'Duplicate element ID: {vertex.id}.')
            seen.add(vertex.id)
            if vertex.has_energy_system and vertex.energy_system is not self:
                raise ValueError('Vertex belongs to another energy system.')
            for port in vertex.ports.values():
                if port.vertex is not vertex:
                    raise ValueError('Every registered port must belong to its vertex.')
        for vertex in vertices:
            if not vertex.has_energy_system:
                vertex._set_energy_system(self)
            self._vertices[vertex.id] = vertex

    def add_edge(self, edge: Edge | BidirectionalEdge) -> None:
        self.add_edges((edge,))

    def add_edges(self, edges: Iterable[Edge | BidirectionalEdge]) -> None:
        from ..core.edge import Edge, BidirectionalEdge
        directed = []
        for edge in edges:
            if isinstance(edge, BidirectionalEdge):
                directed.extend(edge.edges)
            elif isinstance(edge, Edge):
                directed.append(edge)
            else:
                raise TypeError('edges must contain Edge or BidirectionalEdge instances.')
        seen = set(self._vertices) | set(self._edges)
        for edge in directed:
            if edge.id in seen:
                raise ValueError(f'Duplicate element ID: {edge.id}.')
            seen.add(edge.id)
            if edge.energy_system is not None and edge.energy_system is not self:
                raise ValueError('Edge belongs to another energy system.')
            for port in (edge.source_port, edge.sink_port):
                vertex = port.vertex
                if self._vertices.get(vertex.id) is not vertex:
                    raise ValueError('Both endpoint vertices must be registered first.')
                if not vertex.has_energy_system or vertex.energy_system is not self:
                    raise ValueError('Endpoint vertex must belong to this system.')
                if vertex.ports.get(port.id) is not port or port.edge is not edge:
                    raise ValueError('Edge must use registered, connected ports.')
            if (not edge.source_port.is_output or not edge.sink_port.is_input
                    or edge.source_vertex is edge.sink_vertex
                    or edge.source_port.carrier != edge.sink_port.carrier):
                raise ValueError('Invalid edge direction, endpoints, or carriers.')
        for edge in directed:
            edge.energy_system = self
            self._edges[edge.id] = edge

    def remove_edge(self, edge_id: UUID) -> Edge:
        edge = self.get_edge(edge_id)
        if edge.energy_system is not self:
            raise ValueError("Edge ownership has been modified externally.")
        for port in (edge.source_port, edge.sink_port):
            if port.edge is not edge:
                raise ValueError('Edge port connection has been modified externally.')
        del self._edges[edge_id]
        edge.source_port._disconnect()
        edge.sink_port._disconnect()
        edge.energy_system = None
        return edge

    def remove_vertex(self, vertex_id: UUID, *, cascade: bool = False) -> Vertex:
        vertex = self.get_vertex(vertex_id)
        if not vertex.has_energy_system or vertex.energy_system is not self:
            raise ValueError("Vertex ownership has been modified externally.")
        connected = [p.edge for p in vertex.ports.values() if p.edge is not None]
        if connected and not cascade:
            raise ValueError('Vertex has connected edges; use cascade=True to remove them.')
        for edge in connected:
            if self._edges.get(edge.id) is not edge:
                raise ValueError('Vertex has an unregistered connected edge; disconnect it first.')
            if edge.energy_system is not self:
                raise ValueError("Edge ownership has been modified externally.")
            if edge.source_port.edge is not edge or edge.sink_port.edge is not edge:
                raise ValueError('Edge port connection has been modified externally.')
        for edge in connected:
            self.remove_edge(edge.id)
        del self._vertices[vertex_id]
        vertex._remove_energy_system()
        return vertex

    def to_networkx(self):
        """Return a MultiDiGraph; requires the optional networkx package.

        UUID keys preserve duplicate labels and parallel edges. Isolated
        vertices are included. Object and label attributes accompany IDs.
        """
        import networkx as nx
        graph = nx.MultiDiGraph()
        for vertex in self._vertices.values():
            graph.add_node(vertex.id, label=vertex.label, vertex=vertex)
        for edge in self._edges.values():
            graph.add_edge(edge.source_vertex.id, edge.sink_vertex.id,
                           key=edge.id, label=edge.label, edge=edge)
        return graph

    def draw_system_graph(self, *, ax=None):
        """Draw the graph on a matplotlib axis; networkx/matplotlib are optional."""
        import networkx as nx
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()
        graph = self.to_networkx()
        nx.draw(graph, pos=nx.spring_layout(graph, seed=42), ax=ax,
                labels={key: data['label'] for key, data in graph.nodes(data=True)},
                with_labels=True)
        return ax
