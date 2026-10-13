import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd

from btm_modelling_toolbox import (BidirectionalEdge, Bus, Edge, Electricity,
                                   EnergySystem, Sink, Source, Storage, Vertex)


class EnergySystemTests(unittest.TestCase):
    def setUp(self):
        self.system = EnergySystem(time_index=pd.RangeIndex(3))
        self.carrier = Electricity()
        self.source = Source(carrier=self.carrier, variable_cost=2)
        self.sink = Sink(carrier=self.carrier)

    def edge(self):
        return Edge(source_port=self.source.output_ports[0],
                    sink_port=self.sink.input_ports[0], variable_cost=3)

    def test_registration_views_and_cost_context(self):
        bus = Bus(carrier=self.carrier)
        self.system.add_vertices([self.source, self.sink, bus])
        edge = self.edge()
        self.system.add_edge(edge)
        self.assertIs(self.source.energy_system, self.system)
        self.assertIs(edge.energy_system, self.system)
        self.assertEqual(self.source.variable_cost.tolist(), [2.] * 3)
        self.assertEqual(edge.variable_cost.tolist(), [3.] * 3)
        self.assertEqual(set(self.system.components), {self.source.id, self.sink.id})
        self.assertEqual(set(self.system.buses), {bus.id})
        self.assertIs(self.system.get_element(edge.id), edge)
        self.assertIs(self.system.get_vertex(bus.id), bus)
        self.assertIs(self.system.get_edge(edge.id), edge)
        with self.assertRaises(TypeError):
            self.system.vertices[uuid4()] = bus
        with self.assertRaises(KeyError):
            self.system.get_element(uuid4())
        with self.assertRaises(TypeError):
            self.system.get_element('bad')

    def test_preassigned_vertex_and_foreign_system(self):
        vertex = Source(carrier=self.carrier, energy_system=self.system)
        self.system.add_vertex(vertex)
        with self.assertRaises(ValueError):
            EnergySystem().add_vertex(vertex)
        foreign = Sink(carrier=self.carrier, energy_system=EnergySystem())
        with self.assertRaises(ValueError):
            self.system.add_vertices([self.source, foreign])
        self.assertFalse(self.source.has_energy_system)
        self.assertNotIn(self.source.id, self.system.vertices)

    def test_vertex_batch_is_atomic(self):
        for vertices, error in [([self.source, self.source], ValueError),
                                ([self.source, object()], TypeError)]:
            with self.assertRaises(error):
                self.system.add_vertices(vertices)
            self.assertFalse(self.source.has_energy_system)
            self.assertEqual(len(self.system.vertices), 0)
        self.system.add_vertex(self.source)
        with self.assertRaises(ValueError):
            self.system.add_vertex(self.source)

    def test_edge_registration_requires_registered_endpoints(self):
        edge = self.edge()
        with self.assertRaises(ValueError):
            self.system.add_edge(edge)
        self.assertIsNone(edge.energy_system)
        self.system.add_vertices([self.source, self.sink])
        with self.assertRaises(TypeError):
            self.system.add_edges([edge, object()])
        self.assertIsNone(edge.energy_system)
        with self.assertRaises(ValueError):
            self.system.add_edges([edge, edge])
        self.assertEqual(len(self.system.edges), 0)
        edge.energy_system = EnergySystem()
        with self.assertRaises(ValueError):
            self.system.add_edge(edge)
        edge.energy_system = None
        self.source.remove_port(edge.source_port.id)
        with self.assertRaises(ValueError):
            self.system.add_edge(edge)

    def test_remove_edge_and_reconnect(self):
        self.system.add_vertices([self.source, self.sink])
        edge = self.edge()
        self.system.add_edge(edge)
        self.assertIs(self.system.remove_edge(edge.id), edge)
        self.assertIsNone(edge.energy_system)
        self.assertFalse(edge.source_port.is_connected)
        self.assertFalse(edge.sink_port.is_connected)
        replacement = self.edge()
        self.system.add_edge(replacement)
        self.assertIs(self.system.get_edge(replacement.id), replacement)

    def test_vertex_removal_and_cascade(self):
        self.system.add_vertices([self.source, self.sink])
        edge = self.edge()
        self.system.add_edge(edge)
        with self.assertRaises(ValueError):
            self.system.remove_vertex(self.source.id)
        self.assertIs(self.system.remove_vertex(self.source.id, cascade=True), self.source)
        self.assertFalse(self.source.has_energy_system)
        self.assertEqual(len(self.system.edges), 0)
        self.assertFalse(self.sink.input_ports[0].is_connected)
        self.assertIs(self.sink.energy_system, self.system)

    def test_unregistered_connection_blocks_removal(self):
        self.system.add_vertices([self.source, self.sink])
        self.edge()
        with self.assertRaises(ValueError):
            self.system.remove_vertex(self.source.id, cascade=True)
        self.assertIs(self.source.energy_system, self.system)

    def test_bidirectional_registration_and_removal(self):
        a, b = (Storage(carrier=self.carrier, energy_capacity=10) for _ in range(2))
        pair = BidirectionalEdge(forward_source_port=a.output_ports[0],
                                 forward_sink_port=b.input_ports[0],
                                 backward_source_port=b.output_ports[0],
                                 backward_sink_port=a.input_ports[0])
        self.system.add_vertices([a, b])
        self.system.add_edge(pair)
        self.assertEqual(len(self.system.edges), 2)
        self.assertTrue(all(e.energy_system is self.system for e in pair.edges))
        self.system.remove_vertex(a.id, cascade=True)
        self.assertEqual(len(self.system.edges), 0)
        self.assertTrue(all(not p.is_connected for p in b.ports.values()))

    def test_temporal_context(self):
        system = EnergySystem()
        self.assertIsNone(system.time_index)
        with self.assertRaises(RuntimeError):
            _ = system.tau
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        system.set_temporal_context(start, start + timedelta(hours=2), timedelta(minutes=30))
        self.assertEqual(len(system.time_index), 4)
        self.assertEqual(system.time_index[0], int(start.timestamp()))
        self.assertEqual(system.time_increment, 1800)
        self.assertEqual(system.tau, 0.5)
        naive = EnergySystem()
        naive.set_temporal_context(start.replace(tzinfo=None),
                                   (start + timedelta(hours=2)).replace(tzinfo=None),
                                   timedelta(minutes=30))
        self.assertTrue(naive.time_index.equals(system.time_index))

    def test_invalid_time_context_is_atomic(self):
        start = datetime(2026, 1, 1)
        self.system.time_increment = 3600
        old_index = self.system.time_index
        for args, error in [((None, start, timedelta(hours=1)), TypeError),
                            ((start, start, timedelta(hours=1)), ValueError),
                            ((start, start + timedelta(hours=1), 1), TypeError),
                            ((start, start + timedelta(hours=1), timedelta(0)), ValueError),
                            ((start, start + timedelta(hours=1), timedelta(milliseconds=500)), ValueError)]:
            with self.assertRaises(error):
                self.system.set_temporal_context(*args)
            self.assertIs(self.system.time_index, old_index)
            self.assertEqual(self.system.time_increment, 3600)
        for index in [pd.RangeIndex(0), pd.RangeIndex(3, 0, -1)]:
            with self.assertRaises(ValueError):
                EnergySystem(time_index=index)
        with self.assertRaises(TypeError):
            EnergySystem(time_index=pd.Index([0, 1]))
        for value in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                EnergySystem(time_increment=value)
        with self.assertRaises(TypeError):
            EnergySystem(time_increment=True)

    def test_graph_export(self):
        try:
            import networkx
        except ImportError:
            self.skipTest('Optional networkx is not installed')
        isolated = Vertex(label='isolated')
        self.source.label = self.sink.label = 'same label'
        self.system.add_vertices([self.source, self.sink, isolated])
        edge = self.edge()
        self.system.add_edge(edge)
        graph = self.system.to_networkx()
        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(len(graph.edges), 1)
        self.assertIs(graph.edges[self.source.id, self.sink.id, edge.id]['edge'], edge)


    def test_external_mutation_blocks_removal_without_partial_changes(self):
        self.system.add_vertices([self.source, self.sink])
        edge = self.edge()
        self.system.add_edge(edge)
        edge.energy_system = EnergySystem()
        with self.assertRaises(ValueError):
            self.system.remove_vertex(self.source.id, cascade=True)
        self.assertIn(self.source.id, self.system.vertices)
        self.assertIn(edge.id, self.system.edges)
        edge.energy_system = self.system
        edge.sink_port._disconnect()
        with self.assertRaises(ValueError):
            self.system.remove_edge(edge.id)
        self.assertIn(edge.id, self.system.edges)
        self.assertIs(edge.source_port.edge, edge)

    def test_graph_drawing(self):
        try:
            import matplotlib
            import networkx
        except ImportError:
            self.skipTest('Optional visualization packages are not installed')
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.system.add_vertices([self.source, self.sink])
        self.system.add_edge(self.edge())
        ax = self.system.draw_system_graph()
        ax.figure.canvas.draw()
        self.assertEqual(len(ax.texts), 2)
        self.assertTrue(ax.collections)
        plt.close(ax.figure)


if __name__ == '__main__':
    unittest.main()
