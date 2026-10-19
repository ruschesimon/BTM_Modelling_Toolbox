import unittest
from uuid import uuid4

import numpy as np
import pandas as pd

from btm_modelling_toolbox import (Dependency, Edge, Electricity, EnergySystem,
                                   LinearFlowDependency, MutuallyExclusiveFlowDependency,
                                   Sink, Source, Storage, BidirectionalEdge)


class DependencyTestCase(unittest.TestCase):
    def setUp(self):
        self.system = EnergySystem(time_index=pd.RangeIndex(3))
        self.edges = []
        for _ in range(3):
            carrier = Electricity()
            source, sink = Source(carrier=carrier), Sink(carrier=carrier)
            self.system.add_vertices([source, sink])
            edge = Edge(source_port=source.output_ports[0], sink_port=sink.input_ports[0])
            self.edges.append(edge)
            self.system.add_edge(edge)
        self.a, self.b, self.c = self.edges

    def linear(self, **kwargs):
        params = dict(flows_lhs=[self.a, self.b], flows_rhs=self.c,
                      coefficients_lhs={self.a.id: 1., self.b.id: -2.},
                      coefficients_rhs={self.c.id: .5})
        params.update(kwargs)
        return LinearFlowDependency(**params)


class DependencyTests(DependencyTestCase):
    def test_identity_label_and_direct_inheritance(self):
        base = Dependency()
        self.assertTrue(base.label.startswith('Dependency '))
        self.assertNotEqual(base.id, Dependency().id)
        self.assertEqual(Dependency(label='Balance').label, 'Balance')
        self.assertIs(MutuallyExclusiveFlowDependency.__bases__[0], Dependency)
        self.assertIs(LinearFlowDependency.__bases__[0], Dependency)
        with self.assertRaises(TypeError):
            Dependency(label=1)
        self.assertTrue(base.check_system_compatibility(self.system))
        with self.assertRaises(TypeError):
            base.check_system_compatibility(object())

    def test_compatibility_detects_missing_foreign_and_stale_ownership(self):
        for dependency in [self.linear(), MutuallyExclusiveFlowDependency(flows=self.edges)]:
            self.assertTrue(dependency.check_system_compatibility(self.system))
            self.assertFalse(dependency.check_system_compatibility(EnergySystem()))
            self.a.energy_system = EnergySystem()
            self.assertFalse(dependency.check_system_compatibility(self.system))
            self.a.energy_system = self.system
        self.system.remove_edge(self.a.id)
        self.assertFalse(self.linear().check_system_compatibility(self.system))
        self.assertFalse(MutuallyExclusiveFlowDependency(flows=self.a).check_system_compatibility(self.system))

    def test_matching_uuid_alone_does_not_establish_membership(self):
        foreign_source, foreign_sink = Source(carrier=self.a.carrier), Sink(carrier=self.a.carrier)
        foreign = Edge(source_port=foreign_source.output_ports[0], sink_port=foreign_sink.input_ports[0])
        foreign.id = self.a.id
        foreign.energy_system = self.system
        self.assertFalse(MutuallyExclusiveFlowDependency(flows=foreign).check_system_compatibility(self.system))


class MutuallyExclusiveTests(DependencyTestCase):
    def test_defaults_single_flow_generator_and_limits(self):
        dependency = MutuallyExclusiveFlowDependency(flows=(flow for flow in self.edges))
        self.assertEqual(dependency.flows, tuple(self.edges))
        self.assertEqual(dependency.max_count, 1)
        for count in [0, 2, 10, np.int64(2)]:
            model = MutuallyExclusiveFlowDependency(flows=self.a, max_count=count)
            self.assertEqual(model.flows, (self.a,))
            self.assertEqual(model.max_count, count)
        with self.assertRaises(AttributeError):
            dependency.max_count = 2

    def test_invalid_flows_and_counts(self):
        for flows in [None, 'edge', [object()], {self.a.id: self.a}]:
            with self.assertRaises(TypeError):
                MutuallyExclusiveFlowDependency(flows=flows)
        for flows in [[], [self.a, self.a]]:
            with self.assertRaises(ValueError):
                MutuallyExclusiveFlowDependency(flows=flows)
        for count in [True, np.bool_(False), 1., '1']:
            with self.assertRaises(TypeError):
                MutuallyExclusiveFlowDependency(flows=self.a, max_count=count)
        with self.assertRaises(ValueError):
            MutuallyExclusiveFlowDependency(flows=self.a, max_count=-1)


class LinearDependencyTests(DependencyTestCase):
    def test_reference_weighted_equation_and_uuid_access(self):
        model = self.linear()
        self.assertEqual(model.flows_lhs, (self.a, self.b))
        self.assertEqual(model.flows_rhs, (self.c,))
        self.assertEqual(model.flows_lhs_ids, frozenset([self.a.id, self.b.id]))
        self.assertEqual(model.flows_rhs_ids, frozenset([self.c.id]))
        # Example of the legacy equation: 10 - 2*3 == 0.5*8.
        values = {self.a.id: 10., self.b.id: 3., self.c.id: 8.}
        lhs = sum(weight*values[key] for key, weight in model.coefficients_lhs.items())
        rhs = sum(weight*values[key] for key, weight in model.coefficients_rhs.items())
        self.assertEqual(lhs, rhs)

    def test_generator_and_read_only_defensive_copy(self):
        coefficients = {self.a.id: 1., self.b.id: -2.}
        model = self.linear(flows_lhs=(edge for edge in [self.a, self.b]), coefficients_lhs=coefficients)
        coefficients[self.a.id] = 999
        self.assertEqual(model.coefficients_lhs[self.a.id], 1.)
        with self.assertRaises(TypeError):
            model.coefficients_lhs[self.a.id] = 2
        with self.assertRaises(AttributeError):
            model.flows_lhs = ()

    def test_empty_side_zero_coefficient_and_shared_edge(self):
        model = self.linear(flows_rhs=[], coefficients_rhs={})
        self.assertEqual(model.flows_rhs, ())
        self.assertEqual(dict(model.coefficients_rhs), {})
        model = self.linear(flows_lhs=self.a, flows_rhs=self.a,
                            coefficients_lhs={self.a.id: 0}, coefficients_rhs={self.a.id: 0})
        self.assertTrue(model.check_system_compatibility(self.system))
        with self.assertRaises(ValueError):
            self.linear(flows_lhs=[], flows_rhs=[], coefficients_lhs={}, coefficients_rhs={})

    def test_invalid_mappings_and_coefficient_values(self):
        for params, error in [
            (dict(coefficients_lhs={}), ValueError),
            (dict(coefficients_rhs={uuid4(): 1}), ValueError),
            (dict(coefficients_rhs={str(self.c.id): 1}), TypeError),
            (dict(coefficients_rhs=[1]), TypeError),
            (dict(flows_lhs=[self.a, self.a]), ValueError),
            (dict(flows_rhs=[object()]), TypeError),
        ]:
            with self.subTest(params=params), self.assertRaises(error):
                self.linear(**params)
        for value in [True, 'bad', 1j, pd.Series([1]), float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                self.linear(coefficients_rhs={self.c.id: value})


class DependencyRegistrationTests(DependencyTestCase):
    def test_register_retrieve_remove_and_read_only_live_view(self):
        dependency = self.linear()
        view = self.system.dependencies
        self.assertFalse(dependency.has_energy_system)
        with self.assertRaises(RuntimeError):
            _ = dependency.energy_system
        self.system.add_dependency(dependency)
        self.assertIs(view[dependency.id], dependency)
        self.assertIs(self.system.get_dependency(dependency.id), dependency)
        self.assertIs(self.system.get_element(dependency.id), dependency)
        self.assertIs(dependency.energy_system, self.system)
        with self.assertRaises(TypeError):
            view[uuid4()] = Dependency()
        with self.assertRaises(TypeError):
            self.system.get_dependency('bad')
        with self.assertRaises(KeyError):
            self.system.remove_dependency(uuid4())
        self.assertIs(self.system.remove_dependency(dependency.id), dependency)
        self.assertNotIn(dependency.id, view)
        self.assertFalse(dependency.has_energy_system)
        self.assertEqual(len(self.system.edges), 3)
        self.system.add_dependency(dependency)
        self.assertIs(dependency.energy_system, self.system)

    def test_batch_validation_and_duplicate_ids_are_atomic(self):
        valid = MutuallyExclusiveFlowDependency(flows=self.a)
        invalid = MutuallyExclusiveFlowDependency(flows=self.c)
        self.system.remove_edge(self.c.id)
        for values, error in [([valid, invalid], ValueError), ([valid, valid], ValueError),
                              ([valid, object()], TypeError)]:
            with self.subTest(values=values), self.assertRaises(error):
                self.system.add_dependencies(iter(values))
            self.assertEqual(len(self.system.dependencies), 0)
            self.assertFalse(valid.has_energy_system)
        self.system.add_dependency(valid)
        with self.assertRaises(ValueError):
            self.system.add_dependency(valid)

    def test_cross_system_ownership_and_transfer_after_removal(self):
        dependency = Dependency()
        other = EnergySystem()
        self.system.add_dependency(dependency)
        with self.assertRaises(ValueError):
            other.add_dependency(dependency)
        self.assertEqual(len(other.dependencies), 0)
        self.system.remove_dependency(dependency.id)
        other.add_dependency(dependency)
        self.assertIs(dependency.energy_system, other)

    def test_ids_are_unique_across_all_registries(self):
        dependency = Dependency()
        dependency.id = self.a.id
        with self.assertRaises(ValueError):
            self.system.add_dependency(dependency)
        dependency.id = self.a.source_vertex.id
        with self.assertRaises(ValueError):
            self.system.add_dependency(dependency)
        dependency.id = uuid4()
        self.system.add_dependency(dependency)
        vertex = Source(carrier=Electricity())
        vertex.id = dependency.id
        with self.assertRaises(ValueError):
            self.system.add_vertex(vertex)
        self.assertFalse(vertex.has_energy_system)
        # Edge UUID collisions must likewise be caught before registration.
        self.system.remove_edge(self.c.id)
        self.c.id = dependency.id
        with self.assertRaises(ValueError):
            self.system.add_edge(self.c)

    def test_all_references_must_be_removed_before_edge_deletion(self):
        exclusive = MutuallyExclusiveFlowDependency(flows=self.edges, label='Exclusive')
        linear = self.linear(label='Balance')
        self.system.add_dependencies([exclusive, linear])
        with self.assertRaisesRegex(ValueError, 'Exclusive.*Balance'):
            self.system.remove_edge(self.a.id)
        self.assertIs(self.system.get_edge(self.a.id), self.a)
        self.assertIs(self.a.source_port.edge, self.a)
        self.assertIs(self.a.sink_port.edge, self.a)
        self.system.remove_dependency(exclusive.id)
        with self.assertRaisesRegex(ValueError, 'Balance'):
            self.system.remove_edge(self.a.id)
        self.system.remove_dependency(linear.id)
        self.system.remove_edge(self.a.id)
        self.assertFalse(self.a.source_port.is_connected)
        self.assertFalse(self.a.sink_port.is_connected)

    def test_vertex_cascade_preflight_prevents_partial_removal(self):
        carrier = Electricity()
        source = Source(carrier=carrier)
        storage = Storage(carrier=carrier, energy_capacity=10)
        sink = Sink(carrier=carrier)
        self.system.add_vertices([source, storage, sink])
        incoming = Edge(source_port=source.output_ports[0], sink_port=storage.input_ports[0])
        outgoing = Edge(source_port=storage.output_ports[0], sink_port=sink.input_ports[0])
        self.system.add_edges([incoming, outgoing])
        dependency = MutuallyExclusiveFlowDependency(flows=outgoing)
        self.system.add_dependency(dependency)
        with self.assertRaises(ValueError):
            self.system.remove_vertex(storage.id, cascade=True)
        for edge in (incoming, outgoing):
            self.assertIs(self.system.get_edge(edge.id), edge)
            self.assertIs(edge.source_port.edge, edge)
            self.assertIs(edge.sink_port.edge, edge)
        self.assertIs(self.system.get_vertex(storage.id), storage)
        self.assertIs(storage.energy_system, self.system)
        self.system.remove_dependency(dependency.id)
        self.system.remove_vertex(storage.id, cascade=True)
        self.assertNotIn(storage.id, self.system.vertices)
        self.assertNotIn(incoming.id, self.system.edges)
        self.assertNotIn(outgoing.id, self.system.edges)

    def test_bidirectional_edges_are_protected_individually(self):
        carrier = Electricity()
        first, second = [Storage(carrier=carrier, energy_capacity=10) for _ in range(2)]
        self.system.add_vertices([first, second])
        pair = BidirectionalEdge(forward_source_port=first.output_ports[0],
                                 forward_sink_port=second.input_ports[0],
                                 backward_source_port=second.output_ports[0],
                                 backward_sink_port=first.input_ports[0])
        self.system.add_edge(pair)
        dependency = MutuallyExclusiveFlowDependency(flows=pair.edges)
        self.system.add_dependency(dependency)
        with self.assertRaises(ValueError):
            self.system.remove_vertex(first.id, cascade=True)
        for edge in pair.edges:
            with self.assertRaises(ValueError):
                self.system.remove_edge(edge.id)
            self.assertIs(self.system.get_edge(edge.id), edge)

    def test_unreferenced_edge_can_be_deleted(self):
        self.system.add_dependency(MutuallyExclusiveFlowDependency(flows=self.a))
        self.system.remove_edge(self.b.id)
        self.assertNotIn(self.b.id, self.system.edges)
        self.assertIn(self.a.id, self.system.edges)


if __name__ == '__main__':
    unittest.main()
