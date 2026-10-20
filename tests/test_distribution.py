import unittest
from unittest.mock import Mock

import pandas as pd
from pandas.testing import assert_series_equal

from btm_modelling_toolbox import Bus, Distribution, Edge, Electricity, EnergySystem, Sink, Source


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.carrier = Electricity(voltage_level=400)
        self.index = pd.RangeIndex(2, 8, 2, name='time')
        self.system = EnergySystem(time_index=self.index)

    def distribution(self, **kwargs):
        return Distribution(carrier=self.carrier, **kwargs)

    def test_default_scalar_and_series_capacity(self):
        for kwargs in [{}, dict(capacity=0), dict(capacity=12.5),
                       dict(capacity=pd.Series([30., 10., 20.], index=[6, 2, 4]))]:
            with self.subTest(kwargs=kwargs):
                distribution = self.distribution(**kwargs, energy_system=self.system)
                expected = pd.Series(kwargs.get('capacity', float('inf')), index=self.index,
                                     dtype=float, name='capacity')
                assert_series_equal(distribution.capacity, expected)
                result = distribution.capacity
                result.iloc[0] = -1
                assert_series_equal(distribution.capacity, expected)

    def test_missing_system_later_assignment_and_invalid_context(self):
        distribution = self.distribution(capacity=5)
        with self.assertRaises(RuntimeError):
            _ = distribution.capacity
        self.system.add_vertex(distribution)
        self.assertEqual(distribution.capacity.tolist(), [5., 5., 5.])
        for system in [object(), EnergySystem(), Mock(spec=EnergySystem, time_index=pd.Index([0, 1]))]:
            distribution = self.distribution(energy_system=system)
            with self.assertRaises(TypeError):
                _ = distribution.capacity

    def test_invalid_capacity_and_carrier(self):
        for value in [-1, float('-inf'), float('nan'), True, None, 'bad',
                      pd.Series([1, -1]), pd.Series([1, float('nan')])]:
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                self.distribution(capacity=value)
        with self.assertRaises(TypeError):
            Distribution(carrier=object())

    def test_series_index_errors(self):
        for values, error in [(pd.Series([1], index=[2]), KeyError),
                              (pd.Series([1, 2], index=[2, 2]), ValueError)]:
            distribution = self.distribution(capacity=values, energy_system=self.system)
            with self.assertRaises(error):
                _ = distribution.capacity

    def test_inherited_ports_registration_and_connections(self):
        distribution = self.distribution(capacity=10, label='Distribution')
        self.assertIs(Distribution.__bases__[0], Bus)
        self.assertEqual(len(distribution.input_ports), 1)
        self.assertEqual(len(distribution.output_ports), 1)
        for port in distribution.ports.values():
            self.assertIs(port.carrier, self.carrier)
            self.assertIs(port.vertex, distribution)
        source = Source(carrier=Electricity(voltage_level=400))
        sink = Sink(carrier=Electricity(voltage_level=400))
        self.system.add_vertices([source, distribution, sink])
        self.assertIs(self.system.buses[distribution.id], distribution)
        self.assertNotIn(distribution.id, self.system.components)
        incoming = Edge(source_port=source.output_ports[0], sink_port=distribution.input_ports[0])
        outgoing = Edge(source_port=distribution.output_ports[0], sink_port=sink.input_ports[0])
        self.system.add_edges([incoming, outgoing])
        self.assertIs(incoming.sink_vertex, distribution)
        self.assertIs(outgoing.source_vertex, distribution)
        self.assertEqual(distribution.capacity.tolist(), [10.] * 3)


if __name__ == '__main__':
    unittest.main()
