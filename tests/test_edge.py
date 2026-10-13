import unittest
from unittest.mock import Mock

import pandas as pd
from pandas.testing import assert_series_equal

from btm_modelling_toolbox import Edge, Electricity, EnergySystem, Sink, Source


class EdgeCostTests(unittest.TestCase):
    def ports(self):
        carrier = Electricity()
        source, sink = Source(carrier=carrier), Sink(carrier=carrier)
        return dict(source_port=source.output_ports[0], sink_port=sink.input_ports[0])

    def test_default_scalar_and_series(self):
        index = pd.RangeIndex(2, 8, 2, name='time')
        system = EnergySystem(time_index=index)
        for value in [0., 2., -1., pd.Series([3., 1., 2.], index=[6, 2, 4])]:
            with self.subTest(value=value):
                edge = Edge(**self.ports(), variable_cost=value, energy_system=system)
                expected = pd.Series(value, index=index, dtype=float, name='variable_cost')
                assert_series_equal(edge.variable_cost, expected)
                self.assertIs(edge.source_port.edge, edge)
                self.assertIs(edge.sink_port.edge, edge)
        edge = Edge(**self.ports(), energy_system=system)
        assert_series_equal(edge.variable_cost, pd.Series(0., index=index, name='variable_cost'))
        self.assertTrue(edge.label.startswith('Edge '))

    def test_missing_system_and_later_assignment(self):
        edge = Edge(**self.ports(), variable_cost=2)
        with self.assertRaises(RuntimeError):
            _ = edge.variable_cost
        edge.energy_system = EnergySystem(time_index=pd.RangeIndex(3))
        self.assertEqual(edge.variable_cost.tolist(), [2., 2., 2.])

    def test_invalid_system_or_index(self):
        for system in [object(), EnergySystem(time_index=None),
                       Mock(spec=EnergySystem, time_index=pd.Index([0, 1]))]:
            edge = Edge(**self.ports(), energy_system=system)
            with self.assertRaises(TypeError):
                _ = edge.variable_cost

    def test_invalid_cost_does_not_connect_ports(self):
        for value in [True, None, 'bad', float('nan'), pd.Series([1, float('nan')])]:
            ports = self.ports()
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                Edge(**ports, variable_cost=value)
            self.assertFalse(ports['source_port'].is_connected)
            self.assertFalse(ports['sink_port'].is_connected)

    def test_series_index_errors_and_copy(self):
        system = EnergySystem(time_index=pd.RangeIndex(3))
        for value, error in [(pd.Series([1], index=[0]), KeyError),
                             (pd.Series([1, 2], index=[0, 0]), ValueError)]:
            edge = Edge(**self.ports(), variable_cost=value, energy_system=system)
            with self.assertRaises(error):
                _ = edge.variable_cost
        edge = Edge(**self.ports(), variable_cost=pd.Series([1., 2., 3.]), energy_system=system)
        result = edge.variable_cost
        result.iloc[0] = 999
        self.assertEqual(edge.variable_cost.iloc[0], 1)


if __name__ == '__main__':
    unittest.main()
