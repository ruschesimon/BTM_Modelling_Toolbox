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


class EdgeTransmissionTests(unittest.TestCase):
    ports = EdgeCostTests.ports

    def test_capacity_and_efficiency_defaults(self):
        edge = Edge(**self.ports(), energy_system=EnergySystem(time_index=pd.RangeIndex(3)))
        self.assertEqual(edge.capacity.tolist(), [float('inf')] * 3)
        self.assertEqual(edge.efficiency.tolist(), [1.] * 3)
        self.assertEqual(edge.compute_received_flow(5).tolist(), [5.] * 3)
        self.assertEqual(edge.compute_losses(5).tolist(), [0.] * 3)

    def test_time_dependent_transfer_and_conservation(self):
        index = pd.RangeIndex(2, 8, 2, name='time')
        edge = Edge(**self.ports(), energy_system=EnergySystem(time_index=index),
                    capacity=pd.Series([30., 10., 20.], index=[6, 2, 4]),
                    efficiency=pd.Series([0., 1., 0.8], index=[6, 2, 4]))
        sent = pd.Series([10., 20., 30.], index=index)
        received, losses = edge.compute_received_flow(sent), edge.compute_losses(sent)
        assert_series_equal(received, pd.Series([10., 16., 0.], index=index, name='received_flow'))
        assert_series_equal(losses, pd.Series([0., 4., 30.], index=index, name='losses'))
        assert_series_equal((received + losses).rename(None), sent)
        self.assertEqual(edge.capacity.tolist(), [10., 20., 30.])
        with self.assertRaises(ValueError):
            edge.compute_received_flow(11)

    def test_invalid_transmission_parameters_leave_ports_unconnected(self):
        for name, values in [('capacity', [-1, True, None, float('nan'), pd.Series([1, -1])]),
                             ('efficiency', [-0.1, 1.1, float('inf'), float('nan'), True,
                                             pd.Series([0.9, 1.1])])]:
            for value in values:
                ports = self.ports()
                with self.subTest(name=name, value=value), self.assertRaises((TypeError, ValueError)):
                    Edge(**ports, **{name: value})
                self.assertFalse(ports['source_port'].is_connected)
                self.assertFalse(ports['sink_port'].is_connected)

    def test_flow_validation_and_zero_capacity(self):
        edge = Edge(**self.ports(), capacity=0, efficiency=0,
                    energy_system=EnergySystem(time_index=pd.RangeIndex(2)))
        self.assertEqual(edge.compute_received_flow(0).tolist(), [0., 0.])
        for value in [-1, 1, True, float('inf'), float('nan')]:
            with self.assertRaises((ValueError, TypeError)):
                edge.compute_received_flow(value)

    def test_transmission_parameter_context_and_indices(self):
        edge = Edge(**self.ports())
        for name in ('capacity', 'efficiency'):
            with self.assertRaises(RuntimeError):
                getattr(edge, name)
        for name in ('capacity', 'efficiency'):
            for value, error in [(pd.Series([1], index=[0]), KeyError),
                                 (pd.Series([1, 1], index=[0, 0]), ValueError)]:
                edge = Edge(**self.ports(), **{name: value},
                            energy_system=EnergySystem(time_index=pd.RangeIndex(2)))
                with self.assertRaises(error):
                    getattr(edge, name)


if __name__ == '__main__':
    unittest.main()
