import unittest

import numpy as np
import pandas as pd

from btm_modelling_toolbox import (BidirectionalEdge, CurrentType, Edge, Electricity,
                                   EnergySystem, Sink, Source, Storage)


class ElectricityTests(unittest.TestCase):
    def test_defaults_and_current_type_aliases(self):
        carrier = Electricity()
        self.assertIs(carrier.current_type, CurrentType.AC)
        self.assertIsNone(carrier.voltage_level)
        self.assertIs(CurrentType.AC, CurrentType.ALTERNATING_CURRENT)
        self.assertIs(CurrentType.DC, CurrentType.DIRECT_CURRENT)
        self.assertEqual(Electricity(voltage_level=400).voltage_level, 400.)

    def test_compatibility_uses_characteristics_not_identity(self):
        first = Electricity(label='Grid', current_type=CurrentType.AC, voltage_level=400)
        equivalent = Electricity(label='Load', current_type=CurrentType.ALTERNATING_CURRENT,
                                 voltage_level=400.)
        self.assertNotEqual(first.id, equivalent.id)
        self.assertTrue(first.is_compatible_with(equivalent))
        self.assertTrue(equivalent.is_compatible_with(first))
        self.assertTrue(Electricity().is_compatible_with(Electricity()))
        for incompatible in [Electricity(current_type=CurrentType.DC, voltage_level=400),
                             Electricity(voltage_level=230), Electricity(), object()]:
            self.assertFalse(first.is_compatible_with(incompatible))

    def test_invalid_voltage_and_current_type(self):
        for voltage in [0, -1, float('inf'), float('-inf'), float('nan')]:
            with self.subTest(voltage=voltage), self.assertRaises(ValueError):
                Electricity(voltage_level=voltage)
        for voltage in [True, np.bool_(True), '400', 400j, pd.Series([400])]:
            with self.subTest(voltage=voltage), self.assertRaises(TypeError):
                Electricity(voltage_level=voltage)
        for current_type in ['AC', 'DC', None, 1]:
            with self.assertRaises(TypeError):
                Electricity(current_type=current_type)


class ElectricalConnectionTests(unittest.TestCase):
    def ports(self, first, second):
        return Source(carrier=first).output_ports[0], Sink(carrier=second).input_ports[0]

    def test_compatible_distinct_carriers_connect_and_register(self):
        output, input_ = self.ports(Electricity(voltage_level=400), Electricity(voltage_level=400))
        edge = Edge(source_port=output, sink_port=input_)
        system = EnergySystem(time_index=pd.RangeIndex(2))
        system.add_vertices([output.vertex, input_.vertex])
        system.add_edge(edge)
        self.assertIs(system.get_edge(edge.id), edge)
        self.assertIs(edge.carrier, output.carrier)

    def test_incompatible_connections_do_not_change_ports(self):
        for second in [Electricity(current_type=CurrentType.DC, voltage_level=400),
                       Electricity(voltage_level=230), Electricity()]:
            output, input_ = self.ports(Electricity(voltage_level=400), second)
            with self.assertRaises(ValueError):
                Edge(source_port=output, sink_port=input_)
            self.assertFalse(output.is_connected)
            self.assertFalse(input_.is_connected)

    def test_registration_rechecks_modified_characteristics(self):
        output, input_ = self.ports(Electricity(voltage_level=400), Electricity(voltage_level=400))
        edge = Edge(source_port=output, sink_port=input_)
        system = EnergySystem()
        system.add_vertices([output.vertex, input_.vertex])
        input_.carrier.current_type = CurrentType.DC
        with self.assertRaises(ValueError):
            system.add_edge(edge)
        self.assertEqual(len(system.edges), 0)
        self.assertIsNone(edge.energy_system)

    def test_bidirectional_compatibility(self):
        first, second = [Storage(carrier=Electricity(voltage_level=400), energy_capacity=10)
                         for _ in range(2)]
        pair = BidirectionalEdge(forward_source_port=first.output_ports[0],
                                 forward_sink_port=second.input_ports[0],
                                 backward_source_port=second.output_ports[0],
                                 backward_sink_port=first.input_ports[0])
        system = EnergySystem()
        system.add_vertices([first, second])
        system.add_edge(pair)
        self.assertEqual(len(system.edges), 2)

    def test_bidirectional_incompatibility(self):
        first = Storage(carrier=Electricity(voltage_level=400), energy_capacity=10)
        second = Storage(carrier=Electricity(current_type=CurrentType.DC, voltage_level=400),
                         energy_capacity=10)
        with self.assertRaises(ValueError):
            BidirectionalEdge(forward_source_port=first.output_ports[0],
                              forward_sink_port=second.input_ports[0],
                              backward_source_port=second.output_ports[0],
                              backward_sink_port=first.input_ports[0])
        for vertex in (first, second):
            self.assertTrue(all(not port.is_connected for port in vertex.ports.values()))


if __name__ == '__main__':
    unittest.main()
