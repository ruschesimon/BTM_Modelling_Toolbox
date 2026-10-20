import unittest

import numpy as np
import pandas as pd

from btm_modelling_toolbox.core.carrier import CarrierCategory
from test_thermal_converters import FuelCarrier

from btm_modelling_toolbox import (BidirectionalEdge, CurrentType, Edge, Electricity,
                                   EnergySystem, Sink, Source, Storage, Heat, Bus, Converter, HeatPump,
                                   HeatBoiler, CombinedHeatPower)


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


class HeatTests(unittest.TestCase):
    def test_defaults_characteristics_and_normalization(self):
        heat = Heat()
        self.assertIsNone(heat.temperature_level)
        self.assertIsNone(heat.medium)
        self.assertIs(heat.category, CarrierCategory.HEAT)
        self.assertEqual(heat.default_unit, 'MW')
        heat = Heat(label='Heating circuit', temperature_level=80, medium=' Water ')
        self.assertEqual(heat.temperature_level, 80.)
        self.assertEqual(heat.medium, 'water')
        self.assertEqual(heat.compatibility_key, (80., 'water'))
        self.assertEqual(heat.label, 'Heating circuit')
        self.assertEqual(Heat(temperature_level=-20, medium='air').temperature_level, -20.)

    def test_compatibility(self):
        first = Heat(label='Source', temperature_level=80., medium='water')
        equivalent = Heat(label='Load', temperature_level=80., medium='WATER')
        self.assertNotEqual(first.id, equivalent.id)
        self.assertTrue(first.is_compatible_with(equivalent))
        self.assertTrue(equivalent.is_compatible_with(first))
        self.assertTrue(Heat().is_compatible_with(Heat()))
        for other in [Heat(temperature_level=60, medium='water'),
                      Heat(temperature_level=80, medium='steam'),
                      Heat(temperature_level=80), Heat(medium='water'),
                      Heat(), Electricity(), object()]:
            self.assertFalse(first.is_compatible_with(other))

    def test_invalid_temperature_medium_and_label(self):
        for value in [-273.15, -300, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Heat(temperature_level=value)
        for value in [True, np.bool_(True), '80', 80j, pd.Series([80])]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                Heat(temperature_level=value)
        for value in ['', '  ']:
            with self.assertRaises(ValueError):
                Heat(medium=value)
        for value in [1, True, ['water']]:
            with self.assertRaises(TypeError):
                Heat(medium=value)
        with self.assertRaises(TypeError):
            Heat(label=123)

    def test_compatible_heat_connections_and_system_registration(self):
        source = Source(carrier=Heat(temperature_level=80, medium='water'))
        bus = Bus(carrier=Heat(temperature_level=80, medium=' WATER '))
        sink = Sink(carrier=Heat(temperature_level=80, medium='water'))
        system = EnergySystem()
        system.add_vertices([source, bus, sink])
        edges = [Edge(source_port=source.output_ports[0], sink_port=bus.input_ports[0]),
                 Edge(source_port=bus.output_ports[0], sink_port=sink.input_ports[0])]
        system.add_edges(edges)
        self.assertEqual(len(system.edges), 2)
        for edge in edges:
            self.assertIs(edge.energy_system, system)

    def test_incompatible_heat_connections_and_registration_recheck(self):
        for peer in [Heat(temperature_level=60, medium='water'),
                     Heat(temperature_level=80, medium='air'), Heat()]:
            source = Source(carrier=Heat(temperature_level=80, medium='water'))
            sink = Sink(carrier=peer)
            with self.assertRaises(ValueError):
                Edge(source_port=source.output_ports[0], sink_port=sink.input_ports[0])
            self.assertFalse(source.output_ports[0].is_connected)
            self.assertFalse(sink.input_ports[0].is_connected)
        source = Source(carrier=Heat(temperature_level=80, medium='water'))
        sink = Sink(carrier=Heat(temperature_level=80, medium='water'))
        edge = Edge(source_port=source.output_ports[0], sink_port=sink.input_ports[0])
        system = EnergySystem()
        system.add_vertices([source, sink])
        sink.carrier.temperature_level = 60
        with self.assertRaises(ValueError):
            system.add_edge(edge)
        self.assertEqual(len(system.edges), 0)

    def test_native_heat_carrier_in_converters(self):
        heat = Heat(temperature_level=80, medium='water')
        source_heat = Heat(temperature_level=10, medium='air')
        power = Electricity(voltage_level=400)
        fuel = FuelCarrier()
        models = [
            Converter(input_carriers=source_heat, output_carriers=heat,
                      input_conversion=1., output_conversion=.9),
            HeatPump(input_carriers=[power, source_heat], output_carrier=heat,
                     norm_cop=3., norm_supply_temperature=80, norm_source_temperature=10,
                     source_temperature=10, temperature_unit='Celsius',
                     heat_min_capacity=0, heat_max_capacity=100),
            HeatBoiler(input_carrier=fuel, output_carrier=heat,
                       power_min_efficiency=.7, power_max_efficiency=.9,
                       power_min_capacity=10, power_max_capacity=100),
            CombinedHeatPower(input_carrier=fuel, output_carriers=[power, heat],
                              power_min_efficiency=.25, power_max_efficiency=.4,
                              chp_total_efficiency=.9, power_min_capacity=10, power_max_capacity=100),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                port = next(p for p in model.output_ports if p.carrier is heat)
                sink = Sink(carrier=Heat(temperature_level=80, medium='water'))
                edge = Edge(source_port=port, sink_port=sink.input_ports[0])
                system = EnergySystem(time_index=pd.RangeIndex(2))
                system.add_vertices([model, sink])
                system.add_edge(edge)
                self.assertIs(system.get_edge(edge.id), edge)


if __name__ == '__main__':
    unittest.main()
