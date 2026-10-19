"""Electrical carrier characteristics at generic and thermal converter ports."""

from btm_modelling_toolbox import (Converter, CurrentType, Edge, Electricity,
                                   EnergySystem, Sink, Source)
from btm_modelling_toolbox.core.carrier import CarrierCategory
from test_thermal_converters import ThermalTestCase


class ConverterElectricityTests(ThermalTestCase):
    def model(self, kind, current_type=CurrentType.AC, voltage=400.):
        power = Electricity(current_type=current_type, voltage_level=voltage)
        if kind == 'CombinedHeatPower':
            return self.chp(output_carriers=[self.heat, power])
        if kind == 'HeatPump':
            return self.pump(input_carriers=[self.source_heat, power])
        return Converter(input_carriers=power,
                         output_carriers=Electricity(current_type=current_type, voltage_level=voltage),
                         input_conversion=1., output_conversion=.9)

    @staticmethod
    def electrical_ports(model):
        return [port for port in model.ports.values()
                if port.carrier.category == CarrierCategory.ELECTRICITY]

    @staticmethod
    def connection(port, peer_carrier):
        if port.is_input:
            peer = Source(carrier=peer_carrier)
            return peer, dict(source_port=peer.output_ports[0], sink_port=port)
        peer = Sink(carrier=peer_carrier)
        return peer, dict(source_port=port, sink_port=peer.input_ports[0])

    def test_characteristics_preserved_for_all_electrical_converter_ports(self):
        for kind in ('Converter', 'CombinedHeatPower', 'HeatPump'):
            for current_type in (CurrentType.AC, CurrentType.DC):
                for voltage in (230., 400., None):
                    with self.subTest(kind=kind, current_type=current_type, voltage=voltage):
                        model = self.model(kind, current_type, voltage)
                        ports = self.electrical_ports(model)
                        self.assertEqual(len(ports), 2 if kind == 'Converter' else 1)
                        for port in ports:
                            self.assertIs(port.carrier.current_type, current_type)
                            self.assertEqual(port.carrier.voltage_level, voltage)
                            self.assertIs(port.vertex, model)
                            carriers = model.inputs if port.is_input else model.outputs
                            self.assertTrue(any(c is port.carrier for c in carriers))

    def test_compatible_distinct_peers_connect_and_register(self):
        for kind in ('Converter', 'CombinedHeatPower', 'HeatPump'):
            for current_type in (CurrentType.AC, CurrentType.DC):
                with self.subTest(kind=kind, current_type=current_type):
                    model = self.model(kind, current_type)
                    system = EnergySystem()
                    if model.has_energy_system:
                        model._remove_energy_system()
                    system.add_vertex(model)
                    for port in self.electrical_ports(model):
                        peer_carrier = Electricity(current_type=current_type, voltage_level=400.)
                        self.assertNotEqual(peer_carrier.id, port.carrier.id)
                        peer, kwargs = self.connection(port, peer_carrier)
                        edge = Edge(**kwargs)
                        system.add_vertex(peer)
                        system.add_edge(edge)
                        self.assertIs(system.get_edge(edge.id), edge)
                        self.assertIs(edge.energy_system, system)
                        self.assertIs(port.edge, edge)

    def test_wrong_current_type_or_voltage_rejected_on_input_and_output(self):
        for kind in ('Converter', 'CombinedHeatPower', 'HeatPump'):
            for mismatch in ('current_type', 'voltage_level', 'unspecified_voltage'):
                with self.subTest(kind=kind, mismatch=mismatch):
                    model = self.model(kind)
                    for port in self.electrical_ports(model):
                        peer_carrier = Electricity(
                            current_type=CurrentType.DC if mismatch == 'current_type' else CurrentType.AC,
                            voltage_level={'current_type': 400., 'voltage_level': 230.,
                                           'unspecified_voltage': None}[mismatch],
                        )
                        _, kwargs = self.connection(port, peer_carrier)
                        with self.assertRaises(ValueError):
                            Edge(**kwargs)
                        self.assertFalse(kwargs['source_port'].is_connected)
                        self.assertFalse(kwargs['sink_port'].is_connected)

    def test_registration_rechecks_modified_peer_characteristics(self):
        for kind in ('Converter', 'CombinedHeatPower', 'HeatPump'):
            for attribute, value in [('current_type', CurrentType.DC), ('voltage_level', 230.)]:
                with self.subTest(kind=kind, attribute=attribute):
                    model = self.model(kind)
                    if model.has_energy_system:
                        model._remove_energy_system()
                    system = EnergySystem()
                    system.add_vertex(model)
                    for port in self.electrical_ports(model):
                        peer_carrier = Electricity(voltage_level=400.)
                        peer, kwargs = self.connection(port, peer_carrier)
                        edge = Edge(**kwargs)
                        system.add_vertex(peer)
                        setattr(peer_carrier, attribute, value)
                        with self.assertRaises(ValueError):
                            system.add_edge(edge)
                        self.assertNotIn(edge.id, system.edges)
                        self.assertIsNone(edge.energy_system)

    def test_generic_converter_can_bridge_ac_dc_and_voltage_levels(self):
        for output in [Electricity(current_type=CurrentType.DC, voltage_level=800.),
                       Electricity(current_type=CurrentType.AC, voltage_level=230.)]:
            with self.subTest(output=output):
                input_ = Electricity(current_type=CurrentType.AC, voltage_level=400.)
                model = Converter(input_carriers=input_, output_carriers=output,
                                  input_conversion=1., output_conversion=.95)
                system = EnergySystem(time_index=self.index)
                system.add_vertex(model)
                self.assertFalse(input_.is_compatible_with(output))
                for port in model.ports.values():
                    peer_carrier = Electricity(current_type=port.carrier.current_type,
                                               voltage_level=port.carrier.voltage_level)
                    peer, kwargs = self.connection(port, peer_carrier)
                    system.add_vertex(peer)
                    system.add_edge(Edge(**kwargs))
                self.assertEqual(len(system.edges), 2)
                self.assertEqual(model.output_conversion[output.id].tolist(), [.95] * 3)
                self.assertEqual(model.input_conversion[input_.id].tolist(), [1.] * 3)

    def test_equivalent_carriers_keep_distinct_converter_parameter_ids(self):
        model = self.model('Converter')
        self.system.add_vertex(model)
        input_, output = model.inputs[0], model.outputs[0]
        self.assertTrue(input_.is_compatible_with(output))
        self.assertNotEqual(input_.id, output.id)
        self.assertEqual(set(model.input_conversion), {input_.id})
        self.assertEqual(set(model.output_conversion), {output.id})
        self.assertEqual(len(model.ports), 2)

    def test_boiler_rejects_electrical_carriers_in_fuel_and_heat_roles(self):
        for current_type in (CurrentType.AC, CurrentType.DC):
            electricity = Electricity(current_type=current_type, voltage_level=400.)
            for keyword in ('input_carrier', 'output_carrier'):
                with self.subTest(current_type=current_type, keyword=keyword), self.assertRaises(ValueError):
                    self.boiler(**{keyword: electricity})
        boiler = self.boiler()
        self.assertEqual(self.electrical_ports(boiler), [])
