import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from btm_modelling_toolbox import (Converter, CurrentType, Edge, Electricity, EnergySystem,
                                   Heat, Sink, Source, Transformer)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        self.primary = Electricity(voltage_level=20_000)
        self.secondary = Electricity(voltage_level=400)
        self.index = pd.RangeIndex(3, name='time')
        self.system = EnergySystem(time_index=self.index)

    def transformer(self, **kwargs):
        params = dict(primary_carrier=self.primary, secondary_carrier=self.secondary,
                      capacity=10., energy_system=self.system)
        params.update(kwargs)
        return Transformer(**params)

    def assert_constant(self, values, value):
        np.testing.assert_allclose(values.to_numpy(), value)

    def test_defaults_and_four_ports(self):
        model = self.transformer()
        self.assertIs(Transformer.__bases__[0], Converter)
        self.assertEqual(len(model.ports), 4)
        self.assertEqual(len(model.input_ports), 2)
        self.assertEqual(len(model.output_ports), 2)
        self.assertIs(model.primary_input.carrier, self.primary)
        self.assertIs(model.primary_output.carrier, self.primary)
        self.assertIs(model.secondary_input.carrier, self.secondary)
        self.assertIs(model.secondary_output.carrier, self.secondary)
        self.assertTrue(model.primary_input.is_input)
        self.assertTrue(model.secondary_output.is_output)
        self.assertEqual(len({p.label for p in model.ports.values()}), 4)
        self.assert_constant(model.capacity, 10.)
        self.assert_constant(model.efficiency, 1.)
        self.assert_constant(model.no_load_loss, 0.)

    def test_both_port_pairs_have_nonzero_capacity_and_conversion(self):
        model = self.transformer(efficiency=.98, no_load_loss=.01)
        for carrier in (self.primary, self.secondary):
            with self.subTest(carrier=carrier):
                self.assert_constant(model.input_conversion[carrier.id], 1.)
                self.assert_constant(model.input_max[carrier.id], 10.)
                self.assert_constant(model.output_conversion[carrier.id], .98)
                self.assert_constant(model.output_max[carrier.id], 9.8)
                # Loss allocation belongs to the external formulation, not to
                # two independent output offsets charging the same loss twice.
                self.assert_constant(model.input_offset[carrier.id], 0.)
                self.assert_constant(model.output_offset[carrier.id], 0.)
        self.assert_constant(model.no_load_loss, .01)

    def test_series_parameters_are_aligned_on_both_voltage_levels(self):
        model = self.transformer(capacity=pd.Series([10., 20., 30.]),
                                 efficiency=pd.Series([.9, .95, 1.]),
                                 no_load_loss=pd.Series([.01, .02, .03]))
        assert_series_equal(model.capacity, pd.Series([10., 20., 30.], index=self.index, name='capacity'))
        self.assertEqual(model.efficiency.tolist(), [.9, .95, 1.])
        self.assertEqual(model.no_load_loss.tolist(), [.01, .02, .03])
        self.assertEqual(model.input_max[self.secondary.id].tolist(), [10., 20., 30.])
        self.assertEqual(model.output_conversion[self.primary.id].tolist(), [.9, .95, 1.])
        self.assertEqual(model.input_max[self.primary.id].tolist(), [10., 20., 30.])
        result = model.capacity
        result.iloc[0] = 999
        self.assertEqual(model.capacity.iloc[0], 10.)

    def test_all_four_ports_can_be_connected_at_the_same_time(self):
        model = self.transformer()
        self.system.add_vertex(model)
        for port in model.ports.values():
            peer_carrier = Electricity(voltage_level=port.carrier.voltage_level)
            peer = Source(carrier=peer_carrier) if port.is_input else Sink(carrier=peer_carrier)
            self.system.add_vertex(peer)
            edge = (Edge(source_port=peer.output_ports[0], sink_port=port) if port.is_input else
                    Edge(source_port=port, sink_port=peer.input_ports[0]))
            self.system.add_edge(edge)
        self.assertEqual(len(self.system.edges), 4)
        for port in model.ports.values():
            self.assertIs(self.system.get_edge(port.edge.id), port.edge)
            limits = model.input_max if port.is_input else model.output_max
            self.assertTrue((limits[port.carrier.id] > 0).all())

    def test_invalid_parameters_and_carriers(self):
        for kwargs in [dict(capacity=-1), dict(capacity=float('inf')), dict(capacity=True),
                       dict(efficiency=0), dict(efficiency=1.01), dict(efficiency=float('nan')),
                       dict(no_load_loss=-1), dict(no_load_loss=float('inf')),
                       dict(primary_carrier=Heat()), dict(primary_carrier=Electricity()),
                       dict(secondary_carrier=Electricity(voltage_level=20_000)),
                       dict(primary_carrier=Electricity(current_type=CurrentType.DC, voltage_level=20_000)),
                       dict(capacity=pd.Series([1], index=[0]), efficiency=pd.Series([1], index=[1])),
                       dict(no_load_loss=pd.Series([0, 0], index=[0, 0]))]:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                self.transformer(**kwargs)

    def test_no_direction_or_mode_api(self):
        model = self.transformer()
        self.assertFalse(hasattr(model, 'direction'))
        self.assertFalse(hasattr(model, 'set_direction'))
        with self.assertRaises(TypeError):
            self.transformer(direction='primary_to_secondary')

    def test_context_and_missing_profile_labels(self):
        model = self.transformer(energy_system=None)
        for name in ('capacity', 'efficiency', 'no_load_loss', 'input_conversion'):
            with self.assertRaises(RuntimeError):
                getattr(model, name)
        model._set_energy_system(self.system)
        self.assert_constant(model.capacity, 10.)
        model = self.transformer(capacity=pd.Series([10], index=[0]))
        with self.assertRaises(KeyError):
            _ = model.capacity
        with self.assertRaises(KeyError):
            _ = model.input_max

    def test_wrong_voltage_and_dc_external_connections_are_rejected(self):
        for peer in [Electricity(voltage_level=400),
                     Electricity(current_type=CurrentType.DC, voltage_level=20_000)]:
            model = self.transformer()
            source = Source(carrier=peer)
            with self.assertRaises(ValueError):
                Edge(source_port=source.output_ports[0], sink_port=model.primary_input)
            self.assertFalse(model.primary_input.is_connected)
            self.assertFalse(source.output_ports[0].is_connected)


if __name__ == '__main__':
    unittest.main()
