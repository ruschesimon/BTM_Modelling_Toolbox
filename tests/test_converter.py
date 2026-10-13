import unittest
from unittest.mock import Mock
from uuid import uuid4

import pandas as pd

from btm_modelling_toolbox import Converter, Edge, Electricity, EnergySystem, Sink, Source


class ConverterTests(unittest.TestCase):
    names = ('input_conversion', 'output_conversion', 'input_min',
             'input_max', 'output_min', 'output_max')

    def setUp(self):
        self.a, self.b, self.c = (Electricity(label=name) for name in ('A', 'B', 'C'))
        self.system = EnergySystem(time_index=pd.RangeIndex(3))

    def converter(self, **kwargs):
        params = dict(input_carriers=self.a, output_carriers=self.b,
                      input_conversion=1, output_conversion=0.9, energy_system=self.system)
        params.update(kwargs)
        return Converter(**params)

    def test_defaults_ports_and_connections(self):
        converter = self.converter(variable_cost=2, fixed_cost=3)
        self.assertEqual(converter.inputs, (self.a,))
        self.assertEqual(converter.outputs, (self.b,))
        self.assertEqual(converter.input_min[self.a.id].tolist(), [0.] * 3)
        self.assertEqual(converter.output_max[self.b.id].tolist(), [float('inf')] * 3)
        self.assertEqual(converter.output_conversion[self.b.id].tolist(), [0.9] * 3)
        self.assertEqual(converter.variable_cost.tolist(), [2.] * 3)
        self.assertEqual(converter.fixed_cost, 3)
        self.assertEqual(len(converter.input_ports), 1)
        self.assertEqual(len(converter.output_ports), 1)
        source, sink = Source(carrier=self.a), Sink(carrier=self.b)
        incoming = Edge(source_port=next(iter(source.ports.values())),
                        sink_port=converter.input_ports[0])
        outgoing = Edge(source_port=converter.output_ports[0],
                        sink_port=next(iter(sink.ports.values())))
        self.assertIs(incoming.sink_vertex, converter)
        self.assertIs(outgoing.source_vertex, converter)

    def test_multiple_carriers_and_series(self):
        converter = self.converter(
            input_carriers=iter([self.a, self.c]),
            input_conversion={self.a.id: 1, self.c.id: pd.Series([3., 1., 2.], index=[2, 0, 1])},
            input_min={self.a.id: 0, self.c.id: 1},
            input_max={self.a.id: 4, self.c.id: pd.Series([3, 4, 5])},
        )
        self.assertEqual(len(converter.input_ports), 2)
        result = converter.input_conversion
        self.assertEqual(result[self.c.id].tolist(), [1., 2., 3.])
        self.assertIsInstance(result[self.c.id].index, pd.RangeIndex)
        with self.assertRaises(TypeError):
            result[self.a.id] = pd.Series([0])
        result[self.c.id].iloc[0] = 999
        self.assertEqual(converter.input_conversion[self.c.id].iloc[0], 1)
        # UUID port labels avoid collisions between distinct, equally labelled carriers.
        self.c.label = self.a.label
        other = self.converter(input_carriers=[self.a, self.c],
                               input_conversion={self.a.id: 1, self.c.id: 1})
        self.assertEqual(len(other.ports), 3)

    def test_invalid_carriers_and_mappings(self):
        for value, error in [([], ValueError), ([self.a, self.a], ValueError),
                             ('A', TypeError), ([object()], TypeError), (None, TypeError)]:
            with self.subTest(value=value), self.assertRaises(error):
                self.converter(input_carriers=value)
        with self.assertRaises(ValueError):
            self.converter(output_carriers=self.a)
        with self.assertRaises(TypeError):
            self.converter(input_carriers=[self.a, self.c])
        for mapping in [{}, {uuid4(): 1}, {self.a.id: 1, self.c.id: 2}]:
            with self.assertRaises(ValueError):
                self.converter(input_conversion=mapping)

    def test_numeric_validation_and_bounds(self):
        for name in self.names:
            for value in [-1, True, float('nan'), 'bad', pd.Series([1, -1])]:
                with self.subTest(name=name, value=value), self.assertRaises((TypeError, ValueError)):
                    self.converter(**{name: value})
        for side in ('input', 'output'):
            for lower, upper in [(3, 2), (pd.Series([1, 3, 1]), 2),
                                 (1, pd.Series([0, 2, 2])),
                                 (pd.Series([1], index=[0]), pd.Series([2], index=[1]))]:
                with self.assertRaises(ValueError):
                    self.converter(**{f'{side}_min': lower, f'{side}_max': upper})
            with self.assertRaises(ValueError):
                self.converter(**{f'{side}_conversion': float('inf')})
        self.converter(input_conversion=0, output_conversion=3)

    def test_system_lifecycle(self):
        converter = self.converter(energy_system=None)
        for name in self.names:
            with self.assertRaises(RuntimeError):
                getattr(converter, name)
        for system in [object(), Mock(spec=EnergySystem, time_index=pd.Index([0, 1]))]:
            converter._set_energy_system(system)
            with self.assertRaises(TypeError):
                _ = converter.input_conversion
            converter._remove_energy_system()
        converter._set_energy_system(self.system)
        self.assertIs(converter.energy_system, self.system)
        with self.assertRaises(ValueError):
            converter._set_energy_system(self.system)
        self.assertEqual(converter.input_conversion[self.a.id].tolist(), [1.] * 3)

    def test_series_index_errors(self):
        for series, error in [(pd.Series([1], index=[0]), KeyError),
                              (pd.Series([1, 2], index=[0, 0]), ValueError)]:
            converter = self.converter(input_conversion=series)
            with self.assertRaises(error):
                _ = converter.input_conversion


if __name__ == '__main__':
    unittest.main()
