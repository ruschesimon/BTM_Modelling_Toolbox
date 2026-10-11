import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from btm_modelling_toolbox import Edge, Electricity, EnergySystem, Sink, Source, Storage


class StorageTests(unittest.TestCase):
    parameter_names = (
        'energy_capacity', 'min_energy_content', 'charge_capacity',
        'discharge_capacity', 'charge_efficiency', 'discharge_efficiency',
        'fixed_loss_absolute', 'fixed_loss_relative', 'loss_rate',
    )
    operating_names = parameter_names[2:]

    def storage(self, **kwargs):
        return Storage(carrier=kwargs.pop('carrier', Electricity()),
                       energy_capacity=kwargs.pop('energy_capacity', 10), **kwargs)

    def test_defaults(self):
        index = pd.RangeIndex(2, 8, 2, name='time')
        storage = self.storage(energy_system=EnergySystem(time_index=index))
        expected = [10., 0., 10., 10., 1., 1., 0., 0., 0.]
        for name, value in zip(self.parameter_names, expected):
            assert_series_equal(getattr(storage, name), pd.Series(value, index=index, name=name))
        self.assertEqual(storage.initial_state, 5)
        self.assertEqual(storage.final_state, 5)

    def test_state_keywords_and_bounds(self):
        for state, expected in [('min', 2), ('mid', 6), ('max', 10), (3, 3)]:
            storage = self.storage(min_energy_content=2, initial_state=state, final_state=state)
            self.assertEqual(storage.initial_state, expected)
            self.assertEqual(storage.final_state, expected)
        for name in ('initial_state', 'final_state'):
            for value in ['invalid', -1, 11, True, np.nan, np.inf, pd.Series([1])]:
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, TypeError)):
                    self.storage(**{name: value})

    def test_time_dependent_parameters(self):
        index = pd.RangeIndex(3)
        values = pd.Series([0.5, 0.25, 0.75], index=[2, 0, 1])
        storage = self.storage(energy_system=EnergySystem(time_index=index),
                               **dict.fromkeys(self.operating_names, values))
        for name in self.operating_names:
            assert_series_equal(getattr(storage, name),
                                pd.Series([0.25, 0.75, 0.5], index=index, name=name))
        result = storage.charge_capacity
        result.iloc[0] = 100
        self.assertEqual(storage.charge_capacity.iloc[0], 0.25)

    def test_invalid_numeric_parameters(self):
        for name in self.parameter_names:
            for value in [-1, True, 'invalid', None, np.nan, np.inf]:
                if value is None and name in ('charge_capacity', 'discharge_capacity'):
                    continue
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, TypeError)):
                    self.storage(**{name: value})
        for name in ('energy_capacity', 'min_energy_content'):
            with self.assertRaises(TypeError):
                self.storage(**{name: pd.Series([1, 2])})
        with self.assertRaises(ValueError):
            self.storage(min_energy_content=11)

    def test_efficiency_and_loss_ranges(self):
        for name in ('charge_efficiency', 'discharge_efficiency'):
            for value in [0, 1.1, pd.Series([1., 0.]), pd.Series([1., 1.1])]:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.storage(**{name: value})
        for name in ('fixed_loss_relative', 'loss_rate'):
            with self.assertRaises(ValueError):
                self.storage(**{name: pd.Series([0., 1.1])})
        storage = self.storage(energy_capacity=0, loss_rate=1, fixed_loss_relative=1)
        self.assertEqual(storage.initial_state, 0)

    def test_system_lifecycle_and_types(self):
        storage = self.storage()
        self.assertFalse(storage.has_energy_system)
        for name in self.parameter_names:
            with self.assertRaises(RuntimeError):
                getattr(storage, name)
        for system in [object(), EnergySystem(time_index=pd.Index([0, 1]))]:
            storage._set_energy_system(system)
            for name in self.parameter_names:
                with self.assertRaises(TypeError):
                    getattr(storage, name)
            storage._remove_energy_system()
        system = EnergySystem(time_index=pd.RangeIndex(2))
        storage._set_energy_system(system)
        self.assertIs(storage.energy_system, system)
        with self.assertRaises(ValueError):
            storage._set_energy_system(system)
        self.assertEqual(storage.energy_capacity.tolist(), [10., 10.])

    def test_series_index_errors(self):
        for value, error in [(pd.Series([1., 2.], index=[0, 0]), ValueError),
                             (pd.Series([1.], index=[0]), KeyError)]:
            storage = self.storage(charge_capacity=value,
                                   energy_system=EnergySystem(time_index=pd.RangeIndex(2)))
            with self.assertRaises(error):
                _ = storage.charge_capacity

    def test_graph_connections_and_costs(self):
        carrier = Electricity(label='Electricity')
        storage = self.storage(carrier=carrier, label='Battery', variable_cost=2, fixed_cost=3)
        ports = tuple(storage.ports.values())
        self.assertEqual(len(ports), 2)
        charge = next(p for p in ports if p.is_input)
        discharge = next(p for p in ports if p.is_output)
        self.assertEqual(charge.label, 'Battery_Electricity_input')
        self.assertEqual(discharge.label, 'Battery_Electricity_output')
        for port in ports:
            self.assertIs(port.vertex, storage)
            self.assertIs(port.carrier, carrier)
        source = Source(carrier=carrier)
        sink = Sink(carrier=carrier)
        incoming = Edge(source_port=next(iter(source.ports.values())), sink_port=charge)
        outgoing = Edge(source_port=discharge, sink_port=next(iter(sink.ports.values())))
        self.assertIs(incoming.sink_vertex, storage)
        self.assertIs(outgoing.source_vertex, storage)
        self.assertEqual(storage.variable_cost, 2)
        self.assertEqual(storage.fixed_cost, 3)
        with self.assertRaises(TypeError):
            self.storage(carrier=object())


if __name__ == '__main__':
    unittest.main()
