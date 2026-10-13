import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from btm_modelling_toolbox import Converter, Electricity, EnergySystem, Sink, Source, Storage
from btm_modelling_toolbox.components.component import Component


class ComponentCostTests(unittest.TestCase):
    def factories(self):
        carrier = Electricity()
        return [
            ('Component', lambda **kwargs: Component(**kwargs)),
            ('Sink', lambda **kwargs: Sink(carrier=carrier, **kwargs)),
            ('Source', lambda **kwargs: Source(carrier=carrier, **kwargs)),
            ('Storage', lambda **kwargs: Storage(carrier=carrier, energy_capacity=10, **kwargs)),
            ('Converter', lambda **kwargs: Converter(input_carriers=carrier,
                output_carriers=Electricity(), input_conversion=1, output_conversion=1, **kwargs)),
        ]

    def test_scalar_series_and_defaults_across_components(self):
        index = pd.RangeIndex(2, 8, 2, name='time')
        system = EnergySystem(time_index=index)
        for name, factory in self.factories():
            for kwargs in [{}, {'variable_cost': 2}, {'variable_cost': -1},
                           {'variable_cost': pd.Series([3., 1., 2.], index=[6, 2, 4])}]:
                with self.subTest(component=name, kwargs=kwargs):
                    component = factory(**kwargs, fixed_cost=3)
                    component._set_energy_system(system)
                    expected = pd.Series(kwargs.get('variable_cost', 0.), index=index,
                                         dtype=float, name='variable_cost')
                    assert_series_equal(component.variable_cost, expected)
                    self.assertEqual(component.fixed_cost, 3)
                    result = component.variable_cost
                    result.iloc[0] = 999
                    assert_series_equal(component.variable_cost, expected)

    def test_invalid_costs_across_components(self):
        for name, factory in self.factories():
            for value in [True, None, 'bad', float('nan'), pd.Series([1, float('nan')])]:
                with self.subTest(component=name, value=value), self.assertRaises((TypeError, ValueError)):
                    factory(variable_cost=value)

    def test_missing_and_invalid_system(self):
        for name, factory in self.factories():
            with self.subTest(component=name):
                component = factory()
                with self.assertRaises(RuntimeError):
                    _ = component.variable_cost
                for system in [object(), EnergySystem(time_index=None),
                               EnergySystem(time_index=pd.Index([0, 1]))]:
                    component._set_energy_system(system)
                    with self.assertRaises(TypeError):
                        _ = component.variable_cost
                    component._remove_energy_system()

    def test_series_index_errors(self):
        for name, factory in self.factories():
            for value, error in [(pd.Series([1], index=[0]), KeyError),
                                 (pd.Series([1, 2], index=[0, 0]), ValueError)]:
                with self.subTest(component=name):
                    component = factory(variable_cost=value)
                    component._set_energy_system(EnergySystem(time_index=pd.RangeIndex(3)))
                    with self.assertRaises(error):
                        _ = component.variable_cost


if __name__ == '__main__':
    unittest.main()
