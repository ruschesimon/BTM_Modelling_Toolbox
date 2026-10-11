import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from btm_modelling_toolbox import Electricity, EnergySystem, Sink


class SinkTests(unittest.TestCase):
    def sink(self, **kwargs):
        return Sink(carrier=Electricity(), **kwargs)

    def test_defaults_and_later_assignment(self):
        sink = self.sink()
        self.assertIsNone(sink.energy_system)
        for name in ('min_capacity', 'max_capacity'):
            with self.assertRaises(RuntimeError):
                getattr(sink, name)
        index = pd.RangeIndex(2, 8, 2)
        sink.energy_system = EnergySystem(time_index=index)
        assert_series_equal(sink.min_capacity, pd.Series(0., index=index, name='min_capacity'))
        assert_series_equal(sink.max_capacity, pd.Series(float('inf'), index=index, name='max_capacity'))
        self.assertEqual(len(sink.ports), 1)

    def test_scalar_and_series_keywords(self):
        index = pd.RangeIndex(3)
        for minimum, maximum in [(1, 4), (pd.Series([1, 2, 3]), 4),
                                  (1, pd.Series([3, 4, 5])),
                                  (pd.Series([1, 2, 3]), pd.Series([3, 4, 5]))]:
            with self.subTest(minimum=minimum, maximum=maximum):
                sink = self.sink(min_capacity=minimum, max_capacity=maximum,
                                 energy_system=EnergySystem(time_index=index))
                for name, value in [('min_capacity', minimum), ('max_capacity', maximum)]:
                    expected = pd.Series(value, index=index, dtype=float, name=name)
                    assert_series_equal(getattr(sink, name), expected)

    def test_invalid_system_and_index(self):
        for system in [object(), EnergySystem(time_index=pd.Index([0, 1])),
                       EnergySystem(time_index=None)]:
            sink = self.sink(energy_system=system)
            for name in ('min_capacity', 'max_capacity'):
                with self.assertRaises(TypeError):
                    getattr(sink, name)

    def test_series_alignment_and_missing_labels(self):
        sink = self.sink(min_capacity=pd.Series([3, 1, 2], index=[2, 0, 1]),
                         energy_system=EnergySystem(time_index=pd.RangeIndex(3)))
        self.assertEqual(sink.min_capacity.tolist(), [1., 2., 3.])
        sink.energy_system = EnergySystem(time_index=pd.RangeIndex(4))
        with self.assertRaises(KeyError):
            _ = sink.min_capacity

    def test_invalid_capacities(self):
        for kwargs in [dict(min_capacity=-1), dict(max_capacity=float('nan')),
                       dict(min_capacity=True), dict(min_capacity=4, max_capacity=3),
                       dict(min_capacity=pd.Series([1, 5]), max_capacity=4)]:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                self.sink(**kwargs)

    def test_duplicate_series_index(self):
        sink = self.sink(min_capacity=pd.Series([1, 2], index=[0, 0]),
                         energy_system=EnergySystem(time_index=pd.RangeIndex(2)))
        with self.assertRaises(ValueError):
            _ = sink.min_capacity


if __name__ == '__main__':
    unittest.main()
