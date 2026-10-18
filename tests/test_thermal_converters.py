"""Reference equations and integration contracts of the thermal converters."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from btm_modelling_toolbox import (CombinedHeatPower, Converter, Edge, Electricity,
                                   EnergySystem, HeatBoiler, HeatPump, Sink, Source)
from btm_modelling_toolbox.core.carrier import Carrier, CarrierCategory


# Test carriers implement the existing Carrier interface; no thermal carrier
# implementation or additional dependency is required by the converter classes.
class HeatCarrier(Carrier):
    label = 'heat'
    category = CarrierCategory.HEAT
    default_unit = 'MW'
    compatibility_key = ('heat',)


class FuelCarrier(Carrier):
    label = 'fuel'
    category = CarrierCategory.GAS
    default_unit = 'MW'
    compatibility_key = ('fuel',)


class ThermalTestCase(unittest.TestCase):
    def setUp(self):
        self.fuel = FuelCarrier(label='Fuel')
        self.heat = HeatCarrier(label='Heat')
        self.source_heat = HeatCarrier(label='Source heat')
        self.power = Electricity(label='Electricity')
        self.index = pd.RangeIndex(3, name='time')
        self.system = EnergySystem(time_index=self.index)

    def boiler(self, **kwargs):
        params = dict(input_carrier=self.fuel, output_carrier=self.heat,
                      power_min_efficiency=.7, power_max_efficiency=.9,
                      power_min_capacity=10., power_max_capacity=100., energy_system=self.system)
        params.update(kwargs)
        return HeatBoiler(**params)

    def chp(self, **kwargs):
        params = dict(input_carrier=self.fuel, output_carriers=[self.power, self.heat],
                      power_min_efficiency=.25, power_max_efficiency=.4, chp_total_efficiency=.9,
                      power_min_capacity=10., power_max_capacity=100., energy_system=self.system)
        params.update(kwargs)
        return CombinedHeatPower(**params)

    def pump(self, **kwargs):
        params = dict(input_carriers=[self.power, self.source_heat], output_carrier=self.heat,
                      norm_cop=4., norm_supply_temperature=35., norm_source_temperature=10.,
                      source_temperature=10., temperature_unit='Celsius',
                      heat_min_capacity=0., heat_max_capacity=100., energy_system=self.system)
        params.update(kwargs)
        return HeatPump(**params)

    def assert_constant(self, series, value):
        np.testing.assert_allclose(series.to_numpy(), value)
        self.assertTrue(series.index.equals(self.index))


class HeatBoilerTests(ThermalTestCase):
    def test_legacy_curve_and_endpoints(self):
        boiler = self.boiler()
        self.assertIs(HeatBoiler.__bases__[0], Converter)
        # Independent oracle: unchanged equations in the uploaded reference.
        a = ((1/.9) - (1/.7)) / ((1/100.) - (1/10.))
        b = (1/.7) - (((1/.9) - (1/.7)) / ((10./100.) - 1))
        self.assert_constant(boiler.efficiency_absolute, a)
        self.assert_constant(boiler.efficiency_relative, b)
        for power, eta in [(10., .7), (100., .9)]:
            fuel = boiler.efficiency_absolute + boiler.efficiency_relative * power
            self.assert_constant(power / fuel, eta)
        self.assert_constant(boiler.input_offset[self.fuel.id], a)
        self.assert_constant(boiler.output_conversion[self.heat.id], 1.)

    def test_equal_efficiencies_and_series_capacities(self):
        lower = pd.Series([10., 20., 30.])
        upper = pd.Series([100., 200., 300.])
        boiler = self.boiler(power_min_capacity=lower, power_max_capacity=upper,
                             power_min_efficiency=.8, power_max_efficiency=.8,
                             heat_min_capacity=lower, heat_max_capacity=upper)
        self.assert_constant(boiler.efficiency_absolute, 0.)
        self.assert_constant(boiler.efficiency_relative, 1/.8)
        self.assertEqual(boiler.heat_max_capacity.tolist(), upper.tolist())
        self.assertEqual(boiler.power_min_capacity.tolist(), lower.tolist())
        # A decreasing reference efficiency can produce a negative intercept;
        # signed affine coefficients must not be clamped to zero.
        decreasing = self.boiler(power_min_efficiency=.9, power_max_efficiency=.7)
        self.assertTrue((decreasing.efficiency_absolute < 0).all())

    def test_invalid_reference_data(self):
        for params in [dict(power_min_capacity=0), dict(power_max_capacity=10),
                       dict(power_min_capacity=-1), dict(power_max_capacity=float('inf')),
                       dict(power_min_efficiency=0), dict(power_max_efficiency=1.1),
                       dict(power_min_efficiency=True), dict(power_min_efficiency=pd.Series([.8])),
                       dict(heat_min_capacity=10, heat_max_capacity=5),
                       dict(input_carrier=self.power), dict(output_carrier=self.fuel),
                       dict(power_min_capacity=pd.Series([10], index=[0]),
                            power_max_capacity=pd.Series([100], index=[1]))]:
            with self.subTest(params=params), self.assertRaises((ValueError, TypeError)):
                self.boiler(**params)


class CombinedHeatPowerTests(ThermalTestCase):
    def test_legacy_coupling_and_energy_conservation(self):
        chp = self.chp()
        self.assertIs(CombinedHeatPower.__bases__[0], Converter)
        a = ((1/.4) - (1/.25)) / ((1/100.) - (1/10.))
        b = (1/.25) - (((1/.4) - (1/.25)) / ((10./100.) - 1))
        self.assert_constant(chp.efficiency_absolute, a)
        self.assert_constant(chp.efficiency_relative, b)
        self.assert_constant(chp.coupling_absolute, .9*a)
        self.assert_constant(chp.coupling_relative, .9*b-1)
        for p, eta in [(10., .25), (100., .4), (55., None)]:
            fuel = chp.efficiency_absolute + chp.efficiency_relative * p
            heat = chp.coupling_absolute + chp.coupling_relative * p
            np.testing.assert_allclose((heat+p)/fuel, .9)
            if eta is not None:
                self.assert_constant(p/fuel, eta)
        self.assertEqual(len(chp.input_ports), 1)
        self.assertEqual(len(chp.output_ports), 2)
        self.assert_constant(chp.power_min_capacity, 10.)

    def test_negative_heat_slope_and_zero_heat_endpoint(self):
        # Fuel is constant at these two valid reference points; heat decreases
        # with power. Converter's generic nonnegative-factor check must not
        # discard this legacy affine case.
        chp = self.chp(power_min_efficiency=.1, power_max_efficiency=1., chp_total_efficiency=1.)
        self.assert_constant(chp.efficiency_relative, 0.)
        self.assert_constant(chp.coupling_relative, -1.)
        self.assert_constant(chp.coupling_absolute + chp.coupling_relative*100, 0.)

    def test_series_curve_coefficients(self):
        minimum = pd.Series([10., 20., 30.])
        maximum = pd.Series([100., 200., 300.])
        chp = self.chp(power_min_capacity=minimum, power_max_capacity=maximum)
        assert_series_equal(chp.power_min_capacity, minimum.rename('power_min_capacity').rename_axis('time'))
        fuel = chp.efficiency_absolute + chp.efficiency_relative * minimum
        np.testing.assert_allclose(minimum / fuel, .25)

    def test_affine_converter_off_state_has_zero_flows(self):
        model = self.chp()
        for parameters, offsets in ((model.input_conversion, model.input_offset),
                                     (model.output_conversion, model.output_offset)):
            for key, factor in parameters.items():
                self.assert_constant(offsets[key]*0 + factor*0, 0.)

    def test_invalid_carriers_and_efficiencies(self):
        for params in [dict(output_carriers=[self.power]),
                       dict(output_carriers=[self.power, self.power]),
                       dict(output_carriers=[self.heat, HeatCarrier()]),
                       dict(input_carrier=object()), dict(chp_total_efficiency=.2),
                       dict(chp_total_efficiency=float('nan'))]:
            with self.subTest(params=params), self.assertRaises((TypeError, ValueError)):
                self.chp(**params)


class HeatPumpTests(ThermalTestCase):
    def test_nominal_cop_and_energy_conservation(self):
        pump = self.pump()
        self.assertIs(HeatPump.__bases__[0], Converter)
        self.assert_constant(pump.cop, 4.)
        self.assert_constant(pump.input_conversion[self.power.id], 1.)
        self.assert_constant(pump.input_conversion[self.source_heat.id], 3.)
        self.assert_constant(pump.output_conversion[self.heat.id], 4.)
        heat_in = pump.input_conversion[self.source_heat.id] * 10
        power_in = pump.input_conversion[self.power.id] * 10
        heat_out = pump.output_conversion[self.heat.id] * 10
        np.testing.assert_allclose(heat_in + power_in, heat_out)
        self.assertEqual(len(pump.input_ports), 2)
        self.assertEqual(len(pump.output_ports), 1)
        self.assert_constant(pump.source_temperature, 283.15)
        self.assert_constant(pump.supply_temperature, 308.15)

    def test_legacy_carnot_equation_with_series(self):
        supply_c = pd.Series([35., 45., 55.])
        source_c = pd.Series([10., 5., 0.])
        pump = self.pump(supply_temperature=supply_c, source_temperature=source_c)
        nominal_carnot = (308.15+5) / ((308.15+5)-(283.15-5))
        factor = 4 / nominal_carnot
        expected = factor * (supply_c+273.15+5) / ((supply_c+273.15+5)-(source_c+273.15-5))
        np.testing.assert_allclose(pump.cop, expected)
        np.testing.assert_allclose(pump.compute_cop(), expected)
        self.assert_constant(pump.compute_cop(35, 10), 4.)
        np.testing.assert_allclose(pump.cop, expected)  # calculator does not reconfigure ports

    def test_kelvin_celsius_and_fahrenheit_equivalence(self):
        for unit, supply, source in [('Kelvin', 308.15, 283.15), ('Celsius', 35, 10),
                                     ('Fahrenheit', 95, 50), ('°F', 95, 50)]:
            with self.subTest(unit=unit):
                pump = self.pump(temperature_unit=unit, norm_supply_temperature=supply,
                                 norm_source_temperature=source, source_temperature=source)
                self.assert_constant(pump.cop, 4.)
                self.assert_constant(pump.compute_cop(supply, source), 4.)

    def test_unity_cop_zero_environmental_input(self):
        pump = self.pump(norm_cop=1.)
        self.assert_constant(pump.cop, 1.)
        self.assert_constant(pump.input_conversion[self.source_heat.id], 0.)

    def test_series_capacity_bounds_and_off_state(self):
        maximum = pd.Series([10., 20., 30.])
        pump = self.pump(power_max_capacity=maximum, heat_max_capacity=maximum*4)
        self.assertEqual(pump.power_max_capacity.tolist(), maximum.tolist())
        self.assertEqual(pump.heat_max_capacity.tolist(), (maximum*4).tolist())
        self.assert_constant(pump.input_offset[self.power.id], 0.)
        self.assert_constant(pump.output_offset[self.heat.id], 0.)

    def test_invalid_temperature_and_cop(self):
        for params in [dict(norm_cop=0), dict(norm_cop=.9), dict(norm_cop=float('inf')),
                       dict(temperature_unit='Rankine'), dict(temperature_unit=2),
                       dict(source_temperature=-274), dict(temperature_spread=-1),
                       dict(norm_supply_temperature=0, norm_source_temperature=10, temperature_spread=0),
                       dict(source_temperature=45), dict(source_temperature=float('nan')),
                       dict(norm_source_temperature=pd.Series([1.])),
                       dict(source_temperature=pd.Series([10, 11], index=[0, 0])),
                       dict(source_temperature=pd.Series([10], index=[0]),
                            supply_temperature=pd.Series([35], index=[1]))]:
            with self.subTest(params=params), self.assertRaises((ValueError, TypeError)):
                self.pump(**params)
        with self.assertRaises(ValueError):
            self.pump().compute_cop(1000, -200)  # COP < 1 is not a heating-mode transfer

    def test_invalid_carrier_roles_and_bounds(self):
        for params in [dict(input_carriers=[self.power]),
                       dict(input_carriers=[self.power, self.fuel]),
                       dict(output_carrier=self.power), dict(output_carrier=self.source_heat),
                       dict(power_min_capacity=10, power_max_capacity=5),
                       dict(heat_min_capacity=-1)]:
            with self.subTest(params=params), self.assertRaises((TypeError, ValueError)):
                self.pump(**params)


class ThermalIntegrationTests(ThermalTestCase):
    def test_operating_parameters_and_validation(self):
        for factory in (self.boiler, self.chp, self.pump):
            model = factory(min_down_time=2, min_up_time=3, initial_operation=True)
            self.assertEqual(model.min_down_time, 2)
            self.assertEqual(model.min_up_time, 3)
            self.assertTrue(model.initial_operation)
            for params in [dict(min_down_time=-1), dict(min_up_time=True),
                           dict(min_down_time=1.5), dict(initial_operation=1)]:
                with self.subTest(factory=factory, params=params), self.assertRaises((TypeError, ValueError)):
                    factory(**params)
        for factory in (self.boiler, self.chp):
            model = factory(min_gradient=-10, max_gradient=20)
            self.assertEqual(model.min_gradient, -10)
            self.assertEqual(model.max_gradient, 20)
            for params in [dict(min_gradient=2, max_gradient=1), dict(min_gradient=float('nan'))]:
                with self.assertRaises(ValueError):
                    factory(**params)

    def test_registration_costs_and_graph_connections(self):
        for factory in (self.boiler, self.chp, self.pump):
            with self.subTest(factory=factory):
                system = EnergySystem(time_index=self.index)
                model = factory(energy_system=None, variable_cost=pd.Series([1., -1., 2.]), fixed_cost=3.)
                with self.assertRaises(RuntimeError):
                    _ = model.heat_max_capacity
                with self.assertRaises(RuntimeError):
                    _ = model.input_conversion
                system.add_vertex(model)
                for port in model.input_ports:
                    source = Source(carrier=port.carrier)
                    system.add_vertex(source)
                    system.add_edge(Edge(source_port=source.output_ports[0], sink_port=port))
                for port in model.output_ports:
                    sink = Sink(carrier=port.carrier)
                    system.add_vertex(sink)
                    system.add_edge(Edge(source_port=port, sink_port=sink.input_ports[0]))
                self.assertEqual(len(system.edges), len(model.ports))
                self.assertEqual(model.variable_cost.tolist(), [1., -1., 2.])
                self.assertEqual(model.fixed_cost, 3.)
                self.assertIs(model.energy_system, system)

    def test_lazy_index_validation_and_returned_copies(self):
        model = self.chp(power_min_capacity=pd.Series([10.], index=[0]),
                         power_max_capacity=pd.Series([100.], index=[0]))
        with self.assertRaises(KeyError):
            _ = model.efficiency_absolute
        pump = self.pump()
        result = pump.cop
        result.iloc[0] = 999
        self.assert_constant(pump.cop, 4.)


if __name__ == '__main__':
    unittest.main()
