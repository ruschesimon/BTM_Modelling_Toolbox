# Thermal converters

## Architecture review and reuse

The repository uses a `src` layout, Python >=3.11, snake_case modules and
parameters, keyword-only constructors, and properties returning pandas Series.
The hierarchy is `Entity -> Vertex -> Component -> Converter`. `Carrier` is an
abstract interface; concrete `Electricity` and `Heat` are supplied. Fuel currently has categories
but no concrete implementation. No AGENTS.md, CI,
formatter, type-checker configuration or additional runtime version pin was
found. Tests use `unittest` and pandas/numpy assertions. The package is found by
setuptools; no new external dependency is needed for these converters.

Responsibilities already covered by the target:

- `Entity`: UUID and label; `Vertex`: ports, system assignment and graph access.
- `Component`: costs and `_parameter_series`, including EnergySystem/RangeIndex
  validation. `EnergySystem` registers vertices and edges and provides time context.
- `Converter`: carrier normalization, distinct input/output sets, directional
  ports, UUID-keyed parameter dictionaries, finite conversion coefficients,
  nonnegative bounds and their ordering. The new classes inherit these functions.
- `normalize_numeric_input`: numeric validation and scalar/Series normalization;
  `build_time_series`: broadcasting, label alignment and missing-index errors.
- `Storage` has storage-specific efficiency checks, but no reusable part-load,
  Carnot, temperature-conversion or unit-commitment model. No such model was
  duplicated. Small private functions in `converters/_thermal.py` share only the
  new classes' reference-curve and validation logic; they are not another base class.

The three classes in `components/converters/` directly inherit `Converter`.
They retain explicit signatures and delegate infrastructure to the base class.
Imports work from that subpackage or from `btm_modelling_toolbox`.
Existing classes and their signatures remain usable. Converter gains optional
`input_offset`/`output_offset` arguments, both defaulting to zero. The shared
relation is now `flow = offset * operating + conversion * activity`.
`operating` must be supplied by a future solver (1 on, 0 off). No optimization
constraints or simulation state machine are implied.

Generic Converter still rejects negative factors. Only the affine boiler/CHP
specializations allow signed coefficients through a protected class setting:
a negative coefficient can be mathematically necessary while the actual flows
remain nonnegative over the reference operating interval. Offsets are finite,
signed powers; factors are dimensionless. Parameter mappings still use UUIDs,
since existing Carrier dataclasses are not hashable.

## Reference analysis and migration decisions

| Classification | Kept/adapted behavior |
| --- | --- |
| Required domain logic | Two-point affine fuel curve, CHP total-efficiency coupling, temperature-calibrated COP, environmental heat input, operating limits, ramp and on/off metadata. |
| Technical adaptation | Direct Converter inheritance; snake_case; properties and UUID mappings; system time index instead of an explicit timeIndex on every getter; category-based carrier roles; local temperature conversion instead of SciPy. |
| Already covered | Port creation, cost normalization, time-series construction, carrier registration and graph compatibility, scalar/Series validation and bounds checks. |
| Not carried over | Location management, `gesm`/Flow dependencies, setter dispatch through setattr, unused `fix/min/max` arguments, silent `errors='coerce'` conversions, typo `tempartureSpread`, and broken references such as `get_fuel()` and undefined public capacity attributes. |

The old files provide coefficients and metadata; they do not implement dispatch,
ramping enforcement or a unit-commitment solver. The new classes likewise do
not invent those features. Minimum up/down times remain nonnegative integer
metadata with no automatic conversion to seconds. Their exact unit and gradient
time basis were not specified in the reference; any solver must define them
explicitly. Gradients retain signed scalar lower/upper limits (defaults ±infinity).
`initial_operation` is a bool, default False. HP retains minimum-time and initial
operation metadata; its unused gradient setters are not promoted into a new
ramping model.

### HeatBoiler

The reference file names its class `Boiler`; the integrated public name is
`HeatBoiler`. It consumes one fuel carrier (GAS or LIQUID_FUEL category) and
produces one HEAT carrier. For reference powers p_min/p_max and efficiencies
eta_min/eta_max:

```
F_min = p_min / eta_min
F_max = p_max / eta_max
B = (F_max - F_min) / (p_max - p_min)
A = F_min - B * p_min
fuel = A * operating + B * delivered_heat
```

These coefficients are algebraically identical to the legacy reciprocal
expressions exposed by `get_efficiencyAbsolute/Relative`. The new properties
are `efficiency_absolute/relative`. The load axis is delivered heat; this is
the axis for which the reference formula reproduces both supplied efficiencies.
The legacy name `power_min/max_capacity` is retained for its two reference
points, and `heat_min/max_capacity` remain independent delivered-heat limits,
default 0/infinity. No new fuel-input capacity is inferred from ambiguous naming.
All power quantities must use a common energy basis and power unit, e.g. MW.
The code does not infer higher/lower heating values or convert mass fuel flow.

### CombinedHeatPower

One GAS/LIQUID_FUEL input and exactly one ELECTRICITY and one HEAT output are
required; selection uses categories, not iterable order. The same A/B calculation
uses electrical power as its load axis:

```
fuel = A * operating + B * electrical_power
heat = total_efficiency * fuel - electrical_power
     = total_efficiency * A * operating
       + (total_efficiency * B - 1) * electrical_power
```

The properties `coupling_absolute/relative` expose those heat coefficients.
Electrical and thermal limits map to Converter's output bounds. This preserves
part-load behavior, including a decreasing heat slope when appropriate, instead
of replacing it with a fixed electrical/thermal ratio.

Both affine classes accept scalar or Series reference capacities, as the old
setters did, but require `0 < p_min < p_max < infinity`. Scalar efficiencies
must lie in (0, 1]. CHP total efficiency must not be below either electrical
efficiency, so useful heat is nonnegative at both endpoints. Equal efficiencies
are valid (zero intercept); equal/zero reference powers are rejected instead of
dividing by zero. Coefficients are valid over the reference load interval; no
extrapolation or off-state constraints are silently generated. The boiler's
independent heat bounds may further restrict that interval in a solver.

### HeatPump

The actual reference requires TWO inputs: electricity and environmental heat,
plus a distinct output heat carrier. Both heat carriers use category HEAT but
must have distinct IDs. Activity W is electrical power:

```
C_ref = (T_supply_ref + spread) / (T_supply_ref - T_source_ref + 2 * spread)
k = norm_cop / C_ref
COP = k * (T_supply + spread) / (T_supply - T_source + 2 * spread)
Q_out = COP * W
Q_source = (COP - 1) * W
```

This matches the reference's adjusted Carnot formula. Nominal COP and temperatures
are scalars. Actual supply/source temperatures can be scalars or Series.
`source_temperature` is required; the additional `supply_temperature` parameter
binds the reference getter's supply-temperature argument to the Converter
coefficients. If omitted it uses the nominal supply temperature.

`temperature_unit` accepts Kelvin/K, Celsius/C/°C, Fahrenheit/F/°F (case-insensitive).
Temperatures are normalized to Kelvin. `temperature_spread` is always a Kelvin
difference (default 5), including when absolute temperatures are Fahrenheit,
matching the reference arithmetic. Temperature properties report Kelvin.
`cop` returns the configured Series; `compute_cop(supply_temperature=None,
source_temperature=None)` evaluates alternate conditions without modifying it.
Explicit overrides use the configured temperature unit. This deliberately
replaces the old getter's ambiguous unconverted arguments with one documented
unit convention.

Positive Kelvin temperatures, positive effective temperature lift and finite
COP >= 1 are required. These checks reject singular or non-heating operating
conditions that the reference could silently divide by zero or turn into negative
source heat. No additional COP cap, defrost behavior or cooling model is added.
Electrical bounds map to the electrical input and heat bounds to the output;
environmental heat has nonnegative, otherwise unlimited bounds. Flows are positive
in their port directions, so source heat plus electricity equals delivered heat.

## Properties, states and validation

Configuration is supplied through explicit keyword arguments. Series properties
require assignment to an EnergySystem with a pandas RangeIndex; before assignment
they raise RuntimeError. Invalid system/index types raise TypeError. Invalid
physical values, coefficients, bounds or incompatible parameter indices raise
ValueError. Missing runtime index labels raise KeyError. Numeric input is never
silently converted into NaN. Related Series used together in a formula must have
identical indices. The base time-series helper aligns them to the system by label.

Carrier implementations must satisfy the existing `Carrier` interface and return
the appropriate `CarrierCategory`. The built-in `Heat` characterizes a thermal
network by `temperature_level` in Celsius and `medium`; both participate in direct
connection compatibility. HeatPump temperature parameters remain its explicit
operating configuration (normalized to Kelvin), independent of Carrier metadata.
Use consistent power units for all connected carriers. Existing `variable_cost`,
`fixed_cost`, ports and registration APIs remain inherited unchanged.

For example, given suitable fuel/electricity/heat carriers:

```python
from btm_modelling_toolbox import CombinedHeatPower, EnergySystem
import pandas as pd

system = EnergySystem(time_index=pd.RangeIndex(24))
chp = CombinedHeatPower(
    input_carrier=fuel,
    output_carriers=[electricity, heat],
    power_min_efficiency=0.25,
    power_max_efficiency=0.40,
    chp_total_efficiency=0.90,
    power_min_capacity=10,
    power_max_capacity=100,
)
system.add_vertex(chp)
fuel_at_50 = chp.efficiency_absolute + chp.efficiency_relative * 50
heat_at_50 = chp.coupling_absolute + chp.coupling_relative * 50
```

## Validation

Tests use the original coefficient expressions as independent numerical oracles,
check both operating endpoints, equal efficiencies, signed coefficients,
CHP total-efficiency conservation, nominal/variable COP, temperature-unit
equivalence, COP=1, carrier roles, index errors, limits, operating metadata and
full EnergySystem/Edge connections. Generic Converter tests also exercise the
additive offset API and ensure its old negative-factor validation remains intact.

The pre-change baseline had three failing Edge tests referencing the deliberately
removed `compute_received_flow`/`compute_losses` methods. Those unrelated tests
and the removed methods are left unchanged.
