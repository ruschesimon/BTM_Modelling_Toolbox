# Transformer

`Transformer` directly inherits `Converter` and lives in
`components/converters/transformer.py`. It requires two `Electricity` carriers
with known, finite, positive and different AC voltages in volts. The primary
side is a named side, not necessarily the higher voltage side. Both carriers
are reused for their input/output ports, without new artificial carrier IDs.

```python
import pandas as pd
from btm_modelling_toolbox import Electricity, EnergySystem, Transformer

system = EnergySystem(time_index=pd.RangeIndex(24))
transformer = Transformer(
    primary_carrier=Electricity(voltage_level=20_000),
    secondary_carrier=Electricity(voltage_level=400),
    capacity=10.,        # sending-end active power in MW
    efficiency=0.99,     # default 1; must be in (0, 1]
    no_load_loss=0.005,  # MW; default 0
)
system.add_vertex(transformer)

transformer.primary_input
transformer.primary_output
transformer.secondary_input
transformer.secondary_output
```

All four ports remain available concurrently. There is no direction setting,
mode switch or mutually exclusive port-pair constraint in this class.

All three parameters accept real scalars or pandas Series. Capacity and
no_load_loss must be finite and nonnegative. Related Series must have identical,
unique indices and are aligned to the EnergySystem RangeIndex through the
existing utilities. Their properties return Series. Costs, system ownership
and graph registration use the inherited interfaces.

Each input has conversion factor 1 and upper bound capacity; each output has
factor efficiency and upper bound efficiency*capacity. Input/output offsets
are zero. The generic Converter disjoint-ID rule remains the default for other
converters; Transformer permits both carriers on both sides to create four ports.

A separate transformer formulation must relate primary input to secondary
output and secondary input to primary output. The generic Converter's single
common-activity equation alone is not a complete bidirectional transformer
formulation. Directional coupling, any shared device capacity constraint,
operating-state decisions, possible mutual exclusion and allocation of the
no-load loss belong to that external formulation.

no_load_loss is a single device parameter, rather than a loss independently
subtracted at both outputs. For one energized transfer, a formulation can use
`P_received = efficiency*P_sent - no_load_loss*operating`. This class does not
select where to allocate that loss. The voltage ratio does not multiply power.
Capacity is active power in MW, not apparent power in MVA. Reactive power,
impedance, tap changers and quadratic copper-loss curves are not modeled.
