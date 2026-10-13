"""Register a source, a bus, a sink, and the connecting edges."""

from datetime import datetime, timedelta, timezone

from btm_modelling_toolbox import Bus, Edge, Electricity, EnergySystem, Sink, Source


system = EnergySystem(label='Supply system')
start = datetime(2026, 1, 1, tzinfo=timezone.utc)
system.set_temporal_context(start, start + timedelta(days=1), timedelta(hours=1))
carrier = Electricity(label='Electricity')
source = Source(label='Grid', carrier=carrier, variable_cost=0.2)
bus = Bus(label='Electricity bus', carrier=carrier)
sink = Sink(label='Demand', carrier=carrier, max_capacity=10)
system.add_vertices([source, bus, sink])
system.add_edges([
    Edge(source_port=source.output_ports[0], sink_port=bus.input_ports[0]),
    Edge(source_port=bus.output_ports[0], sink_port=sink.input_ports[0]),
])
assert len(system.vertices) == 3
assert len(system.edges) == 2
assert len(source.variable_cost) == 24
assert system.tau == 1
print('Energy system example passed: 3 vertices, 2 edges, 24 hourly steps.')
