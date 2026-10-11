"""Apply the shared capacity and assignment contract to Source."""
from btm_modelling_toolbox import Source, PortDirection
from test_sink import SinkTests as _SinkTests


class SourceTests(_SinkTests):
    component_class = Source
    port_direction = PortDirection.OUTPUT


del _SinkTests
