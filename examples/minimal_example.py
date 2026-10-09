from btm_modelling_toolbox import (
    BidirectionalEdge,
    Edge,
    Electricity,
    Port,
    PortDirection,
    Vertex,
)


# -------------------- Carrier --------------------

electricity = Electricity(
    label="Electricity",
)

assert electricity.label == "Electricity"


# -------------------- Vertices --------------------

grid = Vertex(
    label="Grid",
)

consumer = Vertex(
    label="Consumer",
)

assert grid.label == "Grid"
assert consumer.label == "Consumer"


# -------------------- Unidirectional Edge --------------------

grid_output = Port(
    label="Grid output",
    vertex=grid,
    carrier=electricity,
    direction=PortDirection.OUTPUT,
)

consumer_input = Port(
    label="Consumer input",
    vertex=consumer,
    carrier=electricity,
    direction=PortDirection.INPUT,
)

grid_to_consumer = Edge(
    label="Grid to consumer",
    source_port=grid_output,
    sink_port=consumer_input,
)


# Check edge properties

assert grid_to_consumer.source_port is grid_output
assert grid_to_consumer.sink_port is consumer_input

assert grid_to_consumer.source_vertex is grid
assert grid_to_consumer.sink_vertex is consumer

assert grid_to_consumer.carrier is electricity


# Check port connection

assert grid_output.is_connected
assert consumer_input.is_connected

assert grid_output.edge is grid_to_consumer
assert consumer_input.edge is grid_to_consumer


# -------------------- Bidirectional Edge --------------------

vertex_a = Vertex(
    label="Vertex A",
)

vertex_b = Vertex(
    label="Vertex B",
)

a_output = Port(
    label="A output",
    vertex=vertex_a,
    carrier=electricity,
    direction=PortDirection.OUTPUT,
)

a_input = Port(
    label="A input",
    vertex=vertex_a,
    carrier=electricity,
    direction=PortDirection.INPUT,
)

b_output = Port(
    label="B output",
    vertex=vertex_b,
    carrier=electricity,
    direction=PortDirection.OUTPUT,
)

b_input = Port(
    label="B input",
    vertex=vertex_b,
    carrier=electricity,
    direction=PortDirection.INPUT,
)

bidirectional_edge = BidirectionalEdge(
    label="A <-> B",
    forward_source_port=a_output,
    forward_sink_port=b_input,
    backward_source_port=b_output,
    backward_sink_port=a_input,
)


# Check contained edges

assert bidirectional_edge.forward_edge.source_port is a_output
assert bidirectional_edge.forward_edge.sink_port is b_input

assert bidirectional_edge.backward_edge.source_port is b_output
assert bidirectional_edge.backward_edge.sink_port is a_input


# Check connections

assert a_output.edge is bidirectional_edge.forward_edge
assert b_input.edge is bidirectional_edge.forward_edge

assert b_output.edge is bidirectional_edge.backward_edge
assert a_input.edge is bidirectional_edge.backward_edge


# -------------------- Output --------------------

print("All tests passed.")

print()
print("Unidirectional edge:")
print(
    f"{grid_to_consumer.source_vertex.label} "
    f"-> "
    f"{grid_to_consumer.sink_vertex.label}"
)

print()
print("Bidirectional edge:")
print(
    f"{bidirectional_edge.forward_edge.source_vertex.label} "
    f"-> "
    f"{bidirectional_edge.forward_edge.sink_vertex.label}"
)
print(
    f"{bidirectional_edge.backward_edge.source_vertex.label} "
    f"-> "
    f"{bidirectional_edge.backward_edge.sink_vertex.label}"
)