# Flow dependencies

The three classes in `core/dependency.py` use Entity's UUID/label convention.
`MutuallyExclusiveFlowDependency` and `LinearFlowDependency` inherit directly
from `Dependency`; the other legacy dependency classes are not ported.

Legacy Flow objects are represented by directed Edge instances. The public
parameter names retain `flows`, with snake_case names such as `flows_lhs`,
`coefficients_rhs` and `max_count`. A single Edge or an iterable (including a
single-pass generator) is accepted. BidirectionalEdge is not one directed flow;
pass its `.edges` explicitly to refer to both directions.

## Mutual exclusion

```python
from btm_modelling_toolbox import MutuallyExclusiveFlowDependency

exclusive = MutuallyExclusiveFlowDependency(flows=[purchase, sale], max_count=1)
```

At each model time step at most `max_count` flows may be active. All may be off;
this is not an exactly-one condition. The count must be a nonnegative integer,
not a bool. Zero is allowed, and a count above the number of flows is permitted
as a redundant condition. Empty groups and duplicate IDs are rejected instead
of silently creating a no-op or deduplicating input.

There is no solver or binary indicator implementation in the repository. A
future formulation must define activity and link indicators to the actual flow
limits (possibly requiring finite capacities). No arbitrary numerical threshold
or big-M value is chosen by the data model.

## Linear flow equation

```python
from btm_modelling_toolbox import LinearFlowDependency

balance = LinearFlowDependency(
    flows_lhs=[first, second],
    flows_rhs=third,
    coefficients_lhs={first.id: 1.0, second.id: -2.0},
    coefficients_rhs={third.id: 0.5},
)
```

This declares `flow_first - 2 * flow_second == 0.5 * flow_third` at each time
step. Coefficients are complete UUID-to-real-scalar mappings. Finite positive,
zero and negative values are valid; bool, complex, Series, NaN and infinity are
rejected. Scalar numeric validation reuses `normalize_numeric_input`.

A side may be empty with an empty coefficient mapping (the zero sum); both sides
empty is rejected. Repeated edges within a side are rejected, but the same edge
may appear on both sides. Flows use the sending-end convention of Edge;
reception-side relations must include the corresponding efficiency explicitly.
The caller must choose coefficients consistent with the units of the flows.
There is no automatic unit conversion, transmission-loss adjustment, inequality,
constant term or time-dependent coefficient model.

Properties expose tuples, frozensets of IDs and read-only coefficient mappings.
Mappings are copied during construction; modifying the caller's dictionaries
cannot silently change a declared relation. The __setattr__ dispatch and mutable
sets/dictionaries of the reference are not carried over. Edges themselves remain
normal graph objects, not snapshots.

## System compatibility and current scope

`dependency.check_system_compatibility(system)` verifies membership of every
referenced edge by UUID and object identity, including its energy_system owner.
It returns False for missing/foreign edges and raises TypeError for an invalid
system argument. It does not register edges or modify the system. A base
Dependency has no references and is compatible with any valid EnergySystem.
This replaces the reference base implementation's unconditional True for the
flow specializations.

The classes declare constraints without evaluating or solving flows. Register
edges first, then dependencies:

```python
system.add_dependency(balance)
system.add_dependencies([exclusive])  # iterable; the complete batch is validated first
assert system.get_dependency(balance.id) is balance
assert system.get_element(balance.id) is balance
assert balance.energy_system is system
```

`system.dependencies` is a read-only live UUID mapping. IDs are unique across
vertices, edges and dependencies. Dependencies belong to one system at a time;
registration rejects foreign ownership and unregistered/mismatched edge references.
`remove_dependency(id)` unregisters the constraint, clears its system ownership,
and leaves all vertices, edges and connections untouched.

Deleting an edge referenced by any registered dependency raises ValueError listing
the blocking dependency labels and IDs. Remove ALL referencing dependencies
explicitly before deleting the edge. This policy also applies to
`remove_vertex(id, cascade=True)`: every connected edge is checked before any edge
is disconnected, so a blocked cascade leaves the graph intact. Unregistered
standalone dependency objects do not prevent system graph changes; their
compatibility can be checked again before registration. No dependency is silently
removed or edited as a side effect of deleting graph objects.

Time-varying coefficients, inequalities and solver integration remain separate
possible extensions.
