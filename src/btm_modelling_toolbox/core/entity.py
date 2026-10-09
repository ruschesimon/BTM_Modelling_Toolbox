# -------------------- Public Imports --------------------

from __future__ import annotations
from dataclasses import dataclass, field
from uuid import UUID, uuid4



# -------------------- Class Definition --------------------

@dataclass(slots=True, kw_only=True)
class Entity:
    id: UUID = field(
        default_factory=uuid4,
        init=False,
        repr=True,
        compare=True,
    )
    label: str | None = None

    def __post_init__(self) -> None:
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("label must be a string or None.")

        if self.label is None:
            self.label = f"{self.__class__.__name__} {self.id}"