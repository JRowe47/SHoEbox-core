"""Immutable byte occurrences and a static/live slot dictionary."""
from dataclasses import dataclass

import numpy as np

from ._common import LINKS, SLOTS, integer


def as_bytes(value):
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype != np.uint8:
        raise TypeError("expected bytes, text, or a one-dimensional uint8 array")
    return array.tobytes()


@dataclass(frozen=True)
class ByteEvents:
    values: bytes
    links: bytes
    slots: bytes

    def __post_init__(self):
        for name in ("values", "links", "slots"):
            object.__setattr__(self, name, as_bytes(getattr(self, name)))
        if not len(self.values) == len(self.links) == len(self.slots):
            raise ValueError("event fields must have equal length")
        values, links, slots = (np.frombuffer(x, np.uint8) for x in
                                (self.values, self.links, self.slots))
        if np.any(links >= LINKS) or np.any((links == 0) & (slots != values)):
            raise ValueError("invalid link or static dictionary slot")

    @classmethod
    def static(cls, data):
        data = as_bytes(data)
        return cls(data, bytes(len(data)), data)

    @classmethod
    def stream(cls, data, *, link, slot):
        data = as_bytes(data)
        link = integer(link, "live link", 1, LINKS - 1)
        slot = integer(slot, "slot", 0, SLOTS - 1)
        return cls(data, bytes([link]) * len(data), bytes([slot]) * len(data))

    def __getitem__(self, index):
        if not isinstance(index, slice):
            raise TypeError("slice an event stream with start:stop")
        return ByteEvents(self.values[index], self.links[index], self.slots[index])


class SlotDictionary:
    """Publish a live payload atomically. Materialized occurrences keep their bytes."""

    def __init__(self):
        self._live = {}

    def publish(self, link, slot, data):
        link = integer(link, "live link", 1, LINKS - 1)
        slot = integer(slot, "slot", 0, SLOTS - 1)
        data = as_bytes(data)
        if not data:
            raise ValueError("a published payload must contain at least one byte")
        _, version = self._live.get((link, slot), (b"", 0))
        self._live[link, slot] = data, version + 1
        return version + 1

    def read(self, link, slot):
        link = integer(link, "link", 0, LINKS - 1)
        slot = integer(slot, "slot", 0, SLOTS - 1)
        return (bytes([slot]), 0) if link == 0 else self._live[link, slot]

    def events(self, addresses):
        values, links, slots = bytearray(), bytearray(), bytearray()
        for link, slot in addresses:
            data, _ = self.read(link, slot)
            values.extend(data)
            links.extend(bytes([link]) * len(data))
            slots.extend(bytes([slot]) * len(data))
        return ByteEvents(values, links, slots)
