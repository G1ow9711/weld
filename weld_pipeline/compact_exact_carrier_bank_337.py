"""All original exact domains in a compact layout; no candidate filtering.

Canonical constructor, numeric methods and slot rules remain Carrier327's.
Slots remove per-instance dictionaries. Exact identities share their hash
key. Aliases freeze in place, without retaining two copies of all carriers.
"""
from dataclasses import dataclass
from .analytic_carrier_slots_327 import Carrier, exact_geometry_key
from .streaming_frozen_carrier_bank_333 import candidate_records


@dataclass(frozen=True, slots=True)
class CompactCarrier:
    ordinal: int
    weld_type: str
    kind: str
    length_mm: float
    origin_mm: object
    direction: object
    arc: object
    geometry_identity: str
    aliases: tuple
    structural_events_mm: tuple = ()

    from_record = classmethod(Carrier.from_record.__func__)
    cell_count = Carrier.cell_count
    cell_width_mm = Carrier.cell_width_mm
    slot_count = Carrier.slot_count
    slot_spec = Carrier.slot_spec
    addresses_at = Carrier.addresses_at
    point_at = Carrier.point_at
    project = Carrier.project
    summary = Carrier.summary


def compact_combined_carriers(paths, progress=lambda *args, **kwargs: None):
    by_identity, ordered, count = {}, [], 0
    for path in paths:
        for record in candidate_records(path):
            key = exact_geometry_key(record)
            if key not in by_identity:
                carrier = CompactCarrier.from_record(len(ordered), record)
                assert carrier.geometry_identity == key
                object.__setattr__(carrier, "geometry_identity", key)
                object.__setattr__(carrier, "aliases", [])
                by_identity[key] = carrier
                ordered.append(carrier)
            by_identity[key].aliases.append(record["id"])
            count += 1
            if count % 10000 == 0:
                progress("COMPACT_ALL_EXACT_CANDIDATE_DOMAINS", original_records_consumed=count,
                    exact_alias_capacity_domains=len(ordered))
    by_identity.clear()
    for carrier in ordered:
        object.__setattr__(carrier, "aliases", tuple(carrier.aliases))
    assert sum(len(carrier.aliases) for carrier in ordered) == count
    progress("COMPACT_ALL_DOMAINS_ALIASES_FROZEN", original_records_consumed=count,
        exact_alias_capacity_domains=len(ordered))
    return ordered, count
