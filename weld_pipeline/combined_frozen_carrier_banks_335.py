"""Offline exact327 domains from all original and additional frozen records."""
from dataclasses import replace
from .streaming_frozen_carrier_bank_333 import candidate_records
from .analytic_carrier_slots_327 import Carrier,exact_geometry_key


def combined_carriers(paths):
    groups,order,count = {},[],0
    for path in paths:
        for record in candidate_records(path):
            key = exact_geometry_key(record)
            if key not in groups:
                groups[key] = [Carrier.from_record(len(order),record),[]]
                order.append(key)
            groups[key][1].append(record["id"])
            count += 1
    result = [replace(groups[key][0],aliases=tuple(groups[key][1])) for key in order]
    assert sum(len(carrier.aliases) for carrier in result) == count
    return result,count
