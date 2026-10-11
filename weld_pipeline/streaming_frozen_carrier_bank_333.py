"""Offline bounded JSON reader; preserve every exact327 capacity domain/alias.

Raw support rows remain in the immutable original bank. This constructor stores
geometry and all aliases for representation preflight, never for free inference.
"""
from dataclasses import replace
import json

from .analytic_carrier_slots_327 import Carrier, exact_geometry_key

VERSION = "OFFLINE_ALL_FROZEN_BANK_RECORDS_EXACT327_DOMAINS_STREAMED_333_v1"
PREFIX = '{"candidates":['


def candidate_records(path, block_chars=65536):
    decoder = json.JSONDecoder()
    with path.open("r",encoding="utf-8") as f:
        if f.read(len(PREFIX)) != PREFIX:
            raise ValueError("Exact frozen streaming bank prefix required")
        buffer, position, ended = "", 0, False
        while True:
            while True:
                while position < len(buffer) and buffer[position] in " \r\n\t,":
                    position += 1
                if position < len(buffer):
                    break
                if ended:
                    raise ValueError("Truncated candidate array")
                buffer = buffer[position:] + f.read(block_chars)
                position = 0
                ended = len(buffer) == 0
            if buffer[position] == "]":
                return
            while True:
                try:
                    record, next_position = decoder.raw_decode(buffer,position)
                    break
                except json.JSONDecodeError:
                    extra = f.read(block_chars)
                    if not extra:
                        raise ValueError("Truncated or invalid frozen candidate record")
                    buffer = buffer[position:] + extra
                    position = 0
            if not isinstance(record,dict) or record.get("weld_identity") != "UNSELECTED_GEOMETRY_HYPOTHESIS":
                raise ValueError("Every original record must remain an unselected hypothesis")
            if record.get("material_identity") != "MATERIAL_UNVERIFIED" or not isinstance(record.get("raw_support_rows"),list):
                raise ValueError("Original support references and unreviewed material required")
            yield record
            position = next_position
            if position >= block_chars:
                buffer, position = buffer[position:], 0


def streamed_carriers(path, progress=None):
    groups, order, record_count = {}, [], 0
    for record in candidate_records(path):
        key = exact_geometry_key(record)
        if key not in groups:
            groups[key] = [Carrier.from_record(len(order),record),[]]
            order.append(key)
        groups[key][1].append(record["id"])
        record_count += 1
        if progress is not None and record_count % 10000 == 0:
            progress(record_count,len(order))
    carriers = [replace(groups[key][0],aliases=tuple(groups[key][1])) for key in order]
    assert sum(len(carrier.aliases) for carrier in carriers) == record_count
    return carriers,record_count
