"""E3 migration for known distances above the documented 50 mm range.

The first audit found 79 wall labels stored as value=51. Preserve their raw
records; use the same greater-than-50 rule already bound in process rules 302.
"""
from copy import deepcopy

from .contract_attribute_label_adapter_330 import (
    CENSORED, finite_number, canonical_material_samples,
    canonical_field as canonical_field_v1,
)

VERSION = "OFFLINE_CANONICAL_E16_PAYLOAD_330_v2"


def canonical_field(field, raw, reference_key):
    if (isinstance(raw, dict) and raw.get("known") is True
            and field in CENSORED and raw.get("state") == "value"
            and finite_number(raw.get("value")) and raw["value"] > 50):
        migrated = deepcopy(raw)
        migrated["state"] = "out_of_range"
        result = canonical_field_v1(field, migrated, reference_key)
        result["original_record"] = deepcopy(raw)
        result["original_legacy_state_corrected_for_documented_censor"] = result["known_payload_available"]
        return result
    return canonical_field_v1(field, raw, reference_key)
