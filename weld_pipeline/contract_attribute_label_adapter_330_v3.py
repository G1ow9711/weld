"""Preserve valid material samples serialized in either parameter direction.

Positions are bound by each sample's current-child fraction and XYZ. A parent
path reversal can leave records in descending fraction order; that does not
invalidate a material point or reverse its plate order/signed normal.
"""
from .contract_attribute_label_adapter_330_v2 import canonical_field
from .contract_attribute_label_adapter_330 import (
    canonical_material_samples as material_samples_v1,
    finite_number,
)

VERSION = "OFFLINE_CANONICAL_E16_PAYLOAD_330_v3"


def canonical_material_samples(samples, reference_key):
    if not isinstance(samples, list):
        raise ValueError("Original ordered material sample list required")
    fractions = [s.get("fraction") for s in samples if isinstance(s, dict)]
    if len(fractions) != len(samples) or not all(finite_number(f) for f in fractions):
        direction = "UNRESOLVED_PARAMETERS"
    elif len(fractions) <= 1 or all(b > a for a, b in zip(fractions, fractions[1:])):
        direction = "ASCENDING_CURRENT_CHILD_PARAMETER"
    elif all(b < a for a, b in zip(fractions, fractions[1:])):
        direction = "DESCENDING_CURRENT_CHILD_PARAMETER"
    else:
        direction = "REPEATED_OR_NONMONOTONE_PARAMETERS_REQUIRING_REVIEW"
    result = []
    for index, sample in enumerate(samples):
        row = material_samples_v1([sample], reference_key)[0]
        row["key"] = f"{reference_key}/material_sample:{index}"
        row["original_sample_index"] = index
        row["original_parameter_sequence_order"] = direction
        row["point_position_bound_by_fraction_and_XYZ_not_list_direction"] = True
        result.append(row)
    return result
