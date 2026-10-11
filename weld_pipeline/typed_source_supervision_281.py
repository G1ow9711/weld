"""Full typed loss targets. Never used to build inference inputs or outputs."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from .raw_triangle_neural_features_279 import file_sha
from .typed_prediction_codec_281 import (
    encode_teacher, restore_teacher_representation, SLOT_COLUMNS, FIELDS,
    CATEGORY_NAMES, NUMERIC_NAMES, RELATIONS, DOMAIN_KINDS,
)
from .complete_mixed_domain_metadata_281 import CompleteMixedDomainMetadata

VERSION = "FULL_SOURCE_MIXED_DOMAIN_TYPED_LOSS_TARGETS_281_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def signature(graph):
    value = dict(graph)
    for name in FIELDS:
        value[name] = sorted(graph[name], key=lambda row: (
            row["slot"]["anchor_id"], row["slot"]["role"], row["slot"]["local_slot_ordinal"]))
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compile_loss_targets(root, source_entry, teacher_entry, output):
    """Remap every supervised domain into its complete immutable input registry."""
    root, output = Path(root), Path(output)
    if source_entry["source_index"] != teacher_entry["source_index"]:
        raise ValueError("Input and teacher refer to different original workpieces")
    if source_entry["original_pcd_sha256"] != teacher_entry["original_pcd_sha256"]:
        raise ValueError("Input and teacher have different original PCD identities")
    domains = CompleteMixedDomainMetadata(root, source_entry)
    directory = root / teacher_entry["teacher_directory"]
    names = ("receipt.json", "actual_joint_source_parent_graph.json", "full_original_child_outcomes.json", "joint_original_child_mapping.json")
    bindings = teacher_entry["frozen_teacher_artifact_sha256"]
    for name in names:
        if file_sha(directory / name) != bindings[name]:
            raise ValueError("Immutable complete original teacher artifact changed: " + name)
    receipt, graph, outcomes, mapping = [read(directory / name) for name in names]
    if not receipt["all_original_denominators_retained"] or len(outcomes) != receipt["original_children"]:
        raise ValueError("Full original child denominator required, including unknowns")
    outcome_ids = [row["truth_child_id"] for row in outcomes]
    mapped_ids = [row["truth_child_id"] for row in mapping]
    if len(set(outcome_ids)) != len(outcome_ids) or len(set(mapped_ids)) != len(mapped_ids):
        raise ValueError("Duplicate original supervision identity")
    if not set(mapped_ids).issubset(outcome_ids) or len(mapping) != len(graph["children"]):
        raise ValueError("Represented children must each retain an original mapping")
    pack = encode_teacher(graph, domains.anchor_count, domains.index)
    restored = restore_teacher_representation(pack, domains.record)
    if signature(restored) != signature(graph):
        raise ValueError("Full-domain remapping lost an original teacher field or relationship")
    # Exact rational values stay in this temporary teacher-only schema check.
    # The saved loss arrays contain ordinary numeric values and known masks.
    arrays = {name: value for name, value in pack.items() if isinstance(value, np.ndarray)}
    links = arrays["domain_links"]
    if len(links) and ((links[:, 1] < 0).any() or (links[:, 1] >= domains.total_domain_count).any()):
        raise ValueError("Supervision pointer outside complete original candidate pool")
    output.mkdir(exist_ok=False)
    np.savez(output / "complete_typed_loss_arrays.npz", **arrays)
    complete = len(mapped_ids) == len(outcome_ids)
    occupancy = np.lib.format.open_memmap(output / "full_original_anchor_role_targets.npy", mode="w+", dtype=np.int8,
                                         shape=(domains.anchor_count, len(SLOT_COLUMNS)))
    occupancy[:] = 0 if complete else -1
    for anchor, column, ordinal in arrays["slots"]:
        if ordinal != SLOT_COLUMNS[int(column)][1]:
            raise ValueError("Teacher changes original local slot capacity")
        occupancy[int(anchor) - 1, int(column)] = 1
    occupancy.flush()
    del occupancy
    for name, value in (("full_original_child_outcomes.json", outcomes), ("original_child_slot_mapping.json", mapping)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    known_slots = domains.anchor_count * len(SLOT_COLUMNS)
    result = dict(
        version=VERSION, utc=datetime.now(timezone.utc).isoformat(), source_index=domains.source_index,
        original_pcd_sha256=source_entry["original_pcd_sha256"],
        source_input_receipt_sha256=source_entry["producer_receipt_sha256"],
        source_input_independent_receipt_sha256=source_entry["independent_receipt_sha256"],
        raw_triangle_receipt_sha256=source_entry["raw_triangle_receipt_sha256"],
        raw_triangle_independent_receipt_sha256=source_entry["raw_triangle_independent_receipt_sha256"],
        complete_original_anchor_count=domains.anchor_count,
        complete_legacy_carrier_domain_count=domains.legacy_domain_count,
        complete_raw_triangle_domain_count=domains.raw_triangles.count,
        complete_mixed_domain_count=domains.total_domain_count,
        original_children=len(outcomes), represented_original_children=len(mapping),
        unknown_original_children=len(outcomes) - len(mapping),
        positive_canonical_role_definitions=len(arrays["slots"]),
        all_original_role_slot_denominator=known_slots,
        complete_negative_role_supervision=complete,
        known_negative_role_slots=(known_slots - len(arrays["slots"]) if complete else 0),
        unknown_role_slots=(0 if complete else known_slots - len(arrays["slots"])),
        ordered_role_relations=len(arrays["relations"]), complete_domain_links=len(links),
        gap_groups=len(arrays["gap_classes"]), category_names=CATEGORY_NAMES,
        numeric_names=NUMERIC_NAMES, relation_names=RELATIONS, domain_link_kinds=DOMAIN_KINDS,
        represented_categories=dict(Counter(row["weld_type"] + "_" + row["world_kind"] for row in graph["children"])),
        frozen_teacher_artifact_sha256=bindings,
        supervision_only=True, neural_initial_inputs_created=False,
        exact_teacher_sidecar_saved=False, exact_graph_template_saved=False,
        source_domains_truth_selected=False, unconfirmed_material_truth_supplied=False,
        full25_original_scope=25, full_original_child_scope=1605,
        actual_optimizer_updates=0, new_model_accuracy=False, full_contract_pass=False,
        output_sha256={name: file_sha(output / name) for name in (
            "complete_typed_loss_arrays.npz", "full_original_anchor_role_targets.npy",
            "full_original_child_outcomes.json", "original_child_slot_mapping.json")},
    )
    (output / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result
