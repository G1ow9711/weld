"""Offline complete-source joint witness; never a model accuracy evaluator."""
from collections import Counter
from .analytic_carrier_slots_327 import assemble_parent, analytic_whole_curve_certificate
from .live_json_snapshot_282 import write_json_snapshot as dump
from .world_set_training_285 import utc
from .reports.analytic_candidate_route_327_20261005.construct_joint_carrier_witness_327 import parent_options, assign_joint


def complete_source_witness(carriers, target, directory, metadata, progress):
    parents = target["parents"]
    assert sum(len(parent["children"]) for parent in parents) == target["original_children"]
    options = []
    for index, parent in enumerate(parents):
        options.append(parent_options(carriers, parent))
        if index % 5 == 0 or index + 1 == len(parents):
            progress("ALL_ORIGINAL_PARENTS_CONTINUOUS_JOINT_OPTIONS", complete_current_source_parents=index + 1,
                original_source_parents=len(parents), original_source_G=target["original_children"])
    success, assigned, visits = assign_joint(options)
    witnesses, reserved, categories = [], set(), Counter()
    for index in sorted(assigned):
        parent, chosen = parents[index], assigned[index]
        option, nodes = chosen["option"], chosen["nodes"]
        carrier = option["carrier"]
        assembly = assemble_parent(carrier, nodes, option["children"], parent["gaps"], parent["is_closed"])
        for node in nodes:
            address = (carrier.ordinal, node["slot_index"])
            assert address not in reserved
            reserved.add(address)
        certificates = []
        for child, prediction in zip(parent["children"], assembly["segments"]):
            certificate = analytic_whole_curve_certificate(prediction, child["world_representation"])
            assert certificate["ordered_endpoint_maximum_mm"] <= 3.
            assert certificate["continuous_bidirectional_upper_bound_mm"] <= 3.
            assert child["weld_type"] != "vertical" or prediction["points"][-1][2] < prediction["points"][0][2]
            categories[child["weld_type"] + ":" + child["original_kind"]] += 1
            certificates.append(certificate)
        witnesses.append(dict(original_parent_index_metadata_only=index,
            original_parent_id_metadata_only=parent["original_parent_id"],
            original_child_ids_metadata_only=[c["original_child_id"] for c in parent["children"]],
            carrier_ordinal=carrier.ordinal, exact_geometry_identity=carrier.geometry_identity,
            serialization_aliases=list(carrier.aliases), boundary_nodes=nodes,
            children=option["children"], gaps=parent["gaps"], assembled_parent=assembly,
            certificates=certificates, residual_parameters="ZERO", GT_witness_not_runtime_input=True))
    covered = sum(len(witness["children"]) for witness in witnesses)
    assert not success or covered == target["original_children"]
    missing = [dict(original_parent_index_metadata_only=index,
        original_parent_id_metadata_only=parent["original_parent_id"], original_child_count=len(parent["children"]),
        status="NOT_PROVED") for index, parent in enumerate(parents) if not options[index]]
    summary = dict(metadata, utc=utc(), original_G=target["original_children"], original_parents=len(parents),
        jointly_witnessed_G=covered, complete_joint_witness_exists=success,
        exact_alias_capacity_domains=len(carriers), parent_option_counts=[len(row) for row in options],
        missing_continuous_options=missing, search_visits=visits, category_witnesses=dict(categories),
        maximum_continuous_bidirectional_bound_mm=max((certificate["continuous_bidirectional_upper_bound_mm"]
            for witness in witnesses for certificate in witness["certificates"]), default=None),
        NN_forwards=0, actual_optimizer_updates=0, full_contract_pass=False)
    result = dict(summary=summary, witnesses=witnesses, unique_reserved_slots=len(reserved))
    dump(directory / "joint_witness.json", result)
    return result
