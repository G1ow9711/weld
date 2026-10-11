"""Offline supervision only; never imported by free prediction.

The fixed joint witness assigns legal existing slots. Labels supervise heads,
not candidate generation, sequence features, terminal windows or runtime counts.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
from .carrier_observation_sequences_328 import SequenceLayout
from .carrier_boundary_model_328 import ROLE_FIELDS

VERSION = "COMPLETE_FIXED_JOINT_SLOT_SUPERVISION_329_v1"


def category(carrier):
    return carrier.weld_type + ":" + carrier.kind


@dataclass
class SupervisedSource:
    targets: object
    parents_by_carrier: object
    roles_by_carrier: object
    positive_role_counts: object
    negative_role_counts: object
    boundary_counts: object
    children_by_category: object
    bank_categories: tuple
    reference_categories: tuple
    complete_slots: int
    original_children: int
    gap_counts: object

    @classmethod
    def construct(cls, carriers, targets, witness):
        if (targets["missing_or_process_conflict_children"] or
                targets["incomplete_source_no_unmatched_query_negative_labels"]):
            raise ValueError("This registered complete-source loss requires complete geometry truth")
        if not witness["summary"]["complete_joint_witness_exists"]:
            raise ValueError("Full frozen joint capacity witness is required before training")
        parents = defaultdict(list)
        roles = defaultdict(lambda: [set() for _ in ROLE_FIELDS])
        reserved = defaultdict(set)
        positive = defaultdict(lambda: np.zeros(len(ROLE_FIELDS), np.int64))
        slots_by_category, children, boundaries, gaps = Counter(), Counter(), Counter(), Counter()
        seen = set()
        for carrier in carriers:
            slots_by_category[category(carrier)] += carrier.slot_count
        for assignment in witness["witnesses"]:
            pi, ordinal = assignment["original_parent_index_metadata_only"], assignment["carrier_ordinal"]
            carrier, truth = carriers[ordinal], targets["parents"][pi]
            if pi in seen or assignment["exact_geometry_identity"] != carrier.geometry_identity:
                raise ValueError("Repeated parent or changed pure carrier identity")
            seen.add(pi)
            if (assignment["gaps"] != truth["gaps"] or
                    len(assignment["children"]) != truth["original_child_count"]):
                raise ValueError("Full child and adjacency truth must remain intact")
            nodes = {n["node_id"]: dict(n) for n in assignment["boundary_nodes"]}
            layout = SequenceLayout.from_carrier(carrier)
            node_slots = {}
            for identity, node in nodes.items():
                slot, q = node["slot_index"], node["q_mm"]
                spec = carrier.slot_spec(slot)
                if slot in reserved[ordinal] or not spec["lower_mm"] <= q <= spec["upper_mm"]:
                    raise ValueError("Independent parents must reserve distinct legal slots")
                reserved[ordinal].add(slot)
                layout.location(slot)
                node_slots[identity] = int(slot)
                roles[ordinal][0].add(int(slot))
            child_rows = []
            for index, refs in enumerate(assignment["children"]):
                start, end = node_slots[refs["start_node"]], node_slots[refs["end_node"]]
                gt = truth["children"][index]
                if gt["weld_type"] != carrier.weld_type or gt["original_kind"] != carrier.kind:
                    raise ValueError("Geometry category changed during supervision")
                roles[ordinal][1].add(start)
                roles[ordinal][2].add(end)
                children[category(carrier)] += 1
                child_rows.append(dict(start_slot=start, end_slot=end, original_child_index=index,
                    loss_samples_mm=np.asarray(gt["loss_samples_mm"], np.float64)))
            roles[ordinal][3].add(child_rows[0]["start_slot"])
            roles[ordinal][4].add(child_rows[-1]["end_slot"])
            boundaries[category(carrier)] += len(nodes)
            gaps.update(assignment["gaps"])
            parents[ordinal].append(dict(original_parent_index=pi, nodes=nodes, node_slots=node_slots,
                children=child_rows, gaps=list(assignment["gaps"]), is_closed=truth["is_closed"]))
        if seen != set(range(len(targets["parents"]))) or sum(children.values()) != targets["original_children"]:
            raise ValueError("Every original parent and child must be supervised")
        for ordinal, sets in roles.items():
            positive[category(carriers[ordinal])] += np.array(list(map(len, sets)), np.int64)
        negative = {key: total - positive[key] for key, total in slots_by_category.items()}
        if any(np.any(v <= 0) for v in negative.values()):
            raise ValueError("Registered background means require a nonempty denominator")
        return cls(targets, dict(parents), dict(roles), dict(positive), negative, dict(boundaries),
            dict(children), tuple(sorted(slots_by_category)), tuple(sorted(children)),
            sum(slots_by_category.values()), targets["original_children"], dict(gaps))

    def role_weights(self, carrier):
        key = category(carrier)
        negative = .5 / (len(ROLE_FIELDS) * len(self.bank_categories) * self.negative_role_counts[key])
        counts = self.positive_role_counts.get(key, np.zeros(len(ROLE_FIELDS), np.int64))
        positive = np.divide(.5, len(ROLE_FIELDS) * len(self.reference_categories) * counts,
            out=np.zeros(len(ROLE_FIELDS), np.float64), where=counts > 0)
        return negative, positive

    def child_weight(self, carrier):
        key = category(carrier)
        return 1. / (len(self.reference_categories) * self.children_by_category[key])

    def boundary_weight(self, carrier):
        key = category(carrier)
        return 1. / (len(self.reference_categories) * self.boundary_counts[key])
