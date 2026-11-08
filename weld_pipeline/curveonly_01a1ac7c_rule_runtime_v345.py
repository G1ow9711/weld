"""R35 actual neural forward followed by finite structural rules and export."""
from . import curveonly_01a1ac7c_direction_runtime_v261 as backbone
from . import curveonly_01a1ac7c_preserved_gaps_v342 as rules
from . import curveonly_01a1ac7c_event_interval_model_v304 as geometry
from . import curveonly_01a1ac7c_runtime_v51 as exporter
from . import curveonly_01a1ac7c_corner_publication_v256 as corner
from . import curveonly_01a1ac7c_arc_publication_v6 as publication
from . import curveonly_01a1ac7c as cad


class Proposals:
    def __init__(self, proposals, timing):
        self.proposals, self.last_timing = proposals, timing

    def predict(self, features, packet):
        return self.proposals


def decode(features, packet, model, solids=None, inventory=None):
    raw = model.predict(features, packet)
    if solids and packet.get('pose', {}).get('passed'):
        if inventory is None:
            inventory = geometry.build_geometry(packet, solids)
        proposals, audit = rules.apply(raw, features, packet, inventory, solids)
    else:
        proposals, audit = raw, dict(not_applicable='NO_QUALIFIED_FINITE_STEP', no_input_removed=True)
    proposals, selection = rules.provenance.select(proposals, packet)
    old = publication._fallback
    publication._fallback = corner.fallback
    try:
        result = exporter.decode(features, packet, Proposals(proposals, model.last_timing))
    finally:
        publication._fallback = old
    result['finite_source_selection'] = selection
    result['structural_rule_v338'] = audit
    result['actual_neural_forward_then_structural_rules'] = True
    return result


def extract(input_record, model):
    """Raw XYZ, role-bound material STEP and non-label pose/unit configuration."""
    features, packet = backbone.input_curves(input_record)
    step = input_record.get('STEP')
    solids = cad.load_material(step['path']) if step and packet.get('pose', {}).get('passed') else None
    return decode(features, packet, model, solids)
