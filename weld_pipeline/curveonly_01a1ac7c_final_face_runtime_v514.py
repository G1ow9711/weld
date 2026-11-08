"""Current complete-curve model with final unchanged source/face review."""
from . import curveonly_01a1ac7c_finite_curve_runtime_v505 as baseline
from . import curveonly_01a1ac7c_final_face_rules_v513 as rules


def forward(features, packet, model, solids, inventory, xyz, generated=None):
    before, packet, features, inventory = baseline.forward(
        features, packet, model, solids, inventory, xyz, generated)
    result = rules.correct(before, features, packet, inventory, solids, model.paired.source_expert)
    return result, packet, features, inventory


def decode_existing(features, packet, model, solids, inventory, return_before=False):
    before = baseline.decode_existing(features, packet, model, solids, inventory)
    result = rules.correct(before, features, packet, inventory, solids, model.paired.source_expert)
    return (result, before) if return_before else result


def extract(record, model):
    features, packet = baseline.sources.baseline.root.backbone.input_curves(record)
    step = record.get('STEP')
    solids = baseline.sources.observations.cad.load_material(step['path']) if step and packet.get('pose', {}).get('passed') else None
    inventory = baseline.sources.geometry.build_geometry(packet, solids) if solids else baseline.sources.geometry.native_only(packet)
    xyz = baseline.sources.observations.cad.read_xyz(record['raw']['path'])
    result, _, _, _ = forward(features, packet, model, solids, inventory, xyz)
    return result
