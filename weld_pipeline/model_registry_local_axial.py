from weld_pipeline.local_axial_branch_model import LocalAxialBranchContactNet, MODEL_NAME
from weld_pipeline.model_registry_axial_branch import model_from_checkpoint as previous_factory


def model_from_checkpoint(state):
    if state.get('model_name') != MODEL_NAME:
        return previous_factory(state)
    axes = state['model']['horizontal.plane_axes']
    positions = state['model']['horizontal.cut_positions_mm']
    planes = [{'axis': int(a), 'position_mm': float(p)} for a, p in zip(axes, positions)]
    model = LocalAxialBranchContactNet(state['features'], [],
        width=state.get('model_width', 64), horizontal_planes=planes)
    model.load_state_dict(state['model'], strict=True)
    return model
