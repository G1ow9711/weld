from weld_pipeline.horizontal_refit_branch_model import HorizontalRefitBranchContactNet, MODEL_NAME
from weld_pipeline.model_registry_branch import model_from_checkpoint as previous_factory


def model_from_checkpoint(state):
    if state.get('model_name') != MODEL_NAME:
        return previous_factory(state)
    def planes(prefix):
        return [{'axis': int(a), 'position_mm': float(p)} for a, p in
                zip(state['model'][prefix + '.plane_axes'], state['model'][prefix + '.cut_positions_mm'])]
    model = HorizontalRefitBranchContactNet(state['features'], planes('vertical'),
        width=state.get('model_width', 64), horizontal_planes=planes('horizontal'))
    model.load_state_dict(state['model'], strict=True)
    return model
