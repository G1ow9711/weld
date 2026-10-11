from weld_pipeline.model_registry import model_from_checkpoint as base_model
from weld_pipeline.local_surface_model import LocalSurfaceProcessContactNet


def model_from_checkpoint(state):
    if state.get('model_name') != 'local_surface_process':
        return base_model(state)
    planes = [{'axis':int(a),'position_mm':float(p)} for a,p in
              zip(state['model']['plane_axes'],state['model']['cut_positions_mm'])]
    model = LocalSurfaceProcessContactNet(state['features'],planes)
    model.load_state_dict(state['model'])
    return model
