from weld_pipeline.process_plane_model import model_from_checkpoint as original_model
from weld_pipeline.surface_process_model import SurfaceProcessContactNet


def model_from_checkpoint(state):
    if state.get('model_name') != 'surface_process':
        return original_model(state)
    planes = [{'axis': int(a), 'position_mm': float(p)}
              for a, p in zip(state['model']['plane_axes'], state['model']['cut_positions_mm'])]
    model = SurfaceProcessContactNet(state['features'], planes)
    model.load_state_dict(state['model'])
    return model
