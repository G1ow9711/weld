from weld_pipeline.crossing_plane_model import CrossingPlaneContactNet, MODEL_NAME
from weld_pipeline.model_registry_residual import model_from_checkpoint as previous_factory


def model_from_checkpoint(state):
    if state.get('model_name') != MODEL_NAME:
        return previous_factory(state)
    planes = [{'axis': int(axis), 'position_mm': float(position)}
              for axis, position in zip(state['model']['plane_axes'], state['model']['cut_positions_mm'])]
    model = CrossingPlaneContactNet(state['features'], planes)
    model.load_state_dict(state['model'])
    return model
