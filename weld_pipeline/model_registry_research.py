"""New experimental architectures without changing frozen legacy registries."""
from weld_pipeline.model_registry_extended import model_from_checkpoint as previous_factory
from weld_pipeline.invariant_surface_model import InvariantSurfaceContactNet


def model_from_checkpoint(state):
    if state.get('model_name')!='invariant_surface_process':return previous_factory(state)
    planes=[{'axis':int(a),'position_mm':float(p)} for a,p in
            zip(state['model']['plane_axes'],state['model']['cut_positions_mm'])]
    model=InvariantSurfaceContactNet(state['features'],planes)
    model.load_state_dict(state['model'])
    return model
