from weld_pipeline.context_surface_model import ContextSurfaceContactNet
from weld_pipeline.model_registry_research import model_from_checkpoint as previous_factory


def model_from_checkpoint(state):
    if state.get('model_name')!='context_surface_process':return previous_factory(state)
    planes=[{'axis':int(a),'position_mm':float(p)} for a,p in zip(state['model']['plane_axes'],state['model']['cut_positions_mm'])]
    model=ContextSurfaceContactNet(state['features'],planes)
    model.load_state_dict(state['model']);return model
