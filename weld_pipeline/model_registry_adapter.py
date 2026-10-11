from weld_pipeline.arc_adapter_model import ArcAdapterContactNet,MODEL_NAME
from weld_pipeline.model_registry_context import model_from_checkpoint as previous_factory


def model_from_checkpoint(state):
    if state.get('model_name')!=MODEL_NAME:return previous_factory(state)
    planes=[{'axis':int(a),'position_mm':float(p)} for a,p in zip(state['model']['plane_axes'],state['model']['cut_positions_mm'])]
    model=ArcAdapterContactNet(state['features'],planes);model.load_state_dict(state['model']);return model
