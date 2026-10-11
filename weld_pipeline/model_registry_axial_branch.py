from weld_pipeline.model_registry_branch import model_from_checkpoint as previous_factory


def model_from_checkpoint(state):
    if state.get('model_name')=='axial_branch_surface':
        return previous_factory({**state,'model_name':'typed_branch_surface'})
    return previous_factory(state)
