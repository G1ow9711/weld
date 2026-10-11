"""Current bounded events and separate attribute policies around frozen325."""
from weld_pipeline.normalized_count_conditioned_events_579 import PhysicalTransactionContainer
from weld_pipeline.bounded_set_attribute_head_586 import BoundedSetAttributeHead, VERSION


class BoundedPhysicalTransactionContainer(PhysicalTransactionContainer):
    def __init__(self,original115,original_hybrid,config):
        super().__init__(original115,original_hybrid,config)
        self.attributes = BoundedSetAttributeHead()
        self.path_attributes = BoundedSetAttributeHead(features=84*17,whole_path=True)
        self.config["bounded_attribute_stage"] = VERSION

    @classmethod
    def from_bundle(cls,bundle):
        new_attributes = "bounded_attribute_stage" in bundle["config"]
        if not new_attributes:
            assert bundle["branch_accepted_optimizer_updates"]==0
            assert bundle["actual_native_optimizer_step_calls_this_stage"]==0
            for name in ("attributes","path_attributes"):
                old = bundle["incremental_state"][name]
                assert all(bool((value==0).all()) for key,value in old.items()
                           if not key.startswith("association.0."))
        migration = dict(bundle)
        migration["incremental_state"] = {key:value for key,value in bundle["incremental_state"].items()
                                          if key not in ("attributes","path_attributes")}
        model = super().from_bundle(migration)
        if new_attributes:
            assert bundle["config"]["bounded_attribute_stage"]==VERSION
            for name in ("attributes","path_attributes"):
                getattr(model,name).load_state_dict(bundle["incremental_state"][name],strict=True)
        model.attributes.attach_reference(model.frozen325_attributes)
        model.path_attributes.attach_reference(model.frozen325_path_attributes)
        model.verify_frozen()
        return model
