"""XYZ-only model with reachable scalar association and unchanged weight names."""
from weld_pipeline.bounded_physical_transaction_model_589 import BoundedPhysicalTransactionContainer
from weld_pipeline.reachable_set_attribute_head_637 import BoundedSetAttributeHead,VERSION

class ReachablePhysicalContainer(BoundedPhysicalTransactionContainer):
    def __init__(self,original115,original_hybrid,config):
        super().__init__(original115,original_hybrid,config)
        self.attributes=BoundedSetAttributeHead()
        self.path_attributes=BoundedSetAttributeHead(features=84*17,whole_path=True)
        self.config["reachable_attribute_stage"]=VERSION

