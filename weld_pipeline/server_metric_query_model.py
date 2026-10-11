"""Joint model with an optional explicit input-only physical query scale view."""
from torch import nn
from .server_joint_reference_query_model import JointReferenceModel
class MetricQueryModel(JointReferenceModel):
 def __init__(self,columns=457):
  super().__init__()
  if columns!=457:self.trunk[0]=nn.Linear(columns,256)
