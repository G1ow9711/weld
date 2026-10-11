"""Auxiliary annotated-curve distance head initialized on preserved PTv3.

Distances are reference-proximity predictions, not complete weld extraction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel
from weld_pipeline.sjg_ptv3_801 import cuda_input


class ReferenceDistanceModel(SurfaceMaterialModel):
    def __init__(self):
        super().__init__()
        self.distance_head=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,block):
        feature=self.backbone(cuda_input(block)).feat
        inverse=torch.from_numpy(block['inverse']).cuda()
        log_distance=torch.sigmoid(self.distance_head(feature).squeeze(-1))*math.log1p(100.)
        return dict(log_distance=log_distance[inverse],embedding=self.embedding(feature)[inverse],normal=F.normalize(self.normal(feature),dim=-1)[inverse],boundary=self.boundary(feature).squeeze(-1)[inverse],material=self.material(feature)[inverse])


def bounded_log_distance_loss(prediction,lower_mm,upper_mm):
    low=torch.log1p(lower_mm);high=torch.log1p(upper_mm)
    return (F.relu(low-prediction).square()+F.relu(prediction-high).square()).mean()
