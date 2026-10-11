"""Local material-relation representation conditions weld event prediction."""
import torch
from torch import nn
from weld_pipeline.sjg_local_curve_model_917 import LocalCurveEventModel

class MaterialCurveEventModel(LocalCurveEventModel):
    def __init__(self):
        super().__init__()
        self.material=nn.Sequential(nn.Linear(170,64),nn.GELU(),nn.Linear(64,16),nn.GELU())
        self.material_logit=nn.Linear(16,2)
        self.event=nn.Sequential(nn.Linear(186,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,7))

    def query_events(self,joint,patch,query):
        local=self.patch(patch).amax(1);base=torch.cat([joint,local,query],dim=-1)
        material=self.material(base);output=self.event(torch.cat([base,material],dim=-1))
        return output[:,:5],output[:,5:],self.material_logit(material)


def controls():
    state=torch.random.get_rng_state();torch.manual_seed(953)
    try:
        model=MaterialCurveEventModel()
        joint=torch.randn(4,96);patch=torch.randn(4,16,23);query=torch.randn(4,10)
        event,offset,relation=model.query_events(joint,patch,query)
        assert event.shape==(4,5) and offset.shape==(4,2) and relation.shape==(4,2)
        event.square().mean().backward()
        gradient=sum(float(p.grad.abs().sum()) for p in model.material.parameters() if p.grad is not None)
        assert gradient>0
        return dict(material_latent_conditions_weld_events=True,weld_loss_reaches_material_latent=True,
            inference_requires_no_teacher_ID=True,optimizer_updates=0)
    finally:torch.random.set_rng_state(state)
