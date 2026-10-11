"""Finite-face / material-hypothesis message passing conditions weld curves."""
import torch
from torch import nn
from weld_pipeline.sjg_local_curve_model_917 import LocalCurveEventModel


class SurfaceMaterialJointModel(LocalCurveEventModel):
    def __init__(self):
        super().__init__()
        self.material_encoder=nn.Sequential(nn.Linear(18,96),nn.GELU(),nn.Linear(96,96),nn.LayerNorm(96))
        self.material_updates=nn.ModuleList([nn.Sequential(nn.Linear(192,96),nn.GELU(),nn.LayerNorm(96)) for _ in range(2)])
        self.face_material_updates=nn.ModuleList([nn.Sequential(nn.Linear(192,96),nn.GELU(),nn.LayerNorm(96)) for _ in range(2)])

    def encode_structure(self,data):
        faces=self.encode_faces(data['face_x'],data['neighbors'],data['relation'])
        materials=self.material_encoder(data['material_x'])
        fi=data['membership_face'];mi=data['membership_material'];weight=data['membership_weight'][:,None]
        face_mass=faces.new_zeros((len(faces),1)).index_add(0,fi,weight)
        material_mass=materials.new_zeros((len(materials),1)).index_add(0,mi,weight)
        for update_m,update_f in zip(self.material_updates,self.face_material_updates):
            from_faces=materials.new_zeros(materials.shape).index_add(0,mi,faces[fi]*weight)/material_mass.clamp_min(1e-12)
            materials=materials+update_m(torch.cat([materials,from_faces],-1))*(material_mass>0)
            from_materials=faces.new_zeros(faces.shape).index_add(0,fi,materials[mi]*weight)/face_mass.clamp_min(1e-12)
            faces=faces+update_f(torch.cat([faces,from_materials],-1))*(face_mass>0)
        return faces,materials

    def forward(self,data):
        faces,_=self.encode_structure(data)
        nodes,inverse=torch.unique(data['nodes'],sorted=True,return_inverse=True)
        joints=self.encode_curves(faces,data['pairs'][nodes],data['geometry'][nodes])
        return self.query_events(joints[inverse],data['patch'],data['query'])


def controls():
    state=torch.random.get_rng_state();torch.manual_seed(985)
    try:
        model=SurfaceMaterialJointModel().eval()
        data=dict(face_x=torch.randn(3,46),neighbors=torch.tensor([[0,1],[1,2],[2,0]]),relation=torch.randn(3,2,4),
            material_x=torch.randn(2,18),membership_face=torch.tensor([0,1,1,2]),membership_material=torch.tensor([0,0,1,1]),
            membership_weight=torch.tensor([1.,.5,.5,1.]),nodes=torch.tensor([0,1,0]),pairs=torch.tensor([[0,1],[1,2]]),
            geometry=torch.randn(2,20),patch=torch.randn(3,16,23),query=torch.randn(3,10))
        data['geometry'][:,7]=1
        actual=model(data)
        permuted=dict(data,material_x=data['material_x'].flip(0),membership_material=1-data['membership_material'])
        changed=model(permuted)
        assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(actual,changed))
        actual[0].square().mean().backward()
        assert sum(float(p.grad.abs().sum()) for p in model.material_encoder.parameters() if p.grad is not None)>0
        empty=dict(data,membership_face=torch.empty(0,dtype=torch.long),membership_material=torch.empty(0,dtype=torch.long),membership_weight=torch.empty(0))
        left=model(empty);right=model(dict(empty,material_x=empty['material_x']+100))
        assert all(torch.allclose(a,b) for a,b in zip(left,right))
        return dict(material_ID_permutation_invariant=True,weld_loss_reaches_material_graph=True,
            missing_memberships_do_not_invent_material_context=True,optimizer_updates=0)
    finally:torch.random.set_rng_state(state)
