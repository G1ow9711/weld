"""Learned joint -> finite-curve -> adaptive interval hierarchy.

PTv3 and the existing finite-surface/material encoders provide frozen features.
This module has no label, STEP, source-id, fixed TopK or geometry-NMS access.
Prediction budgets are enforced by the caller for the entire workpiece.
"""
import torch
from torch import nn


def encoder(width,hidden):
    return nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))


class HierarchicalJointModel(nn.Module):
    INTERVAL_STATES=('empty','full_single_child','terminal_partial_single_child','subdivide')

    def __init__(self,relation_features,family_features,interval_features,edge_features=6,hidden=96,weld_types=2):
        super().__init__()
        self.configuration=dict(relation_features=relation_features,family_features=family_features,
             interval_features=interval_features,edge_features=edge_features,hidden=hidden,weld_types=weld_types)
        self.relation_encoder=encoder(relation_features,hidden)
        self.relation_presence=nn.Linear(hidden,1)
        self.family_encoder=encoder(hidden+family_features,hidden)
        self.family_presence=nn.Linear(hidden,1)
        self.interval_encoder=encoder(hidden+interval_features,hidden)
        self.interval_state=nn.Linear(hidden,4)
        self.interval_bounds=nn.Linear(hidden,2)
        self.interval_direction=nn.Linear(hidden,1)
        self.weld_type=nn.Linear(hidden,weld_types)
        self.path_encoder=encoder(2*hidden+edge_features,hidden)
        # Separate semantic decisions. Shared XYZ is not a merge permission.
        self.path_relations=nn.Linear(hidden,5)

    def relations(self,features):
        latent=self.relation_encoder(features)
        return latent,self.relation_presence(latent).squeeze(-1)

    def families(self,relation_latent,family_to_relation,features):
        latent=self.family_encoder(torch.cat((relation_latent[family_to_relation],features),dim=-1))
        return latent,self.family_presence(latent).squeeze(-1)

    def intervals(self,family_latent,interval_to_family,features):
        latent=self.interval_encoder(torch.cat((family_latent[interval_to_family],features),dim=-1))
        raw=self.interval_bounds(latent)
        # Independent continuous endpoints permit both inside the same cell.
        # Do not force their ordering here: an invalid prediction must be
        # diagnosed, not silently sorted into a plausible weld segment.
        return dict(latent=latent,state_logits=self.interval_state(latent),bounds=torch.sigmoid(raw),
                    direction_logit=self.interval_direction(latent).squeeze(-1),weld_type_logits=self.weld_type(latent))

    def links(self,interval_latent,ordered_pairs,features):
        x=torch.cat((interval_latent[ordered_pairs[:,0]],interval_latent[ordered_pairs[:,1]],features),dim=-1)
        return self.path_relations(self.path_encoder(x))


def initialize_priors(model,relation_prior,family_prior,interval_state_priors):
    """Output bias from the NEW training population; no positive reweighting."""
    with torch.no_grad():
        for head,prior in ((model.relation_presence,relation_prior),(model.family_presence,family_prior)):
            if not 0<prior<1:raise ValueError('A finite nondegenerate TRAIN population prior is required')
            head.weight.zero_();head.bias.fill_(float(torch.logit(torch.tensor(prior,dtype=torch.float64))))
        priors=torch.as_tensor(interval_state_priors,dtype=torch.float64)
        if priors.shape!=(4,) or (priors<=0).any() or abs(float(priors.sum())-1)>1e-8:
            raise ValueError('Four positive state priors summing to one are required')
        model.interval_state.weight.zero_();model.interval_state.bias.copy_(priors.log())


def controls():
    saved=torch.random.get_rng_state();torch.manual_seed(1028)
    try:
        model=HierarchicalJointModel(12,10,20)
        r,rl=model.relations(torch.randn(3,12));f,fl=model.families(r,torch.tensor([0,0,1,2]),torch.randn(4,10))
        out=model.intervals(f,torch.tensor([0,1,2,3,0]),torch.randn(5,20))
        pairs=torch.tensor([[0,4],[1,2],[2,3]]);links=model.links(out['latent'],pairs,torch.randn(3,6))
        objective=rl.square().mean()+fl.square().mean()+out['state_logits'].square().mean()+links.square().mean()
        objective.backward()
        assert sum(float(p.grad.abs().sum()) for p in model.relation_encoder.parameters() if p.grad is not None)>0
        assert sum(float(p.grad.abs().sum()) for p in model.interval_encoder.parameters() if p.grad is not None)>0
        initialize_priors(model,.1,.2,[.6,.1,.1,.2])
        _,rl=model.relations(torch.randn(3,12));assert torch.allclose(rl.sigmoid(),torch.full_like(rl,.1))
        assert model.INTERVAL_STATES[-1]=='subdivide'
        return dict(path_and_interval_losses_reach_relation_encoder=True,
            new_population_prior_initialization_verified=True,zero_training_updates=True,
            shape_only_not_extraction_accuracy=True)
    finally:torch.random.set_rng_state(saved)
