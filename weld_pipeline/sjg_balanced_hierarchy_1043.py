"""TRAIN-standardized inputs and balanced-risk proposal scores, not posteriors."""
import math
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_hierarchical_joint_model_1028 import HierarchicalJointModel

FAMILY_BINARY_COLUMNS=(0,7,20,23,27,39,40,43,47,59,63)
RELATION_BINARY_COLUMNS=tuple(352+i for i in FAMILY_BINARY_COLUMNS)+(416,)

class BalancedHierarchicalJointModel(HierarchicalJointModel):
    score_semantics='balanced_risk_screening_score_not_weld_probability'
    def __init__(self,**configuration):
        super().__init__(**configuration)
        for kind in ('relation','family'):
            width=configuration[kind+'_features']
            self.register_buffer(kind+'_input_mean',torch.zeros(width))
            self.register_buffer(kind+'_input_scale',torch.ones(width))

    def set_normalization(self,statistics):
        with torch.no_grad():
            for kind in ('relation','family'):
                getattr(self,kind+'_input_mean').copy_(torch.tensor(statistics[kind]['applied_mean']))
                getattr(self,kind+'_input_scale').copy_(torch.tensor(statistics[kind]['applied_scale']))

    def relations(self,features):
        return super().relations((features-self.relation_input_mean)/self.relation_input_scale)

    def families(self,relation_latent,family_to_relation,features):
        return super().families(relation_latent,family_to_relation,(features-self.family_input_mean)/self.family_input_scale)

def balanced_gate(logits,target,known=None):
    if known is None:known=torch.ones_like(target,dtype=torch.bool)
    positive=known&(target==1);negative=known&(target==0)
    terms=[]
    if positive.any():terms.append(F.binary_cross_entropy_with_logits(logits[positive],target[positive]))
    if negative.any():terms.append(F.binary_cross_entropy_with_logits(logits[negative],target[negative]))
    return sum(terms)/(len(terms)*math.log(2)) if terms else logits.sum()*0

def load_model(state):
    cls=BalancedHierarchicalJointModel if state.get('architecture')=='balanced_normalized_hierarchy' else HierarchicalJointModel
    model=cls(**state['configuration']);model.load_state_dict(state['state_dict']);return model

def controls():
    z=torch.zeros(5,requires_grad=True);y=torch.tensor([1.,0.,0.,0.,-1.]);known=y>=0
    loss=balanced_gate(z,y,known);g=torch.autograd.grad(loss,z)[0]
    assert abs(float(loss)-1)<1e-6 and g[0]<0 and bool((g[1:4]>0).all()) and g[4]==0
    assert torch.allclose(-g[0],g[1:4].sum())
    z=torch.zeros(3,requires_grad=True);loss=balanced_gate(z,torch.zeros(3));g=torch.autograd.grad(loss,z)[0]
    assert abs(float(loss)-1)<1e-6 and bool((g>0).all())
    return dict(positive_negative_gradients_correct=True,class_total_gradients_equal_at_zero=True,unknown_mask_preserved=True,
        empty_positive_case_valid=True,optimizer_updates=0)
