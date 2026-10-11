"""Contact relations ignore world translation; graph gradients reach message layers."""
import json
from pathlib import Path

import numpy as np
import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.joint_relation_heatmaps import make_graph, JointRelationHeatmaps, LOCAL_CHANNELS


def main():
    torch.set_num_threads(2);torch.manual_seed(143)
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    name='224-GR24A1';piece=load_piece(split['piece_roots'][name],verify=True)
    xyz=read_xyz_pcd(piece['record']['input'])*1000.
    with np.load(Path(split['structure_roots'][name])/(name+'.npz')) as context:
        structural,process=context['structural'],context['process']
    original=make_graph(xyz,piece['sequences'],structural,process,FiniteContactPlates(xyz))
    delta=np.array([17.,-23.,13.])
    translated=[{**sequence,'xyz':sequence['xyz'].astype(float)+delta} for sequence in piece['sequences']]
    shifted=make_graph(xyz+delta,translated,structural,process,FiniteContactPlates(xyz+delta))
    comparisons={}
    for key in ['nodes','edge_features']:
        comparisons[key]=dict(maximum_absolute_error=float(np.abs(original[key]-shifted[key]).max()),
                              passed=bool(np.allclose(original[key],shifted[key],atol=2e-5,rtol=1e-5)))
    for key in ['edge_index','edge_type','sequence_indices']:
        comparisons[key]=dict(passed=bool(np.array_equal(original[key],shifted[key])))
    model=JointRelationHeatmaps(True).cuda()
    with torch.no_grad():
        model.condition.weight.normal_(0.,.05)
    args=[torch.from_numpy(original[key]).cuda() for key in ['nodes','edge_index','edge_type','edge_features']]
    selected=next(i for i in original['sequence_indices'] if piece['sequences'][int(i)]['type']=='horizontal')
    node=int(np.flatnonzero(original['sequence_indices']==selected)[0])
    seq=original['canonical'][int(selected)][0]
    points=torch.from_numpy(seq['features'][:256,LOCAL_CHANNELS].T.copy()[None]).cuda()
    logits=model(points,torch.tensor([node],device='cuda'),*args)
    loss=logits.square().mean();loss.backward()
    gradient=sum(float(layer.weight.grad.abs().sum()) for layers in model.relations for layer in layers if layer.weight.grad is not None)
    result=dict(source_sha256=sha256('weld_pipeline/joint_relation_heatmaps.py'),comparisons=comparisons,
                relation_message_gradient_l1=gradient,gradient_finite=bool(np.isfinite(gradient)),
                passed=all(value['passed'] for value in comparisons.values()) and np.isfinite(gradient) and gradient>0.,
                synthetic_parameter_initialization_only=True,full_contract_pass=False)
    out=Path('weld_pipeline/reports/joint_relation_geometry_20261001.json')
    if out.exists():
        raise ValueError('Fixed relation physical receipt exists')
    out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print('RELATION_GEOMETRY',result,flush=True)
    if not result['passed']:
        raise ValueError('Relation geometry/gradient checks failed')


if __name__=='__main__':
    main()
