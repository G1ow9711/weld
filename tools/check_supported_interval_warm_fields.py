"""Actual complete-view control for inherited fields on selected input intervals."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from tools.probe_multiview_visual_assistance import load_xyz
from weld_pipeline.server_cached_local_query_training import CachedLocalQueryInput, CachedLocalTrainingTensors
from weld_pipeline.server_interval_endpoint_checkpoint import verify_complete_interval_endpoint_fit
from weld_pipeline.server_memoized_training_queries import MemoizedRawQueryNeighborhood, MemoizedTrainingAssignments
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_supported_interval_weld_model import SupportedIntervalWeldModel, known_field_losses
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, write


def main(args):
    torch.set_num_threads(2);torch.manual_seed(1031)
    base=ROOT/'server_rebuild/full_dataset_v2';out=base/local_name(args.experiment_name);out.mkdir(exist_ok=False)
    source=(ROOT/args.checkpoint).resolve(strict=True)
    if digest(source)!=args.checkpoint_sha256:raise ValueError('Complete warm checkpoint changed')
    state=torch.load(source,map_location='cpu',weights_only=False)
    verify_complete_interval_endpoint_fit(ROOT,base,state,args.checkpoint_sha256,source.parent/'receipt.json')
    dataset=PartitionedFullShardDataset(ROOT,base);job,version=dataset.versions[args.version]
    xyz,_=load_xyz(dataset.jobs[job]);inp=dataset.input_case(job);view=dataset.supervision(args.version)
    folder=base/'full_task_supervision_v1'/job/args.version;verify(folder,dataset,version)
    with np.load(folder/'owner_witnesses.npz') as z:witnesses={name:z[name] for name in z.files}
    with threadpool_limits(limits=2):
        sample=MemoizedTrainingAssignments(inp).build(view,witnesses,1031)
        cached=CachedLocalQueryInput(inp,sample,MemoizedRawQueryNeighborhood(xyz,inp))
    converter=CachedLocalTrainingTensors(state['mean'],state['std'],'cuda')
    model=SupportedIntervalWeldModel().cuda();model.inherit_complete_query(state['state_dict']);model.eval()
    batch=converter.batch(cached,sample);count=len(sample['features'])
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    with torch.no_grad():
        hidden=model.encode_native(batch,torch.zeros(count,60,device='cuda'))
        old_hidden=model.query.encode_queries(**batch);assert torch.equal(hidden,old_hidden)
        ratio=converter.tensor(sample['features'][:,456]);original=model.query.predict(old_hidden,ratio)
        event=torch.zeros(count,128,device='cuda');bounds=torch.tensor([[0.,1.]],device='cuda').expand(count,-1)
        predicted=model.predict_interval(hidden,event,event,batch['geometry'],bounds,ratio,native=True)
        fields=['weld_type','reference_score','direction','kind','closed','attribute_states','attribute_mm','frame_axes','frame_offsets']
        for field in fields:
            if not torch.equal(predicted[field],original[field]):raise ValueError('Warm field changed: '+field)
    model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    history=[]
    targets={k:converter.tensor(v) for k,v in sample['targets'].items()}
    for step in range(2):
        hidden=model.encode_native(batch,torch.zeros(count,60,device='cuda'))
        predicted=model.predict_interval(hidden,event,event,batch['geometry'],bounds,ratio,native=True)
        losses=known_field_losses(predicted,targets)
        losses['parent_closure']=F.cross_entropy(model.parent_logits(predicted['hidden'],sample['parent_children']),converter.tensor(sample['parent_closed']))
        if len(sample['link_left']):
            link=model.reference_link_logits(predicted['hidden'],converter.tensor(sample['link_left']),converter.tensor(sample['link_right']),converter.tensor(sample['link_relative']))
            losses['reference_links']=F.cross_entropy(link,converter.tensor(sample['link_labels']))
        loss=sum(losses.values());assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
        history.append(dict(loss=float(loss.detach()),components={k:float(v.detach()) for k,v in losses.items()}))
    if any(not torch.equal(frozen[n],p) for n,p in model.named_parameters() if n in frozen):
        raise ValueError('Frozen legacy or unused geometry parameters changed')
    write(out/'receipt.json',dict(status='PASS_COMPLETE_VIEW_WARM_FIELDS_ON_FINITE_INPUT_INTERVALS',
        version=args.version,children=count,known_attributes=int(view.label_known.sum()),known_frames=int(view.frame_algebraically_known.sum()),
        checkpoint_sha256=args.checkpoint_sha256,preserved_fields=fields,actual_updates=2,history=history,
        source_sha256=digest(ROOT/'weld_pipeline/server_supported_interval_weld_model.py'),
        zero_image_residual_used_to_check_warm_preservation=True,endpoints_chosen_as_full_input_extent_for_control=True,
        control_weights_not_promoted=True,full_model_training_or_accuracy_claimed=False,official_validation_or_test_reads=0))
    print('SUPPORTED_INTERVAL_WARM_FIELDS_PASS',count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment-name',required=True);p.add_argument('--version',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--checkpoint-sha256',required=True);main(p.parse_args())
