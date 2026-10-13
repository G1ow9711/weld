"""Rebuild only parent relations from authenticated complete frozen children."""
import argparse
import gzip
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from tools.export_full_dataset_split_view import local_name
from tools.prepare_supported_interval_parent_pool import save_gzip
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_supported_directed_parent_pool import build_supported_parent_pool
from weld_pipeline.server_supported_interval_weld_model import SupportedIntervalWeldModel
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, read, write


def run(args):
    torch.set_num_threads(2);base=ROOT/'server_rebuild/full_dataset_v2'
    source=base/local_name(args.source_pool);out=base/local_name(args.experiment_name);out.mkdir(exist_ok=False)
    dataset=PartitionedFullShardDataset(ROOT,base);key=args.job_key
    if key not in dataset.jobs:raise ValueError('An approved complete input job is required')
    old=source/'pools'/key;freeze=read(old/'child_freeze.json');protocol=read(source/'protocol.json')
    if (freeze['geometry_GT_reads']!=0 or not freeze['audit']['all_input_profiles_and_queries_visited']
            or protocol['registry_sha256']!=dataset.registry_sha256):raise ValueError('Complete input-only child freeze required')
    folder=out/'pools'/key;folder.mkdir(parents=True)
    for name,sha in freeze['files'].items():
        if digest(old/name)!=sha:raise ValueError('Frozen complete child file changed')
        os.link(old/name,folder/name)
    os.link(old/'child_freeze.json',folder/'child_freeze.json')
    checkpoint=source/'query_checkpoint.pt'
    if digest(checkpoint)!=protocol['query_checkpoint_sha256']:raise ValueError('Frozen query changed')
    os.link(checkpoint,out/'query_checkpoint.pt')
    settings=dict(protocol,parent_resume_source=str(source.relative_to(ROOT)),
        parent_resume_child_freeze_sha256=digest(old/'child_freeze.json'),
        max_parents=args.maximum_parents,max_link_edges=args.maximum_link_edges,
        parent_resume_source_sha256=digest(__file__),model_geometry_or_fields_changed=False)
    write(out/'protocol.json',settings)
    with gzip.open(folder/'child_proposals.json.gz','rt') as f:children=json.load(f)
    with gzip.open(folder/'shared_events.json.gz','rt') as f:events=json.load(f)
    with np.load(folder/'child_inputs.npz') as z:hidden=z['child_tokens'][:,:128].copy()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=SupportedIntervalWeldModel().eval();model.load_state_dict(state['state_dict'],strict=True)
    priors=FrozenRun446Priors(ROOT)
    write(folder/'progress.json',dict(status='BUILDING_ALL_PARENT_RELATIONS_FROM_COMPLETE_FROZEN_CHILDREN',pid=os.getpid(),children=len(children),maximum_link_edges=args.maximum_link_edges))
    try:
        with threadpool_limits(limits=2):
            pool=build_supported_parent_pool(children,hidden,priors,model.directed_link,events,
                max_parents=args.maximum_parents,max_link_edges=args.maximum_link_edges)
        save_gzip(folder/'parent_pool.json.gz',pool['parents'])
        np.savez_compressed(folder/'learned_links.npz',**{k:pool[k] for k in ['link_pairs','link_features','link_prior_probabilities','learned_link_logits','link_legal']})
        write(folder/'input_audit.json',dict(query=freeze['audit'],parent=pool['audit']))
        names=['child_proposals.json.gz','child_inputs.npz','shared_events.json.gz','child_freeze.json','parent_pool.json.gz','learned_links.npz','input_audit.json']
        write(folder/'receipt.json',dict(status='ACTUAL_PREDICTION_PARENT_POOL_FROZEN_BEFORE_TRAIN_REFERENCE_READ',
            job_key=key,registry_sha256=dataset.registry_sha256,input_receipt_sha256=digest(dataset.shards/key/'receipt.json'),
            query_checkpoint_sha256=protocol['query_checkpoint_sha256'],query_checkpoint_optimizer_steps=state['optimizer_steps'],
            prior_run446_sha256=priors.checkpoint_sha256,geometry_GT_reads=0,STEP_reads=0,
            competition_mode='lazy',competition_neighbors=16,max_active_competition_pairs=4_000_000,
            children=len(children),parents=len(pool['parents']),queries_scored=freeze['audit']['counts']['queries_scored'],
            complete_first_TRAIN_pass=protocol['complete_first_TRAIN_pass'],partial_training_control=protocol['partial_training_control'],
            files={n:digest(folder/n) for n in names},complete_output_selection_pending=True,full_contract_pass=False,
            frozen_child_geometry_and_fields_reused_unchanged=True))
        print('RESUMED_COMPLETE_PARENT_POOL',len(children),len(pool['parents']),flush=True)
    except BaseException as exc:
        write(folder/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),complete_output_claimed=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-pool',required=True);p.add_argument('--experiment-name',required=True)
    p.add_argument('--job-key',required=True);p.add_argument('--maximum-parents',type=int,default=2_000_000)
    p.add_argument('--maximum-link-edges',type=int,default=12_000_000);run(p.parse_args())
