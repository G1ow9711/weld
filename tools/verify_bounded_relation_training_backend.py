"""Compare original complete TRAIN loading and actual eight-step fitting."""
import argparse
import copy
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tools.bounded_relation_training_backend import configure,relation_arguments
from tools.compact_endpoint_training_backend import control_arguments
from tools.train_full_predicted_parent_selector import load_job,forward_scores
from tools.train_full_recall_parent_dataset import fit_job
from weld_pipeline.server_bounded_cached_relation_prior import BoundedCachedRun446RelationPrior
from weld_pipeline.server_cached_parent_relation_prior import CachedRun446RelationPrior
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_ordered_parent_set_model import OrderedParentSetModel
from weld_pipeline.server_reference_parent_selection import greedy_nullable_set
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/full_dataset_v1'


def main(args):
    for value in [args.experiment_name,args.pool_name,args.job]:
        if not value or value in ('.','..') or Path(value).name!=value:
            raise ValueError('Independent local control names required')
    out=BASE/args.experiment_name;out.mkdir(exist_ok=False)
    started=time.perf_counter()
    try:
        loader,guards,metadata=configure(args.reference_control,args.synthetic_control,
            args.bounded_geometry_control,args.bounded_competition_control,out)
        sources={n:digest(ROOT/n) for n in guards+['tools/verify_bounded_relation_training_backend.py']}
        fake=copy.deepcopy(read(Path(args.bounded_geometry_control)))
        fake['sources']['weld_pipeline/server_bounded_cached_relation_prior.py']='0'*64
        fixture=out/'rejected_bounded_geometry_binding.json';write(fixture,fake)
        try:configure(args.reference_control,args.synthetic_control,fixture,args.bounded_competition_control,out)
        except ValueError:pass
        else:raise AssertionError('Forged bounded geometry source was accepted')
        dataset=FullShardDataset(ROOT)
        if args.job not in dataset.jobs:raise ValueError('One complete actual TRAIN input required')
        pools=BASE/args.pool_name;torch.set_num_threads(2)
        initial=Path(args.selector_checkpoint).resolve();initial.relative_to(ROOT)
        if digest(initial)!=args.selector_sha256:raise ValueError('Original selector initialization differs')
        with threadpool_limits(limits=2):
            original=load_job(dataset,args.job,pools,CachedRun446RelationPrior(FrozenRun446Priors(ROOT)),reference_chunk_size=1024)
            bounded=loader(dataset,args.job,pools,BoundedCachedRun446RelationPrior(FrozenRun446Priors(ROOT)),reference_chunk_size=1024)
            for key in ['flat','offsets','counts','pairs','pair_features','base_penalties','pair_scale']:
                np.testing.assert_array_equal(original[key],bounded[key],err_msg=key)
            assert original['pool']==bounded['pool'] and original['pc']==bounded['pc']
            assert original['pool_receipt_sha256']==bounded['pool_receipt_sha256']
            assert [v['key'] for v in original['views']]==[v['key'] for v in bounded['views']]
            for a,b in zip(original['views'],bounded['views']):
                for key in ['key','G','target','weight','pool_capacity_reference_evaluation']:assert a[key]==b[key],key
                for key in ['costs','tp','child_counts','invalid_children']:
                    np.testing.assert_array_equal(a['table'][key],b['table'][key],err_msg=key)
            device='cuda' if torch.cuda.is_available() else 'cpu'
            histories=[];models=[];optimizers=[];predictions=[]
            initialization=torch.load(initial,map_location='cpu',weights_only=False)['state_dict']
            for name,row in [('original',original),('bounded',bounded)]:
                folder=out/name;folder.mkdir();torch.manual_seed(1031)
                model=OrderedParentSetModel().to(device);model.load_state_dict(initialization,strict=True)
                optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001,eps=1.e-6)
                state=dict(optimizer_steps=0,updated_versions=[],processed_versions=sorted(v['key'] for v in row['views']),
                    job_visits=[],completed=[],passes=1,passes_completed=0,query_full_fit=None)
                history=fit_job(model,optimizer,row,8,state,folder,dataset,0)
                write(folder/'history.json',history)
                assert state['optimizer_steps']==8
                unary,penalties=forward_scores(model,row)
                predictions.append((unary,penalties,greedy_nullable_set(unary,row['pairs'],penalties,row['pc'])))
                histories.append(history);models.append(model);optimizers.append(optimizer)
            assert histories[0]==histories[1]
            for name,value in models[0].state_dict().items():
                assert torch.equal(value.cpu(),models[1].state_dict()[name].cpu()),name
            for index,values in optimizers[0].state_dict()['state'].items():
                for name,value in values.items():
                    other=optimizers[1].state_dict()['state'][index][name]
                    assert torch.equal(value.cpu(),other.cpu()) if isinstance(value,torch.Tensor) else value==other
            for index in [0,1]:np.testing.assert_array_equal(predictions[0][index],predictions[1][index])
            assert predictions[0][2]==predictions[1][2]
        if sources!={n:digest(ROOT/n) for n in sources}:raise ValueError('Control source changed')
        write(out/'receipt.json',dict(status='PASS_BOUNDED_ORIGINAL_FULL_JOB_AND_ACTUAL_EIGHT_STEP_FIT',
            job_key=args.job,pool_name=args.pool_name,parents=len(original['counts']),token_rows=len(original['flat']),
            same_input_versions=[v['key'] for v in original['views']],
            all_input_tokens_and_costs_and_views_bitwise_equal=True,
            all_eight_histories_equal=True,actual_optimizer_updates_per_model=8,
            final_parameter_and_optimizer_tensors_bitwise_equal=True,
            final_unary_penalties_and_selection_bitwise_equal=True,
            forged_bounded_source_binding_rejected=True,
            backend=metadata,sources=sources,selector_initialization_sha256=args.selector_sha256,
            held_reference_reads=0,independent_model_quality_evaluations=0,
            full_model_training_or_new_accuracy_not_claimed=True,seconds=time.perf_counter()-started))
        print('BOUNDED_ORIGINAL_FULL_JOB_AND_EIGHT_STEP_FIT_PASS',len(original['counts']),flush=True)
    except Exception:
        write(out/'failure.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-name',required=True)
    parser.add_argument('--pool-name',required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--selector-checkpoint',required=True)
    parser.add_argument('--selector-sha256',required=True)
    control_arguments(parser);relation_arguments(parser)
    main(parser.parse_args())
