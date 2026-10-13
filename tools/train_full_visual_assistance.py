"""Fit visual assistance on every registered TRAIN view, then augment all inputs.

Only the auxiliary point-region model is fitted here. Full child fields,
ordered parents and final CSV acceptance remain the downstream model's task.
The existing original query/point/parent weights and input shards are immutable.
"""
import argparse
import gc
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from tools.export_full_dataset_split_view import local_name
from tools.probe_multiview_visual_assistance import load_xyz, reference_samples, sampled_fit_rows
from tools.record_training_experiment import git_head
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_tiled_multiview_visual_features import visual_features, SCHEMA
from weld_pipeline.server_visual_candidate_augmentation import predict_point_scores, augmented_profiles
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, read, write

BASE = ROOT / 'server_rebuild/full_dataset_v2'
SOURCES = ['tools/train_full_visual_assistance.py', 'tools/probe_multiview_visual_assistance.py',
    'weld_pipeline/server_multiview_visual_features.py', 'weld_pipeline/server_tiled_multiview_visual_features.py',
    'weld_pipeline/server_visual_candidate_augmentation.py', 'weld_pipeline/server_partitioned_full_dataset.py',
    'weld_pipeline/server_full_shard_dataset.py']


def main(args):
    out = BASE / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    dataset = PartitionedFullShardDataset(ROOT, BASE); dataset.require_complete_inputs()
    sources = {p: digest(ROOT / p) for p in SOURCES}
    storage = Path(tempfile.mkdtemp(prefix='weld-full-visual-fit-', dir='/tmp'))
    (out / 'fit_arrays').symlink_to(storage, target_is_directory=True)
    jobs = sorted(dataset.jobs)
    protocol = dict(source_commit=git_head(ROOT), sources=sources,
        registry_sha256=dataset.registry_sha256, jobs=jobs, versions=sorted(dataset.versions),
        children=sum(v['children'] for _,v in dataset.versions.values()),
        raw_points=sum(j['original_points'] for j in dataset.jobs.values()),
        scope='Complete registered TRAIN auxiliary visual fit and original-profile-preserving augmentation',
        feature_schema=SCHEMA, pixel_mm=2., seed=1031, max_iter=100, max_leaf_nodes=15, max_depth=8,
        learning_rate=.08, min_samples_leaf=40, l2_regularization=1., early_stopping=False,
        positive_reference_sample_distance_mm=3., negative_distance_mm=6.,
        point_sample_cap_per_job=dict(positive=20000, negative=40000),
        fit_mass='Sum of original family_equal_version_weight per raw job, inverse point sampling probability',
        version_targets='Union of all registered versions for auxiliary point proximity only',
        full_point_inference=True, original_candidate_retention=True, new_candidates_require_own_features=True,
        missing_or_difficult_reference_children_removed=False,
        original_best_or_main_model_weights_modified=False, official_validation_or_test_reads=0,
        complete_weld_fields_parent_training_claimed=False, full_optimization_rounds_added=0)
    write(out / 'protocol.json', protocol)
    prepared = []; augmented = []
    try:
        for index,key in enumerate(jobs):
            job = dataset.jobs[key]; start = time.monotonic()
            xyz,binding = load_xyz(job); features,audit = visual_features(xyz)
            input_binding = dict(job_key=key, raw_input=binding, points=len(xyz), audit=audit,
                                 sources=sources, references_read_by_visual_producer=0)
            write(out / (key + '_input.json'), input_binding)
            reference,children = reference_samples(dataset, job)
            distance = cKDTree(reference).query(xyz,workers=2)[0] if len(reference) else np.full(len(xyz),np.inf)
            labels = np.full(len(xyz),-1,np.int8); labels[distance<=3.] = 1; labels[distance>6.] = 0
            ids,weights = sampled_fit_rows(labels,1031+index)
            mass = sum(v['family_equal_version_weight'] for v in job['versions'])
            weights = weights / max(weights.sum(),1.) * mass
            path = storage / (key + '.npz')
            if shutil.disk_usage(storage).free < features[ids].nbytes + 1024**3:
                raise RuntimeError('Insufficient owned fit storage; no TRAIN jobs discarded')
            np.savez_compressed(path,features=features[ids],labels=labels[ids],weights=weights)
            row = dict(job_key=key,fit_array_sha256=digest(path),input_receipt_sha256=digest(out/(key+'_input.json')),
                versions=[v['key'] for v in job['versions']],children=len(children),fit_points=len(ids),
                all_raw_points=len(xyz),positive_points=int((labels==1).sum()),negative_points=int((labels==0).sum()),
                family_version_mass=mass,reference_bindings={v['key']:v['targets_sha256'] for v in job['versions']},
                seconds=time.monotonic()-start)
            prepared.append(row); write(out/'preparation_progress.json',dict(pid=os.getpid(),completed_jobs=len(prepared),
                registered_jobs=len(jobs),completed_versions=sum(len(r['versions']) for r in prepared),
                children=sum(r['children'] for r in prepared),current_job=key,unix_time=time.time()))
            write(out/(key+'_fit_preparation.json'),row)
            print('FULL_VISUAL_INPUT',index+1,len(jobs),key,len(children),len(ids),flush=True)
            del xyz,features,reference,distance,labels
            gc.collect()
        if (sum(r['children'] for r in prepared)!=protocol['children']
                or {k for r in prepared for k in r['versions']}!=set(dataset.versions)):
            raise ValueError('A registered TRAIN reference disappeared')
        write(out/'all_fit_inputs.json',dict(records=prepared))
        count=sum(r['fit_points'] for r in prepared); x=np.empty((count,60),np.float32)
        y=np.empty(count,np.int8); weights=np.empty(count,np.float64); cursor=0
        for row in prepared:
            path=storage/(row['job_key']+'.npz')
            if digest(path)!=row['fit_array_sha256']: raise ValueError('Prepared fit array changed')
            with np.load(path) as z:
                n=len(z['labels']); x[cursor:cursor+n]=z['features']; y[cursor:cursor+n]=z['labels']
                weights[cursor:cursor+n]=z['weights']; cursor+=n
        if cursor!=count or not np.isclose(weights.sum(),1.):
            raise ValueError('Original family/version training mass lost')
        weights*=count
        estimator=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,max_depth=8,learning_rate=.08,
            min_samples_leaf=40,l2_regularization=1.,early_stopping=False,random_state=1031)
        start=time.monotonic(); print('FULL_VISUAL_FIT_STARTED',count,flush=True)
        with threadpool_limits(limits=4): estimator.fit(x,y,sample_weight=weights)
        bundle=dict(schema=SCHEMA,raw_units='mm',estimator=estimator,sources=sources,
                    training_registry_sha256=dataset.registry_sha256,protocol_sha256=digest(out/'protocol.json'))
        joblib.dump(bundle,out/'model.joblib',compress=3)
        fit_receipt=dict(status='ALL_REGISTERED_TRAIN_VISUAL_ASSISTANCE_FIT_COMPLETE',jobs=len(jobs),
            versions=len(dataset.versions),children=protocol['children'],raw_points=protocol['raw_points'],
            sampled_fit_points=count,iterations=int(estimator.n_iter_),seconds=time.monotonic()-start,
            model_sha256=digest(out/'model.joblib'),protocol_sha256=digest(out/'protocol.json'),
            official_validation_or_test_reads=0,full_weld_model_or_accuracy_not_claimed=True)
        write(out/'fit_receipt.json',fit_receipt); print('FULL_VISUAL_FIT_COMPLETE',flush=True)
        del x,y,weights;gc.collect()
        # This is training-input preparation: fitted scores on TRAIN are not generalization results.
        for index,key in enumerate(jobs):
            start=time.monotonic();xyz,binding=load_xyz(dataset.jobs[key])
            scores,audit=predict_point_scores(bundle,xyz)
            inp=dataset.input_case(key); combined,aug_audit=augmented_profiles(inp.profiles,xyz,scores)
            path=out/(key+'_additional_profiles.json')
            write(path,dict(job_key=key,original_profiles_sha256=digest(dataset.shards/key/'profiles.json'),
                original_input_receipt_sha256=digest(dataset.shards/key/'receipt.json'),
                model_sha256=fit_receipt['model_sha256'],raw_input=binding,
                original_profile_count=len(inp.profiles),additional_profiles=combined[len(inp.profiles):],
                visual_input_audit=audit,augmentation_audit=aug_audit,
                fitted_TRAIN_scores_not_generalization=True,reference_geometry_reads_during_augmentation=0,
                seconds=time.monotonic()-start))
            augmented.append(dict(job_key=key,path=path.name,sha256=digest(path),
                                  original_profiles=len(inp.profiles),added_profiles=len(combined)-len(inp.profiles)))
            write(out/'augmentation_progress.json',dict(pid=os.getpid(),completed_jobs=index+1,registered_jobs=len(jobs),
                added_profiles=sum(r['added_profiles'] for r in augmented),unix_time=time.time()))
            print('FULL_VISUAL_AUGMENT',index+1,len(jobs),key,augmented[-1]['added_profiles'],flush=True)
            del xyz,scores,inp,combined;gc.collect()
        if sources!={p:digest(ROOT/p) for p in sources}: raise ValueError('Source changed during complete fit')
        write(out/'receipt.json',dict(status='ALL_TRAIN_VISUAL_MODEL_AND_AUGMENTED_PROFILE_INPUTS_COMPLETE',
            fit_receipt_sha256=digest(out/'fit_receipt.json'),records=augmented,jobs=len(jobs),
            versions=len(dataset.versions),children=protocol['children'],raw_points=protocol['raw_points'],
            additional_profiles=sum(r['added_profiles'] for r in augmented),
            downstream_feature_and_full_weld_training_pending=True,full_extraction_accuracy_claimed=False,
            original_weights_and_inputs_preserved=True,official_validation_or_test_reads=0))
    except BaseException as exc:
        write(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),
            prepared_jobs=len(prepared),augmented_jobs=len(augmented)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment-name',required=True);main(p.parse_args())
