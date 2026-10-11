"""Explicit per-stage pure-input preparation for two additional pilot TRAINs.

Historical output folders and checkpoints remain unchanged. Optional function
arguments route new cases into this run; all defaults retain prior behavior.
"""
import argparse
import importlib
import os
import time
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,reclaim_own_idle_pages

STAGES=(
    ('whole','sjg_whole_joint_features_1228','prepare'),
    ('local','sjg_local_cross_section_1233','prepare'),
    ('admission','sjg_local_presence_pilot_1234','infer'),
    ('sequence28','sjg_dense_joint_sequence_1237','prepare'),
    ('sequence52','sjg_global_neighborhood_sequence_1239','prepare'),
    ('sequence56','sjg_finite_extent_sequence_1240','prepare'),
    ('occupancy','sjg_structured_sequence_1241','infer'),
    ('eventsplane','sjg_geometric_endpoint_events_1242','prepare'),
    ('eventssupport','sjg_observed_curve_boundaries_1243','prepare'),
    ('eventsmultiscale','sjg_multiscale_endpoint_union_1244','prepare'),
    ('eventslegacy','sjg_legacy_endpoint_neighborhood_1250','prepare'),
    ('endpoints','sjg_endpoint_role_features_1245','prepare'),
    ('pairs','sjg_joint_pair_model_1248','infer'),
)


def initialize(dest):
    assert read(OUT/'run413/receipt.json')['status']=='ADDITIONAL_TYPICAL_TRAIN_INPUTS_AND_FROZEN_BASELINE_READY'
    dest.mkdir(exist_ok=False);snap=dest/'source_snapshot';snap.mkdir()
    import shutil
    names={module for _,module,_ in STAGES}|{'sjg_local_interval_pilot_1238','sjg_endpoint_role_train_1246','sjg_typical_case_pipeline_1257'}
    bindings={}
    for name in sorted(names):
        source=ROOT/'weld_pipeline'/f'{name}.py';bindings[name]=digest(source);shutil.copy2(source,snap/source.name)
    for stage,_,_ in STAGES:(dest/stage).mkdir()
    for stage,source,name in [('admission','run384','whole_joint_model.pkl'),
                              ('occupancy','run392','sequence_model.pt'),('pairs','run408','pair_model.pt')]:
        receipt=read(OUT/source/'training_receipt.json')
        assert digest(OUT/source/name)==receipt['model_sha256']
        os.link(OUT/source/name,dest/stage/name)
        os.link(OUT/source/'training_receipt.json',dest/stage/'training_receipt.json')
    write(dest/'registered_protocol.json',dict(status='TWO_TYPICAL_CASES_PURE_INPUT_PREPARATION',
        sources=['25','36'],stages=[s for s,_,_ in STAGES],module_sha256=bindings,
        frozen_admission='run384',frozen_occupancy='run392',frozen_pair_model='run408',
        legacy_proposals='run413',fit_calls=0,GT_reads=0,STEP_reads=0,
        preserve_original_runs=True,full_contract_pass=False,external_consultation_paused=True))


def run_stage(dest,stage,key):
    assert key in ('25','36');entry=next(v for v in STAGES if v[0]==stage)
    protocol=read(dest/'registered_protocol.json');module=importlib.import_module('weld_pipeline.'+entry[1])
    assert digest(module.__file__)==protocol['module_sha256'][entry[1]]
    index=[v[0] for v in STAGES].index(stage)
    if index:assert read(dest/f'stage_{STAGES[index-1][0]}_{key}.json')['status']=='COMPLETE'
    result=dest/f'stage_{stage}_{key}.json'
    if result.exists():
        r=read(result);assert r['status']=='COMPLETE'
        for name,sha in r['output_receipts'].items():assert digest(dest/name)==sha
        print('VERIFIED_EXISTING_STAGE',stage,key,flush=True);return
    paths={name:f'{dest.name}/{name}' for name,_,_ in STAGES}
    kwargs={
        'whole':{},'local':dict(whole_run=paths['whole']),
        'admission':dict(whole_run=paths['whole'],local_run=paths['local']),
        'sequence28':dict(admission_run=paths['admission'],whole_run=paths['whole']),
        'sequence52':dict(base_run=paths['sequence28']),
        'sequence56':dict(base_run=paths['sequence52']),
        'occupancy':dict(input_run=paths['sequence56'],whole_run=paths['whole']),
        'eventsplane':dict(sequence_run=paths['sequence56']),
        'eventssupport':dict(base_run=paths['eventsplane']),
        'eventsmultiscale':dict(base_run=paths['eventssupport'],legacy_run='run413'),
        'eventslegacy':dict(base_run=paths['eventsmultiscale'],legacy_run='run413'),
        'endpoints':dict(event_run=paths['eventslegacy'],sequence_run=paths['sequence56'],occupancy_run=paths['occupancy']),
        'pairs':dict(input_run=paths['endpoints'],context_run=paths['whole'],sequence_run=paths['sequence56'],occupancy_run=paths['occupancy'])}[stage]
    begin=time.perf_counter();before=check_resources();print('INPUT_STAGE_START',key,stage,flush=True)
    control=getattr(module,'controls',None)
    if control is not None and stage not in ('admission','occupancy','pairs'):
        control(dest/stage)
    operation=getattr(module,entry[2])
    if entry[2]=='prepare':operation(key,dest/stage,**kwargs)
    else:operation(dest/stage,key,**kwargs)
    output=dest/stage/(f'source_{key}/receipt.json' if entry[2]=='prepare' else f'proposal_{key}/prediction_receipt.json')
    receipt=read(output);assert receipt['GT_reads']==receipt['STEP_reads']==0
    memory_trim=reclaim_own_idle_pages()
    write(result,dict(status='COMPLETE',stage=stage,source=key,kwargs=kwargs,
        module_sha256=protocol['module_sha256'][entry[1]],
        output_receipts={str(output.relative_to(dest)):digest(output)},fit_calls=0,GT_reads=0,STEP_reads=0,
        seconds=time.perf_counter()-begin,resources_before=before,resources_after=check_resources(),
        after_operation_own_idle_memory_trim=memory_trim))
    print('INPUT_STAGE_COMPLETE',key,stage,round(time.perf_counter()-begin,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True);ap.add_argument('--source');a=ap.parse_args();dest=OUT/a.run
    if a.stage=='initialize':initialize(dest);return
    try:run_stage(dest,a.stage,a.source)
    except BaseException:
        write(dest/f'failure_{a.stage}_{a.source}.json',dict(error=traceback.format_exc(),fit_calls=0));raise


if __name__=='__main__':main()
