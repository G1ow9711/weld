"""Full pure-input sequence/boundary chain under one fixed admission rule.

Keep the final56-channel sequence. Earlier28/52-channel arrays may be reclaimed
only after exact NPY SHA reconstruction from its prefix columns succeeds.
Original datasets, previous runs and final training inputs are never removed.
"""
import argparse
import hashlib
import importlib
import os
import shutil
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_native_whole_presence_1259 import paths

KEYS=('25','36','64','21','BM79A_native_local','10')
STAGES=(
    ('sequence28','sjg_dense_joint_sequence_1237','prepare'),
    ('sequence52','sjg_global_neighborhood_sequence_1239','prepare'),
    ('sequence56','sjg_finite_extent_sequence_1240','prepare'),
    ('occupancy','sjg_structured_sequence_1241','infer'),
    ('eventsplane','sjg_geometric_endpoint_events_1242','prepare'),
    ('eventssupport','sjg_fast_observed_boundaries_1266','prepare_support'),
    ('eventsmultiscale','sjg_fast_observed_boundaries_1266','prepare_multiscale'),
    ('eventslegacy','sjg_legacy_endpoint_neighborhood_1250','prepare'),
    ('endpoints','sjg_endpoint_role_features_1245','prepare'),
)


class HashWriter:
    def __init__(self):self.hash=hashlib.sha256()
    def write(self,data):self.hash.update(data);return len(data)
    def flush(self):pass


def reclaim_prefix_caches(dest,key):
    final=dest/f'sequence56/source_{key}/features.npy';r=read(final.parent/'receipt.json')
    assert digest(final)==r['features_sha256']
    # Buffered reads avoid very slow strided faults on NTFS-compressed maps.
    x=np.load(final);records=[]
    for stage,width in (('sequence28',28),('sequence52',52)):
        path=dest/f'{stage}/source_{key}/features.npy';receipt=read(path.parent/'receipt.json')
        expected=receipt['features_sha256'];stream=HashWriter()
        np.lib.format.write_array(stream,x[:,:width],allow_pickle=False)
        reconstructed=stream.hash.hexdigest()
        if reconstructed!=expected:
            records.append(dict(stage=stage,status='RETAINED_RECONSTRUCTION_HASH_DIFFERS',expected=expected,reconstructed=reconstructed));continue
        resolved=path.resolve();resolved.relative_to(dest.resolve())
        assert resolved.name=='features.npy' and resolved.parent.name==f'source_{key}'
        if resolved.exists():
            assert digest(resolved)==expected
            size=resolved.stat().st_size;resolved.unlink()
        else:size=0
        records.append(dict(stage=stage,status='EXACT_RECONSTRUCTIBLE_PREFIX_RECLAIMED',original_sha256=expected,
            final_file=str(final.relative_to(dest)),prefix_columns=width,reconstructed_sha256=reconstructed,
            byte_length_reclaimed=size,restore='np.save(path, np.load(final_file, mmap_mode="r")[:, :prefix_columns])'))
    del x;write(dest/f'prefix_cache_reconstruction_{key}.json',dict(records=records,final_input_preserved=True))
    print('PREFIX_CACHE_RECLAIMED',key,sum(v.get('byte_length_reclaimed',0) for v in records),flush=True)


def initialize(dest):
    assert (ROOT/'.codex-local/run420_exit.txt').read_text(encoding='utf-8-sig').strip()=='0'
    for key in KEYS:
        r=read(OUT/f'run420/proposal_{key}/prediction_receipt.json')
        assert r['GT_reads']==r['STEP_reads']==0 and r['old_admitted_rows_all_retained']
    dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
    for name in sorted({module for _,module,_ in STAGES}|{'sjg_local_interval_pilot_1238','sjg_corrected_admission_pipeline_1261','sjg_finite_tube_query_1265'}):
        path=ROOT/'weld_pipeline'/f'{name}.py';bindings[name]=digest(path);shutil.copy2(path,snapshot/path.name)
    for stage,_,_ in STAGES:(dest/stage).mkdir()
    for name in ('sequence_model.pt','training_receipt.json'):os.link(OUT/'run392'/name,dest/'occupancy'/name)
    write(dest/'registered_protocol.json',dict(sources=list(KEYS),module_sha256=bindings,
        admission='run420 fixed old/new presence union',frozen_occupancy='run392',
        same_algorithm_for_all_cases=True,GT_reads=0,STEP_reads=0,fit_calls=0,
        exact_reconstructible_intermediate_prefix_reclamation=True,full_contract_pass=False,external_consultation_paused=True))


def run_stage(dest,stage,key):
    assert key in KEYS;protocol=read(dest/'registered_protocol.json');names=[v[0] for v in STAGES]
    index=names.index(stage);_,module_name,method=STAGES[index]
    if index:assert read(dest/f'stage_{names[index-1]}_{key}.json')['status']=='COMPLETE'
    record=dest/f'stage_{stage}_{key}.json'
    if record.exists():
        r=read(record);assert r['status']=='COMPLETE'
        for path,sha in r['output_receipts'].items():assert digest(dest/path)==sha
        print('VERIFIED_EXISTING_STAGE',key,stage,flush=True);return
    module=importlib.import_module('weld_pipeline.'+module_name)
    assert digest(module.__file__)==protocol['module_sha256'][module_name]
    if module_name=='sjg_fast_observed_boundaries_1266':
        assert digest(ROOT/'weld_pipeline/sjg_finite_tube_query_1265.py')==protocol['module_sha256']['sjg_finite_tube_query_1265']
    p={n:f'{dest.name}/{n}' for n in names};whole,_=paths(key);legacy='run413' if key in ('25','36') else 'run379'
    kwargs={
        'sequence28':dict(admission_run='run420',whole_run=whole),
        'sequence52':dict(base_run=p['sequence28']),
        'sequence56':dict(base_run=p['sequence52']),
        'occupancy':dict(input_run=p['sequence56'],whole_run=whole),
        'eventsplane':dict(sequence_run=p['sequence56']),
        'eventssupport':dict(base_run=p['eventsplane']),
        'eventsmultiscale':dict(base_run=p['eventssupport'],legacy_run=legacy),
        'eventslegacy':dict(base_run=p['eventsmultiscale'],legacy_run=legacy),
        'endpoints':dict(event_run=p['eventslegacy'],sequence_run=p['sequence56'],occupancy_run=p['occupancy'])}[stage]
    before=check_resources();begin=time.perf_counter();print('INPUT_STAGE_START',key,stage,flush=True)
    controls=getattr(module,'controls',None);is_prepare=method.startswith('prepare')
    if controls is not None and is_prepare:controls(dest/stage)
    operation=getattr(module,method)
    if is_prepare:operation(key,dest/stage,**kwargs)
    else:operation(dest/stage,key,**kwargs)
    output=dest/stage/(f'source_{key}/receipt.json' if is_prepare else f'proposal_{key}/prediction_receipt.json')
    assert read(output)['GT_reads']==read(output)['STEP_reads']==0
    if stage=='sequence56':reclaim_prefix_caches(dest,key)
    trim=reclaim_own_idle_pages()
    write(record,dict(status='COMPLETE',source=key,stage=stage,kwargs=kwargs,
        module_sha256=protocol['module_sha256'][module_name],output_receipts={str(output.relative_to(dest)):digest(output)},
        seconds=time.perf_counter()-begin,resources_before=before,resources_after=check_resources(),memory_trim=trim,
        fit_calls=0,GT_reads=0,STEP_reads=0))
    print('INPUT_STAGE_COMPLETE',key,stage,round(time.perf_counter()-begin,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--source');a=ap.parse_args();dest=OUT/'run421'
    if a.stage=='initialize':initialize(dest);return
    try:run_stage(dest,a.stage,a.source)
    except BaseException:write(dest/f'failure_{a.stage}_{a.source}.json',dict(error=traceback.format_exc(),fit_calls=0));raise


if __name__=='__main__':main()
