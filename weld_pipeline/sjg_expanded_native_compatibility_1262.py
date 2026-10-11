"""Native compatibility for the fixed expanded admission, TRAIN only.

Background completeness is reconstructed for all canonical carrier rows;
the old admission's restricted row list must not censor newly admitted truths.
"""
import argparse
import gc
import shutil
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_native_whole_presence_1259 import TRAIN,complete_background
from weld_pipeline import sjg_native_carrier_compatibility_1252 as native


def initialize(dest):
    dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir()
    bindings={}
    for name in ('sjg_expanded_native_compatibility_1262.py','sjg_native_whole_presence_1259.py','sjg_native_carrier_compatibility_1252.py'):
        src=ROOT/'weld_pipeline'/name;shutil.copy2(src,snapshot/name);bindings[name]=digest(src)
    write(dest/'registered_protocol.json',dict(TRAIN=list(TRAIN),original_G=269,input_run='run421/endpoints',
        module_sha256=bindings,GT_used_only_after_input_frozen=True,GT_endpoint_insertion=False,
        background_from_all_original_canonical_rows=True,unknowns_not_negatives=True,
        candidate_capacity_not_final_accuracy=True,fit_calls=0,DEV_reads=0,STEP_reads=0,
        external_consultation_paused=True,full_contract_pass=False))
    native.controls(dest)


def prepare(dest,key):
    assert key in TRAIN
    p=read(dest/'registered_protocol.json')
    for name,sha in p['module_sha256'].items():assert digest(ROOT/'weld_pipeline'/name)==sha
    stage=read(OUT/f'run421/stage_endpoints_{key}.json');assert stage['status']=='COMPLETE'
    for path,sha in stage['output_receipts'].items():assert digest(OUT/'run421'/path)==sha
    case=load_case(key);complete,binding=complete_background(case,key);case['x']._mmap.close()
    with np.load(OUT/f'run421/endpoints/source_{key}/role_map.npz') as z:rows=z['rows']
    path=dest/f'background_{key}.json'
    write(path,dict(source=key,rows=[dict(row=int(row),complete_background_supervision_allowed=bool(complete[row])) for row in rows],
        original_canonical_labels=binding,all_carrier_rows=len(complete),new_admission_rows=len(rows),
        derived_from_previous_selected_rows=False,unknowns_not_negatives=True))
    del case;gc.collect();reclaim_own_idle_pages();check_resources()
    result=native.prepare(key,dest,input_run='run421/endpoints',previous_set_run=None,
        known_background_file=str(path.relative_to(OUT)),require_full_coverage=False)
    found={}
    relation=read(dest/f'source_{key}/compatibility.json')
    eligible={r['profile'] for r in relation['rows'] if r['complete_background_supervision_allowed']}
    for profile in relation['profiles']:
        if profile['profile'] in eligible:
            for group in profile['groups']:found[group['owner_index']]=True
    missing=[owner for i,owner in enumerate(relation['owners']) if i not in found]
    write(dest/f'source_{key}/known_training_coverage.json',dict(source=key,G=result['G'],
        covered_G=len(found),missing=missing,capacity_only=True,full_contract_pass=False))
    print('KNOWN_BACKGROUND_OWNER_COVERAGE',key,len(found),'/',result['G'],flush=True)
    completed=[]
    for source in TRAIN:
        file=dest/f'source_{source}/receipt.json'
        if file.exists():completed.append(read(file))
    write(dest/'progress.json',dict(completed_sources=[v['source'] for v in completed],
        G=sum(v['G'] for v in completed),covered_G=sum(v['covered_G'] for v in completed),fit_calls=0))
    if len(completed)==len(TRAIN):
        write(dest/'receipt.json',dict(status='NATIVE_D3_COMPATIBILITY_REBUILT_PENDING_INDEPENDENT_AUDIT',
            cases=completed,G=sum(v['G'] for v in completed),covered_G=sum(v['covered_G'] for v in completed),
            fit_calls=0,DEV_reads=0,STEP_reads=0,full_contract_pass=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['initialize','prepare'],required=True);ap.add_argument('--source');a=ap.parse_args()
    dest=OUT/'run422'
    try:initialize(dest) if a.stage=='initialize' else prepare(dest,a.source)
    except BaseException:
        if dest.exists():write(dest/f'failure_{a.stage}_{a.source or "all"}.json',dict(error=traceback.format_exc(),fit_calls=0))
        raise


if __name__=='__main__':main()
