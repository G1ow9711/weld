"""Five-case pilot with original three-case regression and untouched DEV64.

Additional inputs are frozen before native TRAIN supervision is read. Known
background requires complete original fragment labels on that carrier. Neither
unknown aliases nor original GT coordinates are inserted as candidate geometry.
"""
import argparse
import gc
import os
import shutil
import traceback
from collections import defaultdict
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline import sjg_additive_parent_precision_1254 as raw
from weld_pipeline import sjg_child_parent_supervision_1255 as supervision
from weld_pipeline import sjg_child_parent_model_1256 as model

OLD=('21','BM79A_native_local','10')
NEW=('25','36')
TRAIN=(*OLD,*NEW)
KEYS=(*TRAIN,'64')
FEATURE_RUN='run415'
LABEL_RUN='run416'
MODEL_RUN='run417'


def routes():
    result={key:dict(feature_run='run409') for key in (*OLD,'64')}
    for key in NEW:
        result[key]=dict(baseline_run='run413',proposal_run='run414/pairs',feature_run=FEATURE_RUN,
            pair_inputs=dict(input_run='run414/endpoints',context_run='run414/whole',
                sequence_run='run414/sequence56',occupancy_run='run414/occupancy'),
            known_background_file=f'{LABEL_RUN}/source_{key}/known_background.json')
    return result


def snapshot(dest):
    folder=dest/'source_snapshot';folder.mkdir(exist_ok=True)
    for name in ('sjg_additive_parent_precision_1254.py','sjg_child_parent_supervision_1255.py',
                 'sjg_child_parent_model_1256.py','sjg_five_case_child_pilot_1258.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,folder/name)


def prepare():
    check_resources()
    for key in NEW:
        receipt=read(OUT/'run414'/f'stage_pairs_{key}.json')
        assert receipt['status']=='COMPLETE'
        for name,sha in receipt['output_receipts'].items():assert digest(OUT/'run414'/name)==sha
    dest=OUT/FEATURE_RUN;dest.mkdir(exist_ok=False);snapshot(dest)
    write(dest/'registered_protocol.json',dict(TRAIN=list(TRAIN),DEV=['64'],original_G=269,
        added_typical_cases=list(NEW),new_cases_are_not_independent_holdouts=True,
        fixed_three_models_64_trees_each=True,threshold=.5,baseline='run379',
        parent_child_gap_closure_denominator_unchanged=True,STEP_reads=0,
        external_consultation_paused=True,full_contract_pass=False))
    raw.prepare(dest,keys=NEW,routes=routes())
    label=OUT/LABEL_RUN;label.mkdir(exist_ok=False);snapshot(label)
    write(label/'input_routes.json',dict(routes=routes(),TRAIN=list(TRAIN),DEV=['64'],G=269))
    supervision.prepare(label,keys=KEYS)
    # Old four inputs must be byte-for-byte equal after routing parameterization.
    for key in (*OLD,'64'):
        assert digest(label/f'source_{key}/features.npy')==digest(OUT/f'run411/source_{key}/features.npy')
        assert read(label/f'source_{key}/origin.json')==read(OUT/f'run411/source_{key}/origin.json')
    write(label/'default_input_regression.json',dict(status='PASS',keys=list((*OLD,'64')),
        features_byte_equal=True,origins_exact_equal=True,fit_calls=0,GT_reads=0))


def known_background(key,dest):
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
    case=load_case(key);graph=case['graph'];case['x']._mmap.close()
    label_path=OUT/f'run318/source_{key}_count_targets.npz'
    instances_path=OUT/f'run318/source_{key}_interval_instances.json'
    bind=next(c for c in read(OUT/'run342/receipt.json')['cases'] if c['source']==key)
    assert digest(label_path)==bind['labels_sha256'] and digest(instances_path)==bind['instances_sha256']
    with np.load(label_path) as z:labels=z['labels']
    canonical=np.asarray(case['canonical']);pids=np.searchsorted(graph.offsets,canonical,side='right')-1
    known_count=np.zeros(2*len(graph.profiles),int)
    for side in (0,1):np.add.at(known_count,2*pids+side,labels[side::2]>=0)
    required=np.repeat(np.diff(graph.offsets),2);complete=(known_count==required)&(required>0)
    categories=defaultdict(set)
    for event in read(instances_path)['instances']:
        row=int(event['row']);carrier=2*int(pids[row//2])+row%2
        categories[carrier,tuple(event['owner'])].add(int(event['label']))
    conflicts=[]
    for (carrier,owner),values in categories.items():
        if len(values)>1:
            complete[carrier]=False;conflicts.append(dict(row=carrier,owner=list(owner),categories=sorted(values)))
    map_path=OUT/f'run414/endpoints/source_{key}/role_map.npz'
    assert digest(map_path)==read(map_path.parent/'receipt.json')['map_sha256']
    with np.load(map_path) as z:rows=z['rows'].tolist()
    result=dict(source=key,rows=[dict(row=int(row),complete_background_supervision_allowed=bool(complete[row])) for row in rows],
        known_rows=sum(bool(complete[row]) for row in rows),unknown_rows=sum(not bool(complete[row]) for row in rows),
        conflicts=conflicts,source_labels_sha256=bind['labels_sha256'],source_instances_sha256=bind['instances_sha256'],
        rule='All original carrier fragments need canonical complete labels; category conflicts remain unknown',
        old_positive_owner_mapping_not_used_for_new_geometry_labels=True,DEV_reads=0)
    write(dest/f'source_{key}/known_background.json',result)
    print('KNOWN_BACKGROUND',key,result['known_rows'],'UNKNOWN',result['unknown_rows'],flush=True)
    del case,graph;gc.collect()


def targets():
    dest=OUT/LABEL_RUN
    assert read(dest/'all_inputs_frozen.json')['GT_reads']==0
    for key in NEW:known_background(key,dest)
    supervision.targets(dest,train=NEW,original_g=56)
    new_receipt=read(dest/'supervision_receipt.json')
    write(dest/'additional_cases_supervision.json',new_receipt)
    prior=read(OUT/'run411/supervision_receipt.json')
    for case in prior['cases']:
        key=case['source'];assert key in OLD
        old=OUT/f'run411/source_{key}';new=dest/f'source_{key}'
        assert digest(old/'training_targets.npz')==case['labels_sha256']
        for name in ('training_targets.npz','child_compatibility.json','training_pair_indices.json'):
            os.link(old/name,new/name)
    combined=prior['cases']+new_receipt['cases']
    assert sum(c['G'] for c in combined)==269
    write(dest/'supervision_receipt.json',dict(cases=combined,G=269,TRAIN=list(TRAIN),DEV_reads=0,
        original_three_labels_reused_byte_exact=True,original_parent_scope_preserved=True,
        child_labels_not_full_parent_acceptance=True,full_contract_pass=False))


def audit(label_run=LABEL_RUN,train=TRAIN,routebook=None):
    from weld_pipeline.sjg_recall_set_worker_1162 import target_for,score_value
    dest=OUT/label_run;checks=[];link_rows=0;routebook=routes() if routebook is None else routebook
    frozen=read(dest/'all_inputs_frozen.json')
    for key,sha in frozen['receipts'].items():assert digest(dest/f'source_{key}/receipt.json')==sha
    for key in train:
        target,_=target_for(key);_,_,context,_,base,_=supervision.child_inputs(key,dest)
        flat=[dict(original_child_count=1,is_closed=bool(c.get('original_is_closed',False)),gaps=[],children=[c])
              for p in target['parents'] for c in p['children']]
        ft=dict(target,parents=flat);folder=dest/f'source_{key}'
        c=read(folder/'child_compatibility.json');edges=c['compatible_targets']
        owners=c['original_parent_child_owners'];state=c['forward_state_owners']
        with np.load(folder/'training_targets.npz') as z:y=z['validity'];ly=z['link_y'];dy=z['duplicate_y']
        route=routebook[key]
        known_path=OUT/route.get('known_background_file',f'run406/source_{key}/compatibility.json')
        known={v['row']:v['complete_background_supervision_allowed'] for v in read(known_path)['rows']}
        intervals=read(OUT/route.get('proposal_run','run408')/f'proposal_{key}/selected_intervals.json')['intervals']
        assert len(y)==len(intervals)==len(edges)-base
        for i,label in enumerate(y):
            if label==0:assert known.get(intervals[i]['row'],False)
            if label<0:assert not known.get(intervals[i]['row'],False)
        idx=read(folder/'training_pair_indices.json')
        assert np.array_equal(dy,[int(len(set(edges[a])|set(edges[b]))==1) for a,b in idx['duplicate_pairs']])
        for (a,b),label in zip(idx['link_state_pairs'],ly):
            allowed=set()
            for reverse in (False,True):
                sa,sb=(b^1,a^1) if reverse else (a,b)
                for aa in state[sa]:
                    for bb in state[sb]:
                        pa,ca=owners[aa];pb,cb=owners[bb]
                        if pa!=pb:continue
                        parent=target['parents'][pa];count=len(parent['children'])
                        if cb==ca+1:allowed.add(1 if parent['gaps'][ca]=='SHARED' else 2)
                        elif parent['is_closed'] and count>1 and ca==count-1 and cb==0:allowed.add(1)
            assert int(label) in (allowed or {0}),(key,a,b,label,allowed)
            if label==0:assert not allowed
            link_rows+=1
        assert all(y[i]==1 if e else y[i] in (-1,0) for i,e in enumerate(edges[base:]))
        selected=[]
        for positive in (False,True):
            pool=[i for i,v in enumerate(edges) if bool(v)==positive]
            if pool:selected.extend(pool[i] for i in np.unique(np.linspace(0,len(pool)-1,min(8,len(pool)),dtype=int)))
        for i in selected:
            result=score_value(dict(parents=[context.parents[i]],invalid_predictions=[],total_prediction_children=1),ft,key)['report']
            assert bool(result['TP'])==bool(edges[i]);checks.append(dict(source=key,candidate_index=i,TP=result['TP']))
        print('CHILD_LABEL_AUDIT',key,'CHECKS',len(selected),flush=True)
        del context;gc.collect()
    model.controls(dest)
    write(dest/'prefit_audit.json',dict(status='PASS',child_checks=checks,all_link_rows_checked=link_rows,
        original_parent_order_gap_and_common_orientation_verified=True,
        all_negative_labels_require_complete_background_mask=True,
        DEV_reads=0,fit_calls=0,full_contract_pass=False))


def train():
    dest=OUT/MODEL_RUN;dest.mkdir(exist_ok=False);snapshot(dest)
    write(dest/'registered_protocol.json',dict(TRAIN=list(TRAIN),DEV=['64'],original_G=269,
        trees_each=64,threshold=.5,input_run=LABEL_RUN,source_sha256=digest(__file__),
        exactly_one_orientation_per_retained_child=True,all_retained_children_must_be_output=True,
        frozen_best379=True,external_consultation_paused=True,full_contract_pass=False))
    model.train(dest,OUT/LABEL_RUN)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','targets','audit','train','infer'],required=True)
    ap.add_argument('--source');a=ap.parse_args()
    try:
        if a.stage=='infer':model.infer(OUT/MODEL_RUN,a.source)
        else:globals()[a.stage]()
    except BaseException:
        write(ROOT/'.codex-local'/f'five_case_{a.stage}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
