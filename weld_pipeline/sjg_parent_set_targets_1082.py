"""Compile latent full-parent TRAIN targets after input freezing, without pruning."""
import argparse
import gc
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_parent_anchor_bank_1080 import CarrierBank
from weld_pipeline.sjg_parent_set_runtime_1081 import decode_choices,child_geometry
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents

TRAIN_KEYS=('21','4','10','25','36','42','BM79A_native_local')
SCIENTIFIC_KEYS=('21','BM79A_native_local','10')


def compile_case(key,bank,target):
    sup=read(OUT/f'run186/source_{key}_supervision.json')
    by_child={c['original_child_id']:[] for p in target['parents'] for c in p['children']}
    max_roundoff=0.
    for pid,events in sup['by_profile'].items():
        item=bank.get(int(pid));length=item['curve'].length
        coordinates=np.asarray(item['curve'].at(np.asarray([0.,length])))
        allowance=64*np.finfo(float).eps*max(1.,length,float(np.abs(coordinates).max()))
        for event in events:
            raw=np.asarray([event['lo'],event['hi']],np.float64)
            clipped=np.clip(raw,0.,length)
            delta=float(np.max(np.abs(raw-clipped)));max_roundoff=max(max_roundoff,delta)
            if delta>allowance:raise ValueError(f'Original witness outside finite profile {key} {pid}: {raw} length {length}')
            located=[bank.locate(pid,q) for q in clipped]
            recovered=[bank.coordinate(pid,i,f) for i,f in located]
            assert np.allclose(recovered,clipped,rtol=0,atol=allowance)
            by_child[event['original_child_id']].append(dict(profile=int(pid),
                lo=float(clipped[0]),hi=float(clipped[1]),intervals=[x[0] for x in located],
                fractions=[x[1] for x in located],forward=bool(event['target_direction_forward']),
                closed=bool(event['closed_whole']),wrap=bool(event['wrap']),
                original_compatibility_error_mm=event['maximum_bidirectional_sample_error_mm']))
    parents=[]
    for original in target['parents']:
        if not 1<=len(original['children'])<=5:raise ValueError('Registered child capacity overflow')
        children=[]
        for child in original['children']:
            alternatives=by_child[child['original_child_id']]
            if not alternatives:raise ValueError('Original child lost all compatible witnesses')
            children.append(dict(original_child_id=child['original_child_id'],
                weld_type={'horizontal':0,'vertical':1}.get(child['weld_type'],-1),
                kind=child['original_kind'],representation=child['representation'],
                loss_samples_mm=child['loss_samples_mm'],alternatives=alternatives))
        parents.append(dict(original_parent_id=original['original_parent_id'],
            child_count=len(children),closed=bool(original['is_closed']),gaps=original['gaps'],children=children))
    if len(parents)>256:raise ValueError('Registered parent-query capacity overflow')
    assert sum(p['child_count'] for p in parents)==sup['original_G']
    return dict(source=key,parents=parents,G=sup['original_G'],all_child_witnesses_retained=True,
                maximum_roundoff_clamp_mm=max_roundoff,
                full_cell_anchor_bank=not bank.capacity_only,GT_used_only_for_TRAIN_supervision=True)


def ideal_choices(compiled,bank):
    choices=[]
    for query,parent in enumerate(compiled['parents']):
        children=[];states=[];back=[]
        for ci,child in enumerate(parent['children']):
            alternatives=child['alternatives'];ends=[]
            for a in alternatives:
                item=bank.get(a['profile'])
                points,_=child_geometry(item['curve'],a['lo'],a['hi'],a['forward'],a['closed'],a['wrap'])
                ends.append(points[[0,-1]])
            ends=np.asarray(ends)
            own=np.asarray([a['original_compatibility_error_mm'] for a in alternatives])
            if ci==0:cost=own;back.append(None)
            else:
                previous=states[-1]['ends']
                distance=np.linalg.norm(previous[:,1,None]-ends[None,:,0],axis=-1)
                if parent['gaps'][ci-1]=='POSITIVE_GAP':
                    pair=np.where(distance>0,0.,np.inf)
                elif parent['gaps'][ci-1]=='SHARED':pair=distance
                else:pair=np.zeros_like(distance)
                combined=states[-1]['cost'][:,None]+pair
                indices=combined.argmin(0);cost=combined[indices,np.arange(len(own))]+own
                back.append(indices)
            states.append(dict(ends=ends,cost=cost))
        selected=int(states[-1]['cost'].argmin())
        if not np.isfinite(states[-1]['cost'][selected]):raise ValueError('No topology-consistent complete parent witness')
        selections=[selected]
        for ci in range(len(states)-1,0,-1):
            selected=int(back[ci][selected]);selections.append(selected)
        selections.reverse()
        for ci,(child,selection) in enumerate(zip(parent['children'],selections)):
            a=child['alternatives'][selection]
            children.append(dict(a,weld_type=child['weld_type'],side='UNKNOWN',
                                 connection='UNKNOWN' if ci==0 else parent['gaps'][ci-1]))
        choices.append(dict(query=query,probability=1.,closed=parent['closed'],
                            child_count=parent['child_count'],children=children))
    return choices


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run262');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);start=time.perf_counter();rows=[]
    try:
        freeze=read(OUT/'run260/input_freeze.json')
        assert freeze['status']=='THREE_TRAIN_COMPLETE_INPUTS_WITH_MLS_FROZEN'
        cases={c['source_key']:c for c in read(OUT/'run98/local_cohort.json')['cases']}
        for key in TRAIN_KEYS:
            check_resources();case=cases[key]
            assert digest(OUT/case['target_file'])==case['target_sha256']
            bank=CarrierBank(key,capacity_only=key not in SCIENTIFIC_KEYS)
            target=read(OUT/case['target_file']);compiled=compile_case(key,bank,target)
            write(dest/f'source_{key}_targets.json',compiled)
            # Geometry correspondence is only an oracle expression control, never a prediction metric.
            choices=ideal_choices(compiled,bank);decoded=decode_choices(choices,bank)
            score=score_native_parents(decoded,target,[False]*len(target['parents']))
            row=dict(source=key,G=compiled['G'],parents=len(compiled['parents']),
                alternatives=sum(len(c['alternatives']) for p in compiled['parents'] for c in p['children']),
                maximum_children=max(p['child_count'] for p in compiled['parents']),
                full_cell_anchor_bank=not bank.capacity_only,
                oracle_geometry_score=score['categories']['overall'],
                invalid_oracle_parents=len(decoded['invalid_predictions']),
                maximum_roundoff_clamp_mm=compiled['maximum_roundoff_clamp_mm'],
                target_sha256=case['target_sha256'],compiled_sha256=digest(dest/f'source_{key}_targets.json'))
            write(dest/f'source_{key}_oracle_control.json',dict(row=row,invalid=decoded['invalid_predictions'],matches=score['matches']))
            rows.append(row);write(dest/'progress.json',dict(cases=rows,utc=utc()))
            print('PARENT_EXPRESSION_CONTROL',key,'G',row['G'],'ORACLE',row['oracle_geometry_score'],'INVALID',row['invalid_oracle_parents'],flush=True)
            del bank,target,compiled,choices,decoded,score;gc.collect();reclaim_own_idle_pages()
        passed=all(r['oracle_geometry_score']['TP']==r['G'] and r['oracle_geometry_score']['FP']==0 for r in rows)
        write(dest/'receipt.json',dict(status='PARENT_EXPRESSION_CONTROL_PASSED' if passed else 'PARENT_EXPRESSION_CONTROL_FAILED',
            cases=rows,TRAIN_G=sum(r['G'] for r in rows),TRAIN_parents=sum(r['parents'] for r in rows),
            scientific_training=False,optimizer_updates=0,extraction_correct_rate=None,extraction_precision_mm=None,
            oracle_score_not_model_accuracy=True,DEV_or_DIAG_target_reads=0,
            seconds=time.perf_counter()-start,utc=utc()))
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),optimizer_updates=0,utc=utc()));raise


if __name__=='__main__':main()
