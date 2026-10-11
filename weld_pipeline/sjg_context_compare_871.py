"""Freeze both XYZ relation arms and constrained overlap outputs before scoring."""
import argparse
import gc
import sys
import time
import traceback
import numpy as np
from scipy.optimize import linear_sum_assignment
from weld_pipeline.sjg_prepare_776 import ROOT,OUT
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_material_evaluate_862 import main as infer,score_frozen
from weld_pipeline.sjg_material_comparison_audit_866 import owner_detection
from weld_pipeline.sjg_constrained_material_870 import constrained_memberships,controls

ARMS=['dense_control','context_negative']


def overlap_detection(offsets,membership,owner):
    """One-to-one auxiliary entity recovery; unassigned known points are misses."""
    known=owner>=0;truth_values=np.unique(owner[known]);classes=len(truth_values)
    point=np.repeat(np.arange(len(owner),dtype=np.int64),np.diff(offsets))
    valid=known[point];pred=membership[valid];assigned=point[valid]
    truth=np.searchsorted(truth_values,owner[assigned]);hypotheses,inverse=np.unique(pred,return_inverse=True)
    rows=len(hypotheses);known_count=int(known.sum())
    gtcounts=np.bincount(np.searchsorted(truth_values,owner[known]),minlength=classes)
    if rows and classes:
        table=np.bincount(inverse*classes+truth,minlength=rows*classes).reshape(rows,classes)
        union=table.sum(1)[:,None]+gtcounts[None,:]-table;iou=table/np.maximum(union,1)
        score=(iou>=.5).astype(float)+iou/(min(rows,classes)+1)
        a,b=linear_sum_assignment(-score);hit=iou[a,b]>=.5;count=int(hit.sum())
        matches=[dict(hypothesis=int(hypotheses[r]),reference_owner=int(truth_values[c]),IoU=float(iou[r,c])) for r,c in zip(a[hit],b[hit])]
    else:count=0;matches=[]
    return dict(known_reference_points=known_count,unknown_reference_points=int((~known).sum()),
        predicted_components_with_known_support=rows,reference_owners=classes,matched_reference_owners_at_IoU_half=count,
        missed_reference_owners_at_IoU_half=classes-count,matched_component_precision=count/max(1,rows),
        reference_owner_recall=count/max(1,classes),known_unassigned_points=int((known&(np.diff(offsets)==0)).sum()),
        known_multi_membership_points=int((known&(np.diff(offsets)>1)).sum()),matches=matches,
        auxiliary_not_contract_metric=True,unknown_reference_not_approved_truth=True)


def score_structure(dest):
    teachers={r['source']:r for r in read(OUT/'run22/label_summary.json')};rows=[]
    for record in read(dest/'constrained_predictions_frozen.json')['sources']:
        source=record['source'];assert digest(dest/record['file'])==record['sha256']
        if source not in teachers or 'teacher_file' not in teachers[source]:
            rows.append(dict(source=source,status='NO_OFFLINE_STRUCTURE_REFERENCE',original_points=record['points']));continue
        teacher=OUT/'run22'/teachers[source]['teacher_file'];assert digest(teacher)==teachers[source]['teacher_sha256']
        with np.load(teacher) as t,np.load(dest/record['file']) as q,np.load(dest/f'source_{source:02d}_material_relations.npz') as p:
            owner=t['reference_volume'];simple=owner_detection(p['material_component'],owner)
            overlap=overlap_detection(q['offsets'],q['membership'],owner)
        row=dict(source=source,original_points=record['points'],positive_connected_components=simple,constrained_multi_membership=overlap,
            decoder_stats=record['stats'],full_contract_pass=False)
        rows.append(row);print('CONSTRAINED_ENTITY_RESULT',source,'CC',simple['matched_reference_owners_at_IoU_half'],
            'CONSTRAINED',overlap['matched_reference_owners_at_IoU_half'],'/',overlap['reference_owners'],
            'HYPOTHESES',overlap['predicted_components_with_known_support'],'KNOWN_UNASSIGNED',overlap['known_unassigned_points'],flush=True)
    write(dest/'structure_metrics.json',rows);return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);saved=sys.argv[:]
    assert read(OUT/'run74/receipt.json')['status']=='TWO_CONTEXT_MATERIAL_ARMS_TRAINED'
    write(dest/'registered_protocol.json',dict(stage='CONTEXT_NEGATIVE_XYZ_AND_CONSTRAINED_GROUPING_COMPARISON',training_run='run74',arms=ARMS,
        source_sha256=digest(__file__),decoder_sha256=digest(ROOT/'weld_pipeline/sjg_constrained_material_870.py'),
        all_arms_and_decoders_before_reference_reads=True,fixed_distance_threshold=.5,fixed_boundary_threshold=.5,
        negative_edges_forbid_shared_membership=True,boundaries_can_overlap_but_not_bridge=True,unassigned_remain_unknown=True,
        no_development_selection=True,weld_model_not_replaced=True,full_contract_G=496,full_contract_pass=False,start_utc=utc()))
    write(dest/'controls.json',controls())
    try:
        frozen=[]
        for arm in ARMS:
            sys.argv=[saved[0],'--run',str((dest/arm).relative_to(OUT))]
            infer(checkpoint_run='run74/'+arm,score_outputs=False)
        for arm in ARMS:
            folder=dest/arm;outputs=[]
            for binding in read(folder/'all_predictions_frozen.json')['sources']:
                check_resources();source=binding['source'];assert digest(folder/binding['file'])==binding['sha256']
                print('CONSTRAINED_MATERIAL_START',arm,source,'ORIGINAL_POINTS',binding['points'],flush=True)
                with np.load(folder/binding['file']) as p:
                    started=time.perf_counter();out,stats=constrained_memberships(p['neighbors'],p['material_distance'],p['boundary']);seconds=time.perf_counter()-started
                file=folder/f'source_{source:02d}_constrained_memberships.npz';np.savez_compressed(file,**out)
                outputs.append(dict(source=source,file=file.name,sha256=digest(file),points=binding['points'],
                    relation_sha256=binding['sha256'],seconds=seconds,stats=stats))
                print('CONSTRAINED_MATERIAL_FROZEN',arm,source,stats,'SECONDS',seconds,flush=True);del out;gc.collect()
            write(folder/'constrained_predictions_frozen.json',dict(sources=outputs,all_before_reference_reads=True))
            frozen.append(dict(arm=arm,relation_manifest_sha256=digest(folder/'all_predictions_frozen.json'),
                structure_manifest_sha256=digest(folder/'constrained_predictions_frozen.json')))
        write(dest/'all_arms_frozen_before_reference.json',dict(arms=frozen,all_before_reference=True))
        metrics={};structure={}
        for arm in ARMS:
            metrics[arm]=score_frozen(dest/arm);structure[arm]=score_structure(dest/arm)
            r=read(dest/arm/'receipt.json');r.update(status='EIGHT_RAW_MATERIAL_RELATIONS_FROZEN_AND_EVALUATED',rows=metrics[arm],end_utc=utc());write(dest/arm/'receipt.json',r)
        write(dest/'receipt.json',dict(status='CONTEXT_MATERIAL_COMPARISON_COMPLETE',metrics=metrics,structure=structure,
            optimizer_updates=0,full_contract_G=496,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise
    finally:sys.argv=saved


if __name__=='__main__':main()
