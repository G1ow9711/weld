"""Verify material experiments and expose owner merging hidden by pair accuracy."""
import argparse
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from weld_pipeline.sjg_prepare_776 import OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc


def owner_detection(component,owner):
    known=owner>=0;_,pred=np.unique(component[known],return_inverse=True);_,truth=np.unique(owner[known],return_inverse=True)
    if not len(pred):return dict(known_reference_points=0)
    rows=int(pred.max())+1;cols=int(truth.max())+1
    table=np.bincount(pred*cols+truth,minlength=rows*cols).reshape(rows,cols)
    union=table.sum(1)[:,None]+table.sum(0)[None]-table;iou=table/np.maximum(union,1)
    score=(iou>=.5).astype(float)+iou/(min(rows,cols)+1)
    a,b=linear_sum_assignment(-score);count=int((iou[a,b]>=.5).sum())
    return dict(known_reference_points=len(pred),predicted_components_with_known_support=rows,reference_owners=cols,
        matched_reference_owners_at_IoU_half=count,missed_reference_owners_at_IoU_half=cols-count,
        matched_component_precision=count/rows,reference_owner_recall=count/cols,
        unknown_reference_points=int((~known).sum()),auxiliary_not_contract_metric=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    assert read(OUT/'run71/receipt.json')['status']=='DENSE_MATERIAL_COMPARISON_COMPLETE'
    for arm in ['sampled_control','dense_material']:
        path=OUT/'run70'/arm;r=read(path/'receipt.json');epochs=[json.loads(s) for s in (path/'epochs.jsonl').read_text().splitlines()]
        assert len(epochs)==2 and [e['epoch'] for e in epochs]==[1,2]
        assert r['actual_optimizer_updates']==1168 and r['completed_full_epochs']==2 and digest(path/'last.pt')==r['checkpoint_sha256']
        assert all(e['core_points']==1880869 and e['known_material_points']==1747537 for e in epochs)
        assert all(e['directly_supervised_material_points']==(1747537 if arm=='dense_material' else 64905) for e in epochs)
    for r in read(OUT/'run71/all_arms_frozen_before_reference.json')['arms']:
        assert digest(OUT/'run71'/r['arm']/'all_predictions_frozen.json')==r['sha256']
    teachers={r['source']:r for r in read(OUT/'run22/label_summary.json')}
    original_inputs={r['source']:r for r in read(OUT/'run22/all_inputs_frozen.json')['sources']};summary={}
    for arm,path in [('initial_material',OUT/'run69'),('sampled_control',OUT/'run71/sampled_control'),('dense_material',OUT/'run71/dense_material')]:
        rows=[];metrics={r['source']:r for r in read(path/'offline_material_metrics.json')};total_points=0
        training=OUT/'run68' if arm=='initial_material' else OUT/'run70'/arm
        assert read(path/'registered_protocol.json')['checkpoint_sha256']==read(training/'receipt.json')['checkpoint_sha256']
        for binding in read(path/'all_predictions_frozen.json')['sources']:
            source=binding['source'];file=path/binding['file'];assert digest(file)==binding['sha256']
            assert binding['raw_sha256']==original_inputs[source]['raw_sha256'] and binding['points']==original_inputs[source]['points']
            with np.load(file) as p:
                n=binding['points'];total_points+=n;neighbor=p['neighbors'];distance=p['material_distance']
                assert neighbor.shape==(n,8) and distance.shape==neighbor.shape
                assert np.all(p['core_coverage']==1) and len(p['core_block'])==n
                assert np.all((neighbor>=-1)&(neighbor<n)) and not np.any(neighbor==np.arange(n)[:,None])
                assert not np.any(np.isfinite(distance)&(neighbor<0))
                assert np.all(p['material_component']>=0) and len(p['material_component'])==n
                if source not in teachers or 'teacher_file' not in teachers[source]:
                    rows.append(dict(source=source,status='NO_STRUCTURE_REFERENCE',original_points=n));continue
                teacher=OUT/'run22'/teachers[source]['teacher_file'];assert digest(teacher)==teachers[source]['teacher_sha256']
                with np.load(teacher) as t:owners=owner_detection(p['material_component'],t['reference_volume'])
            r=metrics[source];m=r['relations']['material_distance'];f=r['relations']['surface_distance'];b=r['boundary']
            all_m=m['all'];balanced=.5*(all_m['same_recall']+all_m['different_recall']) if all_m['same_recall'] is not None and all_m['different_recall'] is not None else None
            rows.append(dict(source=source,original_points=n,material_relations=all_m,balanced_relation_recall=balanced,
                cross_surface_same_owner_recall=m['different_surfaces_same_owner']['same_recall'],
                surface_embedding_cross_surface_recall=f['different_surfaces_same_owner']['same_recall'],
                component_Bcubed=r['components'],owner_detection=owners,normal_p95_deg=r['unoriented_normal_p95_deg'],
                boundary_F1=2*b['TP']/max(1,2*b['TP']+b['FP']+b['FN'])))
        assert total_points==2641757;summary[arm]=rows
        print('MATERIAL_COMPARISON_VERIFIED',arm,'all_points',total_points,'development',next(r for r in rows if r['source']==64),flush=True)
    write(dest/'receipt.json',dict(status='MATERIAL_COMPARISON_VERIFIED',summary=summary,
        all_checkpoint_and_prediction_hashes_match=True,all_original_point_mappings_preserved=True,
        optimizer_updates_run70=2336,full_contract_G=496,weld_extraction_not_rerun=True,
        full_contract_pass=False,source_sha256=digest(__file__),end_utc=utc()))


if __name__=='__main__':main()
