"""Freeze all raw-XYZ material relations before reading offline reference owners."""
import argparse
import builtins
import gc
import io
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel
from weld_pipeline.sjg_material_runtime_861 import infer_material_relations,RANKS


def relation_counts(distance,truth_same,eligible):
    observed=np.isfinite(distance);same=distance<=.5
    tp=int((eligible&observed&same&truth_same).sum());fp=int((eligible&observed&same&~truth_same).sum())
    fn=int((eligible&observed&~same&truth_same).sum());tn=int((eligible&observed&~same&~truth_same).sum())
    unseen_same=int((eligible&~observed&truth_same).sum());unseen_different=int((eligible&~observed&~truth_same).sum())
    total=int(eligible.sum());positive=int((eligible&truth_same).sum());negative=total-positive
    return dict(reference_pairs=total,TP=tp,FP=fp,FN_observed=fn,TN=tn,unobserved_same=unseen_same,unobserved_different=unseen_different,
        correct_over_all_known_reference_pairs=(tp+tn)/total if total else None,
        same_recall=tp/positive if positive else None,different_recall=tn/negative if negative else None,
        same_precision=tp/(tp+fp) if tp+fp else None)


def component_metrics(component,owner):
    known=owner>=0;pred=component[known].astype(np.int64);truth=owner[known];n=len(pred)
    if not n:return dict(reference_points=0)
    _,label=np.unique(truth,return_inverse=True);classes=int(label.max())+1
    codes,count=np.unique(pred*classes+label,return_counts=True);row=codes//classes;col=codes%classes
    row_sum=np.bincount(pred);col_sum=np.bincount(label);squared=count.astype(np.float64)**2
    precision=float((squared/row_sum[row]).sum()/n);recall=float((squared/col_sum[col]).sum()/n)
    return dict(reference_points=n,unknown_reference_points=int((~known).sum()),predicted_components_all_points=int(component.max())+1,
        reference_owners=classes,B_cubed_precision=precision,B_cubed_recall=recall,B_cubed_F1=2*precision*recall/max(precision+recall,1e-12),
        largest_component_fraction_all_points=float(np.bincount(component).max()/len(component)),
        single_membership_connected_components_diagnostic_only=True,multi_owner_contacts_not_resolved=True)


def score_frozen(dest):
    teachers={r['source']:r for r in read(OUT/'run22/label_summary.json')};rows=[]
    for record in read(dest/'all_predictions_frozen.json')['sources']:
        source=record['source'];path=dest/record['file'];assert digest(path)==record['sha256']
        if source not in teachers or 'teacher_file' not in teachers[source]:
            rows.append(dict(source=source,status='NO_OFFLINE_STRUCTURE_REFERENCE',original_points=record['points']));continue
        teacher=OUT/'run22'/teachers[source]['teacher_file'];assert digest(teacher)==teachers[source]['teacher_sha256']
        with np.load(path) as p,np.load(teacher) as t:
            owner=t['reference_volume'];instance=t['instance'];neighbor=p['neighbors'];other=neighbor.clip(0)
            valid=neighbor>=0;eligible=valid&(owner[:,None]>=0)&(owner[other]>=0);same=owner[:,None]==owner[other]
            sf=(instance[:,None]>=0)&(instance[other]>=0);equal_surface=instance[:,None]==instance[other]
            groups={'all':eligible,'same_surface':eligible&same&sf&equal_surface,
                'different_surfaces_same_owner':eligible&same&sf&~equal_surface,
                'same_owner_surface_unresolved':eligible&same&~sf,'different_owner':eligible&~same}
            metrics={head:{name:relation_counts(p[head],same,mask) for name,mask in groups.items()} for head in ['material_distance','surface_distance']}
            by_rank={str(rank):relation_counts(p['material_distance'][:,j],same[:,j],eligible[:,j]) for j,rank in enumerate(RANKS)}
            known=t['instance_known'];dot=np.abs((p['normal'][known]*t['normal'][known]).sum(1));angle=np.rad2deg(np.arccos(np.clip(dot,0,1)))
            by=t['boundary_target'];bm=by>=0;positive=by==1;bp=p['boundary']>=.5
            row=dict(source=source,status='OFFLINE_REFERENCE_MATERIAL_DIAGNOSTIC',original_points=len(owner),
                input_pair_slots=neighbor.size,valid_neighbor_pairs=int(valid.sum()),reference_known_pairs=int(eligible.sum()),
                unknown_reference_pairs=int((valid&~eligible).sum()),unobserved_within_block_pairs=int((valid&~np.isfinite(p['material_distance'])).sum()),
                relations=metrics,material_by_neighbor_rank=by_rank,components=component_metrics(p['material_component'],owner),
                unoriented_normal_p95_deg=float(np.quantile(angle,.95)),normal_known_points=int(known.sum()),
                boundary=dict(TP=int((bm&positive&bp).sum()),FP=int((bm&~positive&bp).sum()),FN=int((bm&positive&~bp).sum()),TN=int((bm&~positive&~bp).sum())),
                formal_material_truth=False,full_contract_pass=False)
        rows.append(row);write(dest/f'source_{source:02d}_offline_metrics.json',row)
        print('MATERIAL_REFERENCE_RESULT',source,'relation',metrics['material_distance']['all'],'component',row['components'],flush=True)
    write(dest/'offline_material_metrics.json',rows);return rows


def main(checkpoint_run='run68',score_outputs=True):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);checkpoint=OUT/checkpoint_run/'last.pt'
    assert read(OUT/checkpoint_run/'receipt.json')['status']=='PTV3_MATERIAL_MULTITASK_TRAINING_COMPLETE'
    write(dest/'registered_protocol.json',dict(stage='RAW_XYZ_MATERIAL_RELATIONS_FREEZE_THEN_REFERENCE_SCORE',sources=TRAIN+VALID+ARC,
        checkpoint_run=checkpoint_run,checkpoint_sha256=digest(checkpoint),source_sha256=digest(__file__),runtime_sha256=digest(ROOT/'weld_pipeline/sjg_material_runtime_861.py'),
        neighbor_ranks=RANKS,fixed_material_distance_threshold=.5,all_original_points_preserved=True,
        cross_block_embedding_comparisons=False,unobserved_relations_remain_unknown=True,all8_before_teacher_reads=True,
        components_are_diagnostic_not_final_multi_material_assignment=True,full_contract_G=496,full_contract_pass=False,start_utc=utc()))
    try:
        model=SurfaceMaterialModel().cuda()
        with checkpoint.open('rb') as h:state=torch.load(h,map_location='cuda')
        model.load_state_dict(state['model']);del state;model.eval();inventory=read(OUT/'inventory.json');frozen=[]
        for source in TRAIN+VALID+ARC:
            check_resources();raw=ROOT/inventory[source]['raw']['original_pcd_path'];xyz=read_xyz_pcd(raw)*1000.
            opened=[];original_open=builtins.open;original_io_open=io.open
            def guarded(factory):
                def call(path,*a,**kw):
                    if isinstance(path,(str,bytes,Path)):
                        text=str(path).lower();opened.append(text)
                        if text.endswith(('.step','.stp')) or any(word in text for word in ['teacher','supervision','geometry.json']):raise RuntimeError('Forbidden inference input: '+str(path))
                    return factory(path,*a,**kw)
                return call
            started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            try:
                builtins.open=guarded(original_open);io.open=guarded(original_io_open)
                result=infer_material_relations(model,xyz,resource_check=check_resources,
                    progress=lambda done,total,core,known,pairs:print('RAW_MATERIAL_BLOCK',source,done,'/',total,'CORE',core,'OBSERVED_RELATIONS',known,'/',pairs,flush=True) if done==1 or done%20==0 or done==total else None)
            finally:builtins.open=original_open;io.open=original_io_open
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            file=dest/f'source_{source:02d}_material_relations.npz';np.savez_compressed(file,**result)
            record=dict(source=source,file=file.name,sha256=digest(file),raw_sha256=digest(raw),points=len(xyz),
                inference_seconds=seconds,gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                every_original_point_one_core=bool(np.all(result['core_coverage']==1)),component_count=int(result['component_count']),
                python_file_opens_during_numeric_inference=opened,python_STEP_or_teacher_reads=0,native_file_access_not_instrumented=True)
            frozen.append(record);write(dest/f'source_{source:02d}_freeze.json',record);print('RAW_MATERIAL_FROZEN',record,flush=True)
            del xyz,result;gc.collect();torch.cuda.empty_cache()
        write(dest/'all_predictions_frozen.json',dict(sources=frozen,all_before_teacher_reads=True))
        rows=score_frozen(dest) if score_outputs else None
        write(dest/'receipt.json',dict(status='EIGHT_RAW_MATERIAL_RELATIONS_FROZEN_AND_EVALUATED' if score_outputs else 'EIGHT_RAW_MATERIAL_RELATIONS_FROZEN_SCORE_PENDING',rows=rows,sources=frozen,
            optimizer_updates=0,weld_extraction_not_rerun=True,full_contract_G=496,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),full_contract_pass=False,end_utc=utc()));raise


if __name__=='__main__':main()
