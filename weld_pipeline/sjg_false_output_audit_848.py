"""Posthoc partition of unmatched output children; never an inference filter."""
import argparse
import traceback
import numpy as np
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.contract_geometry_d3_409 import variants,child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment


def audit(decoded,target,score):
    matched={(x['predicted_parent_index'],x['predicted_child_index']) for x in score['matches']}
    references=[[reference_segment(c) for c in t['children']] for t in target['parents']]
    rows=[]
    for pi,parent in enumerate(decoded['parents']):
        possible=set();loose=set()
        for ti,truth in enumerate(target['parents']):
            if parent['child_count']!=truth['original_child_count'] or parent['is_closed']!=truth['is_closed']:continue
            for view in variants(parent,True):
                if view['gaps']!=truth['gaps']:continue
                for ci,(predicted_index,segment) in enumerate(view['segments']):
                    if child_error(segment,truth['children'][ci],references[ti][ci],
                        truth['is_closed'] and len(truth['children'])==1,False) is not None:
                        possible.add(predicted_index)
        for ci,segment in enumerate(parent['segments']):
            if ci in possible:loose.add(ci);continue
            if segment['kind']!='line':continue
            points=np.asarray(segment['points'])[[0,-1]]
            for truth in target['parents']:
                for child in truth['children']:
                    if child['original_kind']!='line' or child['weld_type']!=segment['weld_type']:continue
                    endpoints=np.asarray(child['world_representation']['endpoints_mm'])
                    if min(np.linalg.norm(points-endpoints,axis=1).max(),np.linalg.norm(points[::-1]-endpoints,axis=1).max())<=3:
                        loose.add(ci);break
                if ci in loose:break
        for ci in range(len(parent['segments'])):
            status='MATCHED_TP' if (pi,ci) in matched else (
                'VALID_CORRESPONDENCE_BUT_ASSIGNMENT_COMPETITION' if ci in possible else (
                'CHILD_LINE_GEOMETRY_POSSIBLE_PARENT_STRUCTURE_FAIL' if ci in loose else 'NO_CHILD_KIND_TYPE_ENDPOINT_MATCH'))
            rows.append(dict(predicted_parent=pi,predicted_child=ci,status=status))
    assert len(rows)==decoded['total_prediction_children']
    counts={key:sum(r['status']==key for r in rows) for key in ('MATCHED_TP','VALID_CORRESPONDENCE_BUT_ASSIGNMENT_COMPETITION',
        'CHILD_LINE_GEOMETRY_POSSIBLE_PARENT_STRUCTURE_FAIL','NO_CHILD_KIND_TYPE_ENDPOINT_MATCH')}
    assert counts['MATCHED_TP']==score['categories']['overall']['TP']
    assert sum(v for k,v in counts.items() if k!='MATCHED_TP')==score['categories']['overall']['FP']
    return dict(counts=counts,children=rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);base=OUT/'run57/parents';inventory=read(OUT/'inventory.json');rows=[]
    write(dest/'registered_protocol.json',dict(stage='OFFLINE_FALSE_OUTPUT_ERROR_ATTRIBUTION',sources=TRAIN+VALID+ARC,
        source_sha256=digest(__file__),original_G=496,GT_posthoc_only=True,no_runtime_filter=True,
        assignment_competition_not_all_certified_duplicates=True,optimizer_updates=0,full_contract_pass=False,start_utc=utc()))
    try:
        bindings={r['source']:r for r in read(base/'all_predictions_frozen.json')['sources']}
        for source in TRAIN+VALID+ARC:
            check_resources();binding=bindings[source];path=base/binding['file'];assert digest(path)==binding['sha256']
            target_path=ROOT/inventory[source]['target']['target_path'];assert digest(target_path)==inventory[source]['target']['target_sha256']
            result=audit(read(path),read(target_path),read(base/f'source_{source:02d}_strict_geometry.json'))
            write(dest/f'source_{source:02d}_false_output_audit.json',result);row=dict(source=source,**result['counts']);rows.append(row)
            print('FALSE_OUTPUT_ATTRIBUTION',row,flush=True)
        totals={k:sum(r[k] for r in rows) for k in rows[0] if k!='source'}
        write(dest/'receipt.json',dict(status='FALSE_OUTPUT_ATTRIBUTION_COMPLETE',rows=rows,totals=totals,
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
