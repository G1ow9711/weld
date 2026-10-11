"""Evaluate both registered endpoint arms with one frozen interval/parent head."""
import argparse
import gc
import sys
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline import sjg_free_sequence_823 as free_sequence
from weld_pipeline.sjg_event_pairs_824 import decode_event_pairs
from weld_pipeline.sjg_parent_inputs_829 import interval_inputs,continuation_inputs
from weld_pipeline.sjg_parent_model_831 import IntervalParentModel
from weld_pipeline.sjg_parent_decode_832 import assemble,suppress_parent_hypotheses,decoded
from weld_pipeline.contract_geometry_d3_409 import score_source


def freeze_parent(base,dest):
    dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    checkpoint=OUT/'run47/last.pt';model=IntervalParentModel()
    with checkpoint.open('rb') as h:state=torch.load(h,map_location='cpu')
    assert state['updates']==1152 and state['epoch']==24
    model.load_state_dict(state['model']);del state;model.eval();frozen=[]
    write(dest/'registered_protocol.json',dict(stage='FROZEN_PARENT_CROSS_INPUT_EVALUATION',
        actual_optimizer_updates=0,checkpoint_sha256=digest(checkpoint),base=str(base.relative_to(OUT)),
        source_sha256=digest(__file__),node_threshold=.5,link_threshold=.5,shared_threshold=.5,parent_NMS_mm=1.,
        all_prediction_freezes_before_test_GT=True,GT_not_input=True,full_contract_pass=False))
    with torch.inference_mode():
        for source in TRAIN+VALID+ARC:
            check_resources();children,profiles,nodes=interval_inputs(source,base)
            assert not read(base/f'source_{source:02d}_predictions.json')['invalid_intervals'], 'Do not silently omit invalid emitted predictions'
            pairs,relation=continuation_inputs(children,profiles)
            path=dest/f'source_{source:02d}_inputs.npz';np.savez_compressed(path,nodes=nodes,pairs=pairs,relation=relation)
            h=model.encode(torch.from_numpy(nodes));node_p=model.node_logits(h).sigmoid().numpy();chunks=[]
            for start in range(0,len(pairs),4096):
                chunks.append(model.edge_logits(h,torch.from_numpy(pairs[start:start+4096]),
                    torch.from_numpy(relation[start:start+4096])).sigmoid().numpy())
            edge_p=np.concatenate(chunks) if chunks else np.empty((0,2))
            parents,detail=assemble(children,pairs,node_p,edge_p);kept,suppressed=suppress_parent_hypotheses(parents,1.)
            before=dest/f'source_{source:02d}_parents_before_nms.json';final=dest/f'source_{source:02d}_parents.json'
            write(before,decoded(parents));write(final,decoded(kept))
            write(dest/f'source_{source:02d}_selection.json',dict(node_probability=node_p.tolist(),**detail,suppressed_parent_hypotheses=suppressed))
            record=dict(source=source,file=final.name,sha256=digest(final),before_nms_file=before.name,
                before_nms_sha256=digest(before),input_file=path.name,input_sha256=digest(path),input_intervals=len(children),
                predicted_children=sum(p['child_count'] for p in kept))
            frozen.append(record);print('FROZEN_PARENT_ARM',base.name,record,flush=True)
            del children,profiles,nodes,pairs,relation,h;gc.collect()
    write(dest/'all_predictions_frozen.json',dict(sources=frozen,all_before_test_GT=True))
    write(dest/'receipt.json',dict(status='FROZEN_PARENT_OUTPUTS_SCORE_PENDING',actual_optimizer_updates=0,
        full_contract_pass=False,end_utc=utc()))


def evaluate_parent(dest):
    frozen=read(dest/'all_predictions_frozen.json')['sources'];inventory=read(OUT/'inventory.json');rows=[]
    for record in frozen:
        source=record['source'];target_path=ROOT/inventory[source]['target']['target_path']
        assert digest(target_path)==inventory[source]['target']['target_sha256']
        target=read(target_path);assert digest(dest/record['file'])==record['sha256']
        assert digest(dest/record['before_nms_file'])==record['before_nms_sha256']
        score=score_source(read(dest/record['file']),target)
        before=score_source(read(dest/record['before_nms_file']),target)
        write(dest/f'source_{source:02d}_strict_geometry.json',score)
        row=dict(source=source,strict=score['categories']['overall'],before_nms=before['categories']['overall']);rows.append(row)
        print('FROZEN_PARENT_RESULT',dest.name,row,flush=True)
    write(dest/'receipt.json',dict(status='FROZEN_PARENT_GEOMETRY_EVALUATED',rows=rows,actual_optimizer_updates=0,
        training_run='run47',full_contract_pass=False,material_E16_unknown=True,curves_missing=True,end_utc=utc()))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    assert read(OUT/'run49/receipt.json')['status']=='TWO_ARM_ENDPOINT_TRAINING_COMPLETE'
    write(dest/'registered_protocol.json',dict(stage='REGISTERED_TWO_ARM_FREE_OUTPUT_COMPARISON',
        train_runs=['run49/control','run49/balanced'],sources=TRAIN+VALID+ARC,original_G=496,
        actual_optimizer_updates=0,source_sha256=digest(__file__),
        free_driver_sha256=digest(ROOT/'weld_pipeline/sjg_free_sequence_823.py'),
        features_sha256=digest(ROOT/'weld_pipeline/sjg_parent_inputs_829.py'),
        parent_decoder_sha256=digest(ROOT/'weld_pipeline/sjg_parent_decode_832.py'),
        no_threshold_or_checkpoint_selection=True,all_arms_evaluated=True,full_contract_pass=False,start_utc=utc()))
    comparison={};saved_argv=sys.argv[:]
    try:
        for arm in ('control','balanced'):
            sys.argv=[saved_argv[0],'--run',str((dest/arm).relative_to(OUT)),'--checkpoint-run',f'run49/{arm}']
            free_sequence.main(decode_event_pairs,score_outputs=False)
            freeze_parent(dest/arm,dest/(arm+'_parents'))
            gc.collect()
        write(dest/'both_arms_predictions_frozen.json',dict(all_before_any_evaluation_GT=True,
            manifests={name:digest(dest/name/'all_predictions_frozen.json')
                for name in ('control','balanced','control_parents','balanced_parents')}))
        for arm in ('control','balanced'):
            free_rows=free_sequence.evaluate_outputs(dest/arm)
            rows=evaluate_parent(dest/(arm+'_parents'))
            comparison[arm]=dict(free=free_rows,parents=rows)
        write(dest/'receipt.json',dict(status='TWO_ARM_FREE_AND_FROZEN_PARENT_COMPARISON_COMPLETE',
            comparison=comparison,actual_optimizer_updates=0,original_G=496,
            full_contract_pass=False,E16_unknown=True,curves_missing=True,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_comparison=comparison,error=traceback.format_exc(),end_utc=utc()));raise
    finally:sys.argv=saved_argv


if __name__=='__main__':main()
