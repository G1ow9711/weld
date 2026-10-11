"""Resume only parent inference after run47 completed training, not training."""
import argparse
import json
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_parent_model_831 import IntervalParentModel
from weld_pipeline.sjg_parent_decode_832 import assemble,suppress_parent_hypotheses,decoded
from weld_pipeline.contract_geometry_d3_409 import score_source
from weld_pipeline.sjg_train_parent_833 import load_input,DATA,BASE


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);checkpoint=OUT/'run47/last.pt'
    with checkpoint.open('rb') as h:state=torch.load(h,map_location='cpu')
    epochs=[json.loads(x) for x in (OUT/'run47/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert state['updates']==epochs[-1]['actual_updates']==1152 and state['epoch']==epochs[-1]['epoch']==24 and len(epochs)==24
    model=IntervalParentModel();model.load_state_dict(state['model']);del state;model.eval()
    write(OUT/'run47/training_phase_verification.json',dict(status='TRAINING_COMPLETE_EVALUATION_SAVE_FAILED',
        actual_optimizer_updates=1152,epochs=24,checkpoint_sha256=digest(checkpoint),no_retraining_needed=True))
    write(dest/'registered_protocol.json',dict(stage='PARENT_INFERENCE_RESUME_ONLY',sources=TRAIN+VALID+ARC,
        original_G=496,actual_optimizer_updates=0,training_run='run47',training_updates=1152,training_epochs=24,
        checkpoint_sha256=digest(checkpoint),source_sha256=digest(__file__),
        decoder_sha256=digest(ROOT/'weld_pipeline/sjg_parent_decode_832.py'),
        repair='NumPy bool converted to Python bool for selection JSON only',
        node_threshold=.5,link_threshold=.5,shared_threshold=.5,parent_NMS_mm=1.,
        source_outputs_fixed='run45',GT_not_runtime_input=True,full_contract_pass=False,start_utc=utc()))
    try:
        frozen=[]
        with torch.inference_mode():
            for source in TRAIN+VALID+ARC:
                check_resources();graph=load_input(source);h=model.encode(graph['nodes']);node_p=model.node_logits(h).sigmoid().numpy()
                chunks=[]
                for begin in range(0,len(graph['pairs']),4096):
                    chunks.append(model.edge_logits(h,graph['pairs'][begin:begin+4096],graph['relation'][begin:begin+4096]).sigmoid().numpy())
                edge_p=np.concatenate(chunks) if chunks else np.empty((0,2))
                raw=read(BASE/f'source_{source:02d}_predictions.json');assert not raw['invalid_intervals']
                parents,detail=assemble(raw['predictions'],graph['pairs'].numpy(),node_p,edge_p)
                kept,suppressed=suppress_parent_hypotheses(parents,1.)
                before=dest/f'source_{source:02d}_parents_before_nms.json';path=dest/f'source_{source:02d}_parents.json'
                write(before,decoded(parents));write(path,decoded(kept))
                write(dest/f'source_{source:02d}_selection.json',dict(node_probability=node_p.tolist(),**detail,suppressed_parent_hypotheses=suppressed))
                record=dict(source=source,file=path.name,sha256=digest(path),before_nms_file=before.name,before_nms_sha256=digest(before),
                    input_intervals=len(raw['predictions']),retained_intervals=sum(p['child_count'] for p in parents),
                    final_children=sum(p['child_count'] for p in kept),parents=len(kept),suppressed_parents=len(suppressed))
                frozen.append(record);print('PURE_PARENT_OUTPUT_FROZEN',record,flush=True)
        write(dest/'all_predictions_frozen.json',dict(sources=frozen,all_before_test_GT=True))
        inventory=read(OUT/'inventory.json');rows=[]
        for record in frozen:
            si=record['source'];target_path=ROOT/inventory[si]['target']['target_path']
            assert digest(target_path)==inventory[si]['target']['target_sha256'];target=read(target_path)
            assert digest(dest/record['file'])==record['sha256']
            score=score_source(read(dest/record['file']),target);before=score_source(read(dest/record['before_nms_file']),target)
            write(dest/f'source_{si:02d}_strict_geometry.json',score);write(dest/f'source_{si:02d}_strict_before_nms.json',before)
            row=dict(source=si,strict=score['categories']['overall'],before_nms=before['categories']['overall']);rows.append(row)
            print('PURE_PARENT_STRICT_RESULT',row,flush=True)
        write(dest/'receipt.json',dict(status='PURE_PARENT_TRAIN_AND_STRICT_GEOMETRY_COMPLETE',actual_optimizer_updates=0,
            training_run='run47',training_optimizer_updates=1152,rows=rows,end_utc=utc(),
            full_contract_pass=False,curves_missing=True,material_E16_unresolved=True))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=0,end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
