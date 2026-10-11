"""Input-only replay scores for every nonempty frozen pointer output."""
from pathlib import Path
import time
import numpy as np
import torch
from weld_pipeline.server_arc_boundary_pointer import ArcBoundaryPointer
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    started=time.perf_counter();torch.set_num_threads(2)
    key='new_ac2008a03607';pred=BASE/'arc_pointer_all_candidates_v1';pr=read(pred/'receipt.json');assert digest(pred/'protocol.json')==pr['protocol_sha256'];pp=read(pred/'protocol.json')
    modeldir=BASE/'arc_parent_feedback_model_valid_v1';assert digest(modeldir/'model.pt')==pp['model_sha256'];checkpoint=torch.load(modeldir/'model.pt',map_location='cpu',weights_only=False)
    features=BASE/'training_arc_event_features_v1'/key;events=BASE/'training_arc_vertex_events_v1'/key
    assert digest(features/'receipt.json')==pp['feature_receipt_sha256'];assert digest(events/'receipt.json')==pp['event_receipt_sha256']
    fr=read(features/'receipt.json');fm={r['face']:r for r in fr['faces']}
    mp=read(modeldir/'protocol.json');assert digest(modeldir/'protocol.json')==checkpoint['protocol_sha256'];assert digest('weld_pipeline/server_arc_boundary_pointer.py')==mp['base_module_sha256']
    model=ArcBoundaryPointer().cuda().eval();model.load_state_dict(checkpoint['state_dict']);rows=[]
    for record in pr['faces']:
        path=pred/record['file'];assert digest(path)==record['sha256'];selected=[r for r in read(path)['rows'] if r['selected_local_event_indices']]
        if not selected:continue
        f=fm[record['face']];assert digest(features/f['file'])==f['sha256'];assert digest(events/f"face_{record['face']}.npz")==f['event_file_sha256']
        x=np.load(features/f['file'],mmap_mode='r')
        with np.load(events/f"face_{record['face']}.npz") as z:offsets=z['offsets'];q=z['positions_mm']
        for row in selected:
            a,b=offsets[row['candidate']:row['candidate']+2];values=torch.as_tensor(np.clip((x[a:b]-checkpoint['mean'])/checkpoint['std'],-20,20),dtype=torch.float32,device='cuda');positions=torch.as_tensor(q[a:b],dtype=torch.float64,device='cuda');actions=[];previous=-1
            with torch.no_grad():
                embedded,state=model.prepare(values)
                for chosen in row['selected_local_event_indices']+[int(b-a)]:
                    scores,state=model.step(embedded,state,previous,positions);assert int(scores.argmax())==chosen,'Frozen prediction replay diverged'
                    logp=torch.log_softmax(scores,0);finite=torch.isfinite(logp);entropy=float(-(logp[finite].exp()*logp[finite]).sum());prob=float(logp[chosen].exp())
                    top=scores.topk(min(2,len(scores))).values;gap=float(top[0]-top[1]) if len(top)>1 and torch.isfinite(top[1]) else None
                    actions.append(dict(event_index=chosen if chosen<b-a else None,chosen_probability=prob,entropy=entropy,top_two_logit_margin=gap,stop_probability=float(logp[-1].exp())))
                    previous=chosen
            probabilities=[v['chosen_probability'] for v in actions]
            rows.append(dict(face=record['face'],candidate=row['candidate'],actions=actions,min_action_probability=min(probabilities),mean_action_log_probability=float(np.log(np.maximum(probabilities,1e-300)).mean()),event_count=int(b-a),selected_interval_length_mm=float(q[a+row['selected_local_event_indices'][-1]]-q[a+row['selected_local_event_indices'][0]])))
    out=BASE/'arc_pointer_output_confidence_v1';out.mkdir(exist_ok=False)
    write(out/'scores.json',dict(rows=rows,prediction_receipt_sha256=digest(pred/'receipt.json'),model_sha256=pp['model_sha256'],scope='All nonempty frozen outputs replayed with exact argmax parity, including invalid one-boundary paths. No target reads, calibration, threshold, filtering or deployment claim.',GT_reads=0,DEV_reads=0))
    write(out/'receipt.json',dict(status='INPUT_ONLY_POINTER_CONFIDENCE_FROZEN',outputs=len(rows),scores_sha256=digest(out/'scores.json'),seconds=time.perf_counter()-started,GT_reads=0))
    print('FROZEN',len(rows),flush=True)

if __name__=='__main__':main()
