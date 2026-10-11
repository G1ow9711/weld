"""Publish globally paired intervals and independently assemble parent paths."""
import argparse
import gc
import os
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_pair_model_845 import IntervalPairModel,match_intervals
from weld_pipeline.sjg_parent_model_831 import IntervalParentModel
from weld_pipeline.sjg_parent_inputs_829 import continuation_inputs
from weld_pipeline.sjg_parent_decode_832 import assemble,suppress_parent_hypotheses,decoded
from weld_pipeline.sjg_free_sequence_823 import evaluate_outputs
from weld_pipeline.sjg_endpoint_compare_837 import evaluate_parent

DATA=OUT/'run55'


def evaluate_frozen(dest,checkpoint_run):
    free_rows=evaluate_outputs(dest/'free');parent_rows=evaluate_parent(dest/'parents')
    parent_receipt=read(dest/'parents/receipt.json');parent_receipt.update(interval_gate_training_run=checkpoint_run,
        training_run_is_parent_link_head_only=True);write(dest/'parents/receipt.json',parent_receipt)
    receipt=read(dest/'receipt.json');receipt.update(status='LEARNED_INTERVAL_PAIRING_AND_PARENT_EVALUATED',
        free=free_rows,parents=parent_rows,end_utc=utc());write(dest/'receipt.json',receipt)
    return receipt


def main(checkpoint_run='run56',model_factory=IntervalPairModel,score_outputs=True,data=DATA):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);free=dest/'free';parents_folder=dest/'parents';free.mkdir();parents_folder.mkdir()
    torch.set_num_threads(2);pair_path=OUT/checkpoint_run/'last.pt';parent_path=OUT/'run47/last.pt'
    assert read(OUT/checkpoint_run/'receipt.json')['status']=='INTERVAL_PAIR_TRAINING_COMPLETE'
    pair_model=model_factory();parent_model=IntervalParentModel()
    for model,path in ((pair_model,pair_path),(parent_model,parent_path)):
        with path.open('rb') as h:state=torch.load(h,map_location='cpu')
        model.load_state_dict(state['model']);del state;model.eval()
    protocol=dict(stage='GLOBAL_LEARNED_PAIRING_AND_PARENT_GEOMETRY',sources=TRAIN+VALID+ARC,original_G=496,
        optimizer_updates=0,pair_training_run=checkpoint_run,pair_checkpoint_sha256=digest(pair_path),parent_link_checkpoint_sha256=digest(parent_path),
        source_sha256=digest(__file__),pairing_sha256=digest(ROOT/'weld_pipeline/sjg_pair_model_845.py'),
        parent_decoder_sha256=digest(ROOT/'weld_pipeline/sjg_parent_decode_832.py'),
        learned_interval_gate_replaces_old_parent_node_gate=True,old_parent_link_head_frozen=True,
        candidate_data=str(data.relative_to(OUT)),candidate_manifest_sha256=digest(data/'all_inputs_frozen.json'),
        all_candidates_considered=True,unmatched_score=0.,positive_log_odds_only=True,
        no_topK=True,no_GT_ID_inputs=True,all_predictions_before_GT=True,
        geometric_parent_NMS_mm=1.,cached_input_features_development_only=True,
        no_end_to_end_timing_claim=True,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);write(parents_folder/'registered_protocol.json',protocol)
    try:
        bindings={r['source']:r for r in read(data/'all_inputs_frozen.json')['sources']};frozen=[];parent_freezes=[];selections=[]
        with torch.inference_mode():
            for source in TRAIN+VALID+ARC:
                check_resources();binding=bindings[source];path=data/binding['file'];assert digest(path)==binding['sha256']
                child_path=data/f'source_{source:02d}_predictions.json';assert digest(child_path)==binding['candidate_sha256']
                children=read(child_path)['predictions']
                with np.load(path) as z:features=z['features'];nodes=z['nodes']
                logits=np.concatenate([pair_model(torch.from_numpy(features[start:start+4096])).numpy()
                    for start in range(0,len(features),4096)]) if len(features) else np.empty(0)
                selected=match_intervals(children,logits);probability=torch.from_numpy(logits).sigmoid().numpy()
                chosen=[dict(children[i],parent_hypothesis='learned_endpoint_matching',
                    interval_probability=float(probability[i]),candidate_index=int(i)) for i in selected]
                path=free/f'source_{source:02d}_predictions.json';write(path,dict(predictions=chosen,invalid_intervals=[]))
                fields=read(data/f'source_{source:02d}_fields.json')['files']
                write(free/f'source_{source:02d}_fields.json',dict(files=[dict(x,file=os.path.relpath(data/x['file'],free)) for x in fields]))
                frozen.append(dict(source=source,file=path.name,sha256=digest(path),candidates=len(children),predictions=len(chosen)))
                selection=dict(source=source,candidates=len(children),positive_logit_candidates=int((logits>0).sum()),
                    selected_candidates=selected,logits=logits.tolist())
                write(free/f'source_{source:02d}_selection.json',selection);selections.append({k:v for k,v in selection.items() if k not in ('logits','selected_candidates')})
                profile_path=OUT/f'run34/source_{source:02d}/profiles.json';assert digest(profile_path)==binding['profiles_sha256'];profiles=read(profile_path)
                pairs,relation=continuation_inputs(chosen,profiles);selected_nodes=torch.from_numpy(nodes[np.asarray(selected,np.int64)])
                h=parent_model.encode(selected_nodes);node_p=probability[selected];chunks=[]
                for start in range(0,len(pairs),4096):
                    chunks.append(parent_model.edge_logits(h,torch.from_numpy(pairs[start:start+4096]),torch.from_numpy(relation[start:start+4096])).sigmoid().numpy())
                edge_p=np.concatenate(chunks) if chunks else np.empty((0,2))
                parents,detail=assemble(chosen,pairs,node_p,edge_p);kept,suppressed=suppress_parent_hypotheses(parents,1.)
                before=parents_folder/f'source_{source:02d}_parents_before_nms.json';final=parents_folder/f'source_{source:02d}_parents.json'
                write(before,decoded(parents));write(final,decoded(kept))
                write(parents_folder/f'source_{source:02d}_selection.json',dict(node_probability=node_p.tolist(),
                    node_probability_from=checkpoint_run+'_interval_model',**detail,suppressed_parent_hypotheses=suppressed))
                record=dict(source=source,file=final.name,sha256=digest(final),before_nms_file=before.name,
                    before_nms_sha256=digest(before),input_intervals=len(chosen),predicted_children=sum(p['child_count'] for p in kept))
                parent_freezes.append(record);print('GLOBAL_PAIR_AND_PARENT_FROZEN',record,flush=True)
                del children,profiles,features,nodes,pairs,relation,h;gc.collect()
        write(free/'all_predictions_frozen.json',dict(sources=frozen,all_before_GT=True))
        write(parents_folder/'all_predictions_frozen.json',dict(sources=parent_freezes,all_before_test_GT=True))
        write(dest/'all_outputs_frozen_before_GT.json',dict(free_sha256=digest(free/'all_predictions_frozen.json'),
            parents_sha256=digest(parents_folder/'all_predictions_frozen.json'),all_before_GT=True))
        write(dest/'receipt.json',dict(status='INTERVAL_AND_PARENT_OUTPUTS_FROZEN',
            selections=selections,original_G=496,optimizer_updates=0,full_contract_pass=False,
            curves_missing=True,material_E16_unresolved=True,end_utc=utc()))
        if score_outputs:evaluate_frozen(dest,checkpoint_run)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',optimizer_updates=0,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
