"""Free sequence inference and strict child/parent geometry accounting."""
import argparse
import gc
import inspect
import time
import traceback
import numpy as np
import torch
from scipy.spatial import cKDTree
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_joint_data_816 import load_inputs
from weld_pipeline.sjg_joint_model_817 import JointEventModel
from weld_pipeline.sjg_sequence_model_820 import JointSequenceModel,configure_precision
from weld_pipeline.sjg_sequence_data_821 import sequence_features
from weld_pipeline.sjg_free_events_819 import decode_intervals,geometry_only_score
from weld_pipeline.contract_geometry_d3_409 import score_source

DEVICE='cpu'


def decoded_single_parent_hypotheses(output):
    parents=[]
    for index,p in enumerate(output['predictions']):
        endpoints=np.asarray(p['points']);length=np.linalg.norm(endpoints[1]-endpoints[0])
        points=np.linspace(endpoints[0],endpoints[1],max(2,int(np.ceil(length))+1))
        parents.append(dict(id=index,child_count=1,is_closed=False,gaps=[],
            segments=[dict(kind='line',weld_type=p['weld_type'],points=points)]))
    invalid=[dict(counted_false_children=1,predicted_weld_types=[0 if p['weld_type']=='horizontal' else 1],
        predicted_kinds=[1]) for p in output['invalid_intervals']]
    return dict(parents=parents,invalid_predictions=invalid,total_prediction_children=len(parents)+len(invalid))


def evaluate_outputs(dest):
    inventory=read(OUT/'inventory.json');frozen=read(dest/'all_predictions_frozen.json')['sources'];rows=[]
    for record in frozen:
        si=record['source'];target_path=ROOT/inventory[si]['target']['target_path']
        assert digest(target_path)==inventory[si]['target']['target_sha256']
        assert digest(dest/record['file'])==record['sha256']
        target=read(target_path);output=read(dest/record['file'])
        loose=geometry_only_score(output['predictions'],target);write(dest/f'source_{si:02d}_untyped_geometry.json',loose)
        strict=score_source(decoded_single_parent_hypotheses(output),target)
        write(dest/f'source_{si:02d}_strict_geometry.json',strict)
        row=dict(source=si,G=target['original_children'],predicted=len(output['predictions']),invalid=len(output['invalid_intervals']),
            untyped_child_geometry_matched=loose['matched_geometry'],strict=strict['categories']['overall'])
        rows.append(row);print('SEQUENCE_GEOMETRY_RESULT',row,flush=True)
    write(dest/'receipt.json',dict(status='FREE_SEQUENCE_STRICT_GEOMETRY_EVALUATED',rows=rows,
        original_G=496,full_contract_pass=False,E16_unknown=True,parent_assembly_not_learned=True,
        formal_direction_policy_unconfirmed=True,curves_missing=True,end_utc=utc()))
    return rows


def main(decoder=decode_intervals,score_outputs=True,model_factory=JointSequenceModel,context_factory=None):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--checkpoint-run',default='run41');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);configure_precision()
    assert read(OUT/args.checkpoint_run/'receipt.json')['status']=='CONTINUOUS_SEQUENCE_TRAINING_COMPLETE'
    joint_model=JointEventModel().to(DEVICE);sequence_model=model_factory().to(DEVICE)
    for model,path in ((joint_model,OUT/'run36/last.pt'),(sequence_model,OUT/args.checkpoint_run/'last.pt')):
        with path.open('rb') as h:state=torch.load(h,map_location=DEVICE)
        model.load_state_dict(state['model']);del state;model.eval()
    write(dest/'registered_protocol.json',dict(stage='FREE_CONTINUOUS_SEQUENCE_AND_CONTRACT_GEOMETRY',
        sources=TRAIN+VALID+ARC,original_G=496,source_sha256=digest(__file__),
        joint_checkpoint_sha256=digest(OUT/'run36/last.pt'),sequence_checkpoint_sha256=digest(OUT/args.checkpoint_run/'last.pt'),
        device=DEVICE,grid_step_mm=1.,TF32_disabled=True,profile_threshold=.5,foreground_threshold=.5,event_threshold=.5,
        decoder=decoder.__name__,decoder_source_sha256=digest(inspect.getfile(decoder)),
        sequence_model_class=model_factory.__name__,sequence_model_source_sha256=digest(inspect.getfile(model_factory)),
        extra_context=context_factory.__name__ if context_factory else None,
        context_source_sha256=digest(inspect.getfile(context_factory)) if context_factory else None,
        type_from_learned_logits=True,parent_hypothesis='ONE_CHILD_PER_PARENT_NOT_LEARNED',
        material_identity_unknown=True,no_duplicate_collapse=True,curves_missing=True,
        all_prediction_freezes_before_GT=True,STEP_reads=0,full_contract_pass=False,
        cached_input_features_for_development_only=True,not_end_to_end_speed=True,start_utc=utc()))
    try:
        inventory=read(OUT/'inventory.json');frozen=[]
        with torch.inference_mode():
            for si in TRAIN+VALID+ARC:
                check_resources();started=time.perf_counter();xyz,point,faces,profiles=load_inputs(si,inventory)
                graph_file=OUT/f'run35/source_{si:02d}_inputs.npz'
                binding=next(r for r in read(OUT/'run35/all_inputs_frozen.json')['sources'] if r['source']==si)
                assert digest(graph_file)==binding['sha256']
                with np.load(graph_file) as z:graph={k:torch.as_tensor(z[k],device=DEVICE) for k in z.files}
                h=joint_model.encode_faces(graph['face_x'],graph['neighbors'],graph['relation'])
                joints=[];probabilities=[]
                for begin in range(0,len(profiles),4096):
                    joint=joint_model.encode_joints(h,graph['pairs'][begin:begin+4096],graph['geometry'][begin:begin+4096])
                    joints.append(joint);probabilities.append(joint_model.relevance(joint).squeeze(-1).sigmoid().cpu().numpy())
                joints=torch.cat(joints);probability=np.concatenate(probabilities);selected=np.flatnonzero(probability>=.5)
                context=context_factory(xyz,faces,profiles,selected) if context_factory else None
                if context is not None:write(dest/f'source_{si:02d}_extra_context.json',context.describe())
                tree=cKDTree(xyz);predictions=[];invalid=[];queries=0
                field_folder=dest/f'source_{si:02d}_fields';field_folder.mkdir();field_files=[]
                for number,ni in enumerate(selected,1):
                    ni=int(ni);q,x=sequence_features(ni,profiles,xyz,point,tree)
                    condition=np.r_[joints[ni].cpu().numpy(),x.mean(0),x.std(0)].astype(np.float32)
                    if context is not None:x=np.c_[x,context(ni,q)]
                    logits,offset,type_logits=sequence_model.infer(torch.as_tensor(x,device=DEVICE)[None],
                        torch.as_tensor(condition,device=DEVICE)[None])
                    event_probability=logits[0].sigmoid().cpu().numpy();offset=offset[0].cpu().numpy()
                    type_probability=type_logits[0].softmax(-1).cpu().numpy()
                    field_file=field_folder/f'profile_{ni:06d}.npz'
                    np.savez_compressed(field_file,q_mm=q,probability=event_probability,offset=offset,type_probability=type_probability)
                    field_files.append(dict(profile=ni,file=str(field_file.relative_to(dest)),sha256=digest(field_file)))
                    intervals,bad=decoder(q,event_probability,offset);queries+=len(q)
                    for interval in intervals:
                        ids=(q>=interval['lo'])&(q<=interval['hi'])
                        if not ids.any():ids[np.abs(q-(interval['lo']+interval['hi'])/2).argmin()]=True
                        kind=int(type_probability[ids].mean(0).argmax());p=profiles[ni]
                        ends=np.asarray(p['p0'])+np.asarray([interval['lo'],interval['hi']])[:,None]*p['direction']
                        predictions.append(dict(profile=ni,points=ends.tolist(),kind='line',
                            weld_type=['horizontal','vertical'][kind],parent_hypothesis='single_child',
                            material_pair=None,attributes_status='UNKNOWN',candidate_probability=float(probability[ni]),**interval))
                    for item in bad:
                        index=int(np.abs(q-(item['lo']+item['hi'])/2).argmin())
                        invalid.append(dict(profile=ni,weld_type=['horizontal','vertical'][int(type_probability[index].argmax())],**item))
                    if number==1 or number%25==0 or number==len(selected):
                        print('FREE_SEQUENCE',si,number,'/',len(selected),'QUERIES',queries,'SEGMENTS',len(predictions),flush=True);check_resources()
                file=dest/f'source_{si:02d}_predictions.json';write(file,dict(predictions=predictions,invalid_intervals=invalid))
                write(dest/f'source_{si:02d}_fields.json',dict(files=field_files,all_input_only=True))
                record=dict(source=si,file=file.name,sha256=digest(file),profiles=len(selected),queries=queries,
                    predictions=len(predictions),invalid=len(invalid),cached_decode_seconds=time.perf_counter()-started)
                frozen.append(record);print('FREE_SEQUENCE_FROZEN',record,flush=True)
                del xyz,point,faces,profiles,graph,h,joints,tree;gc.collect()
                if DEVICE=='cuda':torch.cuda.empty_cache()
        write(dest/'all_predictions_frozen.json',dict(sources=frozen,all_before_GT=True))
        if score_outputs:evaluate_outputs(dest)
        else:write(dest/'receipt.json',dict(status='FREE_SEQUENCE_PREDICTIONS_FROZEN_SCORE_PENDING',
            original_G=496,actual_optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
