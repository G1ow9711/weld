"""Fixed-threshold TRAIN path coverage and no-update gradient diagnosis."""
import argparse,gc
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_hierarchical_joint_model_1028 import HierarchicalJointModel
from weld_pipeline.sjg_balanced_hierarchy_1043 import load_model
from weld_pipeline.sjg_hierarchical_gate_audit_1035 import metrics
from weld_pipeline.sjg_hierarchical_train_1031 import entropy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--training-run',default='run218')
    ap.add_argument('--checkpoint',default='final.pt');a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    checkpoint=OUT/a.training_run/a.checkpoint;state=torch.load(checkpoint,map_location='cpu')
    model=load_model(state);model.eval();torch.set_num_threads(2)
    balanced=state.get('architecture')=='balanced_normalized_hierarchy'
    manifest=read(OUT/'run206/training_manifest.json');bindings={b['source_key']:b for b in manifest['static_bindings']}
    cohort={c['source_key']:c for c in read(OUT/'run98/local_cohort.json')['cases']};rows=[]
    relation_priors=[];family_priors=[]
    for case in manifest['cases']:
        key=case['source_key'];b=bindings[key]
        assert digest(OUT/'run206'/b['file'])==b['sha256'] and digest(OUT/'run206'/case['file'])==case['sha256']
        with np.load(OUT/'run206'/b['file']) as z:s={k:torch.from_numpy(z[k]) for k in z.files}
        with np.load(OUT/'run206'/case['file']) as z:ry=z['relation_y'];fy=z['family_y']
        relation_priors.append(float(ry.mean()));family_priors.append(float(fy.mean()))
        with torch.inference_mode():
            r,rl=model.relations(s['relation_x']);f,fl=model.families(r,s['family_relation'],s['family_x'])
            rp=rl.sigmoid().numpy();fp=fl.sigmoid().numpy()
        target=cohort[key];assert digest(OUT/target['target_file'])==target['target_sha256']
        children=[c for p in read(OUT/target['target_file'])['parents'] for c in p['children']]
        ids={c['original_child_id'] for c in children};arcs={c['original_child_id'] for c in children if c['representation']['kind']=='arc'}
        with np.load(OUT/f'run201/source_{key}_joint_mapping.npz') as z:pr=z['profile_relation'];pf=z['profile_family']
        sup=read(OUT/f'run186/source_{key}_supervision.json');representable=set();relation_pass=set();both_pass=set()
        for profile,events in sup['by_profile'].items():
            i=int(profile)
            for event in events:
                child=event['original_child_id'];representable.add(child)
                if rp[pr[i]]>=.5:
                    relation_pass.add(child)
                    if fp[pf[i]]>=.5:both_pass.add(child)
        assert representable==ids,'Registered full-child paths differ from original TRAIN identities'
        row=dict(source=key,G=len(ids),arc_G=len(arcs),relation=metrics(rp,ry),family=metrics(fp,fy),
            relation_complete_child_coverage=len(relation_pass),relation_and_family_complete_child_coverage=len(both_pass),
            relation_arc_coverage=len(relation_pass&arcs),both_arc_coverage=len(both_pass&arcs),
            selected_relations=int((rp>=.5).sum()),selected_families_after_relation=int(((fp>=.5)&(rp[s['family_relation'].numpy()]>=.5)).sum()),
                known_geometric_compatibility_labels=True,material_identity_truth_not_claimed=True,
                score_semantics=getattr(model,'score_semantics','unweighted_sigmoid_model_score'),legacy_probability_field_names_do_not_imply_calibration=True)
        if key=='21':
            model.zero_grad(set_to_none=True);_,logits=model.relations(s['relation_x']);y=torch.from_numpy(ry)
            loss=F.binary_cross_entropy_with_logits(logits,y,reduction='none')/len(y)/entropy(read(OUT/'run207/population_priors.json')['relation'])
            if balanced:
                weight=torch.where(y==1,.5/y.sum(),.5/(1-y).sum())/np.log(2)
                loss=F.binary_cross_entropy_with_logits(logits,y,reduction='none')*weight
            named=list(model.relation_encoder.named_parameters());params=[model.relation_presence.weight,model.relation_presence.bias]+[p for _,p in named]
            gpos=torch.autograd.grad(loss[y.bool()].sum(),[logits]+params,retain_graph=True)
            gneg=torch.autograd.grad(loss[~y.bool()].sum(),params)
            row['gradient_control']=dict(optimizer_updates=0,positive_logit_gradient_all_negative=bool((gpos[0][y.bool()]<0).all()),
                positive_weight_abs_sum=float(gpos[1].abs().sum()),positive_bias_gradient=float(gpos[2].sum()),
                negative_weight_abs_sum=float(gneg[0].abs().sum()),negative_bias_gradient=float(gneg[1].sum()),
                positive_hidden_gradient_abs_sum=sum(float(g.abs().sum()) for g in gpos[3:]))
            del logits,loss,gpos,gneg
        rows.append(row);write(dest/'progress.json',dict(cases=rows))
        print('COMPLETE_GATE',key,'REL',len(relation_pass),'BOTH',len(both_pass),'/',len(ids),'AP',round(row['relation']['average_precision'],4),flush=True)
        del s,r,rl,f,fl,ry,fy,rp,fp;gc.collect()
    relation=sum(x['relation_complete_child_coverage'] for x in rows);arc=sum(x['relation_arc_coverage'] for x in rows)
    write(dest/'receipt.json',dict(status='COMPLETE_TRAIN_GATE_DIAGNOSIS',checkpoint_sha256=digest(checkpoint),checkpoint_updates=state['updates'],
        cases=rows,relation_coverage=relation,both_coverage=sum(x['relation_and_family_complete_child_coverage'] for x in rows),
        relation_arc_coverage=arc,G=363,arc_G=3,intermediate_gate_pass=relation>=360 and arc==3,
        relation_macro_prior=float(np.mean(relation_priors)),family_macro_prior=float(np.mean(family_priors)),
        original_pooled_priors=read(OUT/'run207/population_priors.json'),optimizer_updates=0,not_weld_accuracy=True,full_contract_pass=False,end_utc=utc()))

if __name__=='__main__':main()
