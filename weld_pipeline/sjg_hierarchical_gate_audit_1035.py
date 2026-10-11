"""TRAIN-only checkpoint audit; no optimization or threshold selection."""
import argparse,gc
import numpy as np
import torch
from scipy.stats import rankdata
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,resources
from weld_pipeline.sjg_hierarchical_joint_model_1028 import HierarchicalJointModel

def metrics(p,y):
    y=np.asarray(y,bool);n=int(y.sum());m=len(y)-n
    auc=float((rankdata(p)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
    ends=np.r_[np.flatnonzero(ps[1:]!=ps[:-1]),len(ps)-1]
    true=np.cumsum(ys)[ends];increments=np.diff(np.r_[0,true])
    ap=float(((true/(ends+1))*increments).sum()/n) if n else None
    def quant(a):return np.quantile(a,[0,.5,.95,1]).tolist() if len(a) else []
    return dict(total=len(y),positive=n,positive_probability_min_median_p95_max=quant(p[y]),
        negative_probability_min_median_p95_max=quant(p[~y]),auc=auc,average_precision=ap,
        TP_at_fixed_half=int((y&(p>=.5)).sum()),FP_at_fixed_half=int((~y&(p>=.5)).sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    dest=OUT/a.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    state=torch.load(OUT/'run207/failure_checkpoint.pt',map_location='cpu')
    model=HierarchicalJointModel(**state['configuration']);model.load_state_dict(state['state_dict']);model.eval()
    data=OUT/'run206';manifest=read(data/'training_manifest.json');bindings={s['source_key']:s for s in manifest['static_bindings']};rows=[]
    with torch.inference_mode():
        for c in manifest['cases']:
            key=c['source_key'];b=bindings[key]
            assert digest(data/c['file'])==c['sha256'] and digest(data/b['file'])==b['sha256']
            with np.load(data/b['file']) as z:s={k:torch.from_numpy(z[k]) for k in z.files}
            with np.load(data/c['file']) as z:ry=z['relation_y'];fy=z['family_y']
            r,rl=model.relations(s['relation_x']);f,fl=model.families(r,s['family_relation'],s['family_x'])
            rp=rl.sigmoid().numpy();fp=fl.sigmoid().numpy()
            row=dict(source=key,relation=metrics(rp,ry),family_without_relation_gate=metrics(fp,fy),
                relation_feature_max_abs=float(s['relation_x'].abs().max()),family_feature_max_abs=float(s['family_x'].abs().max()),
                resources=resources());rows.append(row)
            print('GATE_AUDIT',key,'REL_AUC',round(row['relation']['auc'],4),'REL_MAX',float(rp.max()),
                  'POS',int(ry.sum()),'PASS',row['relation']['TP_at_fixed_half'],flush=True)
            del s,r,rl,f,fl,ry,fy,rp,fp;gc.collect()
    write(dest/'receipt.json',dict(status='TRAIN_GATE_AUDITED',optimizer_updates=0,checkpoint_updates=state['updates'],
        checkpoint_sha256=digest(OUT/'run207/failure_checkpoint.pt'),cases=rows,threshold_changed=False,
        not_extraction_accuracy=True,full_contract_pass=False,end_utc=utc()))

if __name__=='__main__':main()
