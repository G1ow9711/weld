"""Exact encoded-input collision audit; not a claim about raw-cloud ambiguity."""
import argparse,gc,hashlib,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources

def hashes(x):
    x=np.array(x,copy=True);assert np.isfinite(x).all();x[x==0]=0
    return [hashlib.sha256(row.tobytes()).digest() for row in x]

def counts(keys,y):
    groups={}
    for key,label in zip(keys,y):
        group=groups.setdefault(key,[0,0]);group[int(label)]+=1
    mixed=[v for v in groups.values() if v[0] and v[1]]
    return dict(rows=len(keys),unique_inputs=len(groups),mixed_input_groups=len(mixed),
        positive_rows_in_mixed_groups=sum(v[1] for v in mixed),negative_rows_in_mixed_groups=sum(v[0] for v in mixed),
        unavoidable_row_errors_for_deterministic_classifier=sum(min(v) for v in mixed),
        not_a_bound_on_child_extraction_accuracy=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    start=time.perf_counter();manifest=read(OUT/'run206/training_manifest.json');bindings={r['source_key']:r for r in manifest['static_bindings']};rows=[]
    for case in manifest['cases']:
        check_resources();key=case['source_key'];b=bindings[key]
        assert digest(OUT/'run206'/b['file'])==b['sha256'] and digest(OUT/'run206'/case['file'])==case['sha256']
        with np.load(OUT/'run206'/b['file']) as z:rx=z['relation_x'];fx=z['family_x'];fr=z['family_relation']
        with np.load(OUT/'run206'/case['file']) as z:ry=z['relation_y'];fy=z['family_y']
        rk=hashes(rx);fk=hashes(fx);combined=[hashlib.sha256(rk[int(r)]+f).digest() for r,f in zip(fr,fk)]
        row=dict(source=key,relation=counts(rk,ry),family_and_full_relation_input=counts(combined,fy));rows.append(row)
        print('EXACT_FEATURE_COLLISION',key,'REL',row['relation']['mixed_input_groups'],'FAMILY',row['family_and_full_relation_input']['mixed_input_groups'],flush=True)
        del rx,fx,fr,ry,fy,rk,fk,combined;gc.collect()
    write(dest/'receipt.json',dict(status='EXACT_ENCODED_INPUT_COLLISIONS_AUDITED',cases=rows,optimizer_updates=0,
        raw_cloud_contradiction_not_claimed=True,no_collisions_does_not_prove_learnability=True,
        elapsed_seconds=time.perf_counter()-start,full_contract_pass=False,end_utc=utc()))

if __name__=='__main__':main()
