"""Report trivial-class baselines for frozen auxiliary supervision."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc

def main():
    source=OUT/'run139';rows=read(source/'manifest.json')['rows'];counts={k:[0,0] for k in ('same_material_possible','different_material')}
    total=known_total=0
    for row in rows:
        assert digest(source/row['file'])==row['sha256']
        with np.load(source/row['file'],allow_pickle=False) as z:
            known=z['material_relation_known'];total+=len(known);known_total+=int(known.sum())
            for field in counts:
                values=z[field][known];counts[field][0]+=int((values<.5).sum());counts[field][1]+=int((values>=.5).sum())
    result=dict(source_run='run139',queries=total,known_queries=known_total,unknown_queries=total-known_total,
        heads={k:dict(negative=v[0],positive=v[1],majority_constant_accuracy=max(v)/sum(v) if sum(v) else None) for k,v in counts.items()},
        interpretation='Auxiliary accuracy alone is not evidence of learned material discrimination; physical same-body negative queries are scarce.',
        no_prediction_or_training_changes=True,time_utc=utc())
    write(OUT/'run139/class_balance_audit.json',result);print(result,flush=True)

if __name__=='__main__':main()
