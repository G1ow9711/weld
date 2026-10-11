"""Replay registered TRAIN crop RNG; audit coverage without training."""
from collections import defaultdict
import numpy as np
from scipy.signal import find_peaks
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import OUT,TRAIN


def main():
    data=OUT/'run38';records=read(data/'manifest.json')['sequences'];metadata=[]
    for record in records:
        with np.load(data/record['file']) as z:
            mask=z['offset_mask'];target=z['targets'];q=z['q_mm']
        event_core=np.flatnonzero(mask.any(1))
        centers=[find_peaks(np.r_[-1.,target[:,h],-1.],height=.95)[0]-1 for h in (1,2)]
        metadata.append(dict(count=len(q),event_core=event_core,centers=centers,
            seen=np.zeros(len(q),bool),event_exposure=[np.zeros(len(ids),np.int64) for ids in centers]))
    epoch_points=[]
    for epoch in range(1,9):
        rng=np.random.default_rng(822+epoch);order=rng.permutation(len(records));points=0
        for index in order:
            item=metadata[int(index)];count=item['count'];start=0;events=item['event_core']
            if count>1024:
                if len(events) and rng.random()<.5:
                    center=int(rng.choice(events));start=int(np.clip(center-rng.integers(128,896),0,count-1024))
                else:start=int(rng.integers(count-1024+1))
            end=min(count,start+1024);points+=end-start;item['seen'][start:end]=True
            for centers,exposure in zip(item['centers'],item['event_exposure']):exposure+=(centers>=start)&(centers<end)
        epoch_points.append(points)
    rows=[]
    for source in TRAIN:
        ids=[i for i,r in enumerate(records) if r['source']==source]
        exposure=np.concatenate([a for i in ids for a in metadata[i]['event_exposure']])
        row=dict(source=source,profiles=len(ids),crops=len(ids)*8,
            original_grid_points=sum(metadata[i]['count'] for i in ids),
            grid_points_seen=sum(int(metadata[i]['seen'].sum()) for i in ids),
            target_event_peaks=len(exposure),event_peaks_never_seen=int((exposure==0).sum()),
            event_exposure_p05=float(np.quantile(exposure,.05)) if len(exposure) else None,
            event_exposure_median=float(np.median(exposure)) if len(exposure) else None)
        rows.append(row);print('CROP_COVERAGE',row,flush=True)
    import json
    logged=[json.loads(line) for line in (OUT/'run41/epochs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['core_points_seen'] for r in logged]==epoch_points
    write(OUT/'run41/sampling_audit_826.json',dict(scope='REGISTERED_TRAIN_RNG_REPLAY_NO_WEIGHT_CHANGE',
        source_sha256=digest(__file__),logged_epoch_core_counts_match=True,rows=rows,
        grid_points=sum(r['original_grid_points'] for r in rows),seen=sum(r['grid_points_seen'] for r in rows),
        peaks_never_seen=sum(r['event_peaks_never_seen'] for r in rows),full_contract_pass=False))


if __name__=='__main__':main()
