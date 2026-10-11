"""Input-only event descriptors from the existing frozen PTv3 point features."""
from pathlib import Path
import time,shutil
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_curve_queries_907 import curve_point_queries
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main():
    key='new_ac2008a03607';started=time.perf_counter();events=BASE/'training_arc_vertex_events_v1'/key;source=BASE/'union_nearest_arc_candidates_v1'/key
    er=read(events/'receipt.json');assert er['status']=='INPUT_ARC_EVENTS_FROZEN';ep=read(events/'protocol.json');assert digest(events/'protocol.json')==er['protocol_sha256'];assert ep['candidate_receipt_sha256']==digest(source/'receipt.json')
    cr=read(source/'receipt.json');candidate_map={r['face']:r for r in cr['faces']}
    folder=BASE/'frozen_features/merged_point_features'/('source_'+key);fr=read(folder/'receipt.json');assert digest(folder/'features.npz')==fr['feature_sha256']
    xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==fr['input_sha256'];xyz=np.load(xyzpath,mmap_mode='r');tree=cKDTree(xyz)
    with np.load(folder/'features.npz') as z:point_features={k:z[k] for k in ('embedding','normal','boundary')};assert np.all(z['core_coverage']==1)
    assert len(point_features['embedding'])==len(xyz)
    out=BASE/'training_arc_event_features_v1'/key;out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(key=key,event_receipt_sha256=digest(events/'receipt.json'),candidate_receipt_sha256=digest(source/'receipt.json'),point_feature_receipt_sha256=digest(folder/'receipt.json'),input_sha256=fr['input_sha256'],query_module_sha256=digest('weld_pipeline/sjg_curve_queries_907.py'),descriptor='Existing16NN curve-frame patch23 features pooled mean/min/max (69) plus original7 query scalars =76. Frozen PTv3 embedding reused; no global coordinates/source IDs.',GT_reads=0,normalization='Raw descriptors; training-only statistics must be fitted later.'))
    rows=[]
    for record in er['faces']:
        assert shutil.disk_usage(BASE).free>=8*1024**3,'Insufficient cache disk headroom'
        path=events/record['file'];assert digest(path)==record['sha256'];cp=source/candidate_map[record['face']]['file'];assert digest(cp)==candidate_map[record['face']]['sha256'];candidates=read(cp)['candidates']
        with np.load(path) as z:offsets=z['offsets'];positions=z['positions_mm']
        assert len(offsets)==len(candidates)+1 and offsets[-1]==len(positions)
        target=out/f"face_{record['face']}.npy";features=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(len(positions),76))
        for index,candidate in enumerate(candidates):
            start,stop=offsets[index:index+2];representation=dict(kind='arc',arc_parameters=candidate['arc_parameters'])
            for a in range(int(start),int(stop),1024):
                b=min(a+1024,int(stop));query=curve_point_queries(representation,positions[a:b],xyz,point_features,tree);patch=query['patch'];value=np.column_stack((patch.mean(1),patch.min(1),patch.max(1),query['query'])).astype(np.float32)
                assert value.shape==(b-a,76) and np.isfinite(value).all();features[a:b]=value
        features.flush();features._mmap.close()
        rows.append(dict(face=record['face'],events=len(positions),file=target.name,sha256=digest(target),event_file_sha256=record['sha256']))
        if record['face']%50==0:print(record['face'],'events',sum(r['events'] for r in rows),flush=True)
    write(out/'receipt.json',dict(status='INPUT_ARC_EVENT_FEATURES_FROZEN',faces=rows,protocol_sha256=digest(out/'protocol.json'),dimensions=76,GT_reads=0,seconds=time.perf_counter()-started))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
