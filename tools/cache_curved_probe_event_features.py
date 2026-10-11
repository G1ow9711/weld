"""All-candidate frozen PTv3 event descriptors plus explicit height/support."""
from pathlib import Path
import time,shutil
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_curve_queries_907 import curve_point_queries
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    started=time.perf_counter();key='new_fcaf2af1abee';events=BASE/'curved_probe_vertex_events_v1'/key;er=read(events/'receipt.json');assert digest(events/'protocol.json')==er['protocol_sha256'];assert digest(events/'events.npz')==er['events_sha256'];ep=read(events/'protocol.json');source=BASE/'curved_upright_probe_v3'/key;assert digest(source/'profiles.json')==ep['profiles_sha256'];profiles=read(source/'profiles.json')
    folder=BASE/'frozen_features/merged_point_features'/('source_'+key);fr=read(folder/'receipt.json');assert digest(folder/'features.npz')==fr['feature_sha256'];path=BASE/'inputs'/key/'xyz_mm.npy';assert digest(path)==fr['input_sha256']==ep['input_sha256'];xyz=np.load(path,mmap_mode='r');tree=cKDTree(xyz)
    with np.load(folder/'features.npz') as z:point_features={k:z[k] for k in ['embedding','normal','boundary']};assert np.all(z['core_coverage']==1)
    with np.load(events/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm'];rawids=z['raw_point_ids'];height=z['source_height_mm']
    out=BASE/'curved_probe_event_features_v1'/key;out.mkdir(parents=True,exist_ok=False)
    assert shutil.disk_usage(BASE).free>=8*1024**3+len(positions)*83*4
    write(out/'protocol.json',dict(event_receipt_sha256=digest(events/'receipt.json'),point_feature_receipt_sha256=digest(folder/'receipt.json'),profiles_sha256=digest(source/'profiles.json'),input_sha256=digest(path),builder_sha256=digest(__file__),query_module_sha256=digest('weld_pipeline/sjg_curve_queries_907.py'),dimensions=83,extra_features=['signed_log1p_source_height_mm','log1p_abs_source_height_mm','is_raw_point_event','in_observed_section_interval','log1p_distance_to_nearest_section_start_mm','log1p_distance_to_nearest_section_end_mm','fraction_carrier_unobserved_at_section'],scope='Original76 frozen PTv3 local descriptors plus7 input evidence values. Height zero on carrier endpoints requires is_raw_point_event=0; section support not contact proof. All input candidates, no reference selection.',GT_reads=0))
    dest=out/'features.npy';features=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(positions),83))
    for index,p in enumerate(profiles):
        start,stop=offsets[index:index+2];intervals=np.asarray(p['support_intervals_mm']);length=p['length_mm'];gap_fraction=1-float(np.sum(intervals[:,1]-intervals[:,0]))/length
        for a in range(int(start),int(stop),1024):
            b=min(a+1024,int(stop));q=positions[a:b];query=curve_point_queries(p['representation'],q,xyz,point_features,tree);patch=query['patch'];base=np.column_stack((patch.mean(1),patch.min(1),patch.max(1),query['query']));in_section=np.any((q[:,None]>=intervals[None,:,0])&(q[:,None]<=intervals[None,:,1]),axis=1);h=height[a:b];extra=np.column_stack((np.sign(h)*np.log1p(abs(h)),np.log1p(abs(h)),rawids[a:b]>=0,in_section,np.log1p(abs(q[:,None]-intervals[None,:,0]).min(1)),np.log1p(abs(q[:,None]-intervals[None,:,1]).min(1)),np.full(len(q),gap_fraction)));value=np.column_stack((base,extra)).astype(np.float32);assert value.shape==(b-a,83) and np.isfinite(value).all();features[a:b]=value
        if index%100==0:print(index,int(stop),flush=True)
    features.flush();features._mmap.close();write(out/'receipt.json',dict(status='INPUT_CURVED_PROBE_EVENT_FEATURES_FROZEN',events=len(positions),profiles=len(profiles),dimensions=83,feature_sha256=digest(dest),protocol_sha256=digest(out/'protocol.json'),seconds=time.perf_counter()-started,GT_reads=0))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
