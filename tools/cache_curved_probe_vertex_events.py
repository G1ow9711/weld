"""Freeze projected endpoint events from all nonhorizontal raw points, no GT."""
from pathlib import Path
import time,shutil
import numpy as np
from weld_pipeline.server_arc_vertex_events import arc_vertex_events
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    started=time.perf_counter();key='new_fcaf2af1abee';source=BASE/'curved_upright_probe_v3'/key;r=read(source/'receipt.json');assert digest(source/'profiles.json')==r['profiles_sha256'];profiles=read(source/'profiles.json');path=BASE/'inputs'/key/'xyz_mm.npy';assert digest(path)==r['input_sha256'];xyz=np.load(path,mmap_mode='r');folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);binding=read(folder/'input_freeze.json');n=np.asarray(r['base_normal']);excluded=np.zeros(len(xyz),bool)
    for name in ['surfaces.json','surface_support.npz']:assert digest(folder/name)==binding['files'][name]
    faces=read(folder/'surfaces.json');basis=np.asarray(faces[r['base_face']]['basis'])
    with np.load(folder/'surface_support.npz') as support:
        for face in faces:
            normal=np.asarray(face.get('normal',[0,0,0]),float);length=np.linalg.norm(normal)
            if length and abs(normal@n/length)>=1-1e-8 and face.get('point_ids_key') in support:excluded[support[face['point_ids_key']]]=True
    assert int(excluded.sum())==r['horizontal_support_points_excluded'];raw_ids=np.flatnonzero(~excluded);points=np.asarray(xyz[raw_ids]);out=BASE/'curved_probe_vertex_events_v1'/key;out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(profiles_sha256=digest(source/'profiles.json'),source_receipt_sha256=digest(source/'receipt.json'),surface_freeze_sha256=digest(folder/'input_freeze.json'),input_sha256=digest(path),builder_sha256=digest(__file__),event_module_sha256=digest('weld_pipeline/server_arc_vertex_events.py'),radial_band_mm=3.,GT_reads=0,scope='All 1920 frozen proposals, project original nonhorizontal-member XYZ into each carrier plane within radial band and finite angular range, plus carrier endpoints. Preserve raw point ID and signed axial distance. Height projection is a candidate only, NOT observed contact or material continuity; explicit original section gaps remain in source profiles.'))
    offsets=[0];positions=[];ids=[];heights=[]
    for i,p in enumerate(profiles):
        assert shutil.disk_usage(BASE).free>=8*1024**3
        arc=CircularArc.from_parameters(p['representation']['arc_parameters']);uv=(points-arc.center_mm)@basis;q,source_ids=arc_vertex_events(arc,uv,arc.center_mm,basis);valid=source_ids>=0;original=source_ids.copy();original[valid]=raw_ids[source_ids[valid]];height=np.zeros(len(q));height[valid]=(points[source_ids[valid]]-arc.center_mm)@arc.plane_normal
        positions.append(q);ids.append(original);heights.append(height);offsets.append(offsets[-1]+len(q))
        if i%200==0:print(i,offsets[-1],flush=True)
    dest=out/'events.npz';np.savez_compressed(dest,offsets=np.asarray(offsets,np.int64),positions_mm=np.concatenate(positions),raw_point_ids=np.concatenate(ids),source_height_mm=np.concatenate(heights));write(out/'receipt.json',dict(status='INPUT_CURVED_PROBE_EVENTS_FROZEN',profiles=len(profiles),events=offsets[-1],raw_nonhorizontal_points=len(points),protocol_sha256=digest(out/'protocol.json'),events_sha256=digest(dest),seconds=time.perf_counter()-started,GT_reads=0,no_automatic_promotion=True))

if __name__=='__main__':main()
