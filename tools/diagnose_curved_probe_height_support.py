"""TRAIN witness diagnosis: distinguish one-section gaps from 3D observations."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def quantiles(values):
    values=np.asarray(values);finite=values[np.isfinite(values)]
    return dict(finite_count=len(finite),total_count=len(values),quantiles_mm=np.quantile(finite,[0,.5,.95,1]).tolist() if len(finite) else [])


def main():
    key='new_fcaf2af1abee';source=BASE/'curved_upright_probe_v3'/key;receipt=read(source/'receipt.json');profiles=read(source/'profiles.json');assert digest(source/'profiles.json')==receipt['profiles_sha256'];cap=read(source/'d3_interval_capacity.json');assert cap['profiles_sha256']==receipt['profiles_sha256']
    path=BASE/'inputs'/key/'xyz_mm.npy';assert digest(path)==receipt['input_sha256'];xyz=np.load(path,mmap_mode='r');n=np.array(receipt['base_normal']);folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);binding=read(folder/'input_freeze.json');assert digest(folder/'surfaces.json')==binding['files']['surfaces.json'];assert digest(folder/'surface_support.npz')==binding['files']['surface_support.npz'];faces=read(folder/'surfaces.json');excluded=np.zeros(len(xyz),bool)
    with np.load(folder/'surface_support.npz') as support:
        for face in faces:
            normal=np.asarray(face.get('normal',[0,0,0]),float);length=np.linalg.norm(normal)
            if length and abs(normal@n/length)>=1-1e-8 and face.get('point_ids_key') in support:excluded[support[face['point_ids_key']]]=True
    assert int(excluded.sum())==receipt['horizontal_support_points_excluded']
    basis=np.asarray(faces[receipt['base_face']]['basis']);off=np.asarray(xyz[~excluded]);uv=off@basis;tree=cKDTree(uv);height=off@n;tree3=cKDTree(off);rows=[]
    for row in cap['rows']:
        candidates=[]
        for match in row['matches']:
            p=profiles[match['profile']];curve=CurveCarrier(p['representation']);lo,hi=match['interval_mm'];q=np.linspace(lo,hi,int(np.ceil(hi-lo))+1);points=curve.at(q);covered=np.zeros(len(q),bool)
            for a,b in p['support_intervals_mm']:covered|=(q>=a)&(q<=b)
            near=tree.query_ball_point(points@basis,3.,workers=2);baseheight=float(curve.arc.center_mm@n);minimum=np.full(len(q),np.inf);below=np.full(len(q),np.inf);above=np.full(len(q),np.inf);counts=np.array([len(ids) for ids in near])
            for i,ids in enumerate(near):
                if not ids:continue
                relative=height[ids]-baseheight;minimum[i]=abs(relative).min()
                if (relative>=0).any():above[i]=relative[relative>=0].min()
                if (relative<=0).any():below[i]=-relative[relative<=0].max()
            distance=tree3.query(points,workers=2)[0]
            candidates.append(dict(profile=match['profile'],samples=len(q),section_missing_samples=int((~covered).sum()),any_height_projected_neighbor_count=int((counts>0).sum()),section_missing_with_any_height_neighbor_count=int(((~covered)&(counts>0)).sum()),section_missing_with_nonhorizontal_3D_neighbor_within3mm=int(((~covered)&(distance<=3)).sum()),nonhorizontal_3D_distance=quantiles(distance),minimum_absolute_height_given_uv_within3mm=quantiles(minimum),nearest_above_height=quantiles(above),nearest_below_height=quantiles(below),missing_section_minimum_abs_height=quantiles(minimum[~covered])))
        rows.append(dict(parent=row['parent'],child=row['child'],matches=candidates))
        print(row['parent'],[(r['profile'],r['section_missing_samples'],r['section_missing_with_any_height_neighbor_count'],r['section_missing_with_nonhorizontal_3D_neighbor_within3mm']) for r in candidates],flush=True)
    out=BASE/'curved_probe_height_support_diagnostic_v1';out.mkdir(exist_ok=False);write(out/'report.json',dict(rows=rows,raw_points=len(xyz),horizontal_members_removed=int(excluded.sum()),remaining_points=len(off),input_sha256=digest(path),profiles_sha256=digest(source/'profiles.json'),capacity_sha256=digest(source/'d3_interval_capacity.json'),surface_freeze_sha256=digest(folder/'input_freeze.json'),script_sha256=digest(__file__),scope='TRAIN reference chooses existing candidate intervals only for diagnosis. All points not assigned to parallel planes are queried; these are not certified upright surfaces. Any-height neighbor is not continuity/contact; no point at base does not prove material absence. No runtime outputs or modified candidates.',DEV_reads=0))

if __name__=='__main__':main()
