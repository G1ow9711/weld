"""Learned normal/boundary evidence reconnects remote support on fitted planes."""
import argparse
import numpy as np
from weld_pipeline.sjg_learned_surfaces_810 import extract as seed_extract,finite_support
from weld_pipeline.sjg_surface_boundary_812 import shared_boundary_support
from weld_pipeline.sjg_surface_pilot_811 import run


def extract(xyz,pred,progress=None):
    seeds,labels,stats=seed_extract(xyz,pred,progress)
    local,_=shared_boundary_support(xyz,seeds,progress)
    faces=[];additions=[]
    for index,plane in enumerate(local):
        selected=[]
        for start in range(0,len(xyz),65536):
            points=xyz[start:start+65536]
            near=np.flatnonzero(np.abs(points@plane['normal']-plane['offset'])<=.2)
            ids=(start+near).astype(np.int64)
            aligned=np.abs(pred['normal'][ids]@plane['normal'])>=np.cos(np.deg2rad(20.))
            boundary=pred['boundary'][ids]>=.5
            selected.append(ids[aligned|boundary])
        ids=np.unique(np.concatenate([plane['point_ids']]+selected))
        candidate={k:plane[k] for k in ('normal','center','offset','basis','learned_component_ids')}
        candidate.update(point_ids=ids,point_membership_multiple_planes_allowed=True,
            remote_support_requires_predicted_normal_or_boundary=True,
            material_ownership_not_inferred_from_geometric_plane=True)
        face=finite_support(xyz,candidate)
        if face is not None:faces.append(face)
        additions.append(dict(seed_plane=index,local_support=len(plane['point_ids']),learned_reassociated_support=len(ids)))
        if progress and index%25==0:progress('LEARNED_REMOTE_MEMBERSHIP',index+1,len(local))
    stats.update(finite_surfaces=len(faces),remote_membership=additions,global_unconditional_plane_fill=False)
    return faces,labels,stats


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    run(args,[21,4],extractor=extract,extra_protocol=dict(name='learned_global_membership',
        minimum_normal_alignment_deg=20.,minimum_boundary_probability=.5,
        shared_boundary=True,point_plane_distance_mm=.2,unconditional_global_fill=False))


if __name__=='__main__':main()
