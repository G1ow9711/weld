"""Offline CAD face audit of native sampling and existing normal/offset seeds."""
import argparse
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,utc
from weld_pipeline.sjg_observed_cylinders_919 import canonical_normals,dominant_axes,plane_levels

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='10')
    xyz=np.load(OUT/case['input_file'],mmap_mode='r');data=canonical_normals(xyz,lambda *v:print(*v,flush=True));axes=dominant_axes(data,64)
    # This saved input-only normal cache has no CAD IDs or supervision.
    np.savez_compressed(dest/'input_only_normals.npz',**data)
    diag=read(OUT/'run127/source_10.json');rows=[]
    for f in diag['required_faces']:
        if f['hypotheses']['matches']:continue
        n=np.asarray(f['normal']);center=np.asarray(f['center_local_mm']);offset=float(center@n)
        ids=np.flatnonzero(abs(xyz@n-offset)<=.01);canonical=abs(data['xyz']@n-offset)<=.01
        normals=data['normal'];angle=np.degrees(np.arccos(np.clip(abs(normals@n),0,1)))
        usable=canonical&data['planar']&(angle<=2)
        delta=np.asarray(xyz[ids])-np.asarray(xyz[ids]).mean(0);values,vectors=np.linalg.eigh(delta.T@delta/max(1,len(ids)))
        axis_angles=np.degrees(np.arccos(np.clip(np.abs(np.asarray(axes)@n),0,1)));ai=int(np.argmin(axis_angles));axis=axes[ai]
        levels,eligible,height=plane_levels(data,dict(axis=axis,origin=xyz.mean(0)))
        expected=float((center-xyz.mean(0))@axis);near=min([abs(v-expected) for v in levels],default=None)
        row=dict(face=f['STEP_face_index'],raw_plane_support_001mm=len(ids),canonical_support=int(canonical.sum()),
            rank2=int((canonical&data['rank2']).sum()),planar=int((canonical&data['planar']).sum()),
            planar_normal_within_2deg=int(usable.sum()),normal_error_deg_quantiles=np.quantile(angle[canonical],[0,.25,.5,.75,1]).tolist(),
            plane_only_fit_max_residual_mm=float(np.max(abs(delta@vectors[:,0]))),plane_only_eigenvalues=values.tolist(),
            nearest_dominant_axis_angle_deg=float(axis_angles[ai]),nearest_level_distance_mm=near,
            diagnostic_STEP_plane_filter_not_deployable=True)
        rows.append(row);print('STEP_SEED_AUDIT',row,flush=True)
    write(dest/'receipt.json',dict(status='STEP_SEED_AUDIT_COMPLETE',rows=rows,optimizer_updates=0,end_utc=utc()))

if __name__=='__main__':main()
