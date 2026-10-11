"""Explicit local XYZ cohort loading and rigid-copy exclusion certificates."""
import argparse
import gc
import traceback
import numpy as np
from scipy.spatial.distance import pdist
from weld_pipeline.sjg_curve_label_inventory_895 import ROOT,OUT,read,write,digest,utc


def load_cohort():
    manifest=read(OUT/'run98/local_cohort.json')
    if manifest['coordinate_frame']!='workpiece_local' or manifest['unit']!='mm':
        raise ValueError('Expected native workpiece-local millimetres')
    if manifest['original_eight_G']!=496:
        raise ValueError('Original evaluation denominator changed')
    return manifest


def load_xyz(case):
    path=OUT/case['input_file']
    if digest(path)!=case['input_sha256']:
        raise ValueError('Frozen local input hash changed')
    points=np.load(path,mmap_mode='r',allow_pickle=False)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('Invalid local XYZ')
    return points


def diameter_bounds(xyz):
    """Witness pair lower bound, enclosing-box upper bounds, full-point scan.

    Diameter changes by at most 2*epsilon for two sets at Hausdorff distance
    epsilon, under any rigid transform. A disjoint interval is a certificate
    of non-duplication at that tolerance, not a general independence claim.
    """
    p=np.asarray(xyz,np.float64);center=p.mean(0)
    centered=p-center;cov=centered.T@centered/len(p)
    _,axes=np.linalg.eigh(cov)
    projected=centered@axes
    upper=min(float(np.linalg.norm(np.ptp(p,axis=0))),float(np.linalg.norm(np.ptp(projected,axis=0))))
    ids=set()
    for axis in range(3):
        ids.update((int(p[:,axis].argmin()),int(p[:,axis].argmax()),
                    int(projected[:,axis].argmin()),int(projected[:,axis].argmax())))
    # Farthest-point witnesses improve the lower bound without N^2 storage.
    anchor=0
    for _ in range(8):
        anchor=int(np.argmax(np.einsum('ij,ij->i',p-p[anchor],p-p[anchor])))
        ids.add(anchor)
    witness=np.asarray(sorted(ids),np.int64)
    lower=float(pdist(p[witness]).max())
    if lower>upper+1e-6:raise AssertionError('Invalid diameter enclosure')
    return dict(points=len(p),diameter_lower_mm=lower,diameter_upper_mm=upper,
                lower_bound_witness_ids=witness.tolist(),axis_extent_mm=np.ptp(p,axis=0).tolist(),
                pca_extent_mm=np.ptp(projected,axis=0).tolist())


def exclusion(a,b,tolerance_mm=3.):
    separation=max(a['diameter_lower_mm']-b['diameter_upper_mm'],
                   b['diameter_lower_mm']-a['diameter_upper_mm'],0.)
    return dict(tolerance_mm=tolerance_mm,diameter_interval_separation_mm=separation,
                certified_rigid_Hausdorff_lower_bound_mm=separation/2,
                rigid_copy_within_tolerance_excluded=bool(separation>2*tolerance_mm+1e-6))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    protocol=dict(stage='LOCAL_XYZ_INPUT_AND_RIGID_COPY_CHECK',start_utc=utc(),
                  source_sha256=digest(__file__),cohort_sha256=digest(OUT/'run98/local_cohort.json'),
                  inputs_only=True,label_reads=0,rigid_copy_tolerance_mm=3.,
                  formal_group_independence_not_claimed=True,optimizer_updates=0)
    write(dest/'registered_protocol.json',protocol)
    try:
        # Exact translated/rotated controls exercise invariant diameter bounds.
        cube=np.array([[x,y,z] for x in (-1,1) for y in (-2,2) for z in (-3,3)],float)
        q,_=np.linalg.qr(np.random.default_rng(902).normal(size=(3,3)))
        a=diameter_bounds(cube);b=diameter_bounds(cube@q+1000);c=diameter_bounds(cube*10)
        assert not exclusion(a,b)['rigid_copy_within_tolerance_excluded']
        assert exclusion(a,c)['rigid_copy_within_tolerance_excluded']
        cohort=load_cohort();stats={}
        for case in cohort['cases']:
            xyz=load_xyz(case);key=case['source_key'];stats[key]=dict(diameter_bounds(xyz),role=case['role'],input_sha256=case['input_sha256'])
            print('LOCAL_XYZ_GEOMETRY',key,len(xyz),'DIAMETER_MM',stats[key]['diameter_lower_mm'],stats[key]['diameter_upper_mm'],flush=True)
            del xyz;gc.collect()
        bm=stats['BM79A_native_local'];comparisons=[]
        for key,item in stats.items():
            if key=='BM79A_native_local':continue
            comparisons.append(dict(source_key=key,role=item['role'],**exclusion(bm,item)))
        heldout=[r for r in comparisons if r['source_key'] in ('64','24')]
        write(dest/'geometry_bounds.json',stats);write(dest/'bm79a_comparisons.json',comparisons)
        write(dest/'receipt.json',dict(status='LOCAL_INPUT_AND_RIGID_COPY_CHECK_COMPLETE',
            original_eight_points=sum(v['points'] for k,v in stats.items() if k!='BM79A_native_local'),
            all_nine_points=sum(v['points'] for v in stats.values()),original_G=496,
            bm79a_rigid_copies_within_3mm_excluded_against_heldout=all(r['rigid_copy_within_tolerance_excluded'] for r in heldout),
            all_eight_rigid_copies_excluded=all(r['rigid_copy_within_tolerance_excluded'] for r in comparisons),
            scaled_or_partial_copy_exclusion_not_proven=True,
            formal_group_independence_not_claimed=True,optimizer_updates=0,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
