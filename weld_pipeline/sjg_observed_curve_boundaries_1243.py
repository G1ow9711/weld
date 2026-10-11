"""Add finite observed curve-support boundaries to geometric event proposals.

Sampling gaps create optional anchors only. They do not force weld splits and
do not prove missing material. Ideal point clouds use a fixed 0.1 mm support
band; no GT or STEP selects the band or the two-median-gap rule.
"""
import argparse
import gc
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case

TUBE_MM = .1
COINCIDENT_MM = 1e-6


def unique_parameters(q):
    q = np.sort(q)
    if not len(q):return q
    starts = np.r_[0, np.flatnonzero(np.diff(q) > COINCIDENT_MM)+1]
    return np.add.reduceat(q, starts)/np.diff(np.r_[starts, len(q)])


def support_anchors(q, length, closed=False):
    q = unique_parameters(q)
    if not len(q):return np.array([]), 0.
    if len(q) == 1:return q, 0.
    gaps = np.diff(q)
    if closed:
        # 0 and L denote the same point. Keep one physical observation.
        if length-q[-1]+q[0] <= COINCIDENT_MM:q=q[:-1]
        if len(q)<2:return np.array([]),0.
        gaps=np.r_[np.diff(q),length-q[-1]+q[0]]
        step=float(np.median(gaps[gaps>COINCIDENT_MM]))
        breaks=np.flatnonzero(gaps>2*step)
        indices=np.unique(np.r_[breaks,(breaks+1)%len(q)])
    else:
        step=float(np.median(gaps[gaps>COINCIDENT_MM]))
        breaks=np.flatnonzero(gaps>2*step)
        indices=np.unique(np.r_[0,breaks,breaks+1,len(q)-1])
    return q[indices],step


def controls(dest):
    q=np.array([0,10,20,70,80,90.])
    a,_=support_anchors(q,100)
    assert np.array_equal(a,[0,20,70,90])
    reverse,_=support_anchors(100-q,100)
    assert np.array_equal(a,np.sort(100-reverse))
    closed,_=support_anchors(np.arange(0,101,10),100,True)
    assert not len(closed)
    assert len(unique_parameters(np.array([0.,1e-10,10.,10.+1e-10])))==2
    write(dest/'controls.json',dict(line_reversal=True,closed_uniform_support_has_no_fake_break=True,
        coincident_projection_duplicates_collapsed=True,all_gaps_optional_anchors_not_weld_splits=True))


def prepare(key,dest,base_run='run393'):
    start=time.perf_counter();folder0=OUT/f'{base_run}/source_{key}';r=read(folder0/'receipt.json')
    assert r['GT_reads']==0 and digest(folder0/'boundary_events.npz')==r['events_sha256']
    with np.load(folder0/'boundary_events.npz') as z:base={k:z[k] for k in z.files}
    case=load_case(key);g=case['graph'];case['x']._mmap.close()
    cohort=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/cohort['input_file'])==cohort['input_sha256']
    xyz=np.load(OUT/cohort['input_file'],mmap_mode='r')
    raw_tree=cKDTree(xyz)
    fields=('q_mm','surface_ids','kinds','hull_signed_excess_mm','nearest_observed_point_mm',
            'plane_residual_max_mm','incidence','finite_support')
    blocks={k:[] for k in fields};offsets=[0];records=[]
    for j,pid in enumerate(base['profile_ids']):
        curve=g.curve(int(pid))
        q,dist=curve.project(xyz)
        mask=dist<=TUBE_MM
        anchors,step=support_anchors(q[mask],curve.length,curve.kind=='arc' and curve.arc.is_closed)
        if len(anchors):
            assert max(float(np.max(-anchors)),float(np.max(anchors-curve.length)))<=1e-8
            anchors=np.clip(anchors,0,curve.length)
        a,b=base['offsets'][j:j+2];n=len(anchors)
        new=dict(q_mm=anchors,surface_ids=np.full(n,-1),kinds=np.full(n,3),
            hull_signed_excess_mm=np.zeros(n),nearest_observed_point_mm=raw_tree.query(curve.at(anchors),workers=2)[0] if n else np.zeros(0),
            plane_residual_max_mm=np.full(n,float(dist[mask].max()) if mask.any() else 0),
            incidence=np.zeros(n),finite_support=np.ones(n,bool))
        order=np.argsort(np.r_[base['q_mm'][a:b],anchors],kind='stable')
        for field in fields:blocks[field].append(np.r_[base[field][a:b],new[field]][order])
        offsets.append(offsets[-1]+b-a+n)
        records.append(dict(profile=int(pid),support_points=int(mask.sum()),median_sampling_step_mm=step,
                            proposed_anchors=n,raw_curve_support_not_material_truth=True))
        if j%128==0:print('OBSERVED_BOUNDARIES',key,j+1,'/',len(base['profile_ids']),flush=True)
    data={k:np.concatenate(v) for k,v in blocks.items()}
    for name,dtype in [('surface_ids',np.int32),('kinds',np.uint8),('finite_support',bool)]:data[name]=data[name].astype(dtype)
    data.update(profile_ids=base['profile_ids'],admitted_rows=base['admitted_rows'],offsets=np.array(offsets))
    folder=dest/f'source_{key}';folder.mkdir(exist_ok=False);np.savez(folder/'boundary_events.npz',**data)
    write(folder/'receipt.json',dict(status='INPUT_BOUNDARY_EVENTS_FROZEN',source=key,
        input_sha256=cohort['input_sha256'],prior_events_sha256=r['events_sha256'],
        events_sha256=digest(folder/'boundary_events.npz'),events=len(data['q_mm']),
        new_observed_support_anchors=int(np.sum(data['kinds']==3)),tube_mm=TUBE_MM,
        GT_reads=0,STEP_reads=0,fits=0,all_prior_events_retained=True,
        kind3_residual_field_is_curve_tube_residual_not_fitted_plane=True,
        gap_does_not_force_weld_split=True,seconds=time.perf_counter()-start))
    write(folder/'support_records.json',dict(records=records))
    xyz._mmap.close();print('OBSERVED_BOUNDARIES_READY',key,'ANCHORS',int(np.sum(data['kinds']==3)),
                              'SECONDS',round(time.perf_counter()-start,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sources',nargs='+',required=True)
    args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),sources=args.sources,
        tube_mm=TUBE_MM,gap_factor=2.,coincident_tolerance_mm=COINCIDENT_MM,
        GT_reads=0,STEP_reads=0,fits=0,baseline379_preserved=True,external_consultation_paused=True))
    controls(dest)
    try:
        for key in args.sources:prepare(key,dest);gc.collect()
    except BaseException:write(dest/'failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
