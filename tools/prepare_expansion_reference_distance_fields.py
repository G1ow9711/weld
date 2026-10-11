"""Bounded distance-to-annotated-finite-curve targets for accepted TRAIN XYZ."""
from pathlib import Path
import time,shutil
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
STEP=.5
CAP=100.


def curve_samples(representation):
    if representation['kind']=='line':
        ends=np.asarray(representation['endpoints_mm'],float);length=float(np.linalg.norm(ends[1]-ends[0]));return ends[0]+np.linspace(0,1,max(1,int(np.ceil(length/STEP)))+1)[:,None]*(ends[1]-ends[0])
    arc=CircularArc.from_parameters(representation['arc_parameters']);return arc.point_at(np.linspace(0,arc.length_mm,max(1,int(np.ceil(arc.length_mm/STEP)))+1))


def bounds(points,samples):
    if not len(samples):return np.full(len(points),CAP),np.full(len(points),CAP)
    distance=cKDTree(samples).query(points,workers=2)[0]
    return np.minimum(np.maximum(distance-STEP/2,0),CAP),np.minimum(distance,CAP)


def controls():
    # Known finite straight line distances (including beyond endpoints).
    line=dict(kind='line',endpoints_mm=[[0.,0,0],[10.,0,0]]);points=np.array([[.25,2,0],[-1,0,0],[11,2,0],[5,0,3]],float);lo,hi=bounds(points,curve_samples(line));truth=np.sqrt(np.array([4.,1.,5.,9.]));assert np.all(lo<=truth) and np.all(truth<=hi)
    arc=CircularArc([0,0,0],10,[0,0,1],[1,0,0],np.pi/2,False);p=arc.point_at(np.linspace(0,arc.length_mm,79));lo,hi=bounds(p,curve_samples(dict(kind='arc',arc_parameters=arc.to_parameters())));assert np.all(lo<1e-12) and np.all(hi<=STEP/2+1e-12)
    lo,hi=bounds(points,np.empty((0,3)));assert np.all(lo==CAP) and np.array_equal(lo,hi)


def main(resume=False):
    controls();out=BASE/'reference_distance_fields_v1';out.mkdir(exist_ok=resume);started=time.perf_counter();summary=read(BASE/'training_label_rebuild_summary.json');groups={g['key']:g for g in read(BASE/'group_manifest.json')['groups']};cases=sorted(summary['successes'],key=lambda c:c['source_key']);assert len(cases)==58
    if not resume:write(out/'protocol.json',dict(group_manifest_sha256=digest(BASE/'group_manifest.json'),label_rebuild_summary_sha256=digest(BASE/'training_label_rebuild_summary.json'),script_sha256=digest(__file__),sample_arc_length_step_mm=STEP,distance_cap_mm=CAP,controls='PASS finite line endpoint/distance bounds, circle on-curve bounds, empty reference set',supervision='Distance to provided annotated finite weld curves, capped100mm. Nearest sampled distance d yields certified true-curve interval [max(0,d-.25),d], then cap. Lower/upper float32 rounded outward. Not physical absence, weld type, owner, or contact labels.',input='Original XYZ only; target arrays never model input',DEV_reads=0))
    if resume:
        original=read(out/'protocol.json');assert original['sample_arc_length_step_mm']==STEP and original['distance_cap_mm']==CAP;assert original['group_manifest_sha256']==digest(BASE/'group_manifest.json');write(out/'resume_protocol.json',dict(original_protocol_sha256=digest(out/'protocol.json'),resume_script_sha256=digest(__file__),algorithm_changed=False,existing_complete_cases_reused_only_after_hash_checks=True))
    records=[]
    for case in cases:
        key=case['source_key'];assert case['role']=='EXPANSION_TRAIN' and groups[key]['role']=='EXPANSION_TRAIN';assert groups[key]['family_id']==case['family_id'];folder=out/key;folder.mkdir(exist_ok=resume);xyzpath=BASE/'inputs'/key/'xyz_mm.npy';ir=read(xyzpath.parent/'receipt.json');assert ir['role']=='EXPANSION_TRAIN' and ir['family_id']==case['family_id'];assert digest(xyzpath)==ir['xyz_sha256'];targetpath=BASE/'labels'/key/'targets.json';target=read(targetpath);assert digest(targetpath)==read(targetpath.parent/'receipt.json')['targets_sha256'];assert target['role']=='TRAIN' and target['unit']=='mm' and target['coordinate_frame']=='workpiece_local';xyz=np.load(xyzpath,mmap_mode='r');assert len(xyz)==ir['original_points'];
        if resume and (folder/'receipt.json').exists():
            r=read(folder/'receipt.json');assert r['input_sha256']==digest(xyzpath) and r['target_sha256']==digest(targetpath);assert digest(folder/'lower_mm.npy')==r['lower_sha256'] and digest(folder/'upper_mm.npy')==r['upper_sha256'];records.append(r);print('REUSED',key,flush=True);continue
        assert not list(folder.iterdir()),'Incomplete existing data must be inspected before resume'
        samples=[curve_samples(c['representation']) for p in target['parents'] for c in p['children']];samples=np.concatenate(samples) if samples else np.empty((0,3));tree=cKDTree(samples) if len(samples) else None;assert shutil.disk_usage(BASE).free>=8*1024**3+len(xyz)*8
        lower=np.lib.format.open_memmap(folder/'lower_mm.npy',mode='w+',dtype=np.float32,shape=(len(xyz),));upper=np.lib.format.open_memmap(folder/'upper_mm.npy',mode='w+',dtype=np.float32,shape=(len(xyz),));near=far=0
        for a in range(0,len(xyz),65536):
            b=min(a+65536,len(xyz));distance=tree.query(xyz[a:b],workers=2)[0] if tree is not None else np.full(b-a,np.inf);lo=np.minimum(np.maximum(distance-STEP/2,0),CAP);hi=np.minimum(distance,CAP);lower[a:b]=np.maximum(0,np.nextafter(lo.astype(np.float32),np.float32(-np.inf)));upper[a:b]=np.minimum(CAP,np.nextafter(hi.astype(np.float32),np.float32(np.inf)));assert np.all(lower[a:b]<=upper[a:b]);near+=int((upper[a:b]<=3).sum());far+=int((lower[a:b]>=6).sum())
        lower.flush();upper.flush();lower._mmap.close();upper._mmap.close();r=dict(key=key,family_id=case['family_id'],points=len(xyz),curve_samples=len(samples),reference_children=target['original_children'],certain_reference_distance_le3mm=near,certain_reference_distance_ge6mm=far,input_sha256=digest(xyzpath),target_sha256=digest(targetpath),input_receipt_sha256=digest(xyzpath.parent/'receipt.json'),label_receipt_sha256=digest(targetpath.parent/'receipt.json'),lower_sha256=digest(folder/'lower_mm.npy'),upper_sha256=digest(folder/'upper_mm.npy'));write(folder/'receipt.json',r);records.append(r);print(key,'points',len(xyz),'near',near,'cases',len(records),flush=True)
    write(out/'receipt.json',dict(status='TRAIN_REFERENCE_DISTANCE_BOUNDS_READY',cases=records,protocol_sha256=digest(out/'protocol.json'),seconds=time.perf_counter()-started,DEV_reads=0))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');main(p.parse_args().resume)
