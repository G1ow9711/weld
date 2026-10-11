"""Bounded pure-XYZ geometric pilot; all candidates frozen before TRAIN scoring."""
import argparse
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_rank2_offset_recovery_947 import recover
from weld_pipeline.sjg_finite_envelope_carriers_941 import envelope_profiles
from weld_pipeline.sjg_local_curve_supervision_915 import supervise

def main(selection='global'):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json',dict(source='10',TRAIN_only=True,maximum_direction_families=16,maximum_new_planes=128,
        offset_bin_mm=.2,minimum_directional_rank2_votes=32,directional_tolerance_deg=2,
        planar_eigenvalue_gate_removed=True,selection=selection,STEP_reads=0,normal_cache='run128/input_only_normals.npz',
        source_sha256=digest(ROOT/'weld_pipeline/sjg_rank2_offset_recovery_947.py'),optimizer_updates=0,start_utc=utc()))
    check_resources();case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='10')
    assert digest(OUT/case['input_file'])==case['input_sha256'];xyz=np.load(OUT/case['input_file'],mmap_mode='r')
    source=OUT/'run125/source_10';faces=read(source/'surfaces.json')
    with np.load(source/'surface_support.npz',allow_pickle=False) as z:
        for f in faces:f['point_ids']=z[f['point_ids_key']]
    with np.load(OUT/'run128/input_only_normals.npz',allow_pickle=False) as z:data={k:z[k] for k in ('xyz','normal','rank2')}
    new,stats=recover(xyz,faces,data,lambda *v:print(*v,flush=True),selection=selection);original=len(faces);faces.extend(new)
    profiles=read(OUT/'run126/source_10/profiles.json');added=envelope_profiles(xyz,faces,profiles,lambda *v:print(*v,flush=True))
    added=[p for p in added if max(p['face_indices'])>=original];profiles.extend(added)
    write(dest/'profiles.json',profiles);serial=[];support={}
    for i,f in enumerate(faces):
        v={k:(x.tolist() if isinstance(x,np.ndarray) else x) for k,x in f.items() if k not in ('point_ids','triangles')}
        v['point_ids_key']='ids_'+str(i);support[v['point_ids_key']]=f['point_ids'];serial.append(v)
    write(dest/'surfaces.json',serial);np.savez_compressed(dest/'surface_support.npz',**support)
    write(dest/'input_freeze.json',dict(STEP_reads=0,target_reads=0,input_sha256=case['input_sha256'],
        profiles_sha256=digest(dest/'profiles.json'),surfaces_sha256=digest(dest/'surfaces.json'),new_planes=len(new),new_profiles=len(added)))
    score=supervise(profiles,read(OUT/case['target_file']));write(dest/'TRAIN_only_geometric_compatibility.json',score)
    write(dest/'receipt.json',dict(status='RANK2_OFFSET_RECOVERY_COMPLETE',G=case['original_G'],covered=score['covered_children'],
        new_planes=len(new),new_profiles=len(added),axes=stats,optimizer_updates=0,not_weld_accuracy=True,full_contract_pass=False,end_utc=utc()))
    print('RANK2_OFFSET_RECOVERY',score['covered_children'],'/',case['original_G'],'NEW_PLANES',len(new),flush=True)

if __name__=='__main__':main()
