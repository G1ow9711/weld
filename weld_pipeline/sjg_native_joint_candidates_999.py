"""Same nine-workpiece pilot: learned finite supports plus retained line/arcs.

Construction never reads targets or STEP. Earlier XYZ hypotheses remain
available; inferred finite supports and material membership are new inputs.
"""
import argparse
import builtins
import copy
import gc
import io
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_ideal_plane_support_997 import extract,controls
from weld_pipeline.sjg_xyz_batch_795 import contacts_from_support,save_surfaces
from weld_pipeline.sjg_unified_local_candidates_950 import load_faces,save_graph_faces
from weld_pipeline.sjg_geometry_776 import serial


def contact_profiles(contacts,faces):
    profiles=[];groups={}
    for c in contacts:
        a,b=c['face_indices'];p0=np.asarray(c['p0'],dtype=float);p1=np.asarray(c['p1'],dtype=float)
        length=float(np.linalg.norm(p1-p0))
        profiles.append(dict(representation=dict(kind='line',endpoints_mm=[p0.tolist(),p1.tolist()]),
            length_mm=length,face_indices=[a,b],support_normal=np.asarray(faces[a]['normal']).tolist(),
            support_intervals_mm=[[0.,length]],unobserved_gaps_mm=[],material_pair=None,weld_type=None,
            status='UNCLASSIFIED_FINITE_OBSERVED_CONTACT',proposal_branch='native_learned_finite_contact'))
        groups.setdefault((a,b),[]).append((p0,p1))
    for (a,b),segments in groups.items():
        if len(segments)<2:continue
        origin=segments[0][0];direction=segments[0][1]-origin;direction/=np.linalg.norm(direction)
        spans=sorted(sorted([float((p-origin)@direction),float((q-origin)@direction)]) for p,q in segments)
        merged=[]
        for lo,hi in spans:
            if merged and lo<=merged[-1][1]+1e-5:merged[-1][1]=max(merged[-1][1],hi)
            else:merged.append([lo,hi])
        if len(merged)<2:continue
        start,end=merged[0][0],merged[-1][1];support=[[lo-start,hi-start] for lo,hi in merged]
        profiles.append(dict(representation=dict(kind='line',endpoints_mm=[(origin+start*direction).tolist(),(origin+end*direction).tolist()]),
            length_mm=end-start,face_indices=[a,b],support_normal=np.asarray(faces[a]['normal']).tolist(),
            support_intervals_mm=support,unobserved_gaps_mm=[[x[1],y[0]] for x,y in zip(support[:-1],support[1:])],
            material_pair=None,weld_type=None,status='UNCLASSIFIED_CARRIER_WITH_EXPLICIT_UNKNOWN_GAPS',
            proposal_branch='native_learned_finite_span',gaps_not_asserted_to_be_welds=True))
    return profiles


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);rows=[]
    sources={r['source_key']:r for r in read(OUT/'run174/receipt.json')['sources']}
    cases=read(OUT/'run98/local_cohort.json')['cases']
    dependencies=['sjg_native_finite_geometry_982.py','sjg_local_sampling_mesh_991.py','sjg_union_sampling_mesh_995.py','sjg_ideal_plane_support_997.py']
    write(dest/'registered_protocol.json',dict(stage='NATIVE_FINITE_MATERIAL_JOINT_CANDIDATES',
        source_sha256=digest(__file__),dependency_sha256={n:digest(ROOT/'weld_pipeline'/n) for n in dependencies},
        source_keys=[c['source_key'] for c in cases],features='run166',instances='run174',backbone='run162',
        retained_XYZ_candidates='run132',original_eight_G=496,supplemental_BM_G=4,
        all_old_line_and_arc_hypotheses_retained=True,STEP_reads=0,target_reads=0,
        geometry_carriers_not_weld_predictions=True,optimizer_updates=0,full_contract_pass=False,start_utc=utc()))
    original_open,original_io=builtins.open,io.open
    def guard(factory):
        def opened(path,*a,**kw):
            if isinstance(path,(str,bytes,Path)):
                name=str(path).lower()
                if name.endswith(('.step','.stp','.xlsx')) or any(v in name for v in ('teacher','target','supervision','geometry.json')):
                    raise RuntimeError('Reference access forbidden: '+str(path))
            return factory(path,*a,**kw)
        return opened
    try:
        write(dest/'controls.json',controls())
        builtins.open=guard(original_open);io.open=guard(original_io)
        for case in cases:
            check_resources();key=case['source_key'];folder=dest/('source_'+key);folder.mkdir();finite=folder/'finite';finite.mkdir()
            start=time.perf_counter();feature=read(OUT/'run166'/('source_'+key+'_freeze.json'));partition=sources[key]
            assert case['input_sha256']==feature['input_sha256']==partition['input_sha256']
            paths=[OUT/case['input_file'],OUT/'run166'/feature['feature_file'],OUT/'run174'/partition['file']]
            hashes=[case['input_sha256'],feature['feature_sha256'],partition['sha256']]
            for p,h in zip(paths,hashes):assert digest(p)==h
            xyz=np.load(paths[0],mmap_mode='r')
            with np.load(paths[1]) as z:pred={k:z[k] for k in ('normal','boundary')}
            with np.load(paths[2]) as z:instance=z['face_instance'];material=z['material_instance']
            def progress(*v):
                check_resources();print('NATIVE_JOINT',key,*v,flush=True)
                write(dest/'progress.json',dict(source_key=key,event=list(v),completed=len(rows),total=len(cases),utc=utc()))
            if key=='21':
                previous=OUT/'run184';freeze=read(previous/'input_freeze.json');protocol=read(previous/'registered_protocol.json')
                assert freeze['input_sha256']==case['input_sha256'] and freeze['instance_sha256']==partition['sha256']
                assert protocol['extractor_sha256']==digest(ROOT/'weld_pipeline/sjg_ideal_plane_support_997.py')
                for name,sha in freeze['files'].items():
                    assert digest(previous/name)==sha;shutil.copy2(previous/name,finite/name)
                faces=load_faces(finite);contacts=read(finite/'contacts.json')
                with np.load(finite/'original_surface_support.npz') as z:full={k:z[k] for k in z.files}
                stats=dict(reused_verified_geometry='run184')
            else:
                faces,full,left,stats=extract(xyz,pred,instance,material,progress)
                np.savez_compressed(finite/'original_surface_support.npz',**full)
                np.save(finite/'unresolved_original_point_ids.npy',left)
                save_surfaces(finite,faces);contacts=contacts_from_support(faces,xyz)
                write(finite/'contacts.json',serial(contacts))
            added=contact_profiles(contacts,faces)
            prior=OUT/'run132'/('source_'+key);binding=read(prior/'input_freeze.json')
            assert binding['input_sha256']==case['input_sha256']
            for name in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(prior/name)==binding['files'][name]
            old_faces=load_faces(prior);profiles=read(prior/'profiles.json');old_count=len(profiles);offset=len(old_faces)
            for profile in added:profile['face_indices']=[i+offset for i in profile['face_indices']]
            for face in faces:
                face['point_ids']=full[face['full_support_key']]
                face['graph_membership_uses_full_original_support']=True
            all_faces=old_faces+faces;profiles.extend(added)
            for p in profiles:assert p['length_mm']>0 and all(0<=i<len(all_faces) for i in p['face_indices'])
            save_graph_faces(folder,all_faces);write(folder/'profiles.json',serial(profiles));write(folder/'support_stats.json',stats)
            write(folder/'input_freeze.json',dict(input_sha256=case['input_sha256'],feature_sha256=feature['feature_sha256'],
                instance_file=str(paths[2].relative_to(OUT)),instance_sha256=partition['sha256'],
                files={n:digest(folder/n) for n in ('surfaces.json','surface_support.npz','profiles.json')},
                finite_files={p.name:digest(p) for p in finite.iterdir() if p.is_file()},
                STEP_reads=0,target_reads=0,original_point_support_preserved=True))
            row=dict(source_key=key,faces=len(all_faces),native_finite_faces=len(faces),retained_profiles=old_count,
                native_profiles=len(added),profiles=len(profiles),elapsed_seconds=time.perf_counter()-start)
            rows.append(row);write(folder/'receipt.json',dict(status='NATIVE_JOINT_CASE_FROZEN',**row,end_utc=utc()))
            print('NATIVE_JOINT_CASE_FROZEN',key,'FACES',len(all_faces),'PROFILES',len(profiles),'NEW',len(added),flush=True)
            del xyz,pred,instance,material,faces,all_faces,old_faces,profiles,added,contacts,full;gc.collect();reclaim_own_idle_pages()
        write(dest/'receipt.json',dict(status='NATIVE_JOINT_NINE_CANDIDATES_FROZEN',sources=rows,
            original_eight_G=496,supplemental_BM_G=4,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_sources=rows,error=traceback.format_exc(),end_utc=utc()));raise
    finally:builtins.open=original_open;io.open=original_io


if __name__=='__main__':main()
