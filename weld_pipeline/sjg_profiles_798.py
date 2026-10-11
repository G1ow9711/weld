"""Input-only contact profiles with explicit supported intervals and unknown gaps.

Profiles consolidate only the same inferred plane pair. Spanning a sampling
void does not certify material or weld continuity. Events must learn that next.
"""
import argparse
import time
import traceback
import numpy as np
from weld_pipeline.sjg_geometry_776 import read,write,digest,serial
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources


def profiles_from_contacts(contacts):
    groups={}
    for i,c in enumerate(contacts):groups.setdefault(tuple(c['face_indices']),[]).append((i,c))
    profiles=[]
    for pair,items in groups.items():
        origin=np.asarray(items[0][1]['p0']);direction=np.asarray(items[0][1]['direction'])
        pieces=[]
        for i,c in items:
            assert abs(direction@c['direction'])>1-1e-8
            endpoints=np.asarray([c['p0'],c['p1']]);q=(endpoints-origin)@direction
            residual=endpoints-(origin+q[:,None]*direction)
            assert np.linalg.norm(residual,axis=1).max()<1e-4
            pieces.append((float(q.min()),float(q.max()),i))
        start=min(v[0] for v in pieces);end=max(v[1] for v in pieces)
        intervals=[]
        for lo,hi,_ in sorted(pieces):
            lo-=start;hi-=start
            if intervals and lo<=intervals[-1][1]+1e-5:intervals[-1][1]=max(intervals[-1][1],hi)
            else:intervals.append([lo,hi])
        profiles.append(dict(p0=origin+start*direction,p1=origin+end*direction,direction=direction,
            length=end-start,face_indices=list(pair),materials=None,
            material_status='UNKNOWN_NOT_INFERRED',support_intervals_mm=intervals,
            original_contact_ids=[i for i,_ in items],
            unobserved_gaps_mm=[[a[1],b[0]] for a,b in zip(intervals[:-1],intervals[1:])],
            sampling_void_is_not_certified_hole=True,weld_continuity_not_assumed=True))
    assert sum(len(p['original_contact_ids']) for p in profiles)==len(contacts)
    return profiles


def span_representability(profiles,target):
    origin=np.asarray([p['p0'] for p in profiles],float).reshape(-1,3)
    direction=np.asarray([p['direction'] for p in profiles],float).reshape(-1,3)
    length=np.asarray([p['length'] for p in profiles]);rows=[]
    for pi,parent in enumerate(target['parents']):
        for ci,child in enumerate(parent['children']):
            if child['original_kind']!='line':
                rows.append(dict(parent=pi,child=ci,representable=False,reason='CURVE_NOT_IMPLEMENTED'));continue
            endpoints=np.asarray(child['world_representation']['endpoints_mm'])
            best=None
            if len(origin):
                error=[]
                for point in endpoints:
                    q=np.einsum('ij,ij->i',point-origin,direction)
                    nearest=origin+np.clip(q,0,length)[:,None]*direction
                    error.append(np.linalg.norm(nearest-point,axis=1))
                best=float(np.maximum(*error).min())
            rows.append(dict(parent=pi,child=ci,representable=best is not None and best<=3.,span_distance_mm=best))
    return dict(original_G=target['original_children'],span_representable=sum(r['representable'] for r in rows),
        rows=rows,not_finite_support_coverage=True,not_weld_accuracy=True,
        unsupported_intervals_need_learned_events=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run18');args=ap.parse_args()
    out=OUT/args.run;out.mkdir(exist_ok=False);started=time.monotonic()
    try:
        inventory=read(OUT/'inventory.json');frozen=[]
        write(out/'registered_protocol.json',dict(stage='PURE_XYZ_PROFILE_REPRESENTATION',sources=TRAIN+VALID+ARC,
            source_sha256=digest(__file__),geometry_source='run16',STEP_reads=0,weld_GT_reads_during_generation=0,
            surface_scores_retained_as_soft_features='run17',same_plane_pair_only=True,
            unsupported_gaps_explicit=True,optimizer_updates=0,full_contract_pass=False))
        for si in TRAIN+VALID+ARC:
            check_resources();folder=OUT/f'run16/source_{si:02d}';binding=read(folder/'input_freeze.json')
            path=folder/'contacts.json';assert digest(path)==binding['files']['contacts.json']
            contacts=read(path);profiles=profiles_from_contacts(contacts)
            path=out/f'source_{si:02d}_profiles.json';write(path,serial(profiles))
            frozen.append(dict(source=si,path=path.name,sha256=digest(path),profiles=len(profiles),
                original_contacts=len(contacts),profiles_with_unobserved_gaps=sum(bool(p['unobserved_gaps_mm']) for p in profiles)))
            print('FROZEN_XYZ_PROFILES',si,len(profiles),'from_contacts',len(contacts),flush=True)
        write(out/'all_profiles_frozen.json',dict(outputs=frozen,all_frozen_before_any_weld_target_read=True))
        rows=[]
        for f in frozen:
            si=f['source'];path=out/f['path'];assert digest(path)==f['sha256']
            target=ROOT/inventory[si]['target']['target_path'];assert digest(target)==inventory[si]['target']['target_sha256']
            score=span_representability(read(path),read(target));write(out/f'source_{si:02d}_span_diagnostic.json',score)
            row=dict(source=si,original_G=score['original_G'],span_representable=score['span_representable'],profiles=f['profiles'])
            rows.append(row);print('SPAN_REPRESENTABILITY_ONLY',row,flush=True)
        write(out/'receipt.json',dict(status='PURE_XYZ_PROFILES_FROZEN_AND_DIAGNOSED',utc=utc(),rows=rows,
            original_G=sum(r['original_G'] for r in rows),span_representable=sum(r['span_representable'] for r in rows),
            not_weld_accuracy=True,not_finite_support_coverage=True,material_status='UNKNOWN',
            optimizer_updates=0,full_contract_pass=False,elapsed_seconds=time.monotonic()-started))
        print('XYZ_PROFILES_COMPLETE | NO WELD ACCURACY CLAIM',flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),optimizer_updates=0));raise


if __name__=='__main__':main()
