"""Freeze third finite surface context for the fixed training/inference profiles."""
import argparse
import gc
import traceback
import numpy as np
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_third_surface_838 import finite_events,adjacency_features


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);inventory=read(OUT/'inventory.json')
    train_records=read(OUT/'run38/manifest.json')['sequences'];rows=[];frozen=[]
    write(dest/'registered_protocol.json',dict(stage='FINITE_THIRD_SURFACE_INPUTS',sources=TRAIN+VALID+ARC,
        original_G=496,train_selection='Unchanged run38 TRAIN profile list; original TRAIN target based positive inclusion retained',
        inference_selection='Unchanged input-only run45 profile list',grid_step_mm=1.,extra_features=12,
        feature_source_sha256=digest(ROOT/'weld_pipeline/sjg_third_surface_838.py'),source_sha256=digest(__file__),
        finite_support_tolerance_mm=3.,no_fixed_endpoint_cutback=True,no_GT_or_STEP_reads=True,
        material_ID_not_inferred=True,all_features_input_only=True,full_contract_pass=False,start_utc=utc()))
    try:
        for source in TRAIN+VALID+ARC:
            check_resources();folder=dest/f'source_{source:02d}';folder.mkdir()
            base=OUT/f'run34/source_{source:02d}';binding=read(base/'input_freeze.json')
            for name in ('profiles.json','surfaces.json'):assert digest(base/name)==binding['files'][name]
            raw=ROOT/inventory[source]['raw']['original_pcd_path'];assert digest(raw)==binding['raw_sha256']
            xyz=read_xyz_pcd(raw)*1000.;faces=read(base/'surfaces.json');profiles=read(base/'profiles.json')
            inference_ids=[r['profile'] for r in read(OUT/f'run45/source_{source:02d}_fields.json')['files']]
            train_ids=[r['profile'] for r in train_records if r['source']==source]
            selected=sorted(set(inference_ids+train_ids));archives={}
            def support_loader(face):
                path=ROOT/face['support_file'];key=str(path)
                if key not in archives:
                    assert digest(path)==face['support_sha256'];archives[key]=np.load(path)
                archive=archives[key];return archive[face['point_ids_key']],archive[face['triangles_key']]
            try:events=finite_events(xyz,faces,profiles,selected,support_loader,lambda *a:print(source,*a,flush=True))
            finally:
                for archive in archives.values():archive.close()
            records=[]
            for ni in selected:
                q=np.linspace(0,profiles[ni]['length'],int(np.ceil(profiles[ni]['length']))+1)
                x=adjacency_features(q,events[ni]);path=folder/f'profile_{ni:06d}.npz'
                np.savez_compressed(path,q_mm=q,extra=x)
                records.append(dict(profile=ni,file=path.name,sha256=digest(path),points=len(q),events=len(events[ni])))
            write(folder/'events.json',dict(events=events,input_only=True))
            write(folder/'manifest.json',dict(profiles=records,inference_ids=inference_ids,train_ids=train_ids,
                raw_sha256=digest(raw),profile_sha256=digest(base/'profiles.json'),surface_sha256=digest(base/'surfaces.json')))
            frozen.append(dict(source=source,manifest_sha256=digest(folder/'manifest.json'),events_sha256=digest(folder/'events.json')))
            row=dict(source=source,profiles=len(selected),profiles_with_events=sum(bool(v) for v in events.values()),
                finite_third_face_events=sum(len(v) for v in events.values()),grid_points=sum(r['points'] for r in records))
            rows.append(row);print('THIRD_SURFACE_INPUTS_FROZEN',row,flush=True)
            del xyz,faces,profiles,events,archives;gc.collect()
        write(dest/'all_inputs_frozen.json',dict(sources=frozen,GT_reads=0,STEP_reads=0))
        write(dest/'receipt.json',dict(status='FINITE_THIRD_SURFACE_FEATURES_COMPLETE',rows=rows,optimizer_updates=0,
            end_utc=utc(),full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
