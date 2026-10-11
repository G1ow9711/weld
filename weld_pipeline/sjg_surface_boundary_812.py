"""Recover shared boundary support around frozen learned plane seeds."""
import argparse
import gc
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read,write,digest,serial
from weld_pipeline.sjg_prepare_776 import ROOT,OUT
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_learned_surfaces_810 import finite_support
from weld_pipeline.sjg_xyz_batch_795 import save_surfaces,contacts_from_support
from weld_pipeline.sjg_xyz_surface_794 import geometry_support
from weld_pipeline.sjg_profiles_798 import profiles_from_contacts,span_representability


def shared_boundary_support(xyz,planes,progress=None):
    faces=[];rows=[]
    for index,plane in enumerate(planes):
        base_ids=plane['point_ids'];base=xyz[base_ids];uv=(base-plane['center'])@plane['basis']
        tree=cKDTree(uv);radii=tree.query(uv,k=min(24,len(uv)),workers=2)[0][:,-1]
        added=[]
        # Exactly one expansion from original support. Newly added points never
        # become seeds for another expansion, preventing unbounded propagation.
        for start in range(0,len(xyz),65536):
            points=xyz[start:start+65536]
            near=np.flatnonzero(np.abs(points@plane['normal']-plane['offset'])<=.2)
            if not len(near):continue
            projected=(points[near]-plane['center'])@plane['basis'];distance,nn=tree.query(projected,workers=2)
            support=distance<=np.maximum(2.,1.5*radii[nn])
            added.append((start+near[support]).astype(np.int64))
        ids=np.unique(np.concatenate([base_ids]+added))
        candidate={**{k:plane[k] for k in ('normal','center','offset','basis','learned_component_ids')},
            'point_ids':ids,'boundary_support_shared':True,'boundary_expansion_iterations':1,
            'original_learned_support_points':len(base_ids),'original_plane_not_refitted_from_expansion':True}
        face=finite_support(xyz,candidate)
        if face is not None:faces.append(face)
        rows.append(dict(plane=index,seed_points=len(base_ids),expanded_points=len(ids),finite_support_retained=face is not None))
        if progress and index%25==0:progress('SHARED_BOUNDARY_SUPPORT',index+1,len(planes))
    return faces,rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);sources=[21,4];inventory=read(OUT/'inventory.json');frozen=[]
    write(dest/'registered_protocol.json',dict(stage='SHARED_BOUNDARY_SUPPORT_ON_LEARNED_PLANES',sources=sources,
        source_plane_run='run29',single_expansion_only=True,point_plane_distance_mm=.2,
        local_radius_rule='max(2mm,1.5*original-support 24NN radius)',
        STEP_reads=0,weld_GT_reads_during_generation=0,optimizer_updates=0,full_contract_pass=False,source_sha256=digest(__file__)))
    try:
        for si in sources:
            check_resources();start=time.perf_counter();folder=dest/f'source_{si:02d}';folder.mkdir()
            original=OUT/'run29'/f'source_{si:02d}';binding=read(original/'input_freeze.json')
            for name in ('surfaces.json','surface_support.npz'):assert digest(original/name)==binding['files'][name]
            raw=ROOT/inventory[si]['raw']['original_pcd_path'];assert digest(raw)==binding['raw_sha256']
            xyz=read_xyz_pcd(raw)*1000.;planes=read(original/'surfaces.json')
            with np.load(original/'surface_support.npz') as arrays:
                for plane in planes:
                    plane['point_ids']=arrays[plane['point_ids_key']]
                    for key in ('normal','center','basis'):plane[key]=np.asarray(plane[key])
            faces,expanded=shared_boundary_support(xyz,planes,lambda *v:print(si,*v,flush=True))
            contacts=contacts_from_support(faces,xyz);profiles=profiles_from_contacts(contacts)
            save_surfaces(folder,faces);write(folder/'contacts.json',serial(contacts));write(folder/'profiles.json',serial(profiles));write(folder/'expansion.json',expanded)
            record=dict(source=si,surfaces=len(faces),contacts=len(contacts),profiles=len(profiles),raw_sha256=digest(raw),
                elapsed_seconds=time.perf_counter()-start,files={name:digest(folder/name) for name in ('surfaces.json','surface_support.npz','contacts.json','profiles.json')})
            write(folder/'input_freeze.json',record);frozen.append(record);print('BOUNDARY_STRUCTURE_FROZEN',record,flush=True)
            del xyz,planes,faces,contacts,profiles;gc.collect()
        write(dest/'all_inputs_frozen.json',dict(sources=frozen,all_frozen_before_weld_GT_read=True))
        rows=[]
        for si in sources:
            folder=dest/f'source_{si:02d}';truth=read(ROOT/inventory[si]['target']['target_path'])
            contacts=read(folder/'contacts.json');profiles=read(folder/'profiles.json')
            for c in contacts:c['p0']=np.asarray(c['p0']);c['direction']=np.asarray(c['direction'])
            support=geometry_support(contacts,truth);span=span_representability(profiles,truth)
            write(folder/'geometry_support_only.json',support);write(folder/'span_only.json',span)
            rows.append(dict(source=si,G=support['original_G'],finite_geometry_supported=support['geometric_supported'],span_representable=span['span_representable'],contacts=len(contacts),profiles=len(profiles)))
        write(dest/'receipt.json',dict(status='LEARNED_SHARED_BOUNDARY_DIAGNOSTIC_COMPLETE',rows=rows,end_utc=utc(),
            full_pilot_G=496,evaluated_G=37,not_run_pilot_G=459,not_weld_extraction_accuracy=True,full_contract_pass=False))
        print('SHARED_BOUNDARY_DIAGNOSTIC',rows,flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
