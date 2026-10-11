"""First two TRAIN cases: learned finite surfaces, frozen before weld labels."""
import argparse
import gc
import time
import traceback
import inspect
import numpy as np
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read,write,digest,serial
from weld_pipeline.sjg_prepare_776 import ROOT,OUT
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_learned_surfaces_810 import extract
from weld_pipeline.sjg_xyz_batch_795 import save_surfaces,contacts_from_support
from weld_pipeline.sjg_xyz_surface_794 import geometry_support
from weld_pipeline.sjg_profiles_798 import profiles_from_contacts,span_representability


def run(args,sources,extractor=extract,extra_protocol=None):
    dest=OUT/args.run;dest.mkdir(exist_ok=False);inventory=read(OUT/'inventory.json');frozen=[]
    write(dest/'registered_protocol.json',dict(stage='LEARNED_PTV3_FINITE_SURFACE_DIAGNOSTIC',sources=sources,
        full_pilot_G=496,evaluated_G=sum(inventory[s]['G'] for s in sources),
        train_only_threshold_audit='run28',embedding_threshold=.4,normal_angle_deg=20.,
        plane_tolerance_mm=.2,min_plane_points=16,per_component_fit_budget=64,
        predicted_feature_checkpoint='run26',features_frozen='run27',STEP_reads=0,
        optimizer_updates=0,full_contract_pass=False,source_sha256=digest(__file__),
        extractor_sha256=digest(inspect.getfile(extractor)),start_utc=utc(),variant=extra_protocol))
    try:
        for source in sources:
            check_resources();folder=dest/f'source_{source:02d}';folder.mkdir();start=time.perf_counter()
            raw=ROOT/inventory[source]['raw']['original_pcd_path'];xyz=read_xyz_pcd(raw)*1000.
            binding=read(OUT/'run27'/f'source_{source:02d}_freeze.json')
            file=OUT/'run27'/binding['feature_file'];assert digest(file)==binding['feature_sha256']
            assert digest(raw)==binding['raw_sha256']
            with np.load(file,allow_pickle=False) as z:pred={k:z[k] for k in ('embedding','normal','boundary')}
            print('LEARNED_SURFACES_START',source,len(xyz),flush=True)
            faces,labels,stats=extractor(xyz,pred,lambda *v:print(source,*v,flush=True))
            np.save(folder/'affinity_components.npy',labels)
            print('LEARNED_SURFACE_FIT',source,stats,flush=True)
            contacts=contacts_from_support(faces,xyz);profiles=profiles_from_contacts(contacts)
            save_surfaces(folder,faces);write(folder/'contacts.json',serial(contacts));write(folder/'profiles.json',serial(profiles))
            record=dict(source=source,original_points=len(xyz),raw_sha256=digest(raw),feature_sha256=binding['feature_sha256'],
                **stats,contacts=len(contacts),profiles=len(profiles),elapsed_seconds=time.perf_counter()-start,
                files={name:digest(folder/name) for name in ('surfaces.json','surface_support.npz','contacts.json','profiles.json','affinity_components.npy')})
            write(folder/'input_freeze.json',record);frozen.append(record)
            print('LEARNED_STRUCTURE_FROZEN',source,'surfaces',len(faces),'contacts',len(contacts),'profiles',len(profiles),flush=True)
            del xyz,pred,faces,labels,contacts,profiles;gc.collect()
        write(dest/'all_inputs_frozen.json',dict(sources=frozen,teacher_and_weld_reads_before_freeze=0))
        rows=[]
        for record in frozen:
            source=record['source'];folder=dest/f'source_{source:02d}'
            target=ROOT/inventory[source]['target']['target_path'];assert digest(target)==inventory[source]['target']['target_sha256']
            truth=read(target);contacts=read(folder/'contacts.json');profiles=read(folder/'profiles.json')
            for c in contacts:c['p0']=np.asarray(c['p0']);c['direction']=np.asarray(c['direction'])
            support=geometry_support(contacts,truth);span=span_representability(profiles,truth)
            write(folder/'geometry_support_only.json',support);write(folder/'span_only.json',span)
            rows.append(dict(source=source,G=support['original_G'],finite_geometry_supported=support['geometric_supported'],
                span_representable=span['span_representable'],surfaces=record['finite_surfaces'],contacts=record['contacts'],profiles=record['profiles']))
            print('LEARNED_GEOMETRY_DIAGNOSTIC',rows[-1],flush=True)
        write(dest/'receipt.json',dict(status='LEARNED_FINITE_SURFACE_DIAGNOSTIC_COMPLETE',end_utc=utc(),rows=rows,
            evaluated_G=sum(row['G'] for row in rows),full_pilot_G=496,
            not_run_pilot_G=496-sum(row['G'] for row in rows),not_weld_extraction_accuracy=True,
            curved_surfaces_and_materials_unresolved=True,optimizer_updates=0,full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),frozen_sources=frozen,error=traceback.format_exc()));raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run(args,[21,4])


if __name__=='__main__':main()
