"""Preserve geometric recall and add learned proposals; caches are training data."""
import argparse
import gc
import time
import traceback
import numpy as np
from weld_pipeline.sjg_geometry_776 import read,write,digest,serial
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_profiles_798 import profiles_from_contacts,span_representability


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);inventory=read(OUT/'inventory.json');freezes=[]
    write(dest/'registered_protocol.json',dict(stage='PURE_INPUT_GEOMETRY_AND_PTV3_PROPOSAL_UNION',
        sources=TRAIN+VALID+ARC,branches=['run16','run33'],original_G=496,
        prior_frontend_replacement_rejected='run33:336/496 vs run16:429/496 finite coverage',
        keep_all_original_geometric_proposals=True,learned_features_from='run27',
        source_labels_not_model_inputs=True,frozen_caches_for_training_only=True,
        deployment_must_recompute_from_input_not_retrieve_by_identity=True,
        STEP_reads=0,weld_GT_reads_before_freeze=0,optimizer_updates=0,full_contract_pass=False,
        source_sha256=digest(__file__),start_utc=utc()))
    try:
        for si in TRAIN+VALID+ARC:
            check_resources();folder=dest/f'source_{si:02d}';folder.mkdir();faces=[];contacts=[];branches=[]
            raw_sha=None
            for branch in ('run16','run33'):
                source=OUT/branch/f'source_{si:02d}';binding=read(source/'input_freeze.json')
                for name in ('surfaces.json','surface_support.npz','contacts.json'):assert digest(source/name)==binding['files'][name]
                current=binding.get('original_PCD_sha256',binding.get('raw_sha256'))
                if raw_sha is not None:assert raw_sha==current
                raw_sha=current;offset=len(faces);descriptors=read(source/'surfaces.json')
                for fi,face in enumerate(descriptors):
                    face.update(proposal_branch=branch,branch_face_id=fi,
                        support_file=str((source/'surface_support.npz').relative_to(ROOT)),
                        support_sha256=binding['files']['surface_support.npz'])
                    faces.append(face)
                original=read(source/'contacts.json')
                for ci,c in enumerate(original):
                    c['face_indices']=[i+offset for i in c['face_indices']]
                    c.update(proposal_branch=branch,branch_contact_id=ci)
                    contacts.append(c)
                branches.append(dict(branch=branch,faces=len(descriptors),contacts=len(original),input_freeze_sha256=digest(source/'input_freeze.json')))
            profiles=profiles_from_contacts(contacts)
            for p in profiles:p['proposal_branch']=faces[p['face_indices'][0]]['proposal_branch']
            write(folder/'surfaces.json',faces);write(folder/'contacts.json',contacts);write(folder/'profiles.json',serial(profiles))
            record=dict(source=si,raw_sha256=raw_sha,branches=branches,surfaces=len(faces),contacts=len(contacts),profiles=len(profiles),
                files={name:digest(folder/name) for name in ('surfaces.json','contacts.json','profiles.json')})
            write(folder/'input_freeze.json',record);freezes.append(record)
            print('HYBRID_CANDIDATES_FROZEN',si,len(faces),len(contacts),len(profiles),flush=True)
            del faces,contacts,profiles;gc.collect()
        write(dest/'all_inputs_frozen.json',dict(sources=freezes,all_before_any_weld_GT_read=True))
        rows=[]
        for item in freezes:
            si=item['source'];folder=dest/f'source_{si:02d}';target=ROOT/inventory[si]['target']['target_path']
            assert digest(target)==inventory[si]['target']['target_sha256'];truth=read(target)
            # Contacts are already individual finite supported intervals; the
            # vectorized bounded-line evaluator is equivalent to endpoint checks.
            finite=span_representability(read(folder/'contacts.json'),truth)
            spans=span_representability(read(folder/'profiles.json'),truth)
            write(folder/'finite_support_diagnostic.json',finite);write(folder/'profile_span_diagnostic.json',spans)
            rows.append(dict(source=si,G=finite['original_G'],finite_geometry_supported=finite['span_representable'],
                span_representable=spans['span_representable'],contacts=item['contacts'],profiles=item['profiles']))
            print('HYBRID_GEOMETRY_DIAGNOSTIC',rows[-1],flush=True)
        write(dest/'receipt.json',dict(status='HYBRID_INPUT_CANDIDATES_FROZEN_AND_EVALUATED',rows=rows,
            original_G=sum(r['G'] for r in rows),finite_geometry_supported=sum(r['finite_geometry_supported'] for r in rows),
            span_representable=sum(r['span_representable'] for r in rows),end_utc=utc(),
            not_weld_extraction_accuracy=True,material_unknown=True,curves_not_implemented=True,full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
