"""TRAIN-only reference-owner coverage on predicted finite surfaces, no relabeling."""
import argparse
from collections import Counter
import numpy as np
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources


def relation(left,right):
    a=set(left);b=set(right)
    if not a or not b:return 'OWNER_UNOBSERVED'
    if len(a)>1 or len(b)>1:return 'MULTI_OWNER_FACE_SUPPORT'
    return 'SINGLE_OBSERVED_OWNER_SAME' if a==b else 'SINGLE_OBSERVED_OWNERS_DIFFERENT'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);teachers={r['source']:r for r in read(OUT/'run22/label_summary.json')}
    input_bindings={r['source']:r for r in read(OUT/'run22/all_inputs_frozen.json')['sources']};rows=[]
    write(dest/'registered_protocol.json',dict(stage='TRAIN_REFERENCE_MATERIAL_OWNER_COVERAGE',sources=TRAIN,
        offline_reference_only=True,reference_owner_not_customer_approved_truth=True,
        unknown_points_retained=True,relations_describe_observed_support_only_not_full_face_truth=True,
        no_majority_vote_owner_assignment=True,no_threshold_selection=True,
        no_inference_filter=True,no_training=True,original_train_G=359,full_contract_G=496,
        source_sha256=digest(__file__),full_contract_pass=False,start_utc=utc()))
    for source in TRAIN:
        check_resources();folder=OUT/f'run34/source_{source:02d}';binding=read(folder/'input_freeze.json')
        assert binding['raw_sha256']==input_bindings[source]['raw_sha256']
        for name in ['surfaces.json','profiles.json']:assert digest(folder/name)==binding['files'][name]
        teacher_path=OUT/'run22'/teachers[source]['teacher_file'];assert digest(teacher_path)==teachers[source]['teacher_sha256']
        with np.load(teacher_path) as z:owner=z['reference_volume'];ambiguous=z['multiple_material']
        assert np.all(owner[ambiguous]<0)
        faces=read(folder/'surfaces.json');profiles=read(folder/'profiles.json');archives={};face_rows=[]
        try:
            for index,face in enumerate(faces):
                path=ROOT/face['support_file'];key=str(path)
                if key not in archives:
                    assert digest(path)==face['support_sha256'];archives[key]=np.load(path)
                ids=archives[key][face['point_ids_key']].astype(np.int64);values=owner[ids];known=values>=0
                unique,counts=np.unique(values[known],return_counts=True)
                face_rows.append(dict(face=index,support_points=len(ids),known_owner_points=int(known.sum()),
                    unknown_owner_points=int((~known).sum()),ambiguous_contact_points=int(ambiguous[ids].sum()),
                    reference_owner_histogram={str(int(k)):int(v) for k,v in zip(unique,counts)},
                    known_owner_fraction=float(known.mean()),owners=unique.tolist()))
        finally:
            for archive in archives.values():archive.close()
        kinds=[]
        for profile in profiles:
            a,b=profile['face_indices'];kinds.append(relation(face_rows[a]['owners'],face_rows[b]['owners']))
        supervision=read(OUT/f'run35/source_{source:02d}_supervision.json');child_rows=[];children_counts=Counter()
        for child in supervision['children']:
            possible={kinds[ni] for ni in child['profiles']};children_counts.update(possible)
            child_rows.append(dict(parent=child['parent'],child=child['child'],profiles=len(child['profiles']),
                possible_reference_relations=sorted(possible)))
        row=dict(source=source,original_G=supervision['original_G'],faces=len(faces),profiles=len(profiles),
            faces_without_known_owner=sum(not r['owners'] for r in face_rows),
            faces_with_one_known_owner=sum(len(r['owners'])==1 for r in face_rows),
            faces_with_multiple_known_owners=sum(len(r['owners'])>1 for r in face_rows),
            faces_with_unknown_support=sum(r['unknown_owner_points']>0 for r in face_rows),
            profiles_by_reference_relation=dict(Counter(kinds)),
            GT_children_with_any_relation_nonexclusive=dict(children_counts),
            geometrically_supported_children=len(child_rows))
        rows.append(row);write(dest/f'source_{source:02d}_owner_coverage.json',dict(summary=row,faces=face_rows,
            compatible_target_children_posthoc=child_rows,teacher_sha256=digest(teacher_path),
            no_reference_owner_used_as_model_input=True));print('REFERENCE_OWNER_COVERAGE',row,flush=True)
    write(dest/'receipt.json',dict(status='TRAIN_MATERIAL_REFERENCE_COVERAGE_AUDITED',rows=rows,
        original_train_G=sum(r['original_G'] for r in rows),nonexclusive_counts_not_accuracy=True,
        offline_reference_not_customer_approved_truth=True,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
