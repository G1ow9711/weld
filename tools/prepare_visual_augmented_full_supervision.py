"""Join visual finite profiles to complete TRAIN owner supervision and input events.

Existing witnesses/parents/fields/frames are retained; only additional supported
curve alternatives are appended. Reference projections are training labels and
capacity measurements, never runtime endpoints or inserted input events.
"""
import argparse
import gc
from pathlib import Path
from types import SimpleNamespace
import traceback

import numpy as np

from tools.export_full_dataset_split_view import local_name
from tools.prepare_full_task_supervision import verify
from tools.probe_multiview_visual_assistance import load_xyz
from weld_pipeline.server_full_reference_witnesses import ReferenceWitnessBuilder
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_supported_interval_events import FiniteInputIntervalEvents
from weld_pipeline.server_visual_augmented_input import VisualAugmentedInput
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, read, write

BASE = ROOT / 'server_rebuild/full_dataset_v2'


def merge_witnesses(old, extra, profile_offset, children):
    if set(old) != set(extra) or not np.array_equal(old['parent_offsets'], extra['parent_offsets']):
        raise ValueError('Complete reference parent/child layout changed')
    extra = dict(extra, profile=extra['profile'] + profile_offset)
    merged = {k: (old[k].copy() if k == 'parent_offsets' else np.concatenate((old[k], extra[k]))) for k in old}
    owners = merged['attribute_row']
    if len(owners) and (owners.min() < 0 or owners.max() >= children):
        raise ValueError('Additional geometry lost original field ownership')
    counts = np.bincount(owners, minlength=children)
    merged['owner_weight'] = (1. / (children * counts[owners]) if len(owners) else np.empty(0, float))
    for name in old:
        if name not in ('owner_weight', 'parent_offsets') and not np.array_equal(merged[name][:len(old[name])], old[name]):
            raise ValueError('Original witness was altered')
    return merged


def main(args):
    out = BASE / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    source = BASE / local_name(args.visual_experiment)
    receipt = read(source / 'receipt.json'); protocol = read(source / 'protocol.json')
    dataset = PartitionedFullShardDataset(ROOT,BASE)
    if (receipt['status'] != 'ALL_TRAIN_VISUAL_MODEL_AND_AUGMENTED_PROFILE_INPUTS_COMPLETE'
            or protocol['registry_sha256'] != dataset.registry_sha256
            or {r['job_key'] for r in receipt['records']} != set(dataset.jobs)):
        raise ValueError('Completed full TRAIN visual input augmentation required')
    sources = {p: digest(ROOT/p) for p in ['tools/prepare_visual_augmented_full_supervision.py',
        'weld_pipeline/server_visual_augmented_input.py','weld_pipeline/server_full_reference_witnesses.py',
        'weld_pipeline/server_supported_interval_events.py']}
    write(out/'protocol.json',dict(sources=sources,registry_sha256=dataset.registry_sha256,
        visual_receipt_sha256=digest(source/'receipt.json'),reference_geometry_used_as_runtime_input=False,
        original_witnesses_and_full_parent_fields_frames_retained=True,additional_events_from_raw_XYZ_only=True,
        all_reference_children_retained=True,official_validation_or_test_reads=0,
        full_model_training_or_accuracy_claimed=False))
    rows=[];events_total=0;profiles_total=0
    try:
        for item in receipt['records']:
            key=item['job_key'];path=source/item['path']
            if digest(path)!=item['sha256']:raise ValueError('Visual input candidates changed')
            addition=read(path)
            if (addition['original_profiles_sha256']!=digest(dataset.shards/key/'profiles.json')
                    or addition['original_input_receipt_sha256']!=digest(dataset.shards/key/'receipt.json')):
                raise ValueError('Original finite input binding changed')
            original=dataset.input_case(key);inp=VisualAugmentedInput(original,addition)
            folder=out/key;folder.mkdir();event_folder=folder/'events';event_folder.mkdir()
            # Native and added curves share the same authenticated raw-input
            # identity. The separate addition binding authenticates new curves.
            xyz,_=load_xyz(dataset.jobs[key]);producer=FiniteInputIntervalEvents(
                xyz, addition['original_input_receipt_sha256'])
            events=[]
            for pid in range(inp.original_profile_count,len(inp.profiles)):
                bank=producer.build(inp.profiles[pid],np.array([[0.,inp.lengths[pid]]]))
                name=f'profile_{pid:07d}.npz'
                np.savez_compressed(event_folder/name,positions_mm=bank['positions_mm'],xyz_mm=bank['xyz_mm'],
                    provenance=bank['provenance'],projection_point_ids=bank['projection_point_ids'])
                events.append(dict(profile_id=pid,path=name,sha256=digest(event_folder/name),
                    curve_representation=bank['curve_representation'],length_mm=bank['length_mm'],audit=bank['audit']))
                events_total+=len(bank['positions_mm']);profiles_total+=1
            write(event_folder/'receipt.json',dict(profiles=events,raw_points=len(xyz),
                  augmented_input_receipt_sha256=digest(path),
                  shared_raw_input_receipt_sha256=addition['original_input_receipt_sha256'],GT_geometry_reads=0))
            builder=ReferenceWitnessBuilder(SimpleNamespace(profiles=addition['additional_profiles']))
            for version in dataset.jobs[key]['versions']:
                vk=version['key'];view=dataset.supervision(vk)
                prior=BASE/'full_task_supervision_v1'/key/vk;verify(prior,dataset,version)
                with np.load(prior/'owner_witnesses.npz') as z:old={k:z[k].copy() for k in z.files}
                new,new_report=builder.build(view)
                merged=merge_witnesses(old,new,inp.original_profile_count,version['children'])
                target=folder/vk;target.mkdir();np.savez_compressed(target/'owner_witnesses.npz',**merged)
                old_owners=set(old['attribute_row'].tolist());new_owners=set(new['attribute_row'].tolist())
                row=dict(job_key=key,version_key=vk,children=version['children'],
                    prior_represented_children=len(old_owners),combined_represented_children=len(old_owners|new_owners),
                    newly_represented_children=sorted(new_owners-old_owners),additional_witnesses=len(new['profile']),
                    original_witnesses_preserved=len(old['profile']),original_label_directory=version['label_directory'],
                    original_label_receipt_sha256=version['label_receipt_sha256'],
                    known_attribute_entries=int(view.label_known.sum()),known_endpoint_frames=int(view.frame_algebraically_known.sum()),
                    original_supervision_receipt_sha256=digest(prior/'receipt.json'),
                    input_addition_sha256=digest(path),witness_sha256=digest(target/'owner_witnesses.npz'),
                    input_events_receipt_sha256=digest(event_folder/'receipt.json'),new_capacity_report=new_report,
                    projected_reference_intervals_are_offline_supervision_only=True)
                write(target/'receipt.json',row);rows.append(row)
            write(out/'progress.json',dict(jobs=sum(1 for p in out.iterdir() if p.is_dir()),versions=len(rows),
                children=sum(r['children'] for r in rows),additional_profiles=profiles_total,additional_events=events_total))
            print('VISUAL_FULL_SUPERVISION',key,len(rows),len(events),flush=True)
            del xyz,producer,inp,original,builder;gc.collect()
        if {r['version_key'] for r in rows}!=set(dataset.versions) or sources!={p:digest(ROOT/p) for p in sources}:
            raise ValueError('Incomplete full supervision or changed source')
        write(out/'receipt.json',dict(status='ALL_REGISTERED_TRAIN_VISUAL_AUGMENTED_SUPERVISION_AND_EVENTS_READY',
            versions=len(rows),children=sum(r['children'] for r in rows),additional_profiles=profiles_total,
            additional_events=events_total,prior_represented_children=sum(r['prior_represented_children'] for r in rows),
            combined_represented_children=sum(r['combined_represented_children'] for r in rows),
            newly_represented_children=sum(len(r['newly_represented_children']) for r in rows),records=rows,
            runtime_endpoints_not_predicted=True,downstream_full_model_training_pending=True,
            full_extraction_accuracy_claimed=False,official_validation_or_test_reads=0))
    except BaseException as exc:
        write(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),completed_versions=len(rows)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment-name',required=True)
    p.add_argument('--visual-experiment',required=True);main(p.parse_args())
