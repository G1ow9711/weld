"""Fit all repaired TRAIN views from a genuine earlier full-weld checkpoint.

The complete original fit loop, losses, sampler, optimizer and checkpoint
writer are reused. Dataset registration and authenticated full-state loading
replace the original anchored initialization. Three fresh complete passes use
the corrected family weights; old updates are recorded as pretraining only.
"""
import argparse
import os
from pathlib import Path
from types import FunctionType

import torch

from tools import train_full_weld_reference_recall as original
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_reference_recall_weld_model import ReferenceRecallWeldModel, ARCHITECTURE
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = original.ROOT
OLD = original.BASE
SOURCES = ['tools/train_partitioned_full_weld_reference_recall.py',
    'weld_pipeline/server_partitioned_full_dataset.py',
    'tools/prepare_partitioned_full_task_supervision.py',
    'tools/reuse_repaired_full_input_shards.py', 'tools/register_repaired_full_dataset.py',
    'tools/qualify_repaired_full_dataset_labels.py', 'tools/repair_full_dataset_family_split.py']


class FullStateReferenceRecallWeldModel(ReferenceRecallWeldModel):
    def inherit_anchored(self, state):
        self.load_state_dict(state,strict=True)


def configure(base, source, expected_sha):
    source = Path(source).resolve();source.relative_to(ROOT)
    if source.parent != OLD/'full_weld_reference_recall_v3' or digest(source) != expected_sha:
        raise ValueError('An immutable actual original full-weld checkpoint is required')
    repaired = PartitionedFullShardDataset(ROOT,base)
    proof = read(base/'registration_repair/receipt.json')
    if proof['bindings'] != {name:digest(base/name) for name in proof['bindings']}:
        raise ValueError('Corrected complete registration changed')
    state = torch.load(source,map_location='cpu',weights_only=False)
    if (state['architecture'] != ARCHITECTURE or state['optimizer_steps'] <= 0
            or state['protocol_sha256'] != digest(OLD/'full_weld_reference_recall_v3/protocol.json')
            or state['sources'] != {name:digest(ROOT/name) for name in state['sources']}):
        raise ValueError('Actual earlier model architecture, fit, protocol or sources changed')
    fitted = {key.split(':',1)[1] for key in state['completed_views']}
    if not fitted or not fitted <= set(repaired.versions):
        raise ValueError('Previously fitted labels cannot become validation, TEST or protected diagnostics')
    comparison = read(base/'qualification_repair/receipt.json')
    if comparison['repaired_qualification_sha256'] != digest(base/'qualification_summary.json'):
        raise ValueError('Corrected labels differ from the verified full requalification')
    verified = {row['key'] for row in comparison['compared'] if row['geometry_attributes_and_raw_frames_unchanged']}
    if not fitted <= verified:
        raise ValueError('Every previous fitted geometry and raw frame must be preserved')
    model = FullStateReferenceRecallWeldModel()
    model.inherit_anchored(state['state_dict'])
    if any(not torch.equal(value.cpu(),state['state_dict'][name].cpu()) for name,value in model.state_dict().items()):
        raise ValueError('Full checkpoint loading changed a model tensor')
    if state['mean'].shape != (457,) or state['std'].shape != (457,) or not (state['std']>0).all():
        raise ValueError('Original TRAIN feature normalization changed')
    bindings = {name:digest(base/name) for name in ['split_manifest.json','training_registry.json',
        'model_use_policy.json','qualification_repair/receipt.json','registration_repair/receipt.json']}
    evidence = dict(status='ACTUAL_FULL_WELD_WEIGHTS_ADMITTED_TO_CORRECTED_TRAIN_ONLY',
        source_checkpoint=str(source.relative_to(ROOT)), source_checkpoint_sha256=expected_sha,
        architecture=ARCHITECTURE, historical_optimizer_updates=state['optimizer_steps'],
        previous_fitted_versions=len(fitted), registered_repaired_versions=len(repaired.versions),
        all_prior_fitted_versions_remain_train=True, all_loaded_model_tensors_identical=True,
        original_train_mean_and_std_retained=True, corrected_curriculum_full_passes_required=True,
        old_optimizer_stored_in_source_checkpoint=True, new_corrected_curriculum_optimizer='FRESH_ORIGINAL_ADAMW',
        historical_updates_not_counted_as_new_full_coverage=True,
        sources={name:digest(ROOT/name) for name in [*original.SOURCES,*SOURCES]},
        repaired_bindings=bindings, model_output_quality_not_yet_evaluated=True,
        geometry_labels_and_signed_frames_preserved=True, held_labels_not_fitted=True)
    path = base/'partitioned_query_initialization_binding.json'
    if path.exists() and read(path) != evidence:
        raise ValueError('Immutable corrected initialization binding already differs')
    if not path.exists():write(path,evidence)

    def initialization(out):
        if digest(source) != expected_sha or bindings != {name:digest(base/name) for name in bindings}:
            raise ValueError('Frozen pretraining model or corrected registration changed')
        destination = out/'warm_initialization.pt'
        if not destination.exists():
            temporary = destination.with_suffix('.incomplete')
            temporary.write_bytes(source.read_bytes())
            if digest(temporary) != expected_sha:raise ValueError('Full checkpoint copy changed')
            os.replace(temporary,destination)
        if digest(destination) != expected_sha:raise ValueError('Warm initialization changed')
        return state,destination

    environment = dict(original.main.__globals__)
    environment.update(BASE=base, FullShardDataset=lambda value:PartitionedFullShardDataset(value,base),
        FullWeldModel=FullStateReferenceRecallWeldModel, load_initialization=initialization,
        SOURCES=list(dict.fromkeys([*original.SOURCES,*SOURCES])))
    return FunctionType(original.main.__code__,environment,original.main.__name__,
                        original.main.__defaults__,original.main.__closure__)


def main(args):
    if Path(args.dataset_name).name != args.dataset_name or args.dataset_name in ('', '.', '..', 'full_dataset_v1'):
        raise ValueError('An independent repaired dataset directory is required')
    if args.epochs != 3 or args.steps_per_view != 8:
        raise ValueError('The complete declared three-pass eight-step curriculum is required')
    run = configure(ROOT/'server_rebuild'/args.dataset_name,args.warm_checkpoint,args.warm_sha256)
    run(args.epochs,args.steps_per_view)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-name',required=True)
    parser.add_argument('--warm-checkpoint',required=True);parser.add_argument('--warm-sha256',required=True)
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--steps-per-view',type=int,default=8)
    main(parser.parse_args())
