"""Reuse authenticated complete XYZ features under the corrected registration.

Input arrays and analytic profiles are linked without modification.
Only registration, family and label-owner metadata is rebound. No supervision
geometry is opened and no model or input feature is fitted or rewritten.
"""
import argparse
import copy
import errno
import gc
import os
from pathlib import Path
import tempfile
import time
import traceback

from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'server_rebuild/full_dataset_v1'


def main(args):
    if Path(args.dataset_name).name != args.dataset_name or args.dataset_name in ('', '.', '..', 'full_dataset_v1'):
        raise ValueError('A repaired independent dataset directory is required')
    base = ROOT/'server_rebuild'/args.dataset_name
    proof = read(base/'registration_repair/receipt.json')
    if proof['status'] != 'ALL_REPAIRED_TRAIN_AND_HELD_VERSIONS_REGISTERED' or proof['bindings'] != {name:digest(base/name) for name in proof['bindings']}:
        raise ValueError('The complete corrected model-use registration is required')
    old, new = FullShardDataset(ROOT), PartitionedFullShardDataset(ROOT,base)
    out = base/args.experiment_name
    if Path(args.experiment_name).name != args.experiment_name or args.experiment_name in ('', '.', '..'):
        raise ValueError('An independent local reuse record is required')
    out.mkdir(exist_ok=False)
    new.shards.mkdir(exist_ok=True)
    source_sha = digest(__file__)
    started = time.perf_counter()
    sources = {name:digest(ROOT/name) for name in ['tools/reuse_repaired_full_input_shards.py',
        'weld_pipeline/server_partitioned_full_dataset.py', 'weld_pipeline/server_full_shard_dataset.py']}
    write(out/'protocol.json',dict(sources=sources, original_registry_sha256=old.registry_sha256,
        repaired_registry_sha256=new.registry_sha256, registration_proof_sha256=digest(base/'registration_repair/receipt.json'),
        GT_geometry_reads=0, model_fit_calls=0, only_feature_bytes_and_metadata_checked=True))
    try:
        reused = []
        for key in old.ready_jobs():
            if key not in new.jobs:
                continue
            source, destination = old.shards/key, new.shards/key
            if destination.exists():
                continue
            before = read(source/'receipt.json')
            old_case = old.input_case(key)
            if before['workpiece_sha256'] != new.jobs[key]['workpiece_sha256']:
                raise ValueError('Only identical complete input clouds can reuse features')
            before_sha = digest(source/'receipt.json')
            with tempfile.TemporaryDirectory(prefix=key+'-',dir=new.shards) as temporary:
                staging = Path(temporary)/'result';staging.mkdir()
                for name, sha in before['files'].items():
                    if Path(name).name != name or digest(source/name) != sha:
                        raise ValueError('A complete original feature artifact changed')
                    try:
                        os.link(source/name,staging/name)
                    except OSError as exc:
                        if exc.errno != errno.EXDEV:
                            raise
                        (staging/name).symlink_to((source/name).resolve(strict=True))
                    if digest(staging/name) != sha:
                        raise ValueError('A reused feature byte changed')
                freeze = copy.deepcopy(read(source/'input_freeze.json'))
                freeze.update(registry_sha256=new.registry_sha256, source_sha256=source_sha,
                    feature_producer_source_sha256=before['source_sha256'],
                    input_feature_values_recomputed_or_changed=False,
                    original_input_receipt=str((source/'receipt.json').relative_to(ROOT)),
                    original_input_receipt_sha256=before_sha,
                    metadata_rebinding_only=True)
                write(staging/'input_freeze.json',freeze)
                write(staging/'supervision_versions.json',dict(versions=new.jobs[key]['versions'],label_geometry_as_input=False))
                receipt=dict(status='FULL_UNSCREENED_INPUT_SHARD_READY_NOT_TRAINED',**freeze,
                    supervision_versions_sha256=digest(staging/'supervision_versions.json'),
                    training_versions=len(new.jobs[key]['versions']),
                    families=sorted({v['family_id'] for v in new.jobs[key]['versions']}))
                write(staging/'receipt.json',receipt)
                if digest(source/'receipt.json') != before_sha:
                    raise ValueError('Original completed input binding changed during reuse')
                os.replace(staging,destination)
            new_case = new.input_case(key)
            if old_case.n != new_case.n or old_case.all_query_count != new_case.all_query_count or old_case.profiles != new_case.profiles:
                raise ValueError('Complete input profile or query coverage changed')
            reused.append(dict(job_key=key, original_receipt_sha256=before_sha,
                repaired_receipt_sha256=digest(destination/'receipt.json'),
                all_input_feature_artifact_sha256=before['files'], feature_bytes_preserved=True))
            del old_case,new_case;gc.collect()
            write(out/'progress.json',dict(status='AUTHENTIC_COMPLETE_INPUT_METADATA_REUSE',controller_pid=os.getpid(),
                completed_jobs=len(reused), GT_geometry_reads=0, model_fit_calls=0,unix_time=time.time()))
            print('REPAIRED_INPUT_SHARD_REUSED',len(reused),key,flush=True)
        if sources != {name:digest(ROOT/name) for name in sources}:
            raise ValueError('Input reuse implementation changed')
        write(out/'receipt.json',dict(status='ORIGINAL_COMPLETE_INPUT_FEATURES_AUTHENTICATED_AND_REBOUND',
            sources=sources,protocol_sha256=digest(out/'protocol.json'),reused=reused,
            newly_reused_input_jobs=len(reused),ready_repaired_input_jobs=len(new.ready_jobs()),
            registered_repaired_input_jobs=len(new.jobs),GT_geometry_reads=0,model_fit_calls=0,
            feature_values_or_original_data_or_models_changed=False,
            remaining_inputs_and_full_model_training_required=True,seconds=time.perf_counter()-started))
        print('COMPLETE_REPAIRED_INPUT_REUSE',len(new.ready_jobs()),'/',len(new.jobs),flush=True)
    except Exception:
        write(out/'failure.json',dict(traceback=traceback.format_exc(),unix_time=time.time()))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-name',required=True)
    parser.add_argument('--experiment-name',required=True);main(parser.parse_args())
