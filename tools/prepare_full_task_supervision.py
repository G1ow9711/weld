"""Stream complete TRAIN owner supervision as all registered inputs arrive.

This is not a model fit. Complete parent/process/frame labels stay in the
qualified persistent label directory; compact carrier/query witnesses are
rebuildable task-owned temporary artifacts. No fixed pilot key list is used.
"""
import fcntl
import gc
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_full_reference_witnesses import ReferenceWitnessBuilder
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
DEST = BASE / 'full_task_supervision_v1'


def storage():
    if DEST.is_symlink():
        target = DEST.resolve()
        if target.parent != Path('/tmp') or not target.name.startswith('weld-full-supervision-') or not target.is_dir():
            raise ValueError('Unexpected supervision cache storage')
        return target
    if DEST.exists():
        raise ValueError('Supervision storage must be an authenticated task-owned link')
    target = Path(tempfile.mkdtemp(prefix='weld-full-supervision-', dir='/tmp'))
    DEST.symlink_to(target, target_is_directory=True)
    write(BASE / 'full_supervision_storage_binding.json', dict(project_link=str(DEST), temporary_directory=str(target),
          scope='Rebuildable geometry/query supervision only; raw data and trained weights remain persistent'))
    return target


def verify(folder, dataset, version):
    receipt = read(folder / 'receipt.json')
    if receipt['registry_sha256'] != dataset.registry_sha256 or receipt['label_receipt_sha256'] != version['label_receipt_sha256']:
        raise ValueError('Supervision registry/label binding changed')
    if receipt['consumer_sha256'] != digest(ROOT / 'weld_pipeline/server_full_shard_dataset.py') or receipt['witness_module_sha256'] != digest(ROOT / 'weld_pipeline/server_full_reference_witnesses.py'):
        raise ValueError('Supervision implementation changed; use a new declared version')
    for name, expected in receipt['files'].items():
        if digest(folder / name) != expected:
            raise ValueError('Owner-bound supervision artifact changed')
    return receipt


def main():
    out = BASE / 'supervision_preparation'
    out.mkdir(exist_ok=True)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        temporary = storage()
        dataset = FullShardDataset(ROOT)
        protocol = dict(status='ALL_QUALIFIED_TRAIN_OWNER_SUPERVISION_PREPARATION_NOT_MODEL_FIT',
                        registry_sha256=dataset.registry_sha256, training_versions=len(dataset.versions),
                        training_families=len(dataset.families), cloud_jobs=len(dataset.jobs),
                        consumer_sha256=digest(ROOT / 'weld_pipeline/server_full_shard_dataset.py'),
                        witness_module_sha256=digest(ROOT / 'weld_pipeline/server_full_reference_witnesses.py'),
                        source_sha256=digest(__file__), original_D3_module_sha256=digest(ROOT / 'weld_pipeline/contract_geometry_d3_409.py'),
                        query_feature_columns=457, fixed_input_grid_spacing_mm=1.,
                        original_parent_child_order_gaps_and_closure_retained=True,
                        process_and_frame_supervision='Complete qualified SupervisionView with explicit known/numeric/applicable/evaluation masks',
                        missing_carrier_targets='Retained in full parent reference and reported; unknown positive witness, not background',
                        same_XYZ_reference_versions='Separate complete views; no cross-view owner/parent merging',
                        physical_negative_labels=False, validation_or_test_label_reads=0, optimizer_updates=0)
        path = out / 'protocol.json'
        if path.exists():
            if read(path) != protocol:
                raise ValueError('Preparation protocol changed; preserve the old supervision version')
        else:
            write(path, protocol)
        completed = {}
        while len(completed) < len(dataset.jobs):
            progressed = False
            ready = set(dataset.ready_jobs())
            for job_key, job in dataset.jobs.items():
                if job_key in completed or job_key not in ready:
                    continue
                folder = DEST / job_key
                folder.mkdir(exist_ok=True)
                if all((folder / version['key'] / 'receipt.json').exists() for version in job['versions']):
                    completed[job_key] = [verify(folder / v['key'], dataset, v) for v in job['versions']]
                    continue
                if shutil.disk_usage(temporary).free < 5 * 1024**3:
                    raise RuntimeError('Supervision cache lacks 5GiB temporary headroom')
                print('SUPERVISION_INPUT_LOAD', job_key, len(completed), '/', len(dataset.jobs), flush=True)
                inp = dataset.input_case(job_key)
                builder = ReferenceWitnessBuilder(inp)
                receipts = []
                for version in job['versions']:
                    destination = folder / version['key']
                    if (destination / 'receipt.json').exists():
                        receipts.append(verify(destination, dataset, version))
                        continue
                    if destination.exists():
                        raise ValueError('Unexpected incomplete published supervision; refuse to overwrite')
                    supervision = dataset.supervision(version['key'])
                    def progress(parent, child, witnesses):
                        if child == 0 and parent % 16 == 0:
                            write(out / 'progress.json', dict(status='PREPARING_FULL_OWNER_SUPERVISION',
                                  controller_pid=os.getpid(), job_key=job_key, version_key=version['key'],
                                  parent=parent, parents=len(supervision.parents), current_witnesses=witnesses,
                                  completed_clouds=len(completed), total_clouds=len(dataset.jobs),
                                  completed_versions=sum(len(value) for value in completed.values()), unix_time=time.time()))
                            print('SUPERVISION_PARENT', version['key'], parent, '/', len(supervision.parents), witnesses, flush=True)
                    start = time.perf_counter()
                    with threadpool_limits(limits=2):
                        arrays, report = builder.build(supervision, progress)
                    with tempfile.TemporaryDirectory(prefix=version['key'] + '-', dir=folder) as staging:
                        result = Path(staging) / 'result'
                        result.mkdir()
                        np.savez_compressed(result / 'owner_witnesses.npz', **arrays)
                        with np.load(result / 'owner_witnesses.npz', allow_pickle=False) as saved:
                            assert all(np.array_equal(saved[key], array) for key, array in arrays.items())
                        write(result / 'coverage.json', report)
                        write(result / 'supervision_binding.json', dict(version=version,
                              complete_qualified_labels=str(ROOT / version['label_directory']),
                              input_receipt_sha256=digest(dataset.shards / job_key / 'receipt.json'),
                              compatibility_status=supervision.compatibility_status, label_geometry_as_runtime_input=False))
                        receipt = dict(status='COMPLETE_REFERENCE_WITH_PARTIAL_PROCESS_AND_OWNER_WITNESSES_READY_NOT_TRAINED',
                                       key=version['key'], job_key=job_key, role='TRAIN',
                                       registry_sha256=dataset.registry_sha256, label_receipt_sha256=version['label_receipt_sha256'],
                                       consumer_sha256=protocol['consumer_sha256'], witness_module_sha256=protocol['witness_module_sha256'],
                                       original_children=report['original_children'], represented_children=report['represented_children'],
                                       witnesses=report['witnesses'], seconds=time.perf_counter() - start,
                                       fit_calls=0, files={p.name: digest(p) for p in result.iterdir()})
                        write(result / 'receipt.json', receipt)
                        result.rename(destination)
                    receipts.append(verify(destination, dataset, version))
                    print('FULL_OWNER_SUPERVISION_READY', version['key'], report['represented_children'], '/',
                          report['original_children'], 'children', report['witnesses'], 'witnesses', flush=True)
                completed[job_key] = receipts
                del inp, builder, supervision, arrays
                gc.collect()
                progressed = True
            if not progressed and len(completed) < len(dataset.jobs):
                write(out / 'progress.json', dict(status='WAITING_FOR_REGISTERED_INPUT_SHARDS', controller_pid=os.getpid(),
                      completed_clouds=len(completed), total_clouds=len(dataset.jobs),
                      completed_versions=sum(len(value) for value in completed.values()), unix_time=time.time()))
                time.sleep(10)
        write(out / 'receipt.json', dict(status='ALL_REGISTERED_FULL_TASK_SUPERVISION_READY_NOT_MODEL_TRAINED',
              clouds=len(completed), versions=sum(len(value) for value in completed.values()),
              original_reference_children=sum(r['original_children'] for value in completed.values() for r in value),
              represented_carrier_children=sum(r['represented_children'] for value in completed.values() for r in value),
              registry_sha256=dataset.registry_sha256, protocol_sha256=digest(path), fit_calls=0))
        print('ALL_QUALIFIED_FULL_TASK_SUPERVISION_READY', len(completed), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(BASE / 'supervision_preparation/failure.json', dict(error=traceback.format_exc(), unix_time=time.time()))
        raise
