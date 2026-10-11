"""Replay every registered TRAIN input with a retained complete native output.

This cohort is selected from actual retained artifacts, not an easy pilot list.
Every registered label version for each retained input is scored separately.
Missing outputs and control failures remain explicit; they are never averaged
away or reported as full-dataset fitting or independent model evaluation.
"""
import fcntl
import os
from pathlib import Path
import time
import traceback

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.verify_complete_native_parent_runtime import main as verify_job, SOURCE_NAMES
from tools.record_training_experiment import record as record_experiment

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
OUT = BASE / 'full_native_retained_inputs_regression_v1'
OLD_CONTROL = 'full_native_parent_runtime_control_v1'


def atomic_json(path, value):
    stage = path.with_name(path.name + '.incomplete')
    write(stage, value)
    os.replace(stage, path)


def main():
    dataset = FullShardDataset(ROOT)
    OUT.mkdir(exist_ok=True)
    with (OUT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        required_files = ['decoded.json', 'unary.npy', 'parent_metadata.json',
                          'final_decoded.json', 'final_prediction.json', 'precision_scores.npz']
        jobs = []
        for key, job in dataset.jobs.items():
            provider = job['prior_input_provider']
            if not provider or not provider['observed_features_complete']:
                continue
            folder = ROOT / provider['runtime_directory'] / 'run379' / ('test_' + job['runtime_key'])
            if not all((folder / name).is_file() for name in required_files):
                raise ValueError('A retained complete input lacks its native comparison output')
            jobs.append(key)
        # Cover line, arc and original large-piece controls early, followed by
        # every remaining retained input. All eligible cases stay in the cohort.
        priorities = {'BM79A_native_local': 0, '21': 1, '25': 2, '36': 3, '10': 4}
        jobs.sort(key=lambda key: (priorities.get(dataset.jobs[key]['runtime_key'], 5), key))
        frozen = read(BASE / OLD_CONTROL / 'protocol.json')
        source_names = SOURCE_NAMES + ['tools/verify_all_retained_native_parents.py']
        sources = {name: digest(ROOT / name) for name in source_names}
        protocol = dict(status='EVERY_RETAINED_ACTUAL_TRAIN_NATIVE_OUTPUT_REPLAY_PROTOCOL',
            registry_sha256=dataset.registry_sha256, jobs=jobs,
            versions=sorted(v['key'] for key in jobs for v in dataset.jobs[key]['versions']),
            native_upstream_checkpoint=frozen['native_upstream_checkpoint'],
            native_selector_bindings=frozen['native_selector_bindings'], sources=sources,
            selection_basis='Every complete retained input in the qualified TRAIN registry',
            original_run446_incremental_branch_executed=False,
            optimizer_updates=0, independent_model_evaluations=0, full_contract_pass=False)
        if (OUT / 'protocol.json').exists():
            if read(OUT / 'protocol.json') != protocol:
                raise ValueError('Complete native regression recipe changed')
        else:
            atomic_json(OUT / 'protocol.json', protocol)
        results, failures = [], []
        for key in jobs:
            if sources != {name: digest(ROOT / name) for name in source_names}:
                raise ValueError('Bound complete native regression source changed')
            name = (OLD_CONTROL if key == frozen['job_key']
                    else 'full_native_retained_' + key.removeprefix('cloud_') + '_v1')
            folder = BASE / name
            atomic_json(OUT / 'progress.json', dict(status='ACTUAL_COMPLETE_NATIVE_REPLAY',
                controller_pid=os.getpid(), job_key=key, completed_jobs=len(results), failed_jobs=len(failures),
                required_jobs=len(jobs), reference_versions_completed=sum(r['complete_reference_versions_evaluated'] for r in results),
                required_versions=len(protocol['versions']), optimizer_updates=0, unix_time=time.time()))
            try:
                if not (folder / 'receipt.json').exists():
                    if folder.exists():
                        raise ValueError('Incomplete native replay exists; diagnose before repeating it')
                    verify_job(key, name)
                receipt, p = read(folder / 'receipt.json'), read(folder / 'protocol.json')
                if (receipt['status'] != 'PASS_ACTUAL_COMPLETE_NATIVE_BRANCH_PARITY_NOT_COMPLETE_RUN446'
                        or p['native_upstream_checkpoint'] != protocol['native_upstream_checkpoint']
                        or p['native_selector_bindings'] != protocol['native_selector_bindings']
                        or p['registry_sha256'] != dataset.registry_sha256
                        or p['sources'] != {name: sources[name] for name in SOURCE_NAMES}
                        or receipt['complete_reference_versions_evaluated'] != len(dataset.jobs[key]['versions'])):
                    raise ValueError('Native replay is not complete under the fixed model and input protocol')
                results.append(receipt)
                record_id = 'native_full_' + digest(OUT / 'protocol.json')[:12] + '_' + key
                if not (ROOT / 'experiment_records' / (record_id + '.json')).exists():
                    selector_root = ROOT / 'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run379'
                    record_experiment(record_id, 'ACTUAL_COMPLETE_NATIVE_REPLAY_TRAIN_DIAGNOSTIC_NOT_FULL_RUN446',
                        [str(p.relative_to(ROOT)) for p in [OUT / 'protocol.json', folder / 'protocol.json',
                            folder / 'prediction_freeze.json', folder / 'native_graph_evidence.json', folder / 'receipt.json']],
                        [protocol['native_upstream_checkpoint']['path']] +
                        [str((selector_root / name).relative_to(ROOT)) for name in protocol['native_selector_bindings']],
                        dict(job_key=key, complete_native_geometry_parity=True,
                            incremental_run446_branch_executed=False, independent_model_evaluations=0, optimizer_updates=0))
                print('EVERY_RETAINED_NATIVE_JOB_COMPLETE', key, len(results), '/', len(jobs), flush=True)
            except Exception:
                failures.append(dict(job_key=key, experiment_name=name, traceback=traceback.format_exc()))
                print('EVERY_RETAINED_NATIVE_JOB_FAILED', key, failures[-1]['traceback'], flush=True)
        atomic_json(OUT / 'summary.json', dict(status=('ALL_RETAINED_NATIVE_CONTROLS_COMPLETE_NOT_FULL_RUN446'
            if not failures and len(results) == len(jobs) else 'INCOMPLETE_NATIVE_CONTROLS_FAILURES_RETAINED'),
            protocol_sha256=digest(OUT / 'protocol.json'), completed_jobs=len(results), required_jobs=len(jobs),
            completed_versions=sum(r['complete_reference_versions_evaluated'] for r in results),
            required_versions=len(protocol['versions']), failures=failures,
            results=[dict(job_key=r['job_key'], selected_children=r['actual_output_children'],
                versions=[dict(version_key=v['version_key'], overall=v['score']['categories']['overall'])
                          for v in r['evaluations']]) for r in results],
            optimizer_updates=0, independent_model_evaluations=0,
            all_registered_train_fitting_not_claimed=True, incremental_run446_branch_executed=False,
            complete_run446_or_contract_pass=False))


if __name__ == '__main__':
    main()
