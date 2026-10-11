"""Finish a saved fit whose output record failed only at NumPy JSON encoding.

No optimizer step is repeated. The original query/pool/fit checkpoint must
still match, and the regenerated selected geometry must equal the already
saved pre-failure geometry. Other failures are deliberately not accepted.
"""
import argparse
import ast
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import shutil

import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_cached_parent_relation_prior import CachedRun446RelationPrior
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_ordered_parent_set_model import OrderedParentSetModel
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.train_full_parent_dataset import post_job_result, atomic_json, progress
from tools.train_full_predicted_parent_selector import load_job
from tools.prepare_full_predicted_parent_pool import load_gzip
from tools.record_training_experiment import record as record_experiment

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'server_rebuild/full_dataset_v1'


def main(name):
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('A local experiment name is required')
    out = BASE/name
    torch.set_num_threads(2)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failure = read(out/'failure.json')
        if 'Object of type int64 is not JSON serializable' not in failure['traceback']:
            raise ValueError('This recovery only accepts the verified output encoding failure')
        key, pass_index = failure['job_key'], failure['pass_index']
        protocol = read(out/'protocol.json')
        for source, expected in protocol['sources'].items():
            if digest(ROOT/source) != expected:
                raise ValueError('Calculation source changed; refuse to relabel a fit as the same run')
        dataset = FullShardDataset(ROOT)
        assert key in dataset.jobs and dataset.registry_sha256 == protocol['registry_sha256']
        checkpoint = out/'checkpoints'/(str(pass_index)+'_'+key+'.pt')
        checkpoint_hash = digest(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['protocol_sha256'] == digest(out/'protocol.json')
        assert saved['completed'][-1] == str(pass_index)+'/'+key
        previous = read(out/'resume.json')
        previous_state = torch.load(out/previous['checkpoint'], map_location='cpu', weights_only=False)
        assert saved['completed'][:-1] == previous_state['completed']
        record = out/'job_records'/str(pass_index)/key
        assert not (record/'receipt.json').exists()
        original_selection = load_gzip(record/'selected_parents.json.gz')
        history = []
        for line in (BASE/(name+'.log')).read_text().splitlines():
            if line.startswith('ALL_TRAIN_PARENT_FIT '):
                item = ast.literal_eval(line.removeprefix('ALL_TRAIN_PARENT_FIT '))
                if item['job_key'] == key and item['pass_index'] == pass_index:
                    history.append(item)
        assert len(history) == protocol['steps_per_job']
        assert [item['local_step'] for item in history] == list(range(1, len(history)+1))
        assert history[-1]['actual_optimizer_steps'] == saved['optimizer_steps']
        priors = CachedRun446RelationPrior(FrozenRun446Priors(ROOT))
        model = OrderedParentSetModel().to('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(saved['state_dict'])
        with threadpool_limits(limits=2):
            row = load_job(dataset, key, out, priors, reference_chunk_size=1024)
            evaluations = post_job_result(model, row, checkpoint, record)
        assert load_gzip(record/'selected_parents.json.gz') == original_selection
        assert digest(checkpoint) == checkpoint_hash
        result = dict(status='COMPLETE_REGISTERED_JOB_FIT_RECORDED_NOT_MODEL_PROMOTION',
            pass_index=pass_index, job_key=key, history=history, training_views=evaluations,
            checkpoint=str(checkpoint.relative_to(out)), checkpoint_sha256=checkpoint_hash,
            protocol_sha256=saved['protocol_sha256'], pool_receipt_sha256=digest(record/'pool_receipt.json'),
            reference_versions_processed=[v['key'] for v in row['views']], independent_evaluations=0,
            original_direct_D3_agreement=True, parent_geometry_builds=priors.parent_geometry_builds,
            genuine_relation_feature_rows=priors.relation_rows,
            recovered_output_serialization_only=True, repeated_optimizer_steps=0,
            pre_failure_selected_geometry_exactly_preserved=True,
            recovery_source_sha256=digest(__file__), JSON_writer_sha256=digest(ROOT/'weld_pipeline/sjg_local_runtime_904.py'))
        atomic_json(record/'receipt.json', result)
        record_id = 'all_parent_'+hashlib.sha256(name.encode()).hexdigest()[:12]+'_'+str(pass_index)+'_'+key
        paths = [out/'protocol.json', record/'pool_receipt.json', record/'prediction_freeze.json', record/'receipt.json']
        record_experiment(record_id, result['status'], [str(p.relative_to(ROOT)) for p in paths],
            [str(checkpoint.relative_to(ROOT)), str((out/'query_checkpoint.pt').relative_to(ROOT))],
            dict(result, protocol_source_commit=protocol['source_commit_at_initialization'],
                 per_job_output_not_final_model_accuracy=True))
        atomic_json(out/'resume.json', dict(checkpoint=result['checkpoint'], sha256=checkpoint_hash))
        (out/'failure_history').mkdir(exist_ok=True)
        archived = out/'failure_history'/(str(pass_index)+'_'+key+'_json_encoding.json')
        assert not archived.exists()
        os.replace(out/'failure.json', archived)
        pool_folder = out/'pools'/key
        storage = (out/'pools').resolve()
        assert storage.parent == Path('/tmp') and storage.name.startswith('weld-predicted-parent-pools-')
        assert pool_folder.resolve().parent == storage
        receipt = read(pool_folder/'receipt.json')
        for filename, expected in receipt['files'].items():assert digest(pool_folder/filename) == expected
        shutil.rmtree(pool_folder)
        priors.clear()
        state = {k: v for k, v in saved.items() if k not in ('state_dict', 'optimizer', 'torch_rng')}
        progress(out, state, dataset, 'OUTPUT_RECORD_RECOVERED_CHECKPOINT_READY_TO_RESUME')
        del row
        gc.collect()
        print('OUTPUT_RECORD_RECOVERED_WITHOUT_RETRAINING', key, saved['optimizer_steps'], evaluations, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    args = parser.parse_args()
    main(args.experiment_name)
