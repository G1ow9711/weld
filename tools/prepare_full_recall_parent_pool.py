"""Freeze an explicitly bound recall query model for complete input prediction.

The shared preparation implementation visits all finite profiles and freezes
all candidates before references. This entry changes only query initialization
and architecture loading, leaving running historical controllers untouched.
"""
import argparse
import os
from pathlib import Path
import shutil
import tempfile

import torch

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_query_checkpoint_model import load_query_model
from weld_pipeline.server_reference_recall_weld_model import ARCHITECTURE
from weld_pipeline.server_runtime_aligned_training_samples import FIXED_QUERY_FRACTIONS
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from tools.prepare_full_predicted_parent_pool import prepare

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def initialize(name, query_source, query_sha256):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('A local independent experiment name is required')
    query_source = Path(query_source).resolve()
    query_source.relative_to(ROOT)
    if not query_sha256 or digest(query_source) != query_sha256:
        raise ValueError('Explicit persistent query checkpoint fingerprint required')
    out = BASE / name
    out.mkdir(exist_ok=True)
    checkpoint = out / 'query_checkpoint.pt'
    if not checkpoint.exists():
        stage = out / 'query_checkpoint.incomplete'
        shutil.copyfile(query_source, stage)
        if digest(stage) != query_sha256:
            raise ValueError('Query snapshot changed while copying')
        state = torch.load(stage, map_location='cpu', weights_only=False)
        _, architecture = load_query_model(state)
        if architecture != ARCHITECTURE:
            raise ValueError('This independent recipe requires the recall query model')
        os.replace(stage, checkpoint)
    if digest(checkpoint) != query_sha256:
        raise ValueError('Existing frozen query checkpoint differs')
    storage_link = out / 'pools'
    if not storage_link.exists():
        storage = Path(tempfile.mkdtemp(prefix='weld-predicted-parent-pools-', dir='/tmp'))
        storage_link.symlink_to(storage, target_is_directory=True)
        write(out / 'storage_binding.json', dict(project_link=str(storage_link),
            temporary_directory=str(storage), scope='Own rebuildable complete prediction pool only'))
    storage = storage_link.resolve()
    if storage.parent != Path('/tmp') or not storage.name.startswith('weld-predicted-parent-pools-'):
        raise ValueError('Unexpected prediction cache ownership')
    return out, storage, checkpoint


def main(args):
    torch.set_num_threads(2)
    out, _, checkpoint = initialize(args.experiment_name, args.query_checkpoint, args.query_sha256)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model, architecture = load_query_model(state)
    dataset = FullShardDataset(ROOT)
    priors = FrozenRun446Priors(ROOT)
    names = ['tools/prepare_full_recall_parent_pool.py', 'tools/prepare_full_predicted_parent_pool.py',
             'weld_pipeline/server_query_checkpoint_model.py', 'weld_pipeline/server_reference_recall_weld_model.py',
             'weld_pipeline/server_full_prediction_runtime.py', 'weld_pipeline/server_predicted_parent_pool.py']
    protocol = dict(query_checkpoint_sha256=args.query_sha256, query_architecture=architecture,
        query_optimizer_steps=state['optimizer_steps'], registry_sha256=dataset.registry_sha256,
        original_run446_prior_sha256=priors.checkpoint_sha256, query_mode='fixed_and_grid',
        fixed_query_fractions=list(FIXED_QUERY_FRACTIONS), competition_mode='lazy', resource_mode='input_scaled_v1',
        sources={name:digest(ROOT / name) for name in names}, reference_geometry_reads=0,
        control_not_full_registered_train_fit=True, independent_model_accuracy_not_claimed=True)
    if (out / 'protocol.json').exists():
        if read(out / 'protocol.json') != protocol:
            raise ValueError('Prediction control changed; use a new experiment')
    else:
        write(out / 'protocol.json', protocol)
    prepare(dataset, args.job, model, state, priors, out, 'fixed_and_grid',
            FIXED_QUERY_FRACTIONS, 'lazy', 'input_scaled_v1')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--query-checkpoint', required=True)
    parser.add_argument('--query-sha256', required=True)
    parser.add_argument('--job', required=True)
    main(parser.parse_args())
