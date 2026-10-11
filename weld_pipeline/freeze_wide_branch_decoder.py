"""Reuse predeclared TRAIN decoder parameters without tuning the branch trial."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from training.labels import sha256
from weld_pipeline.interval_config import load_interval_config
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources


def main():
    ap = argparse.ArgumentParser()
    for name in ['checkpoint', 'prior-checkpoint', 'prior-config', 'outdir']:
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    if (out / 'decoder_config.json').exists():
        raise ValueError('Frozen configuration exists; use a new directory')
    parameters, old_config = load_interval_config(args.prior_config, args.prior_checkpoint)
    with Path(args.checkpoint).open('rb') as stream:
        state = torch.load(stream, map_location='cpu')
    verify_cached_inference_sources(state, args.checkpoint)
    if state.get('model_name') != 'wide_branch_surface' or state['horizontal_initial_sha256'] != sha256(args.prior_checkpoint):
        raise ValueError('Checkpoint not initialized from this frozen horizontal model')
    declared = json.loads((Path(args.checkpoint).parent / 'config.json').read_text(encoding='utf-8'))
    if declared['fixed_decoder_parameters'] != parameters:
        raise ValueError('Parameters differ from predeclared training configuration')
    with Path(args.prior_checkpoint).open('rb') as stream:
        previous = torch.load(stream, map_location='cpu')
    for name, tensor in previous['model'].items():
        if not torch.equal(tensor, state['model']['horizontal.' + name]):
            raise ValueError('Frozen horizontal state changed: ' + name)
    sources = {str(Path('weld_pipeline') / name): sha256(Path('weld_pipeline') / name)
               for name in ['event_gap_decoder.py', 'constrained_decoder.py', 'curved_sequences.py']}
    params = {**parameters, 'event_gap_mm': 0.}
    receipt = {'scope': 'Fixed prior original-TRAIN parameters; no branch-trial decoder search and no validation labels loaded.',
               'checkpoint_sha256': sha256(args.checkpoint), 'prior_checkpoint_sha256': sha256(args.prior_checkpoint),
               'prior_config_sha256': sha256(args.prior_config), 'prior_config': old_config,
               'declared_config_sha256': sha256(Path(args.checkpoint).parent / 'config.json'),
               'horizontal_state_exactly_unchanged': True, 'parameters': params,
               'sources': {**sources, str(Path(__file__)): sha256(__file__),
                           'weld_pipeline/interval_config.py': sha256('weld_pipeline/interval_config.py')}}
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / 'frozen_parameter_receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    config = {'version': 'shared_event_gap_decoder_v1', 'checkpoint_sha256': sha256(args.checkpoint),
              'parameters': params, 'sources': sources, 'frozen_parameter_receipt_sha256': sha256(receipt_path)}
    (out / 'decoder_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    snapshots = out / 'source_snapshot'
    snapshots.mkdir(exist_ok=True)
    for source in receipt['sources']:
        shutil.copy2(source, snapshots / Path(source).name)
    print(json.dumps({'status': 'FROZEN_PRIOR_TRAIN_PARAMETERS', 'parameters': params,
                      'horizontal_state_exactly_unchanged': True}), flush=True)


if __name__ == '__main__':
    main()
