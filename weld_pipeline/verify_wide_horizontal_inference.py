"""Compare real horizontal and arc probabilities against the frozen baseline."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.wide_context_cache import load_piece
from weld_pipeline.curved_cache import load_piece as load_base
from weld_pipeline.model_registry_wide_branch import model_from_checkpoint
from weld_pipeline.typed_contact_model import infer_probabilities


def main():
    ap = argparse.ArgumentParser()
    for name in ['checkpoint', 'baseline', 'data', 'outdir']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--groups', nargs='+', required=True)
    args = ap.parse_args()
    out, root = Path(args.outdir), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / 'checkpoint_used.pt'
    if snapshot.exists():
        raise ValueError('Verification snapshot already exists; use a new directory')
    blob = Path(args.checkpoint).read_bytes()
    snapshot.write_bytes(blob)
    state = torch.load(io.BytesIO(blob), map_location='cpu')
    checkpoint_hash = hashlib.sha256(blob).hexdigest()
    del blob
    with Path(args.baseline).open('rb') as stream:
        baseline_state = torch.load(stream, map_location='cpu')
    verify_cached_inference_sources(state, args.checkpoint)
    verify_cached_inference_sources(baseline_state, args.baseline)
    if state.get('model_name') != 'wide_branch_surface' or state['horizontal_initial_sha256'] != sha256(args.baseline):
        raise ValueError('Mismatched branch baseline')
    for name, value in baseline_state['model'].items():
        if not torch.equal(value, state['model']['horizontal.' + name]):
            raise ValueError('Horizontal parameter differs: ' + name)
    torch.set_num_threads(4)
    model = model_from_checkpoint(state).cuda().eval()
    baseline = model_from_checkpoint(baseline_state).cuda().eval()
    split = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    rows = []
    for group in args.groups:
        folder = (root / split.get('piece_roots', {}).get(group, group)).resolve()
        digest = sha256(folder / 'manifest.json')
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if state['data_hashes'].get(group)!=digest or baseline_state['data_hashes'].get(group)!=manifest['base_manifest_sha256']:
            raise ValueError('Reference cache changed or absent from baseline')
        piece = load_piece(folder)
        base=load_base((folder/manifest['base_folder']).resolve())
        reference = infer_probabilities(baseline, base['sequences'])
        actual = infer_probabilities(model, piece['sequences'])
        selected = [i for i, seq in enumerate(piece['sequences']) if seq['type'] == 'horizontal']
        differences = [float(np.max(np.abs(actual[i] - reference[i]))) for i in selected]
        exact = all(np.array_equal(actual[i], reference[i]) for i in selected)
        row = {'group': group, 'horizontal_sequences': len(selected),
               'horizontal_points': sum(len(actual[i]) for i in selected),
               'arc_sequences': sum(piece['sequences'][i].get('curve_kind') == 'arc' for i in selected),
               'all_seven_probabilities_exact': exact, 'max_absolute_difference': max(differences, default=0.),
               'cache_sha256': digest}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del piece, base, actual, reference
    report = {'status': 'PASS' if all(r['all_seven_probabilities_exact'] for r in rows) else 'FAIL',
              'scope': 'Horizontal/arc inference equivalence only, not vertical accuracy or overall acceptance.',
              'checkpoint_epoch': state['epoch'], 'checkpoint_sha256': checkpoint_hash,
              'baseline_sha256': sha256(args.baseline), 'horizontal_parameters_exact': True, 'groups': rows,
              'sources': {str(Path(__file__)): sha256(__file__), **state['source_hashes']}}
    (out / 'receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if report['status'] != 'PASS':
        raise RuntimeError('Horizontal probability equivalence failed')


if __name__ == '__main__':
    main()
