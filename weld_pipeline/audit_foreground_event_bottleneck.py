"""Fixed FIT-only diagnostic to separate foreground boundaries from learned endpoint events."""
import json
import time
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.foreground_component_decoder import decode_foreground

OUT = Path('weld_pipeline/reports/foreground_event_bottleneck_155_20261001')
RUNS = {151:Path('weld_pipeline/runs/joint_event_vote_capacity_seed151'),
        154:Path('weld_pipeline/runs/joint_workpiece_event_capacity_seed154')}
BASE = Path('weld_pipeline/data/joint_relation_v1')


def main():
    if OUT.exists():
        raise ValueError('Preserve foreground/event diagnosis')
    cfg = json.loads((RUNS[154]/'config.json').read_text('utf8')); names = cfg['train']
    baseline = json.loads((RUNS[151]/'config.json').read_text('utf8'))
    if baseline['train'] != names:
        raise ValueError('Matched training sources required')
    OUT.mkdir(parents=True)
    models = {str(n):sha256(p/'final.pt') for n, p in RUNS.items()}
    protocol = dict(train=names, threshold=.5, minimum_length_filter=None, gap_closing=False,
        representations=['correct_original_FG', 'correct_original_FG_and_cuts',
                         'fixed151_FG', 'fixed151_FG_and_cuts', 'fixed154_FG', 'fixed154_FG_and_cuts'],
        frozen_model_sha256=models, original_labels_changed=False, no_model_training=True,
        no_held_or_observed_read=True, sources={p:sha256(p) for p in [__file__,
            'weld_pipeline/foreground_component_decoder.py', 'weld_pipeline/contract_child_evaluation.py']},
        adoption='Diagnostic only; no decoder/model default replacement or contract acceptance.', full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf8')
    results = {r:{} for r in protocol['representations']}; started = time.perf_counter()
    for name in names:
        manifest = json.loads((BASE/name/'manifest.json').read_text('utf8'))
        piece = load_piece(manifest['source_cache_root'], verify=True)
        sequences = piece['sequences']
        truth = [w for w in piece['truth'] if w['type'] == 'horizontal'
                 and all(s['kind'] == 'line' for s in w['segments'])]
        original = [np.where(target>=0, target, 0.) if s['type'] == 'horizontal'
                    and s.get('curve_kind') != 'arc' else np.zeros_like(target)
                    for s, target in zip(sequences, piece['targets'])]
        variants = {'correct_original':original}
        for n, root in RUNS.items():
            with np.load(root/('FIT_'+name+'_heatmaps.npz'), allow_pickle=False) as cache:
                lengths = cache['lengths']; values = cache['probabilities']
                if lengths.tolist() != [len(s['xyz']) for s in sequences]:
                    raise ValueError('Frozen FIT heatmap query order differs')
                variants['fixed'+str(n)] = np.split(values, np.cumsum(lengths)[:-1])
        for variant, probabilities in variants.items():
            for cuts in (False, True):
                key = variant+'_FG'+('_and_cuts' if cuts else '')
                predictions = decode_foreground(sequences, probabilities, use_cuts=cuts)
                score = score_children(predictions, truth); results[key][name] = score
                (OUT/(key+'_'+name+'.json')).write_text(json.dumps(dict(score=score, predictions=predictions),
                    indent=2, ensure_ascii=False), encoding='utf8')
                print('FOREGROUND_EVENT_DIAGNOSTIC', key, name, score['valid_children'],
                      '/', score['truth_count'], 'FP', score['false_predictions'], flush=True)
    aggregate_results = {r:aggregate(scores) for r, scores in results.items()}
    if any(result['truth_count'] != 1221 for result in aggregate_results.values()):
        raise ValueError('Full 1221 child denominator required')
    if any(sha256(RUNS[int(n)]/'final.pt') != h for n, h in models.items()):
        raise ValueError('Frozen diagnostic model changed')
    complete = dict(config_sha256=sha256(OUT/'config.json'), results=aggregate_results,
        elapsed_seconds=time.perf_counter()-started, diagnostic_only=True,
        no_model_or_decoder_adopted=True, held_or_observed_read=False, full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete, indent=2, ensure_ascii=False), encoding='utf8')
    print('FOREGROUND_EVENT_DIAGNOSTIC_COMPLETE', {r:(s['valid_children'], s['false_predictions'])
                                                for r, s in aggregate_results.items()}, flush=True)


if __name__ == '__main__':
    main()
