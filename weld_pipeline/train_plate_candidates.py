"""TRAIN-only structural candidate heads and full child evaluation."""
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from training.labels import sha256
from weld_pipeline.candidate_frame_tree_model import frames
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.plate_candidate_model import VERSION, predict_vertical
from weld_pipeline.point_plate_context import contact_descriptor, endpoint_descriptor
from weld_pipeline.model_registry_branch import model_from_checkpoint
from weld_pipeline.train_candidate_tree import mapped_references, supervision
from weld_pipeline.typed_contact_model import infer_probabilities


def main():
    root = Path('weld_pipeline/data/point_plate_context_v1')
    if not (root/'complete.json').exists():
        raise ValueError('Structural preprocessing incomplete')
    split_path = root/'split.json'
    split = json.loads(split_path.read_text('utf8'))
    receipt = json.loads((root/'complete.json').read_text('utf8'))
    for path, wanted in receipt['sources'].items():
        if sha256(path) != wanted:
            raise ValueError('Preprocessing source changed')
    out = Path('weld_pipeline/runs/contract_plate_candidates_seed127')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'best.pt').exists():
        raise ValueError('Existing trained model')
    arrays = {}
    for row in receipt['pieces']:
        name = row['name']
        if sha256(root/(name+'.npz')) != row['file_sha256']:
            raise ValueError('Structural cache changed')
        if sha256(Path(split['piece_roots'][name])/'manifest.json') != row['parent_manifest_sha256']:
            raise ValueError('Sequence source changed')
        a = np.load(root/(name+'.npz'), allow_pickle=False)
        arrays[name] = (a['structural'], a['process'])
    train = {n:load_piece(split['piece_roots'][n], verify=True) for n in split['train']}
    def source(name):
        return next((n for n in split['train'] if '__' not in n and (name==n or name.startswith(n+'__') or name.startswith(n+'_f'))), name)
    rows = []
    for name, piece in train.items():
        for row in supervision(piece):
            rows.append({**row, 'name': name})
    counts = Counter((source(r['name']), r['foreground']) for r in rows)
    groups = Counter(label for name, label in counts)
    xf, yf, weights, positive, xe, ye = [], [], [], [], [], []
    for row in rows:
        seq = train[row['name']]['sequences'][row['sequence']]
        structural = arrays[row['name']][0][row['sequence']]
        weight = 1./(16*counts[source(row['name']), row['foreground']]*groups[row['foreground']])
        for transformed, reverse in frames(seq):
            xf.append(np.r_[structural, contact_descriptor(transformed)])
            yf.append(row['foreground'])
            weights.append(weight)
            if row['foreground']:
                positive.append(len(xf)-1)
                xe.append(endpoint_descriptor(transformed))
                ye.append(row['margins'][::-1] if reverse else row['margins'])
    side_rows = []
    for name, piece in train.items():
        structural, process = arrays[name]
        for i, refs in mapped_references(piece).items():
            if len(refs)!=1 or 'IsSplitedForDualRobot (Any)' not in refs[0]['metadata']:
                continue
            seq = piece['sequences'][i]
            length = seq['sample_step_mm']*(len(seq['xyz'])-1)
            side_rows.append((name, int(bool(refs[0]['metadata']['IsSplitedForDualRobot (Any)'])),
                              np.r_[structural[i], process[i], length/1000., seq['type']=='vertical']))
    side_counts = Counter((source(n), y) for n,y,x in side_rows)
    side_groups = Counter(y for n,y in side_counts)
    sw = [1./(side_counts[source(n),y]*side_groups[y]) for n,y,x in side_rows]
    fractions = {'1': [0., 1.]}
    for count in sorted({r['segment_count'] for r in rows if r['foreground']}):
        values = [r['fractions'] for r in rows if r['foreground'] and r['segment_count']==count]
        fractions[str(count)] = np.median(np.array(values), axis=0).tolist()
    for r in rows:
        if r['foreground'] and r['segment_count'] != (3 if r['split_flag'] else 1):
            raise ValueError('Unsupported TRAIN process topology')
    paths = [__file__, 'weld_pipeline/plate_candidate_model.py', 'weld_pipeline/point_plate_context.py',
             'weld_pipeline/candidate_frame_tree_model.py', 'weld_pipeline/candidate_tree_model.py',
             'weld_pipeline/sequence_augmentation.py', 'weld_pipeline/vertical_frame_augmentation.py',
             'weld_pipeline/train_candidate_tree.py', 'weld_pipeline/contract_child_evaluation.py']
    config = {'version': VERSION, 'seed': 127, 'trees': 512, 'split_sha256': sha256(split_path),
              'sources': {p:sha256(p) for p in paths}, 'train': split['train'], 'validation':split['validation'],
              'threshold': .5, 'split_fractions': fractions,
              'foreground_inputs': 'Center point-distance profiles plus plate support/parallel-face distances; no identity/world position.',
              'side_inputs': 'Point-derived plate height/support and physical extents, candidate length and gravity relation.',
              'end_inputs': 'Point-distance profiles at fixed offsets from point-derived candidate ends.',
              'supervision': 'TRAIN only, source/class mass balanced, unknown regions excluded; 16 equivalent V frames.',
              'topology_scope': 'Parent Any to 1/3 correspondence verified on TRAIN, still not a confirmed universal process rule.',
              'full_contract_pass':False}
    (out/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
    archive = out/'source_snapshot'
    archive.mkdir(exist_ok=True)
    for path in paths:
        shutil.copy2(path, archive/Path(path).name)
    start = time.perf_counter()
    foreground = ExtraTreesClassifier(n_estimators=512, max_features=.8, n_jobs=4, random_state=127)
    endpoints = ExtraTreesRegressor(n_estimators=512, max_features=.5, n_jobs=4, random_state=127)
    side = ExtraTreesClassifier(n_estimators=512, max_features=1., n_jobs=4, random_state=127)
    foreground.fit(np.stack(xf), np.array(yf), sample_weight=np.array(weights))
    endpoints.fit(np.stack(xe), np.array(ye), sample_weight=np.array(weights)[positive])
    side.fit(np.stack([x for n,y,x in side_rows]), np.array([y for n,y,x in side_rows]), sample_weight=np.array(sw))
    bundle = {'foreground':foreground, 'endpoints':endpoints, 'side_split':side, 'split_fractions':fractions}
    print('TRAINED_127', len(xf), len(xe), len(side_rows), round(time.perf_counter()-start, 2), flush=True)
    torch.set_num_threads(4)
    baseline_path = Path('weld_pipeline/runs/typed_branch_surface_v3_seed115/best.pt')
    baseline = torch.load(baseline_path, map_location='cpu')
    verify_cached_inference_sources(baseline, baseline_path)
    # One reviewable checkpoint contains both the retained neural branch and
    # all learned candidate heads, avoiding a runtime baseline-file lookup.
    state = {'model_name': VERSION, 'baseline':baseline, 'plate_heads':bundle, 'config':config,
             'source_hashes':config['sources'], 'point_context_sources':receipt['sources']}
    torch.save(state, out/'best.pt')
    model = model_from_checkpoint(baseline).cuda().eval()
    for role,names in [('TRAIN',[n for n in split['train'] if '__' not in n]), ('DEV',split['validation'])]:
        scores, vertical_scores = {}, {}
        for name in names:
            piece = train[name] if name in train else load_piece(split['piece_roots'][name], verify=True)
            v, diagnostic = predict_vertical(bundle, piece['sequences'], *arrays[name])
            h = [p for p in decode_intervals(piece['sequences'], infer_probabilities(model,piece['sequences']),
                 threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.) if p['type']!='vertical']
            pred = h+v
            score = score_children(pred, piece['truth'])
            vertical_score = score_children(v, [w for w in piece['truth'] if w['type']=='vertical'])
            scores[name], vertical_scores[name] = score,vertical_score
            (out/(role+'_'+name+'.json')).write_text(json.dumps({'predictions':pred, 'vertical_diagnostic':diagnostic,
                'score':score, 'vertical_score':vertical_score}, ensure_ascii=False, indent=2), encoding='utf8')
            print(role,name,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],
                  'V',vertical_score['valid_children'],'/',vertical_score['truth_count'],flush=True)
        (out/(role+'_summary.json')).write_text(json.dumps(aggregate(scores), indent=2), encoding='utf8')
        (out/(role+'_vertical_summary.json')).write_text(json.dumps(aggregate(vertical_scores), indent=2), encoding='utf8')
    (out/'complete.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-start,
        'full_contract_pass':False, 'baseline_replaced':False, 'checkpoint_sha256':sha256(out/'best.pt')}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
