"""Preregistered paired business classifiers; source cross-fit is development evidence."""
import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from training.labels import sha256
from weld_pipeline.train_material_support_crossfit import balanced_weights, score_probability
from weld_pipeline.workpiece_relative_context import frame_views, predict_equivalent_views

BASE = Path('weld_pipeline/data/material_support_v1')
DATA = Path('weld_pipeline/data/workpiece_context_v1')
OUT = Path('weld_pipeline/runs/workpiece_context_crossfit_seed153')
SEEDS = (153, 154, 155)
REPRESENTATIONS = ('baseline', 'gravity', 'whole')


def fit_head(x, y, weights, representation, seed):
    rows = np.flatnonzero(weights > 0)
    augmented = np.concatenate(frame_views(x[rows], representation))
    labels = np.tile(y[rows], 16); w = np.tile(weights[rows]/16., 16)
    model = ExtraTreesClassifier(n_estimators=128, max_features=.5,
        min_samples_leaf=2, max_depth=None, n_jobs=2, random_state=seed)
    model.fit(augmented, labels, sample_weight=w)
    del augmented, labels, w; gc.collect()
    return model


def main():
    if OUT.exists():
        raise ValueError('Preserve fixed whole-workpiece classifier trial')
    started = time.perf_counter()
    cfg = json.loads((DATA/'config.json').read_text('utf8'))
    complete = json.loads((DATA/'complete.json').read_text('utf8'))
    raw_path = Path('weld_pipeline/reports/workpiece_context_raw_153_20261001/complete.json')
    raw = json.loads(raw_path.read_text('utf8'))
    collision_path = Path('weld_pipeline/reports/workpiece_context_collisions_153_20261001.json')
    collision = json.loads(collision_path.read_text('utf8'))
    if (sha256(DATA/'config.json') != complete['config_sha256'] or not raw['all_checks_passed']
            or collision['results']['whole']['rounded_6_decimals']['conflicting_groups'] != 0):
        raise ValueError('Current feature consistency/distinction proof required')
    for sources in (cfg['sources'], raw['source_sha256']):
        for path, digest in sources.items():
            if sha256(path) != digest:
                raise ValueError('Relative context feature source changed')
    names = cfg['pieces']; locals_ = []; contexts = []; ys = []; source_rows = {}; row = 0
    for name in names:
        base = json.loads((BASE/name/'manifest.json').read_text('utf8'))
        manifest = json.loads((DATA/name/'manifest.json').read_text('utf8'))
        if (sha256(DATA/name/'manifest.json') != complete['manifests'][name]
                or sha256(BASE/name/'manifest.json') != manifest['base_manifest_sha256']):
            raise ValueError('Fixed sample source manifest changed')
        for root, files in ((BASE/name, base['files']), (DATA/name, manifest['files'])):
            for file, digest in files.items():
                if sha256(root/file) != digest:
                    raise ValueError('Fixed sample array changed')
        local = np.load(BASE/name/'signed.npy', allow_pickle=False)
        context = np.load(DATA/name/'context.npy', allow_pickle=False)
        with np.load(BASE/name/'labels.npz', allow_pickle=False) as target:
            ys.append(target['business'])
        locals_.append(local); contexts.append(context)
        source_rows[name] = (row, row+len(local)); row += len(local)
    local = np.concatenate(locals_); context = np.concatenate(contexts); y = np.concatenate(ys)
    x = {'baseline':local, 'gravity':np.c_[local, context[:, 19:22]], 'whole':np.c_[local, context]}
    OUT.mkdir(parents=True)
    sources = [__file__, 'weld_pipeline/workpiece_relative_context.py',
               'weld_pipeline/train_material_support_crossfit.py', 'weld_pipeline/signed_material_support.py']
    config = dict(seeds=list(SEEDS), representations=list(REPRESENTATIONS), pieces=names,
        source_rows=source_rows, folds=cfg['folds'], n_estimators=128, max_features=.5,
        min_samples_leaf=2, max_depth=None, n_jobs=2, equivalent_frames=16,
        source_and_class_balanced=True, foreground_threshold=.5,
        target='Unchanged original welding foreground; no STEP label used by classifier',
        data_config_sha256=sha256(DATA/'config.json'), data_complete_sha256=sha256(DATA/'complete.json'),
        raw_verification_sha256=sha256(raw_path), collision_audit_sha256=sha256(collision_path),
        sources={p:sha256(p) for p in sources},
        adoption='Every seed: FIT and source cross-fit TP/G >95%, FP/G <=5%; compared with paired baseline FN+FP reduction >=20%, majority sources improve, TP no lower, FP no higher.',
        repeated_source_crossfit_is_development_not_blind=True,
        geometry_children_not_scored=True, observed_check_read=False, full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    snapshot = OUT/'source_snapshot'; snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot/Path(source).name)
    capacity = {}; checkpoints = {}; fold_scores = {}
    oof = {str(s):{r:np.full((row, 2), np.nan) for r in REPRESENTATIONS} for s in SEEDS}
    for seed in SEEDS:
        capacity[str(seed)] = {}
        for representation in REPRESENTATIONS:
            weights = balanced_weights(y, names, source_rows)
            model = fit_head(x[representation], y, weights, representation, seed)
            path = OUT/f'full_fit_seed{seed}_{representation}.joblib'; joblib.dump(model, path)
            checkpoints[path.name] = sha256(path)
            p = predict_equivalent_views(model, x[representation], representation)
            score = score_probability(p, y, model.classes_); capacity[str(seed)][representation] = score
            np.save(OUT/(path.stem+'_probabilities.npy'), p.astype(np.float32))
            (OUT/(path.stem+'.json')).write_text(json.dumps(dict(score=score,
                checkpoint_sha256=checkpoints[path.name], elapsed_seconds=time.perf_counter()-started), indent=2), encoding='utf8')
            print('WORKPIECE_CONTEXT_FIT', seed, representation, 'TP', score['true_foreground'],
                  '/', score['foreground'], 'FP', score['false_foreground'], flush=True)
            del model; gc.collect()
    (OUT/'capacity.json').write_text(json.dumps(capacity, indent=2), encoding='utf8')
    for fold in cfg['folds']:
        fi = fold['fold']; train = [n for n in fold['train_real'] if n in source_rows]
        held = [n for n in fold['held_out_real'] if n in source_rows]
        if set(train)&set(held) or set(train+held) != set(names):
            raise ValueError('Source partition invalid')
        held_indices = np.concatenate([np.arange(*source_rows[n]) for n in held])
        for seed in SEEDS:
            for representation in REPRESENTATIONS:
                weights = balanced_weights(y, train, source_rows)
                model = fit_head(x[representation], y, weights, representation, seed)
                if list(model.classes_) != [0, 1]:
                    raise ValueError('Training fold lacks foreground or background')
                path = OUT/f'fold{fi}_seed{seed}_{representation}.joblib'; joblib.dump(model, path)
                checkpoints[path.name] = sha256(path)
                p = predict_equivalent_views(model, x[representation][held_indices], representation)
                oof[str(seed)][representation][held_indices] = p
                score = score_probability(p, y[held_indices], model.classes_)
                fold_scores[path.stem] = dict(train=train, held=held, score=score,
                    training_rows=int((weights>0).sum()), checkpoint_sha256=checkpoints[path.name])
                (OUT/(path.stem+'.json')).write_text(json.dumps(fold_scores[path.stem], indent=2, ensure_ascii=False), encoding='utf8')
                print('WORKPIECE_CONTEXT_SOURCE', fi, seed, representation, 'TP', score['true_foreground'],
                      '/', score['foreground'], 'FP', score['false_foreground'], flush=True)
                del model; gc.collect()
    aggregates = {}; decisions = {}; by_source = {}
    for seed in SEEDS:
        ss = str(seed); aggregates[ss] = {}; decisions[ss] = {}; by_source[ss] = {}
        for representation in REPRESENTATIONS:
            p = oof[ss][representation]
            if not np.isfinite(p).all():
                raise ValueError('Incomplete source cross-fit prediction')
            aggregates[ss][representation] = score_probability(p, y, [0, 1])
            np.save(OUT/f'oof_seed{seed}_{representation}.npy', p.astype(np.float32))
            by_source[ss][representation] = {n:score_probability(p[a:b], y[a:b], [0, 1])
                for n, (a, b) in source_rows.items()}
        baseline = aggregates[ss]['baseline']
        for representation in ('gravity', 'whole'):
            score = aggregates[ss][representation]
            reduction = (baseline['errors']-score['errors'])/max(baseline['errors'], 1)
            improvements = sum(by_source[ss][representation][n]['errors'] < by_source[ss]['baseline'][n]['errors'] for n in names)
            fit_gate = capacity[ss][representation]['foreground_sample_gate']
            source_gate = score['foreground_sample_gate']
            passed = bool(fit_gate and source_gate and reduction >= .2 and improvements > len(names)/2
                and score['true_foreground'] >= baseline['true_foreground']
                and score['false_foreground'] <= baseline['false_foreground'])
            decisions[ss][representation] = dict(fit_sample_gate=fit_gate, source_sample_gate=source_gate,
                error_reduction=reduction, improved_sources=improvements, total_sources=len(names), adopted=passed)
        print('WORKPIECE_CONTEXT_DECISION', seed, decisions[ss], flush=True)
    (OUT/'by_source.json').write_text(json.dumps(by_source, indent=2, ensure_ascii=False), encoding='utf8')
    eligible = {r:all(decisions[str(s)][r]['adopted'] for s in SEEDS) for r in ('gravity', 'whole')}
    result = dict(config_sha256=sha256(OUT/'config.json'), capacity=capacity,
        source_crossfit=aggregates, source_decisions=decisions, eligible_for_future_geometry_trial=eligible,
        fold_scores=fold_scores, checkpoint_sha256=checkpoints, trained_models=len(checkpoints),
        elapsed_seconds=time.perf_counter()-started, repeated_source_crossfit_is_development_not_blind=True,
        geometry_children_not_scored=True, observed_check_read=False, full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    print('WORKPIECE_CONTEXT_COMPLETE', len(checkpoints), 'ELIGIBLE', eligible, flush=True)


if __name__ == '__main__':
    main()
