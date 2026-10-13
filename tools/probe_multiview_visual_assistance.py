"""Family-disjoint TRAIN cross-fit of point-cloud image assistance.

This bounded priority probe compares one view with three views. It predicts
original-point weld proximity, not complete parents, attributes or CSVs.
Official validation and TEST remain closed. Sampling is limited to fitting;
inference and reporting use every point and every child in selected jobs.
"""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from tools.export_full_dataset_split_view import local_name
from tools.qualify_full_dataset_labels import ply_points
from tools.record_training_experiment import git_head
from weld_pipeline.server_multiview_visual_features import visual_features
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.contract_curve_geometry import sample_arc
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, read, write

BASE = ROOT / 'server_rebuild/full_dataset_v2'
THRESHOLDS = (.25, .5, .75)
SOURCES = ['tools/probe_multiview_visual_assistance.py',
           'weld_pipeline/server_multiview_visual_features.py']


def selected_jobs(dataset, focus, families):
    # Connected components prevent either shared family or shared XYZ leakage.
    parent = {j: j for j in dataset.jobs}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    owners = {}
    for key, job in dataset.jobs.items():
        for v in job['versions']:
            if v['family_id'] in owners:
                parent[find(key)] = find(owners[v['family_id']])
            owners[v['family_id']] = key
    groups = {}
    for key in sorted(parent):
        groups.setdefault(find(key), []).append(key)
    group_order = sorted(groups, key=lambda k: hashlib.sha256('|'.join(groups[k]).encode()).hexdigest())
    chosen = []
    for job in focus:
        if job not in parent:
            raise ValueError('Focus job must belong to registered TRAIN')
        if find(job) not in chosen:
            chosen.append(find(job))
    chosen += [g for g in group_order if g not in chosen][:max(0, families - len(chosen))]
    if len(chosen) != families or families < 4:
        raise ValueError('At least four complete input/family groups are required')
    return [dict(job_key=j, fold=i % 2, group=g) for i, g in enumerate(chosen) for j in groups[g]]


def load_xyz(job):
    provider = job.get('prior_input_provider')
    if provider:
        folder = ROOT / provider['input_directory']
        receipt = read(folder / 'receipt.json')
        if digest(folder / 'receipt.json') != provider['input_receipt_sha256']:
            raise ValueError('Existing raw input receipt changed')
        path = folder / 'xyz_mm.npy'
        if digest(path) != receipt['xyz_sha256']:
            raise ValueError('Raw XYZ array changed')
        xyz = np.load(path)
        binding = dict(path=str(path.relative_to(ROOT)), sha256=receipt['xyz_sha256'])
    else:
        path = ROOT.parent / 'data/G_Data' / job['workpiece_source']
        if digest(path) != job['workpiece_sha256']:
            raise ValueError('Raw PLY changed')
        xyz = ply_points(path)
        binding = dict(raw_ply_sha256=job['workpiece_sha256'])
    if xyz.shape != (job['original_points'], 3) or not np.isfinite(xyz).all():
        raise ValueError('Complete raw point count changed')
    return xyz, binding


def reference_samples(dataset, job):
    points, children = [], []
    offset = 0
    for version in job['versions']:
        supervision = dataset.supervision(version['key'])
        for parent in supervision.parents:
            for child in parent['children']:
                curve = CurveCarrier(child['representation'])
                p = (curve.at(curve.grid(1.)) if curve.kind == 'line'
                     else np.asarray(sample_arc(curve.arc, max_step_mm=1., max_chord_error_mm=.1)['points']))
                points.append(p)
                children.append(dict(version_key=version['key'], child_id=child['original_child_id'],
                                     kind=curve.kind, start=offset, end=offset + len(p)))
                offset += len(p)
    if len(children) != sum(v['children'] for v in job['versions']):
        raise ValueError('A reference child disappeared')
    return (np.concatenate(points) if points else np.empty((0, 3))), children


def sampled_fit_rows(labels, seed):
    rng = np.random.default_rng(seed)
    rows, weights = [], []
    for value, budget in ((1, 20000), (0, 40000)):
        ids = np.flatnonzero(labels == value)
        selected = rng.choice(ids, min(budget, len(ids)), replace=False)
        rows.extend(selected)
        weights.extend([len(ids) / max(1, len(selected))] * len(selected))
    return np.asarray(rows, np.int64), np.asarray(weights, np.float64)


def coverage(xyz, selected, references, children):
    distance = (cKDTree(xyz[selected]).query(references, workers=2)[0]
                if selected.any() and len(references) else np.full(len(references), np.inf))
    hit = distance <= 3.
    rows = [dict(c, fraction_with_selected_input_point_within_3mm=float(hit[c['start']:c['end']].mean()),
                 every_sample_within_3mm=bool(hit[c['start']:c['end']].all())) for c in children]
    return dict(reference_samples=len(hit), covered_samples=int(hit.sum()),
                children=len(rows), all_sample_covered_children=sum(r['every_sample_within_3mm'] for r in rows),
                arc_children=sum(r['kind'] == 'arc' for r in rows),
                all_sample_covered_arc_children=sum(r['kind'] == 'arc' and r['every_sample_within_3mm'] for r in rows),
                child_coverage=rows)


def preview(out, xyz, score, references, job_key):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    views = [(xyz, xyz[:, 2], 'Raw XYZ; color = height'),
             (xyz[score >= .5], score[score >= .5], 'Held-family visual prediction >= 0.5'),
             (references, np.ones(len(references)), 'TRAIN reference; display only')]
    stride = max(1, int(np.ceil(len(xyz) / 200000)))
    for ax, (p, color, title) in zip(axes, views):
        ax.set_facecolor('#101720')
        ax.scatter(xyz[::stride, 0], xyz[::stride, 1], s=.15, c='#596570', linewidths=0)
        ax.scatter(p[:, 0], p[:, 1], c=color, s=.6, cmap='autumn', linewidths=0)
        ax.set_aspect('equal'); ax.set_title(title); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    fig.suptitle('Multiview assistance: point regions, not extracted weld curves')
    fig.savefig(out / (job_key + '_visual.png'), dpi=140)
    plt.close(fig)


def main(args):
    out = BASE / local_name(args.experiment_name)
    out.mkdir(exist_ok=False)
    dataset = PartitionedFullShardDataset(ROOT, BASE)
    plan = selected_jobs(dataset, args.focus_job, args.families)
    sources = {p: digest(ROOT / p) for p in SOURCES}
    storage = Path(tempfile.mkdtemp(prefix='weld-visual-probe-', dir='/tmp'))
    (out / 'inputs').symlink_to(storage, target_is_directory=True)
    protocol = dict(source_commit=git_head(ROOT), sources=sources,
        registry_sha256=dataset.registry_sha256, plan=plan, pixel_mm=2., seed=1031,
        groups=args.families, folds=2, models=['XY_only', 'three_views'], thresholds=list(THRESHOLDS),
        iterations=100, max_leaf_nodes=15, max_depth=8, learning_rate=.08,
        positive_distance_to_1mm_reference_samples_mm=3., negative_distance_mm=6.,
        points_in_3_to_6mm_band_excluded_from_fit_only=True,
        fit_cap_per_job=dict(positive=20000, negative=40000),
        sample_weights='inverse within-class sampling probability, equal job mass',
        union_of_registered_TRAIN_versions_for_auxiliary_point_targets=True,
        scope='Family-disjoint cross-fit inside TRAIN only, all selected raw points scored',
        official_validation_or_test_reads=0, old_model_weights_changed=False,
        complete_extraction_or_field_accuracy_claimed=False)
    write(out / 'protocol.json', protocol)
    inputs = []; fit_data = {}; evaluated = []
    try:
        for i, item in enumerate(plan):
            key = item['job_key']; job = dataset.jobs[key]
            if shutil.disk_usage(storage).free < 2 * 1024**3:
                raise RuntimeError('Less than two GiB input cache headroom')
            start = time.monotonic(); xyz, raw_binding = load_xyz(job)
            features, audit = visual_features(xyz)
            fpath = storage / (key + '_features.npy'); np.save(fpath, features)
            # Bind input features before any reference for this job is loaded.
            binding = dict(item, input=raw_binding, features_sha256=digest(fpath),
                           points=len(xyz), audit=audit, feature_seconds=time.monotonic() - start)
            write(storage / (key + '_input_receipt.json'), binding)
            ref, children = reference_samples(dataset, job)
            distance = (cKDTree(ref).query(xyz, workers=2)[0] if len(ref) else np.full(len(xyz), np.inf))
            labels = np.full(len(xyz), -1, np.int8); labels[distance <= 3] = 1; labels[distance > 6] = 0
            ids, weights = sampled_fit_rows(labels, 1031 + i)
            fit_data[key] = (features[ids].copy(), labels[ids], weights / max(1, weights.sum()))
            np.save(storage / (key + '_positive.npy'), distance <= 3)
            binding.update(reference_children=len(children), positive_input_points=int((distance <= 3).sum()),
                           fit_points=len(ids), reference_label_sha256=[v['targets_sha256'] for v in job['versions']])
            inputs.append(binding)
            write(out / 'progress.json', dict(stage='VISUAL_INPUT_READY', jobs_done=i + 1, jobs=len(plan), job=key))
            print('VISUAL_INPUT_READY', i + 1, len(plan), key, len(children), flush=True)
            del xyz, features, ref, distance
            gc.collect()
        write(out / 'input_receipt.json', dict(inputs=inputs))
        for fold in (0, 1):
            train = [p for p in plan if p['fold'] != fold]; held = [p for p in plan if p['fold'] == fold]
            train_families = {v['family_id'] for p in train for v in dataset.jobs[p['job_key']]['versions']}
            held_families = {v['family_id'] for p in held for v in dataset.jobs[p['job_key']]['versions']}
            if train_families & held_families:
                raise ValueError('Family leakage in auxiliary probe')
            x = np.concatenate([fit_data[p['job_key']][0] for p in train])
            y = np.concatenate([fit_data[p['job_key']][1] for p in train])
            weight = np.concatenate([fit_data[p['job_key']][2] for p in train]); weight *= len(weight) / weight.sum()
            for model_name, cols in [('XY_only', 20), ('three_views', 60)]:
                model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, max_depth=8,
                    learning_rate=.08, min_samples_leaf=40, l2_regularization=1., early_stopping=False, random_state=1031)
                start = time.monotonic()
                with threadpool_limits(limits=2):
                    model.fit(x[:, :cols], y, sample_weight=weight)
                mpath = out / f'fold_{fold}_{model_name}.joblib'; joblib.dump(model, mpath, compress=3)
                write(out / f'fold_{fold}_{model_name}_fit.json', dict(fold=fold, model=model_name,
                    training_jobs=[p['job_key'] for p in train], held_jobs=[p['job_key'] for p in held],
                    disjoint_families=True, fit_rows=len(y), iterations=int(model.n_iter_),
                    seconds=time.monotonic() - start, model_sha256=digest(mpath)))
                print('VISUAL_MODEL_FIT', fold, model_name, len(y), time.monotonic() - start, flush=True)
                for item in held:
                    key = item['job_key']; job = dataset.jobs[key]; start = time.monotonic()
                    features = np.load(storage / (key + '_features.npy'), mmap_mode='r')
                    with threadpool_limits(limits=2):
                        scores = np.concatenate([model.predict_proba(features[b:b + 65536, :cols])[:, 1]
                            for b in range(0, len(features), 65536)]).astype(np.float32)
                    if not np.isfinite(scores).all() or len(scores) != job['original_points']:
                        raise ValueError('Nonfinite or incomplete full-point prediction')
                    np.save(out / f'{key}_{model_name}_scores.npy', scores)
                    inference_seconds = time.monotonic() - start
                    xyz, _ = load_xyz(job); ref, children = reference_samples(dataset, job)
                    positive = np.load(storage / (key + '_positive.npy'))
                    metrics = []
                    for threshold in THRESHOLDS:
                        selected = scores >= threshold
                        cov = coverage(xyz, selected, ref, children)
                        tp = int((selected & positive).sum()); count = int(selected.sum())
                        metrics.append(dict(threshold=threshold, selected_points=count,
                            true_near_reference_points=tp, positive_input_points=int(positive.sum()),
                            point_precision=tp / max(1, count), point_recall=tp / max(1, int(positive.sum())),
                            **cov))
                    row = dict(job_key=key, fold=fold, model=model_name, points=len(xyz),
                               inference_seconds=inference_seconds, thresholds=metrics)
                    if model_name == 'three_views':
                        row['all_input_point_capacity'] = coverage(xyz, np.ones(len(xyz), bool), ref, children)
                        if key in args.focus_job:
                            preview(out, xyz, scores, ref, key)
                    write(out / f'{key}_{model_name}_evaluation.json', row)
                    evaluated.append(row)
                    write(out / 'progress.json', dict(stage='CROSS_FIT_FULL_POINT_EVALUATION',
                          evaluated=len(evaluated), expected=2 * len(plan), fold=fold, model=model_name, job=key))
                    print('VISUAL_EVALUATED', fold, model_name, key, metrics[1]['point_precision'],
                          metrics[1]['point_recall'], flush=True)
                    del features, xyz, ref, scores
                    gc.collect()
            del x, y, weight
        if len(evaluated) != 2 * len(plan) or sources != {p: digest(ROOT / p) for p in sources}:
            raise ValueError('Incomplete probe or changed source')
        summary = []
        for name in ('XY_only', 'three_views'):
            for threshold in THRESHOLDS:
                rows = [m for r in evaluated if r['model'] == name for m in r['thresholds'] if m['threshold'] == threshold]
                fields = ['selected_points', 'true_near_reference_points', 'positive_input_points', 'reference_samples',
                          'covered_samples', 'children', 'all_sample_covered_children', 'arc_children', 'all_sample_covered_arc_children']
                sums = {f: sum(r[f] for r in rows) for f in fields}
                summary.append(dict(model=name, threshold=threshold, **sums,
                    point_precision=sums['true_near_reference_points'] / max(1, sums['selected_points']),
                    point_recall=sums['true_near_reference_points'] / max(1, sums['positive_input_points']),
                    reference_sample_coverage=sums['covered_samples'] / max(1, sums['reference_samples'])))
        write(out / 'receipt.json', dict(status='COMPLETE_TRAIN_FAMILY_DISJOINT_VISUAL_ASSISTANCE_PROBE',
            protocol_sha256=digest(out / 'protocol.json'), jobs=len(plan), groups=args.families,
            total_raw_points=sum(dataset.jobs[p['job_key']]['original_points'] for p in plan),
            reference_children=sum(sum(v['children'] for v in dataset.jobs[p['job_key']]['versions']) for p in plan),
            models_fit=4, summary=summary, official_validation_or_test_reads=0,
            full_extraction_accuracy_measured=False, original_weights_modified=False,
            next_decision='Point-region quality must justify 3D finite-curve recovery before integration'))
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(), inputs_ready=len(inputs), evaluated=len(evaluated)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--families', type=int, default=12)
    parser.add_argument('--focus-job', action='append', default=[])
    main(parser.parse_args())
