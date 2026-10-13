"""Recover finite XY-plane arc proposals from held-family visual point scores.

Narrow auxiliary feasibility experiment, not a general orientation-invariant
extractor. Proposals are frozen before reference geometry is read. A label can
only measure underlying-circle/support capacity, never set cuts or fit circles.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from tools.export_full_dataset_split_view import local_name
from tools.probe_multiview_visual_assistance import BASE, ROOT, load_xyz, reference_samples
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


def circle3(p):
    matrix = 2 * (p[1:] - p[0])
    if abs(np.linalg.det(matrix)) < 1.:
        return None
    center = np.linalg.solve(matrix, np.sum(p[1:] ** 2, axis=1) - np.sum(p[0] ** 2))
    return center, float(np.linalg.norm(p[0] - center))


def propose(xyz, scores, seed=1031):
    chosen = np.flatnonzero(scores >= .5)
    if len(chosen) < 100:
        return [], dict(selected_points=len(chosen), reason='Insufficient visual point support')
    p = xyz[chosen]; bins, counts = np.unique(np.round(p[:, 2] * 2), return_counts=True)
    levels = bins[np.argsort(-counts, kind='stable')[:3]] / 2
    rng = np.random.default_rng(seed); output = []
    max_radius = float(np.linalg.norm(np.ptp(xyz[:, :2], axis=0)) * 10)
    for level in levels:
        ids = chosen[np.abs(p[:, 2] - level) <= .25]
        z = float(np.median(xyz[ids, 2]))
        remaining = xyz[ids, :2].copy()
        for iteration in range(8):
            if len(remaining) < 100:
                break
            subset = remaining[rng.choice(len(remaining), min(6000, len(remaining)), replace=False)]
            best = None; best_count = 0
            for _ in range(384):
                candidate = circle3(subset[rng.choice(len(subset), 3, replace=False)])
                if candidate is None:
                    continue
                center, radius = candidate
                if not 20 <= radius <= max_radius:
                    continue
                distance = np.abs(np.linalg.norm(subset - center, axis=1) - radius)
                n = int((distance <= 1.).sum())
                if n > best_count:
                    best = (center, radius); best_count = n
            if best is None or best_count < 50:
                break
            center, radius = best
            for _ in range(3):
                mask = np.abs(np.linalg.norm(remaining - center, axis=1) - radius) <= 1.
                support = remaining[mask]
                if len(support) < 100:
                    break
                fit = least_squares(lambda v: np.linalg.norm(support - v[:2], axis=1) - v[2],
                                    np.r_[center, radius], max_nfev=30, loss='soft_l1', f_scale=.5)
                center, radius = fit.x[:2], float(fit.x[2])
            mask = np.abs(np.linalg.norm(remaining - center, axis=1) - radius) <= 1.
            support = remaining[mask]
            if len(support) < 100 or not 20 <= radius <= max_radius:
                break
            remaining = remaining[~mask]
            angles = np.sort(np.mod(np.arctan2(support[:, 1] - center[1], support[:, 0] - center[0]), 2 * np.pi))
            # Rotate at the largest observed gap and retain other finite gaps.
            wrap_gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
            cut = (int(np.argmax(wrap_gaps)) + 1) % len(angles)
            unwrapped = np.r_[angles[cut:], angles[:cut] + 2 * np.pi]
            groups = np.split(unwrapped, np.flatnonzero(np.diff(unwrapped) * radius > 10.) + 1)
            for group in groups:
                if len(group) < 30 or (group[-1] - group[0]) * radius < 30:
                    continue
                representation = dict(kind='arc', arc_parameters=dict(center_mm=[*center.tolist(), z],
                    radius_mm=radius, plane_normal=[0., 0., 1.],
                    start_direction=[float(np.cos(group[0])), float(np.sin(group[0])), 0.],
                    sweep_rad=float(group[-1] - group[0]), is_closed=False, units='mm', angle_units='rad'))
                CurveCarrier(representation)  # Reject malformed finite analytic geometry.
                output.append(dict(representation=representation, support_points=len(group),
                    radial_support_tolerance_mm=1., maximum_observed_support_gap_mm=10.,
                    observed_plane_z_mm=z, plane_band_halfwidth_mm=.25, input_generated=True))
    return output, dict(selected_points=len(chosen), input_depth_levels_mm=levels.tolist(),
                        orientation_scope='XY parallel arcs only', raw_points=len(xyz))


def main(args):
    out = BASE / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    source = BASE / local_name(args.visual_experiment)
    receipt = read(source / 'receipt.json'); protocol = read(source / 'protocol.json')
    if receipt['status'] != 'COMPLETE_TRAIN_FAMILY_DISJOINT_VISUAL_ASSISTANCE_PROBE':
        raise ValueError('Completed family-disjoint visual scores required')
    dataset = PartitionedFullShardDataset(ROOT, BASE)
    write(out / 'protocol.json', dict(source_sha256=digest(Path(__file__)),
        visual_receipt_sha256=digest(source / 'receipt.json'), jobs=args.job,
        visual_threshold=.5, circle_fit_seed=1031, trials_per_circle=384, circles_per_level=8,
        input_depth_levels=3, max_support_gap_mm=10., radial_tolerance_mm=1.,
        fit_uses_labels=False, official_validation_or_test_reads=0,
        scope='Finite supported circle capacity only; no full weld selection, cuts, fields or CSV'))
    frozen = []
    for key in args.job:
        row = next(p for p in protocol['plan'] if p['job_key'] == key)
        fit = read(source / f"fold_{row['fold']}_three_views_fit.json")
        if key not in fit['held_jobs'] or key in fit['training_jobs'] or not fit['disjoint_families']:
            raise ValueError('Visual predictions must exclude the complete target family from fitting')
        score_path = source / f'{key}_three_views_scores.npy'
        xyz, binding = load_xyz(dataset.jobs[key]); score = np.load(score_path)
        if score.shape != (len(xyz),) or not np.isfinite(score).all():
            raise ValueError('Incomplete visual point scores')
        start = time.monotonic(); proposals, audit = propose(xyz, score)
        p = out / (key + '_proposals.json')
        write(p, dict(job_key=key, input=binding, scores_sha256=digest(score_path),
                      proposals=proposals, audit=audit, generation_seconds=time.monotonic() - start,
                      reference_geometry_reads=0))
        frozen.append(dict(job_key=key, proposal_sha256=digest(p)))
        print('VISUAL_ARCS_FROZEN', key, len(proposals), time.monotonic() - start, flush=True)
    write(out / 'all_proposals_frozen.json', dict(records=frozen, reference_geometry_reads=0))
    results = []
    for item in frozen:
        key = item['job_key']; p = out / (key + '_proposals.json')
        assert digest(p) == item['proposal_sha256']
        proposed = read(p)['proposals']; curves = [CurveCarrier(r['representation']) for r in proposed]
        points, children = reference_samples(dataset, dataset.jobs[key])
        rows = []
        for child in children:
            if child['kind'] != 'arc':
                continue
            ref = points[child['start']:child['end']]
            full_errors = [float(c.project(ref)[1].max()) for c in curves]
            circle_errors = [float(np.hypot(np.linalg.norm(ref[:, :2] - c.arc.center_mm[:2], axis=1)
                                - c.arc.radius_mm, ref[:, 2] - c.arc.center_mm[2]).max()) for c in curves]
            bidirectional = []
            for c in curves:
                positions = c.project(ref)[0]
                lo, hi = float(positions.min()), float(positions.max())
                # Only offline capacity measurement: these GT-derived cuts are never proposals.
                q = np.linspace(lo, hi, max(2, int(np.ceil(hi - lo)) + 1))
                back = cKDTree(ref).query(c.at(q), workers=2)[0]
                bidirectional.append(max(float(c.project(ref)[1].max()), float(back.max())))
            rows.append(dict(child, best_finite_carrier_oneway_error_mm=min(full_errors) if full_errors else None,
                best_underlying_circle_error_mm=min(circle_errors) if circle_errors else None,
                best_bidirectional_projected_interval_capacity_error_mm=min(bidirectional) if bidirectional else None,
                finite_carrier_supported_within_3mm=bool(full_errors and min(full_errors) <= 3),
                underlying_circle_within_3mm=bool(circle_errors and min(circle_errors) <= 3),
                offline_interval_capacity_within_3mm=bool(bidirectional and min(bidirectional) <= 3)))
        results.append(dict(job_key=key, proposals=len(proposed), arc_children=len(rows),
            underlying_circle_capacity=sum(r['underlying_circle_within_3mm'] for r in rows),
            finite_carrier_capacity=sum(r['finite_carrier_supported_within_3mm'] for r in rows),
            offline_interval_capacity=sum(r['offline_interval_capacity_within_3mm'] for r in rows), children=rows))
    write(out / 'receipt.json', dict(status='VISUAL_INPUT_ONLY_ARC_CAPACITY_PROBE_COMPLETE', results=results,
        predictions_frozen_before_reference_reads=True, offline_reference_projection_not_runtime_cuts=True,
        complete_extraction_accuracy_claimed=False, old_weights_modified=False, official_validation_or_test_reads=0))
    print('VISUAL_ARC_CAPACITY', [(r['job_key'], r['underlying_circle_capacity'], r['finite_carrier_capacity'], r['arc_children']) for r in results], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--experiment-name', required=True)
    p.add_argument('--visual-experiment', required=True); p.add_argument('--job', action='append', required=True)
    main(p.parse_args())
