"""Three fixed consistency/mining controls; full original FIT gates before held use."""
import copy
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.joint_workpiece_event_votes import JointWorkpieceEventVotes
from weld_pipeline.near_event_consistency_loss import loss_fn, center_consistency, hard_background_means
from weld_pipeline.train_near_event_capacity import model_hash, evaluate, TARGETS
from weld_pipeline.train_joint_relation_crossfit import DATA, load_data, seed_all, atomic_save
from weld_pipeline.train_joint_workpiece_event_capacity import attach_context_and_targets, sample

OUT = Path('weld_pipeline/runs/near_event_consistency_160')
BASE = Path('weld_pipeline/runs/near_event_capacity_156')
VARIANTS = (('center_consistency', True, False), ('hard_background', False, True), ('combined', True, True))


def preflight_physics():
    positions = torch.arange(32, dtype=torch.float64)*2.
    targets = torch.zeros((1, 7, 32), dtype=torch.float64)
    known = torch.zeros_like(targets)
    target = (31.3-positions)/16.
    valid = target.abs() <= 1.
    targets[:, 4:7] = target; known[:, 4:7] = valid
    correct = targets.clone().requires_grad_(True)
    value = center_consistency(correct, targets, known)
    if float(value) > 1e-12:
        raise ValueError('Exact stationary centers incur nonzero consistency cost')
    flat = torch.zeros_like(targets).requires_grad_(True)
    penalty = center_consistency(flat, targets, known); penalty.backward()
    gradient = float(flat.grad[:, 4:7].abs().sum())
    if float(penalty) <= 0 or gradient <= 0:
        raise ValueError('Non-converging field does not receive real consistency gradients')
    # Known offsets may be negative, including the valid value -1.
    if not bool(((targets[:, 4:7] < 0)&(known[:, 4:7] > 0)).any()):
        raise ValueError('Signed known-offset physics case missing')
    scores = torch.tensor([[[.1, .2, 5., 10.]]]*3).transpose(0, 1)
    positives = torch.zeros_like(scores); negatives = torch.ones_like(scores)
    hard = hard_background_means(scores, positives, negatives)
    if not torch.isfinite(hard).all():
        raise ValueError('No-positive hard-background case nonfinite')
    return dict(exact_centers_loss=float(value), flat_field_loss=float(penalty),
                flat_field_gradient_l1=gradient, signed_known_offsets=True, no_positive_mining_finite=True)


def main():
    if OUT.exists():
        raise ValueError('Preserve fixed consistency/mining trial')
    parent = json.loads((BASE/'complete.json').read_text('utf8'))
    cfg = json.loads((BASE/'config.json').read_text('utf8'))
    baseline = parent['variants']['physical_matched']; checkpoint = BASE/'physical_matched'/'final.pt'
    support_path = Path('weld_pipeline/reports/near_event_support_localization_158_20261001/complete.json')
    stationarity_path = Path('weld_pipeline/reports/near_vote_stationarity_159_20261001/complete.json')
    support = json.loads(support_path.read_text('utf8'))
    stationarity = json.loads(stationarity_path.read_text('utf8'))
    if (sha256(checkpoint) != baseline['checkpoint_sha256'] or parent['selected'] is not None
            or support['checkpoint_sha256'] != baseline['checkpoint_sha256']
            or stationarity['checkpoint_sha256'] != baseline['checkpoint_sha256']
            or support['results']['learned']['truth_count'] != 1221):
        raise ValueError('Frozen failed model and full causal diagnoses required')
    for source, digest in cfg['sources'].items():
        if sha256(source) != digest:
            raise ValueError('Original paired training/inference source changed')
    names = cfg['train']; target_manifest = json.loads((TARGETS/'manifest.json').read_text('utf8'))
    if target_manifest != parent['target_manifests']:
        raise ValueError('Frozen near targets changed')
    physics = preflight_physics()
    OUT.mkdir(parents=True)
    sources = [__file__, 'weld_pipeline/near_event_consistency_loss.py',
        'weld_pipeline/train_near_event_capacity.py', 'weld_pipeline/near_event_votes.py',
        'weld_pipeline/train_joint_workpiece_event_capacity.py', 'weld_pipeline/joint_workpiece_event_votes.py',
        'weld_pipeline/contract_child_evaluation.py', 'weld_pipeline/event_gap_decoder.py']
    config = dict(cfg, trial_number=160, variants=VARIANTS,
        center_consistency_lags=[1, 4, 8], center_loss_beta_mm=3.2, center_loss_weight=1.,
        same_event_pair_tolerance_normalized=2e-5, known_pair_required=True,
        hard_background_K='min(known_negatives,max(16,known_positive_points))',
        hard_background_and_all_background_equal_mix=True,
        baseline_complete_sha256=sha256(BASE/'complete.json'), baseline_checkpoint_sha256=baseline['checkpoint_sha256'],
        causal_support_audit_sha256=sha256(support_path), stationarity_audit_sha256=sha256(stationarity_path),
        target_manifest_sha256=sha256(TARGETS/'manifest.json'),
        sources={p:sha256(p) for p in sources}, no_pretrained_weights=True, full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    (OUT/'physical_preflight.json').write_text(json.dumps(physics, indent=2), encoding='utf8')
    snap = OUT/'source_snapshot'; snap.mkdir()
    for source in sources:
        shutil.copy2(source, snap/Path(source).name)
    base_data_cfg = json.loads((DATA/'config.json').read_text('utf8'))
    seed_all(151); data = load_data(dict(base_data_cfg, pieces=names)); attach_context_and_targets(data, names)
    for name in names:
        tm = target_manifest[name]
        if (sha256(TARGETS/(name+'_targets.npy')) != tm['targets_sha256']
                or sha256(TARGETS/(name+'_known.npy')) != tm['known_sha256']
                or sha256(DATA/name/'manifest.json') != tm['source_manifest_sha256']):
            raise ValueError('Prepared near-event targets or source changed')
        data[name]['vote_targets'] = np.load(TARGETS/(name+'_targets.npy'), mmap_mode='r')
        data[name]['vote_known'] = np.load(TARGETS/(name+'_known.npy'), mmap_mode='r')
    expected = json.loads((BASE/'physical_matched'/'preflight.json').read_text('utf8'))['initial_parameters_sha256']
    results = {}; started = time.perf_counter()
    for label, consistency, mining in VARIANTS:
        root = OUT/label; root.mkdir(); seed_all(151)
        model = JointWorkpieceEventVotes().cuda()
        if model_hash(model) != expected:
            raise ValueError('Initial parameters differ from the fixed near baseline')
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        rng = np.random.default_rng(151); twin = np.random.default_rng()
        twin.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        x, node, y, known = sample(data[names[0]], rng)
        tx, tn, ty, tk = sample(data[names[0]], twin)
        if (not all(torch.equal(a, b) for a, b in ((x, tx), (node, tn), (y, ty), (known, tk)))
                or rng.bit_generator.state != twin.bit_generator.state):
            raise ValueError('Actual paired sampling changed')
        logits = model(x, node, *data[names[0]]['torch_graph'])[:, :, 64:1088]
        loss = loss_fn(logits, y, known, consistency, mining); loss.backward()
        gradient = float(model.output.weight.grad.abs().sum())
        if not torch.isfinite(loss) or gradient <= 0:
            raise ValueError('Actual consistency/mining gradient invalid')
        optimizer.zero_grad(set_to_none=True)
        (root/'preflight.json').write_text(json.dumps(dict(initial_parameters_sha256=expected,
            paired_sampling_exact=True, negative_known_offsets=True, actual_output_gradient_l1=gradient,
            physical_preflight_sha256=sha256(OUT/'physical_preflight.json')), indent=2), encoding='utf8')
        del tx, ty, tk, tn, logits, loss
        source_steps = {n:0 for n in names}; began = time.perf_counter()
        print('NEAR_CONSISTENCY_TRAIN_START', label, 'EPOCHS', 120, flush=True)
        for epoch in range(1, 121):
            model.train(); losses = []
            for _ in range(64):
                name = names[int(rng.integers(len(names)))]; item = data[name]
                x, node, y, known = sample(item, rng)
                logits = model(x, node, *item['torch_graph'])[:, :, 64:1088]
                loss = loss_fn(logits, y, known, consistency, mining)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite consistency/mining loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step(); losses.append(float(loss.detach())); source_steps[name] += 1
            with (root/'training.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(dict(epoch=epoch, loss=float(np.mean(losses)),
                    elapsed_seconds=time.perf_counter()-began))+'\n')
            if epoch % 16 == 0 or epoch == 120:
                atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                    sampling_rng=rng.bit_generator.state, source_steps=source_steps,
                    config_sha256=sha256(OUT/'config.json')), root/'latest.pt')
                print('NEAR_CONSISTENCY_TRAIN_EPOCH', label, epoch, round(float(np.mean(losses)), 6), flush=True)
        if source_steps != baseline['source_steps']:
            raise ValueError('Training source sampling differs from the original near baseline')
        atomic_save(dict(model=model.state_dict(), epoch=120, variant=label, config=config,
            config_sha256=sha256(OUT/'config.json')), root/'final.pt')
        frozen = sha256(root/'final.pt'); fit = evaluate(model, data, names, root)
        result = dict(variant=label, consistency=consistency, hard_background=mining, fit=fit,
            fit_gate_passed=fit['geometry_core_gates_pass'], final_epoch=120, checkpoint_sha256=frozen,
            source_steps=source_steps, source_sampling_exact_to_baseline=True,
            target_manifest_sha256=sha256(TARGETS/'manifest.json'), elapsed_seconds=time.perf_counter()-began,
            held_or_observed_read=False, full_contract_pass=False)
        (root/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf8'); results[label] = result
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    eligible = [r for r in results.values() if r['fit_gate_passed']]
    eligible.sort(key=lambda r:(-r['fit']['valid_children'], r['fit']['false_predictions'], r['variant']))
    selected = eligible[0]['variant'] if eligible else None
    decision = dict(selected=selected, eligible=[r['variant'] for r in eligible],
        selection_only_uses_FIT=True, held_sources_not_yet_read=True, observed_check_read=False,
        baseline=baseline['fit'], variants={k:r['fit'] for k, r in results.items()}, full_contract_pass=False)
    (OUT/'fit_selection_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf8')
    if sha256(checkpoint) != baseline['checkpoint_sha256'] or any(
            sha256(OUT/n/'final.pt') != r['checkpoint_sha256'] for n, r in results.items()):
        raise ValueError('Frozen baseline or new model changed')
    complete = dict(config_sha256=sha256(OUT/'config.json'), baseline=baseline['fit'], variants=results,
        models_completed=3, selected=selected, baseline_unchanged=True, elapsed_seconds=time.perf_counter()-started,
        target_manifest_sha256=sha256(TARGETS/'manifest.json'),
        held_status='NOT_RUN_ALL_FIT_GATES_FAILED' if selected is None else 'FIT_PASS_REQUIRES_SEPARATE_FIXED_HELD_TRIAL',
        vertical_curve_and_attributes_not_scored=True, full_source_adoption_not_validated=True,
        observed_check_read=False, full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete, indent=2, ensure_ascii=False), encoding='utf8')
    print('NEAR_CONSISTENCY_COMPLETE', {n:(r['fit']['valid_children'], r['fit']['false_predictions'])
          for n, r in results.items()}, 'SELECTED', selected, flush=True)


if __name__ == '__main__':
    main()
