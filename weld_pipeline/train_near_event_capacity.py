"""Two preregistered near-event controls, retaining all original source children."""
import copy
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.joint_workpiece_event_votes import JointWorkpieceEventVotes
from weld_pipeline.near_event_votes import VERSION, RADIUS_MM, infer_votes
from weld_pipeline.train_joint_relation_crossfit import DATA, load_data, seed_all, atomic_save
from weld_pipeline.train_joint_workpiece_event_capacity import (
    attach_context_and_targets, sample, CONTEXT, TARGETS as OLD_TARGETS)

OUT = Path('weld_pipeline/runs/near_event_capacity_156')
TARGETS = Path('weld_pipeline/data/near_event_pilot156')
BASELINE = Path('weld_pipeline/runs/joint_workpiece_event_capacity_seed154')
PROOF = Path('weld_pipeline/reports/near_event_target_proof_156_20261001/complete.json')
AUDIT = Path('weld_pipeline/reports/event_offset_distance_bins_156_20261001/complete.json')
VARIANTS = (('physical_matched', .25), ('millimeter_loss', 16.))


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def loss_fn(logits, target, known, weight):
    bce = F.binary_cross_entropy_with_logits(logits[:, :4], target[:, :4], reduction='none')
    fg = (bce[:, :1]*known[:, :1]).sum()/known[:, :1].sum().clamp_min(1.)
    positive = target[:, 1:4]*known[:, 1:4]
    negative = (1.-target[:, 1:4])*known[:, 1:4]
    pos = (bce[:, 1:4]*positive).sum((0, 2))/positive.sum((0, 2)).clamp_min(1.)
    neg = (bce[:, 1:4]*negative).sum((0, 2))/negative.sum((0, 2)).clamp_min(1.)
    offset = F.smooth_l1_loss(logits[:, 4:7], target[:, 4:7], beta=.2, reduction='none')
    offset = (offset*known[:, 4:7]).sum((0, 2))/known[:, 4:7].sum((0, 2)).clamp_min(1.)
    return fg+.5*(pos+neg).mean()+weight*offset.mean()


def prepare_targets(data, names):
    if TARGETS.exists():
        raise ValueError('Preserve near-event targets')
    TARGETS.mkdir(parents=True); records = {}
    for name in names:
        item = data[name]; old = item['vote_targets']; known = item['vote_known']
        item['baseline_vote_targets'] = old; item['baseline_vote_known'] = known
        near = (known[:, 4:7] > .5) & (np.abs(old[:, 4:7]*64.) <= RADIUS_MM+1e-6)
        new = np.zeros(old.shape, np.float32); mask = np.zeros(known.shape, np.float32)
        new[:, 0] = old[:, 0]; new[:, 1:4] = near
        new[:, 4:7] = np.where(near, old[:, 4:7]*4., 0.)
        mask[:, :4] = known[:, :4]; mask[:, 4:7] = near
        if (not np.array_equal(new[:, 0], old[:, 0])
                or not np.array_equal(mask[:, :4], known[:, :4])
                or np.any(near & (known[:, 4:7] <= 0))):
            raise ValueError('Near support changed original foreground or unknown supervision')
        np.save(TARGETS/(name+'_targets.npy'), new)
        np.save(TARGETS/(name+'_known.npy'), mask)
        item['vote_targets'] = np.load(TARGETS/(name+'_targets.npy'), mmap_mode='r')
        item['vote_known'] = np.load(TARGETS/(name+'_known.npy'), mmap_mode='r')
        records[name] = dict(source_manifest_sha256=sha256(DATA/name/'manifest.json'),
            original_targets_sha256=sha256(OLD_TARGETS/(name+'_targets.npy')),
            original_known_sha256=sha256(OLD_TARGETS/(name+'_known.npy')),
            targets_sha256=sha256(TARGETS/(name+'_targets.npy')),
            known_sha256=sha256(TARGETS/(name+'_known.npy')), rows=len(new),
            support_points=new[:, 1:4].sum(0).astype(int).tolist(),
            known_negative_offsets=int(((new[:, 4:7]<0)&near).sum()),
            original_event_coordinates_unchanged=True,
            signed_offset_known_mask_is_explicit=True)
        print('NEAR_EVENT_PREPARED', name, records[name]['support_points'], flush=True)
    (TARGETS/'manifest.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf8')
    return records


def evaluate(model, data, names, root):
    scores = {}
    for name in names:
        item = data[name]; piece = load_piece(item['manifest']['source_cache_root'], verify=True)
        p = infer_votes(model, item, piece['sequences'])
        predictions = decode_intervals(piece['sequences'], p,
            threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
        truth = [w for w in piece['truth'] if w['type'] == 'horizontal'
                 and all(s['kind'] == 'line' for s in w['segments'])]
        score = score_children(predictions, truth); scores[name] = score
        (root/('FIT_'+name+'.json')).write_text(json.dumps(dict(score=score, predictions=predictions),
            indent=2), encoding='utf8')
        np.savez_compressed(root/('FIT_'+name+'_heatmaps.npz'), probabilities=np.concatenate(p),
                            lengths=np.array([len(v) for v in p], np.int64))
        print('NEAR_EVENT_FIT', root.name, name, score['valid_children'], '/', score['truth_count'],
              'FP', score['false_predictions'], flush=True)
    result = aggregate(scores)
    if result['truth_count'] != 1221:
        raise ValueError('Original full 1221-child denominator changed')
    (root/'FIT_summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return result


def main():
    if OUT.exists():
        raise ValueError('Preserve near-event paired trial')
    proof = json.loads(PROOF.read_text('utf8')); audit = json.loads(AUDIT.read_text('utf8'))
    baseline = json.loads((BASELINE/'complete.json').read_text('utf8'))
    base_config = json.loads((BASELINE/'config.json').read_text('utf8'))
    if (not proof['training_allowed'] or proof['target_representation']['valid_children'] != 1221
            or proof['train_sources'] != base_config['train'] or baseline['eligible_on_fit']
            or sha256(BASELINE/'final.pt') != baseline['checkpoint_sha256']
            or audit['checkpoint_sha256'] != baseline['checkpoint_sha256']):
        raise ValueError('Full representation proof and frozen failed baseline required')
    for path, digest in proof['source_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Near-event representation source changed')
    for path, digest in base_config['sources'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen baseline source changed')
    names = base_config['train']; held = base_config['held']
    if set(names) & set(held):
        raise ValueError('Train/held sources overlap')
    OUT.mkdir(parents=True)
    sources = [__file__, 'weld_pipeline/near_event_votes.py', 'weld_pipeline/verify_near_event_targets.py',
        'weld_pipeline/joint_workpiece_event_votes.py', 'weld_pipeline/train_joint_workpiece_event_capacity.py',
        'weld_pipeline/train_joint_event_vote_capacity.py', 'weld_pipeline/train_joint_relation_crossfit.py',
        'weld_pipeline/contract_child_evaluation.py', 'weld_pipeline/event_gap_decoder.py']
    config = dict(version=VERSION, trial_number=156, train=names, held=held, variants=VARIANTS,
        initialization_and_sampling_seed=151, epochs=120, steps_per_epoch=64, batch_size=8,
        points=1024, halo=64, width=128, messages=False, point_channels=97, node_channels=255,
        radius_mm=16., vote_sigma_mm=1.5, vote_tail_mm=9., vote_scale=4.,
        local_receptive_radius_mm=42., offset_beta=.2, physical_beta_mm=3.2,
        learning_rate=5e-4, weight_decay=1e-4, gradient_clip=5.,
        source_uniform=True, positive_pool_probability=.75, event_focus_probability=.65, cut_batch_probability=.5,
        baseline_config_sha256=sha256(BASELINE/'config.json'),
        baseline_checkpoint_sha256=baseline['checkpoint_sha256'], representation_proof_sha256=sha256(PROOF),
        distance_audit_sha256=sha256(AUDIT), context_complete_sha256=sha256(CONTEXT/'complete.json'),
        original_target_manifest_sha256=sha256(OLD_TARGETS/'manifest.json'),
        sources={p:sha256(p) for p in sources}, no_pretrained_weights=True,
        control_A_physical_offset_loss_matches_old_at_identical_near_points=True,
        control_B_physical_offset_loss_is_millimeter_SmoothL1=True,
        selection='Fixed epoch120. Only strict full1221 FIT gates eligible; highest TP, then lowest FP, then name. Freeze before held read.',
        observed_check_read=False, vertical_curve_and_attributes_not_scored=True, full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    snap = OUT/'source_snapshot'; snap.mkdir()
    for source in sources:
        shutil.copy2(source, snap/Path(source).name)
    cfg = json.loads((DATA/'config.json').read_text('utf8'))
    seed_all(151); data = load_data(dict(cfg, pieces=names))
    attach_context_and_targets(data, names); manifests = prepare_targets(data, names)
    seed_all(151); expected_hash = model_hash(JointWorkpieceEventVotes())
    errors = torch.tensor([-100., -16., -3.2, -1., 0., 1., 3.2, 16., 100.])
    zeros = torch.zeros_like(errors)
    matched_error = float((.25*F.smooth_l1_loss(errors/16., zeros, beta=.2, reduction='none')
                          - F.smooth_l1_loss(errors/64., zeros, beta=.05, reduction='none')).abs().max())
    mm_error = float((16.*F.smooth_l1_loss(errors/16., zeros, beta=.2, reduction='none')
                     - F.smooth_l1_loss(errors, zeros, beta=3.2, reduction='none')).abs().max())
    if matched_error > 1e-6 or mm_error > 1e-5:
        raise ValueError('Near-offset physical loss scaling is not equivalent')
    results = {}; started = time.perf_counter()
    for label, weight in VARIANTS:
        root = OUT/label; root.mkdir(); seed_all(151)
        model = JointWorkpieceEventVotes().cuda()
        if model_hash(model) != expected_hash:
            raise ValueError('Shared initial parameters differ between controls')
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        rng = np.random.default_rng(151); twin = np.random.default_rng()
        twin.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        x, node, y, known = sample(data[names[0]], rng)
        old_item = dict(data[names[0]], vote_targets=data[names[0]]['baseline_vote_targets'],
                        vote_known=data[names[0]]['baseline_vote_known'])
        old_x, old_node, old_y, old_known = sample(old_item, twin)
        if (not torch.equal(x, old_x) or not torch.equal(node, old_node)
                or not torch.equal(y[:, 0], old_y[:, 0]) or not torch.equal(known[:, :4], old_known[:, :4])
                or rng.bit_generator.state != twin.bit_generator.state):
            raise ValueError('Paired points/source RNG or original known labels changed')
        pred = model(x, node, *data[names[0]]['torch_graph'])[:, :, 64:1088]
        loss = loss_fn(pred, y, known, weight); loss.backward()
        offset_gradient = float(model.output.weight.grad[4:7].abs().sum())
        if not torch.isfinite(loss) or offset_gradient <= 0 or not bool(((y[:, 4:7]<0)&(known[:, 4:7]>0)).any()):
            raise ValueError('Actual near offset gradient or known negative offsets invalid')
        optimizer.zero_grad(set_to_none=True)
        (root/'preflight.json').write_text(json.dumps(dict(initial_parameters_sha256=expected_hash,
            point_sampling_exact=True, RNG_exact=True, foreground_exact=True, known_labels_exact=True,
            offset_gradient_l1=offset_gradient, known_negative_offsets=True,
            physical_matched_loss_max_error=matched_error, physical_mm_loss_max_error=mm_error), indent=2), encoding='utf8')
        del old_x, old_y, old_known, old_node, pred, loss
        began = time.perf_counter(); source_steps = {n:0 for n in names}
        print('NEAR_EVENT_TRAIN_START', label, 'EPOCHS', 120, 'OFFSET_WEIGHT', weight, flush=True)
        for epoch in range(1, 121):
            model.train(); losses = []
            for _ in range(64):
                name = names[int(rng.integers(len(names)))]; item = data[name]
                x, node, y, known = sample(item, rng)
                logits = model(x, node, *item['torch_graph'])[:, :, 64:1088]
                loss = loss_fn(logits, y, known, weight)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite near-event training loss')
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
                print('NEAR_EVENT_TRAIN_EPOCH', label, epoch, 'LOSS', round(float(np.mean(losses)), 6), flush=True)
        if source_steps != baseline['source_steps']:
            raise ValueError('Near-event source sampling differs from the frozen baseline')
        atomic_save(dict(model=model.state_dict(), epoch=120, variant=label, offset_weight=weight,
            config=config, config_sha256=sha256(OUT/'config.json')), root/'final.pt')
        frozen = sha256(root/'final.pt'); fit = evaluate(model, data, names, root)
        result = dict(variant=label, fit=fit, fit_gate_passed=fit['geometry_core_gates_pass'],
            final_epoch=120, offset_weight=weight, checkpoint_sha256=frozen,
            source_steps=source_steps, same_source_step_counts=True,
            target_manifest_sha256=sha256(TARGETS/'manifest.json'),
            elapsed_seconds=time.perf_counter()-began, full_contract_pass=False)
        (root/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf8'); results[label] = result
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    eligible = [r for r in results.values() if r['fit_gate_passed']]
    eligible.sort(key=lambda r:(-r['fit']['valid_children'], r['fit']['false_predictions'], r['variant']))
    selected = eligible[0]['variant'] if eligible else None
    decision = dict(eligible_variants=[r['variant'] for r in eligible], selected=selected,
        selection_only_uses_FIT=True, held_sources_not_yet_read=True, observed_check_read=False,
        baseline=baseline['fit'], variants={n:r['fit'] for n, r in results.items()}, full_contract_pass=False)
    (OUT/'fit_selection_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf8')
    if sha256(BASELINE/'final.pt') != baseline['checkpoint_sha256'] or any(
            sha256(OUT/n/'final.pt') != r['checkpoint_sha256'] for n, r in results.items()):
        raise ValueError('Frozen trial or baseline weights changed')
    complete = dict(config_sha256=sha256(OUT/'config.json'), baseline=baseline['fit'],
        variants=results, selected=selected, models_completed=2, target_manifests=manifests,
        held_status='NOT_RUN_ALL_FIT_GATES_FAILED' if not selected else 'FIT_PASS_REQUIRES_SEPARATE_FIXED_HELD_TRIAL',
        elapsed_seconds=time.perf_counter()-started, baseline_unchanged=True,
        observed_check_read=False, full_source_adoption_not_validated=True,
        vertical_curve_and_attributes_not_scored=True, full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete, indent=2, ensure_ascii=False), encoding='utf8')
    print('NEAR_EVENT_CAPACITY_COMPLETE', {n:(r['fit']['valid_children'], r['fit']['false_predictions'])
          for n, r in results.items()}, 'SELECTED', selected, flush=True)


if __name__ == '__main__':
    main()
