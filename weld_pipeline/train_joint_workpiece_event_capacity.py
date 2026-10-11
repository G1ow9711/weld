"""Preregistered 97/255-input capacity trial on the same 16 sources and 1221 children."""
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.joint_event_votes import JointEventVotes
from weld_pipeline.joint_workpiece_event_votes import JointWorkpieceEventVotes, infer_votes, VERSION, LOCAL_COUNT
from weld_pipeline.train_joint_event_vote_capacity import loss_fn, sample as old_sample
from weld_pipeline.train_joint_relation_crossfit import DATA as BASE, load_data, seed_all, atomic_save

CONTEXT = Path('weld_pipeline/data/joint_workpiece_context_pilot154')
TARGETS = Path('weld_pipeline/data/joint_event_votes_pilot151')
BASELINE = Path('weld_pipeline/runs/joint_event_vote_capacity_seed151')
OUT = Path('weld_pipeline/runs/joint_workpiece_event_capacity_seed154')


def attach_context_and_targets(data, names):
    cfg = json.loads((CONTEXT/'config.json').read_text('utf8'))
    complete = json.loads((CONTEXT/'complete.json').read_text('utf8'))
    if sha256(CONTEXT/'config.json') != complete['config_sha256'] or cfg['pieces'] != names:
        raise ValueError('Exactly the original 16 train-source contexts required')
    target_manifest = json.loads((TARGETS/'manifest.json').read_text('utf8'))
    for name in names:
        root = CONTEXT/name; piece = data[name]
        manifest = json.loads((root/'manifest.json').read_text('utf8'))
        if (sha256(root/'manifest.json') != complete['manifests'][name]
                or manifest['base_manifest_sha256'] != sha256(BASE/name/'manifest.json')):
            raise ValueError('Frozen canonical context manifest changed')
        for file, digest in manifest['files'].items():
            if sha256(root/file) != digest:
                raise ValueError('Frozen context array changed')
        tm = target_manifest[name]
        if (tm['original_source_manifest_sha256'] != sha256(BASE/name/'manifest.json')
                or sha256(TARGETS/(name+'_targets.npy')) != tm['targets_sha256']
                or sha256(TARGETS/(name+'_known.npy')) != tm['known_sha256']):
            raise ValueError('Original dense vote supervision changed')
        piece['point_context'] = np.load(root/'point_context.npy', mmap_mode='r')
        node_context = np.load(root/'node_context.npy', allow_pickle=False)
        graph = piece['torch_graph']
        piece['original_torch_graph'] = graph
        nodes = torch.cat((graph[0], torch.from_numpy(node_context).to(graph[0].device)), 1)
        piece['torch_graph'] = (nodes, *graph[1:])
        piece['vote_targets'] = np.load(TARGETS/(name+'_targets.npy'), mmap_mode='r')
        piece['vote_known'] = np.load(TARGETS/(name+'_known.npy'), mmap_mode='r')
        piece['cut_entries'] = []
        for entry in piece['horizontal']:
            ids = np.flatnonzero(piece['targets'][entry['start']:entry['stop'], 3] > .25)
            if len(ids):
                piece['cut_entries'].append(dict(entry, event_indices=ids.tolist()))
    return target_manifest


def sample(piece, rng):
    batch = 8; points = 1024; halo = 64
    x = np.zeros((batch, LOCAL_COUNT, points+2*halo), np.float32)
    y = np.zeros((batch, 7, points), np.float32); known = np.zeros_like(y); nodes = []
    cut_focus = bool(piece['cut_entries'] and rng.random() < .5)
    for bi in range(batch):
        if cut_focus:
            pool = piece['cut_entries']
        else:
            pool = piece['positive'] if piece['positive'] and rng.random() < .75 else piece['horizontal']
        entry = pool[int(rng.integers(len(pool)))]; length = entry['stop']-entry['start']
        if length > points:
            focus = entry['event_indices'] if rng.random() < .65 else entry['foreground_indices']
            if focus and rng.random() < .8:
                centre = focus[int(rng.integers(len(focus)))]
                start = int(np.clip(centre-rng.integers(points//4, 3*points//4), 0, length-points))
            else:
                start = int(rng.integers(length-points+1))
        else:
            start = 0
        stop = min(length, start+points); lo = start-halo
        left, right = max(0, lo), min(length, stop+halo)
        sl = slice(entry['start']+left, entry['start']+right)
        x[bi, :75, left-lo:right-lo] = piece['features'][sl].T
        x[bi, 75:, left-lo:right-lo] = piece['point_context'][sl].T
        a, b = entry['start']+start, entry['start']+stop
        y[bi, :, :stop-start] = piece['vote_targets'][a:b].T
        known[bi, :, :stop-start] = piece['vote_known'][a:b].T
        nodes.append(entry['node'])
    return (torch.from_numpy(x).cuda(), torch.tensor(nodes, device='cuda'),
            torch.from_numpy(y).cuda(), torch.from_numpy(known).cuda())


def evaluate(model, data, names):
    scores = {}
    for name in names:
        piece = load_piece(data[name]['manifest']['source_cache_root'], verify=True)
        probabilities = infer_votes(model, data[name], piece['sequences'])
        predictions = decode_intervals(piece['sequences'], probabilities,
            threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
        truth = [w for w in piece['truth'] if w['type'] == 'horizontal'
                 and all(s['kind'] == 'line' for s in w['segments'])]
        score = score_children(predictions, truth); scores[name] = score
        (OUT/('FIT_'+name+'.json')).write_text(json.dumps(dict(score=score, predictions=predictions),
            indent=2, ensure_ascii=False), encoding='utf8')
        np.savez_compressed(OUT/('FIT_'+name+'_heatmaps.npz'), probabilities=np.concatenate(probabilities),
            lengths=np.array([len(p) for p in probabilities], np.int64))
        print('JOINT_WORKPIECE_FIT', name, score['valid_children'], '/', score['truth_count'],
              'FP', score['false_predictions'], flush=True)
    result = aggregate(scores)
    (OUT/'FIT_summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return result


def main():
    if OUT.exists():
        raise ValueError('Preserve fixed joint workpiece capacity trial')
    cfg = json.loads((BASE/'config.json').read_text('utf8')); fold = cfg['folds'][0]
    names = fold['train_real']
    raw_path = Path('weld_pipeline/reports/joint_workpiece_context_raw_154_20261001/complete.json')
    raw = json.loads(raw_path.read_text('utf8'))
    if not raw['all_checks_passed'] or set(names)&set(fold['held_out_real']):
        raise ValueError('Raw neural input/source partition proof required')
    for source, digest in raw['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Raw neural input proof source changed')
    baseline = json.loads((BASELINE/'complete.json').read_text('utf8'))
    if sha256(BASELINE/'final.pt') != baseline['checkpoint_sha256']:
        raise ValueError('Frozen baseline checkpoint changed')
    baseline_hash = sha256(BASELINE/'final.pt')
    OUT.mkdir(parents=True)
    sources = [__file__, 'weld_pipeline/joint_workpiece_event_votes.py',
        'weld_pipeline/workpiece_relative_context.py', 'weld_pipeline/joint_event_votes.py',
        'weld_pipeline/joint_relation_heatmaps.py', 'weld_pipeline/train_joint_event_vote_capacity.py',
        'weld_pipeline/train_joint_relation_crossfit.py', 'weld_pipeline/contract_child_evaluation.py',
        'weld_pipeline/event_gap_decoder.py']
    config = dict(version=VERSION, trial_number=154, initialization_and_sampling_seed=151,
        train=names, held=fold['held_out_real'], epochs=120, steps_per_epoch=64,
        batch_size=8, points=1024, halo=64, width=128, messages=False,
        point_channels=97, node_channels=255, added_context_columns_start_zero=True,
        public_initial_weights_identical_to_baseline=True, local_receptive_radius_mm=42.,
        learning_rate=5e-4, weight_decay=1e-4, offset_beta=.05, offset_weight=1.,
        vote_radius_mm=64., vote_sigma_mm=1.5, vote_tail_mm=9., vote_scale=4.,
        loss='Unchanged original151 foreground_BCE_balanced_support_BCE_known_offset_SmoothL1',
        source_uniform=True, positive_pool_probability=.75, event_focus_probability=.65, cut_batch_probability=.5,
        context_config_sha256=sha256(CONTEXT/'config.json'), context_complete_sha256=sha256(CONTEXT/'complete.json'),
        target_manifest_sha256=sha256(TARGETS/'manifest.json'), raw_input_proof_sha256=sha256(raw_path),
        baseline_checkpoint_sha256=baseline_hash, sources={p:sha256(p) for p in sources},
        selection='Fixed last epoch. Full 1221 FIT child/core category gates before any new held evaluation.',
        no_pretrained_weights=True, observed_check_read=False,
        vertical_and_curve_supervision_not_run=True, other_contract_fields_not_scored=True, full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    snapshot = OUT/'source_snapshot'; snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot/Path(source).name)
    seed_all(151); data = load_data(dict(cfg, pieces=names))
    target_manifest = attach_context_and_targets(data, names)
    model = JointWorkpieceEventVotes().cuda()
    seed_all(151); old_model = JointEventVotes().cuda()
    old_state = old_model.state_dict(); state = model.state_dict()
    for key, old in old_state.items():
        new = state[key]
        if key == 'point_encoder.0.weight':
            if not torch.equal(new[:, :75], old) or bool(new[:, 75:].any()):
                raise ValueError('Point initialization is not paired')
        elif key == 'node_encoder.0.weight':
            if not torch.equal(new[:, :233], old) or bool(new[:, 233:].any()):
                raise ValueError('Node initialization is not paired')
        elif not torch.equal(new, old):
            raise ValueError('Shared initial parameter differs: '+key)
    rng = np.random.default_rng(151); twin = np.random.default_rng()
    twin.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    x, node, y, known = sample(data[names[0]], rng)
    old_x, old_node, old_y, old_known = old_sample(data[names[0]], twin)
    if (not all(torch.equal(a, b) for a, b in ((x[:, :75], old_x), (node, old_node), (y, old_y), (known, old_known)))
            or rng.bit_generator.state != twin.bit_generator.state):
        raise ValueError('Paired source sampling changed')
    model.eval(); old_model.eval()
    with torch.no_grad():
        pred = model(x, node, *data[names[0]]['torch_graph'])
        old_pred = old_model(old_x, old_node, *data[names[0]]['original_torch_graph'])
        initial_error = float((pred-old_pred).abs().max())
        if not torch.allclose(pred, old_pred, atol=1e-5, rtol=1e-5):
            raise ValueError('Neutral added context does not preserve initial inference')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train(); optimizer.zero_grad(set_to_none=True)
    preflight_loss = loss_fn(model(x, node, *data[names[0]]['torch_graph'])[:, :, 64:1088], y, known)
    preflight_loss.backward()
    context_gradient = float(model.point_encoder[0].weight.grad[:, 75:].abs().sum())
    if (not torch.isfinite(preflight_loss) or context_gradient <= 0
            or not bool(((y[:, 4:7] < 0)&(known[:, 4:7] > 0)).any())):
        raise ValueError('New context gradients/known negative offsets invalid')
    optimizer.zero_grad(set_to_none=True)
    del old_model, pred, old_pred, x, old_x, y, old_y, known, old_known
    torch.cuda.empty_cache()
    (OUT/'preflight.json').write_text(json.dumps(dict(public_parameters_exact=True, sampling_exact=True,
        initial_forward_max_error=initial_error, added_point_context_gradient_l1=context_gradient,
        known_negative_offset_gradient_verified=True, raw_input_proof_sha256=sha256(raw_path)), indent=2), encoding='utf8')
    started = time.perf_counter(); source_steps = {n:0 for n in names}
    print('JOINT_WORKPIECE_TRAIN_START', len(names), 'EPOCHS', 120, 'CONTEXT_GRADIENT', context_gradient, flush=True)
    for epoch in range(1, 121):
        model.train(); losses = []
        for _ in range(64):
            name = names[int(rng.integers(len(names)))]; piece = data[name]
            x, node, y, known = sample(piece, rng)
            logits = model(x, node, *piece['torch_graph'])[:, :, 64:1088]
            loss = loss_fn(logits, y, known)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite whole-context training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); losses.append(float(loss.detach())); source_steps[name] += 1
        with (OUT/'training.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(epoch=epoch, loss=float(np.mean(losses)),
                elapsed_seconds=time.perf_counter()-started))+'\n')
        if epoch % 16 == 0 or epoch == 120:
            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                sampling_rng=rng.bit_generator.state, source_steps=source_steps,
                config_sha256=sha256(OUT/'config.json')), OUT/'latest.pt')
            print('JOINT_WORKPIECE_TRAIN_EPOCH', epoch, 'LOSS', round(float(np.mean(losses)), 6), flush=True)
    if source_steps != baseline['source_steps']:
        raise ValueError('Whole-context trial source sampling differs from baseline')
    atomic_save(dict(model=model.state_dict(), epoch=120, config=config), OUT/'final.pt')
    frozen = sha256(OUT/'final.pt'); fit = evaluate(model, data, names)
    eligible = bool(fit['geometry_core_gates_pass'])
    held_status = 'NOT_RUN_FIT_CAPACITY_GATE_FAILED' if not eligible else 'FIT_GATE_PASSED_REQUIRES_SEPARATE_FROZEN_HELD_TRIAL'
    if sha256(OUT/'final.pt') != frozen or sha256(BASELINE/'final.pt') != baseline_hash:
        raise ValueError('Current or baseline frozen model changed')
    if (fit['truth_count'] != 1221 or set(names)&set(fold['held_out_real'])):
        raise ValueError('Full training-source denominator changed')
    result = dict(config_sha256=sha256(OUT/'config.json'), checkpoint_sha256=frozen, final_epoch=120,
        fit=fit, eligible_on_fit=eligible, held=None, held_status=held_status,
        source_steps=source_steps, source_sampling_exact_to_baseline=True, baseline_checkpoint_unchanged=True,
        target_manifest_sha256=sha256(TARGETS/'manifest.json'), target_manifests=target_manifest,
        elapsed_seconds=time.perf_counter()-started, observed_check_read=False,
        vertical_curve_and_attributes_not_scored=True, three_seed_adoption_not_validated=True, full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    print('JOINT_WORKPIECE_CAPACITY_COMPLETE', fit['valid_children'], '/', fit['truth_count'],
          'FP', fit['false_predictions'], 'HELD', held_status, flush=True)


if __name__ == '__main__':
    main()
