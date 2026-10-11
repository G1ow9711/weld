"""FIT-first bounded deep trial, preserving all original rows plus hard negatives."""
import gc
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.bounded_event_phase_v2 import FEATURE_COUNT
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.continuous_event_decoder import decode_event_sets
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.forest_event_centers import build_inputs
from weld_pipeline.neural_bounded_events import VERSION, BoundedEventNet, predict_fields, assemble_events
from weld_pipeline.train_forest_event_centers import BASE, load_cached
from weld_pipeline.train_neural_physical_events_v3 import state_digest

DATA = Path('weld_pipeline/data/hard_physical_event_pilot175_v2')
PROOF = Path('weld_pipeline/reports/bounded_phase_centers_175v2_20261002')
CONFIDENCE = Path('weld_pipeline/reports/correlated_center_confidence_176_20261002')
OUT = Path('weld_pipeline/runs/deep_bounded_events_175')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def source_weights(y, known, priority, ranges):
    weights = np.zeros_like(y)
    for role in range(7):
        for start, stop in ranges.values():
            valid = known[start:stop, role]>0
            if role<4:
                for positive in (False, True):
                    ids = np.flatnonzero(valid & ((y[start:stop, role]>.5)==positive))+start
                    multiplier = 4. if role>0 and not positive else 1.
                    if len(ids):
                        weights[ids, role] = multiplier/len(ids)
            else:
                ids = np.flatnonzero(valid)+start
                if len(ids):
                    weights[ids, role] = priority[ids]/priority[ids].sum()
        total = weights[:, role].sum()
        if total<=0:
            raise ValueError('A required known event role is missing')
        weights[:, role] *= len(y)/total
    return weights


def evaluate(model, mean, std, names, root, runtime=False, poolings=('mean', 'or')):
    root.mkdir(); pools = {p: {} for p in poolings}
    for p in pools:
        (root/p).mkdir()
    for name in names:
        if runtime:
            manifest = json.loads((BASE/name/'manifest.json').read_text('utf8'))
            piece = load_piece(manifest['source_cache_root'], verify=True)
            sequences, data = build_inputs(read_xyz_pcd(piece['record']['input'])*1000.)
        else:
            data, _, _, piece = load_cached(name); sequences = piece['sequences']
        fields = predict_fields(model, data, sequences, mean, std)
        truth = [w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
        for pooling in pools:
            foregrounds, events = assemble_events(data, sequences, fields, pooling)
            predictions = decode_event_sets(sequences, foregrounds, events)
            score = score_children(predictions, truth); pools[pooling][name] = score
            dump(root/pooling/(name+'.json'), dict(score=score, predictions=predictions, event_sets=events,
                full_contract_pass=False))
            np.savez_compressed(root/pooling/(name+'_foregrounds.npz'), foregrounds=np.concatenate(foregrounds),
                lengths=np.asarray([len(a) for a in foregrounds]))
            print('BOUNDED_DEEP_SCORE', root.name, pooling, name, score['valid_children'], '/',
                score['truth_count'], 'FP', score['false_predictions'], flush=True)
        np.savez_compressed(root/(name+'_raw_fields.npz'), raw=np.concatenate(fields),
            lengths=np.asarray([len(a) for a in fields]))
        del data, piece, fields, foregrounds, events, predictions; gc.collect()
    totals = {p: aggregate(scores) for p, scores in pools.items()}
    for p, total in totals.items():
        dump(root/p/'summary.json', total)
    return totals


def main():
    if OUT.exists():
        raise ValueError('Preserve the bounded deep training version')
    data_receipt = json.loads((DATA/'complete.json').read_text('utf8'))
    data_config = json.loads((DATA/'config.json').read_text('utf8'))
    representation = json.loads((PROOF/'complete.json').read_text('utf8'))
    confidence = json.loads((CONFIDENCE/'complete.json').read_text('utf8'))
    confidence_protocol = json.loads((CONFIDENCE/'protocol.json').read_text('utf8'))
    if not data_receipt['all_original_samples_and_event_positives_preserved']:
        raise ValueError('Original rows and complete event positives must be preserved')
    if (not representation['all_original_1221_and_analytic_cases_represented'] or
            not confidence['all_1221_teacher_and_analytic_targets_represented']):
        raise ValueError('Full bounded and confidence representation proofs required')
    for receipt in (data_config, representation, confidence_protocol):
        for path, digest in receipt['source_sha256'].items():
            if sha256(path)!=digest:
                raise ValueError('A verified representation source changed: '+path)
    old = json.loads(Path('weld_pipeline/runs/deep_physical_events_170_v3/config.json').read_text('utf8'))
    sources = list(dict.fromkeys([__file__, 'weld_pipeline/neural_bounded_events.py',
        'weld_pipeline/bounded_event_phase_v2.py', 'weld_pipeline/bounded_event_phase.py',
        'weld_pipeline/continuous_center_modes_v2.py', 'weld_pipeline/train_neural_physical_events_v3.py',
        *old['source_sha256'].keys()]))
    config = dict(version=VERSION, seed=175, feature_count=FEATURE_COUNT, width=512, residual_blocks=3,
        independent_classifier_and_regressor_parameters=True, independent_optimizers_and_gradient_clipping=True,
        train=data_config['sources'], held=old['held'], epochs=360, batch_size=4096,
        learning_rate=3e-4, final_learning_rate=3e-5, weight_decay=1e-5, gradient_clip_norm=10.,
        classification_loss='source/class balanced BCE; support negative class mass x4; foreground x1',
        regression_loss='source balanced millimeter SmoothL1 beta1; previous large error positives weight x2',
        bounded_residual_is_generic_physical_cell_not_lookup=True, cell_mm=64., radius_mm=16.,
        actual_XYZ_and_query_phases_are_ordinary_shared_neural_inputs=True,
        source_ID_names_fixed_coordinate_lookup_and_reference_runtime_forbidden=True,
        pooling_trials=['mean', 'or'], selection='eligible FIT max TP then min FP then lexical pooling',
        eligibility='complete 1221 horizontal straight FIT: TP/G strictly >.95 and FP/G <=.05; capacity gate only',
        held_is_already_read_development_not_blind=True, no_held_evaluation_if_capacity_gate_fails=True,
        source_sha256={p: sha256(p) for p in sources}, data_complete_sha256=sha256(DATA/'complete.json'),
        bounded_proof_sha256=sha256(PROOF/'complete.json'), confidence_proof_sha256=sha256(CONFIDENCE/'complete.json'),
        full_contract_pass=False)
    OUT.mkdir(parents=True); dump(OUT/'config.json', config)
    snapshot = OUT/'source_snapshot'; snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot/Path(path).name)
    xs=[]; ys=[]; ks=[]; ps=[]; ranges={}; count=0
    for name in config['train']:
        folder=DATA/name
        if sha256(folder/'manifest.json')!=data_receipt['manifests'][name]:
            raise ValueError('Prepared hard-sample manifest changed')
        manifest=json.loads((folder/'manifest.json').read_text('utf8'))
        for file, digest in manifest['files'].items():
            if sha256(folder/file)!=digest:
                raise ValueError('Prepared hard-sample rows changed')
        x=np.load(folder/'features.npy', allow_pickle=False); xs.append(x)
        ys.append(np.load(folder/'targets.npy', allow_pickle=False)); ks.append(np.load(folder/'known.npy', allow_pickle=False))
        ps.append(np.load(folder/'positive_priority.npy', allow_pickle=False))
        ranges[name]=(count, count+len(x)); count+=len(x)
    x=np.concatenate(xs); y=np.concatenate(ys); known=np.concatenate(ks); priority=np.concatenate(ps)
    del xs,ys,ks,ps; gc.collect()
    if count!=data_receipt['samples'] or x.shape!=(count, FEATURE_COUNT):
        raise ValueError('Complete prepared sample table required')
    weights=source_weights(y, known, priority, ranges)
    mean=x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std=np.maximum(x.std(axis=0, dtype=np.float64), 1e-3).astype(np.float32)
    random.seed(175); np.random.seed(175); torch.manual_seed(175); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.manual_seed_all(175)
    model=BoundedEventNet().to(device); initial_digest=state_digest(model.state_dict())
    values=torch.from_numpy((x-mean)/std).to(device); target=torch.from_numpy(y).to(device); weight=torch.from_numpy(weights).to(device)
    del x,known,weights,priority; gc.collect()
    dump(OUT/'runtime.json', dict(device=str(device), gpu_name=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        torch_version=torch.__version__, training_rows=count, original_rows=data_receipt['original_samples'],
        added_unique_rows=data_receipt['added_samples'], initial_state_sha256=initial_digest))
    optimizers=[torch.optim.AdamW(field.parameters(), lr=3e-4, weight_decay=1e-5) for field in (model.classifier, model.regressor)]
    schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=360, eta_min=3e-5) for opt in optimizers]
    rng=np.random.default_rng(175); history=[]; steps_total=0; began=time.perf_counter()
    dump(OUT/'train_progress.json', dict(epoch=0, completed_batch_updates=0, optimizer_updates_each_component=0, state='running'))
    for epoch in range(1, 361):
        model.train(); permutation=rng.permutation(count); losses=np.zeros(2); steps=0
        for start in range(0, count, 4096):
            ids=torch.from_numpy(permutation[start:start+4096]).to(device=device, dtype=torch.long)
            raw=model(values[ids]); yy=target[ids]; ww=weight[ids]
            binary=(F.binary_cross_entropy_with_logits(raw[:, :4], yy[:, :4].clamp(0.,1.), reduction='none')*ww[:, :4]).mean()
            regression=(F.smooth_l1_loss(raw[:, 4:7]*16., yy[:, 4:7]*16., reduction='none', beta=1.)*ww[:, 4:7]).mean()
            loss=binary+regression
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite separate bounded training loss')
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for field, optimizer in zip((model.classifier, model.regressor), optimizers):
                torch.nn.utils.clip_grad_norm_(field.parameters(), 10.); optimizer.step()
            losses += [float(binary.detach()), float(regression.detach())]; steps+=1; steps_total+=1
        for scheduler in schedulers:
            scheduler.step()
        history.append(dict(epoch=epoch, classification_loss=float(losses[0]/steps),
            regression_mm_loss=float(losses[1]/steps), batch_updates=steps, learning_rate=optimizers[0].param_groups[0]['lr']))
        if epoch==1 or epoch%20==0:
            print('BOUNDED_DEEP_TRAINING', epoch, 'CLASS', losses[0]/steps, 'REG_MM', losses[1]/steps, 'STEPS_EACH', steps_total, flush=True)
            dump(OUT/'train_progress.json', dict(epoch=epoch, completed_batch_updates=steps_total,
                optimizer_updates_each_component=steps_total, state='running'))
        if epoch%40==0:
            torch.save(dict(state_dict=model.state_dict(), optimizers=[o.state_dict() for o in optimizers],
                schedulers=[s.state_dict() for s in schedulers], rng_state=rng.bit_generator.state,
                epoch=epoch, completed_batch_updates=steps_total, mean=mean, std=std, config=config), OUT/'latest.pt')
            dump(OUT/'history.json', history)
    checkpoint=OUT/'final.pt'
    torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std, epoch=360,
        completed_batch_updates=steps_total, config=config), checkpoint)
    dump(OUT/'train_progress.json', dict(epoch=360, completed_batch_updates=steps_total,
        optimizer_updates_each_component=steps_total, state='FIT_evaluation'))
    del values,target,weight; torch.cuda.empty_cache() if device.type=='cuda' else None
    fit=evaluate(model, mean, std, config['train'], OUT/'FIT')
    eligible=[p for p,s in fit.items() if s['truth_count']==1221 and s['valid_children']/1221>.95 and s['false_predictions']/1221<=.05]
    selected=sorted(eligible, key=lambda p:(-fit[p]['valid_children'], fit[p]['false_predictions'], p))[0] if eligible else None
    dump(OUT/'selection_before_held.json', dict(selected_pooling=selected, eligibility=config['eligibility'],
        checkpoint_sha256=sha256(checkpoint), FIT_summaries={p:sha256(OUT/'FIT'/p/'summary.json') for p in fit}))
    held=None
    if selected:
        # Evaluation uses fresh raw PCD; selection remains fixed for every source.
        held_all=evaluate(model, mean, std, config['held'], OUT/'DEVELOPMENT', runtime=True, poolings=(selected,))
        held=held_all[selected]
    dump(OUT/'complete.json', dict(config_sha256=sha256(OUT/'config.json'), checkpoint_sha256=sha256(checkpoint),
        initial_state_sha256=initial_digest, actual_trained_shared_deep_models=1, separate_learned_fields=2,
        actual_epochs=360, batch_updates=steps_total, actual_optimizer_updates_each_component=steps_total,
        actual_optimizer_updates_total=2*steps_total, training_samples=count, FIT=fit, selected_before_development=selected,
        development=held, development_read_for_this_trial=held is not None, observed_read_for_tuning=False,
        default_predictor_replaced=False, elapsed_seconds=time.perf_counter()-began, full_contract_pass=False))
    dump(OUT/'train_progress.json', dict(epoch=360, completed_batch_updates=steps_total,
        optimizer_updates_each_component=steps_total, state='complete'))
    print('BOUNDED_DEEP_COMPLETE', {p:(s['valid_children'], s['false_predictions']) for p,s in fit.items()}, 'SELECTED', selected, flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists() and not (OUT/'complete.json').exists():
            progress=json.loads((OUT/'train_progress.json').read_text('utf8')) if (OUT/'train_progress.json').exists() else {}
            dump(OUT/'failure.json', dict(exception_type=type(error).__name__, exception=str(error),
                last_recorded_training_progress=progress, full_contract_pass=False))
        raise
