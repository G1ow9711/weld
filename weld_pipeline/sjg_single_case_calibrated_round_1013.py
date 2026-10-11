"""Fast single-TRAIN feasibility round; fixed final checkpoint then free score.

The full original pilot and contract remain required. Failure here prevents
another expensive full-cohort scan. This is not an independent test result.
"""
import argparse
import gc
import subprocess
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, utc, check_resources
from weld_pipeline.sjg_local_training_data_918 import load_case
from weld_pipeline.sjg_local_graph_data_916 import queries, CanonicalQueryTree
from weld_pipeline.sjg_stratified_event_queries_1010 import event_layout, draw, controls as sampling_controls
from weld_pipeline.sjg_probability_corrected_queries_1007 import labels_for
from weld_pipeline.sjg_population_event_loss_1008 import population_statistics, loss_for, controls as loss_controls
from weld_pipeline.sjg_local_event_train_930 import as_tensors, save_checkpoint
from weld_pipeline.sjg_surface_material_joint_model_985 import SurfaceMaterialJointModel
from weld_pipeline.sjg_local_event_score_932 import geometric_match


def calibration(statistics):
    known = np.asarray(statistics['known_fraction'])
    prior = np.asarray(statistics['conditional_positive_prior'])
    epsilon = 1. / max(2, statistics['total_queries'])
    clipped = np.clip(prior, epsilon, 1-epsilon)
    entropy = -(clipped*np.log(clipped)+(1-clipped)*np.log1p(-clipped))
    # Constant per-head scaling leaves each population-posterior optimum
    # unchanged. It prevents a rare endpoint task from losing its gradient
    # merely because it has vastly fewer positives than the occupancy task.
    active = known > 0
    scale = np.zeros(5)
    scale[active] = known.sum() / (active.sum()*known[active]*entropy[active])
    return dict(statistics, event_head_scale=scale.tolist(), clipped_initial_prior=clipped.tolist(),
                initialization_logit=(np.log(clipped)-np.log1p(-clipped)).tolist(),
                scaling='equal conditional heads, divided by fixed TRAIN prior entropy')


def calibration_controls():
    from torch.nn import functional as F
    target = torch.tensor([0., 0., .1, .2, .7], dtype=torch.float64)
    prior = target.mean()
    bias = torch.logit(prior).detach().requires_grad_()
    entropy = -(prior*prior.log()+(1-prior)*(1-prior).log())
    objective = F.binary_cross_entropy_with_logits(bias.expand_as(target), target)/entropy
    gradient = float(torch.autograd.grad(objective, bias)[0])
    assert abs(gradient) < 1e-12
    no_positive = calibration(dict(known_fraction=[1, 1, 1, 1, 0], conditional_positive_prior=[0, 0, 0, 0, 0],
                                   total_queries=100, offset_known_fraction=[0, 0]))
    assert np.isfinite(no_positive['event_head_scale']).all() and no_positive['event_head_scale'][4] == 0
    return dict(entropy_scaling_preserves_population_optimum=True, optimum_gradient=gradient,
                zero_positive_and_unknown_heads_finite=True, optimizer_updates=0)


def main(default_prepared_run=None, default_prediction_run='run192', default_anchored=False, default_hard_negative_run=None):
    parser = argparse.ArgumentParser();parser.add_argument('--run', required=True)
    args = parser.parse_args();dest = OUT/args.run;dest.mkdir(exist_ok=False)
    snapshots = dest/'source_snapshot';snapshots.mkdir()
    for name in ('sjg_single_case_calibrated_round_1013.py', 'sjg_boundary_anchored_model_1015.py',
                 'sjg_probability_corrected_queries_1007.py', 'sjg_population_event_loss_1008.py',
                 'sjg_stratified_event_queries_1010.py', 'sjg_local_event_infer_931.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name, snapshots/name)
    torch.set_num_threads(2);torch.manual_seed(930);rng = np.random.default_rng(930)
    key = '21';epochs = 32;batch = 256;count = 2048;updates = 0;history = []
    family = 'surface_material_anchored_event_1015' if default_anchored else 'surface_material_joint_985'
    protocol = dict(stage='SINGLE_TRAIN_CALIBRATED_EVENT_FEASIBILITY', source_key=key, source_G=1,
        full_pilot_original_G=496, supplemental_TRAIN_G=4, epochs=epochs, queries_per_epoch=count,
        expected_optimizer_updates=epochs*count//batch, model_family=family,
        backbone='run162_frozen_features_run166', proposals='run185', graph_data='run186',
        sampler='1010_endpoints_interiors_uniform_population', loss='1008_exact_population_importance',
        objective_scaling='fixed_TRAIN_conditional_head_entropy', event_positive_weights=[1]*5,
        initial_trunk='run187_initial.pt', event_output_weights_zero_and_bias_TRAIN_prior=True,
        learning_rate=.001, weight_decay=.0001, gradient_clip=5., seed=930,
        fixed_final_epoch_no_test_selection=True, inference_grid_mm=2, threshold=.5,
        quick_prediction_run=default_prediction_run, prepared_run=default_prepared_run, independent_test=False, full_contract_pass=False,
        hard_negative_prediction_run=default_hard_negative_run,
        external_consultation_sent=False, source_sha256=digest(__file__), start_utc=utc())
    write(dest/'registered_protocol.json', protocol)
    try:
        review = read(OUT/'run190/local_early_stop_review.json')
        assert review['status'] == 'INTENTIONAL_EARLY_STOP_AFTER_VERIFIED_TRAIN_FAILURE'
        assert not (OUT/default_prediction_run).exists()
        write(dest/'controls.json', dict(sampler=sampling_controls(), loss=loss_controls(), calibration=calibration_controls()))
        if default_anchored:
            from weld_pipeline.sjg_boundary_anchored_model_1015 import BoundaryAnchoredModel, adapt_initial_state, boundary_features, controls as anchor_controls
            write(dest/'anchor_controls.json', anchor_controls())
            protocol['boundary_anchor_scales_mm'] = [1, 2, 4, 8, 16, 32, 64, 128]
            protocol['anchor_source_sha256'] = digest(ROOT/'weld_pipeline/sjg_boundary_anchored_model_1015.py')
            protocol['new_anchor_input_weights_initialized_zero'] = True
            write(dest/'registered_protocol.json', protocol)
        check_resources()
        case = next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
        assert case['role']=='TRAIN'
        data = OUT/'run186';proposals = OUT/'run185';features = OUT/'run166'
        datasets = [];query_records = []
        if default_prepared_run:
            prepared = OUT/default_prepared_run
            prior_protocol = read(prepared/'registered_protocol.json')
            assert prior_protocol['source_key']==key and prior_protocol['epochs']==epochs
            assert read(prepared/'round_failure.json')['optimizer_updates']==0
            statistics = read(prepared/'TRAIN_population_statistics.json')
            previous = read(prepared/'TRAIN_query_manifest.json')
            assert previous['supervision_sha256']==digest(data/('source_'+key+'_supervision.json'))
            assert len(previous['rows'])==epochs
            for row in previous['rows']:
                path = prepared/row['file'];assert digest(path)==row['sha256']
                datasets.append(path)
                query_records.append(dict(row, file='../'+default_prepared_run+'/'+row['file']))
            print('REUSE_VERIFIED_PREPARED_QUERIES', len(datasets), 'OPTIMIZER_UPDATES', 0, flush=True)
        else:
            xyz, point, faces, profiles, _ = load_case(case, proposals, features)
            supervision = read(data/('source_'+key+'_supervision.json'))
            assert supervision['original_G']==1 and not supervision['missing']
            layout = event_layout(profiles, supervision)
            if default_hard_negative_run:
                from weld_pipeline.sjg_train_hard_query_mining_1017 import extend_layout
                assert read(OUT/'run195/local_round_review.json')['status']=='SINGLE_TRAIN_FEASIBILITY_FAILED'
                layout = extend_layout(layout, profiles, supervision, default_hard_negative_run, key, dest)
                shutil.copy2(ROOT/'weld_pipeline/sjg_train_hard_query_mining_1017.py', snapshots/'sjg_train_hard_query_mining_1017.py')
            statistics = calibration(population_statistics(profiles, supervision, layout))
            tree = CanonicalQueryTree(xyz)
            for epoch in range(epochs):
                sample = draw(layout, 101300+epoch, count=count)
                labels = labels_for(profiles, supervision, sample['nodes'], sample['q_mm'])
                patch, query = queries(profiles, sample['nodes'], sample['q_mm'], xyz, point, tree)
                path = dest/('epoch_'+str(epoch+1).zfill(2)+'_queries.npz')
                arrays = dict(nodes=sample['nodes'], q_mm=sample['q_mm'], patch=patch, query=query,
                              importance_weight=sample['importance_weight'], sampling_probability=sample['sampling_probability'], **labels)
                np.savez_compressed(path, **arrays)
                query_records.append(dict(epoch=epoch+1, file=path.name, sha256=digest(path)))
                datasets.append(path)
                print('PREPARE_SINGLE_TRAIN', epoch+1, '/', epochs, flush=True)
            del xyz, point, faces, profiles, tree, layout, sample, patch, query, labels, arrays;gc.collect()
        write(dest/'TRAIN_population_statistics.json', statistics)
        write(dest/'TRAIN_query_manifest.json', dict(rows=query_records, source_key=key,
            supervision_sha256=digest(data/('source_'+key+'_supervision.json')), optimizer_updates=0))
        binding = next(r for r in read(data/'all_graphs_frozen_before_labels.json')['cases'] if r['source_key']==key)
        assert digest(data/binding['file'])==binding['sha256']
        graph = as_tensors(data/binding['file'])
        model = BoundaryAnchoredModel() if default_anchored else SurfaceMaterialJointModel()
        state = torch.load(OUT/'run187/initial.pt', map_location='cpu')
        model.load_state_dict(adapt_initial_state(state['state_dict']) if default_anchored else state['state_dict']);del state
        with torch.no_grad():
            model.event[-1].weight[:5].zero_()
            model.event[-1].bias[:5].copy_(torch.tensor(statistics['initialization_logit']))
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        save_checkpoint(dict(state_dict=initial, model_family=family, updates=0), dest/'initial.pt')
        if default_anchored:
            anchor_path = proposals/('source_'+key)/'profiles.json'
            assert digest(anchor_path)==read(anchor_path.parent/'input_freeze.json')['files']['profiles.json']
            anchor_profiles = read(anchor_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        start = time.perf_counter()
        for epoch, path in enumerate(datasets, 1):
            assert digest(path)==query_records[epoch-1]['sha256']
            values = as_tensors(path);order = rng.permutation(count);losses = [];model.train()
            if default_anchored:
                assert values['query'].shape[1]==10
                anchors = boundary_features(anchor_profiles, values['nodes'].numpy(), values['q_mm'].numpy())
                values['query'] = torch.cat([values['query'], torch.from_numpy(anchors)], dim=1)
            for begin in range(0, count, batch):
                ids = torch.as_tensor(order[begin:begin+batch], dtype=torch.long)
                inputs = dict(graph, **{k: values[k][ids] for k in ('nodes', 'patch', 'query')})
                labels = {k: values[k][ids] for k in ('target', 'target_mask', 'offset', 'offset_mask')}
                optimizer.zero_grad(set_to_none=True)
                logits, offset = model(inputs)
                loss, event, regression = loss_for(logits, offset, labels, values['importance_weight'][ids], statistics)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite single-case loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step();updates += 1;losses.append(float(loss.detach()))
            row = dict(epoch=epoch, total_epochs=epochs, updates=updates, expected_updates=epochs*8,
                       mean_loss=float(np.mean(losses)), elapsed_seconds=time.perf_counter()-start)
            history.append(row);write(dest/'progress.json', row)
            print('EPOCH', epoch, '/', epochs, 'UPDATES', updates, '/', epochs*8, 'LOSS', round(row['mean_loss'],6), flush=True)
            del values, inputs, labels, logits, offset, loss, event, regression;gc.collect();check_resources()
        assert updates==epochs*8
        delta = sum(float((v-initial[k]).double().square().sum()) for k,v in model.state_dict().items())**.5
        assert delta>0
        save_checkpoint(dict(state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            updates=updates, epochs=epochs, model_family=family,
            data_manifest_sha256=digest(data/'training_manifest.json'), single_TRAIN_source=key,
            actual_query_manifest_sha256=digest(dest/'TRAIN_query_manifest.json')), dest/'final.pt')
        write(dest/'history.json', history)
        write(dest/'receipt.json', dict(status='LOCAL_TYPED_EVENT_TRAINING_COMPLETE', single_TRAIN_source=key,
            epochs=epochs, optimizer_updates=updates, parameter_change_l2=delta, final_sha256=digest(dest/'final.pt'),
            training_seconds=time.perf_counter()-start, full_contract_pass=False, end_utc=utc()))
        del model, optimizer, initial, graph;gc.collect()
        print('SINGLE_TRAIN_COMPLETE_START_FREE_TEST', key, flush=True)
        child = subprocess.Popen([sys.executable, '-u', '-m', 'weld_pipeline.sjg_local_event_infer_931',
            '--run', default_prediction_run, '--data-run', 'run186', '--training-run', args.run,
            '--proposal-run', 'run185', '--feature-run', 'run166', '--sources', key], cwd=str(ROOT))
        write(dest/'quick_test_process.json', dict(pid=child.pid, run=default_prediction_run, source=key, utc=utc()))
        if child.wait():raise RuntimeError('Single-case free prediction failed')
        frozen = read(OUT/default_prediction_run/('source_'+key)/'input_output_freeze.json')
        prediction_file = OUT/default_prediction_run/('source_'+key)/'predictions.json'
        assert digest(prediction_file)==frozen['files']['predictions.json']
        assert digest(OUT/case['target_file'])==case['target_sha256']
        score = geometric_match(read(prediction_file), read(OUT/case['target_file']))
        write(dest/'quick_geometry_score.json', score)
        passed = score['TP_over_G']>.95 and score['FP_over_G']<=.05
        write(dest/'local_round_review.json', dict(status='SINGLE_TRAIN_FEASIBILITY_PASSED' if passed else 'SINGLE_TRAIN_FEASIBILITY_FAILED',
            score=score, inference_seconds=frozen['elapsed_seconds'], full_contract_pass=False,
            no_auto_expansion=True, external_consultation_sent=False,
            next_action='Review single-case error before any further training; full pilot and all contract fields remain required.', end_utc=utc()))
        print('IMMEDIATE_GEOMETRY_RESULT', key, 'TP', score['TP'], '/', score['G'], 'FP', score['FP'],
              'SINGLE_CASE_PASS', passed, 'FULL_CONTRACT_PENDING', flush=True)
    except Exception:
        write(dest/'round_failure.json', dict(status='FAILED', optimizer_updates=updates, error=traceback.format_exc(), end_utc=utc()))
        raise


if __name__=='__main__':main()
