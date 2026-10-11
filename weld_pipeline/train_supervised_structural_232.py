"""Registered supervised-score training; original source-C budget stays fixed.

No Good/Bad optimization is used in training. Evaluation uses the unchanged
complete, GT-free Runtime solver. Full-contract acceptance is never inferred
from this horizontal-line capacity stage.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

from .source_c_execution_229 import SourceCursor
from .source_c_supervised_232 import complete_supervised_source, ZERO_GRADIENT_POLICY, validate_actual_scalar_loss
from .source_checkpoint_232 import (
    commit_checkpoint, load_checkpoint, finite_state, gradient_receipt, synchronize_model,
    commit_initial_boundary, restore_initial_boundary,
)
from .structural_score_targets_232 import LOSS_KIND
from .supervised_structural_loss_232 import (
    file_sha, object_sha, load_source_targets, source_path_targets,
    bind_runtime_graph, supervised_path_loss, analytic_empty_path,
)
from .train_recovered_source_c_229 import kernel_state

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'weld_pipeline/reports/supervised_structural_score_232_20261002'
VERSION = 'REGISTERED_COMPLETE_SOURCE_C_SUPERVISED_SCORE_RUNNER_232_v2_233_GUARDS'


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def validate_release(path, operation, device, resume=None):
    """All file/method/resource checks precede Torch, CUDA and output creation."""
    release = read(path)
    statuses = dict(preflight=('AUTHORIZED_NO_UPDATE_SUPERVISED_PREFLIGHT_232',),
                    run=('AUTHORIZED_FIRST_COUNTED_SOURCE_232', 'AUTHORIZED_REMAINING_SOURCE_C_232'))
    if release.get('status') not in statuses[operation] or release.get('released_by') != 'root' or release.get('authorized') is not True:
        raise PermissionError('Separate root release for the registered232 method required')
    artifact = release['config_artifact']
    if file_sha(artifact['path']) != artifact['sha256']:
        raise ValueError('Frozen supervised config file changed')
    config = read(artifact['path'])
    if (config['version'] != VERSION or config['loss_kind'] != LOSS_KIND or config['device'] != device
            or config['zero_gradient_policy'] != ZERO_GRADIENT_POLICY
            or config['budget'] != dict(seed=20261002, source_count=16, epochs=80, source_slots=1280)
            or config['training_native_solver_calls_per_path'] != 0
            or config['evaluation_every_actual_updates'] != 1000 or config['evaluation_final'] is not True):
        raise ValueError('Registered method, device, original budget and evaluation schedule required')
    if object_sha(config) != release['config_content_sha256']:
        raise ValueError('Frozen config content binding differs')
    for name, expected in config['files_sha256'].items():
        if file_sha(ROOT / name) != expected:
            raise ValueError('Release-bound source/evidence changed: ' + name)
    original = read(config['original_config_artifact']['path'])
    if file_sha(config['original_config_artifact']['path']) != config['original_config_artifact']['sha256']:
        raise ValueError('Original source-C configuration changed')
    if (config['sources'] != original['sources'] or config['learning_rate'] != original['learning_rate']
            or config['weight_decay'] != original['weight_decay'] or config['chunk_size'] != original['chunk_size']):
        raise ValueError('Original complete source order/data/AdamW/forward chunk changed')
    binding = read(config['initial_binding_artifact']['path'])
    if (file_sha(config['initial_binding_artifact']['path']) != config['initial_binding_artifact']['sha256']
            or file_sha(binding['state_file']) != binding['state_sha256']
            or file_sha(binding['score_file']) != binding['score_sha256']):
        raise ValueError('Original initialized state/scores changed')
    lineage = read(config['lineage_receipt_artifact']['path'])
    formula = read(config['formula_receipt_artifact']['path'])
    if (lineage['status'] != 'PASS_ACTUAL_FULL16_LINEAGE_ALIAS_COMPETITION_CHILD_COVERAGE_AND_ORDERED_MANIFEST'
            or lineage['totals']['truth_children'] != 1221 or lineage['totals']['paths'] != 2011
            or formula['status'] != 'PASS_FULL16_COMPLETE_TARGET_BALANCED_FORMULA_AND_TRANSACTION_FAULTS'):
        raise PermissionError('Actual complete source/target and registered formula evidence required')
    owner = read(config['resource_owner_path'])
    resource = kernel_state(owner)
    if resource['live']:
        raise PermissionError('Same227 native process is active; no second high-memory job: ' + str(resource))
    if operation == 'run':
        for name, required_status in (
            ('actual_no_update_preflight', 'PASS_ACTUAL197_FULL_MAXPATH_SUPERVISED_FORWARD_BACKWARD_NO_UPDATE'),
            ('checkpoint_engineering', 'PASS_CPU_CHECKPOINT_PARAMETER_RNG_BINDING_AND_CORRUPTION_REJECTION_232'),
        ):
            evidence = release.get(name)
            if not evidence or file_sha(evidence['path']) != evidence['sha256']:
                raise PermissionError('Actual root-bound gate required: ' + name)
            proof = read(evidence['path'])
            if proof['status'] != required_status or proof['actual_weld_optimizer_updates'] != 0:
                raise PermissionError('Gate status or model-update accounting differs: ' + name)
            if name == 'checkpoint_engineering' and proof['checkpoint_module_sha256'] != file_sha(ROOT/'weld_pipeline/source_checkpoint_232.py'):
                raise PermissionError('CPU checkpoint gate does not cover the current implementation')
        actual = read(release['actual_no_update_preflight']['path'])
        if (actual['config_content_sha256'] != object_sha(config) or actual['actual_model_device'] != device
                or actual['initial_model_state_hash'] != binding['model_state_hash']
                or actual['finite_gradients'] is not True or actual['all_candidates_and_pairs_retained'] is not True
                or actual['all_supervised_branches_connected'] is not True
                or actual['zero_boundary_restoration']['model_buffers_unchanged'] is not True
                or actual['zero_boundary_restoration']['actual_cuda_rng_restored'] is not True):
            raise PermissionError('Current232 real CUDA consumer gate differs')
        zero_boundary = actual['pre_forward_zero_boundary']
        if file_sha(zero_boundary['path']) != zero_boundary['sha256']:
            raise PermissionError('Committed pre-forward zero boundary changed')
        if release['status'] == 'AUTHORIZED_REMAINING_SOURCE_C_232':
            evidence = release.get('first_complete_source_proof')
            if not evidence or file_sha(evidence['path']) != evidence['sha256']:
                raise PermissionError('Actual first counted source/CUDA checkpoint-load proof required')
            proof = read(evidence['path'])
            if (proof['status'] != 'PASS_FIRST_COUNTED_COMPLETE_SOURCE_AND_ACTUAL_CHECKPOINT_LOAD_232'
                    or proof['config_content_sha256'] != object_sha(config)
                    or proof['cursor']['complete_sources'] != 1
                    or proof['uncounted_real_model_updates'] != 0
                    or proof['checkpoint_loaded']['actual_model_devices'] != [device]
                    or proof['checkpoint_loaded']['actual_cuda_rng_restored'] is not True):
                raise PermissionError('Real first-source or CUDA resume proof differs')
    if resume is not None:
        if operation == 'preflight' or release['status'] == 'AUTHORIZED_FIRST_COUNTED_SOURCE_232':
            raise PermissionError('Original no-update/first-source phases do not resume another checkpoint')
        checkpoint = release.get('resume_checkpoint_commit')
        if not checkpoint or Path(checkpoint['path']).resolve() != Path(resume).resolve() or file_sha(resume) != checkpoint['sha256']:
            raise PermissionError('Exact committed PT/metadata descriptor required for resume')
        if release.get('previous_round_requires_consultation') is True:
            evidence = release.get('completed_round_consultation')
            if not evidence or file_sha(evidence['path']) != evidence['sha256'] or not read(evidence['path'])['generation_stopped']:
                raise PermissionError('Completed current-round Pro consultation required before more updates')
        elif release.get('previous_round_requires_consultation') is not False:
            raise PermissionError('Explicit previous-round consultation state required')
    elif operation == 'run' and release['status'] != 'AUTHORIZED_FIRST_COUNTED_SOURCE_232':
        raise PermissionError('Remaining source-C slots resume the committed first source')
    return config, release, binding, resource


def initialize_original_model(config, binding, device):
    import torch
    from . import train_boundary_incidence_f1_214 as original
    original.seed_runtime(device)
    with Path(binding['state_file']).open('rb') as stream:
        saved = torch.load(stream, map_location='cpu')
    inherited = read(config['original_config_artifact']['path'])
    if saved['optimizer_steps'] != 0 or saved['config'] != inherited:
        raise ValueError('Original pre-update checkpoint required')
    mean, std = original.load_frozen175_normalizer(config['normalizer_path'])
    model = original.BoundaryIncidenceNet(mean, std)
    model.load_state_dict(saved['state_dict'], strict=True)
    if original.model_sha(model) != binding['model_state_hash']:
        raise ValueError('Original initialized model bytes differ')
    return model.to(device).train(), original


def supervised_branch_connections(model, scores):
    """Check every shared/body and corresponding head parameter, allowing zero."""
    import torch
    shared = ('node_encoder.', 'messages.', 'updates.', 'norms.')
    heads = dict(u='unary.', a='child.', d='duplicate.')
    named = tuple(model.named_parameters())
    records = {}
    for name, head in heads.items():
        output = getattr(scores, name)
        if not len(output) or not output.requires_grad:
            raise ValueError('Actual nonempty supervised output required: ' + name)
        expected = [key for key, value in named if key.startswith(shared + (head,))]
        if not expected:
            raise ValueError('Expected shared/body and head parameters absent: ' + name)
        derivatives = torch.autograd.grad(output.sum(), [value for _, value in named],
                                          allow_unused=True, retain_graph=True)
        details = []
        for (key, parameter), derivative in zip(named, derivatives):
            required = key in expected
            if required and derivative is None:
                raise ValueError('Disconnected supervised output/parameter: ' + name + '/' + key)
            if derivative is not None and not bool(torch.isfinite(derivative).all()):
                raise ValueError('Nonfinite branch connection derivative: ' + name + '/' + key)
            details.append(dict(parameter=key, expected_connection=required,
                grad_is_none=derivative is None,
                strict_zero=None if derivative is None else not bool((derivative != 0).any()),
                shape=list(parameter.shape)))
        records[name] = dict(expected_parameters=expected, all_expected_connected=True,
                             output_count=len(output), parameter_details=details)
        del derivatives
    return records


def no_update_preflight(args, config, binding, resource):
    import numpy as np
    import torch
    from .durable_artifacts_223 import AtomicWriter
    from .restore_incidence_snapshot_224 import same_bytes
    writer = AtomicWriter()
    model, original = initialize_original_model(config, binding, args.device)
    before = original.model_sha(model)
    zero_boundary = commit_initial_boundary(args.output/'zero_boundary', model, SourceCursor(), config, writer=writer)
    writer.write_json(args.output/'preflight_progress.json', dict(stage='ZERO_BOUNDARY_COMMITTED_BEFORE_FORWARD',
        pre_forward_zero_boundary=zero_boundary, actual_weld_optimizer_updates=0))
    registered = config['actual_maxpath']
    source_evidence = next(v for v in config['source_target_evidence'] if v['source'] == registered['source'])
    targets = load_source_targets(source_evidence['original_spec'], source_evidence)
    spec = source_evidence['original_spec']
    complete_graph = original.load_runtime_graph(spec['artifact_directory'], spec['source'], spec['expected_sha256'])
    if complete_graph.complete_hashes() != targets.manifest['runtime_hashes']:
        raise ValueError('Real complete runtime source differs from registered targets')
    path_id, graph, _, ordinals = next(v for v in original.complete_path_views(complete_graph)
                                       if v[0] == registered['path_id'])
    del complete_graph
    path = source_path_targets(targets, path_id)
    path = bind_runtime_graph(path, graph, ordinals)
    counts = dict(nodes=len(graph.nodes), children=graph.pair_count('child'),
                  duplicate=graph.pair_count('duplicate'), message=graph.pair_count('message'))
    if counts != registered['complete_counts']:
        raise ValueError('Actual frozen maximum complete graph counts differ')
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    inputs, relations = original.query_graph_195_inputs(graph)
    writer.write_json(args.output/'preflight_progress.json', dict(stage='ACTUAL_COMPLETE197_FORWARD',
        complete_counts=counts, actual_weld_optimizer_updates=0))
    scores = model(inputs, relations, chunk_size=config['chunk_size'])
    detached = original.solver_199_scores(scores)
    if detached.graph_fingerprint != binding['graph_content_hash']:
        raise ValueError('Original initialized complete graph fingerprint differs')
    checks = []
    with np.load(binding['score_file'], allow_pickle=False) as saved:
        for name in ('u', 'a', 'd'):
            checks.append(same_bytes(name+'_native', getattr(scores, name).detach().cpu().numpy(), saved[name+'_native']))
        for name, value in (('u_solve_float64', detached.node_scores), ('a_solve_float64', detached.child_scores),
                            ('d_solve_float64', detached.duplicate_scores),
                            ('node_event_index', targets.nodes[ordinals]),
                            ('original_global_node_ordinals', np.asarray(ordinals, '<i8'))):
            checks.append(same_bytes(name, value, saved[name]))
    if scores.coverage['message'] != [counts['message'], counts['message']]:
        raise ValueError('Both full message passes required')
    del detached
    loss, detail = supervised_path_loss(scores, path)
    validate_actual_scalar_loss(loss, detail)
    writer.write_json(args.output/'preflight_progress.json', dict(stage='CHECKING_ALL_SUPERVISED_BRANCH_CONNECTIONS',
        complete_path_loss=detail, actual_weld_optimizer_updates=0))
    connections = supervised_branch_connections(model, scores)
    writer.commit_json(args.output/'branch_connections.json', connections)
    writer.write_json(args.output/'preflight_progress.json', dict(stage='ACTUAL_COMPLETE197_BACKWARD',
        actual_weld_optimizer_updates=0))
    (loss / path.scope['source_path_denominator']).backward()
    synchronization = synchronize_model(model)
    gradients = gradient_receipt(model)
    if original.model_sha(model) != before or not gradients['finite']:
        raise ValueError('No-update actual consumer changed parameters or produced invalid gradients')
    memory = dict(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(torch.device(args.device))),
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved(torch.device(args.device))),
                  after_sync_free_total_bytes=list(torch.cuda.mem_get_info(torch.device(args.device))))
    # No preflight RNG consumption is carried into the first counted source.
    del loss, scores, inputs, relations, graph, targets, path
    model.zero_grad(set_to_none=True); gc.collect()
    synchronize_model(model)
    restoration = restore_initial_boundary(zero_boundary['path'], model, config)
    writer.commit_json(args.output / 'receipt.json', dict(
        status='PASS_ACTUAL197_FULL_MAXPATH_SUPERVISED_FORWARD_BACKWARD_NO_UPDATE', version=VERSION,
        UTC=datetime.now(timezone.utc).isoformat(), loss_kind=LOSS_KIND,
        config_content_sha256=object_sha(config), actual_model_device=str(next(model.parameters()).device),
        initial_model_state_hash=before, after_model_state_hash=original.model_sha(model),
        finite_gradients=True, all_candidates_and_pairs_retained=True,
        original_initial_restore=dict(source=spec['source'], path_id=path_id, arrays_byte_identical=checks,
            graph_content_hash=binding['graph_content_hash'], complete_counts=counts),
        complete_path_loss=detail, gradients=gradients, all_supervised_branches_connected=True,
        branch_connections_artifact=dict(path=str(args.output/'branch_connections.json'),
                                         sha256=file_sha(args.output/'branch_connections.json')),
        pre_forward_zero_boundary=zero_boundary, zero_boundary_restoration=restoration,
        actual_CUDA_memory=memory,
        synchronization=synchronization, original227_actual_terminal_observation=resource,
        actual_weld_optimizer_updates=0, actual_optimizer_created=False,
        training_native_solver_calls=0, full_contract_pass=False))
    print('232_ACTUAL_COMPLETE197_NO_UPDATE_PREFLIGHT_PASS', flush=True)


def run(args):
    operation = 'preflight' if args.preflight else 'run'
    config, release, binding, resource = validate_release(args.root_release, operation, args.device, args.resume_checkpoint)
    if args.output.exists():
        raise FileExistsError('Preserve every attempt; a fresh output directory is required')
    # Resource/method checks finish before importing any neural framework.
    import torch
    from .durable_artifacts_223 import AtomicWriter
    from .exact_observed_solver_229 import actual_process_identity, ExactObservedAdapter, TrainingTelemetry, plain
    writer = AtomicWriter(); args.output.mkdir(parents=True)
    owner = actual_process_identity()
    writer.commit_json(args.output / 'actual_compute_process_identity.json', owner)
    cursor = SourceCursor(); durable_cursor = None; latest_checkpoint = None
    pending_checkpoint = None; result = None; stage = 'INITIALIZATION'; adapter = None
    telemetry = None; original = None; original_dump = None; original_runtime = None
    try:
        if args.preflight:
            no_update_preflight(args, config, binding, resource)
            return
        model, original = initialize_original_model(config, binding, args.device)
        if release['status'] == 'AUTHORIZED_FIRST_COUNTED_SOURCE_232':
            proof = read(release['actual_no_update_preflight']['path'])
            restored_zero = restore_initial_boundary(proof['pre_forward_zero_boundary']['path'], model, config)
            writer.commit_json(args.output/'first_source_initial_zero_boundary_restored.json', restored_zero)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        evaluated = []
        if args.resume_checkpoint is not None:
            cursor, extra, loaded = load_checkpoint(args.resume_checkpoint, model, optimizer, config)
            writer.commit_json(args.output / 'resume_loaded_checkpoint.json', loaded)
            evaluated = list(extra.get('evaluated_actual_updates', []))
        if release['status'] == 'AUTHORIZED_FIRST_COUNTED_SOURCE_232' and cursor.complete_sources != 0:
            raise ValueError('First-source authorization starts at the original initialization')
        if release['status'] == 'AUTHORIZED_REMAINING_SOURCE_C_232' and cursor.complete_sources < 1:
            raise ValueError('Remaining-source authorization needs the committed counted first source')
        def checkpoint(label, source_receipt=None):
            nonlocal pending_checkpoint
            pending_checkpoint = dict(label=label, observed_cursor=cursor.as_dict(), stage='COMMITTING_PT_METADATA_DESCRIPTOR')
            artifact = commit_checkpoint(args.output/'checkpoints', label, model, optimizer, cursor, config,
                writer=writer, source_receipt=source_receipt, extra=dict(evaluated_actual_updates=evaluated))
            pending_checkpoint = None
            return artifact
        latest_checkpoint = checkpoint('start')
        durable_cursor = cursor
        while not cursor.done:
            validate_release(args.root_release, 'run', args.device, args.resume_checkpoint)
            stage = 'LOADING_COMPLETE_SOURCE'; result = None
            evidence = config['source_target_evidence'][cursor.source_index()]
            spec = config['sources'][cursor.source_index()]
            targets = load_source_targets(spec, evidence)
            graph = original.load_runtime_graph(spec['artifact_directory'], spec['source'], spec['expected_sha256'])
            if graph.complete_hashes() != targets.manifest['runtime_hashes']:
                raise ValueError('Actual full runtime source differs from its registered target graph')
            # No training truth is passed to the runtime graph/view constructor.
            views = original.complete_path_views(graph)
            path_targets = [source_path_targets(targets, p) for p in targets.manifest['full_ordered_paths']]
            by_path = {v.path_id: v for v in path_targets}
            folder = args.output/'sources'/f'completed_slot_{cursor.complete_sources:06d}'
            folder.mkdir(parents=True, exist_ok=False)
            def path_loss(view):
                path_id, local, _, ordinals = view
                bound = bind_runtime_graph(by_path[path_id], local, ordinals)
                if bound.scope['analytic_empty']:
                    return analytic_empty_path(bound)
                inputs, relations = original.query_graph_195_inputs(local)
                scores = model(inputs, relations, chunk_size=config['chunk_size'])
                return supervised_path_loss(scores, bound)
            stage = 'COMPLETE_SOURCE_FORWARD_BACKWARD_AND_DEVICE_UPDATE'
            result = complete_supervised_source(views, [p.scope for p in path_targets],
                manifest_sha256=targets.manifest_sha256, optimizer=optimizer, path_loss=path_loss,
                validate_loss=validate_actual_scalar_loss,
                backward=lambda loss: loss.backward(), gradient_receipt=lambda: gradient_receipt(model),
                record_path=lambda p, detail: writer.commit_json(folder/f'path_{p:06d}.json', detail),
                synchronize=lambda: synchronize_model(model), post_update_validate=lambda: finite_state(model, optimizer),
                model_hash=lambda: original.model_sha(model))
            cursor = cursor.advance(result['optimizer_step_executed'])
            stage = 'COMMITTING_COMPLETE_SOURCE_RECEIPT'
            source_receipt = writer.commit_json(folder/'complete_source.json', dict(result, source=spec['source'],
                after_cursor=cursor.as_dict(), prior_durable_checkpoint=latest_checkpoint))
            stage = 'COMMITTING_REQUIRED_COMPLETE_SOURCE_CHECKPOINT'
            latest_checkpoint = checkpoint(f'complete_source_{cursor.complete_sources:06d}', source_receipt)
            durable_cursor = cursor
            stage = 'VERIFYING_ACTUAL_COMMITTED_CHECKPOINT_LOAD'
            loaded_cursor, extra, loaded = load_checkpoint(latest_checkpoint['path'], model, optimizer, config)
            if loaded_cursor != cursor:
                raise ValueError('Actual loaded immutable cursor differs')
            writer.commit_json(folder/'checkpoint_load_verified.json', loaded)
            writer.write_json(args.output/'training_progress.json', dict(cursor=cursor.as_dict(),
                stage='COMPLETE_SOURCE_CHECKPOINT_LOADED', last_checkpoint=latest_checkpoint,
                last_source=spec['source'], actual_weld_optimizer_updates=cursor.actual_updates))
            print('232_COMPLETE_SOURCE_COMMITTED', cursor.complete_sources, cursor.actual_updates, spec['source'], flush=True)
            del graph, views, targets, path_targets, by_path
            gc.collect()
            if release['status'] == 'AUTHORIZED_FIRST_COUNTED_SOURCE_232':
                writer.commit_json(args.output/'first_source_complete.json', dict(
                    status='PASS_FIRST_COUNTED_COMPLETE_SOURCE_AND_ACTUAL_CHECKPOINT_LOAD_232',
                    loss_kind=LOSS_KIND, config_content_sha256=object_sha(config), cursor=cursor.as_dict(),
                    source_receipt=source_receipt, checkpoint_commit=latest_checkpoint,
                    checkpoint_loaded=loaded, actual_weld_optimizer_updates=cursor.actual_updates,
                    uncounted_real_model_updates=0, full_contract_pass=False))
                return
            evaluate_now = (cursor.actual_updates == 1000 and 1000 not in evaluated) or cursor.done
            if evaluate_now:
                stage = 'STRICT_COMPLETE_RUNTIME_EVALUATION'
                telemetry = TrainingTelemetry(args.output, owner)
                adapter = ExactObservedAdapter(args.output, owner, telemetry).install()
                adapter.set_metadata(operation='STRICT_COMPLETE_RUNTIME_EVALUATION',
                    actual_weld_optimizer_updates=cursor.actual_updates,
                    model_state_hash=original.model_sha(model), config_content_sha256=object_sha(config))
                original_dump, original_runtime = original.dump, original.solve_runtime_graph
                original.dump = lambda p, v: writer.write_json(p, plain(v))
                original.solve_runtime_graph = original.solver213.solve_runtime_graph
                # This inherited config is used solely for the unchanged full Runtime scorer.
                evaluation_config = read(config['original_config_artifact']['path'])
                evaluation = original.evaluate_full16(model, evaluation_config, args.output, cursor.actual_updates)
                adapter.uninstall(); adapter = None
                original.dump, original.solve_runtime_graph = original_dump, original_runtime
                evaluated.append(cursor.actual_updates)
                latest_checkpoint = checkpoint(f'after_evaluation_{cursor.actual_updates:06d}_{cursor.complete_sources:06d}')
                durable_cursor = cursor
                writer.commit_json(args.output/'evaluated_round_requires_consultation.json', dict(
                    status='NEEDS_COMPLETED_CURRENT_ROUND_GPT6_PRO_BEFORE_FURTHER_OPTIMIZATION',
                    evaluation=evaluation, cursor=cursor.as_dict(), checkpoint_commit=latest_checkpoint,
                    full_contract_scope=config['full_contract_scope'],
                    all_types_and_E_scientific_status='NOT_VERIFIED', full_contract_pass=False))
                return
    except Exception as error:
        # A failed state load may already have mutated objects. This exits the
        # run; neither partially restored objects nor a fresh initialization is used.
        detail = {name: getattr(error, name, None) for name in (
            'path_id', 'completed_paths', 'step_entered', 'step_returned_on_host',
            'device_update_synchronized', 'post_update_finite', 'failed_source_gradients_cleared')}
        writer.commit_json(args.output/'failed_round_requires_consultation.json', dict(
            status='FROZEN_FAILED_ROUND_REQUIRES_COMPLETED_GPT6_PRO', stage=stage,
            error_type=type(error).__name__, error=str(error), source_failure=detail,
            observed_cursor=cursor.as_dict(), durable_cursor=durable_cursor.as_dict() if durable_cursor else None,
            last_committed_checkpoint=latest_checkpoint, pending_checkpoint=pending_checkpoint,
            current_source_update_receipt=result, full_contract_pass=False))
        raise
    finally:
        cleanup_errors = []
        if adapter is not None:
            try: adapter.uninstall()
            except Exception as error: cleanup_errors.append(dict(operation='adapter_uninstall', error=str(error)))
        if original is not None and original_dump is not None:
            original.dump, original.solve_runtime_graph = original_dump, original_runtime
        if telemetry is not None:
            try:
                terminal = telemetry.close()
                writer.commit_json(args.output/'runtime_telemetry_terminal_receipt.json', terminal)
            except Exception as error:
                cleanup_errors.append(dict(operation='telemetry_terminal_commit', error=str(error)))
        if cleanup_errors:
            writer.commit_json(args.output/'cleanup_failure.json', dict(cleanup_errors=cleanup_errors,
                original_round_state_preserved=True, full_contract_pass=False))
            import sys
            if sys.exc_info()[0] is None:
                raise RuntimeError('Required runtime cleanup/terminal evidence failed: ' + str(cleanup_errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__': main()
