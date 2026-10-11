"""Recoverable complete-source training; actual updates need a proved release.

The existing H-line capacity stage stays a stage of the whole contract. This
runner never declares full-contract acceptance. New1490 typed observations need
their own supervised head/normalizer and are not silently inserted into197.
"""
import argparse
import ctypes
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import random
import sys

from .source_c_execution_229 import SourceCursor,complete_source_update,CompleteSourceFailure

ROOT=Path(__file__).resolve().parents[1]
VERSION='RECOVERABLE_ORIGINAL_SOURCE_C_WITH_EXACT_NATIVE_229_v1'
CONFIG=ROOT/'weld_pipeline/reports/structured_training_runner_214_20261002/config.json'
INITIAL=CONFIG.parent/'real_maxpath_attempt1'
RECOVERY=ROOT/'weld_pipeline/reports/resume_maxpath_preflight_227_20261002/real_maxpath_recovery_attempt1'
CONFIG_SHA='fe86560d3c9273188978e49d04583064ae4d8b63fe3f8f5d3c65b98262aff378'
MANDATORY_CODE=(
    'weld_pipeline/train_recovered_source_c_229.py','weld_pipeline/exact_observed_solver_229.py',
    'weld_pipeline/source_c_execution_229.py','weld_pipeline/train_boundary_incidence_f1_214.py',
    'weld_pipeline/structured_graph_solver_f1_213.py','weld_pipeline/structured_graph_solver_v1.py',
    'weld_pipeline/boundary_incidence_net_197.py','weld_pipeline/boundary_query_graph_195.py',
    'weld_pipeline/structured_graph_artifacts_200.py','weld_pipeline/highs_backend_215.py',
    'weld_pipeline/highs_observed_backend_225.py','weld_pipeline/fixed_variable_compiler_226.py',
    'weld_pipeline/durable_artifacts_223.py','weld_pipeline/restore_incidence_snapshot_224.py',
)


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def file_sha(path):
    digest=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def kernel_state(owner):
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    api.OpenProcess.restype=ctypes.c_void_p
    api.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(ctypes.c_uint64)]*4
    api.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
    api.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=api.OpenProcess(0x1000,False,owner['process_id'])
    if not handle:
        error=ctypes.get_last_error()
        if error==87:
            return dict(status='ORIGINAL_PROCESS_ABSENT',live=False)
        raise ctypes.WinError(error)
    try:
        times=[ctypes.c_uint64() for _ in range(4)];code=ctypes.c_uint32()
        if not api.GetProcessTimes(handle,*[ctypes.byref(value) for value in times]):
            raise ctypes.WinError(ctypes.get_last_error())
        if not api.GetExitCodeProcess(handle,ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if times[0].value!=owner['creation_filetime_ticks']:
            return dict(status='PID_REUSED_ORIGINAL_GONE',live=False)
        live=code.value==259 and not times[1].value
        return dict(status='ORIGINAL_BOUND_PROCESS_ACTIVE' if live else 'ORIGINAL_BOUND_PROCESS_TERMINATED',
            live=live,process_id=owner['process_id'],creation_filetime_ticks=times[0].value,
            exit_filetime_ticks=times[1].value,kernel_exit_code=int(code.value))
    finally:
        api.CloseHandle(handle)


def readiness():
    owner=read(RECOVERY/'actual_compute_process_identity.json')
    state=kernel_state(owner)
    commits={name:(RECOVERY/name).exists() for name in (
        'good_full_optimal_certificate.json','bad_full_optimal_certificate.json',
        'maxpath_preflight_complete.json','preflight_failure.json','telemetry_terminal_receipt.json')}
    return dict(version=VERSION,actual_kernel_state=state,preflight_commits=commits,
        initial_training_updates=0,
        release_prerequisites_present=not state['live'] and all(commits[name] for name in (
            'good_full_optimal_certificate.json','bad_full_optimal_certificate.json',
            'maxpath_preflight_complete.json','telemetry_terminal_receipt.json')) and not commits['preflight_failure.json'],
        training_release=False,full_contract_pass=False)


def validate_release(path,*,resume=None):
    """No Torch/optimizer initialization occurs before all of these checks."""
    release=read(path)
    if (release.get('status')!='AUTHORIZED_RECOVERED_SOURCE_C_TRAINING_229'
        or release.get('released_by')!='root' or release.get('authorized') is not True
        or release.get('config_sha256')!=CONFIG_SHA or file_sha(CONFIG)!=CONFIG_SHA):
        raise PermissionError('Root release bound to the unchanged214 configuration required')
    config=read(CONFIG)
    binding=read(INITIAL/'initial_snapshot_binding.json')
    if release.get('initial_state_sha256')!=binding['state_sha256'] or file_sha(INITIAL/'initial_state.pt')!=binding['state_sha256']:
        raise ValueError('Original initialized checkpoint binding changed')
    files=release.get('files_sha256',{})
    if set(MANDATORY_CODE)-set(files):
        raise PermissionError('Every actual training source/dependency must be release-bound')
    if any(file_sha(ROOT/name)!=expected for name,expected in files.items()):
        raise ValueError('Release-bound source or prerequisite changed')
    if set(config['gates_required'])-set(release.get('passed_gates',[])):
        raise PermissionError('All original gates, including the real maximum path, required')
    gates=release.get('gate_evidence',{})
    for gate in config['gates_required']+['229_native','229_transactions','229_checkpoint','229_integrated_review']:
        evidence=gates.get(gate)
        if not evidence or file_sha(ROOT/evidence['path'])!=evidence['sha256']:
            raise PermissionError('Missing verified gate evidence: '+gate)
    ready=readiness()
    if not ready['release_prerequisites_present']:
        raise PermissionError('Same227 complete certificates and actual terminal process required')
    if release.get('preflight_complete_sha256')!=file_sha(RECOVERY/'maxpath_preflight_complete.json'):
        raise ValueError('Actual real preflight completion is not release-bound')
    complete=read(RECOVERY/'maxpath_preflight_complete.json')
    if (complete['status']!='PASS_SAME_INSTANCE_MAXPATH_RUNTIME_GOOD_BAD_AND_IDENTITY_BACKWARD'
        or complete['actual_optimizer_steps']!=0 or not complete['finite_gradients']
        or complete['before_after_model_hash']!=binding['model_state_hash']):
        raise ValueError('Real preflight, identity backward or unchanged state unproved')
    for mode in ('good','bad'):
        proof=read(RECOVERY/(mode+'_full_optimal_certificate.json'))
        if (proof['status']!='OPTIMAL' or not proof['optimal_proved']
            or not proof['complete_graph_solution'] or proof['mip_gap'] is None or proof['mip_gap']>1e-9):
            raise ValueError('Complete real'+mode+' certificate unproved')
    if resume is not None:
        evidence=release.get('resume_checkpoint')
        if not evidence or Path(evidence['path']).resolve()!=Path(resume).resolve() or file_sha(resume)!=evidence['sha256']:
            raise PermissionError('Exact proved completed-source checkpoint required for resume')
        if release.get('completed_failure_consultation') is not True:
            raise PermissionError('After a failed round, completed Pro consultation required before further updates')
    return config,release,binding


def checkpoint_payload(model,optimizer,cursor,config,*,model_hash,extra=None):
    """Binary-handle serialization; exact source cursor and all RNGs retained."""
    import numpy as np
    import torch
    gpu=any(parameter.is_cuda for parameter in model.parameters())
    value=dict(version=VERSION,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
        cursor=cursor.as_dict(),config=config,config_sha256=CONFIG_SHA,model_state_hash=model_hash,
        RNG_python=random.getstate(),RNG_numpy=np.random.get_state(),RNG_torch=torch.get_rng_state(),
        RNG_cuda=torch.cuda.get_rng_state_all() if gpu else None,extra=extra or {},
        checkpoint_boundary='COMPLETE_SOURCE_AFTER_RETURNED_UPDATE_OR_ZERO_GRADIENT',
        full_contract_pass=False)
    stream=io.BytesIO();torch.save(value,stream)
    return stream.getvalue()


def load_checkpoint(path,model,optimizer,config,*,expected_model_hash,model_hash_function):
    import numpy as np
    import torch
    with Path(path).open('rb') as stream:
        saved=torch.load(stream,map_location='cpu')
    if (saved['version']!=VERSION or saved['config_sha256']!=CONFIG_SHA or saved['config']!=config
        or saved['checkpoint_boundary']!='COMPLETE_SOURCE_AFTER_RETURNED_UPDATE_OR_ZERO_GRADIENT'):
        raise ValueError('Proved complete-source checkpoint and unchanged config required')
    cursor=SourceCursor(**saved['cursor'])
    model.load_state_dict(saved['state_dict'],strict=True)
    if model_hash_function(model)!=saved['model_state_hash'] or saved['model_state_hash']!=expected_model_hash:
        raise ValueError('Resumed model bytes differ from completed-source binding')
    optimizer.load_state_dict(saved['optimizer'])
    random.setstate(saved['RNG_python']);np.random.set_state(saved['RNG_numpy']);torch.set_rng_state(saved['RNG_torch'])
    if saved['RNG_cuda'] is not None:
        if not any(parameter.is_cuda for parameter in model.parameters()):
            raise ValueError('CUDA checkpoint requires the recorded CUDA device')
        torch.cuda.set_rng_state_all(saved['RNG_cuda'])
    return cursor,saved['extra']


def gradient_receipt(gradient_state):
    """Describe the actual finite accumulated tensors without JSON-encoding them."""
    import numpy as np
    tensors,nonzero=gradient_state()
    receipt={}
    for name,tensor in tensors.items():
        array=np.ascontiguousarray(tensor.detach().cpu().numpy())
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite accumulated gradient: '+name)
        receipt[name]=dict(shape=list(array.shape),dtype=str(array.dtype),
            raw_values_sha256=sha256(array.tobytes()).hexdigest(),finite=True,
            any_nonzero=bool(np.any(array!=0)),
            max_abs=float(np.abs(array).max()) if array.size else 0.)
    if not isinstance(nonzero,bool) or nonzero!=any(value['any_nonzero'] for value in receipt.values()):
        raise ValueError('Actual finite gradient update condition differs from its receipt')
    return receipt,nonzero


def run(args):
    # The guard deliberately runs before importing neural modules or CUDA.
    config,release,binding=validate_release(args.root_release,resume=args.resume_checkpoint)
    import gc
    import numpy as np
    import torch
    from . import train_boundary_incidence_f1_214 as original
    from .restore_incidence_snapshot_224 import restore
    from .durable_artifacts_223 import AtomicWriter
    from .exact_observed_solver_229 import ExactObservedAdapter,TrainingTelemetry,actual_process_identity,plain
    if args.output.exists():
        raise FileExistsError('Preserve all previous attempts; use a fresh run output')
    args.output.mkdir(parents=True)
    writer=AtomicWriter();owner=actual_process_identity()
    writer.commit_json(args.output/'actual_compute_process_identity.json',owner)
    telemetry=TrainingTelemetry(args.output,owner)
    adapter=None;latest_checkpoint=None;cursor=SourceCursor();attempts=0
    durable_cursor=None;pending_checkpoint=None
    source_step_entered=False;source_step_returned=False
    execution_stage='RESTORING_ORIGINAL_INITIAL_STATE'
    original_dump=original.dump
    original_solvers={name:getattr(original,name) for name in (
        'solve_runtime_graph','solve_best_truth_consistent_graph','solve_loss_augmented_graph')}
    def commit_checkpoint(label,extra):
        nonlocal pending_checkpoint
        pending_checkpoint=dict(label=label,cursor=cursor.as_dict(),stage='SERIALIZING')
        data=checkpoint_payload(model,optimizer,cursor,config,model_hash=original.model_sha(model),extra=extra)
        pending_checkpoint['stage']='COMMITTING_STATE_BYTES'
        artifact=writer.commit_bytes(args.output/'checkpoints'/(label+'.pt'),data)
        pending_checkpoint.update(stage='COMMITTING_REQUIRED_METADATA',state_artifact=artifact)
        metadata=writer.commit_json(Path(artifact['path']).with_suffix('.json'),dict(
            owner,artifact=artifact,cursor=cursor.as_dict(),model_state_hash=original.model_sha(model),
            completed_source_boundary=True,version=VERSION))
        pending_checkpoint=None
        return dict(artifact,completed_source_metadata=metadata)
    try:
        # Same original bytes/scores, not a fresh random training initialization.
        model,graph,truth,scores,_,_,restored=restore(CONFIG,INITIAL,args.device)
        writer.commit_json(args.output/'original_initial_restore.json',restored)
        del graph,truth,scores
        gc.collect()
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        if args.resume_checkpoint is not None:
            cursor,previous=load_checkpoint(args.resume_checkpoint,model,optimizer,config,
                expected_model_hash=release['resume_checkpoint']['model_state_hash'],model_hash_function=original.model_sha)
            attempts=int(previous.get('attempted_source_rounds',cursor.complete_sources))
            latest_checkpoint=dict(release['resume_checkpoint'])
            durable_cursor=cursor
        execution_stage='COMMITTING_START_CHECKPOINT'
        latest_checkpoint=commit_checkpoint('start',dict(attempted_source_rounds=attempts,actual_updates=cursor.actual_updates))
        durable_cursor=cursor
        adapter=ExactObservedAdapter(args.output,owner,telemetry).install()
        for name in original_solvers:
            setattr(original,name,getattr(original.solver213,name))
        # Reuse214 official full16 scorer, with reliable required writes.
        original.dump=lambda path,value:writer.write_json(path,plain(value))
        while not cursor.done:
            source_step_entered=False;source_step_returned=False
            execution_stage='CHECKING_RELEASE_AND_LOADING_SOURCE'
            validate_release(args.root_release,resume=args.resume_checkpoint)
            source_index=cursor.source_index();spec=config['sources'][source_index]
            source=spec['source'];attempts+=1
            folder=args.output/'sources'/f'completed_slot_{cursor.complete_sources:06d}'
            folder.mkdir(parents=True,exist_ok=False)
            metadata=dict(epoch=cursor.epoch,slot=cursor.slot,source_index=source_index,source=source,
                actual_optimizer_steps=cursor.actual_updates,model_state_hash=original.model_sha(model),
                full_path_denominator=spec['path_count'],operation='ACTUAL_COMPLETE_SOURCE_TRAINING')
            adapter.set_metadata(**metadata)
            graph=original.load_runtime_graph(spec['artifact_directory'],source,spec['expected_sha256'])
            truth=original.load_training_truth(spec['artifact_directory'],source,graph,spec['expected_sha256'])
            views=original.complete_path_views(graph,truth)
            source_before_hash=original.model_sha(model)
            def path_loss(view):
                path,local,labels,_=view
                adapter.set_metadata(**metadata,path=path)
                def progress(stage,**values):
                    telemetry.phase_event(stage,**metadata,path=path,**values)
                return original.exact_latent_loss(model,local,labels,config['chunk_size'],
                    original.SolveOptions(**config['solver_options']),progress)
            def record_path(path,detail):
                return writer.commit_json(folder/('path_'+str(path).zfill(6)+'.json'),plain(detail))
            execution_stage='COMPLETE_SOURCE_LOSS_BACKWARD_AND_UPDATE'
            result=complete_source_update(views,spec['path_count'],optimizer=optimizer,path_loss=path_loss,
                backward=lambda loss:loss.backward(),
                gradient_state=lambda:gradient_receipt(lambda:original.gradient_state(model)),record_path=record_path)
            source_step_entered=result['optimizer_step_executed']
            source_step_returned=result['optimizer_step_executed']
            if not result['optimizer_step_executed'] and original.model_sha(model)!=source_before_hash:
                raise ValueError('Zero-gradient source changed weights/buffers')
            cursor=cursor.advance(result['optimizer_step_executed'])
            execution_stage='COMMITTING_COMPLETE_SOURCE_RECEIPT'
            source_receipt=writer.commit_json(folder/'complete_source.json',plain(dict(result,
                **metadata,after_cursor=cursor.as_dict(),after_model_state_hash=original.model_sha(model),
                prior_completed_checkpoint=latest_checkpoint)))
            execution_stage='COMMITTING_COMPLETE_SOURCE_CHECKPOINT'
            latest_checkpoint=commit_checkpoint(f'complete_source_{cursor.complete_sources:06d}',dict(
                attempted_source_rounds=attempts,source_receipt=source_receipt,actual_updates=cursor.actual_updates))
            durable_cursor=cursor
            execution_stage='COMPLETE_SOURCE_DURABLY_COMMITTED'
            telemetry.send('training_progress.json',dict(cursor=cursor.as_dict(),last_completed_source=source,
                last_checkpoint=latest_checkpoint,actual_optimizer_steps=cursor.actual_updates))
            del graph,truth,views,result
            gc.collect()
            if cursor.actual_updates and cursor.actual_updates%config['fit_every_actual_updates']==0:
                execution_stage='COMPLETE_RUNTIME_EVALUATION'
                adapter.set_metadata(operation='COMPLETE_RUNTIME_EVALUATION',actual_optimizer_steps=cursor.actual_updates,
                                     model_state_hash=original.model_sha(model))
                evaluation=original.evaluate_full16(model,config,args.output,cursor.actual_updates)
                writer.commit_json(args.output/'evaluated_round_stopped_for_consultation.json',dict(
                    status='NEEDS_COMPLETED_GPT6_PRO_CONSULT_BEFORE_FURTHER_UPDATES',evaluation=evaluation,
                    cursor=cursor.as_dict(),last_complete_source_checkpoint=latest_checkpoint,
                    full_contract_pass=False))
                return
        execution_stage='FINAL_COMPLETE_RUNTIME_EVALUATION'
        adapter.set_metadata(operation='FINAL_COMPLETE_RUNTIME_EVALUATION',actual_optimizer_steps=cursor.actual_updates,
                             model_state_hash=original.model_sha(model))
        evaluation=original.evaluate_full16(model,config,args.output,'final_'+str(cursor.actual_updates))
        writer.commit_json(args.output/'training_receipt.json',dict(
            status='ORIGINAL_BUDGET_COMPLETE_NEEDS_CONSULT_FULL_CONTRACT_UNPROVED',cursor=cursor.as_dict(),
            actual_optimizer_steps=cursor.actual_updates,final_evaluation=evaluation,
            last_complete_source_checkpoint=latest_checkpoint,source_slots_budget=1280,epochs_budget=80,
            full_contract_pass=False))
    except Exception as error:
        # A failed path never reaches step. An exception inside an entered step
        # is explicitly uncertain; only the previous durable boundary may resume.
        failure=dict(status='ROUND_STOPPED_NEEDS_GPT6_PRO_CONSULT',error_type=type(error).__name__,error=str(error),
            cursor=cursor.as_dict(),attempted_source_rounds=attempts,last_complete_source_checkpoint=latest_checkpoint,
            execution_stage=execution_stage,
            optimizer_step_entered=getattr(error,'optimizer_step_entered',source_step_entered),
            optimizer_step_returned=getattr(error,'optimizer_step_returned',source_step_returned),
            observed_completed_source_cursor=cursor.as_dict(),
            last_durable_source_cursor=durable_cursor.as_dict() if durable_cursor is not None else None,
            successful_durable_updates=durable_cursor.actual_updates if durable_cursor is not None else None,
            pending_checkpoint_not_eligible_for_automatic_resume=pending_checkpoint,
            incomplete_entered_optimizer_state_auto_resume=False,full_G_retained=1221,full_contract_pass=False)
        diagnostic_errors=[]
        if 'optimizer' in locals():
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception as cleanup_error:
                diagnostic_errors.append(dict(operation='CLEARING_FAILED_GRADIENTS',
                    error_type=type(cleanup_error).__name__,error=str(cleanup_error)))
        if 'model' in locals():
            try:
                failure['observed_failure_model_state_hash']=original.model_sha(model)
                # Diagnostic weights are not a valid completed-source checkpoint.
                stream=io.BytesIO();torch.save(dict(state_dict=model.state_dict(),failure=failure),stream)
                failure['diagnostic_state_not_for_automatic_resume']=writer.commit_bytes(args.output/'failed_partial_state.pt',stream.getvalue())
            except Exception as diagnostic_error:
                diagnostic_errors.append(dict(operation='SAVING_DIAGNOSTIC_PARTIAL_STATE',
                    error_type=type(diagnostic_error).__name__,error=str(diagnostic_error)))
        failure['diagnostic_errors']=diagnostic_errors
        writer.commit_json(args.output/'training_failure.json',plain(failure))
        raise
    finally:
        original.dump=original_dump
        for name,value in original_solvers.items():
            setattr(original,name,value)
        if adapter is not None:
            adapter.uninstall()
        writer.commit_json(args.output/'telemetry_terminal_receipt.json',telemetry.close())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inspect-ready',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--root-release',type=Path)
    parser.add_argument('--resume-checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.inspect_ready==args.run:
        parser.error('Choose exactly one inspect-ready/run action')
    if args.inspect_ready:
        print(json.dumps(readiness(),ensure_ascii=False,indent=2))
        return
    if args.root_release is None or args.output is None:
        parser.error('Actual training requires root-release/output')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    run(args)


if __name__=='__main__':
    main()
