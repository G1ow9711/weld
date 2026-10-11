"""One-shot desktop recovery controller. A failure consumes this sole attempt."""
import argparse
import os
import shutil
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, digest, utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='run284')
    args = parser.parse_args()
    dest = OUT/args.run
    dest.mkdir(exist_ok=False)
    control = read(OUT/'run283_verified/receipt.json')
    assert control['status'] == 'OWNED_WATCHDOG_BLOCKING_CONTROL_PASSED'
    assert control['watchdog_code_sha256'] == digest(ROOT/'weld_pipeline/sjg_owned_watchdog_1101.py')
    assert read(OUT/'pro_review_run282/receipt.json')['assistant_sentinel_verified']
    # An exclusive lease makes this a one-shot continuation, even under another run name.
    with (OUT/'sole_recovery_attempt_Pro282.json').open('x', encoding='utf-8') as stream:
        import json
        json.dump(dict(recovery_run=args.run, maximum_additional_updates=4, utc=utc()), stream)
    snapshot = dest/'source_snapshot'
    snapshot.mkdir()
    bindings = {}
    for number in (904,1075,1078,1080,1084,1087,1090,1093,1097,1100,1101,1102,1103):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):
            shutil.copy2(path, snapshot/path.name)
            bindings[path.name] = digest(path)
    atomic_json(dest/'registered_protocol.json', dict(stage='ORIGINAL_SCHEDULE_COMPLETION_ONLY',
        original_run='run280', original_completed_updates=26, original_training_seconds=397.55600739998044,
        original_budget_180s_pass=False, maximum_additional_updates=4,
        prescribed_sources=['10','21','BM79A_native_local','10'], wall_budget_seconds=120,
        original_checkpoint_sha256='db33e3bab30f5811b8e12eef9c575e13afec4e76636be34cb421928bf458b32c',
        original_failure_sha256=digest(OUT/'run280/failure.json'),
        code_bindings=bindings, retry_on_failure=False, full_contract_pass=False, utc=utc()))
    os.environ['PYTHONPATH'] = ';'.join(map(str,(ROOT, ROOT/'.venv-ptv3/Lib/site-packages', ROOT/'.venv/Lib/site-packages')))
    command = [str(ROOT/'.venv-ptv3/Scripts/python.exe'), '-u', '-m',
               'weld_pipeline.sjg_four_step_recovery_1102', '--run', args.run]
    try:
        result = supervise(command, dest, 120.)
        committed = []
        previous = read(dest/'registered_protocol.json')['original_checkpoint_sha256']
        for number in range(27,31):
            path = dest/f'committed_{number}.json'
            if not path.exists():
                break
            entry = read(path)
            assert entry['previous_checkpoint_sha256'] == previous
            assert digest(dest/entry['checkpoint_file']) == entry['checkpoint_sha256']
            previous = entry['checkpoint_sha256']
            committed.append(number)
        phase = read(dest/'worker_phase.json') if (dest/'worker_phase.json').exists() else {'phase':'NOT_OBSERVED'}
        if result['termination_reason'] or result['exit_code'] != 0 or result['wall_seconds'] > 120 or committed != [27,28,29,30] or not (dest/'worker_complete.json').exists():
            atomic_json(dest/'failure.json', dict(status='SOLE_RECOVERY_FAILED_RETIRE_FREE_QUERY',
                watchdog=result, verified_committed_updates=committed, last_phase=phase,
                unconfirmed_optimizer_submission_possible=phase['phase'] in ('OPTIMIZER_STEP','CHECKPOINT_WRITE'),
                original_budget_180s_pass=False, retry_permitted=False, utc=utc()))
            raise RuntimeError('Sole recovery did not complete within its resource/wall boundary; retire free-query trial')
        completion = read(dest/'worker_complete.json')
        assert completion['checkpoint_sha256'] == previous and completion['updates'] == 30
        atomic_json(dest/'receipt.json', dict(completion,
            status='PARENT_SET_RECOVERY_COMPLETE_ORIGINAL_BUDGET_FAILED',
            recovery_wall_seconds=result['wall_seconds'], recovery_wall_pass=result['wall_seconds']<=120,
            full_pilot_G=500, full_contract_pass=False))
        print('ORIGINAL_30_UPDATES_COMPLETED_BY_ONE_RECOVERY_ORIGINAL_180S_FAILED', flush=True)
    except BaseException:
        atomic_json(dest/'controller_error.json', dict(error=traceback.format_exc(), utc=utc()))
        if not (dest/'failure.json').exists():
            atomic_json(dest/'failure.json', dict(status='SOLE_RECOVERY_FAILED_RETIRE_FREE_QUERY',
                reason='Controller or state verification failure; inspect controller_error.json',
                original_budget_180s_pass=False, retry_permitted=False, utc=utc()))
        raise


if __name__ == '__main__':
    main()
