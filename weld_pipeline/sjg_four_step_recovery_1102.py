"""Exactly one continuation of run280 updates 27--30, under external supervision."""
import argparse
import gc
import hashlib
import random
import shutil
import time
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, digest, utc, check_resources, reclaim_own_idle_pages
from weld_pipeline.sjg_owned_watchdog_1101 import atomic_json
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write


def run(name):
    worker_started = time.perf_counter()
    dest = OUT / name
    phase = lambda value, **fields: atomic_json(dest/'worker_phase.json', dict(phase=value, utc=utc(), **fields))
    phase('IMPORTING', confirmed_updates=26)
    import numpy as np
    import torch
    from weld_pipeline.sjg_persistent_query_network_1097 import PersistentQueryModel
    from weld_pipeline.sjg_roboweldar_pair_support_1093 import PairParentInput
    from weld_pipeline.sjg_parent_anchor_bank_1080 import CarrierBank
    from weld_pipeline.sjg_parent_set_loss_1084 import ParentTargets, ParentSetObjective
    from weld_pipeline.sjg_parent_stream_encoder_1087 import stream_encode
    from weld_pipeline.sjg_parent_resource_start_1090 import start_cuda

    def tree_hash(value):
        h = hashlib.sha256()
        def add(v):
            if torch.is_tensor(v):
                a = v.detach().cpu().contiguous().numpy()
                h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
            elif isinstance(v, np.ndarray):
                h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
            elif isinstance(v, dict):
                for k in sorted(v, key=str):
                    h.update(repr(k).encode()); add(v[k])
            elif isinstance(v, (tuple, list)):
                for item in v:
                    add(item)
            else:
                h.update(repr(v).encode())
        add(value)
        return h.hexdigest()

    def rng():
        return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate())

    updates = 26
    history = []
    train_started = None
    checkpoint_seconds = 0.
    try:
        registration = read(dest/'registered_protocol.json')
        assert registration['maximum_additional_updates'] == 4 and registration['wall_budget_seconds'] == 120
        for file, sha in registration['code_bindings'].items():
            assert digest(ROOT/'weld_pipeline'/file) == sha
        source = OUT/'run280'
        original_failure = read(source/'failure.json')
        assert digest(source/'failure.json') == registration['original_failure_sha256']
        assert original_failure['actual_updates'] == 26 and original_failure['training_seconds'] > 180
        assert not (source/'receipt.json').exists() and not (source/'final.pt').exists()
        original_sha = digest(source/'interrupted.pt')
        assert original_sha == registration['original_checkpoint_sha256']
        pro = read(OUT/'pro_review_run282/receipt.json')
        assert pro['generation_stopped'] and pro['assistant_sentinel_verified']
        assert hashlib.sha256((OUT/'pro_review_run282/answer.md').read_text(encoding='utf-8').encode()).hexdigest() == pro['answer_sha256']
        torch.set_num_threads(2)
        reclaim_own_idle_pages()
        atomic_json(dest/'cuda_startup.json', start_cuda())
        with (source/'interrupted.pt').open('rb') as stream:
            checkpoint = torch.load(stream, map_location='cpu')
        assert checkpoint['updates'] == 26 and checkpoint['status'] == 'INCOMPLETE'
        assert checkpoint['protocol']['decoder'] == 'persistent_query_1097'
        phase('RESTORING', confirmed_updates=26)
        model = PersistentQueryModel().cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        old = [p for n, p in model.named_parameters() if not n.startswith('query_attention.')]
        optimizer = torch.optim.AdamW(old, lr=.0003, weight_decay=.0001)
        optimizer.add_param_group(dict(params=list(model.query_attention.parameters()), lr=.0003, weight_decay=.0001))
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        model_hash = tree_hash(model.state_dict())
        adam_hash = tree_hash(optimizer.state_dict())
        assert model_hash == tree_hash(checkpoint['state_dict'])
        assert adam_hash == tree_hash(checkpoint['optimizer_state'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        rng_hash = tree_hash(rng())
        assert rng_hash == tree_hash({k: checkpoint[k] for k in rng()})
        optimizer.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in model.parameters())
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert all(layer.dropout == 0 for layer in model.query_attention)
        protocol = checkpoint['protocol']
        del checkpoint
        gc.collect()
        atomic_json(dest/'restored_boundary.json', dict(model_sha256=model_hash, Adam_sha256=adam_hash,
            RNG_sha256=rng_hash, optimizer_and_moments_exact=True, uncommitted_gradients_cleared=True,
            completed_updates=26, next_source='10', dtype='float32', torch_version=torch.__version__,
            GPU=torch.cuda.get_device_name(0), torch_threads=torch.get_num_threads(),
            RNG_boundary_basis='Training forward/backward paths have no sampling; self-attention dropout=0. Constructors consume RNG before restoration. Per-step RNG equality is checked before any update.',
            cross_device_bitwise_equivalence_claimed=False))
        phase('LOADING_FROZEN_INPUTS', confirmed_updates=26)
        for file in ('normalization.npz', 'input_bindings.json'):
            shutil.copy2(source/file, dest/file)
        with np.load(dest/'normalization.npz') as z:
            normalization = {k: z[k] for k in z.files}
        bindings = {b['source']: b for b in read(source/'input_bindings.json')['cases']}
        target_bindings = {r['source']: r for r in read(OUT/'run264/receipt.json')['cases']}
        inputs, objectives = {}, {}
        for key in ('21', 'BM79A_native_local', '10'):
            inputs[key] = PairParentInput(key, normalization)
            assert inputs[key].binding() == bindings[key]
            path = OUT/f'run264/source_{key}_targets.json'
            assert digest(path) == target_bindings[key]['compiled_sha256']
            bank = CarrierBank(key)
            objectives[key] = ParentSetObjective(model, ParentTargets(read(path), bank, 'cuda'), common_denominator=True)
            del bank
            reclaim_own_idle_pages(); check_resources()
            print('RECOVERY_INPUT_READY', key, 'COMPLETE_PROFILES', len(inputs[key]), flush=True)
        assert tree_hash(rng()) == rng_hash
        preparation_seconds = time.perf_counter() - worker_started
        train_started = time.perf_counter()
        schedule = ('10', '21', 'BM79A_native_local', '10')
        previous_sha = original_sha
        for key in schedule:
            next_update = updates + 1
            phase('FORWARD_BACKWARD', confirmed_updates=updates, attempted_update=next_update, source=key)
            check_resources()
            optimizer.zero_grad(set_to_none=True)
            before_rng = tree_hash(rng())
            result = model.forward_encoded(stream_encode(model, inputs[key]))
            loss, parts = objectives[key](result)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            torch.cuda.synchronize()
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite gradient')
            assert tree_hash(rng()) == before_rng, 'Unexpected stochastic operation; boundary cannot be certified'
            phase('OPTIMIZER_STEP', confirmed_updates=updates, attempted_update=next_update, source=key)
            optimizer.step()
            torch.cuda.synchronize()
            updates += 1
            optimizer.zero_grad(set_to_none=True)
            item = dict(epoch=(updates-1)//3+1, update=updates, source=key,
                        loss=float(loss.detach()), parts=parts, gradient_norm=float(norm),
                        resumed_training_seconds_at_snapshot=time.perf_counter()-train_started)
            del result, loss
            phase('CHECKPOINT_WRITE', last_previously_confirmed_update=updates-1, attempted_update=updates)
            tick = time.perf_counter()
            path = dest/f'update_{updates}.pt'
            snapshot = dict(state_dict=model.state_dict(), optimizer_state=optimizer.state_dict(), **rng(),
                            updates=updates, next_source=schedule[updates-26] if updates<30 else None,
                            protocol=protocol, original_failure_run='run280', original_budget_180s_pass=False,
                            original_checkpoint_sha256=original_sha, previous_checkpoint_sha256=previous_sha,
                            normalization_sha256=digest(dest/'normalization.npz'),
                            input_bindings_sha256=digest(dest/'input_bindings.json'),
                            code_bindings=registration['code_bindings'],
                            cumulative_training_seconds_at_snapshot=original_failure['training_seconds']+time.perf_counter()-train_started)
            atomic_binary_write(path, lambda stream: torch.save(snapshot, stream))
            del snapshot
            current_sha = digest(path)
            checkpoint_seconds += time.perf_counter()-tick
            item.update(checkpoint_file=path.name, checkpoint_sha256=current_sha,
                        previous_checkpoint_sha256=previous_sha,
                        resumed_training_seconds_at_commit=time.perf_counter()-train_started,
                        cumulative_training_seconds_at_commit=original_failure['training_seconds']+time.perf_counter()-train_started)
            atomic_json(dest/f'committed_{updates}.json', item)
            previous_sha = current_sha
            history.append(item)
            atomic_json(dest/'history.json', history)
            phase('COMMITTED', confirmed_updates=updates)
            print('EPOCH', item['epoch'], '/10 UPDATE', updates, '/30 SOURCE', key,
                  'LOSS', round(item['loss'],5), 'RESUMED_SECONDS', round(time.perf_counter()-train_started,3), flush=True)
        seconds = time.perf_counter()-train_started
        assert updates == 30 and digest(source/'interrupted.pt') == original_sha
        atomic_json(dest/'worker_complete.json', dict(status='FOUR_ORIGINAL_UPDATES_SAVED', updates=30,
            additional_updates=4, checkpoint_file='update_30.pt', checkpoint_sha256=previous_sha,
            original_training_seconds=original_failure['training_seconds'], resumed_training_seconds=seconds,
            cumulative_training_seconds=original_failure['training_seconds']+seconds,
            original_budget_180s_pass=False, preparation_seconds=preparation_seconds,
            checkpoint_io_seconds_in_training_interval=checkpoint_seconds,
            worker_wall_seconds=time.perf_counter()-worker_started,
            original_failure_sha256=digest(source/'failure.json'), original_checkpoint_sha256=original_sha,
            original_checkpoint_and_failure_unchanged=True, utc=utc()))
        phase('COMPLETE', confirmed_updates=30)
    except BaseException:
        atomic_json(dest/'worker_failure.json', dict(error=traceback.format_exc(), in_memory_updates=updates,
            confirmed_commit_files=[p.name for p in dest.glob('committed_*.json')],
            resumed_interval_seconds=0 if train_started is None else time.perf_counter()-train_started,
            worker_wall_seconds=time.perf_counter()-worker_started, utc=utc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    run(parser.parse_args().run)
