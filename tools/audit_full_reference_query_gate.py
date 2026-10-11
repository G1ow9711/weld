"""Audit the frozen legacy query gate at actual complete TRAIN assignments.

Use all eight first-epoch assignment seeds per ready supervised label version.
This measures supervised assignment admission, not complete candidate capacity,
physical absence, model accuracy or independent validation. No weights change.
"""
import gc
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_runtime_aligned_training_samples import RuntimeAlignedTrainingAssignments
from weld_pipeline.server_joint_reference_query_model import JointReferenceModel
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.prepare_full_task_supervision import verify as verify_supervision

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
OUT = BASE / 'full_reference_query_gate_audit_v1'


def atomic_json(path, value):
    stage = path.with_name(path.name + '.incomplete')
    write(stage, value)
    os.replace(stage, path)


def main():
    OUT.mkdir(exist_ok=False)
    checkpoint = OUT / 'query_checkpoint.pt'
    shutil.copyfile(BASE / 'full_weld_anchored_v2/last.pt', checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    model = JointReferenceModel().to(device).eval()
    model.load_state_dict({name.removeprefix('legacy.'): value for name, value in state['state_dict'].items()
                           if name.startswith('legacy.')}, strict=True)
    mean, std = [torch.as_tensor(state[name], device=device, dtype=torch.float32) for name in ('mean', 'std')]
    dataset = FullShardDataset(ROOT)
    jobs = [key for key in dataset.ready_jobs() if all(
        (BASE / 'full_task_supervision_v1' / key / v['key'] / 'receipt.json').is_file()
        for v in dataset.jobs[key]['versions'])]
    protocol = dict(status='ALL_READY_TRAIN_COMPLETE_REFERENCE_ASSIGNMENT_GATE_AUDIT',
        registry_sha256=dataset.registry_sha256, jobs=jobs,
        versions=sorted(v['key'] for key in jobs for v in dataset.jobs[key]['versions']),
        query_checkpoint_sha256=digest(checkpoint), query_optimizer_steps=state['optimizer_steps'],
        assignment_seeds='Exactly first-epoch steps 0..7, seed1031 and full registered version index',
        threshold_logit=0., roundoff_band_logit=2e-5, optimizer_updates=0,
        query_score_is_reference_task_prior_not_physical_absence=True,
        candidate_capacity_or_accuracy_not_evaluated=True,
        sources={name:digest(ROOT / name) for name in ['tools/audit_full_reference_query_gate.py',
            'weld_pipeline/server_runtime_aligned_training_samples.py',
            'weld_pipeline/server_full_training_samples.py', 'weld_pipeline/server_joint_reference_query_model.py']})
    write(OUT / 'protocol.json', protocol)
    results = []
    (OUT / 'versions').mkdir()
    start = time.perf_counter()
    for key in jobs:
        inp = dataset.input_case(key)
        builder = RuntimeAlignedTrainingAssignments(inp)
        for version in dataset.jobs[key]['versions']:
            vkey = version['key']
            supervision = dataset.supervision(vkey)
            witness_folder = BASE / 'full_task_supervision_v1' / key / vkey
            verify_supervision(witness_folder, dataset, version)
            with np.load(witness_folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
                witnesses = {name: arrays[name] for name in arrays.files}
            scores, known = [], []
            for step in range(8):
                seed = np.random.SeedSequence([1031, 0, list(dataset.versions).index(vkey), step])
                with threadpool_limits(limits=2):
                    sample = builder.build(supervision, witnesses, seed)
                raw = torch.tensor(sample['features'], device=device, dtype=torch.float32)
                normalized = torch.clamp((raw - mean) / std, -20, 20)
                logits = []
                with torch.inference_mode():
                    for begin in range(0, len(raw), 2048):
                        h = model.trunk(normalized[begin:begin + 2048])
                        logits.append(model.reference(h).squeeze(1).cpu().numpy())
                score = np.concatenate(logits) if logits else np.empty(0, np.float32)
                if len(score) != supervision.target['original_children'] or not np.isfinite(score).all():
                    raise ValueError('Reference gate audit lost a child or produced nonfinite scores')
                scores.append(score)
                known.append(sample['targets']['interval_known'])
            scores = np.asarray(scores)
            best, worst = scores.max(0), scores.min(0)
            result = dict(version_key=vkey, job_key=key, role='TRAIN_ASSIGNMENT_DIAGNOSTIC_ONLY',
                reference_children=len(best), assignment_score_evaluations=int(scores.size),
                assigned_children_below_gate_all_eight_seeds=int((best < -2e-5).sum()),
                assigned_children_above_gate_any_seed=int((best > 2e-5).sum()),
                assigned_children_in_roundoff_band=int((abs(best) <= 2e-5).sum()),
                every_assigned_seed_above_gate=int((worst > 2e-5).sum()),
                carrier_compatible_assignment_visits=int(np.asarray(known).sum()),
                carrier_incompatible_keeps_free_geometry=True,
                minimum_best_logit=float(best.min()) if len(best) else None,
                maximum_best_logit=float(best.max()) if len(best) else None,
                label_receipt_sha256=version['label_receipt_sha256'],
                owner_witness_receipt_sha256=digest(witness_folder / 'receipt.json'),
                another_unassigned_query_may_still_match_reference=True,
                full_candidate_unreachability_or_extraction_accuracy_not_claimed=True)
            np.savez_compressed(OUT / 'versions' / (vkey + '_scores.npz'), logits=scores)
            result['score_artifact_sha256'] = digest(OUT / 'versions' / (vkey + '_scores.npz'))
            write(OUT / 'versions' / (vkey + '.json'), result)
            results.append(result)
            atomic_json(OUT / 'progress.json', dict(status='ACTUAL_COMPLETE_TRAIN_ASSIGNMENT_GATE_AUDIT',
                controller_pid=os.getpid(), completed_versions=len(results), required_versions=len(protocol['versions']),
                job_key=key, version_key=vkey, optimizer_updates=0, unix_time=time.time()))
            print('ACTUAL_REFERENCE_QUERY_GATE', vkey, result['assigned_children_below_gate_all_eight_seeds'],
                  '/', result['reference_children'], flush=True)
        del inp, builder
        gc.collect()
    result = dict(status='ALL_FROZEN_READY_TRAIN_ASSIGNMENT_GATE_AUDITS_COMPLETE_NOT_MODEL_ACCURACY',
        protocol_sha256=digest(OUT / 'protocol.json'), registered_versions=len(dataset.versions),
        audited_versions=len(results), audited_jobs=len(jobs),
        reference_children=sum(r['reference_children'] for r in results),
        assigned_children_below_gate_all_eight_seeds=sum(r['assigned_children_below_gate_all_eight_seeds'] for r in results),
        assigned_children_above_gate_any_seed=sum(r['assigned_children_above_gate_any_seed'] for r in results),
        another_unassigned_query_may_still_match_reference=True,
        full_candidate_unreachability_or_extraction_accuracy_not_claimed=True,
        optimizer_updates=0, validation_or_test_or_protected_diagnostic_reads=0,
        full_contract_pass=False, seconds=time.perf_counter() - start,
        versions=[dict(version_key=r['version_key'], reference_children=r['reference_children'],
            below_gate_all_seeds=r['assigned_children_below_gate_all_eight_seeds'],
            receipt_sha256=digest(OUT / 'versions' / (r['version_key'] + '.json'))) for r in results])
    write(OUT / 'receipt.json', result)
    print('ACTUAL_COMPLETE_REFERENCE_GATE_AUDIT_FINISHED', result['audited_versions'],
          result['assigned_children_below_gate_all_eight_seeds'], '/', result['reference_children'], flush=True)


if __name__ == '__main__':
    main()
