"""Real-data initialization, admission-gradient and geometry isolation control.

Eight optimizer updates train only the new admission correction on an actual
complete TRAIN assignment. No output accuracy or full-data training is claimed.
"""
from pathlib import Path
import argparse
import shutil

import numpy as np
import torch

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_runtime_aligned_training_samples import RuntimeAlignedTrainingAssignments
from weld_pipeline.server_legacy_anchored_full_weld_model import LegacyAnchoredFullWeldModel
from weld_pipeline.server_reference_recall_weld_model import ReferenceRecallWeldModel, reference_responsibility_loss, ARCHITECTURE
from weld_pipeline.sjg_local_runtime_904 import digest, read, write
from tools.train_full_weld_anchored import TrainingTensors as Converter
from tools.prepare_full_task_supervision import verify as verify_supervision

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main(experiment_name):
    if Path(experiment_name).name != experiment_name:
        raise ValueError('Control experiment must be a single directory name')
    OUT = BASE / experiment_name
    OUT.mkdir(exist_ok=False)
    checkpoint = OUT / 'anchored_initialization.pt'
    shutil.copyfile(BASE / 'full_weld_anchored_v2/last.pt', checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline = LegacyAnchoredFullWeldModel().to(device).eval()
    baseline.load_state_dict(state['state_dict'])
    model = ReferenceRecallWeldModel().to(device).eval()
    model.inherit_anchored(state['state_dict'])
    dataset = FullShardDataset(ROOT)
    # This real TRAIN case had a complete retained input and a failed native
    # output. Its references are authorized training data, never held labels.
    job_key = 'cloud_66daa89ba06a47e2'
    inp = dataset.input_case(job_key)
    version = dataset.jobs[job_key]['versions'][0]
    folder = BASE / 'full_task_supervision_v1' / job_key / version['key']
    verify_supervision(folder, dataset, version)
    supervision = dataset.supervision(version['key'])
    with np.load(folder / 'owner_witnesses.npz', allow_pickle=False) as arrays:
        witnesses = {name: arrays[name] for name in arrays.files}
    seed = np.random.SeedSequence([1031, 0, list(dataset.versions).index(version['key']), 0])
    sample = RuntimeAlignedTrainingAssignments(inp).build(supervision, witnesses, seed)
    converter = Converter(state['mean'], state['std'], device)
    batch = converter.batch(inp, sample)
    q = converter.tensor(sample['features'][:, 456])
    with torch.inference_mode():
        old = baseline.predict(baseline.encode_queries(**batch), q)
        initial = model.predict(model.encode_queries(**batch), q)
    if any(not torch.equal(old[name], initial[name]) for name in old):
        raise AssertionError('Zero-correction initialization changed a real anchored output')
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.reference_delta.parameters(), lr=.001, eps=1e-6, weight_decay=.0001)
    history = []
    for step in range(8):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.predict(model.encode_queries(**batch), q)
        losses = reference_responsibility_loss(prediction)
        loss = losses['reference_assigned_positive'] + .001 * losses['reference_prior_correction']
        loss.backward()
        if any(p.grad is not None for name, p in model.named_parameters() if not name.startswith('reference_delta.')):
            raise AssertionError('Admission-only gradient reached geometry or the frozen original model')
        torch.nn.utils.clip_grad_norm_(model.reference_delta.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        history.append(float(loss.detach()))
    with torch.inference_mode():
        after = model.predict(model.encode_queries(**batch), q)
    for name, value in model.state_dict().items():
        if not name.startswith('reference_delta.') and not torch.equal(value.cpu(), before[name]):
            raise AssertionError('Admission control changed another model parameter')
    geometry_names = [name for name in old if name != 'reference_score']
    if any(not torch.equal(old[name], after[name]) for name in geometry_names):
        raise AssertionError('Admission-only correction changed geometry, fields, frames or parent inputs')
    if not history[-1] < history[0] or torch.equal(after['reference_score'], old['reference_score']):
        raise AssertionError('Real admission correction failed to learn from assigned complete reference queries')
    result = dict(status='PASS_ACTUAL_REFERENCE_RECALL_INITIALIZATION_AND_GRADIENT_CONTROL_NOT_MODEL_ACCURACY',
        architecture=ARCHITECTURE, job_key=job_key, version_key=version['key'], complete_reference_children=len(q),
        initial_all_prediction_tensors_bitwise_equal=True, original_legacy_parameters_unchanged=True,
        all_non_admission_parameters_and_outputs_bitwise_unchanged=True, actual_optimizer_updates=8,
        unassigned_queries_relabelled_background=0, physical_absence_labels_fabricated=False,
        initial_assigned_query_logits_above_gate=int((old['reference_score'] > 0).sum()),
        after_assigned_query_logits_above_gate=int((after['reference_score'] > 0).sum()),
        complete_output_or_independent_accuracy_not_evaluated=True, history=history,
        frozen_initialization_sha256=digest(checkpoint), initialization_optimizer_steps=state['optimizer_steps'],
        full_data_training_not_claimed=True, held_or_diagnostic_label_reads=0,
        source_sha256=digest(__file__), model_source_sha256=digest(ROOT / 'weld_pipeline/server_reference_recall_weld_model.py'))
    write(OUT / 'receipt.json', result)
    print('ACTUAL_REFERENCE_RECALL_CONTROL_PASS', len(q), history[0], history[-1], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', default='reference_recall_model_control_v3_r2')
    args = parser.parse_args()
    main(args.experiment_name)
