"""Actual native and visual finite curves through shared prediction and parent APIs.

An explicitly bounded input-component control, not complete workpiece accuracy.
Original coordinates/features are retained; complete queries/events of the two
selected finite input curves are visited with reference file access denied.
"""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from tools.export_full_dataset_split_view import local_name
from tools.probe_multiview_visual_assistance import load_xyz
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors
from weld_pipeline.server_supported_interval_prediction_runtime import SupportedIntervalPredictionRuntime, supported_parent_data
from weld_pipeline.server_supported_interval_weld_model import SupportedIntervalWeldModel
from weld_pipeline.server_supported_directed_parent_pool import geometry_fingerprint
from weld_pipeline.server_supported_interval_events import FiniteInputIntervalEvents, interval_child
from weld_pipeline.server_visual_augmented_input import VisualAugmentedInput
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, read, write


def run(args):
    torch.set_num_threads(2)
    base = ROOT / 'server_rebuild/full_dataset_v2'; out = base / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    dataset = PartitionedFullShardDataset(ROOT, base); key = 'cloud_dfea4a82bc9967c4'
    raw, _ = load_xyz(dataset.jobs[key]); original = dataset.input_case(key)
    addition_path = base / 'full_171_visual_assistance_and_candidates_20261013_v1' / (key + '_additional_profiles.json')
    addition = read(addition_path)
    native_pid = min((i for i, length in enumerate(original.lengths) if length >= 10.),
                     key=lambda i: original.lengths[i])
    extra = min(addition['additional_profiles'], key=lambda p: CurveCarrier(p['representation']).length)
    a, b = original.offsets[native_pid:native_pid + 2]
    native = SimpleNamespace(profiles=[original.profiles[native_pid]], lengths=original.lengths[[native_pid]],
        offsets=np.array([0, b - a]), spans=original.spans[a:b],
        features_at=lambda pid, q: original.features_at(native_pid, q) if pid == 0 else None,
        tokens=lambda pid: original.tokens(native_pid) if pid == 0 else None)
    inp = VisualAugmentedInput(native, dict(original_profile_count=1, additional_profiles=[extra]))
    checkpoint = base / local_name(args.training_control) / 'last.pt'
    receipt = read(checkpoint.parent / 'receipt.json')
    if digest(checkpoint) != receipt['checkpoint_sha256']: raise ValueError('Actual fitted control changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SupportedIntervalWeldModel().eval(); model.load_state_dict(state['state_dict'], strict=True)
    priors = FrozenRun446Priors(ROOT)
    labels = [str((ROOT / v['label_directory']).resolve()) for _, v in dataset.versions.values()]
    attempted = []
    def deny(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes, Path)):
            path = str(Path(values[0]).resolve())
            if any(path == p or path.startswith(p + '/') for p in labels):
                attempted.append(path); raise AssertionError('Prediction attempted a reference read')
    sys.addaudithook(deny)
    runtime = SupportedIntervalPredictionRuntime(model, state['mean'], state['std'], query_batch=4,
        maximum_event_query_product=128, maximum_children=25_000, reference_probability=0.,
        whole_pair_beam=args.whole_pair_beam)
    input_sha = digest(dataset.shards / key / 'receipt.json')
    with threadpool_limits(limits=2):
        proposals = runtime.collect(inp, raw, input_sha)
        pool = supported_parent_data(proposals, model, priors, max_chain_children=2,
                                     max_parents=100_000, max_link_edges=2_000_000)
    counts = proposals['audit']['counts']
    assert counts['input_profiles_visited'] == 2 and len(proposals['children']) > 0
    assert {c['profile'] for c in proposals['children']} == {0, 1}
    producer = FiniteInputIntervalEvents(raw, input_sha); banks = [producer.build(p,
        native.spans[:, :2] if i == 0 else [[0., inp.lengths[i]]]) for i, p in enumerate(inp.profiles)]
    for c in proposals['children']:
        bank = banks[c['profile']]; positions = bank['positions_mm']
        assert c['input_interval_mm'] == positions[c['input_event_pair']].tolist()
        assert c['length_mm'] > 0 and c['finite_input_geometry_preserved']
        assert np.isfinite(c['frame_origins_mm']).all() and np.isfinite(c['raw_export_frame_axes']).all()
    assert all(p['geometry_preserved_exactly'] for p in pool['pool']['parents'])
    np.savez_compressed(out / 'child_inputs.npz', **{k: proposals[k] for k in ['child_tokens', 'query_xyz', 'query_bases']})
    write(out / 'children.json', proposals['children'])
    write(out / 'receipt.json', dict(status='PASS_ACTUAL_NATIVE_AND_VISUAL_COMPLETE_CURVE_OUTPUT_AND_PARENT_INPUTS',
        actual_control_checkpoint_sha256=digest(checkpoint), native_original_profile=native_pid,
        curve_lengths_mm=inp.lengths.tolist(), children=len(proposals['children']),
        parents=len(pool['pool']['parents']), parent_tokens=len(pool['flat']),
        input_audit=proposals['audit'], parent_audit=pool['pool']['audit'],
        reference_file_open_attempts=attempted, all_selected_curve_queries_and_events_scored=True,
        exact_finite_input_geometry_and_fields=True, whole_workpiece_output_claimed=False,
        nullable_parent_selection_still_to_train=True, full_extraction_accuracy_claimed=False,
        source_sha256=digest(ROOT / 'weld_pipeline/server_supported_interval_prediction_runtime.py')))
    print('SUPPORTED_INTERVAL_INPUT_ONLY_OUTPUT_PASS', len(proposals['children']), len(pool['pool']['parents']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--experiment-name', required=True)
    p.add_argument('--whole-pair-beam', type=int, default=64)
    p.add_argument('--training-control', default='supported_interval_complete_training_control_20261013_v3'); run(p.parse_args())
