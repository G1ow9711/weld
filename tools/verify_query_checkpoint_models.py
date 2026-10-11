"""Exercise both real query architectures and reject incomplete TRAIN claims.

Uses actual input-only fixed positions. Bad architecture and forged-completion
documents are explicit negative controls, never model fitting evidence.
"""
import argparse
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_legacy_anchored_full_weld_model import LegacyAnchoredFullWeldModel
from weld_pipeline.server_reference_recall_weld_model import ReferenceRecallWeldModel, ARCHITECTURE
from weld_pipeline.server_query_checkpoint_model import load_query_model, verify_complete_query_fit
from weld_pipeline.server_full_prediction_runtime import CompleteQueryProposals
from weld_pipeline.server_runtime_aligned_training_samples import FIXED_QUERY_FRACTIONS
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main(args):
    if not args.experiment_name or Path(args.experiment_name).name != args.experiment_name:
        raise ValueError('A local control experiment name is required')
    out = BASE / args.experiment_name
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    dataset = FullShardDataset(ROOT)
    inp = dataset.input_case(args.job)
    denied = []
    label_folders = [str((ROOT / v['label_directory']).resolve())
                     for j in dataset.jobs.values() for v in j['versions']]
    def audit(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes, Path)):
            p = str(Path(values[0]).resolve())
            if any(p == folder or p.startswith(folder + '/') for folder in label_folders):
                denied.append(p)
                raise AssertionError('Input-only query control opened a reference')
    sys.addaudithook(audit)
    results = []
    loaded = []
    for source, fingerprint, cls in [(args.anchored_checkpoint, args.anchored_sha256, LegacyAnchoredFullWeldModel),
                                    (args.recall_checkpoint, args.recall_sha256, ReferenceRecallWeldModel)]:
        source = Path(source).resolve()
        source.relative_to(ROOT)
        if digest(source) != fingerprint:
            raise ValueError('Control checkpoint fingerprint differs')
        state = torch.load(source, map_location='cpu', weights_only=False)
        model, architecture = load_query_model(state)
        direct = cls().eval()
        direct.load_state_dict(state['state_dict'], strict=True)
        runtime = CompleteQueryProposals(model, state['mean'], state['std'], device='cpu')
        old = CompleteQueryProposals(direct, state['mean'], state['std'], device='cpu')
        checked = 0
        # ABI control only: full prediction is verified by a separate complete
        # pool consumer. No sampled subset is presented as model quality.
        ids = np.flatnonzero(np.diff(inp.offsets) > 0)
        selected = np.unique(ids[np.linspace(0, len(ids)-1, min(8, len(ids)), dtype=int)])
        for pid in selected:
            begin, end = inp.offsets[int(pid):int(pid)+2]
            spans = inp.spans[int(begin):min(int(end), int(begin)+3), :2]
            positions = np.unique((spans[:, :1] + np.asarray(FIXED_QUERY_FRACTIONS)
                                   * (spans[:, 1:2]-spans[:, :1])).ravel())
            tokens = torch.tensor(inp.tokens(int(pid))[0], dtype=torch.float32)
            a, _ = runtime.score_queries(inp, int(pid), positions, tokens)
            b, _ = old.score_queries(inp, int(pid), positions, tokens)
            if set(a) != set(b) or any(not np.array_equal(a[k], b[k]) for k in a):
                raise AssertionError('Factory dropped or changed an actual query prediction field')
            checked += len(positions)
        loaded.append(state)
        results.append(dict(architecture=architecture, checkpoint_sha256=fingerprint,
            sampled_input_queries=checked, all_prediction_fields_bitwise_equal=True))
    bad = []
    value = copy.copy(loaded[1]);value['architecture'] = 'LEGACY_ANCHORED_FULL_WELD_V2'
    bad.append(('recall_wrong_architecture', value))
    value = copy.copy(loaded[1]);value['state_dict'] = dict(value['state_dict'])
    del value['state_dict']['reference_delta.bias']
    bad.append(('missing_recall_parameter', value))
    value = copy.copy(loaded[0]);value['architecture'] = ARCHITECTURE
    bad.append(('anchored_falsely_tagged_recall', value))
    value = copy.copy(loaded[1]);value['state_dict'] = dict(value['state_dict'])
    value['state_dict']['unexpected.control.weight'] = torch.zeros(1)
    bad.append(('unexpected_parameter', value))
    rejected = []
    for name, value in bad:
        try:
            load_query_model(value)
        except (ValueError, RuntimeError):
            rejected.append(name)
        else:
            raise AssertionError('Invalid actual checkpoint architecture was accepted: '+name)
    # Preserve the genuine protocol bytes so rejection must be based on actual
    # missing visits, rather than an absent or mismatched protocol file.
    shutil.copyfile(BASE / 'full_weld_reference_recall_v3/protocol.json', out / 'protocol.json')
    protocol = read(out / 'protocol.json')
    epochs = protocol['settings']['epochs']
    forged = dict(status='ALL_TRAIN_REFERENCE_RECALL_FITTED_NOT_PROMOTED_NOT_COMPLETE_INFERENCE',
        negative_control_not_actual_fit_receipt=True, architecture=ARCHITECTURE,
        checkpoint_sha256=args.recall_sha256, protocol_sha256=loaded[1]['protocol_sha256'],
        training_versions=len(dataset.versions), validation_or_test_label_reads=0,
        epochs=epochs, completed_views=epochs*len(dataset.versions), optimizer_steps=loaded[1]['optimizer_steps'])
    write(out / 'invalid_complete_query_claim.json', forged)
    try:
        verify_complete_query_fit(ROOT, loaded[1], args.recall_sha256, out / 'invalid_complete_query_claim.json')
    except ValueError as exc:
        if 'actually fitted every registered TRAIN view' not in str(exc):
            raise AssertionError('Incomplete query was rejected for an unrelated reason') from exc
        rejected.append('forged_completion_without_actual_registered_visits')
    else:
        raise AssertionError('A real partial query checkpoint authorized full fitting')
    write(out / 'receipt.json', dict(status='PASS_ACTUAL_QUERY_ARCHITECTURES_AND_PARTIAL_FIT_REJECTION_NOT_MODEL_ACCURACY',
        job_key=args.job, architectures=results, rejected_invalid_checkpoints=rejected,
        qualified_reference_open_attempts=len(denied), held_label_reads=0, optimizer_updates=0,
        full_prediction_or_extraction_accuracy_not_evaluated=True,
        sources={n:digest(ROOT/n) for n in ['tools/verify_query_checkpoint_models.py',
            'weld_pipeline/server_query_checkpoint_model.py','tools/evaluate_full_weld_predictions.py']}))
    print('ACTUAL_QUERY_ARCHITECTURES_CONTROL_PASS', len(results), len(rejected), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--anchored-checkpoint', required=True)
    parser.add_argument('--anchored-sha256', required=True)
    parser.add_argument('--recall-checkpoint', required=True)
    parser.add_argument('--recall-sha256', required=True)
    main(parser.parse_args())
