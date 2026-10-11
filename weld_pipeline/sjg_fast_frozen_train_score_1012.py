"""User-requested immediate score of completed, frozen TRAIN predictions.

This does not replace the all-nine independent score. It reads no held-out
targets and never changes the checkpoint, decoder, thresholds or predictions.
"""
import gc
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc
from weld_pipeline.sjg_local_event_score_932 import geometric_match


def main():
    keys = ['21', '4', '10', '25', '36', '42']
    cohort = {c['source_key']: c for c in read(OUT / 'run98/local_cohort.json')['cases']}
    checkpoint = OUT / 'run187/final.pt';checkpoint_sha = digest(checkpoint)
    bindings = {}
    for key in keys:
        assert cohort[key]['role'] == 'TRAIN'
        folder = OUT / 'run188' / ('source_' + key)
        binding = read(folder / 'input_output_freeze.json')
        assert binding['source_key'] == key
        assert digest(folder / 'predictions.json') == binding['files']['predictions.json']
        bindings[key] = binding
    dest = OUT / 'run190'
    write(dest / 'fast_TRAIN_score_protocol.json', dict(source_keys=keys, partial_TRAIN_only=True,
        source_sha256=digest(__file__), checkpoint_sha256=checkpoint_sha,
        all_selected_prediction_hashes_verified_before_target_reads=True,
        user_requested_immediate_partial_result=True, complete_nine_score_still_pending=True,
        fixed_directed_endpoint_tolerance_mm=3, bidirectional_curve_sampling_mm=1,
        all_false_predictions_retained=True, full_contract_pass=False, start_utc=utc()))
    scores = [];endpoint = [];curve_error = []
    for key in keys:
        case = cohort[key]
        assert digest(OUT / case['target_file']) == case['target_sha256']
        target = read(OUT / case['target_file'])
        predictions = read(OUT / 'run188' / ('source_' + key) / 'predictions.json')
        score = geometric_match(predictions, target)
        assert score['G'] == case['original_G']
        for match in score['matches']:
            endpoint.extend(match['endpoint_errors_mm'])
            curve_error.append(match['bidirectional_sampled_maximum_mm'])
        write(dest / ('fast_TRAIN_source_' + key + '_geometry.json'), score)
        row = dict(source_key=key, **{k: score[k] for k in ('G', 'TP', 'FP', 'FN', 'predicted', 'TP_over_G', 'FP_over_G')},
                   inference_seconds=bindings[key]['elapsed_seconds'], queries=bindings[key]['queries'])
        scores.append(row)
        print('FROZEN_TRAIN_SCORE', key, 'TP', score['TP'], '/', score['G'], 'FP', score['FP'], flush=True)
        del predictions, target, score;gc.collect()
    totals = {key: sum(row[key] for row in scores) for key in ('G', 'TP', 'FP', 'FN', 'predicted', 'queries', 'inference_seconds')}
    totals.update(TP_over_G=totals['TP']/totals['G'], FP_over_G=totals['FP']/totals['G'])
    def stats(values):
        return dict(count=len(values), mean_mm=float(np.mean(values)), p95_mm=float(np.quantile(values, .95)), max_mm=float(np.max(values))) if values else None
    assert digest(checkpoint) == checkpoint_sha
    result = dict(status='COMPLETED_PARTIAL_TRAIN_GEOMETRY_SCORE', rows=scores, totals=totals,
                  matched_only_endpoint_error=stats(endpoint), matched_only_curve_error=stats(curve_error),
                  heldout_accuracy_not_established=True, full_contract_pass=False,
                  weld_type_parent_structure_E16_not_assessed=True, checkpoint_sha256=checkpoint_sha,
                  checkpoint_unchanged=True, optimizer_updates=0, end_utc=utc())
    write(dest / 'fast_TRAIN_score_summary.json', result)
    print('PARTIAL_TRAIN_TOTAL', totals, flush=True)
    print('MATCHED_ONLY_ENDPOINT_ERROR', result['matched_only_endpoint_error'], flush=True)
    print('MATCHED_ONLY_CURVE_ERROR', result['matched_only_curve_error'], flush=True)


if __name__ == '__main__':
    main()
