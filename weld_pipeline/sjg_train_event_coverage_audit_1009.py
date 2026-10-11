"""TRAIN-only query supervision coverage, not extraction or validation scores."""
import argparse
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_probability_corrected_queries_1007 import grid_layout, draw, labels_for


def event_coverage(profiles, supervision, nodes, q, labels):
    observations = {}
    for child in supervision['children'] + supervision['missing']:
        key = (child['parent'], child['child'])
        observations[key] = dict(parent=key[0], child=key[1], kind=child['kind'],
                                 occupancy_queries=0, direction_queries=0,
                                 start_peak_queries=0, end_peak_queries=0,
                                 start_offset_queries=0, end_offset_queries=0,
                                 same_carrier_has_both_peaks=False, closed_whole=False)
    for raw_node, events in supervision['by_profile'].items():
        node = int(raw_node)
        ids = np.flatnonzero(nodes == node)
        if not len(ids):
            continue
        local_q = q[ids]
        curve = CurveCarrier(profiles[node]['representation'])
        for event in events:
            item = observations[(event['parent'], event['child'])]
            inside = np.zeros(len(ids), bool)
            for lo, hi in event['intervals']:
                inside |= (local_q >= lo) & (local_q <= hi)
            item['occupancy_queries'] += int(inside.sum())
            direction = labels['target_mask'][ids, 4] & (labels['target'][ids, 4] == event['target_direction_forward'])
            item['direction_queries'] += int((inside & direction).sum())
            item['closed_whole'] |= bool(event['closed_whole'])
            if event['closed_whole']:
                continue
            peak_counts = []
            for head, endpoint in enumerate(('lo', 'hi')):
                delta = local_q - event[endpoint]
                reconstructed_delta = local_q + labels['offset'][ids, head] * 10 - event[endpoint]
                if curve.kind == 'arc' and curve.arc.is_closed:
                    delta = (delta + curve.length / 2) % curve.length - curve.length / 2
                    reconstructed_delta = (reconstructed_delta + curve.length / 2) % curve.length - curve.length / 2
                # float32 offset targets can have a few micrometres of roundoff.
                assigned = labels['offset_mask'][ids, head] & (abs(reconstructed_delta) <= 1e-5)
                peaks = int((assigned & (abs(delta) <= 2.)).sum())
                prefix = 'start' if head == 0 else 'end'
                item[prefix + '_peak_queries'] += peaks
                item[prefix + '_offset_queries'] += int(assigned.sum())
                peak_counts.append(peaks)
            item['same_carrier_has_both_peaks'] |= all(v > 0 for v in peak_counts)
    rows = list(observations.values())
    assert len(rows) == supervision['original_G']
    open_rows = [r for r in rows if not r['closed_whole']]
    closed_rows = [r for r in rows if r['closed_whole']]
    return dict(original_G=len(rows), open_children=len(open_rows), closed_children=len(closed_rows),
                occupancy_seen=sum(r['occupancy_queries'] > 0 for r in rows),
                directed_occupancy_seen=sum(r['direction_queries'] > 0 for r in rows),
                both_endpoint_peaks_seen=sum(r['start_peak_queries'] > 0 and r['end_peak_queries'] > 0 for r in open_rows),
                both_offset_targets_seen=sum(r['start_offset_queries'] > 0 and r['end_offset_queries'] > 0 for r in open_rows),
                both_peaks_on_same_carrier=sum(r['same_carrier_has_both_peaks'] for r in open_rows),
                closed_occupancy_seen=sum(r['occupancy_queries'] > 0 for r in closed_rows), children=rows)


def controls():
    from weld_pipeline.sjg_local_curve_supervision_915 import compatible_interval
    profile = dict(representation=dict(kind='line', endpoints_mm=[[0, 0, 0], [100, 0, 0]]))
    event = compatible_interval(profile, dict(kind='line', endpoints_mm=[[20, 0, 0], [40, 0, 0]]))
    event.update(parent=0, child=0)
    sup = dict(original_G=2, children=[dict(parent=0, child=0, kind='line')],
               missing=[dict(parent=1, child=0, kind='arc')], by_profile={'0': [event]})
    nodes = np.array([0, 0]);q = np.array([20., 40.])
    labels = labels_for([profile], sup, nodes, q)
    good = event_coverage([profile], sup, nodes, q, labels)
    assert good['original_G'] == 2 and good['both_endpoint_peaks_seen'] == 1
    labels['offset_mask'][:] = False
    hidden = event_coverage([profile], sup, nodes, q, labels)
    assert hidden['both_offset_targets_seen'] == 0
    assert hidden['occupancy_seen'] == 1
    return dict(missing_child_keeps_denominator=True, masked_offsets_do_not_count_as_exposure=True,
                observed_both_endpoints_counted=True, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-run', default='run190')
    parser.add_argument('--sampler', choices=['mixture', 'stratified'], default='mixture')
    args = parser.parse_args()
    make_layout, sample_queries = grid_layout, draw
    if args.sampler == 'stratified':
        from weld_pipeline.sjg_stratified_event_queries_1010 import event_layout, draw as stratified_draw
        make_layout, sample_queries = event_layout, stratified_draw
    data = OUT / 'run186';proposals = OUT / 'run185'
    protocol = read(data / 'registered_protocol.json')
    assert protocol['features'] == 'run166'
    manifest = read(data / 'training_manifest.json')
    rows = []
    for case in manifest['cases']:
        key = case['source_key']
        assert key in protocol['train'] and key not in ('64', '24')
        profiles_path = proposals / ('source_' + key) / 'profiles.json'
        binding = read(profiles_path.parent / 'input_freeze.json')
        assert digest(profiles_path) == binding['files']['profiles.json']
        assert digest(data / case['input_file']) == case['input_sha256']
        assert digest(data / case['label_file']) == case['label_sha256']
        profiles = read(profiles_path)
        sup_path = data / ('source_' + key + '_supervision.json')
        sup = read(sup_path)
        with np.load(data / case['input_file'], allow_pickle=False) as z:
            nodes, q = z['nodes'], z['q_mm']
        with np.load(data / case['label_file'], allow_pickle=False) as z:
            labels = {k: z[k] for k in z.files}
        old = event_coverage(profiles, sup, nodes, q, labels)
        layout = make_layout(profiles, sup)
        drawn = [sample_queries(layout, 100900 + i, count=2048) for i in range(8)]
        nn = np.concatenate([s['nodes'] for s in drawn]);qq = np.concatenate([s['q_mm'] for s in drawn])
        candidate = event_coverage(profiles, sup, nn, qq, labels_for(profiles, sup, nn, qq))
        row = dict(source_key=key, supervision_sha256=digest(sup_path), actual_run187_fixed_queries=old,
                   proposed_eight_resampled_epochs=candidate, proposed_optimizer_updates=0,
                   candidate_unique_grid_queries=len(np.unique(np.concatenate([s['grid_indices'] for s in drawn]))),
                   inference_grid_queries=int(layout['ptr'][-1]))
        if args.sampler == 'stratified':
            per_epoch = []
            for sample in drawn:
                observed = event_coverage(profiles, sup, sample['nodes'], sample['q_mm'],
                    labels_for(profiles, sup, sample['nodes'], sample['q_mm']))
                assert observed['both_endpoint_peaks_seen'] == observed['open_children']
                assert observed['closed_occupancy_seen'] == observed['closed_children']
                assert observed['occupancy_seen'] == observed['original_G']
                per_epoch.append(observed['both_endpoint_peaks_seen'] + observed['closed_occupancy_seen'])
            row['per_epoch_children_with_required_events'] = per_epoch
        rows.append(row)
        print('TRAIN_EVENT_EXPOSURE', key, 'OLD_BOTH_PEAKS', old['both_endpoint_peaks_seen'],
              'PROPOSED', candidate['both_endpoint_peaks_seen'], 'OPEN_G', old['open_children'], flush=True)
    assert sum(r['actual_run187_fixed_queries']['original_G'] for r in rows) == 363
    filename = 'TRAIN_event_exposure_audit.json' if args.sampler == 'mixture' else 'TRAIN_stratified_event_exposure_audit.json'
    write(OUT / args.output_run / filename,
          dict(controls=controls(), rows=rows, sampler=args.sampler, original_training_G=363, source_sha256=digest(__file__),
               old_queries_repeated_all_eight_epochs=True, proposed_queries_have_not_trained_any_model=True,
               endpoint_peak_radius_mm=2, known_offset_radius_mm=8, extraction_accuracy=False,
               optimizer_updates=0, end_utc=utc()))


if __name__ == '__main__':
    main()
