"""Exercise customer scalar semantics, complete curves and signed CSV frames."""
import argparse
import copy
import csv
import io
import json
from pathlib import Path
import traceback

import numpy as np

from weld_pipeline.contract_curve_geometry import CircularArc, sample_arc
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.server_full_weld_csv_export import export_decoded, verify_roundtrip, field_encoding
from weld_pipeline.sjg_local_runtime_904 import write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
SOURCES = ['tools/verify_full_weld_csv_export.py', 'tools/export_full_weld_predictions.py',
           'weld_pipeline/server_full_weld_csv_export.py', 'weld_pipeline/runtime_csv_export_287.py',
           'weld_pipeline/contract_curve_geometry.py', 'weld_pipeline/world_set_runtime_285.py']


def main(args):
    if not args.experiment_name or args.experiment_name in ('.', '..') or Path(args.experiment_name).name != args.experiment_name:
        raise ValueError('Independent local control required')
    out = BASE / args.experiment_name; out.mkdir(exist_ok=False)
    sources = {name: digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(sources=sources, reference_geometry_reads=0, fit_calls=0))
    try:
        short = line_segment([-.0, -3., 2.], [.125, -3., 2.])
        short.update(weld_type='horizontal', predicted_side='UNKNOWN',
            endpoint_attributes=dict(back_wall=dict(state='NO_BLOCKER_OR_DISTANCE_GT_50', value_mm=None),
                start_base_edge=dict(state='DISTANCE_GT_50', value_mm=None),
                end_hole_height=dict(state='NO_HOLE', value_mm=None),
                start_height=dict(state='EXACT_MM', value_mm=743., confidence=.73)),
            raw_export_frame_axes=[[[0., -1., -0.], [1., 0., 0.], [0., 0., 1.]],
                                   [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]],
            frame_origins_mm=[[-.0, -3., 2.], [.125, -3., 2.]], frame_business_semantics_certified=False)
        arc = sample_arc(CircularArc([2., -3., 4.], 5., [0., 0., 1.], [1., 0., 0.], -np.pi))
        arc.update(weld_type='vertical', predicted_side='UNKNOWN', endpoint_attributes=dict(
            back_wall=dict(state='EXACT_MM', value_mm=12.75),
            start_height=dict(state='NOT_REQUIRED', value_mm=None)))
        circle = sample_arc(CircularArc([9., -2., 1.], 3., [0., 0., 1.], [1., 0., 0.], -2*np.pi, True))
        circle.update(weld_type='horizontal', predicted_side='UNKNOWN', endpoint_attributes={})
        decoded = dict(parents=[dict(child_count=2, segments=[short, arc], gaps=['POSITIVE_GAP'],
            is_closed=False, IsSplitedForDualRobot=True, HasMid=True),
            dict(child_count=1, segments=[circle], gaps=[], is_closed=True)],
            total_prediction_children=3, invalid_predictions=[], GT_or_STEP_reads=0)
        original = copy.deepcopy(decoded)
        folder = out / 'complete_geometry'
        schema = export_decoded(folder, 'csv_control', decoded)
        roundtrip = verify_roundtrip(folder, decoded)
        assert decoded == original and schema['published_rows'] == 3
        with (folder / 'welds.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert float(rows[0]['length_mm']) == .125
        assert rows[0]['back_wall_legacy_code'] == '-1'
        assert rows[0]['start_base_edge_legacy_code'] == '51'
        assert rows[0]['end_hole_height_legacy_code'] == '0'
        assert rows[0]['front_wall_value_mm'] == '' and rows[0]['front_wall_state'] == 'UNKNOWN'
        assert rows[0]['start_height_value_mm'] == '743.0' and rows[1]['start_height_value_mm'] == ''
        assert rows[1]['back_wall_required'] == 'True' and rows[1]['start_height_required'] == 'False'
        assert rows[1]['raw_frame_axes_state'] == 'UNKNOWN_OR_MALFORMED'
        assert all(row['unmodeled_child_flags_state'].startswith('UNKNOWN') for row in rows)
        assert json.loads((folder / 'curves/w000001.json').read_text())['arc_parameters']['sweep_rad'] < 0
        assert rows[2]['is_closed'] == 'True'
        # An exact scalar at the cutoff is distinct from the censored 51 code.
        assert field_encoding('start_base_edge', dict(state='EXACT_MM', value_mm=50), True)['legacy_code'] == 50
        bad = field_encoding('start_base_edge', dict(state='EXACT_MM', value_mm=51), True)
        assert bad['value_mm'] == 51 and not bad['encoding_valid'] and bad['legacy_code'] is None
        bad = field_encoding('back_wall', dict(state='EXACT_MM', value_mm=-7), True)
        assert bad['value_mm'] == -7 and not bad['encoding_valid'] and bad['legacy_code'] is None
        assert not field_encoding('back_wall', dict(state='NO_HOLE'), True)['encoding_valid']
        original_bytes = (folder / 'welds.csv').read_bytes()
        rows[0]['start_frame_X_y'] = '1.0'
        stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
        (folder / 'welds.csv').write_bytes(stream.getvalue().encode('utf-8-sig'))
        try:
            verify_roundtrip(folder, decoded)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed signed frame was accepted')
        (folder / 'welds.csv').write_bytes(original_bytes)
        verify_roundtrip(folder, decoded)
        for name, fixture in [('wrong_count', dict(decoded, total_prediction_children=2)),
                              ('reference_input', dict(decoded, GT_or_STEP_reads=1)),
                              ('invalid_not_dropped', dict(decoded, invalid_predictions=[dict(reason='invalid')]))]:
            try:
                export_decoded(out / name, 'csv_control', fixture)
            except ValueError:
                pass
            else:
                raise AssertionError('Unsafe or incomplete export was accepted: ' + name)
        empty = dict(parents=[], invalid_predictions=[], total_prediction_children=0, GT_or_STEP_reads=0)
        export_decoded(out / 'empty_prediction', 'empty', empty)
        assert verify_roundtrip(out / 'empty_prediction', empty)['complete_child_rows'] == 0
        assert sources == {name: digest(ROOT / name) for name in sources}
        write(out / 'receipt.json', dict(status='PASS_COMPLETE_MODEL_CSV_SEMANTICS_AND_SIGNED_ROUNDTRIP',
            sources=sources, all_ordered_parents_and_children_unchanged=True,
            arc_and_closed_curve_signed_parameters_retained=True, signed_zero_and_axis_roundtrip_checked=True,
            changed_signed_axis_rejected=True, short_child_retained=True, parent_Any_not_broadcast=True,
            wall_minus_one_edge_51_and_no_hole_zero_not_exact_distance=True,
            unknown_blank_not_zero=True, invalid_geometry_not_silently_dropped=True,
            censored_or_incompatible_states_not_rewritten=True, vertical_only_back_wall_required=True,
            empty_output_supported=True, roundtrip=roundtrip, reference_geometry_reads=0, fit_calls=0,
            business_frames_and_formal_customer_layout_not_certified=True, full_contract_pass=False))
        print('COMPLETE_MODEL_CSV_SEMANTICS_PASS', flush=True)
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
