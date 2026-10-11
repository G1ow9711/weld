"""Reuse every original D3 semantic control and exercise unsampled compact lines."""
import argparse
from pathlib import Path
from types import FunctionType

import numpy as np

from tools import verify_exact_endpoint_reference as original
from weld_pipeline.server_compact_endpoint_reference import compile_compact_endpoint_reference
from weld_pipeline.server_predicted_parent_pool import decoded_subset
from weld_pipeline.server_full_prediction_runtime import compact_segment, sampled_segment
from weld_pipeline.sjg_local_runtime_904 import digest, write

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT/'weld_pipeline/server_compact_endpoint_reference.py'


def main(output_name):
    original_helper = ROOT/'weld_pipeline/server_exact_endpoint_reference.py'
    def bound_digest(path):
        return digest(HELPER if Path(path) == original_helper else path)
    def bound_write(path, value):
        a = original.line([0.,0.,0.],[10.,0.,0.])
        b = original.line([10.,0.,0.],[20.,0.,0.])
        wrong = original.line([0.,20.,0.],[10.,20.,0.])
        truth = original.target([original.parent([a,b],['SHARED'])])
        parents = [original.parent([compact_segment(wrong),compact_segment(b)],['SHARED']),
                   original.parent([compact_segment(wrong),compact_segment(wrong)],['SHARED'])]
        table, audit = compile_compact_endpoint_reference(parents,truth,decoded_subset,1)
        expected, _ = original.compile_reference_table(decoded_subset(parents,np.arange(2)),truth,[False])
        for key in ('costs','tp','child_counts','invalid_children'):
            np.testing.assert_array_equal(table[key],expected[key])
        assert table['tp'][0,0] == 1 and audit['sampled_parents'] == 1
        stale = compact_segment(a)
        stale['points'] = original.line([0.,20.,0.],[10.,20.,0.])['points']
        actual = [original.parent([stale])]
        one = original.target([original.parent([a])])
        stale_table, _ = compile_compact_endpoint_reference(actual,one,decoded_subset)
        assert stale_table['tp'][0,0] == 1, 'Stored points must not override the original frozen sampler'
        for segment in [a,b,wrong,original.line([0.,0.,0.],[np.nextafter(0.,1.),1.e-12,0.])]:
            compact=compact_segment(segment)
            np.testing.assert_array_equal(np.asarray(sampled_segment(compact)['points'])[[0,-1]],
                                          np.asarray(compact['endpoints_mm'],np.float64))
        try:
            compile_compact_endpoint_reference(parents,truth,original.decode)
        except ValueError:
            pass
        else:
            raise AssertionError('Unverified compact endpoint decoder was accepted')
        value.update(reused_original_control_source_sha256=digest(original.__file__),
            compact_sampled_endpoints_bitwise_equal=True,
            unavailable_compact_row_unsampled_and_FP_retained=True,
            original_runtime_decoder_required_for_compact_rows=True,
            original_sampler_ignores_stored_points_preserved=True)
        write(path,value)
    environment=dict(original.main.__globals__)
    environment.update(compile_exact_endpoint_reference=compile_compact_endpoint_reference,
                       __file__=__file__,digest=bound_digest,write=bound_write)
    verify=FunctionType(original.main.__code__,environment,original.main.__name__,
                        original.main.__defaults__,original.main.__closure__)
    verify(output_name)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-name',required=True)
    main(parser.parse_args().output_name)
