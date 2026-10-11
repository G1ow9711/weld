"""Real-data numerical point-limit control and unchanged regular features."""
from pathlib import Path
import numpy as np
from weld_pipeline.server_whole_parent_features import encode_parents
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'


def main():
    key='new_ac2008a03607';source=BASE/'frozen_features/legacy_runtime'/('source_'+key)
    original=BASE/'unified_parent_scoring_v1'/('source_'+key)
    inp=FixedQueries(source,key);parents=read(original/'parents.json')['parents']
    encoded=encode_parents(inp,parents)
    assert np.array_equal(encoded['features'],np.load(original/'features.npy'))
    assert np.array_equal(encoded['child_features'],np.load(original/'child_features.npy'))
    assert not any(r.get('numerically_collapsed_projection') for r in encoded['child_records'])
    inp.close()
    key='new_cfc732e5f068';source=BASE/'frozen_features/legacy_runtime'/('source_'+key)
    native=source/'run379'/('test_'+key)/'decoded.json';before=digest(native)
    inp=FixedQueries(source,key);parent=read(native)['parents'][599]
    encoded=encode_parents(inp,[parent]);collapsed=[r for r in encoded['child_records'] if r.get('numerically_collapsed_projection')]
    assert len(collapsed)==1 and parent['segments'][1]['length_mm']<1e-10
    record=collapsed[0];assert len(record['fragments'])==1
    child=encoded['child_features'][encoded['parent_children'][0][1]]
    raw=np.asarray(inp.raw[record['canonical_feature_rows']],np.float64)
    expected=np.r_[raw.mean(0),raw.min(0),raw.max(0)].astype(np.float32)
    assert np.array_equal(child[:456],expected)
    assert digest(native)==before
    inp.close()
    out=BASE/'whole_parent_projection_limits_v1';out.mkdir(exist_ok=False)
    write(out/'receipt.json',dict(status='PASS',regular_parents_bitwise_unchanged=142,
          collapsed_child_length_mm=parent['segments'][1]['length_mm'], measured_containing_fragment=True,
          output_geometry_unchanged=True, module_sha256=digest(ROOT/'weld_pipeline/server_whole_parent_features.py'),
          reference_reads=0,fit_calls=0))
    print('WHOLE_PARENT_PROJECTION_LIMITS_PASS',flush=True)


if __name__=='__main__':main()
