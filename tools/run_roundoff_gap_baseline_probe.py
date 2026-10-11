"""Rebuild edges/solve/decode/select with frozen weights, isolated source10."""
from pathlib import Path
import time
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as runtime
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
from weld_pipeline.server_roundoff_gap_links import guarded_relative_link_blocks
import weld_pipeline.sjg_relative_forest_runtime_1139 as relative


def main(resume=False):
    key='10';source=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime/source_10';base=ROOT/'server_rebuild/roundoff_gap_baseline_probe_v1';case=base/'legacy_runtime/source_10'
    dest=case/'run379'
    if not resume:
        base.mkdir(exist_ok=False);case.mkdir(parents=True)
        for p in source.iterdir():
            if p.name=='run379' or not p.is_dir():continue
            (case/p.name).symlink_to(p,target_is_directory=True)
        dest.mkdir()
        for p in (source/'run379').iterdir():
            if p.is_file() and not p.name.startswith('screen_64'):(dest/p.name).symlink_to(p)
        write(base/'protocol.json',dict(source=str(source),source_screen_sha256=digest(source/'run379/screen_10.json'),guard_module_sha256=digest(ROOT/'weld_pipeline/server_roundoff_gap_links.py'),script_sha256=digest(__file__),scope='Frozen rebuilt node/features/weights; regenerate all connection features/probabilities and solve/decode/select with numerical gap guard. Raw XYZ upstream unchanged. Original artifacts not modified.',protected_DEV_reads=0))
    else:
        protocol=read(base/'protocol.json');assert protocol['guard_module_sha256']==digest(ROOT/'weld_pipeline/server_roundoff_gap_links.py')
        assert protocol['source_screen_sha256']==digest(dest/'screen_10.json')
        assert (base/'edge_graph_complete.json').exists() and (base/'solve_complete.json').exists()
        assert not (base/'decode_complete.json').exists()
        write(base/'resume_protocol.json',dict(initial_protocol_sha256=digest(base/'protocol.json'),resume_script_sha256=digest(__file__),resume_stage='decode',reuse_complete_graph_and_solution=True))
    runtime.CASE=case;runtime.BASE=base;relative.relative_link_blocks=guarded_relative_link_blocks
    with threadpool_limits(limits=2):
        for stage in (('decode','select') if resume else ('edge_graph','solve','decode','select')):
            start=time.perf_counter();runtime.forest_stage(key,stage);write(base/f'{stage}_complete.json',dict(stage=stage,seconds=time.perf_counter()-start));print('DONE',stage,flush=True)
    from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions
    from weld_pipeline.sjg_recall_set_worker_1162 import score_value
    target=read(ROOT/'server_rebuild/labels_v1/10/targets.json');pred=dest/'test_10/final_decoded.json';result=score_value(read_shared_predictions(pred),target,key)
    write(base/'evaluation.json',result);write(base/'receipt.json',dict(status='FROZEN_BASELINE_PROBE_COMPLETE',report=result['report'],decoded_sha256=digest(pred),scope='Single TRAIN baseline branch probe only; not full run446 or acceptance'))
    print(result['report'],flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');main(p.parse_args().resume)
