"""Compare all samples and full original D3 of an already frozen real output."""
from pathlib import Path
import numpy as np
from weld_pipeline.server_compact_line_predictions import build_compact_lines,decode_compact_lines,line_segment_array
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.server_array_cached_d3_evaluation import score_native_parents_cached
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'compact_line_output_controls_v1'
def main():
 OUT.mkdir(exist_ok=False);key='new_cfc732e5f068';source=BASE/'boundary_event_provenance_selector_full_output_v1'/key;fr=read(source/'prediction_freeze.json');assert digest(source/'predictions.npz')==fr['files']['predictions.npz'];p=np.load(source/'predictions.npz');old=BASE/'joint_reference_full_output'/key;f=read(old/'prediction_freeze.json');assert digest(old/'predictions.npz')==f['predictions_sha256'];roles=np.load(old/'predictions.npz')['weld_type'][p['selected_query_ids']];profiles=read(BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key)/'profiles.json');decoded,compact=build_compact_lines(profiles,p['profile_ids'],p['interval_mm'],roles,p['selected_query_ids'],p['event_ids']);count=0
 for parent in decoded['parents']:
  segment=parent['segments'][0];reference=line_segment(*segment['endpoints_mm']);assert np.array_equal(segment['points'],np.asarray(reference['points']));assert {k:v for k,v in segment.items() if k not in ('points','weld_type','predicted_side')}=={k:v for k,v in reference.items() if k!='points'};count+=len(segment['points'])
 np.savez(OUT/'compact_predictions.npz',**compact)
 with np.load(OUT/'compact_predictions.npz') as loaded:restored=decode_compact_lines(loaded)
 for a,b in zip(decoded['parents'],restored['parents']):assert np.array_equal(a['segments'][0]['points'],b['segments'][0]['points'])
 target=read(BASE/'labels'/key/'targets.json');score=score_native_parents_cached(restored,target,[False]*len(target['parents']));expected=read(source/'new_score.json');assert score==expected
 write(OUT/'receipt.json',dict(status='PASS',key=key,predicted=decoded['total_prediction_children'],all_line_samples_exactly_preserved=count,full_D3_result_all_fields_equal=True,lossless_npz_roundtrip=True,compact_sha256=digest(OUT/'compact_predictions.npz'),source_prediction_freeze_sha256=digest(source/'prediction_freeze.json'),original_line_segment_source_sha256=digest('weld_pipeline/world_set_runtime_285.py'),compact_module_sha256=digest('weld_pipeline/server_compact_line_predictions.py'),scope='Evaluation storage representation equivalence, not timing acceptance or new extraction accuracy'));print('PASS',count,score['categories']['overall'],flush=True)
if __name__=='__main__':main()
