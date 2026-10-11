"""Compare cached scorer to completed original scorer on unchanged full output."""
from pathlib import Path
import time
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.server_array_cached_d3_evaluation import score_native_parents_cached
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 source=BASE/'joint_reference_full_output/new_cfc732e5f068';freeze=read(source/'prediction_freeze.json');assert (source/'receipt.json').exists();assert digest(source/'new_decoded.json')==freeze['new_output_sha256'];original=read(source/'new_score.json');start=time.perf_counter();decoded=read(source/'new_decoded.json');load_seconds=time.perf_counter()-start;target=read(BASE/'labels/new_cfc732e5f068/targets.json');start=time.perf_counter();cached=score_native_parents_cached(decoded,target,[False]*len(target['parents']));seconds=time.perf_counter()-start;assert original==cached;dest=BASE/'array_cached_d3_validation_v1';write(dest/'full_joint_output_equivalence.json',dict(status='PASS_FULL_OUTPUT_RESULT_EXACT',predicted_children=decoded['total_prediction_children'],original_score_sha256=digest(source/'new_score.json'),decoded_sha256=freeze['new_output_sha256'],model_sha256=freeze['model_sha256'],cached_score_seconds=seconds,json_load_seconds=load_seconds,original_inference_and_both_scores_total_seconds=read(source/'receipt.json')['seconds'],timings_not_like_for_like=True,all_result_fields_equal=True,no_gate_coordinate_or_sample_change=True));print(read(dest/'full_joint_output_equivalence.json'),flush=True)
if __name__=='__main__':main()
