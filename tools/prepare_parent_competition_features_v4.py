"""Input-only pair features for learned competition between parent hypotheses.

Coordinates are never merged; these are learned selection features, not claims of
material identity. Candidate pairs: 16 nearest parent centres plus radius 25 mm.
"""
from pathlib import Path
import time
import numpy as np
from weld_pipeline.server_parent_competition_v4 import build_pair_features
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def run(key):
 start=time.perf_counter();f=BASE/('source_'+key);dest=f/'competition_v4';dest.mkdir(exist_ok=False);r=read(f/'receipt.json');assert digest(f/'parents.json')==r['files']['parents.json'];parents=read(f/'parents.json')['parents'];meta=read(f/'child_metadata.json');pc=read(f/'parent_children.json');pairs,x=build_pair_features(parents,meta,pc)
 np.save(dest/'pairs.npy',pairs);np.save(dest/'features.npy',x);write(dest/'receipt.json',dict(source=key,schema='PARENT_COMPETITION29_V4',pairs=len(pairs),columns=29,nearest_other_parents=16,centre_radius_mm=25.,radius_is_candidate_search_not_duplicate_threshold=True,nearest_distance_ties_complete=True,tie_tolerance_mm=1e-7,analytic_per_child_arclength=True,points_per_child=33,all_child_endpoints_preserved=True,no_gap_interpolation=True,full_circle_supported=False,old_model_incompatible=True,sampling_module_sha256=digest(ROOT/'weld_pipeline/server_parent_geometry_sampling.py'),feature_round_decimals=6,feature_module_sha256=digest(ROOT/'weld_pipeline/server_parent_competition_v4.py'),no_coordinate_merging=True,GT_reads=0,DEV_reads=0,parents_sha256=digest(f/'parents.json'),input_receipt_sha256=digest(f/'receipt.json'),files={n:digest(dest/n) for n in ['pairs.npy','features.npy']},source_sha256=digest(__file__),seconds=time.perf_counter()-start));print(key,len(pairs),round(time.perf_counter()-start,2),flush=True)
if __name__=='__main__':
 for k in ['21','BM79A_native_local','10','25','36']:run(k)
