"""Audit model score versus actual predicted interval geometry, after freeze."""
from pathlib import Path
import argparse
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def run(key):
 folder=BASE/'joint_reference_full_output'/key;r=read(folder/'prediction_freeze.json');assert digest(folder/'predictions.npz')==r['predictions_sha256'];capacity=read(folder/'child_reference_capacity.json');assert capacity['runtime_receipt_sha256']==digest(folder/'prediction_freeze.json');z=np.load(folder/'predictions.npz');matches=np.zeros(len(z['valid']),bool);ids=sorted({i for row in capacity['rows'] for i in row['compatible_query_ids']});matches[ids]=True;selected=z['valid']&z['line_eligible']&(z['reference_occupancy_score']>=.5)
 model=BASE/'positive_query_head_v1/joint_reference_v1';mr=read(model/'receipt.json');positive=Path(mr['binding_paths'][key]);records=read(positive/'records.json');local=np.load(model/(key+'_heldout.npz'));query_ids=np.array([v['feature_rows'][0]//2*4+v['query_index'] for v in records]);diff=abs(z['interval_ratios'][query_ids]-local['interval_ratios']);score_diff=abs(z['reference_occupancy_score'][query_ids]-local['reference_occupancy_score']);assert diff.max()<=2e-6 and score_diff.max()<=2e-6;assert np.array_equal(z['weld_type'][query_ids],local['weld_type']);inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key);x=np.load(positive/'features.npy')
 for first in range(0,len(query_ids),4096):assert np.array_equal(inp.features(query_ids[first:first+4096]),x[first:first+4096])
 inp.close();scores=z['reference_occupancy_score'];buckets=[]
 for lo,hi in [(0,.1),(.1,.5),(.5,.9),(.9,1.00001)]:
  take=z['valid']&z['line_eligible']&(scores>=lo)&(scores<hi);buckets.append(dict(lo=lo,hi=min(hi,1.),queries=int(take.sum()),actual_interval_matching_queries=int((take&matches).sum())))
 report=dict(status='AUDIT_COMPLETE',key=key,selected_queries=int(selected.sum()),selected_actual_geometry_matching_queries=int((selected&matches).sum()),all_actual_matching_queries=int(matches.sum()),children_with_any_candidate=capacity['children_with_compatible_query'],children_with_selected_candidate=sum(any(selected[i] for i in row['compatible_query_ids']) for row in capacity['rows']),score_buckets=buckets,positive_subset_features_exact=True,positive_subset_ratio_max_difference=float(diff.max()),positive_subset_score_max_difference=float(score_diff.max()),physical_absence_claim=False,scope='Query-level geometry/score audit; does not account for duplicates or parent matching, final original D3 score remains authoritative',source_sha256=digest(__file__),fit_calls=0,DEV_reads=0);write(folder/'selection_audit.json',report);print(report,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);run(p.parse_args().key)
