"""Conservative moment-based candidate retrieval, not duplicate classification."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'migration_checks';F=BASE/'input_geometry_fingerprints_v1'
def main():
 source=read(F/'summary.json');rows=source['inputs'];descriptors=[]
 for r in rows:
  descriptors.append(np.r_[np.sqrt(np.maximum(0,r['covariance_eigenvalues_raw_units_squared'])),r['sample_radius_quantiles_raw_units'][1:]])
 x=np.asarray(descriptors);scale=np.sqrt(np.array([sum(r['covariance_eigenvalues_raw_units_squared']) for r in rows]));assert np.all(scale>0);relative=np.max(abs(x[:,None,:]-x[None,:,:])/np.maximum(scale[:,None,None],scale[None,:,None]),axis=2);np.fill_diagonal(relative,np.inf);pairs=set()
 for i in range(len(rows)):
  for j in np.argsort(relative[i])[:5]:pairs.add(tuple(sorted((i,int(j)))))
  for j in np.flatnonzero(relative[i]<=.025):pairs.add(tuple(sorted((i,int(j)))))
 records=[dict(a=rows[i]['pcd_sha256'],b=rows[j]['pcd_sha256'],a_directory=rows[i]['directory'],b_directory=rows[j]['directory'],max_relative_descriptor_difference=float(relative[i,j]),within_conservative_retrieval_band=bool(relative[i,j]<=.025)) for i,j in sorted(pairs,key=lambda p:relative[p])]
 write(BASE/'geometry_duplicate_pair_candidates_v1.json',dict(pairs=records,inputs=len(rows),candidate_pairs=len(records),nearest_neighbors_per_input=5,relative_descriptor_retrieval_band=.025,threshold_is_not_material_identity_or_duplicate_proof=True,GT_reads=0,STEP_reads=0,split_assigned=False,cropped_variant_coverage_not_proved=True,fingerprint_summary_sha256=digest(F/'summary.json'),source_sha256=digest(__file__)));print('CANDIDATE_PAIRS',len(records),'WITHIN_BAND',sum(r['within_conservative_retrieval_band'] for r in records),flush=True)
if __name__=='__main__':main()
