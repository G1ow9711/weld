"""Offline comparison of identical fixed queries through two inference paths."""
import argparse
from pathlib import Path
import numpy as np
from tools.query_reference_ordered_runtime import Inputs,BASE
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

def run(key):
 model=BASE/'positive_query_head_v1/ordered_context_v1';r=read(model/'receipt.json');folder=Path(r['binding_paths'][key]);assert digest(folder/'receipt.json')==r['bindings'][key];records=read(folder/'records.json');features=np.load(folder/'features.npy');split='training' if key in r['training_keys'] else 'heldout';path=model/(key+'_'+split+'.npz');assert digest(path)==r['files'][path.name]
 with np.load(path) as z:expected=z['interval_ratios'].astype(np.float64);role=z['weld_type']
 inp=Inputs(key);ids=np.array([v['feature_rows'][0]//2*4+v['query_index'] for v in records]);assert len(ids)==len(expected);assert np.array_equal(inp.pid[ids//4],[v['profile'] for v in records]);max_feature=0.
 for start in range(0,len(ids),4096):
  actual=inp.features(ids[start:start+4096])[:,:457];target=features[start:start+4096];assert np.array_equal(actual,target);max_feature=max(max_feature,float(np.max(abs(actual-target))))
 difference=abs(inp.ratio[ids]-expected);physical=difference*inp.length[ids//4,None];ratio_max=float(difference.max());assert ratio_max<=2e-6;assert np.array_equal(inp.role[ids],role)
 report=dict(status='PASS',key=key,queries=len(ids),feature_max_difference=max_feature,ratio_max_difference=ratio_max,endpoint_mm_max_difference=float(physical.max()),roles_exact=True,tolerance_ratio=2e-6,scope='GPU batch/padding arithmetic equivalence on same fixed input queries; original geometry scoring tolerances unchanged',model_sha256=r['model_sha256'],runtime_receipt_sha256=digest(inp.folder/'receipt.json'),source_sha256=digest(__file__),fit_calls=0,DEV_reads=0)
 write(inp.folder/'positive_subset_runtime_equivalence.json',report);print(report,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);run(p.parse_args().key)
