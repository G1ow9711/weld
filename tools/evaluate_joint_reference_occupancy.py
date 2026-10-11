"""Frozen held-family occupancy evaluation, never a training-label extension."""
import argparse
import numpy as np
from tools.query_reference_utility_runtime import BASE
from weld_pipeline.server_fixed_fragment_queries import FixedQueries,GRID
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

def run(key):
 out=BASE/'joint_reference_full_output'/key;frozen=read(out/'prediction_freeze.json');protocol=read(BASE/'exhaustive_carrier_supervision_v1/held_evaluation_only/protocol.json');assert key in protocol['keys'] and frozen['model_sha256']==protocol['model_sha256'];assert digest(out/'predictions.npz')==frozen['predictions_sha256'];z=np.load(out/'predictions.npz');inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key);positive=np.zeros(inp.n,bool);folder=BASE/'exhaustive_carrier_supervision_v1/positive_intervals'/key;r=read(folder/'receipt.json');assert digest(folder/'witnesses.json')==r['witnesses_sha256']
 for w in read(folder/'witnesses.json')['positive_witnesses']:
  if w['kind']!='line':continue
  pid=w['profile'];lo,hi=w['interval_mm'];a=np.searchsorted(inp.pid,pid,'left');b=np.searchsorted(inp.pid,pid,'right');f=np.arange(a,b);span=inp.spans[inp.canonical[f],:2];q=span[:,0,None]+GRID*(span[:,1]-span[:,0])[:,None];ids=f[:,None]*4+np.arange(4);positive[ids[(q>=lo)&(q<=hi)]]=True
 eligible=z['valid']&z['line_eligible'];selected=eligible&(z['reference_occupancy_score']>=.5);cap=read(out/'child_reference_capacity.json');geometry=np.zeros(inp.n,bool);geometry[sorted({i for row in cap['rows'] for i in row['compatible_query_ids']})]=True
 report=dict(key=key,status='HELD_REFERENCE_OCCUPANCY_EVALUATION_ONLY',reference_positive_queries=int((eligible&positive).sum()),selected_queries=int(selected.sum()),reference_occupancy_TP=int((selected&positive).sum()),reference_occupancy_FP=int((selected&~positive).sum()),reference_occupancy_FN=int((eligible&positive&~selected).sum()),geometry_correct_queries=int((geometry&eligible).sum()),geometry_correct_also_reference_positive=int((geometry&eligible&positive).sum()),geometry_correct_selected=int((geometry&selected).sum()),reference_positive_score_quantiles=np.quantile(z['reference_occupancy_score'][positive&eligible],[0,.25,.5,.75,.9,.99,1]).tolist(),physical_negative_claim=False,fit_calls=0,threshold_changes=0,model_sha256=frozen['model_sha256'],witness_receipt_sha256=digest(folder/'receipt.json'),source_sha256=digest(__file__),scope='Query occupancy task metrics, not endpoint/parent extraction accuracy; held labels evaluation only');write(out/'reference_occupancy_evaluation.json',report);inp.close();print(report,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);run(p.parse_args().key)
