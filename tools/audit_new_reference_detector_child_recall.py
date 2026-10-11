"""Post-freeze query detector child-coverage diagnosis, no endpoint oracle export."""
import argparse
import numpy as np
from pathlib import Path
from weld_pipeline.server_fixed_fragment_queries import FixedQueries,GRID
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def run(key,arm):
 folder=BASE/('reference_detector_'+arm+'_full_output_v1')/key;freeze=read(folder/'prediction_freeze.json');assert digest(folder/'predictions.npz')==freeze['predictions_sha256'];z=np.load(folder/'predictions.npz');ids=np.flatnonzero(z['valid']&z['line_eligible']&(z['reference_occupancy_score']>=.5));inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key);querypid=inp.pid[ids//4];span=inp.spans[inp.canonical[ids//4],:2];q=span[:,0]+GRID[ids%4]*(span[:,1]-span[:,0]);byprofile={}
 for pid in np.unique(querypid):byprofile[int(pid)]=q[querypid==pid]
 witness=BASE/'exhaustive_carrier_supervision_v1/positive_intervals'/key;r=read(witness/'receipt.json');assert digest(witness/'witnesses.json')==r['witnesses_sha256'];rows=read(witness/'witnesses.json')['positive_witnesses'];covered=set();same_profile_event=set();event=read(BASE/'sampled_surface_boundary_audit_v1'/key/'receipt.json');event_ok={(v['parent'],v['child'],v['profile']) for v in event['records'] if v['passing_interval'] is not None}
 for w in rows:
  values=byprofile.get(w['profile'])
  if values is None:continue
  lo,hi=w['interval_mm']
  if np.any((values>=lo)&(values<=hi)):
   child=(w['parent'],w['child']);covered.add(child)
   if (*child,w['profile']) in event_ok:same_profile_event.add(child)
 truth=read(BASE/'labels'/key/'targets.json');complete=sum(all((pi,ci) in covered for ci in range(len(p['children']))) for pi,p in enumerate(truth['parents']));report=dict(key=key,selected_queries=len(ids),reference_children_with_selected_occupancy_query=len(covered),reference_parents_all_children_with_selected_query=complete,selected_query_and_previous_sampled_event_witness_on_same_profile=len(same_profile_event),G=sum(len(p['children']) for p in truth['parents']),covered_children=sorted(covered),covered_with_prior_event_witness=sorted(same_profile_event),scope='Diagnostic necessary coverage under frozen .5 detector. Does not predict endpoints or certify assembly. Event match only prior audited profiles, not exhaustive.',fit_calls=0,threshold_changes=0,physical_absence_claim=False,model_sha256=freeze['model_sha256'],source_sha256=digest(__file__));write(folder/'reference_query_child_coverage.json',report);inp.close();print({k:v for k,v in report.items() if not isinstance(v,list)},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--arm',required=True,choices=['empirical','balanced']);a=p.parse_args();run(a.key,a.arm)
