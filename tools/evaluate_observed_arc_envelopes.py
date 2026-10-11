"""Offline TRAIN coverage of frozen XYZ-only circle proposals."""
import argparse
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key):
 folder=BASE/'observed_arc_envelopes_v1'/key;receipt=read(folder/'receipt.json');assert digest(folder/'profiles.json')==receipt['profiles_sha256'];profiles=read(folder/'profiles.json');curves=[CurveCarrier(p['representation']) for p in profiles];target=BASE/'labels'/key/'targets.json';rows=[]
 for pi,p in enumerate(read(target)['parents']):
  for ci,c in enumerate(p['children']):
   if c['original_kind']!='arc':continue
   curve=CurveCarrier(c['representation']);samples=curve.at(np.linspace(0,curve.length,65));errors=[]
   for carrier in curves:errors.append(float(carrier.project(samples)[1].max()))
   order=np.argsort(errors)[:3];rows.append(dict(parent=pi,child=ci,covered_65_samples=bool(errors and min(errors)<=3.),nearest=[dict(profile=int(i),max_error_mm=errors[i]) for i in order]))
 report=dict(key=key,arc_children=len(rows),covered_65_samples=sum(r['covered_65_samples'] for r in rows),rows=rows,profiles_sha256=digest(folder/'profiles.json'),target_sha256=digest(target),scope='65 GT analytic sample coverage only, not continuous/contact/parent D3 acceptance, not model output',GT_only_after_proposal_freeze=True,fit_calls=0,DEV_reads=0);write(folder/'coverage.json',report);print(key,report['covered_65_samples'],'/',len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);run(p.parse_args().key)
