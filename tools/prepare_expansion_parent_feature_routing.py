"""Make auditable path-only copies of the already-tested parent input builders."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'server_rebuild/expansion_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b,1)
rows=[]
for old,new in [('prepare_baseline_child_features','prepare_expansion_child_features'),('generate_residual_parent_hypotheses','generate_expansion_parent_hypotheses'),('prepare_unified_parent_scoring_inputs','prepare_expansion_parent_scoring_inputs')]:
 src=ROOT/'tools'/(old+'.py');dst=ROOT/'tools'/(new+'.py');assert not dst.exists();s=src.read_text()
 s=s.replace("server_rebuild/best_baseline_restore","server_rebuild/expansion_v1/frozen_features").replace("server_rebuild/child_residual_v1","server_rebuild/expansion_v1/child_residual_v1").replace("server_rebuild/parent_hypotheses_v1","server_rebuild/expansion_v1/parent_hypotheses_v1").replace("server_rebuild/unified_parent_scoring_v1","server_rebuild/expansion_v1/unified_parent_scoring_v1")
 if old=='prepare_baseline_child_features':
  s=replace_once(s,'def main():','def run(key):');s=replace_once(s,'DEST.mkdir(exist_ok=False);reports=[]','DEST.mkdir(parents=True,exist_ok=True);reports=[]');s=replace_once(s,"for key in ['21','BM79A_native_local','10','25','36']:",'for key in [key]:');s=replace_once(s,"DEST/'inputs_frozen.json'","DEST/('inputs_frozen_'+key+'.json')");s=replace_once(s,"if __name__=='__main__':main()","if __name__=='__main__':\n import argparse\n p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();assert a.key in {c['key'] for c in read(ROOT/'server_rebuild/expansion_v1/coverage_cohort_protocol.json')['cases']};run(a.key)")
 else:
  s=replace_once(s,"choices=['21','BM79A_native_local','10','25','36']","choices=[c['key'] for c in read(ROOT/'server_rebuild/expansion_v1/coverage_cohort_protocol.json')['cases']]")
  if old=='prepare_unified_parent_scoring_inputs':s=replace_once(s,"runtime.CASE=runtime.BASE/", "runtime.BASE=ROOT/'server_rebuild/expansion_v1/frozen_features';runtime.CASE=runtime.BASE/")
 dst.write_text(s);rows.append(dict(source=str(src.relative_to(ROOT)),source_sha256=sha(src),destination=str(dst.relative_to(ROOT)),destination_sha256=sha(dst)))
(OUT/'parent_feature_routing_sources.json').write_text(json.dumps(dict(changes='Only output/input routing, per-case invocation and registered CLI keys; source algorithms and frozen model paths unchanged',files=rows,fit_calls=0),indent=2)+'\n')
