"""Post-freeze child/parent availability diagnostics; no model updates."""
import sys,time,subprocess,traceback,collections
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'bounded_raw_capacity_diagnostics_followup'
def main():
 OUT.mkdir(exist_ok=False);ready=BASE/'bounded_raw_endpoint_evaluation_followup/receipt.json';proc=Path('/proc/774864');identity=(proc/'stat').read_text().split()[21] if proc.exists() else None;deadline=time.monotonic()+14400
 while not ready.exists():
  if identity is None or not proc.exists() or (proc/'stat').read_text().split()[21]!=identity:raise RuntimeError('Evaluation producer terminal without receipt')
  if time.monotonic()>deadline:raise TimeoutError('Observation expired; producer not stopped')
  time.sleep(5)
 rows=[]
 for arm in ['control','raw']:
  for key in ['new_cfc732e5f068','new_ceeae49d6777']:
   with (OUT/(arm+'_'+key+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/evaluate_bounded_raw_child_capacity.py'),'--arm',arm,'--key',key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   f=BASE/('bounded_raw_'+arm+'_full_output_v1')/key;r=read(f/'child_reference_capacity.json');groups=collections.defaultdict(list)
   for child in r['rows']:groups[child['parent']].append(bool(child['compatible_query_ids']))
   row=dict(arm=arm,key=key,GT_children=r['GT_children'],children_with_compatible_query=r['children_with_compatible_query'],queries_matching_reference_child=r['unique_queries_matching_any_GT_child'],parents_with_all_children_available=sum(all(v) for v in groups.values()),children_in_fully_available_parents=sum(len(v) for v in groups.values() if all(v)),actual_new_D3=read(f/'new_score.json')['categories']['overall'],capacity_sha256=digest(f/'child_reference_capacity.json'));rows.append(row);print(row,flush=True)
 write(OUT/'receipt.json',dict(status='POST_FREEZE_CHILD_CAPACITY_DIAGNOSTICS_COMPLETE',rows=rows,scope='Availability does not prove unique matching, parent assembly or physical weldability. Complete actual D3 reported separately.',fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
