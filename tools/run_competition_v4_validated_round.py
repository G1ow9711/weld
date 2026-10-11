"""One sequential experimental round, gated on the already running control job."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main(pid):
 control=BASE/'competition_geometry_v3_10.json';deadline=time.monotonic()+900
 while not control.exists():
  proc=Path('/proc')/str(pid)/'cmdline'
  if not proc.exists() or b'check_parent_competition_geometry_v3.py\x00--key\x0010' not in proc.read_bytes():raise RuntimeError('Original validation process terminal without result; no restart')
  if time.monotonic()>deadline:raise TimeoutError('Observation timeout; original validation was not stopped')
  print('WAIT_EXISTING_CONTROL',pid,flush=True);time.sleep(15)
 assert json.loads(control.read_text())['all_pass']
 for script in ['verify_competition_v4_unchanged_open_cases','train_parent_competition_head_v4','infer_unified_parent_competition_support_v5','evaluate_unified_parent_competition_support_v5','infer_unified_parent_competition_dual_heldout','evaluate_unified_parent_competition_dual_heldout']:
  print('START',script,flush=True)
  subprocess.run([sys.executable,str(ROOT/'tools'/(script+'.py'))],cwd=ROOT,check=True)
  print('DONE',script,flush=True)
 print('ROUND_COMPLETE_NOT_PROMOTED',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--existing-control-pid',required=True,type=int);main(p.parse_args().existing_control_pid)
