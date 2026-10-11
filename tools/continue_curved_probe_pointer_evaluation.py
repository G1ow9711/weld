from pathlib import Path
import os,time,subprocess,sys
from weld_pipeline.sjg_local_runtime_904 import write
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
while not (BASE/'curved_probe_boundary_pointer_v1/receipt.json').exists():
    os.kill(59752,0)
    assert Path('/proc/59752/stat').read_text().split(') ',1)[1].split()[0]!='Z','Training exited without receipt'
    time.sleep(10)
for script in ['infer_curved_probe_boundary_pointer.py','evaluate_curved_probe_pointer.py']:
    print('START',script,flush=True)
    subprocess.run([sys.executable,str(ROOT/'tools'/script)],cwd=ROOT,check=True)
    print('COMPLETE',script,flush=True)
write(BASE/'curved_probe_pointer_all_candidates_v1/followup_receipt.json',dict(status='INFERENCE_AND_DIAGNOSTIC_COMPLETE'))
