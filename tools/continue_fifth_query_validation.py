"""Wait for the live original feature producer, then freeze ninth-head inference."""
import time,subprocess,sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];b=root/'server_rebuild/expansion_v1';key='new_8b1899c643ee';ready=b/'frozen_features/legacy_runtime'/('source_'+key)/'run329'/('source_'+key)/'receipt.json';deadline=time.monotonic()+1800
while not ready.exists():
 producer=Path('/proc/408878/cmdline')
 if not producer.exists() or b'continue_expansion_geometry_features.py' not in producer.read_bytes():raise RuntimeError('Feature producer is terminal before required receipt; inspect failure, do not restart blindly')
 if time.monotonic()>deadline:raise TimeoutError('Observation budget expired; producer is not thereby terminal')
 time.sleep(10)
for script in ['infer_positive_query_interval_proposals_nine.py','evaluate_all_query_proposal_capacity_nine.py']:
 subprocess.run([sys.executable,str(root/'tools'/script),'--key',key],cwd=root,check=True)
(b/'fifth_query_validation_complete.json').write_text(json.dumps(dict(key=key,status='FROZEN_NINE_FAMILY_HEAD_FULL_INPUT_QUERY_PROBE_COMPLETE',new_fit_calls=0,not_acceptance=True)))
