from pathlib import Path
import numpy as np
from weld_pipeline.server_multiscale_chain_windows import chain_windows
from weld_pipeline.sjg_local_runtime_904 import write,digest

def identities(rows):return {min(tuple(r),tuple(r[::-1])) for r in rows}
for n in [4,5,6,9,18,37,128]:
    c=np.arange(n);windows=chain_windows(c);assert identities(windows)==identities(chain_windows(c[::-1]));assert len(windows)==len(identities(windows))
    for w in windows:assert len(w)>=5 and np.all(np.diff(w)==1) and (len(w)-1)&(len(w)-2)==0
c=np.r_[np.arange(19),0];rows=chain_windows(c);rot=np.r_[np.roll(np.arange(19),7),np.roll(np.arange(19),7)[0]];assert identities(rows)==identities(chain_windows(rot))==identities(chain_windows(c[::-1]))
for w in rows:assert np.all((np.diff(w)%19)==1)
out=Path('server_rebuild/expansion_v1/multiscale_chain_window_controls_v1');out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='PASS',module_sha256=digest('weld_pipeline/server_multiscale_chain_windows.py'),controls=['open reversal','closed reversal and cyclic start','existing contiguous edges only','dyadic length and unique windows','short chain empty'],scope='Mechanics only, no reference-based slicing or curve accuracy claim'))
print('PASS')
