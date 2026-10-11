from collections import Counter
from pathlib import Path
import numpy as np
from weld_pipeline.server_direction_continuous_chains import trace_direction_continuous
from weld_pipeline.sjg_local_runtime_904 import write,digest

def normalized(chains):return sorted(min(tuple(c),tuple(c[::-1])) for c in chains)
def checked(v,e):
 c=trace_direction_continuous(v,e);actual=Counter(tuple(sorted((int(a),int(b)))) for p in c for a,b in zip(p,p[1:]));assert actual==Counter(tuple(sorted(x)) for x in e);return c
v=np.array([[-2,0],[-1,0],[0,0],[1,0],[2,0],[0,-1],[0,1]],float);e=np.array([[0,1],[1,2],[2,3],[3,4],[2,5],[2,6]])
c=checked(v,e);assert normalized(c)==[(0,1,2,3,4),(5,2,6)]
assert normalized(c)==normalized(checked(v,e[::-1,::-1]))
a=.73;r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);assert normalized(c)==normalized(checked(v@r.T+[4,9],e))
# A true missing edge remains a break.
assert len(checked(v,e[[0,2,3]]))==2
# Right-angle corner does not get a fabricated tangent continuation.
assert len(checked(np.array([[0,0],[1,0],[1,1]]),np.array([[0,1],[1,2]])))==2
# Symmetric ambiguity is left unpaired, independent of vertex IDs.
theta=np.arange(3)*2*np.pi/3;star=np.vstack(([0,0],np.column_stack((np.cos(theta),np.sin(theta)))));assert len(checked(star,np.array([[0,1],[0,2],[0,3]])))==3
# A smooth closed loop stays closed and covers each edge once.
theta=np.arange(16)*2*np.pi/16;v=np.column_stack((np.cos(theta),np.sin(theta)));e=np.column_stack((np.arange(16),np.roll(np.arange(16),-1)));c=checked(v,e);assert len(c)==1 and c[0][0]==c[0][-1]
out=Path('server_rebuild/expansion_v1/direction_continuous_chain_controls_v1');out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='PASS',module_sha256=digest('weld_pipeline/server_direction_continuous_chains.py'),controls=['crossing continuation','edge order/reversal','rigid transform','missing edge','right-angle stop','ambiguous star stop','closed loop','all input edges exactly once'],scope='Mechanics only, no geometry quality claim'))
print('PASS')
