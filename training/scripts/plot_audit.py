"""All-workpiece coordinate overlay sheets from audited raw geometry and reference curves."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

base=ROOT/'training/data/pilot_v1'
out=ROOT/'training/reports/pilot_v1'
out.mkdir(exist_ok=True,parents=True)
m=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
for page in range(4):
    fig=plt.figure(figsize=(12,10))
    for j,g in enumerate(m['groups'][page*4:page*4+4]):
        ax=fig.add_subplot(2,2,j+1,projection='3d')
        p=base/g['name']; xyz=np.load(p/'geometry_sample.npz')['xyz']/1000
        center=xyz.mean(axis=0); xyz-=center
        ax.scatter(*xyz.T,s=.4,c='#b0b0b0',alpha=.3,rasterized=True)
        curves=json.loads((p/'curves_world_mm.json').read_text(encoding='utf-8'))
        for w in curves:
            for s in w['segments']:
                q=np.array([s['start_world_mm'],s['end_world_mm']])/1000-center
                ax.plot(*q.T,color='#cf2929',lw=.8)
        ax.set_title(f"{g['name']} | {g['reference_welds']} reference welds",fontsize=10)
        ax.set_box_aspect(np.maximum(np.ptp(xyz,axis=0),.3))
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_zticks([])
        ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=8)
    fig.suptitle('Coordinate audit: grey = input cloud; red = supplied reference curves',fontsize=13)
    fig.tight_layout()
    fig.savefig(out/f'audit_overlay_{page+1}.png',dpi=140)
    plt.close(fig)
print('Saved four coordinate overlay sheets')
