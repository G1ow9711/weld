"""Separate oracle diagnostics from raw geometric candidates on one training part."""
import json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from training.dataio import read_xyz_pcd
from weld_pipeline.truth import load_truth
from weld_pipeline.geometry_candidates import horizontal_candidates
from weld_pipeline.evaluation import score_welds


def main():
    folder=Path('DataSet/9月20日/161-LB18A')
    start=time.perf_counter()
    xyz=read_xyz_pcd(next(folder.glob('*AllFaces.pcd')))*1000
    candidates,meta=horizontal_candidates(xyz)
    meta['seconds_pcd_read_and_candidates']=time.perf_counter()-start
    out=Path('weld_pipeline/reports/geometry_v2');out.mkdir(parents=True,exist_ok=True)
    (out/'161-LB18A_candidates.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    # Truth is loaded only after the point-only candidate output has been written.
    truth,_=load_truth(folder)
    all_segments=[np.asarray(s['points'])[[0,-1]] for w in truth for s in w['segments']]
    lines=meta['plane_lines'];oracle=[]
    for ab in all_segments:
        residual=[np.max(np.abs(ab[:,:2]@np.asarray(l['normal_xy'])-l['offset_mm'])) for l in lines]
        best=int(np.argmin(residual)) if residual else -1
        oracle.append({'line_residual_mm':float(min(residual)) if residual else None,'plane_id':best})
    meta['oracle_segment_support_line_recall_3mm']=float(np.mean([r['line_residual_mm']<=3 for r in oracle]))
    meta['oracle_scope']='Unbounded infinite support-line recall using reference for diagnosis only; NOT endpoint/instance accuracy.'
    meta['raw_candidate_weld_score']=score_welds(candidates,truth)
    (out/'161-LB18A_diagnostics.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(14,8),sharex=True,sharey=True)
    for ax in axs:
        ax.scatter(xyz[::15,0],xyz[::15,1],s=.2,c='#bbbbbb');ax.set_aspect('equal');ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)')
    for ab in all_segments:axs[0].plot(ab[:,0],ab[:,1],c='green',lw=.8)
    for w in candidates:
        ab=np.asarray(w['segments'][0]['points']);axs[1].plot(ab[:,0],ab[:,1],c='red',lw=.8)
    axs[0].set_title('Customer reference (evaluation only)');axs[1].set_title('Unselected geometric contact candidates')
    fig.tight_layout();fig.savefig(out/'161-LB18A_candidates.png',dpi=150);plt.close(fig)
    print({k:v for k,v in meta.items() if k not in ['raw_candidate_weld_score','plane_lines']})
    print({k:v for k,v in meta['raw_candidate_weld_score'].items() if k not in ['per_truth','unmatched_prediction_indices']})


if __name__=='__main__':main()
