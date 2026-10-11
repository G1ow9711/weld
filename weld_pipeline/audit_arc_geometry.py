"""Fit reference arcs offline to determine a supportable future curve representation."""
import argparse,json
from pathlib import Path
import numpy as np
from training.labels import sha256
from weld_pipeline.truth import load_truth


def fit_circle(points):
    p=np.asarray(points,float)
    center=p.mean(0);_,_,axes=np.linalg.svd(p-center,full_matrices=False)
    q=(p-center)@axes[:2].T
    lhs=np.column_stack([2*q,np.ones(len(q))]);rhs=(q*q).sum(1)
    solution,_,rank,_=np.linalg.lstsq(lhs,rhs,rcond=None)
    if rank<3:raise ValueError('Reference arc is degenerate')
    c=solution[:2];radius=float(np.sqrt(solution[2]+c@c))
    theta=np.unwrap(np.arctan2(q[:,1]-c[1],q[:,0]-c[0]))
    plane=np.abs((p-center)@axes[2]);radial=np.abs(np.linalg.norm(q-c,axis=1)-radius)
    return {'center_world_mm':(center+c@axes[:2]).tolist(),'radius_mm':radius,
            'plane_normal':axes[2].tolist(),'plane_max_error_mm':float(plane.max()),
            'radial_max_error_mm':float(radial.max()),'span_degrees':float(abs(theta[-1]-theta[0])*180/np.pi),
            'sampled_points':len(p),'reference_z_span_mm':float(np.ptp(p[:,2]))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',default='data/supplemental/TD2-LB8A')
    ap.add_argument('--out',default='weld_pipeline/reports/arc_reference_representation.json');args=ap.parse_args()
    folder=Path(args.folder);truth,missing=load_truth(folder);records=[]
    for w in truth:
        for i,s in enumerate(w['segments']):
            if s['kind']=='arc':records.append({'weld_id':w['id'],'name':w['name'],'type':w['type'],'segment_index':i,**fit_circle(s['points'])})
    result={'scope':'Customer reference geometry audit only; circle parameters were NOT obtained from the point cloud and are forbidden inference inputs.',
            'folder':str(folder),'reference_welds':len(truth),'missing_curves':missing,'arc_segments':len(records),
            'reference_hashes':{str(p):sha256(p) for p in list(folder.glob('*.xlsx'))+list(folder.glob('*welds.json'))},
            'arcs':records}
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'welds':len(truth),'arc_segments':len(records),
        'max_plane_error_mm':max((r['plane_max_error_mm'] for r in records),default=None),
        'max_radial_error_mm':max((r['radial_max_error_mm'] for r in records),default=None),
        'radii_mm':sorted(set(round(r['radius_mm'],3) for r in records)),
        'z_spans_mm':sorted(set(round(r['reference_z_span_mm'],3) for r in records))}),flush=True)


if __name__=='__main__':main()
