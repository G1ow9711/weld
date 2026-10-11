"""Point-only surface candidate diagnostics, then independent reference audit."""
import argparse,json,time
from pathlib import Path
import numpy as np
from training.dataio import read_xyz_pcd
from weld_pipeline.truth import load_truth
from weld_pipeline.surface_planes import surface_candidates


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',nargs='+',default=['161-LB18A','224-GR24A1','527-FR146A','521-CM56A','527-LB24A1','GR768A'])
    parser.add_argument('--local',action='store_true');parser.add_argument('--outdir',default='weld_pipeline/reports/surface_candidates_v1');a=parser.parse_args()
    generator=surface_candidates
    if a.local:
        from weld_pipeline.surface_planes_local import surface_candidates as generator
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    for name in a.groups:
        folder=Path('DataSet/9月20日')/name
        start=time.perf_counter();xyz=read_xyz_pcd(next(folder.glob('*AllFaces.pcd')))*1000
        candidates,meta=generator(xyz);seconds=time.perf_counter()-start
        (out/f'{name}_candidates.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
        truth,_=load_truth(folder);counts={};misses=[]
        for typ in ['horizontal','vertical']:
            gt=[w for w in truth if w['type']==typ];pp=[w for w in candidates if w['type']==typ];covered=0
            if pp:
                ab=np.array([c['segments'][0]['points'] for c in pp]);u=ab[:,1]-ab[:,0];u=u/np.linalg.norm(u,axis=1)[:,None]
            for w in gt:
                pts=np.array([p for s in w['segments'] for p in [s['points'][0],s['points'][-1]]]);best=np.inf
                if pp:
                    errors=np.linalg.norm(np.cross(pts[:,None]-ab[None,:,0],u[None]),axis=2).max(0)
                    best=float(errors.min())
                if best<=3:covered+=1
                else:misses.append({'name':w['name'],'type':typ,'support_line_error_mm':float(best),'ends':pts[[0,-1]].tolist()})
            counts[typ]={'reference':len(gt),'infinite_line_oracle_covered':covered,'candidates':len(pp)}
        report={'seconds_raw_to_candidate':seconds,'counts':counts,'missing':misses,'geometry':meta,
            'scope':'Candidate support coverage oracle, not trained selection/endpoint acceptance.'}
        (out/f'{name}_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(name,round(seconds,3),'planes',len(meta['vertical_planes']),counts,flush=True)


if __name__=='__main__':main()
