import argparse,json,time
from pathlib import Path
import numpy as np
from training.dataio import read_xyz_pcd
from weld_pipeline.truth import load_truth
from weld_pipeline.geometry_extended import contact_candidates
from weld_pipeline.evaluation import score_welds


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',default=['224-GR24A1','527-FR146A'])
    p.add_argument('--min-support',type=int,default=6);p.add_argument('--ridges',action='store_true');a=p.parse_args()
    if a.min_support!=6:
        import weld_pipeline.geometry_candidates as geometry
        original_support=geometry.support_lines
        geometry.support_lines=lambda xy,**kwargs:original_support(xy,**{**kwargs,'min_support':a.min_support})
    suffix='' if a.min_support==6 else f'_min{a.min_support}'
    if a.ridges:suffix+='_ridges'
    out=Path('weld_pipeline/reports/extended_candidates'+suffix);out.mkdir(parents=True,exist_ok=True)
    for name in a.groups:
        folder=Path('DataSet/9月20日')/name;start=time.perf_counter();xyz=read_xyz_pcd(next(folder.glob('*AllFaces.pcd')))*1000
        pred,meta=contact_candidates(xyz,include_ridges=a.ridges);seconds=time.perf_counter()-start
        (out/f'{name}_candidates.json').write_text(json.dumps(pred,indent=2),encoding='utf-8')
        # Separate evaluator; no labels touched until candidate generation ends.
        truth,_=load_truth(folder);counts={}
        for typ in ['horizontal','vertical']:
            gt=[w for w in truth if w['type']==typ];pp=[w for w in pred if w['type']==typ];covered=0
            for w in gt:
                pts=np.array([p for s in w['segments'] for p in [s['points'][0],s['points'][-1]]])
                for c in pp:
                    ab=np.array(c['segments'][0]['points']);u=ab[1]-ab[0];u=u/np.linalg.norm(u)
                    if np.max(np.linalg.norm(np.cross(pts-ab[0],u),axis=1))<=3:
                        covered+=1;break
            counts[typ]={'reference_welds':len(gt),'infinite_line_oracle_covered':covered,'candidates':len(pp)}
        score=score_welds(pred,truth);report={'seconds_raw_pcd_to_candidates':seconds,'counts':counts,
            'raw_candidate_weld_score':score,'geometry':meta,'not_a_model_prediction':True}
        (out/f'{name}_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(name,seconds,counts,'raw_endpoint_matches',score['valid_welds'],flush=True)


if __name__=='__main__':main()
