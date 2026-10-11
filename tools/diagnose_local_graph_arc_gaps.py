"""TRAIN-only angular gaps on a diagnostic face, no runtime gap filling."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.contract_curve_geometry import CircularArc

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main():
    key='new_ac2008a03607';old=BASE/'local_graph_arc_candidates_v1'/key;new=BASE/'local_graph_arc_extended_candidates_v1'/key
    paths=[]
    for folder in (old,new):
        receipt=read(folder/'receipt.json');record=next(r for r in receipt['faces'] if r['face']==1);path=folder/record['file'];assert digest(path)==record['sha256'];paths.append(path)
    originals=read(paths[0])['candidates'];extended=read(paths[1])['candidates'];arcs=[CircularArc.from_parameters(c['arc_parameters']) for c in originals]
    gp=BASE/'local_spacing_face_graphs_v1'/key/'face_1.npz'
    gr=read(gp.parent/'receipt.json');assert digest(gp)==next(r['sha256'] for r in gr['faces'] if r['face']==1)
    with np.load(gp) as g:origin=g['origin'];basis=g['basis']
    target=BASE/'labels'/key/'targets.json';rows=[];tau=2*np.pi
    for pi,parent in enumerate(read(target)['parents']):
        for ci,child in enumerate(parent['children']):
            if child['representation']['kind']!='arc':continue
            ref=CircularArc.from_parameters(child['representation']['arc_parameters'])
            index=int(np.argmin([np.linalg.norm(a.center_mm-ref.center_mm)+abs(a.radius_mm-ref.radius_mm) for a in arcs]));circle=arcs[index]
            points=ref.point_at(np.linspace(0,ref.length_mm,int(np.ceil(ref.length_mm))+1));uv=(points-circle.center_mm)@basis
            angles=np.unwrap(np.arctan2(uv[:,1],uv[:,0]));lo=float(angles.min());hi=float(angles.max())
            intervals=[c['angular_support_interval_rad'] for c in extended if c.get('source_circle_candidate')==index];clipped=[]
            for a,b in intervals:
                for shift in (-2,-1,0,1,2):
                    l=max(lo,a+shift*tau);h=min(hi,b+shift*tau)
                    if h>l:clipped.append((l,h))
            gaps=[];cursor=lo
            for a,b in sorted(clipped):
                if a>cursor:gaps.append((cursor,a))
                cursor=max(cursor,b)
            if cursor<hi:gaps.append((cursor,hi))
            widths=[(b-a)*circle.radius_mm for a,b in gaps]
            row=dict(parent=pi,child=ci,diagnostic_face=1,closest_circle_candidate=index,
                circle_parameter_error_mm=float(np.linalg.norm(circle.center_mm-ref.center_mm)+abs(circle.radius_mm-ref.radius_mm)),
                support_interval_count=len(intervals),gaps_rad=gaps,gap_lengths_mm=widths,
                uncovered_total_mm=sum(widths),maximum_gap_mm=max(widths,default=0.))
            rows.append(row);print(pi,ci,'gaps',len(gaps),'max',row['maximum_gap_mm'],'total',sum(widths),flush=True)
    out=BASE/'local_graph_arc_gap_diagnostic_v1';out.mkdir(exist_ok=False)
    write(out/'report.json',dict(rows=rows,input_face_files_sha256=[digest(p) for p in paths],target_sha256=digest(target),scope='TRAIN diagnostic selects closest circle on face1 using reference geometry. Not runtime selection, not permission to merge gaps.',fit_calls=0,DEV_reads=0))

if __name__=='__main__':main()
