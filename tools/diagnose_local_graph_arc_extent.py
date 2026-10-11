"""TRAIN diagnostic separating circle fit errors from finite angular coverage."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.contract_curve_geometry import CircularArc

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main(key='new_ac2008a03607', graph='mutual'):
    folder=BASE/({'mutual':'local_graph_arc_candidates_v1','union':'union_graph_arc_seeds_v1','direction':'direction_union_arc_seeds_v1','geometric':'geometric_refined_arc_seeds_v1','multiscale':'multiscale_arc_seeds_v1','joint':'joint_fragment_arc_seeds_v1'}[graph])/key
    receipt=read(folder/'receipt.json');assert receipt['status']=='INPUT_ARC_CANDIDATES_FROZEN'
    assert digest(folder/'protocol.json')==receipt['protocol_sha256']
    records=[];curves=[]
    for face in receipt['faces']:
        path=folder/face['file'];assert digest(path)==face['sha256']
        for index,candidate in enumerate(read(path)['candidates']):
            curves.append(CurveCarrier(dict(kind='arc',arc_parameters=candidate['arc_parameters'])))
            records.append(dict(face=face['face'],candidate=index,radial_error_max_mm=candidate['radial_error_max_mm'],support_vertices=candidate['support_vertices']))
    centers=np.asarray([c.arc.center_mm for c in curves]);radii=np.asarray([c.arc.radius_mm for c in curves]);normals=np.asarray([c.arc.plane_normal for c in curves])
    truthpath=BASE/'labels'/key/'targets.json';rows=[]
    for pi,parent in enumerate(read(truthpath)['parents']):
        for ci,child in enumerate(parent['children']):
            if child['representation']['kind']!='arc':continue
            arc=CircularArc.from_parameters(child['representation']['arc_parameters']);points=arc.point_at(np.linspace(0,arc.length_mm,int(np.ceil(arc.length_mm))+1))
            circle_error=np.empty(len(curves))
            for start in range(0,len(curves),32):
                end=min(start+32,len(curves));relative=points[None]-centers[start:end,None];height=np.einsum('npi,ni->np',relative,normals[start:end])
                radial=np.linalg.norm(relative-height[:,:,None]*normals[start:end,None],axis=2)
                circle_error[start:end]=np.hypot(height,radial-radii[start:end,None]).max(1)
            finite_error=np.asarray([c.project(points)[1].max() for c in curves])
            parameter_error=np.linalg.norm(centers-arc.center_mm,axis=1)+abs(radii-arc.radius_mm)
            closest=[]
            for i in np.argsort(parameter_error)[:8]:
                closest.append(dict(records[i],circle_parameter_error_mm=float(parameter_error[i]),infinite_circle_sample_error_mm=float(circle_error[i]),finite_arc_sample_error_mm=float(finite_error[i]),candidate_length_mm=curves[i].length))
            rows.append(dict(parent=pi,child=ci,reference_length_mm=arc.length_mm,infinite_circle_sample_compatible_count=int((circle_error<=3).sum()),finite_arc_sample_compatible_count=int((finite_error<=3).sum()),best_infinite_circle_sample_error_mm=float(circle_error.min()),best_finite_arc_sample_error_mm=float(finite_error.min()),closest_circles=closest))
            print(pi,ci,'circle',int((circle_error<=3).sum()),'finite',int((finite_error<=3).sum()),'best',float(finite_error.min()),flush=True)
    out=BASE/(('local_graph_arc_extent_diagnostic_v1' if key=='new_ac2008a03607' else 'local_graph_arc_extent_fc_diagnostic_v1') if graph=='mutual' else graph+'_graph_arc_extent_'+key+'_v1');out.mkdir(exist_ok=False)
    write(out/'report.json',dict(rows=rows,candidate_receipt_sha256=digest(folder/'receipt.json'),target_sha256=digest(truthpath),scope='TRAIN post-freeze diagnostics only. Infinite circles are not runtime proposals; sampled coverage is not D3 or contact proof.',fit_calls=0,DEV_reads=0,heldout_reads=0))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--key',choices=['new_ac2008a03607','new_fcaf2af1abee'],default='new_ac2008a03607');parser.add_argument('--graph',choices=['mutual','union','direction','geometric','multiscale','joint'],default='mutual');args=parser.parse_args();main(args.key,args.graph)
