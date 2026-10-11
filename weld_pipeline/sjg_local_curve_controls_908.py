"""Geometry controls for the actual line/arc query and boundary fitter interfaces."""
import argparse
import traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier,curve_point_queries
from weld_pipeline.sjg_planar_boundary_curves_894 import face_boundary_arcs,fit_chain_arcs
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc


def controls():
    rng=np.random.default_rng(908);rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(rotation)<0:rotation[:,0]*=-1
    center=np.array([120.,-400.,35.]);results=[]
    for sweep,closed in ((.7,False),(-.7,False),(4.9,False),(-4.9,False),(2*np.pi,True),(-2*np.pi,True)):
        arc=CircularArc(center,80.,rotation[:,2],rotation[:,0],sweep,closed)
        curve=CurveCarrier(dict(kind='arc',arc_parameters=arc.to_parameters()))
        q=curve.grid();points=curve.at(q);projected,distance=curve.project(points)
        assert distance.max()<1e-8
        assert np.linalg.norm(curve.at(projected)-points,axis=1).max()<1e-8
        assert np.diff(q).max()<=1.+1e-10
        assert np.allclose(np.linalg.norm(curve.tangent(q),axis=1),1.)
        assert np.abs(np.einsum('ij,ij->i',curve.tangent(q),points-center)).max()<1e-8
        sub=CurveCarrier(curve.segment(curve.length*.2,curve.length*.8))
        assert np.allclose(sub.at([0.,sub.length]),curve.at([curve.length*.2,curve.length*.8]),atol=1e-8)
        raised=points+rotation[:,2]*5
        assert np.allclose(curve.project(raised)[1],5.,atol=1e-8)
        if not closed:
            # A point immediately before the start projects onto the start,
            # including the case where atan2 crosses its branch cut.
            angle=-np.sign(sweep)*.01
            outside=center+80*(np.cos(angle)*rotation[:,0]+np.sin(angle)*rotation[:,1])
            assert curve.project(outside[None])[0][0]<1e-8
        results.append(dict(test='3D_CARRIER',sweep=sweep,closed=closed,max_roundtrip_mm=float(distance.max())))
    line=CurveCarrier(dict(kind='line',endpoints_mm=[center.tolist(),(center+rotation[:,0]*100).tolist()]))
    q=line.grid();assert np.allclose(line.project(line.at(q))[0],q)
    assert np.allclose(line.project(np.stack([center-rotation[:,0]*10,center+rotation[:,0]*120]))[0],[0,100])
    # Actual feature queries must transform with their geometry, without
    # feeding absolute workpiece position into the local geometric channels.
    cloud=rng.normal(size=(200,3))*30+center
    features=dict(embedding=rng.normal(size=(200,16)),normal=np.tile(rotation[:,2],(200,1)),boundary=np.zeros(200))
    rep=dict(kind='arc',arc_parameters=CircularArc(center,20.,rotation[:,2],rotation[:,0],1.2).to_parameters())
    a=curve_point_queries(rep,np.linspace(0,24,20),cloud,features,cKDTree(cloud))
    turn,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(turn)<0:turn[:,0]*=-1
    shift=np.array([2000.,100.,-50.]);arc=CircularArc.from_parameters(rep['arc_parameters'])
    transformed=CircularArc(arc.center_mm@turn+shift,arc.radius_mm,arc.plane_normal@turn,arc.start_direction@turn,arc.sweep_rad)
    cloud2=cloud@turn+shift;f2={**features,'normal':features['normal']@turn}
    b=curve_point_queries(dict(kind='arc',arc_parameters=transformed.to_parameters()),np.linspace(0,24,20),cloud2,f2,cKDTree(cloud2))
    assert np.array_equal(a['original_point_ids'],b['original_point_ids'])
    assert np.allclose(a['patch'],b['patch'],atol=2e-6) and np.allclose(a['query'],b['query'],atol=2e-6)
    results.append(dict(test='LOCAL_QUERY_RIGID_EQUIVARIANCE_FIXED_EMBEDDINGS',passed=True,PTv3_rotation_invariance_not_claimed=True))
    # A fan mesh tests the actual boundary traversal and full-circle closure.
    angle=np.arange(128)*2*np.pi/128;uv=np.c_[np.cos(angle),np.sin(angle)]*100
    xyz=np.vstack([center,center+uv@rotation[:,:2].T]);tri=np.array([[0,i+1,(i+1)%128+1] for i in range(128)])
    face=dict(center=center,basis=rotation[:,:2]);arcs,stats=face_boundary_arcs(xyz,face,np.arange(len(xyz)),tri)
    assert stats['boundary_edges']==128 and len(arcs)==1
    found=CircularArc.from_parameters(arcs[0]['arc_parameters'])
    assert found.is_closed and abs(found.radius_mm-100)<1e-7 and np.linalg.norm(found.center_mm-center)<1e-7
    results.append(dict(test='TILTED_FULL_CIRCLE_BOUNDARY',passed=True,edges=stats['boundary_edges']))
    # Straight support cannot silently become a curved target.
    assert not fit_chain_arcs(np.c_[np.arange(101),np.zeros(101)],.02)
    # Disconnected fan components must not be joined into a full loop.
    tri_gap=tri[np.r_[0:30,64:94]]
    arcs,stats=face_boundary_arcs(xyz,face,np.arange(len(xyz)),tri_gap)
    assert all(not a['arc_parameters']['is_closed'] for a in arcs)
    assert all(len(set(a['support_point_ids'])&set(range(2,31)))==0 or len(set(a['support_point_ids'])&set(range(66,95)))==0 for a in arcs)
    results.append(dict(test='DISCONNECTED_BOUNDARY_COMPONENTS_NOT_JOINED',passed=True,candidates=len(arcs)))
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json',dict(stage='LOCAL_CURVE_INTERFACE_CONTROLS',source_sha256=digest(__file__),
        optimizer_updates=0,no_data_accuracy_claim=True,start_utc=utc()))
    try:
        rows=controls()
        # Real local labels exercise representation; no learned prediction score.
        cohort=read(OUT/'run98/local_cohort.json');counts={'line':0,'arc':0};maximum=0.
        for case in cohort['cases']:
            path=OUT/case['target_file'];assert digest(path)==case['target_sha256'];target=read(path)
            for parent in target['parents']:
                for child in parent['children']:
                    curve=CurveCarrier(child['representation']);q=curve.grid()
                    distance=curve.project(curve.at(q))[1];maximum=max(maximum,float(distance.max()))
                    assert distance.max()<1e-6
                    counts[curve.kind]+=1
        assert sum(counts.values())==500 and counts['arc']==27
        write(dest/'receipt.json',dict(status='LOCAL_LINE_ARC_QUERY_CONTROLS_PASSED',controls=rows,
            native_children=counts,real_label_carrier_roundtrip_mm=maximum,
            original_G=496,new_training_candidate_G=4,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
        print('LOCAL_CURVE_CONTROLS_PASSED',counts,'ROUNDTRIP_MM',maximum,flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
