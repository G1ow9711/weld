"""Exhaustive small mathematical controls for Pro359's exact simplification."""
import time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import write
from weld_pipeline.sjg_sparse_path_solver_1168 import assemble
from weld_pipeline.sjg_compact_resources_1169 import compact_resources,compare_legacy
from weld_pipeline.sjg_request_competition_1125 import competition_resources
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph


def control(dest):
    begin=time.perf_counter();reports=[]
    cases=[(3,[(0,1),(1,0),(1,2)],r,True) for r in ([],[[0,2]],[[0,1],[1,2]])]
    cases.extend([(3,[(0,1),(1,0)],[],False),(4,[(0,1),(2,3)],[[0,1],[2,3]],True)])
    for n,links,resource,optional in cases:
        links=np.asarray(links,np.int32);edge_from=np.repeat(links[:,0],3);edge_to=np.repeat(links[:,1],3)
        m=len(edge_from);b=np.resize(np.array([-2.,0.,1.]),m);a=np.arange(n,dtype=float)
        resource=[np.array(r,np.int32) for r in resource]
        matrix,c,upper,lower=assemble(n,edge_from,edge_to,b,resource,a,optional)
        numbers=np.arange(1<<(n+m),dtype=np.uint32)
        values=((numbers[:,None]>>np.arange(n+m,dtype=np.uint32))&1).astype(np.float64)
        x,y=values[:,:n],values[:,n:];old=np.ones(len(values),bool)
        if not optional:old&=np.all(x==1,axis=1)
        # Literal old per-state, per-node, directed-per-pair and resource rows.
        for j in range(m):old&=(y[:,j]<=x[:,edge_from[j]])&(y[:,j]<=x[:,edge_to[j]])
        for i in range(n):
            old&=y[:,edge_from==i].sum(1)<=x[:,i];old&=y[:,edge_to==i].sum(1)<=x[:,i]
        for e in range(len(links)):old&=y[:,3*e:3*e+3].sum(1)<=1
        for members in resource:old&=x[:,members].sum(1)<=1
        new=np.all(values@matrix.toarray().T<=upper,axis=1)&np.all(values>=lower,axis=1)
        assert np.array_equal(old,new),'Redundant-row deletion changed feasible set'
        objective=values@(-c);maximum=objective[old].max();optimal=old&(objective==maximum)
        assert np.all(y[optimal][:,b<0]==0),'Negative state in an exact optimum'
        keep=b>=0;columns=np.r_[np.arange(n),n+np.flatnonzero(keep)]
        reduced,rc,ru,rl=assemble(n,edge_from[keep],edge_to[keep],b[keep],resource,a,optional)
        permitted=np.all(y[:,~keep]==0,axis=1)
        rf=np.all(values[:,columns]@reduced.toarray().T<=ru,axis=1)&np.all(values[:,columns]>=rl,axis=1)&permitted
        ro=values[:,columns]@(-rc);rm=ro[rf].max();restored=rf&(ro==rm)
        assert rm==maximum and np.array_equal(restored,optimal)
        reports.append(dict(nodes=n,edge_states=m,assignments=len(values),resources=[v.tolist() for v in resource],
            feasible_sets_exact=True,optimal_sets_exact_after_zero_extension=True,maximum=float(maximum),
            zero_states_retained=int((b==0).sum()),both_directions_present=bool(np.any(edge_from>edge_to))))
    # Certified interval boundaries and independent side resources, no targets.
    profiles=[dict(representation=dict(kind='line',endpoints_mm=[[0.,0.,0.],[20.,0.,0.]]),
        support_intervals_mm=[[0.,20.]],unobserved_gaps_mm=[]) for _ in range(2)]
    graph=JointSegmentGraph.build(profiles)
    frozen=dict(requests=[dict(members=[0,1],origin=[0.,0.,0.],direction=[1.,0.,0.])])
    nodes=[dict(fragment=0,side=0,lo=0.,hi=10.),dict(fragment=1,side=0,lo=5.,hi=15.),
        dict(fragment=0,side=1,lo=0.,hi=10.),dict(fragment=1,side=1,lo=10.,hi=20.)]
    resources=compact_resources(nodes,graph,frozen);compare_legacy(resources,competition_resources(nodes,graph,frozen))
    write(dest/'sparse_mechanism_control.json',dict(status='SPARSE_EXHAUSTIVE_EQUIVALENCE_PASSED',cases=reports,
        resource_sweep_exact=True,seconds=time.perf_counter()-begin,real_fit_calls=0,GT_reads=0,new_extraction_result=False))
    print('SPARSE_EXHAUSTIVE_EQUIVALENCE_PASSED',sum(r['assignments'] for r in reports),'ASSIGNMENTS',flush=True)
