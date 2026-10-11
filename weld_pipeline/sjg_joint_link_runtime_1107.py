"""CPU link features and exact finite-path selection for supplied node states.

The caller owns the registered object selection policy. This module never reads
targets, chooses a presence threshold, applies NMS, or truncates candidate count.
All legal links between supplied states are scored by the caller before MILP.
"""
import time
import numpy as np


def node_geometry(node, graph):
    curve=graph.curve(graph.profile(node['fragment']))
    lo,hi=sorted((float(node['lo']),float(node['hi'])))
    from weld_pipeline.sjg_parent_set_runtime_1081 import child_geometry
    points,arc=child_geometry(curve,lo,hi,node['forward'],node.get('closed',False),node.get('wrap',False))
    if arc is None:
        direction=(points[-1]-points[0])/np.linalg.norm(points[-1]-points[0]);tangent=np.stack((direction,direction))
    else:
        tangent=np.sign(arc.sweep_rad)*np.cross(arc.plane_normal,
                    (points[[0,-1]]-arc.center_mm)/arc.radius_mm)
    return points[[0,-1]],tangent


def pair_features(a,b,graph,geometries=None):
    pa,pb=graph.profile(a['fragment']),graph.profile(b['fragment'])
    ca,cb=graph.curve(pa),graph.curve(pb)
    (ea,ta),(eb,tb)=geometries or (node_geometry(a,graph),node_geometry(b,graph))
    aa,bb=graph.bounds(a['fragment']),graph.bounds(b['fragment'])
    broad=float(np.linalg.norm(np.maximum(0.,np.maximum(aa[0]-bb[1],bb[0]-aa[1]))))
    distance=float(np.linalg.norm(ea[-1]-eb[0]))
    same=(a['side']==b['side'] and a.get('weld_type')==b.get('weld_type') and
          graph.connection(a['fragment'],b['fragment'],'SAME_CHILD',a['forward'],b['forward']))
    shared=distance<=6. and graph.connection(a['fragment'],b['fragment'],'SHARED',a['forward'],b['forward'])
    gap=distance>0. and graph.connection(a['fragment'],b['fragment'],'POSITIVE_GAP',a['forward'],b['forward'])
    legal=np.asarray([same,shared,gap],bool)
    fa=graph.profiles[pa].get('face_indices',[])[:2];fb=graph.profiles[pb].get('face_indices',[])[:2]
    if a['side']:fa=fa[::-1]
    if b['side']:fb=fb[::-1]
    roles=[int(i<len(fa) and j<len(fb) and fa[i]==fb[j]) for i,j in ((0,0),(0,1),(1,0),(1,1))]
    sa,sb=graph.spans[a['fragment']],graph.spans[b['fragment']]
    values=[broad,distance,*ta[-1],*tb[0],float(ta[-1]@tb[0]),
            ca.kind=='line',ca.kind=='arc',cb.kind=='line',cb.kind=='arc',pa==pb,*roles,
            a['side']==0,a['side']==1,b['side']==0,b['side']==1,sa[2],sb[2],sa[3],sb[3],
            0,0,1,*legal]
    if len(values)!=32:raise AssertionError('Frozen 32-channel link schema changed')
    return np.asarray(values,np.float32),legal


def build_links(nodes,graph):
    geometry=[node_geometry(n,graph) for n in nodes]
    links=[];features=[];masks=[]
    for i,a in enumerate(nodes):
        for j,b in enumerate(nodes):
            if i==j:continue
            row,legal=pair_features(a,b,graph,(geometry[i],geometry[j]))
            if legal.any():links.append((i,j));features.append(row);masks.append(legal)
    return (np.asarray(links,np.int32).reshape(-1,2),
            np.asarray(features,np.float32).reshape(-1,32),np.asarray(masks,bool).reshape(-1,3))


def select_paths(nodes,links,link_probabilities,legal,wall_seconds=10.,competition_resources=None,allow_node_selection=False):
    """Pro286: all supplied positive nodes are fixed selected; optimize links.

    Timeout has no certified partial-output fallback. The connector cannot
    remove an isolated or inconvenient positive to improve reported precision.
    """
    begin=time.perf_counter()
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import coo_matrix
    n=len(nodes);links=np.asarray(links,np.int32).reshape(-1,2)
    p=np.asarray(link_probabilities,np.float64);legal=np.asarray(legal,bool)
    if p.shape!=(len(links),4) or legal.shape!=(len(links),3):raise ValueError('Link probability/mask shape')
    if np.any(~np.isfinite(p)) or np.any(p<0) or np.any(p>1):raise ValueError('Invalid link probabilities')
    present=np.asarray([x['probability'] for x in nodes],np.float64)
    if np.any(~np.isfinite(present)) or np.any(present<0) or np.any(present>1):raise ValueError('Invalid object probabilities')
    if np.any(present<.5):raise ValueError('Runtime node set must be the registered fixed 0.5 positives')
    eps=1e-6;present=np.clip(present,eps,1-eps);p=np.clip(p,eps,1-eps)
    a=np.log(present)-np.log1p(-present);b=np.log(p[:,1:])-np.log(p[:,0,None])
    # Components include legal edges and shared input occupancy, never geometric
    # aliases belonging to different unconfirmed profiles or side hypotheses.
    root=list(range(n))
    def find(x):
        while root[x]!=x:root[x]=root[root[x]];x=root[x]
        return x
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y:root[y]=x
    for i,j in links:union(int(i),int(j))
    resources=[] if competition_resources is None else competition_resources
    if resources and not allow_node_selection:
        raise ValueError('Competition requires the explicitly registered joint node selection policy')
    for resource in resources:
        members=resource['nodes']
        for i in members[1:]:union(int(members[0]),int(i))
    occupancy={}
    for i,node in enumerate(nodes):
        key=(int(node['fragment']),int(node['side']),int(node.get('k',1))) if allow_node_selection else (int(node['fragment']),int(node['side']))
        if key in occupancy:raise ValueError('Duplicate fixed input occupancy; must count as invalid, not drop a positive')
        occupancy.setdefault(key,[]).append(i)
    groups={}
    for i in range(n):groups.setdefault(find(i),[]).append(i)
    edge_groups={}
    for e,(i,j) in enumerate(links):edge_groups.setdefault(find(int(i)),[]).append(e)
    selected=[];chosen=[];receipts=[]
    for component,ids in groups.items():
        remaining=wall_seconds-(time.perf_counter()-begin)
        if remaining<=0:raise TimeoutError('Complete graph selection exceeded registered cumulative time')
        edges=edge_groups.get(component,[])
        if not edges and len(ids)==1:
            selected.extend(ids)
            continue
        local={i:k for k,i in enumerate(ids)};variables=[]
        for e in edges:
            for kind in range(3):
                if legal[e,kind]:variables.append((e,kind))
        objective=np.r_[-a[ids],[-b[e,k] for e,k in variables]]
        ri=[];ci=[];data=[];lo=[];hi=[]
        def constraint(values,lower,upper):
            row=len(lo)
            for index,value in values:ri.append(row);ci.append(index);data.append(value)
            lo.append(lower);hi.append(upper)
        incoming={i:[] for i in ids};outgoing={i:[] for i in ids};per_pair={}
        for k,(e,kind) in enumerate(variables,len(ids)):
            i,j=map(int,links[e]);incoming[j].append(k);outgoing[i].append(k);per_pair.setdefault(e,[]).append(k)
            constraint([(k,1),(local[i],-1)],-np.inf,0)
            constraint([(k,1),(local[j],-1)],-np.inf,0)
        for i in ids:
            for edges_at_node in (incoming[i],outgoing[i]):
                if edges_at_node:constraint([(k,1) for k in edges_at_node]+[(local[i],-1)],-np.inf,0)
        for values in per_pair.values():constraint([(v,1) for v in values],-np.inf,1)
        for values in occupancy.values():
            present_ids=[local[i] for i in values if i in local]
            if len(present_ids)>1:constraint([(i,1) for i in present_ids],-np.inf,1)
        seen_resources=set()
        for resource in resources:
            members=tuple(sorted(local[i] for i in resource['nodes'] if i in local))
            if len(members)>1 and members not in seen_resources:
                constraint([(i,1) for i in members],-np.inf,1);seen_resources.add(members)
        matrix=coo_matrix((data,(ri,ci)),shape=(len(lo),len(objective))).tocsc()
        matrix.indices=matrix.indices.astype(np.int32);matrix.indptr=matrix.indptr.astype(np.int32)
        lower_bounds=np.r_[np.zeros(len(ids)) if allow_node_selection else np.ones(len(ids)),np.zeros(len(variables))]
        result=milp(objective,integrality=np.ones(len(objective),np.int32),bounds=Bounds(lower_bounds,1.),
                    constraints=LinearConstraint(matrix,np.asarray(lo),np.asarray(hi)),
                    options={'time_limit':max(.001,wall_seconds-(time.perf_counter()-begin)),'presolve':True})
        receipts.append(dict(nodes=len(ids),edge_states=len(variables),status=int(result.status),message=str(result.message)))
        if result.status!=0 or result.x is None:
            raise RuntimeError('No certified complete MILP solution: '+str(receipts[-1]))
        selected.extend(ids[i] for i in np.flatnonzero(result.x[:len(ids)]>.5))
        chosen.extend((int(links[e,0]),int(links[e,1]),int(kind)+1)
                      for e,kind in [variables[i] for i in np.flatnonzero(result.x[len(ids):]>.5)])
    elapsed=time.perf_counter()-begin
    if elapsed>wall_seconds:raise TimeoutError('Complete graph selection exceeded registered cumulative time')
    if not allow_node_selection:assert set(selected)==set(range(n)), 'Connector removed a fixed positive'
    return dict(selected_nodes=selected,selected_links=chosen,seconds=elapsed,components=receipts,
                complete_optimal=True,all_positive_nodes_fixed_selected=not allow_node_selection,
                admitted_nodes=n,unselected_explanations=n-len(selected),
                input_competition_atoms=len(resources),
                probability_numeric_clip=eps,GT_reads=0)
