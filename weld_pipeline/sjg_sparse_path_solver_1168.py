"""Same path objective, exact negative-state elimination, directly filled CSC."""
import time
from fractions import Fraction
import numpy as np
from weld_pipeline.sjg_compact_resources_1169 import resource_rows
from weld_pipeline.sjg_solver_certificate_1176 import numerical_certificate,validate_integer_rows


def coefficients(probabilities):
    p=np.clip(np.asarray(probabilities,np.float64),1e-6,1-1e-6)
    return np.log(p[:,1:])-np.log(p[:,0,None])


def objective_exact(nodes,links,p,selection):
    present=np.clip(np.asarray([v['probability'] for v in nodes],np.float64),1e-6,1-1e-6)
    a=np.log(present)-np.log1p(-present);value=sum((Fraction.from_float(float(a[i])) for i in selection['selected_nodes']),Fraction())
    wanted={(int(i),int(j)):int(k)-1 for i,j,k in selection['selected_links']}
    found=0
    for start in range(0,len(links),65536):
        block=links[start:start+65536];b=coefficients(p[start:start+65536])
        # Selection is sparse, so avoid constructing a dictionary for all edges.
        for (i,j),kind in wanted.items():
            take=np.flatnonzero((block[:,0]==i)&(block[:,1]==j))
            assert len(take)<=1
            if len(take):value+=Fraction.from_float(float(b[int(take[0]),kind]));found+=1
    assert found==len(wanted)
    return value


def validate_original(nodes,links,legal,resources,selection,allow_node_selection=True):
    n=len(nodes);chosen=np.zeros(n,bool);ids=selection['selected_nodes']
    assert len(set(ids))==len(ids);chosen[ids]=True
    if not allow_node_selection:assert chosen.all()
    incoming=np.zeros(n,np.int32);outgoing=np.zeros(n,np.int32);pairs=set()
    selected={(int(i),int(j)):int(k)-1 for i,j,k in selection['selected_links']}
    assert len(selected)==len(selection['selected_links'])
    for i,j,k in selection['selected_links']:
        assert i!=j and chosen[i] and chosen[j] and 1<=k<=3
        assert (i,j) not in pairs;pairs.add((i,j));outgoing[i]+=1;incoming[j]+=1
    assert np.all(incoming<=chosen) and np.all(outgoing<=chosen)
    found=0
    for start in range(0,len(links),65536):
        block=links[start:start+65536]
        for (i,j),kind in selected.items():
            take=np.flatnonzero((block[:,0]==i)&(block[:,1]==j))
            assert len(take)<=1
            if len(take):assert legal[start+int(take[0]),kind];found+=1
    assert found==len(selected)
    for members in resource_rows(resources):assert int(chosen[members].sum())<=1


def assemble(n,edge_from,edge_to,scores,resource_members,a,allow_node_selection):
    """Two passes over membership. No edge tuples or triplet lists."""
    from scipy.sparse import csc_matrix
    m=len(scores);member_count=sum(len(v) for v in resource_members)
    counts=np.full(n,2,np.int64)
    for members in resource_members:counts[members]+=1
    indptr=np.empty(n+m+1,np.int64);indptr[0]=0
    np.cumsum(counts,out=indptr[1:n+1]);indptr[n+1:]=indptr[n]+2*np.arange(1,m+1,dtype=np.int64)
    assert indptr[-1]==2*n+member_count+2*m
    if indptr[-1]>np.iinfo(np.int32).max or n+m>np.iinfo(np.int32).max:raise MemoryError('CSC int32 capacity exceeded')
    indptr=indptr.astype(np.int32);indices=np.empty(int(indptr[-1]),np.int32);data=np.ones(len(indices),np.float64)
    cursor=indptr[:n].astype(np.int64)
    for i in range(n):
        indices[cursor[i]:cursor[i]+2]=2*i,2*i+1;data[cursor[i]:cursor[i]+2]=-1.;cursor[i]+=2
    for row,members in enumerate(resource_members,2*n):
        positions=cursor[members];indices[positions]=row;cursor[members]+=1
    edge_positions=indptr[n:-1]
    left=2*np.asarray(edge_to,np.int32);right=2*np.asarray(edge_from,np.int32)+1
    indices[edge_positions]=np.minimum(left,right);indices[edge_positions+1]=np.maximum(left,right)
    matrix=csc_matrix((data,indices,indptr),shape=(2*n+len(resource_members),n+m),copy=False)
    assert matrix.has_sorted_indices
    upper=np.r_[np.zeros(2*n),np.ones(len(resource_members))]
    lower_variables=np.r_[np.zeros(n) if allow_node_selection else np.ones(n),np.zeros(m)]
    return matrix,np.r_[-a,-scores],upper,lower_variables


def select_paths(nodes,links,p,legal,resources,wall_seconds=60.,allow_node_selection=True,phase=None):
    begin=time.perf_counter();deadline=begin+wall_seconds
    def mark(name,**extra):
        if phase:phase(name,seconds=time.perf_counter()-begin,**extra)
        if time.perf_counter()>=deadline:raise TimeoutError('Shared path assembly/solve 60-second sublimit')
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.optimize import milp,Bounds,LinearConstraint
    n=len(nodes);e=len(links)
    assert p.shape==(e,4) and legal.shape==(e,3)
    occupancy=set()
    for node in nodes:
        key=(int(node['fragment']),int(node['side']),int(node.get('k',1))) if allow_node_selection else (int(node['fragment']),int(node['side']))
        assert key not in occupancy,'Duplicate registered input occupancy';occupancy.add(key)
    present=np.asarray([v['probability'] for v in nodes],np.float64)
    assert np.isfinite(present).all() and np.all((present>=.5)&(present<=1.))
    present=np.clip(present,1e-6,1-1e-6);a=np.log(present)-np.log1p(-present)
    # All legal edges determine original components, including eliminated states.
    graph=coo_matrix((np.ones(e,np.uint8),(links[:,0],links[:,1])),shape=(n,n)).tocsr()
    _,labels=connected_components(graph,directed=False);del graph
    parent=list(range(n))
    def find(x):
        x=int(x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for members in resource_rows(resources):
        if len(members)>1:
            root=find(labels[members[0]])
            for member in members[1:]:parent[find(labels[member])]=root
    labels=np.array([find(x) for x in labels],np.int32)
    groups={}
    for i,label in enumerate(labels):groups.setdefault(int(label),[]).append(i)
    mark('components_complete',components=len(groups),legal_edges=e)
    total=kept=negative=zero=0
    for start in range(0,e,65536):
        q=p[start:start+65536];assert np.isfinite(q).all() and np.all((q>=0)&(q<=1))
        b=coefficients(q);mask=legal[start:start+65536]
        count=int(np.count_nonzero(mask));total+=count
        negative+=int(np.count_nonzero(mask&(b<0)));zero+=int(np.count_nonzero(mask&(b==0)))
        kept+=int(np.count_nonzero(mask&(b>=0)))
    edge_ids=np.empty(kept,np.int32);kinds=np.empty(kept,np.uint8);scores=np.empty(kept,np.float64)
    original_ids=np.empty(kept,np.int64);original_to_compact=np.full(total,-1,np.int32)
    offset=at=0
    for start in range(0,e,65536):
        b=coefficients(p[start:start+65536]);mask=legal[start:start+65536]
        flat=np.flatnonzero(mask.ravel());keep=b.ravel()[flat]>=0;f=flat[keep];count=len(f)
        edge_ids[at:at+count]=start+f//3;kinds[at:at+count]=f%3;scores[at:at+count]=b.ravel()[f]
        ids=offset+np.flatnonzero(keep);original_ids[at:at+count]=ids
        original_to_compact[ids]=np.arange(at,at+count,dtype=np.int32)
        offset+=len(flat);at+=count
    assert total==kept+negative and at==kept and offset==total
    mark('dominated_variables_removed',original_states=total,retained_states=kept,negative_states=negative,zero_states_retained=zero)
    selected=[];chosen=[];receipts=[];selected_compact=[]
    edge_components=labels[links[edge_ids,0]]
    resource_groups={label:[] for label in groups}
    for members in resource_rows(resources):
        if len(members)>1:
            label=int(labels[members[0]]);assert np.all(labels[members]==label);resource_groups[label].append(members)
    original_degree=np.bincount(links.ravel(),minlength=n) if e else np.zeros(n,np.int64)
    for label,raw_ids in groups.items():
        ids=np.asarray(raw_ids,np.int32);take=np.flatnonzero(edge_components==label)
        if len(ids)==1 and original_degree[ids[0]]==0 and not resource_groups[label]:selected.append(int(ids[0]));continue
        local=np.full(n,-1,np.int32);local[ids]=np.arange(len(ids),dtype=np.int32)
        component_edges=edge_ids[take];members=[local[v] for v in resource_groups[label]]
        matrix,c,upper,lower=assemble(len(ids),local[links[component_edges,0]],local[links[component_edges,1]],scores[take],members,a[ids],allow_node_selection)
        mark('CSC_complete',nodes=len(ids),variables=len(c),nonzeros=matrix.nnz,
            CSC_bytes=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes)
        result=milp(c,integrality=np.ones(len(c),np.int32),bounds=Bounds(lower,1.),
            constraints=LinearConstraint(matrix,np.full(len(upper),-np.inf),upper),
            options={'time_limit':max(.001,deadline-time.perf_counter()),'presolve':True,'mip_rel_gap':0.})
        info=dict(nodes=len(ids),edge_states=len(take),status=int(result.status),message=str(result.message),
            mip_gap=None if getattr(result,'mip_gap',None) is None else float(result.mip_gap),
            mip_dual_bound=None if getattr(result,'mip_dual_bound',None) is None else float(result.mip_dual_bound))
        if result.status!=0 or result.x is None:raise RuntimeError('No certified completed solution: '+str(info))
        rounded=validate_integer_rows(result.x,lower,upper,matrix)
        fun=float(c@rounded);tol=64*np.finfo(float).eps*max(1.,float(np.abs(c[rounded==1]).sum()))
        certificate=numerical_certificate(result.status,result.fun,fun,info['mip_dual_bound'],info['mip_gap'],tol,success=result.success)
        info.update(certificate);receipts.append(info)
        selected.extend(int(i) for i in ids[np.flatnonzero(rounded[:len(ids)]==1)])
        for index in take[np.flatnonzero(rounded[len(ids):]==1)]:
            edge=int(edge_ids[index]);chosen.append((int(links[edge,0]),int(links[edge,1]),int(kinds[index])+1))
            selected_compact.append(int(index))
        del matrix,c,upper,lower,result;mark('component_solved',selected_nodes=len(selected),selected_links=len(chosen))
    result=dict(selected_nodes=selected,selected_links=chosen,seconds=time.perf_counter()-begin,components=receipts,
        complete_optimal=True,all_positive_nodes_fixed_selected=not allow_node_selection,admitted_nodes=n,
        unselected_explanations=n-len(selected),input_competition_atoms=len(resources['offsets'])-1,
        probability_numeric_clip=1e-6,GT_reads=0,original_edge_states=total,negative_states_eliminated=negative,
        zero_states_retained=zero,retained_edge_states=kept,development_solve_limit_seconds=wall_seconds)
    # Original rows are verified directly using chosen variables, with eliminated states zero.
    chosen_mask=np.zeros(n,bool);chosen_mask[selected]=True
    incoming=np.zeros(n,np.int32);outgoing=np.zeros(n,np.int32);seen=set()
    for i,j,k in chosen:
        assert chosen_mask[i] and chosen_mask[j] and (i,j) not in seen
        seen.add((i,j));incoming[j]+=1;outgoing[i]+=1
    assert np.all(incoming<=chosen_mask) and np.all(outgoing<=chosen_mask)
    for members in resource_rows(resources):assert chosen_mask[members].sum()<=1
    value=sum((Fraction.from_float(float(a[i])) for i in selected),Fraction())
    for index in selected_compact:value+=Fraction.from_float(float(scores[index]))
    result['objective_exact_numerator']=str(value.numerator);result['objective_exact_denominator']=str(value.denominator)
    mark('original_constraints_verified',objective_float=float(value))
    result['seconds']=time.perf_counter()-begin
    return result,dict(original_to_compact=original_to_compact,compact_to_original=original_ids,edge_ids=edge_ids,kinds=kinds)
