"""Full frozen-query candidate graph; no label, source-name or model I/O.

One admitted canonical query times role supplies one proposal. Role-legal
same-path child edges are enumerated completely, independent of coordinates.
Candidate IDs only index proposals; neural features never include IDs.
"""
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
import math

import numpy as np

from weld_pipeline.continuous_center_modes_v2 import RESOLUTION_MM
from weld_pipeline.boundary_graph_representation_193 import (
    BoundaryNode,ParentHypothesis,OrderedChildEdge,BoundaryGraph,export_welds)

VERSION='full_frozen_query_boundary_graph_195_v1'
ROLES=('start','end','cut')
CHILD_ROLE_PAIRS=((0,1),(0,2),(2,2),(2,1))
PAIR_FEATURE_NAMES=('query_delta_mm','centre_delta_mm','xyz_delta_x_mm','xyz_delta_y_mm','xyz_delta_z_mm',
    'xyz_distance_mm','abs_query_delta_mm','interval_FG_mean','interval_FG_std','interval_FG_trapezoid_integral_mm',
    'source_start','source_end','source_cut','target_start','target_end','target_cut')
NODE_GEOMETRY_NAMES=('query_position_mm','predicted_position_mm','centre_minus_query_mm',
    'predicted_x_relative_to_path_origin_mm','predicted_y_relative_to_path_origin_mm','predicted_z_relative_to_path_origin_mm','support')


class QueryGraphError(ValueError):pass


def ordinal(value,name='ordinal'):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<0:
        raise QueryGraphError(name+': nonnegative integer required')
    return int(value)


def array_sha(value):
    array=np.ascontiguousarray(value)
    h=sha256();h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class CanonicalPathInput:
    path_id: int
    origin_mm: object
    direction: object
    step_mm: float
    raw: object
    feature_provider: object

    def __post_init__(self):
        object.__setattr__(self,'path_id',ordinal(self.path_id,'path_id'))
        origin=np.asarray(self.origin_mm,float);direction=np.asarray(self.direction,float);raw=np.asarray(self.raw)
        if origin.shape!=(3,) or direction.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(direction).all():
            raise QueryGraphError('finite path origin/direction3 required')
        if not math.isclose(float(np.linalg.norm(direction)),1.,rel_tol=1e-8,abs_tol=1e-8):raise QueryGraphError('unit path direction required')
        if raw.dtype!=np.float32 or raw.ndim!=2 or raw.shape[1]!=7 or not np.isfinite(raw).all():raise QueryGraphError('frozen finite float32 raw[N,7] required')
        if not math.isfinite(float(self.step_mm)) or self.step_mm<=0.:raise QueryGraphError('positive finite step_mm required')
        if not callable(self.feature_provider):raise QueryGraphError('PCD feature_provider(indices) required')
        object.__setattr__(self,'origin_mm',tuple(origin));object.__setattr__(self,'direction',tuple(direction))
        object.__setattr__(self,'raw',raw);object.__setattr__(self,'step_mm',float(self.step_mm))


@dataclass(frozen=True)
class BoundaryProposal:
    runtime_id: tuple
    path_id: int
    role: str
    query_position_mm: float
    predicted_position_mm: float
    predicted_xyz_mm: object
    support: float
    raw7: object
    pcd_features425: object

    def __post_init__(self):
        if not isinstance(self.runtime_id,(tuple,list)) or len(self.runtime_id)!=3:raise QueryGraphError('runtime_id=(path,query,role) required')
        identity=tuple(ordinal(v,'runtime_id') for v in self.runtime_id);path=ordinal(self.path_id,'path_id')
        if self.role not in ROLES or identity[0]!=path or identity[2]!=ROLES.index(self.role):raise QueryGraphError('runtime ID path/role differs from proposal')
        xyz=np.asarray(self.predicted_xyz_mm,float);raw=np.asarray(self.raw7);features=np.asarray(self.pcd_features425)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise QueryGraphError('finite predictedXYZ3 required')
        if raw.shape!=(7,) or raw.dtype!=np.float32 or not np.isfinite(raw).all():raise QueryGraphError('frozen float32 raw7 required')
        if features.shape!=(425,) or features.dtype!=np.float32 or not np.isfinite(features).all():raise QueryGraphError('finite PCD float32 features425 required')
        if not all(math.isfinite(float(v)) for v in (self.query_position_mm,self.predicted_position_mm,self.support)) or not 0.<=self.support<=1.:
            raise QueryGraphError('finite proposal query/centre/support required')
        raw=raw.copy();features=features.copy();raw.setflags(write=False);features.setflags(write=False)
        object.__setattr__(self,'runtime_id',identity);object.__setattr__(self,'path_id',path)
        object.__setattr__(self,'predicted_xyz_mm',tuple(float(v) for v in xyz))
        object.__setattr__(self,'raw7',raw);object.__setattr__(self,'pcd_features425',features)


@dataclass(frozen=True)
class PathContext:
    path_id: int
    query_count: int
    step_mm: float
    origin_mm: tuple
    direction: tuple
    raw_sha256: str
    fg_prefix: object
    fg_square_prefix: object

    def __post_init__(self):
        object.__setattr__(self,'path_id',ordinal(self.path_id,'context path_id'))
        count=ordinal(self.query_count,'query_count');object.__setattr__(self,'query_count',count)
        if not math.isfinite(float(self.step_mm)) or self.step_mm<=0.:raise QueryGraphError('finite positive context step required')
        for name in ('fg_prefix','fg_square_prefix'):
            value=np.asarray(getattr(self,name),float)
            if value.shape!=(count+1,) or not np.isfinite(value).all() or value[0]!=0. or np.any(np.diff(value)<-1e-10):raise QueryGraphError('finite canonical FG prefix required')
            value.setflags(write=False);object.__setattr__(self,name,value)


def frozen_query_fields(raw,step_mm):
    raw=np.asarray(raw)
    if raw.dtype!=np.float32 or raw.ndim!=2 or raw.shape[1]!=7 or not np.isfinite(raw).all():raise QueryGraphError('frozen raw[N,7] required')
    if not math.isfinite(float(step_mm)) or step_mm<=0.:raise QueryGraphError('positive finite step required')
    q=np.arange(len(raw),dtype=float)*float(step_mm)
    centres=np.floor(q[:,None]/64.)*64.+raw[:,4:7].astype(float)*16.
    support=1./(1.+np.exp(-np.clip(raw[:,1:4].astype(float),-60.,60.)))
    offsets=(centres-q[:,None])/16.
    support_ok=support>=.5;radius_ok=np.abs(offsets)<=1.+1e-6
    span=q[-1] if len(q) else 0.
    span_ok=(centres>=-RESOLUTION_MM)&(centres<=span+RESOLUTION_MM)
    return q,centres,support,support_ok&radius_ok&span_ok,dict(support=support_ok,radius=radius_ok,span=span_ok)


def same_proposal(a,b):
    return (a.runtime_id==b.runtime_id and a.path_id==b.path_id and a.role==b.role and
        a.query_position_mm==b.query_position_mm and a.predicted_position_mm==b.predicted_position_mm and
        a.predicted_xyz_mm==b.predicted_xyz_mm and a.support==b.support and
        np.array_equal(a.raw7,b.raw7) and np.array_equal(a.pcd_features425,b.pcd_features425))


class QueryGraph:
    def __init__(self,proposals=(),path_contexts=(),admission_receipts=None):
        contexts={}
        for context in path_contexts:
            if context.path_id in contexts:
                old=contexts[context.path_id]
                if (old.query_count,old.step_mm,old.origin_mm,old.direction,old.raw_sha256)!=(context.query_count,context.step_mm,context.origin_mm,context.direction,context.raw_sha256) or not np.array_equal(old.fg_prefix,context.fg_prefix) or not np.array_equal(old.fg_square_prefix,context.fg_square_prefix):
                    raise QueryGraphError('conflicting repeated canonical path ID')
            else:contexts[context.path_id]=context
        unique={};duplicates=0
        for node in proposals:
            if not isinstance(node,BoundaryProposal):raise QueryGraphError('BoundaryProposal required')
            if node.path_id not in contexts:raise QueryGraphError('proposal path context missing')
            if node.runtime_id in unique:
                if not same_proposal(unique[node.runtime_id],node):raise QueryGraphError('conflicting repeated explicit query-role record')
                duplicates+=1
            else:unique[node.runtime_id]=node
        self.nodes=tuple(unique[key] for key in sorted(unique));self.path_contexts=contexts
        self.index={node.runtime_id:i for i,node in enumerate(self.nodes)}
        self.paths={path:np.asarray([i for i,n in enumerate(self.nodes) if n.path_id==path],np.int64) for path in sorted(contexts)}
        self.roles={path:tuple(np.asarray([i for i in ids if self.nodes[i].role==role],np.int64) for role in ROLES) for path,ids in self.paths.items()}
        self.explicit_duplicate_records_removed=duplicates;self.admission_receipts=admission_receipts or {}
        self._child_buckets=[];offset=0
        for path,roles in self.roles.items():
            for a,b in CHILD_ROLE_PAIRS:
                source,target=roles[a],roles[b];count=len(source)*len(target)-(len(source) if a==b else 0)
                if count:self._child_buckets.append((offset,offset+count,path,a,b,source,target))
                offset+=count

    def role_counts(self):return {path:{role:len(self.roles[path][r]) for r,role in enumerate(ROLES)} for path in self.roles}

    def pair_count(self,kind='child'):
        if kind=='child':
            return sum(sum(len(roles[a])*len(roles[b])-(len(roles[a]) if a==b else 0) for a,b in CHILD_ROLE_PAIRS) for roles in self.roles.values())
        if kind=='message':return sum(len(ids)*(len(ids)-1) for ids in self.paths.values())
        if kind=='duplicate':return sum(len(ids)*(len(ids)-1)//2 for ids in self.paths.values())
        raise QueryGraphError('pair kind child/message/duplicate required')

    def iter_pairs(self,kind='child',chunk_size=65536):
        if kind not in ('child','message','duplicate'):raise QueryGraphError('pair kind child/message/duplicate required')
        chunk_size=ordinal(chunk_size,'chunk_size')
        if not chunk_size:raise QueryGraphError('nonzero chunk_size required')
        for path,ids in self.paths.items():
            groups=[(self.roles[path][a],self.roles[path][b],a==b) for a,b in CHILD_ROLE_PAIRS] if kind=='child' else [(ids,ids,True)]
            for source,target,may_self in groups:
                if not len(source) or not len(target):continue
                for start in range(0,len(source)*len(target),chunk_size):
                    linear=np.arange(start,min(start+chunk_size,len(source)*len(target)),dtype=np.int64)
                    pairs=np.c_[source[linear//len(target)],target[linear%len(target)]]
                    if kind=='duplicate':pairs=pairs[pairs[:,0]<pairs[:,1]]
                    elif may_self:pairs=pairs[pairs[:,0]!=pairs[:,1]]
                    if len(pairs):yield pairs

    def iter_child_pairs(self,chunk_size=65536):return self.iter_pairs('child',chunk_size)
    def iter_message_pairs(self,chunk_size=65536):return self.iter_pairs('message',chunk_size)
    def iter_duplicate_pairs(self,chunk_size=65536):return self.iter_pairs('duplicate',chunk_size)

    def has_child_pair(self,source,target):
        try:source,target=ordinal(source,'source node'),ordinal(target,'target node')
        except QueryGraphError:return False
        if source==target or not 0<=source<len(self.nodes) or not 0<=target<len(self.nodes):return False
        a,b=self.nodes[source],self.nodes[target]
        return a.path_id==b.path_id and (ROLES.index(a.role),ROLES.index(b.role)) in CHILD_ROLE_PAIRS

    def child_edge_id(self,source,target):
        if not self.has_child_pair(source,target):raise QueryGraphError('pair absent from complete child graph')
        node=self.nodes[source];a=ROLES.index(node.role);b=ROLES.index(self.nodes[target].role)
        for begin,end,path,ra,rb,sources,targets in self._child_buckets:
            if (path,ra,rb)!=(node.path_id,a,b):continue
            i=int(np.searchsorted(sources,source));j=int(np.searchsorted(targets,target))
            local=i*(len(targets)-1)+(j if j<i else j-1) if a==b else i*len(targets)+j
            return begin+local
        raise QueryGraphError('child edge bucket missing')

    def child_edge_pair(self,identity):
        identity=ordinal(identity,'edge_id')
        for begin,end,path,a,b,source,target in self._child_buckets:
            if begin<=identity<end:
                local=identity-begin
                if a==b:
                    i,j=divmod(local,len(target)-1);j=j if j<i else j+1
                else:i,j=divmod(local,len(target))
                return int(source[i]),int(target[j])
        raise QueryGraphError('child edge ID outside complete graph')

    def duplicate_pair_id(self,left,right):
        left,right=sorted((ordinal(left),ordinal(right)))
        if left==right or right>=len(self.nodes) or self.nodes[left].path_id!=self.nodes[right].path_id:raise QueryGraphError('unordered same-path distinct pair required')
        offset=0
        for path,ids in self.paths.items():
            if path==self.nodes[left].path_id:
                i=int(np.searchsorted(ids,left));j=int(np.searchsorted(ids,right));n=len(ids)
                return offset+i*(2*n-i-1)//2+j-i-1
            offset+=len(ids)*(len(ids)-1)//2
        raise QueryGraphError('duplicate pair path missing')

    def node_arrays(self):
        if hasattr(self,'_node_arrays_cache'):return dict(self._node_arrays_cache)
        n=len(self.nodes)
        result=dict(runtime_ids=np.asarray([v.runtime_id for v in self.nodes],np.int64).reshape(n,3),
            features425=np.asarray([v.pcd_features425 for v in self.nodes],np.float32).reshape(n,425),
            raw7=np.asarray([v.raw7 for v in self.nodes],np.float32).reshape(n,7),
            roles=np.asarray([ROLES.index(v.role) for v in self.nodes],np.int64),
            query_position_mm=np.asarray([v.query_position_mm for v in self.nodes],float),
            predicted_position_mm=np.asarray([v.predicted_position_mm for v in self.nodes],float),
            predicted_xyz_mm=np.asarray([v.predicted_xyz_mm for v in self.nodes],float).reshape(n,3),
            support=np.asarray([v.support for v in self.nodes],float))
        for array in result.values():array.setflags(write=False)
        self._node_arrays_cache=result
        return dict(result)

    def node_tensors(self):
        a=self.node_arrays()
        relative=a['predicted_xyz_mm'].copy()
        for path,ids in self.paths.items():relative[ids]-=np.asarray(self.path_contexts[path].origin_mm)
        geometry=np.c_[a['query_position_mm'],a['predicted_position_mm'],a['predicted_position_mm']-a['query_position_mm'],relative,a['support']]
        return dict(node_features=a['features425'],raw7=a['raw7'],roles=a['roles'],node_geometry=geometry)

    def pair_features(self,pairs):
        pairs=np.asarray(pairs)
        if pairs.size and pairs.dtype.kind not in 'iu':raise QueryGraphError('integer pair indexes required')
        pairs=pairs.astype(np.int64,copy=False)
        if pairs.size==0:pairs=np.empty((0,2),np.int64)
        if pairs.ndim!=2 or pairs.shape[1]!=2 or (len(pairs) and (pairs.min()<0 or pairs.max()>=len(self.nodes))):raise QueryGraphError('pair node indexes[N,2] required')
        out=np.empty((len(pairs),len(PAIR_FEATURE_NAMES)),np.float32)
        if not len(pairs):return out
        a=self.node_arrays();s,t=pairs.T;dq=a['query_position_mm'][t]-a['query_position_mm'][s];dc=a['predicted_position_mm'][t]-a['predicted_position_mm'][s]
        delta=a['predicted_xyz_mm'][t]-a['predicted_xyz_mm'][s]
        out[:,:7]=np.c_[dq,dc,delta,np.linalg.norm(delta,axis=1),np.abs(dq)]
        source_paths=a['runtime_ids'][s,0];target_paths=a['runtime_ids'][t,0]
        if not np.array_equal(source_paths,target_paths):raise QueryGraphError('pair features require same path')
        for path in np.unique(source_paths):
            context=self.path_contexts[int(path)];mask=source_paths==path
            if not mask.any():continue
            qs=np.rint(a['query_position_mm'][s[mask]]/context.step_mm).astype(np.int64)
            qt=np.rint(a['query_position_mm'][t[mask]]/context.step_mm).astype(np.int64)
            lo=np.minimum(qs,qt);hi=np.maximum(qs,qt);count=hi-lo+1
            if lo.min()<0 or hi.max()>=context.query_count:raise QueryGraphError('query position outside canonical path')
            total=context.fg_prefix[hi+1]-context.fg_prefix[lo];square=context.fg_square_prefix[hi+1]-context.fg_square_prefix[lo]
            mean=total/count;std=np.sqrt(np.maximum(square/count-mean*mean,0.))
            std[count==1]=0.
            first=context.fg_prefix[lo+1]-context.fg_prefix[lo];last=context.fg_prefix[hi+1]-context.fg_prefix[hi]
            integral=(total-.5*(first+last))*context.step_mm
            out[mask,7:10]=np.c_[mean,std,integral]
        out[:,10:13]=np.eye(3,dtype=np.float32)[a['roles'][s]];out[:,13:16]=np.eye(3,dtype=np.float32)[a['roles'][t]]
        if not np.isfinite(out).all():raise QueryGraphError('finite pair features required')
        return out

    def complete_hashes(self,chunk_size=65536):
        arrays=self.node_arrays();identities=array_sha(arrays['runtime_ids']);h=sha256()
        for key in sorted(arrays):h.update(key.encode());h.update(array_sha(arrays[key]).encode())
        result=dict(node_count=len(self.nodes),node_identity_sha256=identities,node_payload_sha256=h.hexdigest(),pairs={})
        for kind in ('child','message','duplicate'):
            digest=sha256();digest.update((VERSION+'|'+kind+'|'+identities).encode());count=0
            for pairs in self.iter_pairs(kind,chunk_size):digest.update(pairs.astype('<i8',copy=False).tobytes());count+=len(pairs)
            if count!=self.pair_count(kind):raise QueryGraphError('complete enumeration count differs from analytic count')
            result['pairs'][kind]=dict(count=count,sha256=digest.hexdigest(),complete_enumeration=True,no_GT_truncation=True)
        return result


def build_runtime_graph(inputs):
    nodes=[];contexts=[];receipts={}
    for path in inputs:
        if not isinstance(path,CanonicalPathInput):raise QueryGraphError('CanonicalPathInput required')
        q,centre,support,admitted,flags=frozen_query_fields(path.raw,path.step_mm)
        query_ids=np.flatnonzero(admitted.any(axis=1));features=np.asarray(path.feature_provider(query_ids))
        if features.dtype!=np.float32 or features.shape!=(len(query_ids),425) or not np.isfinite(features).all():raise QueryGraphError('PCD provider must return float32[len(indices),425]')
        for row,query in enumerate(query_ids):
            for role in np.flatnonzero(admitted[query]):
                c=float(centre[query,role]);xyz=np.asarray(path.origin_mm)+np.asarray(path.direction)*c
                nodes.append(BoundaryProposal((path.path_id,int(query),int(role)),path.path_id,ROLES[role],float(q[query]),c,xyz,
                    float(support[query,role]),path.raw[query],features[row]))
        fg=1./(1.+np.exp(-np.clip(path.raw[:,0].astype(float),-60.,60.)))
        contexts.append(PathContext(path.path_id,len(q),path.step_mm,path.origin_mm,path.direction,array_sha(path.raw),np.r_[0.,np.cumsum(fg)],np.r_[0.,np.cumsum(fg*fg)]))
        receipts[path.path_id]=dict(queries=len(q),admitted_by_role={role:int(admitted[:,r].sum()) for r,role in enumerate(ROLES)},
            gate_sha256={key:array_sha(value) for key,value in flags.items()},admitted_sha256=array_sha(admitted),
            raw_sha256=array_sha(path.raw),provider_admitted_unique_queries=len(query_ids),provider425_sha256=array_sha(features),
            support_threshold=.5,radius_mm=16.,radius_offset_tolerance=1e-6,span_tolerance_mm=RESOLUTION_MM,
            no_clip_or_rint_of_predicted_centres=True,all_admitted_queries_retained=True)
    return QueryGraph(nodes,contexts,receipts)


def validate_selected(graph,pairs):
    pairs=np.asarray(pairs)
    if pairs.size and pairs.dtype.kind not in 'iu':raise QueryGraphError('integer selected edge indexes required')
    pairs=pairs.astype(np.int64,copy=False)
    if pairs.size==0:pairs=np.empty((0,2),np.int64)
    if pairs.ndim!=2 or pairs.shape[1]!=2:raise QueryGraphError('selected edges[index,index] required')
    selected=set();incoming={};outgoing={}
    for source,target in pairs.tolist():
        if not graph.has_child_pair(source,target):raise QueryGraphError('selected edge absent from full role-legal runtime graph')
        if (source,target) in selected:raise QueryGraphError('duplicate selected child edge')
        selected.add((source,target));incoming.setdefault(target,[]).append(source);outgoing.setdefault(source,[]).append(target)
    used=set(incoming)|set(outgoing)
    for index in used:
        ni=len(incoming.get(index,()));no=len(outgoing.get(index,()));role=graph.nodes[index].role
        legal=(ni==0 and no==1) if role=='start' else ((ni==1 and no==0) if role=='end' else (ni==no==1))
        if not legal:raise QueryGraphError('selected boundary degree violates S(0,1)/E(1,0)/C(1,1)')
    chains=[];visited=set();edge_visited=set()
    for start in sorted(index for index in used if graph.nodes[index].role=='start'):
        chain=[start];current=start
        while graph.nodes[current].role!='end':
            target=outgoing[current][0]
            if target in chain or target in visited:raise QueryGraphError('cycle or reused boundary in selected graph')
            edge_visited.add((current,target));chain.append(target);current=target
        visited.update(chain);chains.append(tuple(chain))
    if visited!=used or edge_visited!=selected:raise QueryGraphError('selected disconnected cut cycle violates acyclic S-to-E chains')
    return dict(selected_nodes=len(used),selected_children=len(selected),selected_parents=len(chains),
        chains=chains,all_role_degrees_valid=True,acyclic=True,unselected_nodes=len(graph.nodes)-len(used),
        empty_graph_legal=True,no_interparent_WIS_or_coordinate_merge=True)


def export_selected(graph,pairs):
    validation=validate_selected(graph,pairs);nodes={};parents=[];edges=[]
    for parent_id,chain in enumerate(validation['chains']):
        first=graph.nodes[chain[0]];local=[]
        for index in chain:
            proposal=graph.nodes[index]
            node=BoundaryNode(proposal.path_id,proposal.path_id,index,proposal.predicted_xyz_mm,
                proposal.predicted_position_mm,proposal.support,proposal.role)
            nodes[index]=node;local.append(node)
        edge_ids=[]
        for order,(a,b) in enumerate(zip(local[:-1],local[1:])):
            identity=len(edges);edges.append(OrderedChildEdge(identity,a.id,b.id,parent_id,order));edge_ids.append(identity)
        direction=1 if local[1].physical_q_mm>local[0].physical_q_mm else -1
        parents.append(ParentHypothesis(parent_id,first.path_id,first.path_id,'horizontal',tuple(edge_ids),
            ('shared_cut',)*(len(edge_ids)-1),direction,min(n.confidence for n in local)))
    runtime_graph=BoundaryGraph(tuple(nodes.values()),tuple(parents),tuple(edges))
    return export_welds(runtime_graph),validation,runtime_graph
