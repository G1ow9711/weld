"""Training-only targets for the registered supervised structural-score method.

Runtime candidates and pair order stay those of frozen195. GT identities are
used only to construct loss masks/targets; they are never neural features.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

VERSION='COMPLETE_ALIAS_BALANCED_STRUCTURAL_SCORE_TARGETS_232_v1'
LOSS_KIND='DIRECT_BALANCED_STRUCTURAL_SCORE_232'
EXPECTED_195_SHA='9cee6158b0b44852cfe0bb42e8c949ee07f0f2ead9556098e259f05930c24187'
ROLES=('start','end','cut')
CHILD_ROLE_PAIRS=((0,1),(0,2),(2,2),(2,1))


def array_sha(value):
    array=np.ascontiguousarray(value)
    digest=sha256()
    digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def immutable(value):
    array=np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


class TrainingRelationLayout:
    """Exact195 relation enumeration from literal IDs, without copying features."""
    def __init__(self,runtime_ids,ordered_path_ids):
        ids=np.asarray(runtime_ids)
        paths=tuple(ordered_path_ids)
        if (ids.dtype!=np.int64 or ids.ndim!=2 or ids.shape[1]!=3 or np.any(ids<0)
            or any(isinstance(value,bool) or not isinstance(value,(int,np.integer)) for value in paths)
            or tuple(sorted(set(paths)))!=paths):
            raise ValueError('Original int64 IDs and complete ordered path manifest required')
        if len(ids) and (not np.all((ids[:,2]>=0)&(ids[:,2]<3))
                        or any(tuple(a)>=tuple(b) for a,b in zip(ids[:-1],ids[1:]))):
            raise ValueError('Original strict lexicographic runtime IDs and roles required')
        if not set(ids[:,0].tolist())<=set(paths):
            raise ValueError('Node path outside the original complete manifest')
        self.runtime_ids=immutable(ids)
        self.paths={int(path):np.flatnonzero(ids[:,0]==path).astype(np.int64) for path in paths}
        self.roles={path:tuple(indices[ids[indices,2]==role] for role in range(3)) for path,indices in self.paths.items()}

    def pair_count(self,kind):
        if kind=='child':
            return sum(sum(len(roles[first])*len(roles[second])-(len(roles[first]) if first==second else 0)
                           for first,second in CHILD_ROLE_PAIRS) for roles in self.roles.values())
        if kind=='duplicate':return sum(len(ids)*(len(ids)-1)//2 for ids in self.paths.values())
        if kind=='message':return sum(len(ids)*(len(ids)-1) for ids in self.paths.values())
        raise ValueError('Complete195 child/duplicate/message relation kind required')

    def iter_pairs(self,kind,chunk_size=65536):
        # Literal frozen195 enumeration; avoid its transitive Torch import.
        if kind not in ('child','duplicate','message') or isinstance(chunk_size,bool) or not isinstance(chunk_size,int) or chunk_size<1:
            raise ValueError('Original relation kind and positive integer chunk size required')
        for path,ids in self.paths.items():
            groups=[(self.roles[path][first],self.roles[path][second],first==second)
                    for first,second in CHILD_ROLE_PAIRS] if kind=='child' else [(ids,ids,True)]
            for source,target,may_self in groups:
                if not len(source) or not len(target):continue
                for start in range(0,len(source)*len(target),chunk_size):
                    linear=np.arange(start,min(start+chunk_size,len(source)*len(target)),dtype=np.int64)
                    pairs=np.c_[source[linear//len(target)],target[linear%len(target)]]
                    if kind=='duplicate':pairs=pairs[pairs[:,0]<pairs[:,1]]
                    elif may_self:pairs=pairs[pairs[:,0]!=pairs[:,1]]
                    if len(pairs):yield pairs


@dataclass(frozen=True)
class TargetBundle:
    nodes: np.ndarray
    children: np.ndarray
    duplicates: np.ndarray
    event_ids: tuple
    child_ids: tuple
    audit: dict


def compile_targets(layout,node_event_index,events,children,*,runtime_hashes,label_binding):
    if sha256(Path(__file__).with_name('boundary_query_graph_195.py').read_bytes()).hexdigest()!=EXPECTED_195_SHA:
        raise ValueError('Frozen195 pair order changed')
    index=np.asarray(node_event_index)
    ids=layout.runtime_ids
    if index.dtype!=np.int64 or index.shape!=(len(ids),) or np.any(index<-2):
        raise ValueError('Exact aligned int64 known/unknown event identities required')
    table={int(value['id']):value for value in events}
    if len(table)!=len(events) or any(identity<0 for identity in table):
        raise ValueError('Unique original event identities required')
    if set(index[index>=0].tolist())-set(table):
        raise ValueError('Positive candidate event absent from original table')
    event_paths={};alias_counts={}
    incoming={identity:0 for identity in table};outgoing=incoming.copy()
    for identity,event in table.items():
        nodes=np.flatnonzero(index==identity)
        if not len(nodes):raise ValueError('Original true event has no candidate: '+str(identity))
        paths=set(ids[nodes,0].tolist())
        if len(paths)!=1:raise ValueError('Crosspath aliases have no complete competition in this registered scope')
        if np.any(ids[nodes,2]!=ROLES.index(event['role'])):
            raise ValueError('Original event role and candidate role differ')
        declared=event.get('alternative_node_indices')
        if declared is not None and nodes.tolist()!=declared:
            raise ValueError('Original admitted alias manifest changed')
        event_paths[identity]=next(iter(paths));alias_counts[identity]=len(nodes)
    child_table={};child_keys={};expected_child_counts={}
    for child in children:
        identity=int(child['id']);first=int(child['start_event']);second=int(child['end_event'])
        if identity in child_table or (first,second) in child_keys or first==second or first not in table or second not in table:
            raise ValueError('Unique original ordered Hline child relation required')
        start,end=table[first],table[second]
        if (start['parent_id']!=end['parent_id'] or start['boundary_order']>=end['boundary_order']
            or event_paths[first]!=event_paths[second]
            or (ROLES.index(start['role']),ROLES.index(end['role'])) not in CHILD_ROLE_PAIRS):
            raise ValueError('True child is not an ordered same-parent same-path admissible relation')
        expected=alias_counts[first]*alias_counts[second]
        if child.get('alternative_edge_count',expected)!=expected:
            raise ValueError('Original complete child alias count changed')
        incoming[second]+=1;outgoing[first]+=1
        child_table[identity]=child;child_keys[first,second]=identity;expected_child_counts[identity]=expected
    for identity,event in table.items():
        wanted={'start':(0,1),'end':(1,0),'cut':(1,1)}[event['role']]
        if (incoming[identity],outgoing[identity])!=wanted:
            raise ValueError('Registered chain-union/role assumption is not established: '+str(identity))
    a=np.full(layout.pair_count('child'),-2,np.int64);cursor=0
    for pairs in layout.iter_pairs('child'):
        left,right=index[pairs[:,0]],index[pairs[:,1]]
        values=np.full(len(pairs),-2,np.int64)
        false=(left==-1)|(right==-1)
        known=(left>=0)&(right>=0)
        values[false|known]=-1
        for ordinal in np.flatnonzero(known):
            values[ordinal]=child_keys.get((int(left[ordinal]),int(right[ordinal])),-1)
        a[cursor:cursor+len(pairs)]=values;cursor+=len(pairs)
    if cursor!=len(a):raise ValueError('Complete child pair order/count changed')
    positive_ids,positive_counts=np.unique(a[a>=0],return_counts=True)
    observed_child_counts={int(identity):int(count) for identity,count in zip(positive_ids,positive_counts)}
    if observed_child_counts!=expected_child_counts:
        raise ValueError('Loss masks did not cover every true child alternative')
    d=np.full(layout.pair_count('duplicate'),-2,np.int64);cursor=0
    for pairs in layout.iter_pairs('duplicate'):
        left,right=index[pairs[:,0]],index[pairs[:,1]]
        values=np.full(len(pairs),-2,np.int64)
        false=(left==-1)|(right==-1);known=(left>=0)&(right>=0)
        values[false|known]=-1
        same=known&(left==right);values[same]=left[same]
        d[cursor:cursor+len(pairs)]=values;cursor+=len(pairs)
    if cursor!=len(d):raise ValueError('Complete duplicate pair order/count changed')
    duplicate_ids,duplicate_counts=np.unique(d[d>=0],return_counts=True)
    duplicate_table={int(identity):int(count) for identity,count in zip(duplicate_ids,duplicate_counts)}
    observed_dup_counts={identity:duplicate_table.get(identity,0) for identity in table}
    if any(observed_dup_counts[identity]!=count*(count-1)//2 for identity,count in alias_counts.items()):
        raise ValueError('Original event competition pool is incomplete')
    event_count=len(table);child_count=len(child_table)
    per_path=[];child_offset=0;duplicate_offset=0
    for path,ordinals in layout.paths.items():
        role_groups=layout.roles[path]
        child_size=sum(len(role_groups[first])*len(role_groups[second])-(len(role_groups[first]) if first==second else 0)
                       for first,second in CHILD_ROLE_PAIRS)
        duplicate_size=len(ordinals)*(len(ordinals)-1)//2
        local_events=tuple(sorted(identity for identity,value in event_paths.items() if value==path))
        local_children=tuple(sorted(identity for identity,child in child_table.items()
                                    if event_paths[int(child['start_event'])]==path))
        path_a=a[child_offset:child_offset+child_size]
        path_d=d[duplicate_offset:duplicate_offset+duplicate_size]
        per_path.append(dict(path_id=path,node_count=len(ordinals),child_count=child_size,duplicate_count=duplicate_size,
            original_global_node_ordinals_sha256=array_sha(ordinals),literal_runtime_ids_sha256=array_sha(ids[ordinals]),
            child_offset=child_offset,duplicate_offset=duplicate_offset,event_ids=list(local_events),child_ids=list(local_children),
            path_units_denominator=max(1,len(local_events)+len(local_children)),
            masks_sha256=dict(nodes=array_sha(index[ordinals]),children=array_sha(path_a),duplicates=array_sha(path_d)),
            empty_GT=not local_events and not local_children,trivial_no_candidates_and_no_GT=not len(ordinals) and not local_events and not local_children,
            events_without_duplicate_pairs=[identity for identity in local_events if observed_dup_counts[identity]==0]))
        child_offset+=child_size;duplicate_offset+=duplicate_size
    if child_offset!=len(a) or duplicate_offset!=len(d):
        raise ValueError('Per-path masks do not partition the complete original pair space')
    audit=dict(version=VERSION,loss_kind=LOSS_KIND,node_count=len(index),child_count=len(a),duplicate_count=len(d),
        truth_events=event_count,truth_children=child_count,path_units_denominator=max(1,event_count+child_count),
        target_units=dict(u_positive=1.,u_negative=-1.,a_positive=1.,a_negative=-1.,d_same_event=3.,d_known_non_same_event=0.),
        full_ordered_paths=list(layout.paths),per_path=per_path,
        empty_GT_paths=[path for path in layout.paths if path not in set(event_paths.values())],
        zero_candidate_paths=[path for path,values in layout.paths.items() if not len(values)],
        runtime_hashes=runtime_hashes,label_binding=label_binding,
        masks_sha256=dict(nodes=array_sha(index),children=array_sha(a),duplicates=array_sha(d)),
        counts={name:dict(positive=int(np.sum(values>=0)),known_negative=int(np.sum(values==-1)),unknown=int(np.sum(values==-2)))
                for name,values in (('nodes',index),('children',a),('duplicates',d))},
        all_true_events_single_path=True,all_true_child_alternatives_covered=True,
        all_event_competition_pairs_present=True,registered_chain_union_roles_verified=True,
        aliases_never_permanently_selected_or_deleted=True,negative_endpoint_makes_child_known_false=True,
        unknown_is_never_automatically_negative=True,training_labels_not_runtime_features=True,
        real_neural_forward_backward='NOT_RUN',training_release=False,full_contract_pass=False)
    return TargetBundle(immutable(index),immutable(a),immutable(d),tuple(sorted(table)),tuple(sorted(child_table)),audit)
