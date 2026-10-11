"""Explicit runtime-ordinal boundary topology; no discovery, scoring or label I/O.

Every parent and child relation is supplied by the caller. The module never
collects all cuts inside an interval, merges nearby coordinates, reads source
names or chooses a fixed number of children. Arc export uses the 178 parser.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from numbers import Integral
import math

import numpy as np

from weld_pipeline.contract_curve_geometry import CircularArc, sample_arc, verify_sampled_arc

VERSION = 'runtime_ordinal_boundary_graph_193_v1'


class GraphValidationError(ValueError):
    pass


def _ordinal(value,name):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<0:
        raise GraphValidationError(name+': nonnegative runtime integer ordinal required')
    return int(value)


def _node_key(value):
    if not isinstance(value,(tuple,list)) or len(value)!=3:
        raise GraphValidationError('node_id: (sequence_id,joint_id,instance_id) required')
    return tuple(_ordinal(x,'node_id component') for x in value)


def _probability(value,name):
    number=float(value)
    if not math.isfinite(number) or not 0.<=number<=1.:
        raise GraphValidationError(name+': finite probability in[0,1] required')
    return number


@dataclass(frozen=True)
class BoundaryNode:
    sequence_id: int
    joint_id: int
    instance_id: int
    xyz: object
    physical_q_mm: float
    confidence: float
    role: str

    def __post_init__(self):
        for key in ('sequence_id','joint_id','instance_id'):
            object.__setattr__(self,key,_ordinal(getattr(self,key),key))
        xyz=np.asarray(self.xyz,float)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise GraphValidationError('xyz: finite3D point required')
        q=float(self.physical_q_mm)
        if not math.isfinite(q):raise GraphValidationError('physical_q_mm: finite candidate parameter required')
        if self.role not in ('start','end','cut','boundary'):raise GraphValidationError('role: start/end/cut/boundary required')
        object.__setattr__(self,'xyz',tuple(float(v) for v in xyz))
        object.__setattr__(self,'physical_q_mm',q)
        object.__setattr__(self,'confidence',_probability(self.confidence,'node confidence'))

    @property
    def id(self):
        return self.sequence_id,self.joint_id,self.instance_id


@dataclass(frozen=True)
class ParentHypothesis:
    id: int
    sequence_id: int
    joint_id: int
    weld_type: str
    child_edge_ids: tuple
    connections: tuple = ()
    direction: int = 1
    confidence: float = 1.

    def __post_init__(self):
        for key in ('id','sequence_id','joint_id'):object.__setattr__(self,key,_ordinal(getattr(self,key),'parent '+key))
        if self.weld_type not in ('horizontal','vertical'):raise GraphValidationError('weld_type: horizontal or vertical required')
        if isinstance(self.direction,bool) or self.direction not in (-1,1):raise GraphValidationError('parent direction: +1 or -1 required')
        children=tuple(_ordinal(x,'child edge_id') for x in self.child_edge_ids)
        if not children or len(set(children))!=len(children):raise GraphValidationError('parent requires unique explicitly ordered child IDs')
        connections=tuple(self.connections)
        if len(connections)!=len(children)-1 or any(c not in ('shared_cut','gap') for c in connections):
            raise GraphValidationError('connections: one explicit shared_cut/gap for each child adjacency required')
        object.__setattr__(self,'child_edge_ids',children);object.__setattr__(self,'connections',connections)
        object.__setattr__(self,'confidence',_probability(self.confidence,'parent confidence'))


@dataclass(frozen=True)
class OrderedChildEdge:
    id: int
    start_node_id: tuple
    end_node_id: tuple
    parent_id: int
    order: int
    kind: str = 'line'
    arc_parameters: object = None

    def __post_init__(self):
        for key in ('id','parent_id','order'):object.__setattr__(self,key,_ordinal(getattr(self,key),'edge '+key))
        object.__setattr__(self,'start_node_id',_node_key(self.start_node_id));object.__setattr__(self,'end_node_id',_node_key(self.end_node_id))
        if self.kind not in ('line','arc'):raise GraphValidationError('edge kind: line/arc supported; other curves NOT_IMPLEMENTED')
        if self.kind=='line' and self.arc_parameters is not None:raise GraphValidationError('line cannot carry arc_parameters')
        if self.kind=='arc':
            if not isinstance(self.arc_parameters,Mapping):raise GraphValidationError('arc requires178 analytic parameter mapping')
            object.__setattr__(self,'arc_parameters',CircularArc.from_parameters(self.arc_parameters).to_parameters())


@dataclass(frozen=True)
class BoundaryGraph:
    nodes: tuple = ()
    parents: tuple = ()
    edges: tuple = ()


def validate_graph(graph):
    if not isinstance(graph,BoundaryGraph):raise GraphValidationError('BoundaryGraph required')
    nodes={};parents={};edges={}
    for node in graph.nodes:
        if not isinstance(node,BoundaryNode) or node.id in nodes:raise GraphValidationError('missing/duplicate BoundaryNode ID')
        nodes[node.id]=node
    for parent in graph.parents:
        if not isinstance(parent,ParentHypothesis) or parent.id in parents:raise GraphValidationError('missing/duplicate ParentHypothesis ID')
        parents[parent.id]=parent
    for edge in graph.edges:
        if not isinstance(edge,OrderedChildEdge) or edge.id in edges:raise GraphValidationError('missing/duplicate OrderedChildEdge ID')
        edges[edge.id]=edge
        if edge.parent_id not in parents:raise GraphValidationError('edge references missing parent ID')
        if edge.start_node_id not in nodes or edge.end_node_id not in nodes:raise GraphValidationError('edge references missing boundary ID')
        parent=parents[edge.parent_id];start=nodes[edge.start_node_id];end=nodes[edge.end_node_id]
        if any((n.sequence_id,n.joint_id)!=(parent.sequence_id,parent.joint_id) for n in (start,end)):
            raise GraphValidationError('edge endpoints must use explicitly declared parent runtime path')
        span=parent.direction*(end.physical_q_mm-start.physical_q_mm)
        if not math.isfinite(span) or span<=0.:raise GraphValidationError('edge parameter span must be positive in parent direction; no length threshold')
        if edge.kind=='line':
            length=float(np.linalg.norm(np.asarray(end.xyz)-np.asarray(start.xyz)))
            if not math.isfinite(length) or length<=0.:raise GraphValidationError('degenerate straight edge: coincident endpoints')
            if not math.isclose(length,span,rel_tol=1e-9,abs_tol=1e-9):raise GraphValidationError('straight xyz length differs from physical parameter span')
        else:
            arc=CircularArc.from_parameters(edge.arc_parameters)
            expected=arc.point_at(np.asarray([0.,arc.length_mm]))
            if not math.isclose(arc.length_mm,span,rel_tol=1e-9,abs_tol=1e-9):raise GraphValidationError('arc length differs from physical parameter span')
            if np.linalg.norm(expected-np.asarray([start.xyz,end.xyz]),axis=1).max()>1e-9:
                raise GraphValidationError('analytic arc endpoints disagree with boundary nodes')
    consumed=set();node_parent_refs={}
    for parent in graph.parents:
        ordered=[]
        for order,identity in enumerate(parent.child_edge_ids):
            if identity not in edges:raise GraphValidationError('parent references missing child edge ID')
            edge=edges[identity]
            if edge.parent_id!=parent.id or edge.order!=order:raise GraphValidationError('child parent/order differs from explicit parent sequence')
            if identity in consumed:raise GraphValidationError('child edge belongs to more than one parent')
            consumed.add(identity);ordered.append(edge)
            for node_id in (edge.start_node_id,edge.end_node_id):node_parent_refs.setdefault(node_id,set()).add(parent.id)
        for connection,left,right in zip(parent.connections,ordered[:-1],ordered[1:]):
            end=nodes[left.end_node_id];start=nodes[right.start_node_id]
            if connection=='shared_cut':
                if left.end_node_id!=right.start_node_id or end.role!='cut':raise GraphValidationError('continuous cut requires the same explicit cut node ID')
            else:
                if left.end_node_id==right.start_node_id:raise GraphValidationError('gap requires distinct boundary instance IDs')
                if parent.direction*(start.physical_q_mm-end.physical_q_mm)<=0.:raise GraphValidationError('explicit gap must have positive ordered parameter separation')
    if consumed!=set(edges):raise GraphValidationError('orphan child edges are not listed by their parent')
    if any(nodes[node_id].role=='cut' and len(ps)>1 for node_id,ps in node_parent_refs.items()):
        raise GraphValidationError('cross-parent cut sharing needs a separate reviewed relation schema: NOT_IMPLEMENTED')
    return dict(version=VERSION,node_count=len(nodes),parent_count=len(parents),child_count=len(edges),
        unconnected_boundary_instances=sum(node_id not in node_parent_refs for node_id in nodes),
        arc_parser_connected_to178=True,no_coordinate_identity_merge=True,no_length_gap_or_child_count_filter=True,
        association_or_selection_learned=False,full_contract_pass=False)


def export_welds(graph):
    """Export only explicitly supplied relations; never run WIS or select cuts."""
    validate_graph(graph);nodes={n.id:n for n in graph.nodes};edges={e.id:e for e in graph.edges}
    point_cache={identity:list(n.xyz) for identity,n in nodes.items()};out=[]
    for parent in graph.parents:
        segments=[]
        for identity in parent.child_edge_ids:
            edge=edges[identity]
            if edge.kind=='line':
                segment=dict(kind='line',points=[point_cache[edge.start_node_id],point_cache[edge.end_node_id]])
            else:
                segment=sample_arc(CircularArc.from_parameters(edge.arc_parameters))
                segment['points'][0]=point_cache[edge.start_node_id];segment['points'][-1]=point_cache[edge.end_node_id]
                verify_sampled_arc(segment)
            segment['boundary_graph_edge_id']=edge.id;segment['start_node_id']=list(edge.start_node_id);segment['end_node_id']=list(edge.end_node_id)
            segments.append(segment)
        out.append(dict(id='parent_'+str(parent.id),type=parent.weld_type,segments=segments,confidence=parent.confidence,
                        boundary_graph_version=VERSION,runtime_parent_id=parent.id))
    return out


def to_dict(graph):
    validate_graph(graph)
    return dict(version=VERSION,nodes=[dict(sequence_id=n.sequence_id,joint_id=n.joint_id,instance_id=n.instance_id,
        xyz=list(n.xyz),physical_q_mm=n.physical_q_mm,confidence=n.confidence,role=n.role) for n in graph.nodes],
        parents=[dict(id=p.id,sequence_id=p.sequence_id,joint_id=p.joint_id,weld_type=p.weld_type,
            child_edge_ids=list(p.child_edge_ids),connections=list(p.connections),direction=p.direction,confidence=p.confidence) for p in graph.parents],
        edges=[dict(id=e.id,start_node_id=list(e.start_node_id),end_node_id=list(e.end_node_id),parent_id=e.parent_id,
            order=e.order,kind=e.kind,arc_parameters=e.arc_parameters) for e in graph.edges])


def from_dict(value):
    if not isinstance(value,Mapping) or value.get('version')!=VERSION:raise GraphValidationError('unsupported graph schema version')
    try:
        graph=BoundaryGraph(tuple(BoundaryNode(**r) for r in value['nodes']),tuple(ParentHypothesis(**r) for r in value['parents']),
                            tuple(OrderedChildEdge(**r) for r in value['edges']))
    except (KeyError,TypeError) as error:raise GraphValidationError('missing or invalid graph field: '+str(error)) from error
    validate_graph(graph);return graph
