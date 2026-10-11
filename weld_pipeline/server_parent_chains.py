"""Provisional geometric parent assembly from selected, directed model spans.

No labels or IDs are used. Only shared endpoints, a common measured face and
continuous directed tangents can link children. Material identity is pending.
"""
import numpy as np
from weld_pipeline.server_geometry_events import geometry
from weld_pipeline.world_set_evaluation_285 import reference_segment


def assemble(chosen,candidates):
    endpoints=[];tangents=[];segments=[]
    for row in chosen:
        length,point,tangent,arc=geometry(row['representation']);s=np.array([0.,length]);endpoints.append(point(s));tangents.append(tangent(s));segments.append(dict(reference_segment(dict(world_representation=row['representation'])),weld_type=row['weld_type']))
    edges=[]
    for i,a in enumerate(chosen):
        for j,b in enumerate(chosen):
            if i==j or a['weld_type']!=b['weld_type']:continue
            if not set(candidates[a['candidate']]['face_indices'])&set(candidates[b['candidate']]['face_indices']):continue
            distance=float(np.linalg.norm(endpoints[i][1]-endpoints[j][0]));dot=float(tangents[i][1]@tangents[j][0])
            if distance<=3 and dot>=.9:edges.append((distance+1-dot,i,j))
    successor={};predecessor={};links=[]
    for cost,i,j in sorted(edges):
        if i in successor or j in predecessor:continue
        successor[i]=j;predecessor[j]=i;links.append(dict(source=i,target=j,geometric_cost=cost))
    visited=set();parents=[];groups=[]
    for start in sorted(range(len(chosen)),key=lambda i:(i in predecessor,i)):
        if start in visited:continue
        group=[];at=start
        while at not in visited:
            visited.add(at);group.append(at)
            if at not in successor:break
            at=successor[at]
        closed=bool(group and successor.get(group[-1])==group[0])
        parents.append(dict(child_count=len(group),is_closed=closed,gaps=['SHARED']*(len(group)-1),segments=[segments[i] for i in group]));groups.append(group)
    return dict(parents=parents,invalid_predictions=[],total_prediction_children=len(chosen),parent_linkage_status='GEOMETRIC_SHARED_ENDPOINT_PROVISIONAL',links=links,groups=groups)
