"""Pure whole-parent chart correspondence; original physical plate slots stay fixed."""
import numpy as np
from weld_pipeline.contract_geometry_d3_409 import variants,synchronized_endpoint_fields

def target_variants(record):
    p=np.asarray(record['child_parameter_endpoints_mm'],float)
    n=record['original_child_count'];assert p.shape==(n,2)
    d=np.sign(p[:,1]-p[:,0]);assert np.all(d==d[0]) and d[0]!=0
    assert np.all(np.diff(p[:,0])*d[0]>=0),'Independent child flips are not a parent correspondence'
    parent=dict(segments=[dict(kind='line',points=[[a,0.,0.],[z,0.,0.]]) for a,z in p],
        gaps=record['original_gap_status'],is_closed=record['parent_is_closed'])
    policy=record.get('direction_semantics_confirmed')
    for view in variants(parent,policy is not True):
        # A linear candidate chart has one finite cut; a cyclic view is only
        # represented when its entire ordered chart can be expressed there.
        positions=np.array([[s['points'][0][0],s['points'][-1][0]] for _,s in view['segments']])
        negative=positions[0,1]<positions[0,0]
        ordered=positions[::-1,::-1] if negative else positions
        if np.any(np.diff(ordered[:,0])<0):continue
        yield dict(record,child_parameter_endpoints_mm=positions.tolist(),
            original_child_ids=[record['original_child_ids'][i] for i,s in view['segments']] if 'original_child_ids' in record else [],
            parent_chart_direction=-1 if negative else 1,original_gap_status=view['gaps'],
            reversed_whole_parent=view['reversed_whole_parent'],cyclic_child_shift=view['cyclic_child_shift'])

def canonical_child_fields(fields,negative_chart):
    return synchronized_endpoint_fields(fields,bool(negative_chart))
