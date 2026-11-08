"""Keep the original-edge index domain separate from generated source IDs."""
import copy
import types
import numpy as np
from . import curveonly_01a1ac7c_native_boundary_export_v521 as previous

native,provenance=previous.native,previous.provenance
NUMERIC,CONVERSION,KINDS=previous.NUMERIC,previous.CONVERSION,previous.KINDS
correct=previous.correct


def source_events(view, raw):
    candidates=list(view['candidates'])
    withheld=[]
    for ci,candidate in enumerate(candidates):
        members=candidate.get('original_native_members')
        if candidate.get('source_material_path') or not members:continue
        parts=[];reason=None
        for member in members:
            primitive=member.get('curve')
            if not isinstance(primitive,(int,np.integer)) or not 0<=primitive<len(raw):
                reason='MEMBER_IS_NOT_AN_ORIGINAL_MATERIAL_EDGE_INDEX';break
            if 'reversed' not in member:
                reason='ORIGINAL_EDGE_ORIENTATION_UNRESOLVED';break
            points=raw[int(primitive)]['points']
            parts.append(points[::-1] if member['reversed'] else points)
        if reason is None:
            replay=np.vstack(parts);points=np.asarray(candidate['points'])
            if replay.shape!=points.shape or not np.allclose(replay,points,atol=1e-9,rtol=0):
                reason='ORIGINAL_EDGE_POINTS_NOT_EXACTLY_REPLAYED'
        if reason:
            # This is an observation-only metadata view. Actual source points,
            # model features, original metadata and material paths stay intact.
            candidates[ci]=dict(candidate,original_native_members=[])
            withheld.append(dict(candidate=ci,reason=reason,
                original_member_metadata=copy.deepcopy(members),
                source_endpoints_and_explicit_material_path_boundaries_retained=True,
                no_original_edge_boundary_fabricated=True,process_semantics='UNKNOWN'))
    safe=dict(view,candidates=candidates)
    events,checks=native.source_events(safe,raw)
    return events,checks,withheld


def bank(packet,decoded,record=None,solids=None):
    withheld=[]
    def collect(view,raw):
        events,checks,unknown=source_events(view,raw)
        withheld.extend(unknown)
        return events,checks
    proxy=types.SimpleNamespace(**dict(native.__dict__,source_events=collect))
    build=types.FunctionType(previous.bank.__code__,dict(previous.__dict__,native=proxy))
    result=build(packet,decoded,record,solids)
    if withheld:result['withheld_unreplayed_original_members']=withheld
    return result
