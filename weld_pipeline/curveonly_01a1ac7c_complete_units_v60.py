"""Complementary full-circle geometry across inferred native owner changes.

An inferred STEP carrier owner is not a customer-certified physical joint ID.
All original process boundaries/owners remain in evidence. Only geometry is
normalized; side, process, direction and attributes are not thereby accepted.
"""
import math
import types
import numpy as np
from . import curveonly_01a1ac7c_complete_units_v36 as previous
from . import curveonly_01a1ac7c_geometry_units_v26 as legacy

VERSION='CURVEONLY_COMPLETE_CIRCLE_INFERRED_OWNER_V60'
OWNER_SOURCE='NATIVE_CARRIER_OWNER_INFERRED_NOT_CUSTOMER_JOINT_ID'
POLICY=dict(previous.POLICY,
    complementary_closed_circle_exception='Both ends identity<=.001mm, sameanalyticcircleidentity<=.001mm, complementary singleturn, inferredownersonly, no confirmedjoint/side/type/cut change.',
    inferred_native_owner_alone_not_physical_joint_identity=True,
    known_physical_joint_identity_changes_still_block=True,
    original_owner_changes_and_process_boundaries_preserved_in_evidence=True)


def complementary(a,b):
    if a.get('_material_identity_source')!=OWNER_SOURCE or b.get('_material_identity_source')!=OWNER_SOURCE:return False
    if a.get('_material') is None or b.get('_material') is None or a['_material']==b['_material']:return False
    if a.get('_confirmed_joint_id') is not None and b.get('_confirmed_joint_id') is not None and a['_confirmed_joint_id']!=b['_confirmed_joint_id']:return False
    # This substitution removes only the inferred-owner barrier for the check.
    if legacy.compatible(a,dict(b,_material=a['_material']))!='SAME_ANALYTIC_CIRCLE_GEOMETRY_ONLY':return False
    x,y=a.get('circle'),b.get('circle')
    if x is None or y is None or abs(abs(x['sweep'])+abs(y['sweep'])-2*math.pi)>1e-6:return False
    p=np.asarray(a['points'])[[0,-1]];q=np.asarray(b['points'])[[0,-1]]
    d=np.linalg.norm(p[:,None]-q[None],axis=2)
    return bool(min(max(d[0,0],d[1,1]),max(d[0,1],d[1,0]))<=.001)


def link_evidence(a,b,ae,be):
    if a.get('_confirmed_joint_id') is not None and b.get('_confirmed_joint_id') is not None and a['_confirmed_joint_id']!=b['_confirmed_joint_id']:return None
    old=previous.link_evidence(a,b,ae,be)
    if old is not None:return old
    if not complementary(a,b):return None
    if ae in a.get('_confirmed_process_cut_ends',[]) or be in b.get('_confirmed_process_cut_ends',[]):return None
    d=float(np.linalg.norm(np.asarray(a['points'])[0 if ae==0 else-1]-np.asarray(b['points'])[0 if be==0 else-1]))
    if d>.001:return None
    u,v=legacy.tangent(a,ae),legacy.tangent(b,be)
    if u is None or v is None or -u@v<math.cos(math.pi/36):return None
    return dict(distance_mm=d,reason='EXACT_COMPLEMENTARY_CIRCULAR_GEOMETRY_INFERRED_OWNER_CHANGE',
        inferred_native_materials=[a['_material'],b['_material']],
        physical_joint_and_process_identity='UNKNOWN_NOT_CERTIFIED_BY_NORMALIZATION',
        process_cut_status='ORIGINAL_BOUNDARIES_RETAINED_NOT_CANCELED')


def units(curves):
    return types.FunctionType(previous.units.__code__,dict(previous.__dict__,link_evidence=link_evidence))(curves)
