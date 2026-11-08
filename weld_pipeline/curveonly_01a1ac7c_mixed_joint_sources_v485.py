"""Finite wall/observed base-plane sources, including curved-to-straight paths.

The actual finite wall bounds every proposed edge. The base plane proposes a
height only; finite base-face observations and all nominal gaps remain explicit.
These are model inputs, never automatic should-weld decisions.
"""
import math
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_sampler_v5 as sampler
from . import curveonly_01a1ac7c_paired_wall_native_caps_v432 as caps
from . import curveonly_01a1ac7c_contact_relations_v47 as relations
from . import curveonly_01a1ac7c_geometry_units_v26 as geometry
from . import curveonly_01a1ac7c_composite_v48 as prior
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.gp import gp_Pln,gp_Pnt,gp_Dir

NUMERIC=caps.NUMERIC


def native_edge(edge,R,t):
    edge=cad.TopoDS.Edge_s(edge);ad=cad.BRepAdaptor_Curve(edge);a,b=float(ad.FirstParameter()),float(ad.LastParameter())
    if not np.isfinite([a,b]).all() or b<=a:return None
    linear=caps.linear_edge(edge)
    if linear is not None:
        p=np.asarray([cad.vec(ad.Value(a)),cad.vec(ad.Value(b))]);kind='CERTIFIED_NATIVE_LINEAR'
    else:
        p=sampler.curve_points(edge);kind='NATIVE_NONLINEAR'
        if p is None:return None
    p=p@R.T+t;length=float(cad.arclength(p)[-1])
    if length<=1e-10:return None
    return dict(points=p.tolist(),circle=cad.circle_fit(p),native_geometry_kind=kind,
        native_type=str(ad.GetType()),native_linearity_certificate=linear,length_mm=length,
        source_curve_discretization_mm=0. if linear is not None else sampler.TOLERANCE_MM)


def build(packet,inventory,solids):
    if not solids or not packet.get('pose',{}).get('passed'):
        return dict(primitives=[],candidates=[],reason='NO_QUALIFIED_FINITE_STEP')
    context=caps.previous.step.Context(solids,packet,inventory);R,t=context.geometry.R,context.geometry.t
    relation_context=relations.Context(solids,packet['pose']['transform_STEP_to_XYZ'])
    primitives=[];sections=[]
    for base,panel in context.panels.items():
        for face_index in panel['dominant_faces']:
            face=inventory['faces'][face_index];n=np.asarray(face['normal']);offset=float(face['offset_mm'])
            plane=gp_Pln(gp_Pnt(*map(float,n*offset)),gp_Dir(*map(float,n)))
            for owner,solid in enumerate(solids):
                if owner==base:continue
                box=cad.bbox(solid);low=sum(n[i]*(box[i] if n[i]>=0 else box[i+3]) for i in range(3));high=sum(n[i]*(box[i+3] if n[i]>=0 else box[i]) for i in range(3))
                if not low-NUMERIC<=offset<=high+NUMERIC:continue
                operation=BRepAlgoAPI_Section(solid,plane,False);operation.SetRunParallel(False);operation.Build();ids=[]
                if operation.IsDone():
                    seen=set()
                    for edge_index,edge in enumerate(cad.items(operation.Shape(),cad.TopAbs_EDGE)):
                        value=native_edge(edge,R,t)
                        if value is None:continue
                        p=np.asarray(value['points']);sample=cad.resample(p,max(value['length_mm']/16.,.1))
                        key=min(np.round(sample,6).tobytes(),np.round(sample[::-1],6).tobytes())
                        if key in seen:continue
                        seen.add(key);ids.append(len(primitives))
                        primitives.append(dict(value,material=owner,STEP=True,pose=True,base=base,base_face=face_index,
                            base_local_face=face['local_face'],native_section_edge=edge_index,
                            source_kind='FINITE_WALL_SECTION_AT_OBSERVED_BASE_PLANE',
                            height_plane_is_only_proposal=True,finite_base_support_not_implied=True,
                            material_extent=np.sort(box[3:]-box[:3]).tolist()))
                sections.append(dict(base=base,base_face=face_index,wall=owner,done=operation.IsDone(),primitives=ids))
    options=[];choices={}
    for i,a in enumerate(primitives):
        if a['native_geometry_kind']!='NATIVE_NONLINEAR':continue
        for j,b in enumerate(primitives):
            if b['native_geometry_kind']!='CERTIFIED_NATIVE_LINEAR' or a['material']==b['material']:continue
            if (a['base'],a['base_face'])!=(b['base'],b['base_face']):continue
            for ae in [0,1]:
                for be in [0,1]:
                    pa=np.asarray(a['points'][0 if ae==0 else -1]);pb=np.asarray(b['points'][0 if be==0 else -1])
                    gap=float(np.linalg.norm(pa-pb))
                    if gap>prior.ENDPOINT_MM:continue
                    u,v=geometry.tangent(a,ae),geometry.tangent(b,be)
                    if u is None or v is None or -u@v<math.cos(math.radians(prior.TANGENT_DEGREES)):continue
                    qa,qb=np.asarray(relations.near_endpoint(a,ae)),np.asarray(relations.near_endpoint(b,be))
                    world=np.vstack([qa,(qa[0]+qb[0])/2.,qb]);support=None
                    for band in [sampler.TOLERANCE_MM,relations.OBSERVED_NEAR_MM]:
                        support=relation_context.face(world,allowed={a['base']},limit=band)
                        if support is not None and support['face']==a['base_local_face']:break
                        support=None
                    if support is None:continue
                    entry=dict(curve=i,line=j,curve_end=ae,line_end=be,nominal_endpoint_clearance_mm=gap,
                        tangent_deflection_deg=float(np.degrees(np.arccos(np.clip(-u@v,-1,1)))),
                        finite_base_observation=support,no_weld_gap_or_should_weld_permission=True)
                    index=len(options);options.append(entry)
                    for end in [(i,ae),(j,be)]:choices.setdefault(end,[]).append(index)
    candidates=[]
    for index,entry in enumerate(options):
        i,j,ae,be=entry['curve'],entry['line'],entry['curve_end'],entry['line_end']
        if len(choices[i,ae])!=1 or len(choices[j,be])!=1:continue
        a,b=primitives[i],primitives[j];left=np.asarray(a['points']);right=np.asarray(b['points'])
        if ae==0:left=left[::-1]
        if be==1:right=right[::-1]
        la=float(cad.arclength(left)[-1]);lb=float(cad.arclength(right)[-1]);gap=entry['nominal_endpoint_clearance_mm']
        points=np.vstack([left,right]);length=float(cad.arclength(points)[-1]);assert abs(length-la-lb-gap)<1e-7*max(1.,length)
        candidates.append(dict(points=points.tolist(),circle=cad.circle_fit(points),material=-2,STEP=True,pose=True,
            source_material_path=[dict(curve=i,material=a['material'],start_mm=0.,end_mm=la,reversed=ae==0),
                dict(curve=j,material=b['material'],start_mm=la+gap,end_mm=length,reversed=be==1)],
            source_connections=[entry],nominal_clearance_intervals=[dict(start_mm=la,end_mm=la+gap,length_mm=gap,
                source='ORIGINAL_FINITE_SECTION_ENDPOINT_CLEARANCE',weld_gap_or_process_cut_status='UNKNOWN_NOT_AUTOMATIC')],
            source_curve_count=2,source_length_mm=la+lb,nominal_clearance_length_mm=gap,
            base=a['base'],base_face=a['base_face'],
            source_kind='FINITE_CURVED_TO_LINEAR_DIFFERENT_WALL_BASE_PLANE_CANDIDATE',
            joint_identity_accepted=False,process_continuity_accepted=False,
            no_fixed_radius_extension_length_name_or_coordinates=True))
    return dict(primitives=primitives,candidates=candidates,sections=sections,
        original_sampling_endpoint_and_tangent_bounds_unchanged=True,
        all_finite_base_nearness_is_observation_not_should_weld=True,not_final_model_output=True)
