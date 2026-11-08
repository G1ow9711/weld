"""Input-only tangent-connected carrier context, with original edges retained.

This never chooses should-weld intervals. Cheap existing station observations
are reordered and concatenated; no additional CAD or point-neighborhood query.
"""
import time
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_geometry_units_v26 as geometry


def groups(packet):
    curves=[]
    for ci,c in enumerate(packet['candidates']):
        mi=c.get('material',-1)
        curves.append(dict(c,weld_type='INPUT_GEOMETRY_ONLY',weld_side=None,
            _material=mi,_carrier='material_'+str(mi)if mi>=0 else'isolated_'+str(ci)))
    return geometry.units(curves)


def build(features,packet):
    begin=time.perf_counter();chains=groups(packet);candidates=[];all_x=[];offsets=[0]
    for chain in chains:
        xx=[];ss=[]
        for member in chain['members']:
            ci=member['curve'];c=packet['candidates'][ci];a,b=packet['offsets'][ci:ci+2]
            x=np.asarray(features[a:b],np.float32).copy();p=np.array(c['stations'])
            if member['reversed']:
                x=x[::-1].copy();p=p[::-1]
                for left,right in ((23,24),(27,28),(29,30),(12,14)):x[:,[left,right]]=x[:,[right,left]]
            if ss and np.linalg.norm(ss[-1][-1]-p[0])<1e-9:p=p[1:];x=x[1:]
            xx.append(x);ss.append(p)
        points=np.array(chain['points']);stations=np.vstack(ss);x=np.concatenate(xx)
        s=cad.arclength(stations);length=float(cad.arclength(points)[-1]);circle=chain['circle']
        first=packet['candidates'][chain['members'][0]['curve']]
        scale=float(np.expm1(x[0,0])/max(x[0,33],1e-12))
        tangent=np.gradient(stations,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)
        turn=abs(circle['sweep'])if circle else float(np.arccos(np.clip((tangent[1:]*tangent[:-1]).sum(1),-1,1)).sum())
        x[:,0]=np.log1p(length);x[:,1]=np.log1p(circle['radius'])if circle else 0.;x[:,2]=turn
        x[:,3]=circle is not None;x[:,4]=np.linalg.norm(points[0]-points[-1])<.1
        x[:,9]=s/max(s[-1],1e-12);x[:,10]=np.log1p(np.minimum(s,s[-1]-s));x[:,33]=length/max(scale,1.)
        candidate=dict(points=points.tolist(),stations=stations.tolist(),station_s_mm=s.tolist(),circle=circle,
            material=first['material'],STEP=first['STEP'],pose=first['pose'],material_extent=first.get('material_extent',[0,0,0]),
            other_distance=[float(x[0,12]),float(x[len(x)//2,13]),float(x[-1,14])],
            original_native_members=chain['members'],input_geometry_connections=chain['connections'],
            source_curve_count=len(chain['members']),source_length_mm=chain['source_length_mm'],
            numerical_bridge_length_mm=chain['numerical_bridge_length_mm'],joint_identity_accepted=False)
        assert len(x)==len(stations)and x.shape[1]==34 and np.isfinite(x).all()
        all_x.append(x);candidates.append(candidate);offsets.append(offsets[-1]+len(x))
    out=dict(packet,candidates=candidates,offsets=offsets,
        carrier_union=dict(original_candidates=len(packet['candidates']),connected_candidates=len(candidates),
            no_should_weld_rule=True,no_new_geometric_queries=True,original_edges_and_connections_retained=True,
            other_distance_columns_remain_piece_local_sparse_contacts=True,CPU_seconds=time.perf_counter()-begin))
    return np.concatenate(all_x)if all_x else np.empty((0,34),np.float32),out
