"""Strict workpiece-local millimetre input; no implicit scale or world conversion."""
from pathlib import Path
import numpy as np


def read_workpiece_local_surface(path):
    path=Path(path)
    if not path.name.endswith('_weld_workpiece.ply'):
        raise ValueError('Use the geometry-only workpiece export; overview/weld-label clouds are not inputs')
    header=[]
    with path.open('r',encoding='ascii') as stream:
        for line in stream:
            header.append(line.strip())
            if line.strip()=='end_header':break
        if 'format ascii 1.0' not in header or not any('unit millimeter; frame reference_workpiece' in s for s in header):
            raise ValueError('Explicit millimetre / reference_workpiece declaration is required')
        count=int(next(s.split()[-1] for s in header if s.startswith('element vertex ')))
        properties=[s.split()[-1] for s in header if s.startswith('property ')]
        if properties[:3]!=['x','y','z']:raise ValueError('XYZ must be the first three vertex properties')
        points=np.loadtxt(stream,usecols=(0,1,2),dtype=np.float64,ndmin=2)
    if points.shape!=(count,3) or not np.isfinite(points).all():raise ValueError('Invalid original point count or XYZ')
    return points,dict(coordinate_frame='workpiece_local',unit='mm',original_points=count,
        coordinate_transform_applied=False,scale_applied=1.,only_XYZ_used=True,header=header)


def validate_local_pair(input_metadata,labels):
    for item in (input_metadata,labels):
        if item.get('coordinate_frame')!='workpiece_local' or item.get('unit')!='mm':
            raise ValueError('Training requires explicitly matched workpiece-local millimetre frames')
    children=[c for p in labels['parents'] for c in p['children']]
    if len(children)+labels.get('unresolved_children',0)!=labels['original_children']:
        raise ValueError('Original child denominator changed')
    for child in children:
        if 'world_representation' in child:raise ValueError('Legacy world representation cannot enter local training silently')
        representation=child['representation']
        if representation['kind']=='line':
            points=np.asarray(representation['endpoints_mm'])
            if points.shape!=(2,3) or not np.isfinite(points).all():raise ValueError('Invalid local line')
        elif representation['kind']=='arc':
            from weld_pipeline.contract_curve_geometry import CircularArc
            CircularArc.from_parameters(representation['arc_parameters'])
        else:raise ValueError('Unknown geometry must be retained explicitly, not relabelled')
    return True
