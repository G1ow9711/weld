"""Gravity-frame endpoint plane measurements derived only from finite XYZ."""
import numpy as np
from scipy.spatial import cKDTree

VERSION='endpoint_plane_physical_observations_v1'
RADII_MM=(5.,15.,40.,100.)


class EndpointPlanes:
    def __init__(self,xyz):
        self.xyz=np.asarray(xyz,float)
        if self.xyz.ndim!=2 or self.xyz.shape[1]!=3 or not np.isfinite(self.xyz).all():
            raise ValueError('Finite XYZ in millimeters required')
        self.projections={};self.planes={}
        z,count=np.unique(np.round(self.xyz[:,2],2),return_counts=True)
        self.levels=z[count>=max(20,len(xyz)*.001)]

    def plane(self,normal,point):
        n=np.asarray(normal,float).copy();n/=np.linalg.norm(n)
        if n[np.argmax(np.abs(n))]<0:
            n=-n
        axis=tuple(n.round(7));offset=float(np.asarray(point)@n)
        key=(*axis,round(offset,3))
        if key not in self.planes:
            if axis not in self.projections:
                self.projections[axis]=self.xyz@n
            p=self.xyz[abs(self.projections[axis]-offset)<=.15]
            self.planes[key]=(p,cKDTree(p[:,:2]) if len(p) else None)
        return self.planes[key],n

    def observe(self,point,normals):
        point=np.asarray(point,float);rows=[]
        for normal in np.asarray(normals).reshape(2,3):
            (p,tree),n=self.plane(normal,point)
            profile={'normal':n.tolist(),'support':len(p),'vertical_normal':abs(float(n[2]))<.4,'windows':{}}
            if len(p):
                profile['global_z_min']=float(p[:,2].min());profile['global_z_max']=float(p[:,2].max())
                profile['global_span']=float(np.ptp(p[:,2]))
            for radius in RADII_MM:
                ids=tree.query_ball_point(point[:2],radius) if tree else []
                z=p[ids,2] if len(ids) else np.empty(0)
                if len(z):
                    profile['windows'][str(radius)]={'support':len(z),'span':float(np.ptp(z)),
                        'top_above_endpoint':float(z.max()-point[2]),'bottom_below_endpoint':float(point[2]-z.min()),
                        'q95_above_endpoint':float(np.quantile(z,.95)-point[2]),'top_z':float(z.max()),'bottom_z':float(z.min())}
                else:
                    profile['windows'][str(radius)]={'support':0}
            rows.append(profile)
        above=self.levels[self.levels>=point[2]-.15];below=self.levels[self.levels<=point[2]+.15]
        return {'endpoint_z':float(point[2]),'planes':rows,
            'nearest_horizontal_level_above':float(above.min()) if len(above) else None,
            'nearest_horizontal_level_below':float(below.max()) if len(below) else None,
            'levels_mm':self.levels.tolist()}


def physical_height_candidates(observation):
    values={}
    for i,p in enumerate(observation['planes']):
        if not p['vertical_normal']:
            continue
        if p['support']:
            values['plane_'+str(i)+'_global_span']=p['global_span']
        for radius,w in p['windows'].items():
            if not w['support']:
                continue
            for field in ['span','top_above_endpoint','q95_above_endpoint']:
                values['plane_'+str(i)+'_r'+radius+'_'+field]=w[field]
            if observation['nearest_horizontal_level_below'] is not None:
                values['plane_'+str(i)+'_r'+radius+'_top_from_local_level']=w['top_z']-observation['nearest_horizontal_level_below']
    return values
