"""Thickness/material-outward candidates from parallel point-supported faces."""
import numpy as np
from scipy.spatial import cKDTree

from weld_pipeline.point_frame import dominant_xy_frame

VERSION='point_parallel_plate_attributes_v2'


class PointPlateAttributes:
    def __init__(self,xyz):
        self.xyz=np.asarray(xyz,float)
        self.tree=cKDTree(self.xyz)
        self.frame=None

    def face(self,point,normal,tangent):
        normal=np.asarray(normal,float)
        normal/=np.linalg.norm(normal)
        cross=np.cross(tangent,normal)
        cross/=np.linalg.norm(cross)
        ids=self.tree.query_ball_point(point,150.)
        p=self.xyz[ids]-point
        p=p[(np.abs(p@tangent)<=25.) & (np.abs(p@normal)<=50.) & (np.abs(p@cross)<=120.)]
        if len(p)<8:
            return {'state':'unknown','reason':'INSUFFICIENT_SURFACE_SUPPORT'}
        d=p@normal
        def support(offset):
            selected=p[np.abs(d-offset)<.15]
            along=selected@tangent
            side=selected@cross
            if len(selected)<4 or np.ptp(along)<5. or np.ptp(side)<20.:
                return None
            covariance=np.cov(np.column_stack([along,side]).T)
            if np.linalg.eigvalsh(covariance)[0]<1.:
                return None
            return {'count':len(selected),'low':float(side.min()),'high':float(side.max()),
                    'offset':float(np.median(selected@normal)),'side_span':float(np.ptp(side))}
        root=support(0.)
        if root is None:
            return {'state':'unknown','reason':'CONTACT_FACE_NOT_SUPPORTED'}
        bins,counts=np.unique(np.round(d/.05)*.05,return_counts=True)
        options=[]
        for offset in bins[(counts>=4) & (np.abs(bins)>=1.) & (np.abs(bins)<=50.)]:
            opposite=support(offset)
            if opposite is None:
                continue
            overlap=max(0.,min(root['high'],opposite['high'])-max(root['low'],opposite['low']))
            if overlap<20.:
                continue
            score=overlap/min(root['side_span'],opposite['side_span'])
            # Surface extent matching takes priority over vertex/edge rows.
            options.append({**opposite,'score':score,'overlap':overlap})
        if not options:
            return {'state':'unknown','reason':'OPPOSING_PARALLEL_FACE_NOT_SUPPORTED'}
        options.sort(key=lambda a:(-a['score'],-a['overlap'],-a['count'],abs(a['offset'])))
        best=options[0]
        competing=[a for a in options[1:] if np.sign(a['offset'])!=np.sign(best['offset']) and
                   a['score']>=best['score']-.02 and a['overlap']>=best['overlap']*.95]
        if competing:
            return {'state':'unknown','reason':'MATERIAL_SIDE_AMBIGUOUS','parallel_candidates':options[:6]}
        distance=best['offset']-root['offset']
        outward=-np.sign(distance)*normal
        confidence=float(min(1.,best['score'])*min(1.,best['overlap']/50.))
        return {'state':'estimated','source':'AllFaces_XYZ_parallel_surface_support',
                'thickness_mm':abs(float(distance)),'normal_material_outward':outward.tolist(),
                'confidence':confidence,'support':{'contact':root,'opposite':best},
                'direction_policy':'Outward from the material interval between matched faces'}

    def line(self,points):
        points=np.array(points,float)
        u=points[-1]-points[0]
        u/=np.linalg.norm(u)
        if abs(u[2])>.95:
            if self.frame is None:
                self.frame=dominant_xy_frame(self.xyz)
            axes=[np.r_[n,0.] for n in self.frame]
        elif abs(u[2])<.05:
            axes=[np.cross(u,[0.,0.,1.]),np.array([0.,0.,1.])]
        else:
            return {'state':'unknown','reason':'OBLIQUE_CONTACT_FRAME_NOT_IMPLEMENTED','samples':[]}
        samples=[]
        for fraction in [0.,.25,.5,.75,1.]:
            q=points[0]*(1-fraction)+points[-1]*fraction
            faces=[self.face(q,n,u) for n in axes]
            valid=all(f['state']=='estimated' for f in faces)
            if valid:
                normals=[np.array(f['normal_material_outward']) for f in faces]
                if np.cross(normals[0],normals[1])@u<0:
                    faces=faces[::-1]
            samples.append({'fraction':fraction,'xyz_mm':q.tolist(),'sides':faces,'both_estimated':valid})
        return {'state':'estimated' if all(s['both_estimated'] for s in samples) else 'partial_or_unknown',
                'side_convention':'cross(normal_1,normal_2) dot directed weld tangent positive',
                'samples':samples,'customer_side_convention_confirmed':False}
