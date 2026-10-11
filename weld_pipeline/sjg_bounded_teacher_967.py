"""Exact boundary-band labels without computing unused far-boundary distances."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_surface_learning_796 import inside_finite_face


def band_distance(points,face,band=1.):
    result=np.full(len(points),np.inf)
    if not len(points):return result
    tree=cKDTree(points)
    for line in face['lines']:
        for a,b in zip(np.asarray(line)[:-1],np.asarray(line)[1:]):
            delta=b-a;length=float(np.linalg.norm(delta))
            if length<1e-8:continue
            pieces=max(1,int(np.ceil(length/20.)));half=length/(2*pieces)
            centers=a+((np.arange(pieces)+.5)/pieces)[:,None]*delta
            # Every point <=band from the segment is within one such sphere,
            # including endpoint caps. Final exact segment distance decides.
            radius=half+band+1e-9
            for ids in tree.query_ball_point(centers,radius,workers=2):
                if not ids:continue
                ids=np.asarray(ids,np.int64);p=points[ids];t=np.clip((p-a)@delta/(length*length),0,1)
                distance=np.linalg.norm(p-a-t[:,None]*delta,axis=1)
                inside=distance<=band;selected=ids[inside]
                result[selected]=np.minimum(result[selected],distance[inside])
    return result


def point_teachers(xyz,faces,source):
    n=len(xyz);count=np.zeros(n,np.int16);instance=np.full(n,-1,np.int32);normal=np.zeros((n,3),np.float32)
    material=np.full(n,-1,np.int32);multiple=np.zeros(n,bool);boundary=np.full(n,np.inf,np.float32)
    for fi,face in enumerate(faces):
        lo=np.asarray(face['lo'])-.2;hi=np.asarray(face['hi'])+.2;direction=np.asarray(face['normal']);center=np.asarray(face['center'])
        eligible=[]
        for begin in range(0,n,65536):
            part=xyz[begin:begin+65536];ids=np.flatnonzero(np.all((part>=lo)&(part<=hi),axis=1))
            ids=ids[np.abs((part[ids]-center)@direction)<=.2]
            if len(ids):ids=ids[inside_finite_face(part[ids],face)]
            if len(ids):eligible.append(ids+begin)
        ids=np.concatenate(eligible) if eligible else np.empty(0,np.int64)
        if len(ids):
            fresh=ids[count[ids]==0];volume=int(face['volume']);instance[fresh]=fi;normal[fresh]=direction;material[fresh]=volume
            multiple[ids]|=material[ids]!=volume;count[ids]+=1
            boundary[ids]=np.minimum(boundary[ids],band_distance(xyz[ids],face))
        if fi%25==0:print('BOUNDED_NATIVE_TEACHER',source,'FACE',fi+1,'/',len(faces),flush=True)
    single=count==1;covered=count>0;instance[~single]=-1;normal[~single]=0;material[multiple|~covered]=-1
    return dict(instance=instance,normal=normal,instance_known=single,membership_count=count,reference_volume=material,
        multiple_material=multiple,boundary_mm=boundary,boundary_target=np.where(covered,(boundary<=1).astype(np.int8),-1),
        primitive_type=np.where(single,0,-1).astype(np.int8))


def controls():
    from weld_pipeline.sjg_point_teacher_803 import point_teachers as reference
    from weld_pipeline.sjg_local_runtime_904 import OUT,read
    faces=read(OUT/'run159/conditional_native_faces.json')['faces']
    xyz=np.load(OUT/'run96/source_21_xyz_mm.npy',mmap_mode='r')
    ids=np.linspace(0,len(xyz)-1,4096,dtype=np.int64);points=xyz[ids]
    expected=reference(points,faces,'control');actual=point_teachers(points,faces,'control')
    for key in expected:
        if key=='boundary_mm':
            near=expected[key]<=1;assert np.allclose(actual[key][near],expected[key][near],atol=1e-7)
        else:assert np.array_equal(actual[key],expected[key]),key
    # Endpoint caps, long segments, and out-of-band samples are covered.
    line=dict(lines=[[[0.,0,0],[100,0,0]]]);p=np.array([[-1.,0,0],[0,1,0],[50,1,0],[101,0,0],[50,1.0001,0]])
    d=band_distance(p,line);assert np.allclose(d[:4],1) and np.isinf(d[4])
    return dict(real_native_points=4096,all_training_labels_equal_to_803=True,boundary_band_distances_equal=True,
        distances_beyond_band_censored_not_used_by_training=True,endpoint_caps_and_long_segments_checked=True)
