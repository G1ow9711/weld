"""Explicit material hypotheses and finite-face memberships from free XYZ outputs."""
import numpy as np


def material_graph(xyz,normal,boundary,material,face_original_ids):
    if not (len(xyz)==len(normal)==len(boundary)==len(material)):raise ValueError('Original point count mismatch')
    identities,inverse=np.unique(material,return_inverse=True)
    if (identities<0).any():raise ValueError('Unknown material memberships require an explicit mask, not a material node')
    order=np.argsort(inverse,kind='stable');cuts=np.r_[0,np.flatnonzero(np.diff(inverse[order]))+1,len(order)]
    features=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        ids=order[lo:hi];points=np.asarray(xyz[ids],np.float64);center=points.mean(0);delta=points-center
        covariance=delta.T@delta/max(1,len(ids));eigen=np.linalg.eigvalsh(covariance).clip(0)
        n=np.abs(normal[ids]);b=boundary[ids]
        row=np.r_[center/1000.,np.log1p(np.ptp(points,axis=0))/10.,eigen/max(eigen.sum(),1e-12),
            n.mean(0),n.std(0),b.mean(),b.std(),np.log1p(len(ids))/15.]
        assert row.shape==(18,);features.append(row)
    face_index=[];material_index=[];weight=[]
    for i,raw in enumerate(face_original_ids):
        ids=np.unique(np.asarray(raw,np.int64));assert len(ids) and np.all((ids>=0)&(ids<len(xyz)))
        owner,count=np.unique(inverse[ids],return_counts=True)
        face_index.extend([i]*len(owner));material_index.extend(owner.tolist());weight.extend((count/len(ids)).tolist())
    result=dict(material_x=np.asarray(features,np.float32),membership_face=np.asarray(face_index,np.int64),
        membership_material=np.asarray(material_index,np.int64),membership_weight=np.asarray(weight,np.float32))
    assert all(np.isfinite(v).all() for v in result.values())
    # Material IDs are only a graph indexing choice, never numeric model features.
    return result,dict(predicted_material_id_mapping=identities.tolist(),material_nodes=len(identities),
        membership_edges=len(weight),material_id_values_not_input_features=True,
        predicted_groups_not_customer_material_truth=True,all_original_points_used=True)
