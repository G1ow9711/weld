"""Post-freeze TRAIN diagnosis: trace known endpoints through mesh topology."""
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc
from weld_pipeline.sjg_local_sampling_mesh_991 import two_dimensional_radius
from weld_pipeline.sjg_xyz_batch_795 import intervals


def main():
    folder=OUT/'run182'
    freeze=read(folder/'input_freeze.json')
    for name,sha in freeze['files'].items():
        assert digest(folder/name)==sha
    xyz=np.load(OUT/'run96/source_21_xyz_mm.npy',mmap_mode='r')
    faces=read(folder/'surfaces.json')
    supports=np.load(folder/'surface_support.npz')
    old=read(OUT/'run181/endpoint_membership_audit.json')
    endpoints=np.array([r['target_endpoint'] for r in old['rows']])
    output=[]
    for fid in (1,5):
        face=faces[fid]
        for key in ('normal','center','basis'):face[key]=np.asarray(face[key])
        ids=supports[face['point_ids_key']];tri=supports[face['triangles_key']]
        uv=(xyz[ids]-face['center'])@face['basis']
        radius,used=two_dimensional_radius(uv)
        all_tri=Delaunay(uv).simplices
        keep=np.all(used[all_tri]>0,axis=1)
        edges=[]
        for i,j in ((0,1),(1,2),(2,0)):
            edge=np.linalg.norm(uv[all_tri[:,i]]-uv[all_tri[:,j]],axis=1)
            limit=np.maximum(2.,1.5*np.minimum(radius[all_tri[:,i]],radius[all_tri[:,j]]))
            edges.append(edge/limit)
            keep &= edge<=limit
        assert np.array_equal(all_tri[keep],tri)
        tree=cKDTree(xyz[ids]);rows=[]
        for endpoint in endpoints:
            dist,near=tree.query(endpoint,k=4)
            incident=np.flatnonzero(np.isin(all_tri,near[:1]).any(1))
            rows.append(dict(endpoint=endpoint.tolist(),nearest_original_ids=ids[near].tolist(),
                nearest_distances_mm=dist.tolist(),nearest_vertex_uv=uv[near].tolist(),
                radius_mm=radius[near].tolist(),neighborhood_k=used[near].tolist(),
                raw_incident_triangles=len(incident),kept_incident_triangles=int(keep[incident].sum()),
                incident=[dict(kept=bool(keep[t]),original_ids=ids[all_tri[t]].tolist(),
                    xyz=xyz[ids[all_tri[t]]].tolist(),radius=radius[all_tri[t]].tolist(),
                    used=used[all_tri[t]].tolist(),edge_over_limit=[float(e[t]) for e in edges]) for t in incident]))
        face.update(point_ids=ids,triangles=tri)
        output.append(dict(face=fid,normal=face['normal'].tolist(),endpoints=rows))
        print('AUDIT_FACE',fid,[(r['nearest_distances_mm'][0],r['raw_incident_triangles'],r['kept_incident_triangles']) for r in rows],flush=True)
    a,b=faces[1],faces[5]
    a.update(point_ids=supports[a['point_ids_key']],triangles=supports[a['triangles_key']])
    for key in ('normal','center','basis'):b[key]=np.asarray(b[key])
    b.update(point_ids=supports[b['point_ids_key']],triangles=supports[b['triangles_key']])
    normals=np.stack([a['normal'],b['normal']]);offset=np.array([a['offset'],b['offset']])
    origin=normals.T@np.linalg.solve(normals@normals.T,offset)
    direction=np.cross(*normals);direction/=np.linalg.norm(direction)
    if direction[np.argmax(abs(direction))]<0:direction=-direction
    spans=[intervals(f,xyz,origin,direction) for f in (a,b)]
    value=dict(stage='POST_FREEZE_TRAIN_ENDPOINT_MESH_DIAGNOSIS',rows=output,
        pair_1_5_origin=origin.tolist(),pair_1_5_direction=direction.tolist(),
        pair_1_5_intervals=spans,endpoint_projection=((endpoints-origin)@direction).tolist(),
        reference_only_for_diagnosis=True,optimizer_updates=0,end_utc=utc())
    write(folder/'endpoint_topology_audit.json',value)
    print('PAIR_INTERVALS',spans,flush=True)


if __name__=='__main__':main()
