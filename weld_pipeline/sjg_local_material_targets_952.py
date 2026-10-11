"""Offline finite STEP ownership at TRAIN queries; no reference IDs as inputs."""
import argparse
import gc
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_surface_learning_796 import inside_finite_face
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier


def query_relations(points,normals,paired,faces):
    volumes=sorted({int(f['volume']) for f in faces});columns={v:i for i,v in enumerate(volumes)}
    membership=np.zeros((len(points),2,len(volumes)),bool)
    for side in (0,1):
        for face in faces:
            n=np.asarray(face['normal']);center=np.asarray(face['center'])
            possible=paired&(abs(normals[:,side]@n)>=np.cos(np.deg2rad(.1)))
            possible&=abs((points-center)@n)<=.2
            possible&=np.all((points>=np.asarray(face['lo'])-.2)&(points<=np.asarray(face['hi'])+.2),axis=1)
            ids=np.flatnonzero(possible)
            if not len(ids):continue
            ids=ids[inside_finite_face(points[ids],face)]
            membership[ids,side,columns[int(face['volume'])]]=True
    count=membership.sum(2);ambiguous=count>1;owners=np.full((len(points),2),-1,np.int32)
    for side in (0,1):
        unique=np.flatnonzero(count[:,side]==1)
        if len(unique):owners[unique,side]=np.asarray(volumes)[membership[unique,side].argmax(1)]
    known=paired&np.all(count>0,axis=1)
    common=(membership[:,0]&membership[:,1]).sum(1)
    same=((common>0)&known).astype(np.float32)
    different=((count[:,0]*count[:,1]>common)&known).astype(np.float32)
    # These two existence labels may both be true. They do not select a
    # unique material pair, and ambiguous owner identities remain -1.
    return different,known,owners,ambiguous,same


def controls():
    def face(n,volume):
        p=np.array([[-10.,-10,0],[10,-10,0],[10,10,0],[-10,10,0],[-10,-10,0]])
        if n[1]:p=p[:,[0,2,1]]
        return dict(normal=n,volume=volume,center=[0,0,0],lo=p.min(0),hi=p.max(0),lines=[p])
    p=np.zeros((1,3));n=np.array([[[0.,0,1],[0,1,0]]]);paired=np.array([True])
    a=face([0,0,1],1);b=face([0,1,0],2)
    y,m,_,_,same=query_relations(p,n,paired,[a,b]);assert m[0] and y[0]==1 and same[0]==0
    b['volume']=1;y,m,_,_,same=query_relations(p,n,paired,[a,b]);assert m[0] and y[0]==0 and same[0]==1
    extra=dict(a,volume=3);y,m,owners,ambiguous,same=query_relations(p,n,paired,[a,b,extra])
    assert m[0] and y[0]==1 and same[0]==1 and ambiguous[0,0] and owners[0,0]==-1
    return dict(same_body_edge_separate_from_contact=True,multiple_relation_hypotheses_retained=True,
        ambiguous_material_identity_remains_unknown=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--data-run',default='run133');ap.add_argument('--proposal-run',default='run132');a=ap.parse_args()
    dest=OUT/a.run;dest.mkdir(exist_ok=False);data=OUT/a.data_run;proposals=OUT/a.proposal_run
    inventory=read(OUT/'inventory.json');manifest=read(data/'training_manifest.json');cases={c['source_key']:c for c in read(OUT/'run98/local_cohort.json')['cases']}
    assert {r['source_key'] for r in manifest['cases']}=={'21','4','10','25','36','42','BM79A_native_local'}
    write(dest/'registered_protocol.json',dict(stage='LOCAL_QUERY_MATERIAL_RELATION_TEACHER',data_run=a.data_run,proposal_run=a.proposal_run,
        sources=[r['source_key'] for r in manifest['cases']],TRAIN_only=True,STEP_offline_only=True,
        ambiguous_owner_identity_unknown=True,relation_targets='same_possible_and_different_possible_independent_bits',
        existence_is_not_unique_material_assignment=True,reference_material_not_customer_approved_truth=True,source_sha256=digest(__file__),start_utc=utc()))
    write(dest/'controls.json',controls());rows=[]
    for row in manifest['cases']:
        check_resources();key=row['source_key'];path=data/row['input_file'];assert digest(path)==row['input_sha256']
        with np.load(path,allow_pickle=False) as z:nodes=z['nodes'];q=z['q_mm']
        y=np.zeros(len(nodes),np.float32);same=np.zeros(len(nodes),np.float32);known=np.zeros(len(nodes),bool);evidence={};reason='NO_NATIVE_REFERENCE_TRANSFORM'
        if key.isdigit():
            source=int(key);native_folder=(ROOT/inventory[source]['raw']['original_pcd_path']).parent
            exports=list(native_folder.glob('*_weld_welds.json'));cache=OUT/f'source_{source:02d}'
            if len(exports)==1 and (cache/'geometry.json').exists():
                native=read(exports[0]);assert native['coordinateFrame']=='reference_workpiece' and native['unit']=='millimeter'
                t=np.asarray(native['exportToWorld']);r=t[:3,:3];shift=t[:3,3];assert np.allclose(r.T@r,np.eye(3),atol=1e-7)
                frozen=read(cache/'input_freeze.json');assert digest(cache/'geometry.json')==frozen['geometry_sha256']
                teacher=read(cache/'geometry.json')['faces']
                for f in teacher:
                    f['normal']=(np.asarray(f['normal'])@r).tolist();f['center']=((np.asarray(f['center'])-shift)@r).tolist()
                    f['lines']=[(np.asarray(v)-shift)@r for v in f['lines']];vertices=np.concatenate(f['lines'])
                    f['lo']=vertices.min(0);f['hi']=vertices.max(0)
                folder=proposals/('source_'+key);binding=read(folder/'input_freeze.json')
                for name in ('profiles.json','surfaces.json'):assert digest(folder/name)==binding['files'][name]
                profiles=read(folder/'profiles.json');surfaces=read(folder/'surfaces.json')
                points=np.empty((len(nodes),3));normals=np.zeros((len(nodes),2,3));paired=np.zeros(len(nodes),bool)
                for node in np.unique(nodes):
                    ids=np.flatnonzero(nodes==node);profile=profiles[int(node)];points[ids]=CurveCarrier(profile['representation']).at(q[ids])
                    if len(profile['face_indices'])!=2:continue
                    first,second=profile['face_indices'];paired[ids]=first!=second
                    normals[ids,0]=surfaces[first]['normal'];normals[ids,1]=surfaces[second]['normal']
                y,known,owners,ambiguity,same=query_relations(points,normals,paired,teacher)
                reason='FINITE_REFERENCE_QUERY_OWNERSHIP';evidence=dict(native_transform_sha256=digest(exports[0]),
                    reference_geometry_sha256=digest(cache/'geometry.json'),ambiguous_queries=int(ambiguity.any(1).sum()),
                    reference_frame_provenance=inventory[source]['step']['verified_offline_frames'][0])
                del profiles,surfaces,teacher,points,normals
        file=dest/('source_'+key+'_material_relation.npz');np.savez(file,different_material=y,same_material_possible=same,material_relation_known=known)
        record=dict(source_key=key,file=file.name,sha256=digest(file),query_input_sha256=row['input_sha256'],queries=len(nodes),
            known=int(known.sum()),same=int((known&(same==1)).sum()),different=int((known&(y==1)).sum()),
            both_possible=int((known&(same==1)&(y==1)).sum()),unique_material_assignment_not_claimed=True,reason=reason,**evidence)
        rows.append(record);print('LOCAL_MATERIAL_RELATION',key,'KNOWN',record['known'],'SAME',record['same'],'DIFFERENT',record['different'],flush=True);gc.collect()
    write(dest/'manifest.json',dict(rows=rows,data_manifest_sha256=digest(data/'training_manifest.json')))
    write(dest/'receipt.json',dict(status='LOCAL_MATERIAL_QUERY_TARGETS_COMPLETE',rows=rows,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))

if __name__=='__main__':main()
