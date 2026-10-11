"""PTv3 face graph and joint inputs, frozen before separate weld supervision."""
import argparse
import gc
import traceback
import numpy as np
from scipy.spatial import cKDTree
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read, write, digest
from weld_pipeline.sjg_prepare_776 import ROOT, OUT, TRAIN, VALID, ARC
from weld_pipeline.sjg_train_776 import check_resources, utc
from weld_pipeline.sjg_pure_data_799 import support_distance

PROPOSALS = OUT/'run34'
FEATURES = OUT/'run27'
SEED = 816


def load_inputs(si, inventory):
    folder = PROPOSALS/f'source_{si:02d}'
    binding = read(folder/'input_freeze.json')
    for name in ('surfaces.json', 'profiles.json'):
        assert digest(folder/name) == binding['files'][name]
    f = read(FEATURES/f'source_{si:02d}_freeze.json')
    assert digest(FEATURES/f['feature_file']) == f['feature_sha256']
    raw = ROOT/inventory[si]['raw']['original_pcd_path']
    assert digest(raw) == binding['raw_sha256'] == f['raw_sha256']
    xyz = read_xyz_pcd(raw)*1000.
    with np.load(FEATURES/f['feature_file']) as z:
        point = {k:z[k] for k in ('embedding', 'normal', 'boundary')}
    faces = read(folder/'surfaces.json')
    profiles = read(folder/'profiles.json')
    return xyz, point, faces, profiles


def graph_inputs(xyz, point, faces, profiles):
    archives = {}
    face_x = []
    try:
        for face in faces:
            path = ROOT/face['support_file']
            if str(path) not in archives:
                assert digest(path) == face['support_sha256']
                archives[str(path)] = np.load(path, allow_pickle=False)
            ids = archives[str(path)][face['point_ids_key']].astype(np.int64)
            e = point['embedding'][ids]
            normal = np.asarray(face['normal'])
            agreement = np.abs(point['normal'][ids] @ normal)
            edge = point['boundary'][ids]
            # A soft confidence weight retains every observed support point.
            weights = .1 + .9*agreement*(1-.75*edge)
            weights /= weights.sum()
            mean = (e*weights[:, None]).sum(0)
            std = np.sqrt(((e-mean)**2*weights[:, None]).sum(0))
            local = xyz[ids]-np.asarray(face['center'])
            residual = np.abs(local@normal)
            face_x.append(np.r_[mean, std, normal, np.log1p(len(ids))/12,
                agreement.mean(), agreement.std(), edge.mean(), edge.std(),
                np.log1p(np.ptp(local, axis=0))/10, residual.mean(), residual.max(),
                float(face['proposal_branch']=='run33')])
    finally:
        for archive in archives.values(): archive.close()
    face_x = np.asarray(face_x, np.float32)
    assert face_x.shape == (len(faces), 46)
    centers = np.asarray([f['center'] for f in faces])
    normals = np.asarray([f['normal'] for f in faces])
    knn = cKDTree(centers).query(centers, k=min(16, len(faces)))[1]
    if knn.ndim == 1: knn = knn[:, None]
    neighbors = np.asarray(knn, np.int64)
    relation = np.concatenate([(centers[neighbors]-centers[:, None])/1000,
        np.abs((normals[neighbors]*normals[:, None]).sum(-1))[..., None]], -1).astype(np.float32)
    pairs = np.asarray([p['face_indices'] for p in profiles], np.int64)
    geometry = []
    for p, (i, j) in zip(profiles, pairs):
        center = (np.asarray(p['p0'])+p['p1'])/2
        length = max(p['length'], 1e-8)
        coverage = sum(b-a for a, b in p['support_intervals_mm'])/length
        gap = max([b-a for a, b in p['unobserved_gaps_mm']], default=0.)/length
        geometry.append(np.r_[p['direction'], np.log1p(length)/10,
            abs(normals[i]@normals[j]), (centers[i]-center)/1000,
            (centers[j]-center)/1000, coverage, gap,
            np.log1p(len(p['support_intervals_mm']))/5,
            np.log1p(len(p['original_contact_ids']))/5, 1.])
    geometry = np.asarray(geometry, np.float32)
    assert geometry.shape == (len(profiles), 16)
    assert np.isfinite(face_x).all() and np.isfinite(geometry).all()
    return dict(face_x=face_x, neighbors=neighbors, relation=relation,
                pairs=pairs, geometry=geometry)


def supervision(profiles, target):
    origin = np.asarray([p['p0'] for p in profiles])
    direction = np.asarray([p['direction'] for p in profiles])
    lengths = np.asarray([p['length'] for p in profiles])
    labels = np.zeros(len(profiles), np.float32)
    children = []
    by_profile = {}
    missing = []
    for pi, parent in enumerate(target['parents']):
        for ci, child in enumerate(parent['children']):
            if child['original_kind'] != 'line':
                missing.append(dict(parent=pi, child=ci, reason='CURVE_NOT_IMPLEMENTED'))
                continue
            points = np.asarray(child['world_representation']['endpoints_mm'])
            q = np.stack([np.einsum('ij,ij->i',p-origin,direction) for p in points], 1)
            nearest = origin[:, None]+np.clip(q,0,lengths[:, None])[:, :, None]*direction[:, None]
            error = np.linalg.norm(nearest-points[None],axis=-1).max(1)
            options = np.flatnonzero(error<=3.)
            if not len(options):
                missing.append(dict(parent=pi, child=ci, reason='NO_PROFILE_WITHIN_3MM'))
                continue
            labels[options] = 1
            choices = []
            for k in options:
                lo, hi = np.clip(np.sort(q[k]), 0, lengths[k])
                event = dict(parent=pi, child=ci, lo=float(lo), hi=float(hi),
                             direction=bool(q[k,1]>=q[k,0]))
                by_profile.setdefault(str(int(k)), []).append(event)
                choices.append(int(k))
            children.append(dict(parent=pi, child=ci, profiles=choices))
    return labels, dict(original_G=target['original_children'], covered_children=len(children),
        children=children, missing=missing, by_profile=by_profile,
        labels_mean_geometrically_compatible_weld_support=True,
        duplicate_material_identity_not_resolved=True, not_extraction_accuracy=True)


def point_queries(profiles, nodes, q, xyz, point, tree):
    positions = np.empty((len(q),3)); extra = np.empty((len(q),3))
    for ni in np.unique(nodes):
        ids = np.flatnonzero(nodes==ni); p=profiles[int(ni)]
        positions[ids] = np.asarray(p['p0'])+q[ids,None]*p['direction']
        distance = support_distance(p, q[ids])
        extra[ids] = np.c_[distance/50, (distance<=1e-6).astype(float), np.ones(len(ids))]
    distance, ids = tree.query(positions, k=16, workers=2)
    patch = np.concatenate([point['embedding'][ids], point['normal'][ids],
        point['boundary'][ids,None], np.clip((xyz[ids]-positions[:,None])/50,-20,20)], -1).astype(np.float32)
    length = np.asarray([profiles[int(i)]['length'] for i in nodes])
    query = np.c_[q/length,q/1000,(length-q)/1000,distance[:,0]/50,extra].astype(np.float32)
    assert patch.shape == (len(q),16,23) and query.shape == (len(q),7)
    return patch, query


def training_queries(si, profiles, labels, truth, xyz, point):
    rng=np.random.default_rng(SEED+si); examples=[]
    by_profile=truth['by_profile']; children=truth['children']
    # Equal child sampling prevents a child with many duplicate proposals from
    # dominating supervision. Positive-profile background is explicitly sampled.
    for _ in range(4096):
        child=children[int(rng.integers(len(children)))]
        ni=int(rng.choice(child['profiles']))
        events=by_profile[str(ni)]
        event=next(e for e in events if e['parent']==child['parent'] and e['child']==child['child'])
        mode=int(rng.integers(3))
        if mode==0: q=rng.uniform(event['lo'],event['hi'])
        elif mode==1: q=float(rng.choice([event['lo'],event['hi']]))+rng.uniform(-8,8)
        else: q=rng.uniform(0,profiles[ni]['length'])
        examples.append((ni,float(np.clip(q,0,profiles[ni]['length']))))
    negative=np.flatnonzero(labels==0)
    for _ in range(4096):
        ni=int(rng.choice(negative)); examples.append((ni,float(rng.uniform(0,profiles[ni]['length']))))
    nodes=np.asarray([a for a,b in examples],np.int64); q=np.asarray([b for a,b in examples])
    targets=np.zeros((len(q),3),np.float32); offsets=np.zeros((len(q),2),np.float32); mask=np.zeros_like(offsets)
    for ni in np.unique(nodes):
        events=by_profile.get(str(int(ni)),[])
        if not events: continue
        ids=np.flatnonzero(nodes==ni); values=q[ids]
        for e in events: targets[ids,0]=np.maximum(targets[ids,0],(values>=e['lo'])&(values<=e['hi']))
        for head,key in enumerate(('lo','hi')):
            delta=np.asarray([e[key] for e in events])[None]-values[:,None]
            nearest=delta[np.arange(len(values)),np.abs(delta).argmin(1)]
            targets[ids,head+1]=np.exp(-.5*(nearest/2)**2)
            offsets[ids,head]=nearest/2; mask[ids,head]=np.abs(nearest)<=4
    patch,query=point_queries(profiles,nodes,q,xyz,point,cKDTree(xyz))
    return dict(nodes=nodes,q_mm=q,patch=patch,query=query,targets=targets,
        offsets=offsets,offset_mask=mask)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); args=ap.parse_args()
    dest=OUT/args.run; dest.mkdir(exist_ok=False); inventory=read(OUT/'inventory.json'); frozen=[]
    write(dest/'registered_protocol.json',dict(stage='PTV3_JOINT_GRAPH_AND_EVENT_DATA',
        source_sha256=digest(__file__),train=TRAIN,development=VALID,arc_diagnostic=ARC,
        original_G=496,face_features=46,pair_geometry_features=16,point_patch_features=23,
        PTv3_weights_frozen_from='run26',proposals='run34',STEP_reads=0,
        no_source_identity_as_features=True,labels_after_all_input_freezes=True,
        material_unknown=True,curves_pending=True,full_contract_pass=False,start_utc=utc()))
    try:
        for si in TRAIN+VALID+ARC:
            check_resources(); xyz,point,faces,profiles=load_inputs(si,inventory)
            data=graph_inputs(xyz,point,faces,profiles)
            path=dest/f'source_{si:02d}_inputs.npz'; np.savez_compressed(path,**data)
            frozen.append(dict(source=si,file=path.name,sha256=digest(path),
                faces=len(faces),profiles=len(profiles),raw_points=len(xyz)))
            print('PTV3_JOINT_INPUT_FROZEN',si,len(faces),len(profiles),flush=True)
            del xyz,point,faces,profiles,data;gc.collect()
        write(dest/'all_inputs_frozen.json',dict(sources=frozen,all_before_any_GT_read=True))
        rows=[]
        for si in TRAIN+VALID+ARC:
            check_resources(); profiles=read(PROPOSALS/f'source_{si:02d}/profiles.json')
            target=ROOT/inventory[si]['target']['target_path']
            assert digest(target)==inventory[si]['target']['target_sha256']
            labels,truth=supervision(profiles,read(target))
            np.save(dest/f'source_{si:02d}_labels.npy',labels)
            write(dest/f'source_{si:02d}_supervision.json',truth)
            queries=0
            if si in TRAIN:
                xyz,point,faces,_=load_inputs(si,inventory)
                data=training_queries(si,profiles,labels,truth,xyz,point)
                np.savez_compressed(dest/f'source_{si:02d}_queries.npz',**data);queries=len(data['nodes'])
                del xyz,point,faces,data
            row=dict(source=si,original_G=truth['original_G'],covered_children=truth['covered_children'],
                profiles=len(profiles),compatible_profiles=int(labels.sum()),training_queries=queries)
            rows.append(row);print('JOINT_SUPERVISION_READY',row,flush=True)
            del profiles,truth,labels;gc.collect()
        write(dest/'receipt.json',dict(status='PTV3_JOINT_DATA_COMPLETE',rows=rows,
            end_utc=utc(),optimizer_updates=0,full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__': main()
