"""Freeze source/augmentation groups and detect geometric near-duplicate leakage."""
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.contract_child_evaluation import child_units
from weld_pipeline.point_frame import dominant_xy_frame


def main():
    started = time.perf_counter()
    out = Path('weld_pipeline/reports/source_family_audit_20261001')
    if out.exists():
        raise ValueError('Frozen grouping diagnostic exists')
    out.mkdir(parents=True)
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    real = [name for name in split['train'] if '__' not in name]
    augment = [name for name in split['train'] if '__' in name]
    config = dict(role='PREPARATION_FOR_FAMILY_CROSS_FIT_NOT_ACCEPTANCE', real_FIT=real,
        augmentations=augment, observed_check_excluded=split['validation'],
        split_sha256=sha256(split_path), source_sha256=sha256(__file__),
        grouping_policy='Source hash identity; additionally bidirectional raw-point-cloud geometric proximity after gravity-frame translation/XY symmetries.',
        near_duplicate=dict(sample_points=4096, bidirectional_q95_max_mm=3., q99_max_mm=10.,
                            symmetric_gravity_frame=True, used_weld_labels_for_similarity=False),
        folds=3, fold_seed=172, runtime_model_input=False, full_contract_pass=False)
    (out/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    shutil.copy2(__file__, out/Path(__file__).name)
    geometry, records, parents = {}, {}, {}
    hashes = {}
    for name in real:
        root = Path(split['piece_roots'][name])
        manifest = json.loads((root/'manifest.json').read_text('utf8'))
        record = manifest['record']; pcd = Path(record['input'])
        actual_hash = sha256(pcd)
        if actual_hash != record['pcd_sha256']:
            raise ValueError('Original point cloud changed: '+name)
        for path, digest in record.get('reference_hashes', {}).items():
            if sha256(path) != digest:
                raise ValueError('Original reference changed: '+path)
        hashes.setdefault(actual_hash, []).append(name)
        xyz = read_xyz_pcd(pcd)*1000.
        frame = dominant_xy_frame(xyz)
        canonical = np.column_stack((xyz[:,:2]@frame.T, xyz[:,2]))
        canonical -= (canonical.min(0)+canonical.max(0))/2.
        rng = np.random.default_rng(172)
        ids = rng.choice(len(canonical), min(4096,len(canonical)), replace=False)
        geometry[name] = dict(points=canonical, tree=cKDTree(canonical), samples=canonical[ids],
                              extents=np.ptp(canonical, axis=0))
        children = child_units(manifest['truth'])
        category_counts = {key:sum(child[field]==value for child in children) for key,field,value in
            [('horizontal','type','horizontal'),('vertical','type','vertical'),('line','kind','line'),('curve','kind','arc')]}
        records[name] = dict(input=str(pcd), input_sha256=actual_hash, points=len(xyz),
            children=len(children), categories=category_counts, reference_hashes=record.get('reference_hashes',{}),
            gravity_frame=frame.tolist(), normalized_extents_mm=geometry[name]['extents'].tolist(),
            cache_manifest_sha256=sha256(root/'manifest.json'))
        parents[name] = name
        print('FAMILY_SOURCE', name, len(xyz), len(children), flush=True)
    def find(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name
    def join(a,b):
        a,b = find(a),find(b)
        if a!=b:
            parents[max(a,b)] = min(a,b)
    for group in hashes.values():
        for name in group[1:]:
            join(group[0],name)
    pairs=[]
    transforms=[]
    for swap in (False,True):
        for signs in itertools.product((-1.,1.), repeat=2):
            matrix=np.eye(3)
            if swap:
                matrix=matrix[[1,0,2]]
            matrix[0]*=signs[0];matrix[1]*=signs[1]
            transforms.append(matrix)
    for a,b in itertools.combinations(real,2):
        ga,gb=geometry[a],geometry[b]
        if (abs(ga['extents'][2]-gb['extents'][2])>10. or
            np.max(abs(np.sort(ga['extents'][:2])-np.sort(gb['extents'][:2])))>10.):
            continue
        best=None
        for matrix in transforms:
            extent=np.abs(matrix)@gb['extents']
            if np.max(abs(ga['extents']-extent))>10.:
                continue
            dist_a=ga['tree'].query(gb['samples']@matrix.T,k=1,workers=1)[0]
            dist_b=gb['tree'].query(ga['samples']@matrix,k=1,workers=1)[0]
            q95=float(max(np.quantile(dist_a,.95),np.quantile(dist_b,.95)))
            q99=float(max(np.quantile(dist_a,.99),np.quantile(dist_b,.99)))
            candidate=dict(a=a,b=b,bidirectional_q95_mm=q95,bidirectional_q99_mm=q99,
                           transform_b_to_a=matrix.tolist())
            if best is None or (q95,q99)<(best['bidirectional_q95_mm'],best['bidirectional_q99_mm']):
                best=candidate
        if best is not None:
            best['near_duplicate_family']=best['bidirectional_q95_mm']<=3. and best['bidirectional_q99_mm']<=10.
            pairs.append(best)
            if best['near_duplicate_family']:
                join(a,b)
    groups={}
    for name in real:
        groups.setdefault(find(name),[]).append(name)
    augmentation_sources={}
    for name in augment:
        manifest=json.loads((Path(split['piece_roots'][name])/'manifest.json').read_text('utf8'))
        record=manifest['record']
        possible=hashes.get(record['pcd_sha256'],[])
        source=record.get('source_group')
        if source is not None and source not in possible:
            raise ValueError('Augmentation provenance source differs: '+name)
        if source is None:
            if len(possible)!=1:
                raise ValueError('Augmentation source identity not unique: '+name)
            source=possible[0]
        augmentation_sources[name]=source
    group_names=sorted(groups)
    rng=np.random.default_rng(172);order=rng.permutation(len(group_names))
    group_fold={group_names[int(index)]:position%3 for position,index in enumerate(order)}
    folds=[]
    for fold in range(3):
        held=[name for name in real if group_fold[find(name)]==fold]
        train=[name for name in real if name not in held]
        train_aug=[name for name,source in augmentation_sources.items() if source in train]
        held_aug=[name for name,source in augmentation_sources.items() if source in held]
        counts={role:{cat:sum(records[name]['categories'][cat] for name in names) for cat in ['horizontal','vertical','line','curve']}
                for role,names in [('train',train),('held_out',held)]}
        if any(augmentation_sources[name] in held for name in train_aug):
            raise ValueError('Source-family augmentation leakage')
        folds.append(dict(fold=fold,train_real=train,train_augmentations=train_aug,
                          held_out_real=held,held_out_augmentations_excluded=held_aug,
                          category_counts=counts,
                          unavailable_training_categories=[cat for cat in counts['held_out']
                              if counts['held_out'][cat]>0 and counts['train'][cat]==0]))
    result=dict(config=config,sources=records,source_family_groups=groups,
        augmentation_sources=augmentation_sources,comparable_geometry_pairs=pairs,folds=folds,
        curve_positive_source_families=len({find(name) for name in real if records[name]['categories']['curve']}),
        held_out_predictions_not_yet_trained=True,elapsed_seconds=time.perf_counter()-started,
        source_data_preserved=True,full_contract_pass=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('FAMILY_AUDIT_COMPLETE','SOURCES',len(real),'AUG',len(augment),'FAMILIES',len(groups),
          'CURVE_FAMILIES',result['curve_positive_source_families'],flush=True)


if __name__=='__main__':
    main()
