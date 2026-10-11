"""Prepare original and audited rigid-augmentation axial context, without relabeling."""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
import numpy as np
from training.labels import sha256
from training.dataio import read_xyz_pcd
from weld_pipeline.curved_cache import load_piece as load_base
from weld_pipeline.prepare_wide_context_dataset import augmentation_transform
from weld_pipeline.axial_context_features import append_axial_context
from weld_pipeline.axial_context_cache import write_piece,load_piece,STORAGE_FORMAT,FEATURE_VERSION


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args()
    root,out=Path(a.parent),Path(a.outdir);split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    files=[Path(__file__),*[Path('weld_pipeline')/n for n in ['axial_context_features.py','axial_context_cache.py','prepare_wide_context_dataset.py','indexed_surface_features.py','curved_cache.py','mapped_cache.py']]]
    sources={str(p):sha256(p) for p in files};spec={'parent_split_sha256':sha256(root/'split.json'),'sources':sources,'feature_version':FEATURE_VERSION}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'build_spec.json').exists() and json.loads((out/'build_spec.json').read_text(encoding='utf-8'))!=spec:raise ValueError('Existing build specification differs')
    (out/'build_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    variants={v['name']:v for v in split['augmentation_policy']['variants']};rows=[]
    for name in split['train']+split['validation']:
        started=time.perf_counter();folder=(root/split.get('piece_roots',{}).get(name,name)).resolve()
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));variant=variants.get(name)
        if variant:
            if name not in split['train'] or variant['source_group'] not in split['augmentation_policy']['original_train']:raise ValueError('Non-TRAIN augmentation')
            if sha256(folder/'manifest.json')!=variant['manifest_sha256']:raise ValueError('Variant manifest changed')
        rotation,translation,audit=augmentation_transform(name,variant,manifest)
        source=Path(manifest['record']['input'])
        if sha256(source)!=manifest['record']['pcd_sha256']:raise ValueError('Raw input changed')
        provenance={'sources':sources,'parent_manifest_sha256':sha256(folder/'manifest.json'),'raw_input':str(source),
                    'raw_input_sha256':sha256(source),'rotation':rotation.tolist(),'translation_mm':translation.tolist(),'augmentation_audit':audit}
        dest=out/name
        if (dest/'manifest.json').exists():
            result=json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
            if result['provenance']!=provenance:raise ValueError('Prepared provenance changed')
            piece=load_piece(dest)
        else:
            base=load_base(folder);xyz=(read_xyz_pcd(source)*1000.)@rotation.T+translation
            sequences,_=append_axial_context(xyz,base['sequences'])
            result=write_piece(dest,folder,sequences,provenance);piece=load_piece(dest)
            if piece['truth']!=base['truth'] or any(not np.array_equal(x,y) or y.flags.writeable for x,y in zip(base['targets'],piece['targets'])):
                raise ValueError('Original truth or unknown supervision changed')
            del base,xyz,sequences
        row={'group':name,'vertical_sequences':len(result['vertical_indices']),'context_bytes':(dest/'axial_context.npy').stat().st_size,
             'manifest_sha256':sha256(dest/'manifest.json'),'seconds':time.perf_counter()-started};rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True);del piece
    new=copy.deepcopy(split);new.update(feature_version=FEATURE_VERSION,storage_format=STORAGE_FORMAT,piece_roots={r['group']:r['group'] for r in rows},parent_split_sha256=spec['parent_split_sha256'])
    for v in new['augmentation_policy']['variants']:v['manifest_sha256']=sha256(out/v['name']/'manifest.json')
    (out/'split.json').write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
    ready={'status':'READY_PARTIAL','required_trainer_capability':'partial_supervision_v1','required_feature_loader':STORAGE_FORMAT,
           'split_sha256':sha256(out/'split.json'),'source_hashes':sources,'pieces':rows,
           'scope':'Context preparation only. Original coordinates, labels and candidate output extents unchanged.'}
    (out/'preprocessing_complete.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2),encoding='utf-8')
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in files:shutil.copy2(p,snapshot/p.name)
    print('READY',len(rows),'BYTES',sum(r['context_bytes'] for r in rows),flush=True)


if __name__=='__main__':main()
