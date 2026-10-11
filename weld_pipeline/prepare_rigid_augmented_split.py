"""Index original TRAIN assemblies and audited pose variants without split leakage."""
import json,shutil
from pathlib import Path
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece,write_piece


def main():
    original=Path('weld_pipeline/data/mixed_surface_v1')
    out=Path('weld_pipeline/data/rigid_augmented_surface_v1');out.mkdir(parents=True,exist_ok=True)
    if (out/'split.json').exists():raise ValueError('Dataset split already frozen')
    split=json.loads((original/'split.json').read_text(encoding='utf-8'))
    if json.loads((original/'preprocessing_complete.json').read_text())['status']!='READY':raise ValueError('Original dataset not ready')
    parents=['223-FR83A','226-FR137F','227-BK148A'];original_train=list(split['train'])
    roots={name:str(Path('../mixed_surface_v1')/name) for name in split['train']+split['validation']}
    variants=[];sources={}
    for parent in parents:
        if parent not in original_train or parent in split['validation']:raise ValueError('Augmentation parent leaks validation')
        folder=Path('weld_pipeline/reports/rigid_vertical_augmentation')/parent
        receipt=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
        if receipt['status']!='REPRESENTATION_READY':raise ValueError('Augmentation not ready')
        if sha256(original/parent/'manifest.json')!=receipt['source_manifest_sha256']:raise ValueError('Original manifest changed')
        for name,h in receipt['source_hashes'].items():
            if sha256(name)!=h:raise ValueError('Audit code changed before freezing augmentation')
        piece=load_piece(folder/'candidate_cache');name=piece['record']['name']
        if not all(w['type']=='vertical' and len(w['segments'])==1 for w in piece['truth']):raise ValueError('Unexpected augmented labels')
        runtime_paths=['fast_mixed_sequences.py','bounded_sequences.py','robust_surface_planes.py','indexed_surface_features.py',
                       'mixed_sequences.py','curved_sequences.py','typed_sequences.py','arc_tangent_sequences.py',
                       'arc_surface_candidates.py','local_tangent_circles.py','tangent_circle_candidates.py','circle_candidates.py']
        feature_sources={str(Path('weld_pipeline')/p):sha256(Path('weld_pipeline')/p) for p in runtime_paths}
        provenance={'source_group':parent,'source_manifest_sha256':receipt['source_manifest_sha256'],
                    'augmentation_receipt_sha256':sha256(folder/'receipt.json'),'feature_source_hashes':feature_sources,
                    'rotation':receipt['rotation'],'translation_mm':receipt['translation_mm'],
                    'valid_supervision':'Geometric foreground, endpoints, preserved internal segmentation, and false source split flags only.',
                    'scope':'TRAIN pose augmentation; same parent assembly; never counted as an independent workpiece.'}
        write_piece(out/name,piece,provenance);roots[name]=name;split['train'].append(name)
        variants.append({'name':name,'source_group':parent,'welds':len(piece['truth']),'manifest_sha256':sha256(out/name/'manifest.json')})
        sources.update(feature_sources)
        dest=out/'augmentation_evidence'/parent;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(folder/'receipt.json',dest/'receipt.json')
        for source in receipt['source_hashes']:shutil.copy2(source,dest/Path(source).name)
    split['piece_roots']=roots
    split['augmentation_policy']={'independent_training_groups':len(original_train),'original_train':original_train,'variants':variants,
        'validation_unchanged':True,'no_augmented_variant_in_validation':True,'source_split_sha256':sha256(original/'split.json')}
    (out/'split.json').write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf-8')
    readiness={'status':'READY','training_entries':len(split['train']),'independent_training_groups':len(original_train),
        'augmentation_entries':len(variants),'augmented_single_vertical_welds':sum(v['welds'] for v in variants),
        'validation_groups':split['validation'],'original_readiness_sha256':sha256(original/'preprocessing_complete.json'),
        'split_sha256':sha256(out/'split.json'),'source_hashes':{str(Path(__file__)):sha256(Path(__file__)),**sources},
        'scope':'Representation readiness only, not model accuracy or additional independent customers.'}
    (out/'preprocessing_complete.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:readiness[k] for k in ['status','training_entries','independent_training_groups','augmentation_entries','augmented_single_vertical_welds']}),flush=True)


if __name__=='__main__':main()
