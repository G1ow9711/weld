"""Freeze partial TRAIN augmentation; old trainers reject READY_PARTIAL."""
import json,os,shutil
from pathlib import Path
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece,write_piece
from weld_pipeline.partial_supervision import clean_targets


def main():
    previous=Path('weld_pipeline/data/rigid_augmented_surface_v1')
    out=Path('weld_pipeline/data/partial_augmented_surface_v1');out.mkdir(parents=True,exist_ok=True)
    if (out/'split.json').exists():raise ValueError('Frozen split exists')
    split=json.loads((previous/'split.json').read_text(encoding='utf-8'))
    if json.loads((previous/'preprocessing_complete.json').read_text())['status']!='READY':raise ValueError('Previous representations not ready')
    folder=Path('weld_pipeline/reports/partial_vertical_augmentation/EDH2P-LB18E-0918-李欣_f102')
    audit=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
    verification=json.loads((folder/'cache_verification.json').read_text())
    if audit['status']!='MASKED_REPRESENTATION_READY' or verification['status']!='PASS':raise ValueError('Partial augmentation not verified')
    if sha256(folder/'receipt.json')!=verification['audit_sha256'] or sha256(folder/'candidate_cache/manifest.json')!=verification['manifest_sha256']:
        raise ValueError('Partial evidence changed')
    parent=audit['source_group']
    if parent not in split['augmentation_policy']['original_train'] or parent in split['validation']:raise ValueError('TRAIN parent required')
    parent_folder=(previous/split['piece_roots'][parent]).resolve()
    if sha256(parent_folder/'manifest.json')!=audit['source_manifest_sha256']:raise ValueError('Parent labels changed')
    for source,h in audit['source_hashes'].items():
        if sha256(source)!=h:raise ValueError('Partial audit source changed')
    piece=load_piece(folder/'candidate_cache');name=piece['record']['name'];known=unknown=0
    for target in piece['targets']:
        _,valid=clean_targets(target);known+=int(valid.sum());unknown+=int((valid==0).sum())
    if known!=audit['supervised_points'] or unknown!=audit['unknown_points']:raise ValueError('Supervision count differs')
    for key,path in list(split['piece_roots'].items()):split['piece_roots'][key]=os.path.relpath((previous/path).resolve(),out.resolve())
    provenance={'source_group':parent,'source_manifest_sha256':audit['source_manifest_sha256'],
        'audit_sha256':sha256(folder/'receipt.json'),'target_policy':'all_channel_minus_one_is_unknown',
        'required_trainer_capability':'partial_supervision_v1','known_points':known,'unknown_points':unknown,
        'scope':'TRAIN augmentation, not an independent piece; unknown context is not negative supervision.'}
    write_piece(out/name,piece,provenance);split['piece_roots'][name]=name;split['train'].append(name)
    split['augmentation_policy']['variants'].append({'name':name,'source_group':parent,'welds':len(piece['truth']),
        'partial_supervision':True,'known_points':known,'unknown_points':unknown,'manifest_sha256':sha256(out/name/'manifest.json')})
    split['augmentation_policy']['prior_split_sha256']=sha256(previous/'split.json')
    split['required_trainer_capability']='partial_supervision_v1'
    (out/'split.json').write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf-8')
    readiness={'status':'READY_PARTIAL','required_trainer_capability':'partial_supervision_v1','training_entries':len(split['train']),
        'independent_training_groups':len(split['augmentation_policy']['original_train']),
        'augmentation_entries':len(split['augmentation_policy']['variants']),'validation_groups':split['validation'],
        'partial_supervision':{name:{'known_points':known,'unknown_points':unknown}},'split_sha256':sha256(out/'split.json'),
        'source_sha256':sha256(Path(__file__)),'scope':'Representation readiness only; existing unmasked trainers must reject this dataset.'}
    (out/'preprocessing_complete.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding='utf-8')
    dest=out/'partial_evidence';dest.mkdir(exist_ok=True)
    for filename in ['receipt.json','cache_verification.json']:shutil.copy2(folder/filename,dest/filename)
    for source in audit['source_hashes']:shutil.copy2(source,dest/Path(source).name)
    print(json.dumps({k:readiness[k] for k in ['status','training_entries','independent_training_groups','augmentation_entries','partial_supervision']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
