"""Freeze a development split and cache point-only features before references."""
import argparse,json,time,shutil
from pathlib import Path
from training.dataio import read_xyz_pcd
from training.labels import sha256
from training.scripts.train_pilot import atomic_save
from weld_pipeline.typed_sequences import make_sequences,training_targets,FEATURE_VERSION
from weld_pipeline.typed_contact_model import decode_typed
from weld_pipeline.truth import load_truth
from weld_pipeline.evaluation import score_welds

SPLIT={
    'train':['161-LB18A','203-TT1A','206-TT1A','204-BS1A','225-BS1A','224-GR24A1','207-FR79M','207-FR80A'],
    'validation':['222-TT1A','527-FR146A','527-LB24A1','234-FR71A'],
    'reserved':['232-TT1A','224-FR71A']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outdir',default='weld_pipeline/data/typed_pilot_v2')
    ap.add_argument('--groups',nargs='+');a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    config={'status':'FROZEN_DEVELOPMENT_SPLIT','feature_version':FEATURE_VERSION,**SPLIT,
        'scope':'12 development assemblies; two reserved for this model. Earlier SpUNet test data were already exposed; not a new blind acceptance set.',
        'type_coverage':'GR24A1 verticals train, FR146A verticals validation. No arc training in this straight-line experiment.',
        'family_limit':'TT and FR groups share structural families; these splits do not establish new-family generalization.'}
    split_path=out/'split.json'
    if split_path.exists():
        if json.loads(split_path.read_text(encoding='utf-8'))!=config:raise ValueError('Frozen split differs')
    else:split_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    names=a.groups or SPLIT['train']+SPLIT['validation']
    sources=[Path('weld_pipeline')/n for n in ['typed_sequences.py','typed_contact_model.py','surface_planes_local.py','surface_planes.py','geometry_candidates.py','truth.py','evaluation.py','prepare_typed_pilot.py']]
    hashes={str(p):sha256(p) for p in sources}
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in sources:
        target=snapshot/p.name
        if target.exists() and sha256(target)!=sha256(p):raise ValueError('Preprocessing source changed; use a new dataset directory')
        if not target.exists():shutil.copy2(p,target)
    for name in names:
        dest=out/(name+'.pt')
        if dest.exists():print('EXISTING',name,flush=True);continue
        folder=Path('DataSet/9月20日')/name;pcd=next(folder.glob('*AllFaces.pcd'));start=time.perf_counter()
        xyz=read_xyz_pcd(pcd)*1000;sequences,geometry=make_sequences(xyz)
        feature_seconds=time.perf_counter()-start
        # No reference is opened before this point.
        truth,missing=load_truth(folder);targets,mapping=training_targets(sequences,truth)
        oracle=score_welds(decode_typed(sequences,targets),truth)
        record={'name':name,'input':str(pcd),'pcd_sha256':sha256(pcd),'source_hashes':hashes,
            'feature_version':FEATURE_VERSION,'feature_channels':sequences[0]['features'].shape[1],
            'features_before_truth_read':True,'feature_seconds':feature_seconds,'geometry':geometry,
            'mapping':mapping,'missing_source_curves':missing,'representation_oracle':oracle,
            'reference_hashes':{str(p):sha256(p) for p in [*folder.glob('*.xlsx'),*folder.glob('*welds.json')]}}
        atomic_save({'sequences':sequences,'targets':targets,'truth':truth,'record':record},dest)
        (out/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print('PREPARED',name,'seconds',round(feature_seconds,2),'seq',len(sequences),'points',geometry['feature_points'],
              'mapped',mapping['mapped_to_sequences'],'/',len(truth),'oracle_valid',oracle['valid_welds'],'fp',oracle['false_or_duplicate_predictions'],flush=True)


if __name__=='__main__':main()
