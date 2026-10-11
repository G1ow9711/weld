"""Freeze 8-case pair inputs, then add known-owner labels on TRAIN only."""
import argparse
import gc
import numpy as np
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_material_atom_audit_873 import support_matrix
from weld_pipeline.sjg_relation_features_880 import pair_features,FEATURE_NAMES


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    base=OUT/'run75/dense_control';relations={r['source']:r for r in read(base/'all_predictions_frozen.json')['sources']}
    ambiguities={r['source']:r for r in read(OUT/'run80/all_predictions_frozen.json')['sources']}
    raws={r['source']:r for r in read(OUT/'run22/all_inputs_frozen.json')['sources']};frozen=[]
    write(dest/'registered_protocol.json',dict(stage='XYZ_FINITE_SUPPORT_PAIR_FEATURES',sources=TRAIN+VALID+ARC,features=FEATURE_NAMES,
        all8_inputs_before_reference=True,cached_XYZ_geometry_not_final_deployment_timing=True,
        unknown_reference_not_negative=True,source_sha256=digest(__file__),full_contract_G=496,full_contract_pass=False,start_utc=utc()))
    for source in TRAIN+VALID+ARC:
        check_resources();print('RELATION_FEATURE_START',source,flush=True);r=relations[source];a=ambiguities[source];b=raws[source]
        assert digest(base/r['file'])==r['sha256'] and digest(OUT/'run80'/a['file'])==a['sha256']
        assert digest(OUT/'run22'/b['xyz_file'])==b['xyz_sha256'] and r['raw_sha256']==b['raw_sha256']==a['raw_sha256']
        geometry_binding=read(OUT/f'run34/source_{source:02d}/input_freeze.json');assert geometry_binding['raw_sha256']==b['raw_sha256']
        xyz=np.load(OUT/'run22'/b['xyz_file']);support=support_matrix(source,len(xyz))
        with np.load(base/r['file']) as p,np.load(OUT/'run80'/a['file']) as q:
            x,flat=pair_features(xyz,p['neighbors'],p['material_distance'],p['surface_distance'],p['normal'],p['boundary'],
                q['material_ambiguity_probability'],support,p['core_block'])
        assert len(np.unique(flat))==len(flat) and np.all((flat>=0)&(flat<len(xyz)*8))
        xf=dest/f'source_{source:02d}_features.npy';ix=dest/f'source_{source:02d}_flat.npy';np.save(xf,x);np.save(ix,flat)
        rec=dict(source=source,points=len(xyz),observed_pairs=len(flat),features_file=xf.name,features_sha256=digest(xf),
            flat_file=ix.name,flat_sha256=digest(ix),raw_sha256=b['raw_sha256'],relation_sha256=r['sha256'],ambiguity_sha256=a['sha256'],
            geometry_binding_sha256=digest(OUT/f'run34/source_{source:02d}/input_freeze.json'))
        frozen.append(rec);print('RELATION_FEATURE_FROZEN',source,len(flat),'FEATURES',x.shape[1],flush=True);del xyz,support,x,flat;gc.collect()
    write(dest/'all_inputs_frozen.json',dict(sources=frozen,all_before_reference_reads=True))
    teachers={r['source']:r for r in read(OUT/'run22/label_summary.json')};targets=[]
    for r in frozen:
        source=r['source']
        if source not in TRAIN:continue
        t=OUT/'run22'/teachers[source]['teacher_file'];assert digest(t)==teachers[source]['teacher_sha256']
        with np.load(t) as z:owner=z['reference_volume']
        flat=np.load(dest/r['flat_file']);a=flat//8
        with np.load(base/relations[source]['file']) as z:b=z['neighbors'].ravel()[flat]
        known=(owner[a]>=0)&(owner[b]>=0);y=np.full(len(flat),-1,np.int8);y[known]=(owner[a[known]]!=owner[b[known]]).astype(np.int8)
        file=dest/f'source_{source:02d}_target.npy';np.save(file,y)
        target=dict(source=source,file=file.name,sha256=digest(file),same=int((y==0).sum()),different=int((y==1).sum()),unknown=int((y<0).sum()))
        targets.append(target);print('RELATION_TRAIN_TARGET',target,flush=True)
    write(dest/'training_targets.json',dict(rows=targets))
    write(dest/'receipt.json',dict(status='PAIR_RELATION_INPUTS_AND_TRAIN_TARGETS_READY',all_points=sum(r['points'] for r in frozen),
        known_pairs=sum(r['same']+r['different'] for r in targets),different=sum(r['different'] for r in targets),same=sum(r['same'] for r in targets),
        unknown_train_pairs=sum(r['unknown'] for r in targets),full_contract_G=496,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
