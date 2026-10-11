"""Preserve every old sample; add verified false event-support examples."""
import gc
import json
import shutil
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.bounded_event_phase import VERSION,FEATURE_COUNT,CELL_MM,PERIODS_MM,bounded_features,cell_residuals,cell_offsets
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.train_forest_event_centers import load_cached

RUN=Path('weld_pipeline/runs/deep_physical_events_170_v3')
RAW=Path('weld_pipeline/reports/deep_physical_head_diagnosis_173_20261002')
OLD=Path('weld_pipeline/data/forest_event_centers_pilot166')
OUT=Path('weld_pipeline/data/hard_physical_event_pilot175')
QUOTA_PER_SOURCE_ROLE=8000


def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf8')


def diverse_top(groups,quota):
    if not groups:return np.empty(0,np.int64)
    first=[];per_group=max(1,quota//len(groups))
    for rows,scores in groups:
        order=np.lexsort((rows,-scores));first.append(rows[order[:per_group]])
    selected=np.unique(np.concatenate(first));rows=np.concatenate([g[0] for g in groups]);scores=np.concatenate([g[1] for g in groups])
    order=np.lexsort((rows,-scores))
    if len(selected)>quota:return rows[order[:quota]]
    remaining=rows[order[~np.isin(rows[order],selected)]]
    return np.unique(np.r_[selected,remaining[:quota-len(selected)]])


def main():
    if OUT.exists():raise ValueError('Preserve the hard-event training-data version')
    config=json.loads((RUN/'config.json').read_text('utf8'))
    diagnosis=json.loads((RAW/'complete.json').read_text('utf8'))
    if not diagnosis['full_16_frozen_learned_outputs_exact']:raise ValueError('Complete frozen full-FIT head diagnosis required')
    OUT.mkdir(parents=True);snapshot=OUT/'source_snapshot';snapshot.mkdir()
    for path in [__file__,'weld_pipeline/bounded_event_phase.py']:shutil.copy2(path,snapshot/Path(path).name)
    protocol=dict(version=VERSION,sources=config['train'],feature_count=FEATURE_COUNT,
        cell_mm=CELL_MM,periods_mm=PERIODS_MM.tolist(),quota_per_source_support_role=QUOTA_PER_SOURCE_ROLE,
        keep_every_original_sample=True,preserve_all_original_known_event_positive_rows=True,
        mined_negative_requires_original_known_label_and_actual_vote_eligibility=True,
        candidate_diversity_then_high_score_order=True,large_error_positive_sample_weight_factor_next_training=2.,
        data_is_training_development_only_not_runtime_reference=True,
        source_sha256={p:sha256(p) for p in [__file__,'weld_pipeline/bounded_event_phase.py']},
        full_contract_pass=False)
    dump(OUT/'config.json',protocol);manifests={};total_old=0;total_new=0;maximum_inverse_error=0.
    for name in config['train']:
        data,near,known,piece=load_cached(name)
        old=OLD/name;old_manifest=json.loads((old/'manifest.json').read_text('utf8'))
        for file,digest in old_manifest['files'].items():
            if sha256(old/file)!=digest:raise ValueError('Original sampled data changed')
        old_indices=np.load(old/'sample_indices.npy',allow_pickle=False)
        if sha256(RAW/(name+'.json'))!=diagnosis['piece_receipts'][name]:raise ValueError('Frozen raw diagnosis changed')
        raw_path=RAW/(name+'_raw_heads.npz')
        with np.load(raw_path,allow_pickle=False) as archive:raw=archive['raw'];lengths=archive['lengths']
        if not np.array_equal(lengths,[len(s['xyz']) for s in piece['sequences']]):raise ValueError('Full raw event rows differ')
        starts=np.r_[0,np.cumsum(lengths)];groups={role:[] for role in range(3)};positive=[];hard_positive=[]
        for e in data['entries']:
            if e['type']!='horizontal':continue
            a,b=e['start'],e['stop'];pred=raw[starts[e['sequence']]:starts[e['sequence']+1]]
            yy=np.asarray(near[a:b]);kk=np.asarray(known[a:b])
            positive.extend((a+np.flatnonzero(np.any((yy[:,1:4]>.5)&(kk[:,1:4]>0),axis=1))).tolist())
            hard_positive.extend((a+np.flatnonzero(np.any((kk[:,4:7]>0)&(np.abs(pred[:,4:7]-yy[:,4:7])*16.>3.),axis=1))).tolist())
            for role in range(3):
                mask=(kk[:,role+1]>0)&(yy[:,role+1]<.5)&(pred[:,role+1]>=0.)&(np.abs(pred[:,role+4])<=1.+1e-6)
                ids=np.flatnonzero(mask)
                if len(ids):groups[role].append((a+ids,pred[ids,role+1].astype(float)))
        positive=np.asarray(positive,np.int64)
        if not np.isin(positive,old_indices).all():raise ValueError('All original event positives must already be preserved')
        mined={role:diverse_top(groups[role],QUOTA_PER_SOURCE_ROLE) for role in range(3)}
        indices=np.unique(np.concatenate([old_indices,*mined.values()])).astype(np.int64)
        if not np.isin(old_indices,indices).all():raise ValueError('An original training row was removed')
        x=np.zeros((len(indices),FEATURE_COUNT),np.float32);assigned=np.zeros(len(indices),bool)
        target=np.asarray(near[indices],np.float32).copy();k=np.asarray(known[indices],np.float32).copy()
        original_offsets=target[:,4:7].copy();max_error=0.
        for e in data['entries']:
            if e['type']!='horizontal':continue
            a,b=e['start'],e['stop'];lo,hi=np.searchsorted(indices,[a,b]);local=indices[lo:hi]-a
            seq,_,transform=canonical_frame(piece['sequences'][e['sequence']])
            if transform!=e['transform']:raise ValueError('Canonical hard-event query order differs')
            x[lo:hi]=bounded_features(seq,data,e,local);assigned[lo:hi]=True
            positions=local.astype(float)*float(seq['sample_step_mm'])
            residuals=cell_residuals(target[lo:hi,4:7],positions).astype(np.float32)
            recovered=cell_offsets(residuals,positions);valid=k[lo:hi,4:7]>0
            if valid.any():max_error=max(max_error,float(np.abs(recovered-original_offsets[lo:hi])[valid].max()*16.))
            target[lo:hi,4:7]=np.where(valid,residuals,0.)
        if not assigned.all() or max_error>1e-5:raise ValueError('Physical cell transformation lost a known label')
        valid=k[:,4:7]>0
        if np.any((target[:,4:7][valid]<-1.-1e-5)|(target[:,4:7][valid]>5.+1e-5)):
            raise ValueError('Known 16 mm events must have bounded 64 mm cell residuals')
        hard_positive=np.unique(np.asarray(hard_positive,np.int64));priority=np.where(np.isin(indices,hard_positive),2.,1.).astype(np.float32)
        folder=OUT/name;folder.mkdir()
        for file,array in [('features.npy',x),('targets.npy',target),('known.npy',k),('sample_indices.npy',indices),('positive_priority.npy',priority)]:np.save(folder/file,array)
        receipt=dict(original_samples=len(old_indices),samples=len(indices),added_unique_samples=len(indices)-len(old_indices),
            mined_per_role={str(role):len(rows) for role,rows in mined.items()},all_original_samples_and_event_positives_preserved=True,
            hard_position_positive_rows=len(hard_positive),max_inverse_error_mm=max_error,
            raw_head_sha256=sha256(raw_path),original_sample_manifest_sha256=sha256(old/'manifest.json'),
            files={file:sha256(folder/file) for file in ['features.npy','targets.npy','known.npy','sample_indices.npy','positive_priority.npy']},
            original_labels_unchanged=True,full_contract_pass=False)
        dump(folder/'manifest.json',receipt);manifests[name]=sha256(folder/'manifest.json')
        total_old+=len(old_indices);total_new+=len(indices);maximum_inverse_error=max(maximum_inverse_error,max_error)
        print('HARD_PHYSICAL_PREPARED',name,len(old_indices),'TO',len(indices),'ADDED',len(indices)-len(old_indices),'INVERSE_MM',max_error,flush=True)
        del data,near,known,piece,raw,x,target,k;gc.collect()
    dump(OUT/'complete.json',dict(config_sha256=sha256(OUT/'config.json'),manifests=manifests,
        original_samples=total_old,samples=total_new,added_samples=total_new-total_old,max_inverse_error_mm=maximum_inverse_error,
        all_original_samples_and_event_positives_preserved=True,actual_trained_models=0,
        new_representation_oracle_still_required_before_training=True,full_contract_pass=False))
    print('HARD_PHYSICAL_DATA_COMPLETE',total_old,'TO',total_new,'INVERSE_MM',maximum_inverse_error,flush=True)


if __name__=='__main__':main()
