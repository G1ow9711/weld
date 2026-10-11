"""Exact three-column endpoint update without copying unchanged geometry arrays.

The feature identity is explicitly composite, not the hash of a nonexistent
features.npy. Original candidate positions and 177 geometric columns stay frozen.
"""
import gc
import hashlib
import json
import shutil
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources

COLUMNS=(168,169,170)
KEYS=('21','BM79A_native_local','10','25','36','64')
IDENTITY_KIND='SHA256_COMPOSITE_BASE_PLUS_EXACT_COLUMN_OVERLAY_V1'


def identity(spec):
    keys=('base_sha256','override_sha256','columns','shape','dtype')
    value=json.dumps({k:spec[k] for k in keys},sort_keys=True,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(value).hexdigest()


class OverlayRows:
    def __init__(self,base,override):
        self.base=base;self.override=override;self.shape=base.shape;self.dtype=base.dtype;self.nbytes=base.nbytes
        assert self.shape[1]==180 and override.shape==(len(base),3) and override.dtype==self.dtype
        self._mmap=self

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        value=np.array(self.base[index],copy=True)
        value[...,list(COLUMNS)]=self.override[index]
        return value

    def close(self):
        for x in (self.base,self.override):
            mmap=getattr(x,'_mmap',None)
            if mmap is not None and not mmap.closed:mmap.close()


def load_overlay(folder,receipt):
    spec=read(folder/'feature_overlay.json')
    assert spec['columns']==list(COLUMNS)
    assert receipt['feature_identity_kind']==IDENTITY_KIND
    assert digest(folder/'feature_overlay.json')==receipt['overlay_spec_sha256']
    assert identity(spec)==receipt['feature_sha256']
    base=OUT/spec['base_file'];override=folder/spec['override_file']
    assert digest(base)==spec['base_sha256'] and digest(override)==spec['override_sha256']
    result=OverlayRows(np.load(base,mmap_mode='r'),np.load(override,mmap_mode='r'))
    assert list(result.shape)==spec['shape'] and str(result.dtype)==spec['dtype']
    return result


def controls():
    rng=np.random.default_rng(1272);base=rng.normal(size=(1031,180)).astype(np.float32)
    override=rng.random((1031,3)).astype(np.float32);full=base.copy();full[:,list(COLUMNS)]=override
    view=OverlayRows(base,override)
    for index in (0,-1,slice(0,1),slice(511,1025),slice(2,2)):
        assert np.array_equal(view[index],full[index])
    changed=view[0];changed[:]=0.;assert np.array_equal(view[0],full[0])
    return dict(status='PASS',exact_integer_and_slice_overlay=True,original_arrays_not_mutated=True,
                unchanged_columns=177,replaced_columns=list(COLUMNS),lossy_operations=0)


def prepare():
    check_resources();dest=OUT/'run438';dest.mkdir(exist_ok=False)
    snapshot=dest/'source_snapshot';snapshot.mkdir();shutil.copy2(__file__,snapshot/'sjg_density_endpoint_overlay_1279.py')
    model=read(OUT/'run437/training_receipt.json')
    assert model['status']=='NATIVE_AUXILIARY_OCCUPANCY_TRAINED' and model['G']==269 and model['epochs']==20
    assert digest(OUT/'run437/sequence_model.pt')==model['model_sha256']
    write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),keys=list(KEYS),
        base_endpoints='run421/endpoints',new_occupancy='run437',
        feature_identity_kind=IDENTITY_KIND,unchanged_columns=177,replaced_columns=list(COLUMNS),
        candidate_positions_and_role_map_frozen=True,GT_reads=0,STEP_reads=0,
        old_model_event_positions_retained_as_input_candidates=True,
        old_occupancy_probability_values_not_used=True,full_contract_pass=False))
    write(dest/'controls.json',controls());cases=[]
    for key in KEYS:
        base=OUT/f'run421/endpoints/source_{key}';br=read(base/'receipt.json')
        assert digest(base/'features.npy')==br['feature_sha256'] and digest(base/'role_map.npz')==br['map_sha256']
        folder=dest/f'source_{key}';folder.mkdir();shutil.copy2(base/'role_map.npz',folder/'role_map.npz')
        with np.load(base/'role_map.npz') as z:em={k:z[k] for k in z.files}
        seq=OUT/f'run421/sequence56/source_{key}';sr=read(seq/'receipt.json')
        assert digest(seq/'sequence_map.npz')==sr['map_sha256']
        assert digest(seq/'features.npy')==sr['features_sha256']
        with np.load(seq/'sequence_map.npz') as z:sm={k:z[k] for k in z.files}
        occ=OUT/f'run437/proposal_{key}';pr=read(occ/'prediction_receipt.json')
        assert pr['GT_reads']==pr['STEP_reads']==0 and pr['model_sha256']==model['model_sha256']
        assert digest(occ/'occupancy.npz')==pr['occupancy_sha256']
        assert pr['features_sha256']==sr['features_sha256'] and pr['map_sha256']==sr['map_sha256']
        with np.load(occ/'occupancy.npz') as z:o={k:z[k] for k in z.files}
        assert np.array_equal(em['rows'],o['admitted_rows']) and np.array_equal(em['rows'],sm['admitted_rows'])
        lookup={int(p):j for j,p in enumerate(sm['profile_ids'])}
        sequence=np.load(seq/'features.npy')
        overrides=np.empty((2*len(em['q_mm']),3),np.float32)
        for i,row in enumerate(em['rows']):
            a,b=map(int,em['offsets'][i:i+2]);q=em['q_mm'][a:b];n=b-a
            j=lookup[int(row)//2];sa,sb=map(int,sm['offsets'][j:j+2]);grid=sm['q_mm'][sa:sb]
            lo,hi=map(int,o['offsets'][i:i+2]);prob=o['probabilities'][lo:hi]
            assert len(grid)==len(prob) and grid[0]==0.
            length=grid[-1];closed=bool(sequence[sa,27]>.5)
            for role,inward in enumerate((1,-1)):
                queries=(q,q+inward*50.,q-inward*50.)
                queries=[np.mod(v,length) if closed else np.clip(v,0,length) for v in queries]
                overrides[2*a+role*n:2*a+(role+1)*n]=np.column_stack([np.interp(v,grid,prob) for v in queries])
        assert np.isfinite(overrides).all() and overrides.min()>=0 and overrides.max()<=1
        np.save(folder/'occupancy_overrides.npy',overrides)
        raw=np.load(base/'features.npy',mmap_mode='r')
        spec=dict(base_file=str((base/'features.npy').relative_to(OUT)),base_sha256=br['feature_sha256'],
            override_file='occupancy_overrides.npy',override_sha256=digest(folder/'occupancy_overrides.npy'),
            columns=list(COLUMNS),shape=list(raw.shape),dtype=str(raw.dtype))
        write(folder/'feature_overlay.json',spec)
        receipt=dict(br,status='PURE_INPUT_ENDPOINT_FEATURE_OVERLAY_FROZEN',feature_identity_kind=IDENTITY_KIND,
            feature_sha256=identity(spec),overlay_spec_sha256=digest(folder/'feature_overlay.json'),
            occupancy_sha256=pr['occupancy_sha256'],occupancy_model_sha256=model['model_sha256'],
            base_feature_sha256=br['feature_sha256'],map_sha256=digest(folder/'role_map.npz'),
            unchanged_geometry_columns=177,old_occupancy_probability_values_not_used=True,GT_reads=0,STEP_reads=0)
        write(folder/'receipt.json',receipt)
        view=load_overlay(folder,receipt);unchanged=np.array([k for k in range(180) if k not in COLUMNS])
        for first in range(0,len(raw),8192):
            x=view[first:first+8192]
            assert np.array_equal(x[:,unchanged],raw[first:first+8192,unchanged])
            assert np.array_equal(x[:,list(COLUMNS)],overrides[first:first+8192])
        view.close();raw._mmap.close();del sequence,overrides;gc.collect()
        result=dict(source=key,feature_sha256=receipt['feature_sha256'],map_sha256=receipt['map_sha256'],
            all_rows_177_columns_exact=True,all_probability_replacements_exact=True,
            receipt_sha256=digest(folder/'receipt.json'))
        cases.append(result);print('ENDPOINT_OCCUPANCY_OVERLAY_COMPLETE',key,flush=True);check_resources()
    write(dest/'receipt.json',dict(status='ALL_SIX_ENDPOINT_OVERLAYS_FROZEN_BEFORE_LABEL_BINDING',cases=cases,
        model_sha256=model['model_sha256'],GT_reads=0,STEP_reads=0,full_contract_pass=False))


if __name__=='__main__':prepare()
