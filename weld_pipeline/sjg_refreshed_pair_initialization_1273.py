"""Bind unchanged native interval targets to repaired inputs and fresh scaling."""
import gc
import shutil
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_endpoint_occupancy_overlay_1272 import load_overlay,IDENTITY_KIND
from weld_pipeline.sjg_joint_pair_model_1248 import PairModel

TRAIN=('21','BM79A_native_local','10','25','36')


def prepare():
    check_resources();dest=OUT/'run431';dest.mkdir(exist_ok=False)
    snapshot=dest/'source_snapshot';snapshot.mkdir();shutil.copy2(__file__,snapshot/'sjg_refreshed_pair_initialization_1273.py')
    frozen=read(OUT/'run430/receipt.json')
    assert frozen['status']=='ALL_SIX_ENDPOINT_OVERLAYS_FROZEN_BEFORE_LABEL_BINDING' and len(frozen['cases'])==6
    source=read(OUT/'run423/receipt.json');assert source['G']==269
    write(dest/'registered_protocol.json',dict(TRAIN=list(TRAIN),DEV=[],G=269,
        source_sha256=digest(__file__),frozen_inputs_sha256=digest(OUT/'run430/receipt.json'),
        source_supervision='run423',input_run='run430',occupancy_run='run429',
        target_geometry_and_all_record_values_unchanged=True,normalization_current_TRAIN_only=True,
        initialization='fresh seed1031 PairModel',fit_calls=0,full_contract_pass=False))
    sums=np.zeros(180);squares=np.zeros(180);csum=np.zeros(160);csquares=np.zeros(160)
    count=ccount=0;cases=[];bindings=[]
    for case in source['cases']:
        key=case['source'];assert key in TRAIN
        path=OUT/f'run423/source_{key}_set_targets.json';assert digest(path)==case['target_sha256']
        data=read(path);folder=OUT/f'run430/source_{key}';ir=read(folder/'receipt.json')
        frozen_case=next(v for v in frozen['cases'] if v['source']==key)
        assert digest(folder/'receipt.json')==frozen_case['receipt_sha256']
        assert ir['feature_identity_kind']==IDENTITY_KIND and data['map_sha256']==ir['map_sha256']
        assert data['feature_sha256']==ir['base_feature_sha256']
        # Mapping and all 177 geometry columns already checked on every row.
        x=load_overlay(folder,ir)
        whole='run414/whole' if key in ('25','36') else 'run380'
        context_folder=OUT/whole/f'source_{key}';cr=read(context_folder/'receipt.json')
        assert cr['GT_reads']==cr['STEP_reads']==0 and digest(context_folder/'joint_features.npy')==cr['feature_sha256']
        context=np.load(context_folder/'joint_features.npy')
        for r in data['records']:
            first=r['feature_offset'];last=first+2*r['n']
            for offset in range(first,last,8192):
                v=x[offset:min(offset+8192,last)].astype(np.float64)
                assert np.isfinite(v).all();sums+=v.sum(0);squares+=(v*v).sum(0);count+=len(v)
            c=np.asarray(context[r['row']],np.float64);assert np.isfinite(c).all()
            csum+=c;csquares+=c*c;ccount+=1
        x.close();del context,v;gc.collect()
        data.update(previous_feature_sha256=data['feature_sha256'],feature_sha256=ir['feature_sha256'],
            feature_identity_kind=IDENTITY_KIND,unchanged_geometry_source_supervision='run423')
        output=dest/f'source_{key}_set_targets.json';write(output,data)
        reloaded=read(output)
        assert reloaded['records']==data['records'] and reloaded['original_G']==data['original_G']
        newcase=dict(case,target_sha256=digest(output));cases.append(newcase)
        bindings.append(dict(source=key,source_target_sha256=case['target_sha256'],new_target_sha256=newcase['target_sha256'],
            feature_sha256=ir['feature_sha256'],context_sha256=cr['feature_sha256'],all_supervision_records_roundtrip_exact=True))
        print('REFRESHED_PAIR_BINDING',key,'G',data['original_G'],'RECORDS',len(data['records']),flush=True)
        del data,reloaded;gc.collect();check_resources()
    assert sum(c['G'] for c in cases)==269
    def normalization(total,squared,n):
        mean=total/n;scale=np.sqrt(np.maximum(0,squared/n-mean*mean));scale[scale<1e-6]=1.
        return mean.astype(np.float32),scale.astype(np.float32)
    mean,scale=normalization(sums,squares,count);cm,cs=normalization(csum,csquares,ccount)
    norm=dict(mean=mean,scale=scale,context_mean=cm,context_scale=cs)
    torch.manual_seed(1031);model=PairModel(list(range(5)))
    payload=dict(classes=list(range(5)),state_dict=model.state_dict(),norm=norm,
        initialization='fresh untrained weights',input_run='run430')
    atomic_binary_write(dest/'pair_initialization.pt',lambda f:torch.save(payload,f))
    write(dest/'initialization_receipt.json',dict(status='UNTRAINED_INITIALIZATION_CURRENT_TRAIN_NORMALIZATION',
        model_sha256=digest(dest/'pair_initialization.pt'),seed=1031,TRAIN=list(TRAIN),G=269,
        endpoint_rows=count,context_rows=ccount,bindings=bindings,fit_calls=0,DEV_reads=0,STEP_reads=0,full_contract_pass=False))
    write(dest/'receipt.json',dict(source,cases=cases,input_run='run430',native_run='run422',
        source_supervision_run='run423',input_values_changed_supervision_records_unchanged=True,
        bindings=bindings,GT_used_to_regenerate_geometry=False,full_contract_pass=False))
    write(dest/'prefit_binding_audit.json',dict(status='PASS',G=269,all_six_inputs_frozen_before_TRAIN_binding=True,
        all_target_records_and_role_maps_unchanged=True,all_geometric_feature_columns_unchanged=True,
        new_feature_identity_kind=IDENTITY_KIND,fit_calls=0,DEV_reads=0,full_contract_pass=False))
    print('FRESH_PAIR_INITIALIZATION_READY_NO_FIT',flush=True)


if __name__=='__main__':prepare()
