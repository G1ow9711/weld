"""Freeze macro TRAIN normalization before any new label/loss control."""
import argparse,gc,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_balanced_hierarchy_1043 import FAMILY_BINARY_COLUMNS,RELATION_BINARY_COLUMNS,controls

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    start=time.perf_counter();manifest=read(OUT/'run206/training_manifest.json');bindings={b['source_key']:b for b in manifest['static_bindings']}
    means={k:[] for k in ('relation','family')};seconds={k:[] for k in means};frozen=[]
    for case in manifest['cases']:
        key=case['source_key'];b=bindings[key];path=OUT/'run206'/b['file'];assert digest(path)==b['sha256']
        with np.load(path) as z:
            for kind in means:
                x=z[kind+'_x'].astype(np.float64);assert np.isfinite(x).all()
                means[kind].append(x.mean(0));seconds[kind].append(np.square(x).mean(0));del x
        frozen.append(dict(source=key,file=b['file'],sha256=b['sha256']));gc.collect()
    statistics=dict(TRAIN_sources=[c['source_key'] for c in manifest['cases']],case_weight='one seventh each',variance='macro second moment minus macro mean squared',
        statistic_dtype='float64',threshold_std=1e-6,no_clipping=True,raw_label_reads=0,bindings=frozen)
    for kind,binary in [('relation',RELATION_BINARY_COLUMNS),('family',FAMILY_BINARY_COLUMNS)]:
        mean=np.mean(means[kind],axis=0);variance=np.maximum(0,np.mean(seconds[kind],axis=0)-mean**2);std=np.sqrt(variance)
        scale=np.where(std<1e-6,1,std);applied=mean.copy();applied[list(binary)]=0;scale[list(binary)]=1
        statistics[kind]=dict(raw_mean=mean.tolist(),raw_std=std.tolist(),applied_mean=applied.tolist(),applied_scale=scale.tolist(),
            binary_columns_unscaled=list(binary),continuous_columns=[i for i in range(len(mean)) if i not in binary])
    write(dest/'normalization.json',statistics)
    ranges=[]
    for case in manifest['cases']:
        b=bindings[case['source_key']]
        with np.load(OUT/'run206'/b['file']) as z:
            for kind,binary in [('relation',RELATION_BINARY_COLUMNS),('family',FAMILY_BINARY_COLUMNS)]:
                x=z[kind+'_x'];assert np.isin(x[:,list(binary)],[0,1]).all()
                value=(x-np.asarray(statistics[kind]['applied_mean'],np.float32))/np.asarray(statistics[kind]['applied_scale'],np.float32)
                assert np.isfinite(value).all();ranges.append(dict(source=case['source_key'],kind=kind,maximum_abs=float(abs(value).max())))
    write(dest/'gradient_controls.json',controls())
    write(dest/'receipt.json',dict(status='MACRO_TRAIN_NORMALIZATION_READY',normalization_sha256=digest(dest/'normalization.json'),
        finite_all_TRAIN=True,ranges=ranges,optimizer_updates=0,no_heldout_statistics=True,seconds=time.perf_counter()-start,end_utc=utc()))
    print('TRAIN_NORMALIZATION_READY','sources',len(frozen),'control updates',0,flush=True)

if __name__=='__main__':main()
