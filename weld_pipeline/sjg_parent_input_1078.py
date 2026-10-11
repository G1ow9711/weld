"""Hash-bound, input-only combination of PT/geometry context and MLS residual bins."""
import argparse
import gc
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources,resources,reclaim_own_idle_pages

BASE_RUN={'21':'run253','BM79A_native_local':'run253','10':'run258'}


class ParentInput:
    def __init__(self,key,normalization=None,verify=True):
        self.key=key;self.base=OUT/BASE_RUN[key]/('source_'+key)
        self.geometry=OUT/'run259'/('source_'+key)
        self.base_receipt=read(self.base/'receipt.json');self.geo_receipt=read(self.geometry/'receipt.json')
        b,g=self.base_receipt,self.geo_receipt
        assert b['status']=='ALL_PROFILE_ORDERED_BINS_READY' and g['status']=='ALL_PROFILE_MLS_GEOMETRY_READY'
        assert b['profiles']==g['profiles']
        if verify:
            assert digest(self.base/'ordered_bins.npy')==b['bins_sha256']
            assert digest(self.geometry/'geometry_ordered_bins.npy')==g['bin_sha256']
            assert digest(OUT/b['context_file'])==b['context_sha256']
            assert digest(OUT/f'run201/source_{key}_joint_mapping.npz')==b['mapping_sha256']
        with np.load(OUT/f'run201/source_{key}_joint_mapping.npz') as z:self.family=z['profile_family']
        self.context=np.load(OUT/b['context_file'],mmap_mode='r')
        self.bins=np.load(self.base/'ordered_bins.npy',mmap_mode='r')
        self.geobins=np.load(self.geometry/'geometry_ordered_bins.npy',mmap_mode='r')
        assert self.bins.shape==(len(self.family),16,188)
        assert self.geobins.shape==(len(self.family),16,54)
        self.normalization=normalization

    def __len__(self):return len(self.family)

    def arrays(self,indices):
        context=np.asarray(self.context[self.family[indices]],np.float32)
        bins=np.concatenate((self.bins[indices],self.geobins[indices]),axis=-1).astype(np.float32,copy=False)
        if self.normalization is not None:
            n=self.normalization
            context=(context-n['context_mean'])/n['context_scale']
            bins=(bins-n['bin_mean'])/n['bin_scale']
        assert np.isfinite(context).all() and np.isfinite(bins).all()
        return context,bins

    def binding(self):
        return dict(source=self.key,profiles=len(self),base_receipt_sha256=digest(self.base/'receipt.json'),
            geometry_receipt_sha256=digest(self.geometry/'receipt.json'),context_sha256=self.base_receipt['context_sha256'],
            base_bins_sha256=self.base_receipt['bins_sha256'],geometry_bins_sha256=self.geo_receipt['bin_sha256'],
            mapping_sha256=self.base_receipt['mapping_sha256'],combined_shape=[len(self),16,242])


def prepare_normalization():
    dest=OUT/'run260';dest.mkdir(exist_ok=False);means=[];squares=[];bindings=[]
    for key in BASE_RUN:
        check_resources();data=ParentInput(key)
        cm=np.zeros(669);cs=np.zeros(669);bm=np.zeros((16,242));bs=np.zeros((16,242))
        for first in range(0,len(data),512):
            c,b=data.arrays(slice(first,first+512))
            cm+=c.sum(0,dtype=np.float64);cs+=np.square(c,dtype=np.float64).sum(0)
            bm+=b.sum(0,dtype=np.float64);bs+=np.square(b,dtype=np.float64).sum(0)
            if first%8192==0 and resources()['available_physical_bytes']<1879048192:
                reclaim_own_idle_pages();check_resources()
        means.append((cm/len(data),bm/len(data)));squares.append((cs/len(data),bs/len(data)))
        bindings.append(data.binding());print('PARENT_INPUT_NORMALIZATION',key,len(data),'profiles',flush=True)
        del data,c,b;gc.collect();reclaim_own_idle_pages()
    cm=np.mean([x[0] for x in means],axis=0);bm=np.mean([x[1] for x in means],axis=0)
    cs=np.sqrt(np.maximum(np.mean([x[0] for x in squares],axis=0)-cm*cm,0))
    bs=np.sqrt(np.maximum(np.mean([x[1] for x in squares],axis=0)-bm*bm,0))
    cs[cs<1e-6]=1.;bs[bs<1e-6]=1.
    np.savez(dest/'input_normalization.npz',context_mean=cm.astype(np.float32),context_scale=cs.astype(np.float32),
             bin_mean=bm.astype(np.float32),bin_scale=bs.astype(np.float32))
    write(dest/'input_freeze.json',dict(status='THREE_TRAIN_COMPLETE_INPUTS_WITH_MLS_FROZEN',cases=bindings,
        normalization='macro by 21/BM/10 complete input populations; per ordered bin/channel',
        normalization_sha256=digest(dest/'input_normalization.npz'),
        user_requested_geometry_preprocessing=True,all_candidates_retained=True,
        target_reads=0,STEP_reads=0,DEV_supervision=False,optimizer_updates=0,
        training_and_full_decoder_pending=True,utc=utc()))


if __name__=='__main__':prepare_normalization()
