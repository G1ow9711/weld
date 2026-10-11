"""Bounded exact cache of input-only relation columns, independent of weights/GT."""
from collections import OrderedDict
import hashlib,shutil
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations


class RelationColumnCache:
    def __init__(self,root,bindings,memory_limit=64*1024*1024,disk_limit=512*1024*1024):
        self.root=root;root.mkdir(exist_ok=False);self.bindings=bindings
        self.memory_limit=memory_limit;self.disk_limit=disk_limit;self.memory=OrderedDict()
        self.memory_bytes=0;self.disk_bytes=0
        self.stats=dict(computed_columns=0,memory_hits=0,disk_hits=0,uncached_columns=0,disk_write_skips=0)
        for key,binding in bindings.items():
            folder=root/key;folder.mkdir();write(folder/'binding.json',binding)

    def column(self,key,index,context):
        slot=(key,int(index));shape=(len(context.parents),16)
        if slot in self.memory:
            self.stats['memory_hits']+=1;self.memory.move_to_end(slot);return self.memory[slot]
        folder=self.root/key;path=folder/(str(index)+'.npy');receipt=folder/(str(index)+'.json')
        if path.exists():
            r=read(receipt);assert digest(path)==r['sha256']
            value=np.load(path);assert value.shape==shape and value.dtype==np.float64 and np.isfinite(value).all()
            self.stats['disk_hits']+=1
        else:
            value=np.zeros(shape,np.float64)
            for i in range(len(context.parents)):
                if i!=index:value[i]=context.pair(i,index)
            self.stats['computed_columns']+=1
            required=value.nbytes+256
            if self.disk_bytes+required<=self.disk_limit and shutil.disk_usage(self.root).free-required>=5368709120+1024*1024:
                atomic_binary_write(path,lambda stream:np.save(stream,value,allow_pickle=False))
                write(receipt,dict(sha256=digest(path),shape=list(shape),dtype='float64',candidate_index=int(index)))
                self.disk_bytes+=path.stat().st_size+receipt.stat().st_size
            else:self.stats['disk_write_skips']+=1
        if value.nbytes<=self.memory_limit:
            while self.memory and self.memory_bytes+value.nbytes>self.memory_limit:
                _,old=self.memory.popitem(last=False);self.memory_bytes-=old.nbytes
            self.memory[slot]=value;self.memory_bytes+=value.nbytes
        else:self.stats['uncached_columns']+=1
        return value

    def receipt(self):return dict(self.stats,memory_bytes=self.memory_bytes,disk_bytes=self.disk_bytes,
        memory_limit=self.memory_limit,disk_limit=self.disk_limit,GT_reads=0,model_weights_in_cache_key=False)


class CachedParentRelations(ParentRelations):
    def __init__(self,parents,metadata,key,cache):
        super().__init__(parents,metadata);self.key=key;self.cache=cache

    def add(self,index):
        if index in self.selected:raise ValueError('Repeated candidate')
        column=self.cache.column(self.key,index,self)
        self.selected.append(index);mask=np.ones(len(self.parents),bool);mask[self.selected]=False
        self.total[mask]+=column[mask];self.maximum[mask]=np.maximum(self.maximum[mask],column[mask])
