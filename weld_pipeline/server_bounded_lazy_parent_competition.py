"""Keep exact full-input competition with bounded reusable feature storage.

All requested edges receive the original features and prior penalties. Output
arrays still contain every edge; eviction affects only reuse between calls.
The original neighbour radii, ties, pairs and resource limits are inherited.
"""
from collections import OrderedDict

import numpy as np

from .server_lazy_parent_competition import LazyParentCompetition


class BoundedLazyParentCompetition(LazyParentCompetition):
    def __init__(self, parents, priors, neighbors=16, max_active_pairs=4_000_000,
                 feature_chunk_size=2048, max_cached_pairs=8192):
        if type(max_cached_pairs) is not int or max_cached_pairs < 1:
            raise ValueError('A positive relation reuse-cache capacity is required')
        super().__init__(parents,priors,neighbors,max_active_pairs,feature_chunk_size)
        self.cache=OrderedDict()
        self.max_cached_pairs=max_cached_pairs
        self.feature_cache_evictions=0
        self.peak_cached_pairs=0

    def features(self,pairs):
        pairs=np.asarray(pairs,np.int64).reshape(-1,2)
        if (len(pairs)>self.max_active_pairs or np.any(pairs<0)
                or np.any(pairs>=len(self.parents)) or np.any(pairs[:,0]>=pairs[:,1])
                or len(np.unique(pairs,axis=0))!=len(pairs)):
            raise ValueError('Unique valid unordered competition pairs within declared resources required')
        x=np.empty((len(pairs),32),np.float32)
        base=np.empty(len(pairs),float)
        for start in range(0,len(pairs),self.feature_chunk_size):
            block=pairs[start:start+self.feature_chunk_size]
            values=[];missing=[]
            for position,pair in enumerate(block):
                key=tuple(int(v) for v in pair)
                value=self.cache.get(key)
                values.append(value)
                if value is None:missing.append(position)
                else:self.cache.move_to_end(key)
            if missing:
                indices=block[missing]
                features,probability=self.priors.duplicate_probabilities(self.parents,indices)
                scale=2*np.minimum(self.counts[indices[:,0]],self.counts[indices[:,1]])
                for position,row,penalty in zip(missing,features,probability*scale):
                    value=(row.copy(),float(penalty));values[position]=value
                    key=tuple(int(v) for v in block[position])
                    self.cache[key]=value
                    while len(self.cache)>self.max_cached_pairs:
                        self.cache.popitem(last=False);self.feature_cache_evictions+=1
                    self.peak_cached_pairs=max(self.peak_cached_pairs,len(self.cache))
            x[start:start+len(block)]=[value[0] for value in values]
            base[start:start+len(block)]=[value[1] for value in values]
        return x,base
