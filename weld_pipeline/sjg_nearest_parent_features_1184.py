"""Input-only nearest whole-parent relation, preserving original max/mean."""
import numpy as np
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations


class NearestParentRelations(ParentRelations):
    def __init__(self,parents,metadata,key=None,cache=None):
        super().__init__(parents,metadata)
        n=len(parents);self.key=key;self.cache=cache
        self.nearest=np.zeros((n,16),np.float64)
        self.nearest_distance=np.full(n,np.inf)
        self.nearest_index=np.full(n,-1,dtype=np.int64)

    def add_column(self,index,column):
        if index in self.selected:raise ValueError('Repeated candidate')
        column=np.asarray(column,dtype=np.float64)
        if column.shape!=(len(self.parents),16) or not np.isfinite(column).all():
            raise ValueError('Invalid complete relation column')
        self.selected.append(index)
        mask=np.ones(len(self.parents),bool);mask[self.selected]=False
        self.total[mask]+=column[mask]
        self.maximum[mask]=np.maximum(self.maximum[mask],column[mask])
        distance=column[:,:3].max(axis=1)
        replace=mask & ((distance<self.nearest_distance) |
            ((distance==self.nearest_distance)&(index<self.nearest_index)))
        self.nearest[replace]=column[replace]
        self.nearest_distance[replace]=distance[replace]
        self.nearest_index[replace]=index

    def add(self,index):
        if index in self.selected:raise ValueError('Repeated candidate')
        if self.cache is not None:column=self.cache.column(self.key,index,self)
        else:
            column=np.zeros((len(self.parents),16),np.float64)
            for i in range(len(self.parents)):
                if i not in self.selected and i!=index:column[i]=self.pair(i,index)
        self.add_column(index,column)

    def features(self):
        return np.concatenate((super().features(),self.nearest),axis=1)
