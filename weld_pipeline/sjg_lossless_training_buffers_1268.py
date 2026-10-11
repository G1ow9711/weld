"""Exact in-memory compressed row blocks; bounded decompression cache."""
from collections import OrderedDict
import hashlib
import zlib
import numpy as np


class LosslessRows:
    def __init__(self, mapped, block_rows=512, cache_blocks=4):
        self.shape=mapped.shape;self.dtype=mapped.dtype
        assert len(self.shape)==2 and not self.dtype.hasobject
        self.block_rows=block_rows;self.cache_blocks=cache_blocks
        self.blocks=[];self.cache=OrderedDict();sha=hashlib.sha256()
        self.nbytes=int(mapped.nbytes)
        for first in range(0,len(mapped),block_rows):
            raw=np.ascontiguousarray(mapped[first:first+block_rows]).tobytes()
            shuffled=np.frombuffer(raw,np.uint8).reshape(-1,self.dtype.itemsize).T.copy().tobytes()
            packed=zlib.compress(shuffled,1)
            restored=np.frombuffer(zlib.decompress(packed),np.uint8).reshape(self.dtype.itemsize,-1).T.copy().tobytes()
            assert restored==raw
            self.blocks.append(packed);sha.update(raw)
        self.raw_sha256=sha.hexdigest()
        self.stored_bytes=sum(map(len,self.blocks))

    def __len__(self):return self.shape[0]

    def _block(self,index):
        if index not in self.cache:
            shuffled=zlib.decompress(self.blocks[index])
            raw=np.frombuffer(shuffled,np.uint8).reshape(self.dtype.itemsize,-1).T.copy().tobytes()
            self.cache[index]=np.frombuffer(raw,dtype=self.dtype).reshape(-1,self.shape[1])
            if len(self.cache)>self.cache_blocks:self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self,key):
        if isinstance(key,(int,np.integer)):
            key=int(key)
            if key<0:key+=len(self)
            if not 0<=key<len(self):raise IndexError(key)
            return self._block(key//self.block_rows)[key%self.block_rows]
        if not isinstance(key,slice):raise TypeError('Only row integers and slices are supported')
        start,stop,step=key.indices(len(self))
        if step!=1:raise ValueError('Only contiguous slices are supported')
        if stop<=start:return np.empty((0,self.shape[1]),dtype=self.dtype)
        pieces=[]
        while start<stop:
            block=self._block(start//self.block_rows);offset=start%self.block_rows
            count=min(stop-start,len(block)-offset)
            pieces.append(block[offset:offset+count]);start+=count
        return pieces[0] if len(pieces)==1 else np.concatenate(pieces,axis=0)


def controls():
    rng=np.random.default_rng(1268)
    for shape in ((1,3),(512,5),(1031,7)):
        x=rng.normal(size=shape).astype(np.float32);packed=LosslessRows(x)
        for start in (0,1,511,512,1023):
            for width in (0,1,2,511,700):
                assert np.array_equal(packed[start:start+width],x[start:start+width])
        for index in (0,len(x)-1,-1):assert np.array_equal(packed[index],x[index])
        assert len(packed.cache)<=4
    return dict(status='PASS',exact_slice_boundary_and_integer_controls=True,lossy_operations=0)
