"""Bounded, graph-local execution of the unchanged finite-void gap predicate."""
from collections import OrderedDict
import hashlib
import numpy as np


class GapResolver:
    def __init__(self, graph, fragments, audit=False, matrix_limit=64*1024*1024, lru_limit=65536):
        self.graph=graph
        self.fragments=np.unique(np.asarray(fragments,np.int64))
        self.index={int(f):i for i,f in enumerate(self.fragments)}
        self.profiles=np.array([graph.profile(f) for f in self.fragments],np.int64)
        self.boxes=np.array([graph.bounds(f) for f in self.fragments],np.float64).reshape(-1,2,3)
        self.size=len(self.fragments); self.lru_limit=lru_limit; self.lru=OrderedDict()
        self.cache=np.zeros((self.size,self.size),np.uint8) if self.size**2<=matrix_limit else None
        self.possible=np.zeros((self.size,self.size),bool) if self.cache is not None else None
        self.reference=np.zeros_like(self.possible) if audit and self.possible is not None else None
        self.audit=audit; self.summary_hash=hashlib.sha256()
        self.stats=dict(pending_checks=0,necessary_false_skipped=0,exact_calls=0,cache_hits=0,
            audit_checks=0,audit_differences=0,uncertain_source_profiles=0,
            cache_bytes=0 if self.cache is None else self.cache.nbytes,
            prefilter_bytes=0 if self.possible is None else self.possible.nbytes,
            reference_bytes=0 if self.reference is None else self.reference.nbytes,
            cache_scope='this graph and its frozen geometry only',lru_limit=lru_limit)
        if self.possible is not None:self._prepare()

    def _points(self,pid):
        g=self.graph; curve=g.curve(pid)
        if pid not in g._gap_points:
            gaps=np.asarray(g.profiles[pid].get('unobserved_gaps_mm',[]),np.float64).reshape(-1,2)
            g._gap_points[pid]=curve.at(np.clip(gaps,0,curve.length))
        return g._gap_points[pid]

    def _prepare(self):
        # Retain complete-profile padding and each void's paired endpoints.
        for pid in np.unique(self.profiles):
            points=self._points(int(pid)); count=len(points)
            if not count:continue
            own=np.flatnonzero(self.profiles==pid)
            # A large/uncertain source goes to the original exact predicate.
            if self.size*count*2>32*1024*1024 or not np.isfinite(points).all():
                self.possible[own,:]=True;self.possible[:,own]=True
                if self.reference is not None:self.reference[own,:]=True;self.reference[:,own]=True
                self.stats['uncertain_source_profiles']+=1;continue
            pad=64*np.finfo(float).eps*max(1.,float(np.abs(points).max()))
            near=np.empty((self.size,count,2),bool)
            for start in range(0,self.size,64):
                boxes=self.boxes[start:start+64]
                for first in range(0,count,128):
                    p=points[first:first+128]
                    delta=np.maximum(0.,np.maximum(boxes[:,None,None,0,:]-p[None],p[None]-boxes[:,None,None,1,:]))
                    near[start:start+len(boxes),first:first+len(p)]=np.linalg.norm(delta,axis=-1)<=3.+pad
            for gap in range(count):
                a=np.flatnonzero(near[:,gap,0]);b=np.flatnonzero(near[:,gap,1])
                ao=a[self.profiles[a]==pid];bo=b[self.profiles[b]==pid]
                self.possible[np.ix_(ao,b)]=True;self.possible[np.ix_(b,ao)]=True
                self.possible[np.ix_(bo,a)]=True;self.possible[np.ix_(a,bo)]=True
            if self.reference is not None:
                # Independent pairing expression from graph.connection, with
                # its original gap axis retained until the final any().
                for i in own:
                    q=np.any((near[i,:,0]&near[:,:,1])|(near[i,:,1]&near[:,:,0]),axis=1)
                    self.reference[i]|=q;self.reference[:,i]|=q
        self.stats['potential_fragment_pairs']=int(self.possible.sum())

    def _original_necessary(self,a,b):
        aa,bb=self.boxes[a],self.boxes[b]
        for pid in (self.profiles[a],self.profiles[b]):
            points=self._points(int(pid))
            if not len(points):continue
            if not np.isfinite(points).all():return True
            pad=64*np.finfo(float).eps*max(1.,float(np.abs(points).max()))
            near=[]
            for box in (aa,bb):
                delta=np.maximum(0.,np.maximum(box[0]-points,points-box[1]))
                near.append(np.linalg.norm(delta,axis=-1)<=3.+pad)
            if np.any((near[0][:,0]&near[1][:,1])|(near[0][:,1]&near[1][:,0])):return True
        return False

    def resolve(self,fragment,others):
        a=self.index[int(fragment)];ids=np.fromiter((self.index[int(v)] for v in others),np.int64,count=len(others))
        possible=self.possible[a,ids] if self.possible is not None else np.fromiter(
            (self._original_necessary(a,int(b)) for b in ids),bool,count=len(ids))
        self.stats['pending_checks']+=len(ids)
        self.stats['necessary_false_skipped']+=int((~possible).sum())
        if self.audit:
            ref=self.reference[a,ids] if self.reference is not None else np.fromiter(
                (self._original_necessary(a,int(b)) for b in ids),bool,count=len(ids))
            differences=int(np.count_nonzero(ref!=possible));self.stats['audit_checks']+=len(ids)
            self.stats['audit_differences']+=differences
            self.summary_hash.update(np.asarray([a,len(ids)],np.int64).tobytes())
            self.summary_hash.update(ids.tobytes());self.summary_hash.update(possible.tobytes())
            if differences:raise AssertionError('Finite-void necessary-condition mismatch')
        result=np.zeros(len(ids),bool)
        for j in np.flatnonzero(possible):
            b=int(ids[j]);lo,hi=sorted((a,b));key=(lo,hi)
            value=int(self.cache[lo,hi]) if self.cache is not None else self.lru.get(key,0)
            if not value:
                value=1+int(self.graph.connection(int(self.fragments[lo]),int(self.fragments[hi]),'POSITIVE_GAP'))
                self.stats['exact_calls']+=1
                if self.cache is not None:self.cache[lo,hi]=value
                else:
                    self.lru[key]=value
                    if len(self.lru)>self.lru_limit:self.lru.popitem(last=False)
            else:
                self.stats['cache_hits']+=1
                if self.cache is None:self.lru.move_to_end(key)
            result[j]=value==2
        return result

    def receipt(self):
        return dict(self.stats,fragment_count=self.size,summary_sha256=self.summary_hash.hexdigest(),
            cache_entries=len(self.lru) if self.cache is None else int(np.count_nonzero(self.cache)))
