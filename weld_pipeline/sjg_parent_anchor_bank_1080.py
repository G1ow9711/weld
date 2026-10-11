"""Lazy, full-domain endpoint interval dictionary from input geometry only."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier


class CarrierBank:
    def __init__(self,key,capacity_only=False):
        self.key=key;self.cache={}
        base=OUT/f'run185/source_{key}';binding=read(base/'input_freeze.json')
        assert digest(base/'profiles.json')==binding['files']['profiles.json']
        self.profiles=read(base/'profiles.json')
        db=next(c for c in read(OUT/'run201/all_domains_frozen_before_labels.json')['cases'] if c['source_key']==key)
        assert digest(OUT/'run201'/db['json_file'])==db['json_sha256']
        assert digest(OUT/'run201'/db['mapping_file'])==db['mapping_sha256']
        self.domain=read(OUT/'run201'/db['json_file'])
        with np.load(OUT/'run201'/db['mapping_file']) as z:
            self.family=z['profile_family'];self.relation=z['profile_relation'];self.transform=z['local_to_canonical']
        self.capacity_only=bool(capacity_only)
        if capacity_only:
            self.cells=None;self.offsets=None
        elif key=='21':
            frozen=read(OUT/'run224/input_freeze.json')
            assert digest(OUT/'run224/input_domain.npz')==frozen['files']['input_domain.npz']
            with np.load(OUT/'run224/input_domain.npz') as z:self.cells=z['cells'];family=z['family']
            self.offsets=np.searchsorted(family,np.arange(len(self.domain['families'])+1))
        else:
            folder=OUT/'run252'/('source_'+key);r=read(folder/'receipt.json')
            for name in ('cells.f64','family_offsets.npy'):assert digest(folder/name)==r['files'][name]
            self.cells=np.memmap(folder/'cells.f64',mode='r',dtype='<f8',shape=(r['cells'],4))
            self.offsets=np.load(folder/'family_offsets.npy')

    def __len__(self):return len(self.profiles)

    def get(self,profile):
        profile=int(profile)
        if profile in self.cache:return self.cache[profile]
        p=self.profiles[profile];curve=CurveCarrier(p['representation'])
        fid=int(self.family[profile]);family=self.domain['families'][fid]
        start,sign,length=self.transform[profile]
        if self.capacity_only:
            local=np.asarray([v for key in ('support_intervals_mm','unobserved_gaps_mm') for row in p.get(key,[]) for v in row],np.float64)
        else:
            first,last=self.offsets[fid:fid+2]
            canonical=np.unique(self.cells[first:last,:2])
            local=sign*(canonical-start)
            if family['circumference_mm']:local%=family['circumference_mm']
        # Exact equality only. Nearby independent anchors and all finite extent remain.
        anchors=np.unique(np.r_[0.,curve.length,local[(local>0)&(local<curve.length)]])
        left,right=anchors[:-1],anchors[1:];width=right-left
        assert len(left)>0 and np.all(width>0)
        midpoint=(left+right)/2
        support=np.zeros(len(left));gap=np.zeros(len(left))
        for lo,hi in p.get('support_intervals_mm',[]):support=np.maximum(support,(midpoint>=lo)&(midpoint<=hi))
        for lo,hi in p.get('unobserved_gaps_mm',[]):gap=np.maximum(gap,(midpoint>lo)&(midpoint<hi))
        features=np.c_[left/curve.length,right/curve.length,width/1000,np.log1p(width)/10,
                       np.full(len(left),curve.kind=='arc'),
                       np.full(len(left),curve.kind=='arc' and curve.arc.is_closed),support,gap].astype(np.float32)
        result=dict(profile=profile,family=fid,relation=int(self.relation[profile]),curve=curve,
                    anchors_mm=anchors,interval_features=features,
                    ordered_face_indices=p.get('face_indices',[]),
                    side_hypothesis=p.get('weld_side',p.get('side_hypothesis')),
                    material_identity='UNKNOWN')
        self.cache[profile]=result
        return result

    def locate(self,profile,q):
        """Training/control helper; never called with a GT coordinate in inference."""
        item=self.get(profile);edges=item['anchors_mm'];q=float(q)
        if not np.isfinite(q) or q<0 or q>edges[-1]:raise ValueError('Endpoint outside original finite profile')
        i=min(int(np.searchsorted(edges,q,side='right')-1),len(edges)-2)
        fraction=(q-edges[i])/(edges[i+1]-edges[i])
        return i,float(fraction)

    def coordinate(self,profile,interval,fraction):
        item=self.get(profile);edges=item['anchors_mm'];i=int(interval);fraction=float(fraction)
        if not 0<=i<len(edges)-1 or not 0<=fraction<=1:raise ValueError('Invalid endpoint interval/fraction')
        return float(edges[i]+fraction*(edges[i+1]-edges[i]))
