"""Pure-input features for joint/family/adaptive interval prediction."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_geometry_query_bank_1025 import basis


def positions(family,q):
    d=np.asarray(family['descriptor']);q=np.asarray(q,dtype=np.float64)
    if family['kind']=='line':return d[:3]+q[...,None]*d[3:6]
    u,v=basis(d[3:6]);return d[:3]+d[6]*(np.cos(q/d[6])[...,None]*u+np.sin(q/d[6])[...,None]*v)


def static_features(domain,graph,face_latent):
    geometry=graph['geometry'];family_rows=[]
    for family in domain['families']:
        ids=np.asarray(family['profiles']);g=geometry[ids];ranges=family['domains_mm']
        total=sum(b-a for a,b in ranges)
        family_rows.append(np.r_[g.mean(0),g.min(0),g.max(0),np.log1p(len(ids))/10,
            np.log1p(total)/10,np.log1p(len(ranges))/10,float(family['kind']=='arc')])
    family_x=np.asarray(family_rows,np.float32);relation_rows=[]
    for relation in domain['relations']:
        faces=relation['ordered_face_indices'];a=face_latent[faces[0]] if faces else np.zeros(96)
        b=face_latent[faces[1]] if len(faces)>1 else np.zeros(96)
        g=family_x[relation['families']]
        relation_rows.append(np.r_[a,b,np.abs(a-b),g.mean(0),g.max(0),
            float(len(faces)>1),np.log1p(len(relation['profiles']))/10,np.log1p(len(g))/10])
    relation_x=np.asarray(relation_rows,np.float32)
    family_relation=np.asarray([f['relation'] for f in domain['families']],np.int64)
    assert np.isfinite(relation_x).all() and np.isfinite(family_x).all()
    return dict(relation_x=relation_x,family_x=family_x,family_relation=family_relation)


class IntervalFeatures:
    """Only XYZ and frozen PTv3 features enter this query provider."""
    def __init__(self,domain,profiles,transforms,xyz,point,max_queries=None):
        self.domain=domain;self.xyz=xyz;self.point=point;self.max_queries=max_queries
        # Original-point nearest neighbours; no reference-coordinate filtering.
        self.tree=cKDTree(xyz);self.cache={};self.query_count=0;self.anchors=[]
        for family in domain['families']:
            anchors=[]
            for p in family['profiles']:
                start,sign,length=transforms[p]
                for lo,hi in profiles[p]['support_intervals_mm']:
                    anchors.extend((start+sign*lo,start+sign*hi))
            self.anchors.append(np.unique(anchors))

    def point_summary(self,fid,q):
        key=(int(fid),float(q))
        if key in self.cache:return self.cache[key]
        self._fill([key]);return self.cache[key]

    def _fill(self,keys):
        unique=list(dict.fromkeys(k for k in keys if k not in self.cache))
        if self.max_queries is not None and self.query_count+len(unique)>self.max_queries:
            raise RuntimeError('Whole workpiece failed: adaptive XYZ query budget exceeded')
        for first in range(0,len(unique),4096):
            part=unique[first:first+4096]
            xyz=np.asarray([positions(self.domain['families'][fid],q) for fid,q in part])
            distance,ids=self.tree.query(xyz,k=16,workers=2)
            embedding=self.point['embedding'][ids];normal=self.point['normal'][ids];boundary=self.point['boundary'][ids]
            value=np.c_[embedding.mean(1),embedding.std(1),normal.mean(1),boundary.mean(1),boundary.max(1),
                         distance[:,0]/50,distance.mean(1)/50,(self.xyz[ids]-xyz[:,None]).mean(1)/50].astype(np.float32)
            assert value.shape[1]==42 and np.isfinite(value).all()
            self.cache.update(zip(part,value));self.query_count+=len(part)

    def batch(self,families,lo,hi):
        families=np.asarray(families,np.int64);lo=np.asarray(lo,float);hi=np.asarray(hi,float)
        if not (len(families)==len(lo)==len(hi)) or np.any(hi<=lo):raise ValueError('Nonempty finite cells required')
        q=np.c_[lo,(lo+hi)/2,hi]
        self._fill([(int(fid),float(v)) for fid,row in zip(families,q) for v in row])
        rows=[];scales=2.**np.arange(8)
        for fid,left,right,qs in zip(families,lo,hi,q):
            family=self.domain['families'][int(fid)];anchors=self.anchors[int(fid)]
            patches=np.concatenate([self.cache[(int(fid),float(v))] for v in qs])
            distances=[]
            for v in qs:
                index=np.searchsorted(anchors,v)
                before=v-anchors[index-1] if index else 1e6
                after=anchors[index]-v if index<len(anchors) else 1e6
                distances.extend(np.exp(-.5*(np.asarray([before,after])[:,None]/scales)**2).ravel())
            low=min(a for a,b in family['domains_mm']);high=max(b for a,b in family['domains_mm']);span=max(high-low,1e-8)
            midpoint=positions(family,(left+right)/2)/1000
            shape=np.r_[left/1000,right/1000,(right-left)/1000,np.log1p(right-left)/10,
                (left-low)/span,(right-low)/span,float(family['kind']=='arc'),
                100/family['descriptor'][6] if family['kind']=='arc' else 0,
                float(bool(family['circumference_mm'])),midpoint]
            rows.append(np.r_[patches,distances,shape])
        result=np.asarray(rows,np.float32).reshape(-1,186)
        assert np.isfinite(result).all();return result


def link_candidates(families,lo,hi,domain,limit=100000):
    """Local ordered proposals; proximity alone never asserts connectivity."""
    families=np.asarray(families,np.int64);lo=np.asarray(lo);hi=np.asarray(hi)
    pairs=set()
    for fid in np.unique(families):
        ids=np.flatnonzero(families==fid);ids=ids[np.argsort(lo[ids],kind='stable')]
        pairs.update(zip(ids[:-1].tolist(),ids[1:].tolist()))
    starts=np.asarray([positions(domain['families'][int(f)],q) for f,q in zip(families,lo)]).reshape(-1,3)
    ends=np.asarray([positions(domain['families'][int(f)],q) for f,q in zip(families,hi)]).reshape(-1,3)
    if len(starts):
        tree=cKDTree(starts)
        for i,candidates in enumerate(tree.query_ball_point(ends,6.)):
            for j in candidates:
                if i!=j:pairs.add((i,j))
            if len(pairs)>limit:raise RuntimeError('Whole workpiece failed: path candidate budget exceeded')
    pairs=np.asarray(sorted(pairs),np.int64).reshape(-1,2)
    if not len(pairs):return pairs,np.empty((0,6),np.float32)
    i,j=pairs.T;delta=starts[j]-ends[i];same=families[i]==families[j]
    x=np.c_[delta/1000,np.linalg.norm(delta,axis=1)/1000,same,np.where(same,lo[j]-hi[i],0)/1000].astype(np.float32)
    return pairs,x


def link_index(families,lo,hi,domain):
    """Complete edge domain in CSR form; no TopK or candidate truncation."""
    families=np.asarray(families,np.int64);lo=np.asarray(lo);hi=np.asarray(hi);successor={}
    for fid in np.unique(families):
        ids=np.flatnonzero(families==fid);ids=ids[np.argsort(lo[ids],kind='stable')]
        successor.update(zip(ids[:-1].tolist(),ids[1:].tolist()))
    starts=np.asarray([positions(domain['families'][int(f)],q) for f,q in zip(families,lo)]).reshape(-1,3)
    ends=np.asarray([positions(domain['families'][int(f)],q) for f,q in zip(families,hi)]).reshape(-1,3)
    offsets=[0];targets=[];tree=cKDTree(starts) if len(starts) else None
    for first in range(0,len(starts),1024):
        for local,neighbors in enumerate(tree.query_ball_point(ends[first:first+1024],6.)):
            i=first+local;row=set(neighbors);row.discard(i)
            if i in successor:row.add(successor[i])
            target=np.asarray(sorted(row),np.int32);targets.append(target);offsets.append(offsets[-1]+len(target))
    return dict(link_offsets=np.asarray(offsets,np.int64),
                link_targets=np.concatenate(targets) if targets else np.empty(0,np.int32),
                leaf_starts=starts,leaf_ends=ends)
