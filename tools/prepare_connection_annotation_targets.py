"""Conservative annotation-relation targets on frozen candidate edges.

Unknown=-1; these targets describe reference assembly compatibility, not physical
absence. This stage does not train or filter any runtime candidate.
"""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest

AUDIT=ROOT/'server_rebuild/connection_fragment_ownership_audit_v1'
DEST=ROOT/'server_rebuild/connection_annotation_targets_v1'
KEYS=('21','BM79A_native_local','10','25','36')

def classify(a,b,owner,child,start,end,length,counts,closed,gaps):
    known=(owner[a]>=0)&(owner[b]>=0)
    y=np.full(len(a),-1,np.int8)
    y[known&(owner[a]!=owner[b])]=0
    same=known&(owner[a]==owner[b]);direction=np.sign(end-start)
    aligned=direction[a]==direction[b]
    # Numerical continuity only; neither an engineering gap nor contact proof.
    tol=64*np.finfo(float).eps*np.maximum(1,np.maximum(length[a],length[b]))
    touch=np.abs(end[a]-start[b])<=tol
    y[same&(child[a]==child[b])&aligned&touch]=1
    for j in np.flatnonzero(same&(child[a]!=child[b])&aligned):
        u,v=int(a[j]),int(b[j]);p=int(owner[u]);ca,cb=int(child[u]),int(child[v]);forward=direction[u]>0
        adjacent=(cb==ca+1) if forward else (ca==cb+1)
        wrap=bool(closed[p] and ((ca==counts[p]-1 and cb==0) if forward else (ca==0 and cb==counts[p]-1)))
        if not adjacent and not wrap:
            y[j]=0;continue
        # Fragment must reach the corresponding annotated child boundaries.
        terminal=length[u] if forward else 0.
        initial=0. if forward else length[v]
        if abs(end[u]-terminal)>3. or abs(start[v]-initial)>3.:continue
        kind='SHARED' if wrap else gaps[p][min(ca,cb)]
        if kind in ('SHARED','POSITIVE_GAP'):y[j]=2 if kind=='SHARED' else 3
    return y

def controls():
    # Forward/reverse chain, overlapping fragment, cross-parent and unknown.
    owner=np.array([0,0,0,1,-1,0,0]);child=np.zeros(7,int)
    start=np.array([0,5,4,0,0,10,5.]);end=np.array([5,10,9,5,5,5,0.]);length=np.full(7,10.)
    a=np.array([0,0,0,0,5]);b=np.array([1,2,3,4,6])
    assert classify(a,b,owner,child,start,end,length,[1,1],[False,False],[[],[]]).tolist()==[1,-1,0,-1,1]
    owner=np.zeros(4,int);child=np.array([0,1,1,0]);start=np.array([0,0,10,10.]);end=np.array([10,10,0,0.]);length=np.full(4,10.)
    assert classify(np.array([0,2]),np.array([1,3]),owner,child,start,end,length,[2],[False],[['SHARED']]).tolist()==[2,2]
    assert classify(np.array([0]),np.array([1]),owner,child,start,end,length,[2],[False],[['POSITIVE_GAP']]).tolist()==[3]

def main():
    controls();DEST.mkdir(exist_ok=False)
    write(DEST/'protocol.json',dict(keys=list(KEYS),classes={'unknown':-1,'NO_LINK':0,'SAME_CHILD':1,'SHARED':2,'POSITIVE_GAP':3},rules='Unique finite-line annotation owner only. Distinct parents/nonadjacent children NO_LINK. SAME_CHILD requires same direction and numerical projected continuity (64eps*reference length). Adjacent children require same direction, each boundary within original 3mm tolerance, reference parent gap class. Ambiguous/unmatched/arcs unknown.',scope='Annotation assembly compatibility, not physical absence; no runtime GT lookup. Raw targets retained even when runtime legal class is false.',controls_pass=True,fit_calls=0))
    summaries=[]
    for key in KEYS:
        ap=AUDIT/(key+'.json');audit=read(ap)
        for rel,sha in audit['bindings'].items():assert digest(ROOT/rel)==sha
        target=read(ROOT/'server_rebuild/labels_v1'/key/'targets.json');assert target['role']=='TRAIN'
        rows=audit['rows'];n=len(rows);owner=np.full(n,-1,int);child=owner.copy();start=np.zeros(n);end=np.zeros(n);length=np.zeros(n)
        for i,r in enumerate(rows):
            if len(r['owners'])!=1:continue
            o=r['owners'][0];owner[i]=o['parent'];child[i]=o['child'];start[i],end[i]=o['projected_endpoints_mm'];length[i]=o['reference_length_mm']
        parents=target['parents'];counts=[len(p['children']) for p in parents];closed=[p['is_closed'] for p in parents];gaps=[p['gaps'] for p in parents]
        folder=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime'/('source_'+key)/'run379'/('test_'+key)
        m=audit['summary']['edges'];edges=np.memmap(folder/'edges.i32',dtype='<i4',mode='r',shape=(m,2));legal=np.memmap(folder/'legal.u8',dtype='u1',mode='r',shape=(m,3))
        labels=np.empty(m,np.int8);invalid=np.zeros(4,np.int64)
        for first in range(0,m,65536):
            a,b=edges[first:first+65536].T;y=classify(a,b,owner,child,start,end,length,counts,closed,gaps);labels[first:first+len(a)]=y
            for k in (1,2,3):invalid[k]+=np.sum((y==k)&~legal[first:first+len(a),k-1].astype(bool))
        dest=DEST/key;dest.mkdir();np.save(dest/'labels.npy',labels)
        summary=dict(source=key,edges=m,label_counts={str(k):int((labels==k).sum()) for k in (-1,0,1,2,3)},positive_but_illegal_counts=invalid.tolist(),labels_sha256=digest(dest/'labels.npy'),ownership_sha256=digest(ap),edges_sha256=digest(folder/'edges.i32'))
        write(dest/'receipt.json',summary);summaries.append(summary);print(summary,flush=True)
        edges._mmap.close();legal._mmap.close()
    write(DEST/'receipt.json',dict(status='ANNOTATION_TARGETS_READY',cases=summaries,fit_calls=0,protected_DEV_reads=0,promoted=False,scope='Conservative line supervision; arcs and ambiguous nodes masked; frozen upstream not independent cross-validation.'))

if __name__=='__main__':main()
