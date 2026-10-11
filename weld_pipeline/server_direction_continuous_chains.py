"""Reversible tangent continuation on existing undirected edges only."""
import numpy as np


def trace_direction_continuous(vertices, edges):
    vertices=np.asarray(vertices,float);edges=np.asarray(edges,np.int64)
    if not len(edges):return []
    if np.any(edges[:,0]==edges[:,1]):raise ValueError('Self edges are not allowed')
    if len(np.unique(np.sort(edges,axis=1),axis=0))!=len(edges):raise ValueError('Duplicate undirected edges')
    adjacency=[[] for _ in vertices]
    for i,(a,b) in enumerate(edges):adjacency[a].append((i,int(b)));adjacency[b].append((i,int(a)))
    continuation={}
    for vertex,neighbors in enumerate(adjacency):
        if len(neighbors)<2:continue
        vectors=vertices[[n for _,n in neighbors]]-vertices[vertex];length=np.linalg.norm(vectors,axis=1)
        if np.any(length==0):raise ValueError('Zero length graph edge')
        unit=vectors/length[:,None];cosine=unit@unit.T;np.fill_diagonal(cosine,np.inf);best=np.argmin(cosine,axis=1)
        unique=[]
        for i,j in enumerate(best):
            tied=np.isclose(cosine[i],cosine[i,j],rtol=0,atol=1e-12)
            unique.append(cosine[i,j]<-1e-12 and int(tied.sum())==1)
        for i,j in enumerate(best):
            if unique[i] and unique[j] and best[j]==i:
                continuation[(vertex,neighbors[i][0])]=neighbors[j][0]
    used=np.zeros(len(edges),bool);chains=[]
    def trace(start,edge):
        path=[start];current=start
        while not used[edge]:
            used[edge]=True;a,b=edges[edge];current=int(b if current==a else a);path.append(current)
            following=continuation.get((current,edge))
            if following is None or used[following]:break
            edge=following
        chains.append(np.asarray(path,np.int64))
    for edge,(a,b) in enumerate(edges):
        if used[edge]:continue
        if (int(a),edge) not in continuation:trace(int(a),edge)
        elif (int(b),edge) not in continuation:trace(int(b),edge)
    for edge,(a,_) in enumerate(edges):
        if not used[edge]:trace(int(a),edge)
    assert used.all()
    assert sum(len(c)-1 for c in chains)==len(edges)
    return chains
