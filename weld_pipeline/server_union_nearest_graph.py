"""Local-spacing graph augmented by symmetric union of nearest-neighbor edges."""
import numpy as np
from scipy.spatial import cKDTree
from .server_local_spacing_graph import local_spacing_graph as mutual_graph


def local_spacing_graph(points):
    vertices,edges,spacing=mutual_graph(points)
    if len(vertices)<2:return vertices,edges,spacing
    tree=cKDTree(vertices);added=[]
    for start in range(0,len(vertices),1024):
        stop=min(start+1024,len(vertices))
        neighborhoods=tree.query_ball_point(vertices[start:stop],spacing[start:stop]*(1+1e-12),workers=2)
        for i,neighbors in enumerate(neighborhoods,start):
            for j in neighbors:
                if i!=j:added.append((min(i,j),max(i,j)))
    if added:edges=np.unique(np.concatenate((edges,np.asarray(added,np.int64))),axis=0)
    return vertices,edges,spacing
