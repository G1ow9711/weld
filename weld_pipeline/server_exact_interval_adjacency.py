"""Input-only directed adjacency for predicted intervals sharing exact endpoints."""
import numpy as np


def interval_adjacency(profile_ids, roles, intervals):
    starts={};intervals=np.asarray(intervals,np.float64);edges=[]
    for i,(pid,role,(lo,hi)) in enumerate(zip(profile_ids,roles,intervals)):
        if np.isfinite([lo,hi]).all() and hi>lo:
            starts.setdefault((int(pid),int(role),float(lo)),[]).append(i)
    for i,(pid,role,(lo,hi)) in enumerate(zip(profile_ids,roles,intervals)):
        if not np.isfinite([lo,hi]).all() or hi<=lo:continue
        for j in starts.get((int(pid),int(role),float(hi)),[]):edges.append((i,j))
    return np.asarray(edges,np.int64).reshape(-1,2)
