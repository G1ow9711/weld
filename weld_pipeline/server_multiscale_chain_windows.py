"""Input-index windows with dyadic edge counts and reversal-symmetric starts."""
import numpy as np


def chain_windows(chain):
    chain=np.asarray(chain,np.int64)
    if len(chain)<5:return []
    closed=chain[0]==chain[-1];vertices=chain[:-1] if closed else chain;n=len(vertices);result=[];seen=set();edges=4
    while edges+1<=n:
        size=edges+1
        if closed:
            starts=range(n)
        else:
            stride=edges//2;forward=set(range(0,n-size+1,stride));starts=sorted(forward|{n-size-s for s in forward}|{n-size})
        for start in starts:
            part=vertices[(start+np.arange(size))%n] if closed else vertices[start:start+size]
            identity=min(tuple(part),tuple(part[::-1]))
            if identity not in seen:seen.add(identity);result.append(part.copy())
        edges*=2
    return result
