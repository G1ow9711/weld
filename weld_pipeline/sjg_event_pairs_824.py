"""Use learned start/end events; foreground crossings alone emit no interval."""
import numpy as np
from scipy.signal import find_peaks


def decode_event_pairs(q,probability,offset):
    spacing=float(q[1]-q[0]) if len(q)>1 else 1.
    distance=max(1,int(np.ceil(2/max(spacing,1e-6))))
    starts=find_peaks(np.r_[-1.,probability[:,1],-1.],height=.5,distance=distance)[0]-1
    ends=find_peaks(np.r_[-1.,probability[:,2],-1.],height=.5,distance=distance)[0]-1
    result=[];used=set();previous_end=-np.inf
    for s in starts:
        lo=float(np.clip(q[s]+np.clip(2*offset[s,0],-4,4),q[0],q[-1]))
        if lo<previous_end-1e-6:continue
        for e in ends:
            e=int(e)
            if e<s or e in used:continue
            hi=float(np.clip(q[e]+np.clip(2*offset[e,1],-4,4),q[0],q[-1]))
            if hi<=lo+1e-6:continue
            confidence=float(probability[s:e+1,0].mean())
            if confidence<.5:continue
            result.append(dict(lo=lo,hi=hi,foreground_confidence=confidence,
                start_event_confidence=float(probability[s,1]),end_event_confidence=float(probability[e,2])))
            used.add(e);previous_end=hi;break
    # Candidate peak pairs that cannot form a positive interval are rejected
    # before emission, never removed by a posthoc GT comparison.
    return result,[]


if __name__=='__main__':
    from weld_pipeline.sjg_free_sequence_823 import main
    main(decoder=decode_event_pairs)
