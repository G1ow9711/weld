"""Choose competing weld representations using identical finite source intervals.

No proximity-based contact identity is inferred. Only the same original input
curve and a measured overlapping native interval establish a conflict.
"""
import time
import math
import numpy as np
from . import curveonly_01a1ac7c as cad
from .curveonly_01a1ac7c_composite_v48 import ENDPOINT_MM

MIN_OVERLAP_MM=ENDPOINT_MM


def primitive_intervals(ci,a,b,packet,lengths,seen=()):
    if ci in seen:raise ValueError('Cyclic finite source provenance')
    c=packet['candidates'][ci];length=lengths[ci];p=np.asarray(c['points'])
    closed=np.linalg.norm(p[0]-p[-1])<=1e-6
    if closed:
        windows=[]
        for turn in range(math.floor(a/length),math.floor((b-1e-10)/length)+1):
            lo=max(a,turn*length)-turn*length;hi=min(b,(turn+1)*length)-turn*length
            if hi>lo+1e-10:windows.append((lo,hi))
    else:windows=[(max(0.,a),min(length,b))]
    path=c.get('source_material_path')
    if not path:return [(ci,lo,hi) for lo,hi in windows if hi>lo+1e-10]
    result=[]
    for lo,hi in windows:
        for m in path:
            aa=max(lo,m['start_mm']);bb=min(hi,m['end_mm'])
            if bb<=aa+1e-10:continue
            child=m['curve'];child_length=lengths[child]
            if abs(m['end_mm']-m['start_mm']-child_length)>1e-7*max(1.,child_length):
                raise ValueError('Finite source length mapping mismatch')
            aa-=m['start_mm'];bb-=m['start_mm']
            if m['reversed']:aa,bb=child_length-bb,child_length-aa
            result.extend(primitive_intervals(child,aa,bb,packet,lengths,seen+(ci,)))
    return result


def select(proposals,packet):
    start=time.perf_counter();lengths=[float(cad.arclength(c['points'])[-1]) for c in packet['candidates']]
    spans=[]
    for r in proposals:
        s=r['interval'];spans.append(primitive_intervals(r['candidate'],s['start_mm'],s['end_mm'],packet,lengths))
    order=sorted(range(len(proposals)),key=lambda i:(-proposals[i]['confidence'],
        -(proposals[i]['interval']['end_mm']-proposals[i]['interval']['start_mm']),i))
    kept=[];rejected=[]
    for i in order:
        r=proposals[i];conflicts=[]
        for j in kept:
            old=proposals[j]
            if r['weld_type']!=old['weld_type'] or r.get('weld_side')!=old.get('weld_side'):continue
            overlap=[]
            for ci,a,b in spans[i]:
                for cj,c,d in spans[j]:
                    if ci!=cj:continue
                    lo,hi=max(a,c),min(b,d)
                    if hi-lo>MIN_OVERLAP_MM:overlap.append(dict(primitive=ci,start_mm=lo,end_mm=hi,overlap_mm=hi-lo))
            if overlap:conflicts.append(dict(kept_proposal=j,kept_candidate=old['candidate'],kept_confidence=old['confidence'],shared_finite_intervals=overlap))
        if conflicts:
            rejected.append(dict(proposal=i,candidate=r['candidate'],confidence=r['confidence'],interval=r['interval'],
                source_primitive_intervals=spans[i],conflicts=conflicts,reason='SAME_FINITE_SOURCE_INTERVAL_COMPETING_REPRESENTATION'))
        else:kept.append(i)
    summary=dict(input_proposals=len(proposals),kept=len(kept),rejected=rejected,minimum_overlap_mm=MIN_OVERLAP_MM,
        source='INPUT_ONLY_FINITE_CURVE_PARAMETER_PROVENANCE',new_geometry_or_reference_reads=0,
        gap_intervals_and_source_points_unchanged=True,seconds=time.perf_counter()-start)
    return [proposals[i] for i in kept],summary


class SourceSelection:
    def __init__(self,base):
        self.base=base;self.last_timing=None;self.selection=None

    def predict(self,features,packet):
        proposals=self.base.predict(features,packet);out,self.selection=select(proposals,packet)
        self.last_timing=dict(self.base.last_timing,finite_source_selection=self.selection)
        return out
