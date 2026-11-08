"""Exact acceleration after bitwise certification of the literal conflict graph.

Coordinates below are solely an embedding of graph edges. They do not change
native ranges or infer material continuity, weld identity or process gaps.
"""
import types
from functools import lru_cache
import numpy as np
from . import curveonly_01a1ac7c_structural_whole_model_v324 as previous
from . import curveonly_01a1ac7c_whole_intervals_v316 as original

NAMES,SETTINGS,ACTIVE,REMOVED=previous.NAMES,previous.SETTINGS,previous.ACTIVE,previous.REMOVED
base,profile,provenance,TOL=previous.base,previous.profile,previous.provenance,previous.TOL
pairs,features,fields=previous.pairs,previous.features,previous.fields


class CertificateUnavailable(RuntimeError):pass


def interval_dp(ids,bounds,weights):
    order=sorted(ids,key=lambda i:(bounds[i][1],bounds[i][0],i));ends=np.array([bounds[i][1] for i in order])
    scores,choices=[0.],[()]
    for k,i in enumerate(order):
        j=int(np.searchsorted(ends[:k],bounds[i][0],side='right'))
        score=scores[j]+weights[i];chosen=tuple(sorted(choices[j]+(i,)))
        if score>scores[-1]+1e-12 or abs(score-scores[-1])<=1e-12 and (len(chosen),chosen)<(len(choices[-1]),choices[-1]):
            scores.append(score);choices.append(chosen)
        else:scores.append(scores[-1]);choices.append(choices[-1])
    return float(scores[-1]),choices[-1]


def linear_components(packet,lengths,records,footprints,stats,require_certificate):
    old=original.linear_components(packet,lengths,records,footprints);tolerance=provenance.MIN_OVERLAP_MM;cache={}
    def mapping(ci,offset=0.,sign=1.,seen=()):
        assert ci not in seen
        path=packet['candidates'][ci].get('source_material_path')
        if not path:return [(ci,offset,sign)]
        value=[]
        for part in path:
            origin=part['end_mm'] if part['reversed'] else part['start_mm']
            value+=mapping(part['curve'],offset+sign*origin,sign*(-1 if part['reversed'] else 1),seen+(ci,))
        return value
    def table(ci):
        if ci not in cache:
            entries=mapping(ci);keys=[p for p,o,s in entries]
            if len(keys)!=len(set(keys)):cache[ci]={};return cache[ci]
            entries.sort(key=lambda q:min(q[1],q[1]+q[2]*lengths[q[0]]));offset,value=0.,{}
            for p,unused,sign in entries:
                value[p]=(offset,sign,lengths[p]);offset+=max(0.,lengths[p]-tolerance)
            cache[ci]=value
        return cache[ci]
    def solve(ids,mask,weights,adjacency):
        primitives={ci for i in ids for ci,a,b in footprints[i]}
        for ci in sorted({records[i]['candidate'] for i in ids},key=lambda i:-lengths[i]):
            embedding=table(ci)
            if not primitives<=embedding.keys():continue
            bounds={}
            for i in ids:
                pieces=[]
                for p,a,b in footprints[i]:
                    if b-a<=tolerance:continue
                    offset,sign,length=embedding[p]
                    pieces.append((offset+a,offset+b-tolerance) if sign>0 else
                        (offset+length-b,offset+length-a-tolerance))
                if not pieces:break
                bounds[i]=(min(a for a,b in pieces),max(b for a,b in pieces))
            if len(bounds)!=len(ids):continue
            expected,active={i:0 for i in ids},[]
            for a,b,i in sorted((a,b,i) for i,(a,b) in bounds.items()):
                active=[(end,j,start) for end,j,start in active if end>a]
                for end,j,start in active:
                    if min(b,end)-max(a,start)>0.:
                        expected[i]|=1<<j;expected[j]|=1<<i
                active.append((b,i,a))
            stats['embedding_attempts']+=1
            bad={i:(expected[i]^adjacency[i])&mask for i in ids};remaining=mask;cover=[]
            while True:
                errors=[i for i in ids if remaining&(1<<i) and bad[i]&remaining]
                if not errors:break
                vertex=max(errors,key=lambda i:((bad[i]&remaining).bit_count(),-i));cover.append(vertex);remaining&=~(1<<vertex)
                if len(cover)>16:break
            if len(cover)>16:continue
            stats['max_exception_vertices']=max(stats['max_exception_vertices'],len(cover));calls=[0]
            @lru_cache(None)
            def certified(submask):
                calls[0]+=1
                if require_certificate and calls[0]>4096:raise CertificateUnavailable('Qualification subproblem budget, no approximate result')
                if not submask:return 0.,()
                errors=[i for i in cover if submask&(1<<i) and bad[i]&submask]
                if not errors:
                    subset=[i for i in ids if submask&(1<<i)]
                    assert all(expected[i]&submask==adjacency[i]&submask for i in subset)
                    stats['bit_exact_interval_subgraphs']+=1;stats['largest_interval_subgraph']=max(stats['largest_interval_subgraph'],len(subset))
                    return interval_dp(subset,bounds,weights)
                vertex=max(errors,key=lambda i:((bad[i]&submask).bit_count(),-i));bit=1<<vertex
                a,aa=certified(submask&~bit)
                b,bb=certified(submask&~bit&~adjacency[vertex]);b+=weights[vertex];bb=tuple(sorted(bb+(vertex,)))
                if b>a+1e-12 or abs(b-a)<=1e-12 and (len(bb),bb)<(len(aa),aa):return b,bb
                return a,aa
            return certified(mask)
        stats['fallback_components']+=1
        if require_certificate:raise CertificateUnavailable('No certified material interval embedding, no approximate result')
        return old(ids,mask,weights,adjacency)
    return solve


def select(proposals,packet,require_certificate=False):
    stats=dict(embedding_attempts=0,max_exception_vertices=0,bit_exact_interval_subgraphs=0,largest_interval_subgraph=0,fallback_components=0)
    namespace=dict(original.__dict__,linear_components=lambda *args:linear_components(*args,stats,require_certificate))
    result,diagnostic=types.FunctionType(original.select.__code__,namespace)(proposals,packet)
    diagnostic['material_interval_graph_certificate']=stats
    diagnostic['literal_conflicts_and_log_odds_objective_unchanged']=True
    diagnostic['score_arithmetic_tie']=1e-12
    return result,diagnostic


class WholeIntervalModel(previous.WholeIntervalModel):
    predict=types.FunctionType(original.WholeIntervalModel.predict.__code__,dict(original.__dict__,features=features,select=select))


decode,input_curves=previous.decode,previous.input_curves
