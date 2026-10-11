"""Label-free exact interval-set selection with symmetric endpoint uncertainty."""
import numpy as np


def compile_plan(rows):
    groups={}
    for i,row in enumerate(rows):groups.setdefault((row['candidate'],row['weld_type']),[]).append(i)
    result=[]
    for ids in groups.values():
        lo=np.array([min(rows[i]['start'],rows[i]['end']) for i in ids],dtype=float);hi=np.array([max(rows[i]['start'],rows[i]['end']) for i in ids],dtype=float);trim=np.minimum(1.5,.25*(hi-lo));lo+=trim;hi-=trim
        order=np.lexsort((np.array(ids),lo,hi));idx=np.array(ids,dtype=np.int64)[order];lo=lo[order];hi=hi[order];previous=np.searchsorted(hi,lo,side='right');previous=np.minimum(previous,np.arange(len(idx)))
        result.append((idx,previous))
    return result


def select(utility,plan):
    selected=[]
    for idx,previous in plan:
        values=utility[idx];best=np.zeros(len(idx)+1);take=np.zeros(len(idx),bool)
        for k,value in enumerate(values):
            score=best[previous[k]]+value
            if score>best[k]:best[k+1]=score;take[k]=True
            else:best[k+1]=best[k]
        at=len(idx)
        while at:
            k=at-1
            if take[k]:selected.append(int(idx[k]));at=int(previous[k])
            else:at-=1
    return np.array(selected,dtype=np.int64)


def controls():
    def row(a,b):return dict(candidate=0,weld_type='horizontal',start=a,end=b)
    rows=[row(0,10),row(0,20),row(10,20),row(0,1)]
    plan=compile_plan(rows);assert set(select(np.array([2.,3.,2.,-1.]),plan))=={0,2}
    assert len(select(np.full(4,-1.),plan))==0
    tiny=compile_plan([row(0,.2),row(.2,.4)]);assert len(select(np.array([1.,1.]),tiny))==2
    return dict(nonoverlap_global_optimum=True,all_negative_empty=True,submillimeter_adjacent_segments_preserved=True)
