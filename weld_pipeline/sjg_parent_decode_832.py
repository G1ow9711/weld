"""Assemble oriented interval ports into paths; no truth or CAD access."""
import numpy as np


def sampled_segment(points,weld_type):
    p=np.asarray(points);length=np.linalg.norm(p[1]-p[0])
    return dict(kind='line',weld_type=weld_type,
        points=np.linspace(p[0],p[1],max(2,int(np.ceil(length))+1)).tolist())


def assemble(children,pairs,node_probability,edge_probability):
    keep=set(np.flatnonzero(node_probability>=.5).tolist());connection={};grouped={}
    for row,probability in zip(pairs,edge_probability):
        i,oi,j,oj=map(int,row)
        if i not in keep or j not in keep:continue
        a,b=(i,1-oi),(j,oj);key=tuple(sorted((a,b)))
        grouped.setdefault(key,[]).append(probability)
    candidates=[(float(np.mean(p,axis=0)[0]),float(np.mean(p,axis=0)[1]),key) for key,p in grouped.items()]
    candidates.sort(key=lambda x:(-x[0],x[2]));accepted=[];closed_seeds=[]
    parent={i:i for i in keep}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for probability,shared_probability,(a,b) in candidates:
        if probability<.5:break
        if a in connection or b in connection:continue
        gap=np.linalg.norm(np.asarray(children[a[0]]['points'][a[1]])-children[b[0]]['points'][b[1]])
        shared=bool(shared_probability>=.5 and gap<=6.)
        ra,rb=root(a[0]),root(b[0]);cycle=ra==rb
        if cycle and not shared:continue
        connection[a]=(b,shared,probability);connection[b]=(a,shared,probability)
        if cycle:closed_seeds.append(a)
        else:parent[ra]=rb
        accepted.append(dict(first=list(a),second=list(b),probability=probability,shared=shared,
            shared_probability=shared_probability,endpoint_gap_mm=float(gap),closes_cycle=cycle))
    covered=set();parents=[]
    starts=sorted((i,port) for i in keep for port in (0,1) if (i,port) not in connection)
    starts.extend(closed_seeds)
    for initial in starts:
        if initial[0] in covered:continue
        current=initial;segments=[];ids=[];links=[];closed=False
        while True:
            i,entry=current
            if i in covered:
                assert i==initial[0];closed=True;break
            covered.add(i);ids.append(i)
            points=np.asarray(children[i]['points'],float)
            if entry:points=points[::-1].copy()
            segments.append(dict(points=points,weld_type=children[i]['weld_type']))
            exit_port=(i,1-entry)
            if exit_port not in connection:break
            other,shared,probability=connection[exit_port];links.append((shared,probability));current=other
        for k,(shared,probability) in enumerate(links):
            if not shared:continue
            next_index=(k+1)%len(segments)
            coordinate=(segments[k]['points'][1]+segments[next_index]['points'][0])/2
            segments[k]['points'][1]=coordinate;segments[next_index]['points'][0]=coordinate
        gaps=['SHARED' if shared else 'POSITIVE_GAP' for shared,_ in links[:len(segments)-1]]
        confidence=float(np.mean([node_probability[i] for i in ids]+[p for _,p in links]))
        parents.append(dict(child_count=len(ids),segments=[sampled_segment(s['points'],s['weld_type']) for s in segments],
            gaps=gaps,is_closed=closed,source_child_ids=ids,confidence=confidence,material_pair=None))
    assert covered==keep,'Every retained interval must appear in exactly one path'
    return parents,dict(kept_children=sorted(keep),rejected_children=sorted(set(range(len(children)))-keep),accepted_links=accepted)


def parent_equivalent(a,b,tolerance=1.):
    if a['child_count']!=b['child_count'] or a['is_closed']!=b['is_closed']:return False
    ap=np.asarray([[s['points'][0],s['points'][-1]] for s in a['segments']])
    at=[s['weld_type'] for s in a['segments']]
    bp=np.asarray([[s['points'][0],s['points'][-1]] for s in b['segments']]);bt=[s['weld_type'] for s in b['segments']]
    for reverse in (False,True):
        points=bp[::-1,::-1] if reverse else bp;types=bt[::-1] if reverse else bt
        gaps=b['gaps'][::-1] if reverse else b['gaps']
        shifts=range(len(points)) if a['is_closed'] else (0,)
        for shift in shifts:
            shifted=np.roll(points,-shift,axis=0);ts=types[shift:]+types[:shift]
            gs=gaps
            if shift:gs=(gaps+['SHARED'])[shift:]+(gaps+['SHARED'])[:shift];gs=gs[:-1]
            if at==ts and a['gaps']==gs and np.linalg.norm(ap-shifted,axis=-1).max()<=tolerance:return True
    return False


def suppress_parent_hypotheses(parents,tolerance=1.):
    kept=[];suppressed=[]
    for index in sorted(range(len(parents)),key=lambda i:(-parents[i]['confidence'],i)):
        duplicate=next((j for j in kept if parent_equivalent(parents[index],parents[j],tolerance)),None)
        if duplicate is None:kept.append(index)
        else:suppressed.append(dict(parent=index,retained_parent=duplicate,maximum_endpoint_tolerance_mm=tolerance))
    return [parents[i] for i in kept],suppressed


def decoded(parents):
    return dict(parents=parents,invalid_predictions=[],total_prediction_children=sum(p['child_count'] for p in parents))
