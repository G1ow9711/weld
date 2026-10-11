"""Original certified occupied intervals, swept directly into member CSR."""
from collections import defaultdict
import shutil
import numpy as np
from weld_pipeline.sjg_request_competition_1125 import interval_projection


def atoms(intervals,ports=()):
    events=defaultdict(lambda:[[],[]])
    for port in ports:events[float(port)]
    for lo,hi,node in intervals:
        if hi>lo:
            events[float(lo)][1].append(int(node));events[float(hi)][0].append(int(node))
    active=defaultdict(int);coordinates=sorted(events)
    for lo,hi in zip(coordinates[:-1],coordinates[1:]):
        ending,starting=events[lo]
        for node in ending:
            active[node]-=1;assert active[node]>=0
        for node in starting:active[node]+=1
        members=np.fromiter(sorted(i for i,count in active.items() if count),np.int32)
        if len(members)>1 and hi>lo:yield lo,hi,members


def compact_resources(nodes,graph,frozen,folder=None):
    lookup={pid:i for i,r in enumerate(frozen['requests']) for pid in r['members']}
    intervals=defaultdict(list);ports=defaultdict(set);port_seen=set();local=defaultdict(list)
    for index,node in enumerate(nodes):
        fragment=int(node['fragment']);pid=graph.profile(fragment);curve=graph.curve(pid);side=int(node['side'])
        if pid in lookup:
            group=lookup[pid];request=frozen['requests'][group];resource=(group,side)
            direction=np.asarray(request['direction']);origin=np.asarray(request['origin'])
            pair=curve.at(np.asarray([0.,curve.length]));shift=float((pair[0]-origin)@direction)
            slope=float((pair[1]-pair[0])@direction/curve.length);assert slope>0
            profile=graph.profiles[pid]
            if (pid,side) not in port_seen:
                for name in ('support_intervals_mm','unobserved_gaps_mm'):
                    for lo,hi in profile.get(name,[]):ports[resource].update((shift+slope*lo,shift+slope*hi))
                ports[resource].update((shift,shift+slope*curve.length));port_seen.add((pid,side))
            for lo,hi in profile.get('support_intervals_mm',[]):
                left,right=max(float(lo),node['lo']),min(float(hi),node['hi'])
                if right>left:
                    lower=interval_projection(curve,left,origin,direction)[1]
                    upper=interval_projection(curve,right,origin,direction)[0]
                    if upper>lower:intervals[resource].append((lower,upper,index))
        domain=graph.spans[fragment,:2];pieces=[(float(node['lo']),float(node['hi']))]
        if node.get('closed'):pieces=[tuple(domain)]
        elif node.get('wrap'):pieces=[(float(domain[0]),float(node['lo'])),(float(node['hi']),float(domain[1]))]
        for left,right in pieces:
            if not right>left:continue
            if curve.kind=='line':
                lower=interval_projection(curve,left,curve.ends[0],curve.direction)[1]
                upper=interval_projection(curve,right,curve.ends[0],curve.direction)[0]
            else:
                allowance=64*np.finfo(float).eps*max(1.,curve.length);lower=left+allowance;upper=right-allowance
            if upper>lower:local[fragment,side].append((lower,upper,index))
    groups=[('local_fragment_'+str(f),s,v,()) for (f,s),v in local.items()]
    groups.extend((g,s,v,ports[g,s]) for (g,s),v in intervals.items())
    rows=members=0
    for _,_,values,p in groups:
        for lo,hi,ids in atoms(values,p):rows+=1;members+=len(ids)
    required=(rows+1)*8+members*4+rows*20
    if folder is not None and shutil.disk_usage(folder).free-required-1024*1024<5368709120:
        raise RuntimeError('Full resource CSR would cross original disk floor')
    def allocate(name,shape,dtype):
        if folder is None:return np.empty(shape,dtype)
        return np.lib.format.open_memmap(folder/('resources_'+name+'.npy'),mode='w+',dtype=dtype,shape=shape)
    offsets=allocate('offsets',(rows+1,),np.int64);indices=allocate('indices',(members,),np.int32)
    bounds=allocate('bounds',(rows,2),np.float64);group_ids=allocate('group_ids',(rows,),np.int32)
    keys=[dict(request=g,side=s) for g,s,_,_ in groups]
    row=at=0;offsets[0]=0
    for gi,(_,_,values,p) in enumerate(groups):
        for lo,hi,ids in atoms(values,p):
            indices[at:at+len(ids)]=ids;at+=len(ids);offsets[row+1]=at
            bounds[row]=lo,hi;group_ids[row]=gi;row+=1
    assert row==rows and at==members
    if folder is not None:
        for array in (offsets,indices,bounds,group_ids):array.flush()
    return dict(offsets=offsets,indices=indices,bounds=bounds,group_ids=group_ids,keys=keys)


def resource_rows(resources):
    offsets=resources['offsets'];indices=resources['indices']
    for i in range(len(offsets)-1):yield indices[offsets[i]:offsets[i+1]]


def compare_legacy(resources,legacy):
    assert len(legacy)==len(resources['offsets'])-1
    for i,(members,old) in enumerate(zip(resource_rows(resources),legacy)):
        key=resources['keys'][int(resources['group_ids'][i])]
        assert members.tolist()==old['nodes'] and key['request']==old['request'] and key['side']==old['side']
        assert resources['bounds'][i,0]==old['lo'] and resources['bounds'][i,1]==old['hi']
