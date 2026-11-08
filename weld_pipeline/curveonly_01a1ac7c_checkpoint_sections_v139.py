"""Resumable exact finite-solid queries; keep V124 query points and features."""
import hashlib
import json
import os
import time
import types
from pathlib import Path
import numpy as np
from . import curveonly_01a1ac7c_material_sections_v124 as original


class InjectedCheckpointStop(RuntimeError):
    pass


def array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)


def atomic_array(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb')as f:np.save(f,value)
    os.replace(temp,path)


class CheckpointProbe(original.SolidProbe):
    def __init__(self,solids,folder,context,abort_after_calls=None):
        super().__init__(solids)
        self.folder=Path(folder);self.context=context;self.abort_after_calls=abort_after_calls
        self.actual_calls=0;self.next_candidate=0;self.active_folder=None;self.active_signature=None
        self.reused_candidates=0;self.loaded_cursors=0;self.last_save=0.

    def save(self,inside,on,own,available,mi,index,completed_in_call,signature):
        path=self.active_folder/'QUERY_STATE.npz';temp=path.with_name(path.name+'.tmp')
        with temp.open('wb')as f:
            np.savez_compressed(f,inside=inside,on=on,owner_occupied=own,available=available,
                material=np.int64(mi),next_query=np.int64(index),completed_in_call=np.int64(completed_in_call),
                signature=np.asarray(signature))
        os.replace(temp,path);self.last_save=time.monotonic()
        atomic_json(self.folder/'PROGRESS.json',dict(candidate=self.next_candidate-1,
            next_material=mi,next_query=index,logical_classifier_calls=self.calls,
            fresh_classifier_calls=self.actual_calls,checkpoint=str(path),epoch=time.time()))

    def classify(self,points,owners):
        points=np.asarray(points,float);owners=np.asarray(owners,int)
        assert points.shape==(len(owners),3)
        signature=value_hash(dict(source=self.active_signature,points=array_hash(points),owners=array_hash(owners),
            boxes=array_hash(self.boxes),tolerance=original.CLASSIFY_TOLERANCE_MM))
        n=len(points);inside=np.zeros(n,np.int16);on=np.zeros(n,np.int16)
        own=np.zeros(n,bool);available=np.ones(n,bool);mi_start=0;query_start=0;completed=0
        path=self.active_folder/'QUERY_STATE.npz'
        if path.exists():
            with np.load(path,allow_pickle=False)as state:
                assert str(state['signature'])==signature,'Checkpoint query context differs'
                inside=state['inside'].copy();on=state['on'].copy();own=state['owner_occupied'].copy();available=state['available'].copy()
                mi_start=int(state['material']);query_start=int(state['next_query']);completed=int(state['completed_in_call'])
            assert all(q.shape==(n,)for q in (inside,on,own,available))and 0<=mi_start<=len(self.solids)
            self.calls+=completed;self.loaded_cursors+=1
        base_calls=self.calls-completed
        for mi,(box,classifier)in enumerate(zip(self.boxes,self.classifiers)):
            if mi<mi_start:continue
            indices=np.flatnonzero(np.all((points>=box[:3]-original.CLASSIFY_TOLERANCE_MM)
                                         &(points<=box[3:]+original.CLASSIFY_TOLERANCE_MM),axis=1))
            start=query_start if mi==mi_start else 0;assert 0<=start<=len(indices)
            for qi in range(start,len(indices)):
                index=indices[qi]
                classifier.Perform(original.cad.gp_Pnt(*map(float,points[index])),original.CLASSIFY_TOLERANCE_MM)
                state=classifier.State();self.calls+=1;self.actual_calls+=1
                if state==original.TopAbs_IN:inside[index]+=1
                elif state==original.TopAbs_ON:on[index]+=1
                elif state!=original.TopAbs_OUT:available[index]=False
                if mi==owners[index]and state in (original.TopAbs_IN,original.TopAbs_ON):own[index]=True
                next_mi,next_q=(mi+1,0)if qi+1==len(indices)else(mi,qi+1)
                stop=self.abort_after_calls is not None and self.actual_calls>=self.abort_after_calls
                if stop or time.monotonic()-self.last_save>=5.:
                    self.save(inside,on,own,available,next_mi,next_q,self.calls-base_calls,signature)
                if stop:raise InjectedCheckpointStop('Intentional interruption after a complete exact query')
            self.save(inside,on,own,available,mi+1,0,self.calls-base_calls,signature)
        return dict(inside=inside,on=on,owner_occupied=own,available=available)


def section_values(points,tangent,base_axis,owners,transform,probe):
    ci=probe.next_candidate;probe.next_candidate+=1
    folder=probe.folder/f'candidate_{ci:05d}';folder.mkdir(parents=True,exist_ok=True)
    signature=value_hash(dict(context=probe.context,candidate=ci,points=array_hash(points),tangent=array_hash(tangent),
        base=array_hash(base_axis),owners=array_hash(owners),transform=array_hash(transform)))
    done=folder/'COMPLETE.json'
    if done.exists():
        data=json.loads(done.read_text());assert data['signature']==signature
        path=folder/'features.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==data['feature_sha256']
        values=np.load(path);assert values.shape==(len(points),len(original.NAMES))and values.dtype==np.float32
        probe.calls+=data['logical_calls'];probe.reused_candidates+=1
        return values
    probe.active_folder=folder;probe.active_signature=signature
    before=probe.calls
    values=original.section_values(points,tangent,base_axis,owners,transform,probe)
    atomic_array(folder/'features.npy',values)
    atomic_json(done,dict(signature=signature,logical_calls=probe.calls-before,rows=len(values),
        feature_sha256=hashlib.sha256((folder/'features.npy').read_bytes()).hexdigest(),epoch=time.time()))
    return values


def build(packet,solids,folder,context,abort_after_calls=None):
    captured={}
    def factory(items):
        captured['probe']=CheckpointProbe(items,folder,context,abort_after_calls)
        return captured['probe']
    namespace=dict(original.__dict__,SolidProbe=factory,section_values=section_values)
    values,receipt=types.FunctionType(original.build.__code__,namespace)(packet,solids)
    probe=captured.get('probe')
    receipt.update(checkpointed_exact_queries=True,
        actual_fresh_classifier_calls=probe.actual_calls if probe else 0,
        reused_candidate_blocks=probe.reused_candidates if probe else 0,
        resumed_query_cursors=probe.loaded_cursors if probe else 0,
        original_full_candidate_query_points_and_feature_math=True,
        logical_call_count_includes_reused_queries=True)
    return values,receipt
