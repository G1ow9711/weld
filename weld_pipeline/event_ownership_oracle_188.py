"""Offline original-target ownership diagnostics; never an inference label lookup."""
import numpy as np

from weld_pipeline.event_gap_decoder import event_peaks
from weld_pipeline.joint_event_votes import dense_targets
from weld_pipeline.continuous_center_modes_v2 import RESOLUTION_MM

VERSION = 'original_generator_ownership_oracle_188_v1'
O, B, I, UNKNOWN = 0, 1, 2, -1


def trace_original_owners(original, old_targets, old_known, near_targets, near_known, step):
    """Replay 151 then156 exactly; identity is the original extracted peak ordinal."""
    original=np.asarray(original,np.float32)
    if original.ndim!=2 or original.shape[1]!=4 or step!=2.:
        raise ValueError('Original151 canonical four-head labels use the fixed2mm grid')
    dense=dense_targets(original,step)
    valid=np.c_[original>=0.,(original[:,1:4]>=0.)&(dense[:,1:4]>.5)]
    generated151=np.where(valid,dense,0.).astype(np.float32)
    if not np.array_equal(generated151,old_targets) or not np.array_equal(valid.astype(np.float32),old_known):
        raise ValueError('Original151 frozen labels/known did not replay exactly')
    near=(np.asarray(old_known[:,4:7])>.5)&(np.abs(np.asarray(old_targets[:,4:7])*64.)<=16.+1e-6)
    generated156=np.zeros_like(old_targets); generated_known=np.zeros_like(old_known)
    generated156[:,0]=old_targets[:,0]; generated156[:,1:4]=near
    generated156[:,4:7]=np.where(near,old_targets[:,4:7]*4.,0.)
    generated_known[:,:4]=old_known[:,:4]; generated_known[:,4:7]=near
    if not np.array_equal(generated156,near_targets) or not np.array_equal(generated_known,near_known):
        raise ValueError('Original156 frozen labels/known did not replay exactly')
    q=np.arange(len(original),dtype=float)*step
    owner=np.where(np.asarray(near_known[:,1:4])>0,-1,-2).astype(np.int32)
    peaks=[]; tie_counts=[]
    for role in range(3):
        events=np.asarray([p*step for p,_ in event_peaks(np.maximum(original[:,role+1],0.),step,.5)],float)
        peaks.append(events)
        if not len(events):
            tie_counts.append(0); continue
        distances=events[None,:]-q[:,None]
        nearest=np.argmin(np.abs(distances),axis=1)
        minimum=np.abs(distances)[np.arange(len(q)),nearest]
        tie_counts.append(int(np.sum(np.sum(np.abs(distances)==minimum[:,None],axis=1)>1)))
        expected=(distances[np.arange(len(q)),nearest]/64.).astype(np.float32)
        supervised=np.asarray(old_known[:,role+4])>0
        if not np.array_equal(expected[supervised],np.asarray(old_targets[:,role+4])[supervised]):
            raise ValueError('Original argmin source event differs from151 supervision')
        positive=(np.asarray(near_known[:,role+4])>0)&(np.asarray(near_targets[:,role+1])>.5)
        owner[positive,role]=nearest[positive]
    return owner,peaks,tie_counts


def encode_obi(owner):
    owner=np.asarray(owner,np.int32)
    if owner.ndim!=2 or owner.shape[1]!=3 or np.any(owner<-2):
        raise ValueError('Three roles with known O=-1 and unknown=-2 required')
    labels=np.full(owner.shape,UNKNOWN,np.int8)
    for role in range(3):
        active=-1
        for index,identity in enumerate(owner[:,role]):
            if identity==-2:
                continue  # Unknown neither becomes O nor closes a known block.
            if identity==-1:
                labels[index,role]=O; active=-1
            else:
                labels[index,role]=I if identity==active else B
                active=int(identity)
    return labels


def decode_obi(labels):
    labels=np.asarray(labels,np.int8)
    if labels.ndim!=2 or labels.shape[1]!=3 or np.any(~np.isin(labels,(UNKNOWN,O,B,I))):
        raise ValueError('Three O/B/I roles with explicit unknown masks required')
    blocks=[]; membership=np.full(labels.shape,-1,np.int32)
    membership[labels==UNKNOWN]=-2
    for role in range(3):
        role_blocks=[]; active=None
        for index,value in enumerate(labels[:,role]):
            if value==UNKNOWN:
                continue
            if value==O:
                active=None
            elif value==B:
                role_blocks.append([]); active=len(role_blocks)-1
                role_blocks[active].append(index); membership[index,role]=active
            else:
                if active is None:
                    raise ValueError('I cannot start a complete query sequence')
                role_blocks[active].append(index); membership[index,role]=active
        blocks.append([np.asarray(ids,np.int64) for ids in role_blocks])
    return membership,blocks


def frozen_admission(raw,step):
    raw=np.asarray(raw,np.float32)
    if raw.ndim!=2 or raw.shape[1]!=7 or not np.isfinite(raw).all() or step<=0.:
        raise ValueError('Finite frozen175 raw fields required')
    q=np.arange(len(raw),dtype=float)*step
    centres=np.floor(q[:,None]/64.)*64.+raw[:,4:7].astype(float)*16.
    offsets=(centres-q[:,None])/16.
    support=1./(1.+np.exp(-np.clip(raw[:,1:4].astype(float),-60.,60.)))
    span=q[-1]
    support_ok=support>=.5
    radius_ok=np.abs(offsets)<=1.+1e-6
    span_ok=(centres>=-RESOLUTION_MM)&(centres<=span+RESOLUTION_MM)
    return centres,support,support_ok&radius_ok&span_ok,dict(support=support_ok,radius=radius_ok,span=span_ok)


def weighted_median(values,weights):
    values=np.asarray(values,float); weights=np.asarray(weights,float)
    if not len(values) or values.shape!=weights.shape or not np.isfinite(values).all() or np.any(weights<=0.):
        raise ValueError('Nonempty finite positive-weight observations required')
    order=np.argsort(values,kind='stable'); cumulative=np.cumsum(weights[order])
    index=int(np.searchsorted(cumulative,.5*cumulative[-1],side='left'))
    return float(values[order[index]])


def fuse_owned_block(centres,support,admitted,query_ids,role,span):
    # Query identity, not coordinates, eliminates repeated copies of one record.
    ids=np.unique(np.asarray(query_ids,np.int64))
    eligible=ids[np.asarray(admitted)[ids,role]]
    if not len(eligible):
        return None,eligible
    c=weighted_median(np.asarray(centres)[eligible,role],np.asarray(support)[eligible,role])
    confidence=float(np.mean(np.asarray(support)[eligible,role]))
    return [float(np.clip(c,0.,span)),min(confidence,1.-1e-4)],eligible
