"""Pure candidate features and frozen-teacher priors for incremental heads.

No source IDs, filenames, references, targets, STEP or cache readers. Arrays
are produced in the current raw run; offline TRAIN may use its sealed copy.
"""
import numpy as np
from weld_pipeline.candidate_tree_model import descriptor
from weld_pipeline.event_gap_decoder import select_intervals
from weld_pipeline.constrained_decoder import refined_peaks

VERSION='PURE_XYZ_FROZEN_TEACHER_INCREMENTAL_CANDIDATE_FEATURES_v1'
def logit(probability):
 p=float(np.clip(probability,1e-6,1.-1e-6));return np.log(p)-np.log1p(-p)

def prepare_candidate_inputs(sequences,probabilities,structural,process,vertical_diagnostic,
                           split_fractions,parent_capacity,child_capacity):
 assert len(sequences)==len(probabilities)==len(structural)==len(process)
 assert parent_capacity>=1 and child_capacity>=1
 count=len(sequences);q=parent_capacity;c=child_capacity
 base=dict(presence_logits=np.full((count,q),-6.,np.float32),
  count_logits=np.full((count,q,c+1),-6.,np.float32),positions_mm=np.zeros((count,q,c,2),np.float32),
  gap_logits=np.full((count,q,max(0,c-1)),-6.,np.float32),
  direction_logits=np.full((count,q),6.,np.float32),closed_logits=np.full((count,q),-6.,np.float32),
  full_candidate_domain_mm=np.zeros((count,2),np.float32))
 diag={int(row['sequence']):row for row in vertical_diagnostic};features=[];overflow=[]
 for si,seq in enumerate(sequences):
  span=(len(seq['xyz'])-1)*float(seq['sample_step_mm']);assert span>0
  base['full_candidate_domain_mm'][si]=[0.,span]
  base['count_logits'][si,:,1]=6.
  base['positions_mm'][si,:,:]=np.stack([np.arange(c)*span/c,np.arange(1,c+1)*span/c],axis=1)
  base['positions_mm'][si,:,0]=[0.,span]
  p=np.asarray(probabilities[si],np.float32);assert p.shape==(len(seq['xyz']),7)
  teacher=[];tree=np.zeros(6,np.float32)
  if seq['type']=='vertical':
   row=diag[si];n=int(row['segment_count']);assert n in (1,3)
   start=float(row['margins_mm'][0]);end=span-float(row['margins_mm'][1]);confidence=float(row['foreground_probability'])
   cuts=start+np.asarray(split_fractions[str(n)],float)*(end-start)
   assert len(cuts)==n+1
   teacher=[dict(confidence=confidence,segments=np.stack([cuts[:-1],cuts[1:]],axis=1))]
   tree=np.array([confidence,start/1000.,(span-end)/1000.,row['side_split_probability'],n/16.,span/1000.],np.float32)
  else:
   step=float(seq['sample_step_mm']);internal=refined_peaks(p[:,3],step,.5)
   for a,z,pa,pz in select_intervals(p,step,.5,2.,.25,0.):
    cuts=np.array([a]+[v for v,height in internal if a+1.5/step<v<z-1.5/step]+[z])*step
    teacher.append(dict(confidence=min(pa,pz),segments=np.stack([cuts[:-1],cuts[1:]],axis=1)))
  for query,parent in enumerate(teacher):
   n=len(parent['segments'])
   if query>=q or n>c:
    overflow.append(dict(sequence_index=si,frozen_teacher_parent_query=query,declared_children=n,
     reason='REGISTERED_INCREMENTAL_CAPACITY_SMALLER_THAN_FROZEN_TEACHER_OUTPUT'))
    continue
   base['presence_logits'][si,query]=logit(parent['confidence'])
   base['count_logits'][si,query,:]=-6.;base['count_logits'][si,query,n]=6.
   base['positions_mm'][si,query,:n]=parent['segments']
  features.append(np.r_[descriptor(seq),np.asarray(structural[si]),np.asarray(process[si]),
   p.mean(axis=0),p.max(axis=0),tree].astype(np.float32))
 if not features:return np.empty((0,1724),np.float32),base,overflow
 x=np.stack(features);assert np.isfinite(x).all() and x.shape[1]==1724
 return x,base,overflow
