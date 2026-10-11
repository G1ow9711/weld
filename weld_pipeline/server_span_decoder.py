"""Decode scored spans without reading targets; parent linkage still pending."""
import numpy as np
from weld_pipeline.world_set_evaluation_285 import reference_segment,distance_to_primitive


def decode(rows,scores,candidates,threshold=.5):
    chosen=[];segments=[]
    for i in np.argsort(-scores,kind='stable'):
        if scores[i]<threshold:continue
        row=rows[i];segment=reference_segment(dict(world_representation=row['representation']));ends=np.asarray(segment['points'])[[0,-1]];length=float(np.linalg.norm(np.diff(np.asarray(segment['points']),axis=0),axis=1).sum());duplicate=False
        for old,prev in zip(chosen,segments):
            if old['weld_type']!=row['weld_type'] or prev['kind']!=segment['kind']:continue
            if candidates[old['candidate']]['face_indices']!=candidates[row['candidate']]['face_indices']:continue
            oldends=np.asarray(prev['points'])[[0,-1]]
            if np.linalg.norm(ends-oldends,axis=1).max()>1:continue
            oldlen=float(np.linalg.norm(np.diff(np.asarray(prev['points']),axis=0),axis=1).sum())
            if min(length,oldlen)/max(length,oldlen)<.95:continue
            if max(distance_to_primitive(segment['points'],prev).max(),distance_to_primitive(prev['points'],segment).max())<=1:duplicate=True;break
        if duplicate:continue
        chosen.append(dict(row,score=float(scores[i]),proposal_index=int(i)));segments.append(dict(segment,weld_type=row['weld_type']))
    # Explicitly provisional singleton parents: strict parent evaluation will
    # count unresolved multi-child parents as missed rather than infer truth.
    decoded=dict(parents=[dict(child_count=1,is_closed=False,gaps=[],segments=[s]) for s in segments],invalid_predictions=[],total_prediction_children=len(segments),parent_linkage_status='UNIMPLEMENTED_SINGLETONS_ONLY')
    return chosen,decoded
