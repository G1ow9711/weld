"""Offline TRAIN negative exclusion for original unresolved GT regions.

A disjoint bounding box proves a candidate cannot touch any missing child
within 3mm. Overlap remains unknown. This never admits a positive, changes
GT, alters the acceptance tolerance, or creates an inference candidate.
"""
import numpy as np
from weld_pipeline.world_set_evaluation_285 import reference_segment


def known_background_by_candidate(metadata,targets,representation_misses):
    by_id={c['original_child_id']:c for p in targets['parents'] for c in p['children']}
    regions=[]
    for item in representation_misses:
        for cid in item['original_child_ids']:
            child=by_id[cid];points=np.asarray(reference_segment(child)['points'],float)
            # Lines have exact endpoint bounds. Sampled arc bounds receive
            # the existing <=0.1mm sagitta enclosure, only for exclusion.
            margin=3. if child['original_kind']=='line' else 3.1
            regions.append((child['weld_type'],points.min(0)-margin,points.max(0)+margin))
    known=[]
    for item in metadata:
        relevant=[(a,z) for typ,a,z in regions if typ==item['type']]
        if not relevant:known.append(True);continue
        if item['kind']!='line':known.append(False);continue
        origin=np.asarray(item['origin']);end=origin+float(item['span'])*np.asarray(item['direction'])
        low=np.minimum(origin,end);high=np.maximum(origin,end)
        possible=any(bool(np.all(low<=z)&np.all(high>=a)) for a,z in relevant)
        known.append(not possible)
    return known
