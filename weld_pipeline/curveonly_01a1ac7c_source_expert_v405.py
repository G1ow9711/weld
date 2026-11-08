"""Frozen TRAIN-trained source expert on the unchanged base feature prefix.

Votes are learned should-weld hypotheses, not geometric absence certificates.
This module does not fit weights or choose a workpiece-specific configuration.
"""
import numpy as np
from . import curveonly_01a1ac7c_graph_context_v65 as graph


def classify(features,packet,model):
    width=len(graph.cad.NAMES)
    assert features.ndim==2 and features.shape[1]>=width
    context=graph.candidate_context(packet)
    pooled=np.asarray([graph.pooled(features[a:b,:width],c) for (a,b),c in zip(
        zip(packet['offsets'][:-1],packet['offsets'][1:]),context)],np.float32)
    if not len(pooled):return []
    assert pooled.shape[1]==model.count.n_features_in_
    probability=model.count.predict_proba(pooled);classes=model.count.classes_
    result=[]
    for ci,row in enumerate(probability):
        index=int(np.argmax(row));label=int(classes[index]);confidence=float(row[index])
        result.append(dict(candidate=ci,count_class=label,confidence=confidence,
            endorsed=bool(label>0 and confidence>=.5),rejected=bool(label==0 and confidence>=.5),
            base_feature_columns=width,pooled_columns=pooled.shape[1],
            frozen_original_expert_threshold=.5))
    return result
