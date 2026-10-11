"""Single-checkpoint inference combining learned H and point-derived V heads."""
import numpy as np

from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.plate_candidate_model import predict_vertical
from weld_pipeline.point_plate_context import PlateContext
from weld_pipeline.typed_contact_model import infer_probabilities

MODEL_NAME='full_available_fit_hybrid_v1'


def infer_from_sequences(state,model,sequences,structural,process):
    horizontal=[p for p in decode_intervals(sequences,infer_probabilities(model,sequences),threshold=.5,
                    weld_penalty=2.,mask_weight=.25,event_gap_mm=0.) if p['type']=='horizontal']
    vertical,diagnostic=predict_vertical(state['plate_heads'],sequences,structural,process)
    return horizontal+vertical,diagnostic


def point_structure(xyz,sequences):
    context=PlateContext(xyz)
    structural=np.zeros((len(sequences),50),np.float32)
    process=np.zeros((len(sequences),6),np.float32)
    for i,seq in enumerate(sequences):
        if seq['type']=='vertical':
            a,b,diagnostic=context.describe(seq)
            structural[i],process[i]=a,b
    return structural,process
