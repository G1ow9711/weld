"""Use the same point-only context for horizontal and vertical line contacts."""
import numpy as np

from weld_pipeline.point_plate_context import PlateContext

VERSION='fresh_all_straight_contact_context_v1'


def point_structure(xyz,sequences):
    context=PlateContext(xyz)
    structural=np.zeros((len(sequences),50),np.float32)
    process=np.zeros((len(sequences),6),np.float32)
    for index,sequence in enumerate(sequences):
        if sequence.get('curve_kind')!='arc':
            structural[index],process[index],_=context.describe(sequence)
    return structural,process
