"""R5 composition: R4 feature34 and model, consistent analytic ARC30 export."""
from .curveonly_01a1ac7c_runtime_v5 import *
from . import curveonly_01a1ac7c_runtime_v5 as previous
from . import curveonly_01a1ac7c_arc_publication_v6 as publication

def decode(features,packet,model,threshold=.5):
    return publication.publish(previous.decode(features,packet,model,threshold))
