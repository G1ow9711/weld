"""R16: frozen ordered-set model and source, bounded local-arc publication."""
from .curveonly_01a1ac7c_runtime_v22 import *
from . import curveonly_01a1ac7c_runtime_v22 as previous
from . import curveonly_01a1ac7c_arc_publication_v24 as publication


def decode(features,packet,model,threshold=.5):
    return publication.publish(previous.decode(features,packet,model,threshold))
