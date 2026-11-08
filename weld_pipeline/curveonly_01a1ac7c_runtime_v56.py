"""Source interval representation selection, no fitted weight changes."""
from .curveonly_01a1ac7c_runtime_v51 import *
from . import curveonly_01a1ac7c_runtime_v51 as previous
from .curveonly_01a1ac7c_source_selection_v56 import SourceSelection


def decode(features,packet,model,threshold=.5):
    proxy=SourceSelection(model);out=previous.decode(features,packet,proxy,threshold)
    out['finite_source_selection']=proxy.selection
    return out
