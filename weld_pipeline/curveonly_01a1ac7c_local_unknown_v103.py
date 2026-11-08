"""Preserve unknown owner spans in nominal composite paths.

Only the availability handling changes from V94. Known-owner numerical code
is reused verbatim, and no curve or source interval is deleted or declared
unweldable. A nominal clearance's owner is allowed to be unknown.
"""
import inspect
from . import curveonly_01a1ac7c_local_material_v94 as previous
NAMES=previous.NAMES
DEFLECTION_MM=previous.DEFLECTION_MM
closest=previous.closest
slog=previous.slog


def build(packet,solids):
    source=inspect.getsource(previous.build)
    old="if np.any(owners<0)or np.any(owners>=len(solids)):raise ValueError('Qualified source has unresolved owner')"
    assert source.count(old)==1
    source=source.replace(old,'valid_owner=(owners>=0)&(owners<len(solids))')
    old='(owners!=mi)&(lower[:,mi]<=second+1e-7)'
    assert source.count(old)==1
    source=source.replace(old,'valid_owner&(owners!=mi)&(lower[:,mi]<=second+1e-7)')
    namespace=dict(previous.__dict__);exec(compile(source,__file__,'exec'),namespace)
    values,receipt=namespace['build'](packet,solids)
    receipt=dict(receipt,unknown_owner_policy='Keep all original source geometry and raw observations; material roles unavailable only on unresolved-owner stations.',
        unknown_material_rows=int((values[:,0]==0).sum()),known_owner_computation_unchanged_from_v94=True,
        no_unknown_to_nonweld_conversion=True)
    return values,receipt
