"""Preserve exact polyline corners when midpoint arc subdivision cannot converge.

No proposal is dropped, no original vertex moves, and no minimum weld length is
introduced. The exceptional legacy fit is withdrawn as a circle certificate.
"""
import math
import numpy as np
from . import curveonly_01a1ac7c_ordered_spans_v235 as previous
from . import curveonly_01a1ac7c_arc_publication_v6 as publication

LEGACY_ERROR='A zero-length curve cannot have a certified circular turn'
legacy_fallback=publication._fallback


def vertex_fallback(parent):
    segments=[];records=[]
    for source_index,original in enumerate(parent['segments']):
        points=np.asarray(original['points'],float)
        assert points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all()
        # Removing identical consecutive representations changes no path extent.
        keep=np.r_[True,np.any(points[1:]!=points[:-1],axis=1)]
        unique=points[keep]
        if len(unique)<2:
            raise ValueError('Invalid original zero-extent polyline, not a short-weld filter')
        vectors=np.diff(unique,axis=0);norm=np.linalg.norm(vectors,axis=1)
        direction=vectors/norm[:,None]
        angle=np.arctan2(np.linalg.norm(np.cross(direction[:-1],direction[1:]),axis=1),
            np.sum(direction[:-1]*direction[1:],axis=1))
        cuts=np.r_[0,np.flatnonzero(angle>publication.TURN_LIMIT)+1,len(unique)-1]
        cuts=np.unique(cuts);edge_fallback=False
        pieces=[]
        for a,b in zip(cuts[:-1],cuts[1:]):
            local=unique[a:b+1]
            candidate=dict(original,points=local.tolist(),length_mm=float(previous.arclength(local)[-1]))
            try:
                parts=legacy_fallback(dict(parent,segments=[candidate]))['segments']
            except ValueError as exc:
                if str(exc)!=LEGACY_ERROR:raise
                edge_fallback=True
                parts=[dict(original,points=local[i:i+2].tolist(),
                    length_mm=float(previous.arclength(local[i:i+2])[-1]),
                    status='EXACT_SOURCE_POLYLINE_EDGE',reason='UNRESOLVABLE_CIRCULAR_FIT_REPRESENTATION')
                    for i in range(len(local)-1)]
            for part in parts:
                p=np.asarray(part['points'],float)
                if len(p)==2:
                    part=dict(part,geometry_type='bspline_polyline',status='EXACT_SOURCE_POLYLINE_EDGE',
                        reason='GEOMETRIC_REPRESENTATION_ONLY_NOT_PROCESS_CUT')
                pieces.append(part)
        assert pieces
        for p in pieces:
            p=dict(p,segment_index=len(segments),source_representation_segment=source_index)
            segments.append(p)
        records.append(dict(source_segment=source_index,original_points=len(points),
            exact_duplicate_points_removed=int((~keep).sum()),
            original_vertex_cuts=cuts[1:-1].tolist(),cut_turn_rad=angle[cuts[1:-1]-1].tolist(),
            source_length_mm=float(previous.arclength(points)[-1]),
            published_length_mm=sum(s['length_mm']for s in pieces),
            fallback_to_exact_positive_edges=edge_fallback))
    for a,b in zip(segments[:-1],segments[1:]):
        assert np.array_equal(np.asarray(a['points'])[-1],np.asarray(b['points'])[0])
    assert np.array_equal(np.asarray(segments[0]['points'])[0],np.asarray(parent['points'])[0])
    assert np.array_equal(np.asarray(segments[-1]['points'])[-1],np.asarray(parent['points'])[-1])
    return dict(parent,segments=segments,circle=None,source_circle_fit=parent.get('circle'),
        arc_publication_status='EXACT_POLYLINE_CORNERS_RETAINED',
        corner_publication=dict(records=records,reason='LEGACY_MIDPOINT_SUBDIVISION_DID_NOT_RESOLVE_VERTEX',
            original_parent_points_unchanged=True,no_prediction_or_positive_length_removed=True,
            no_new_process_cut_or_empty_interval=True,minimum_weld_length_mm=None))


def fallback(parent):
    try:
        return legacy_fallback(parent)
    except ValueError as exc:
        if str(exc)!=LEGACY_ERROR:raise
        return vertex_fallback(parent)


def decode(features,packet,model,threshold=.5):
    # Worker processes decode one item at a time. Restore the original module
    # binding even if inference/export fails; never edit the frozen source file.
    old=publication._fallback
    publication._fallback=fallback
    try:
        return previous.decode(features,packet,model,threshold)
    finally:
        publication._fallback=old
