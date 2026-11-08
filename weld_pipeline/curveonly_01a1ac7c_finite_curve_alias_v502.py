"""Interpret explicit original-polyline fallback by its authoritative points.

The existing publisher and frozen evaluator use analytic parameters only when
they are present. An uncertified ARC tag alone does not create an analytic arc.
This changes a private conversion-check view, never the published geometry.
"""
import copy,types
from .curveonly_01a1ac7c_finite_curve_alias_v500 import *
from . import curveonly_01a1ac7c_finite_curve_alias_v500 as previous


def polyline_conversion(parent,graph):
    if any('arc_parameters' in s for s in parent['segments']):return previous.polyline_conversion(parent,graph)
    if parent.get('arc_publication_status') not in ('CONVERSION_UNCERTIFIED_ORIGINAL_GEOMETRY_RETAINED','EXACT_POLYLINE_CORNERS_RETAINED'):
        return previous.polyline_conversion(parent,graph)
    view=copy.deepcopy(parent)
    for segment in view['segments']:
        if segment['geometry_type']=='arc':segment['geometry_type']='polyline'
    proof=previous.polyline_conversion(view,graph)
    return dict(proof,actual_geometry='ORIGINAL_CONTINUOUS_POINT_POLYLINE_WITHOUT_ANALYTIC_PARAMETERS',
        diagnostic_view_only_published_tags_points_and_parameters_unchanged=True,
        no_arc_certificate_or_new_geometry_invented=True)


inspect=types.FunctionType(previous.inspect.__code__,dict(previous.__dict__,polyline_conversion=polyline_conversion))
correct=types.FunctionType(previous.correct.__code__,dict(previous.__dict__,inspect=inspect))
