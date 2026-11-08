"""Publish explicit original-point fallbacks without an uncertified ARC tag.

Coordinates, lengths, intervals, sides and process segmentation never change.
Existing analytic arcs remain byte-for-byte intact. No arc parameters are fit.
"""
import copy

KNOWN_POINT_FALLBACKS={'CONVERSION_UNCERTIFIED_ORIGINAL_GEOMETRY_RETAINED','EXACT_POLYLINE_CORNERS_RETAINED'}


def correct(decoded):
    result=copy.deepcopy(decoded);changed=[];unknown=[]
    for i,parent in enumerate(result['parents']):
        original=copy.deepcopy(parent);records=[]
        for segment in parent.get('segments',[]):
            if segment.get('geometry_type')!='arc' or 'arc_parameters' in segment:continue
            if parent.get('arc_publication_status') not in KNOWN_POINT_FALLBACKS:
                unknown.append(dict(parent_index=i,segment_index=segment.get('segment_index'),reason='UNKNOWN_ANALYTIC_OR_POINT_PROVENANCE'));continue
            records.append(dict(segment_index=segment['segment_index'],original_geometry_type='arc',original_status=segment.get('status')))
            segment['geometry_type']='bspline_polyline';segment['status']='EXPLICIT_ORIGINAL_CONTINUOUS_POINT_GEOMETRY'
        if records:
            if parent.get('circle') is not None:
                parent['uncertified_nominal_parent_circle']=copy.deepcopy(parent['circle']);parent['circle']=None
            parent['arc_publication_status']='GENERAL_CURVE_ORIGINAL_POINT_GEOMETRY'
            assert parent['points']==original['points'] and parent['interval']==original['interval']
            for a,b in zip(parent['segments'],original['segments']):
                assert a['points']==b['points'] and a['length_mm']==b['length_mm'] and a['segment_index']==b['segment_index']
                assert {k:v for k,v in a.items() if k not in ('geometry_type','status')}=={k:v for k,v in b.items() if k not in ('geometry_type','status')}
            changed.append(dict(parent_index=i,source_candidate=parent.get('source_candidate'),segments=records,
                original_status=original.get('arc_publication_status'),actual_points_lengths_intervals_and_process_fields_exact=True))
    result['point_geometry_schema_rule']=dict(changed=changed,unknown=unknown,
        existing_analytic_parameters_unchanged=True,no_new_fit_or_coordinate_or_segmentation_change=True,
        existing_point_geometry_made_explicit_not_an_accuracy_gain=True)
    return result
