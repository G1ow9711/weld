"""Customer label semantics clarified by diagrams on 2026-09-28.

Reference labels are training/evaluation metadata, never prediction inputs.
Keep censored or ambiguous distances out of exact-distance regression.
"""
import math
from weld_pipeline.customer_label_corrections import apply_label_corrections

SEMANTICS_VERSION='customer_process_labels_v4'
PASSAGE_SPLIT_LIMIT_MM=35.

SIDE_PLATE_FIELD='IsSplitedForDualRobot (Any)'
WALL_FIELDS=('FrontWallDistance','BackWallDistance')
EDGE_FIELDS=('StartBasePlateEdgeDistance','EndBasePlateEdgeDistance')
HEIGHT_FIELDS=('startHeight','endHeight','StartViaHoleHeight','EndViaHoleHeight')
HOLE_FIELDS=('StartViaHoleHeight','EndViaHoleHeight')


def clearance_requires_split(distance_mm):
    """Customer-confirmed rule for an identified passage clearance, not any distance."""
    if distance_mm is None:return None
    if isinstance(distance_mm,bool):raise ValueError('Boolean is not a clearance')
    distance=float(distance_mm)
    if not math.isfinite(distance) or distance<0:raise ValueError('Clearance must be finite and nonnegative')
    return distance<=PASSAGE_SPLIT_LIMIT_MM


def distance_label(field,value):
    if field not in WALL_FIELDS+EDGE_FIELDS+HEIGHT_FIELDS:raise ValueError('Unknown distance field')
    if value is None:return {'raw':None,'state':'missing','exact_mm':None}
    if isinstance(value,bool):raise ValueError('Boolean is not a distance')
    number=float(value)
    if not math.isfinite(number):raise ValueError('Non-finite distance')
    result={'raw':number,'state':'exact','exact_mm':number}
    if field in WALL_FIELDS and number==-1:
        result.update(state='no_wall_or_over_50',exact_mm=None)
    elif field in EDGE_FIELDS and number==51:
        result.update(state='over_50',exact_mm=None,lower_bound_mm=50.,lower_bound_inclusive=False)
    elif field in EDGE_FIELDS and number>50:
        result.update(state='invalid_censored_encoding',exact_mm=None)
    elif number<0:
        result.update(state='negative_sentinel_unconfirmed',exact_mm=None)
    elif field in HOLE_FIELDS and number==0:
        result.update(state='no_hole',exact_mm=None,hole_present=False)
    elif field in HEIGHT_FIELDS and number==0:
        result.update(state='zero_meaning_unconfirmed',exact_mm=None)
    if field in HOLE_FIELDS:
        result.setdefault('hole_present',True if number>0 else None)
    return result


def process_labels(row,group=None):
    effective,corrections=apply_label_corrections(group,row)
    distances={k:distance_label(k,effective.get(k)) for k in WALL_FIELDS+EDGE_FIELDS+HEIGHT_FIELDS}
    for correction in corrections:
        label=distances[correction['field']]
        label.update(raw=correction['raw_value'],effective_value=correction['effective_value'],correction=correction)
    return {'semantics_version':SEMANTICS_VERSION,'distance_unit':'millimeter',
            'side_plate_split_any':row.get(SIDE_PLATE_FIELD),
            'connected_overhead_eave_any':row.get('HasEave (Any)'),
            'distances':distances,'label_corrections':corrections,
            'segmentation_policy':{'retain_geometric_breakpoints':True,'retain_separate_weld_records':True},
            'intersection_distance':{'state':'not_present_in_supplied_excel_schema','unit':'millimeter',
                                     'threshold_mm':PASSAGE_SPLIT_LIMIT_MM,'split_at_equal_threshold':True},
            'scope':'Customer reference supervision only; original values and explicit corrections retained.'}
