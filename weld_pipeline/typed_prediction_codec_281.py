"""Predictable fields for the whole260 graph; teacher JSON is never decoded.

The exact numeric sidecar belongs only to teacher representation review.
decode_prediction accepts discrete predictions, ordinary numeric predictions,
source-domain metadata and predicted links. Physical acceptance is separate.
"""
from fractions import Fraction
import math
import numpy as np

from .source_graph_targets_275 import SLOT_COLUMNS, identity, canonical
from .mixed_parent_graph_260 import VERSION as GRAPH_VERSION

VERSION = "FULL_TYPED_PREDICTION_FIELDS_AND_ORDERED_LINKS_281_v1"
FIELDS = ("coordinates", "geometry_spans", "children", "parents")
CATEGORIES = {
    "coordinate_form": ("position", "curve_endpoint", "physical_ref"),
    "position_form": ("uv_mm", "barycentric", "occurrence", "curve_endpoint"),
    "occurrence_kind": ("VERTEX", "EDGE_INTERIOR"),
    "span_kind": ("line", "arc", "polyline", "carrier_slice"),
    "world_kind": ("line", "arc", "polyline", "compound"),
    "weld_type": ("horizontal", "vertical"),
    "closed": (False, True),
    "direction": (-1, 1),
    "frame_kind": ("line", "arc", "carrier", "piecewise"),
    "arc_prototype_closed": (False, True),
    "arc_interval_present": (False, True),
    "arc_authority_present": (False, True),
}
CATEGORY_NAMES = tuple(CATEGORIES)
NUMERIC_NAMES = (
    "uv_u_m", "uv_v_m", "barycentric_u", "barycentric_v",
    "occurrence_ordinal", "occurrence_alpha", "analytic_endpoint_index",
    "center_u_m", "center_v_m", "radius_m", "u_axis_u", "u_axis_v",
    "v_axis_u", "v_axis_v", "sweep_rad", "arc_prototype_turn_count",
    "arc_interval_start", "arc_interval_end", "carrier_extra_turns",
    "child_order_index", "parent_turn_count", "parent_axis_x", "parent_axis_y", "parent_axis_z",
)
RELATIONS = (
    "parent", "children", "start_boundary", "end_boundary", "geometry_spans",
    "internal_geometry_nodes", "phase_boundary", "authority_span", "frame_authority_span",
    "physical_ref", "curve_endpoint", "frame_canonical_span_order", "gap_left", "gap_right", "gap_span",
)
DOMAIN_KINDS = ("own", "parent_allowed", "parent_frame")
GAP_KINDS = ("SHARED_BOUNDARY", "POSITIVE_GAP")
PREDICTION_KEYS = {"slots", "categories", "numeric", "relations", "domain_links", "gap_classes"}


def _fields(row, required, optional=()):
    if not isinstance(row, dict) or not set(required) <= set(row) or set(row) - set(required) - set(optional):
        raise ValueError("Undeclared or missing typed prediction field: " + str(row))


def _scalar(value):
    if isinstance(value, dict) and set(value) == {"numerator", "denominator"}:
        return float(Fraction(int(value["numerator"]), int(value["denominator"])))
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite explicit scalar required")
    return float(value)


def _integer(value):
    number = _scalar(value)
    if number != int(number):
        raise ValueError("Discrete predictions must be explicitly integer; no silent rounding")
    return int(number)


def encode_teacher(graph, anchor_count, domain_index):
    """Loss-only encoding, including every ordered link and optional field."""
    _fields(graph, ("version", *FIELDS))
    if graph["version"] != GRAPH_VERSION:
        raise ValueError("Whole260 graph version required")
    definitions = []
    for field in FIELDS:
        for row in graph[field]:
            definitions.append((identity(row["slot"], anchor_count), field, row))
    definitions.sort(key=lambda value: value[0])
    keys = [value[0] for value in definitions]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate canonical source role")
    lookup = {key: i for i, key in enumerate(keys)}
    categories = np.full((len(keys), len(CATEGORIES)), -1, np.int64)
    numeric = np.zeros((len(keys), len(NUMERIC_NAMES)), np.float64)
    known = np.zeros(numeric.shape, np.bool_)
    exact_numeric, links, domains, gaps = [{} for _ in keys], [], [], []

    def cat(i, name, value):
        if value not in CATEGORIES[name] or type(value) != type(CATEGORIES[name][0]):
            raise ValueError("Explicit category outside fixed schema: " + name)
        categories[i, CATEGORY_NAMES.index(name)] = CATEGORIES[name].index(value)

    def num(i, name, value, divisor=1):
        column = NUMERIC_NAMES.index(name)
        numeric[i, column] = _scalar(value) / divisor
        known[i, column] = True
        exact_numeric[i][name] = value

    def ref(i, kind, value, order=0, group=-1):
        key = identity(value, anchor_count)
        if key not in lookup:
            raise ValueError("Dangling predicted role reference")
        links.append([i, lookup[key], RELATIONS.index(kind), order, group])

    def refs(i, kind, values, group=-1):
        if not isinstance(values, list):
            raise ValueError("Ordered reference list required")
        for order, value in enumerate(values):
            ref(i, kind, value, order, group)

    def domain(i, kind, value, order=0):
        domains.append([i, int(domain_index(value)), DOMAIN_KINDS.index(kind), order])

    def endpoint(i, value):
        _fields(value, ("span", "endpoint_index"))
        ref(i, "curve_endpoint", value["span"])
        num(i, "analytic_endpoint_index", value["endpoint_index"])

    def position(i, value):
        if not isinstance(value, dict) or len(value) != 1:
            raise ValueError("One explicit source position form required")
        form = next(iter(value))
        cat(i, "position_form", form)
        data = value[form]
        if form in ("uv_mm", "barycentric"):
            if not isinstance(data, list) or len(data) != 2:
                raise ValueError("Source position requires exactly two parameters")
            names = ("uv_u_m", "uv_v_m") if form == "uv_mm" else ("barycentric_u", "barycentric_v")
            for name, item in zip(names, data):
                num(i, name, item, 1000 if form == "uv_mm" else 1)
        elif form == "occurrence":
            _fields(data, ("kind", "ordinal"), ("alpha",))
            cat(i, "occurrence_kind", data["kind"])
            num(i, "occurrence_ordinal", data["ordinal"])
            if data["kind"] == "EDGE_INTERIOR":
                if "alpha" not in data:
                    raise ValueError("Strict edge interior alpha required")
                num(i, "occurrence_alpha", data["alpha"])
            elif "alpha" in data:
                raise ValueError("Vertex must not introduce an extra alpha")
        elif form == "curve_endpoint":
            endpoint(i, data)
        else:
            raise ValueError("Unsupported source position")

    for i, (key, field, row) in enumerate(definitions):
        if field == "coordinates":
            _fields(row, ("slot", "domain"), ("position", "curve_endpoint", "physical_ref", "local_position"))
            forms = [name for name in ("position", "curve_endpoint", "physical_ref") if name in row]
            if len(forms) != 1:
                raise ValueError("One defining coordinate form required")
            domain(i, "own", row["domain"])
            form = forms[0]
            cat(i, "coordinate_form", form)
            if form == "position":
                if "local_position" in row:
                    raise ValueError("Unexpected own position")
                position(i, row[form])
            elif form == "curve_endpoint":
                if "local_position" in row:
                    raise ValueError("Unexpected own position")
                endpoint(i, row[form])
            else:
                if "local_position" not in row:
                    raise ValueError("Physical reference needs its own source position")
                ref(i, "physical_ref", row[form])
                position(i, row["local_position"])
        elif field == "geometry_spans":
            _fields(row, ("slot", "domain", "kind", "start_boundary", "end_boundary", "internal_geometry_nodes", "is_closed"),
                    ("parameters", "parameter_interval", "authority_span"))
            domain(i, "own", row["domain"])
            cat(i, "span_kind", row["kind"])
            cat(i, "closed", row["is_closed"])
            ref(i, "start_boundary", row["start_boundary"])
            ref(i, "end_boundary", row["end_boundary"])
            refs(i, "internal_geometry_nodes", row["internal_geometry_nodes"])
            if row["kind"] == "arc":
                p = row["parameters"]
                _fields(p, ("center_mm", "radius_mm", "u_axis", "v_axis", "sweep_rad", "is_closed", "turn_count"))
                for field_name, names, divisor in (("center_mm", ("center_u_m", "center_v_m"),1000),
                    ("u_axis",("u_axis_u","u_axis_v"),1),("v_axis",("v_axis_u","v_axis_v"),1)):
                    if len(p[field_name]) != 2:
                        raise ValueError("Two-dimensional source arc frame required")
                    for name, value in zip(names,p[field_name]):
                        num(i,name,value,divisor)
                num(i,"radius_m",p["radius_mm"],1000)
                num(i,"sweep_rad",p["sweep_rad"])
                num(i,"arc_prototype_turn_count",p["turn_count"])
                cat(i,"arc_prototype_closed",p["is_closed"])
                cat(i,"arc_interval_present","parameter_interval" in row)
                cat(i,"arc_authority_present","authority_span" in row)
                if "parameter_interval" in row:
                    if len(row["parameter_interval"]) != 2:
                        raise ValueError("Two interval limits required")
                    for name, value in zip(("arc_interval_start","arc_interval_end"),row["parameter_interval"]):
                        num(i,name,value)
                if "authority_span" in row:
                    ref(i,"authority_span",row["authority_span"])
            elif row["kind"] == "carrier_slice":
                _fields(row["parameters"],("direction","extra_turns"))
                cat(i,"direction",row["parameters"]["direction"])
                num(i,"carrier_extra_turns",row["parameters"]["extra_turns"])
            elif set(row) & {"parameters","parameter_interval","authority_span"}:
                raise ValueError("Unrepresented free span parameters")
        elif field == "children":
            _fields(row,("slot","parent","order_index","start_boundary","end_boundary","geometry_spans","geometry_kind","world_kind","weld_type"))
            if row["geometry_kind"] != ("line" if row["world_kind"] == "line" else "curve"):
                raise ValueError("World and broad geometry kinds conflict")
            cat(i,"world_kind",row["world_kind"])
            cat(i,"weld_type",row["weld_type"])
            num(i,"child_order_index",row["order_index"])
            for name in ("parent","start_boundary","end_boundary"):
                ref(i,name,row[name])
            refs(i,"geometry_spans",row["geometry_spans"])
        else:
            _fields(row,("slot","children","start_boundary","end_boundary","allowed_domains","direction","is_closed","phase_boundary","turn_count","gaps","frame"))
            cat(i,"direction",row["direction"])
            cat(i,"closed",row["is_closed"])
            num(i,"parent_turn_count",row["turn_count"])
            refs(i,"children",row["children"])
            for name in ("start_boundary","end_boundary","phase_boundary"):
                ref(i,name,row[name])
            for order, value in enumerate(row["allowed_domains"]):
                domain(i,"parent_allowed",value,order)
            frame = row["frame"]
            cat(i,"frame_kind",frame["kind"])
            if frame["kind"] == "line":
                _fields(frame,("kind","axis_world"))
                if len(frame["axis_world"]) != 3:
                    raise ValueError("Three-dimensional parent line axis required")
                for name,value in zip(("parent_axis_x","parent_axis_y","parent_axis_z"),frame["axis_world"]):
                    num(i,name,value)
            elif frame["kind"] == "arc":
                _fields(frame,("kind","authority_span"))
                ref(i,"frame_authority_span",frame["authority_span"])
            elif frame["kind"] == "carrier":
                _fields(frame,("kind","domain"))
                domain(i,"parent_frame",frame["domain"])
            else:
                _fields(frame,("kind","canonical_span_order"))
                refs(i,"frame_canonical_span_order",frame["canonical_span_order"])
            for group,gap in enumerate(row["gaps"]):
                _fields(gap,("relation","left_boundary","right_boundary","geometry_spans"))
                gaps.append([i,group,GAP_KINDS.index(gap["relation"])])
                ref(i,"gap_left",gap["left_boundary"],group=group)
                ref(i,"gap_right",gap["right_boundary"],group=group)
                refs(i,"gap_span",gap["geometry_spans"],group)
    return dict(version=VERSION,anchor_count=anchor_count,
        slots=np.asarray(keys,np.int64).reshape(-1,3),categories=categories,numeric=numeric,numeric_known=known,
        relations=np.asarray(links,np.int64).reshape(-1,5),domain_links=np.asarray(domains,np.int64).reshape(-1,4),
        gap_classes=np.asarray(gaps,np.int64).reshape(-1,3),exact_numeric_values=exact_numeric,
        supervision_only=True,teacher_sidecar_used_for_runtime=False)


def _decode(prediction, anchor_count, domain_record, exact_teacher_scalars=None):
    if set(prediction) != PREDICTION_KEYS:
        raise ValueError("Runtime accepts predicted arrays only, never teacher graph/sidecar")
    slots = np.asarray(prediction["slots"])
    if slots.ndim != 2 or slots.shape[1] != 3:
        raise ValueError("Canonical slot array required")
    slot_records = []
    for anchor,column,ordinal in slots.tolist():
        if type(anchor) is not int or type(column) is not int or not 0 <= column < len(SLOT_COLUMNS):
            raise ValueError("Explicit canonical source slot indices required")
        role,expected_ordinal = SLOT_COLUMNS[column]
        item = dict(anchor_id=anchor,role=role,local_slot_ordinal=ordinal)
        if ordinal != expected_ordinal or identity(item,anchor_count) != (anchor,column,ordinal):
            raise ValueError("Canonical role capacity changed")
        slot_records.append(item)
    if len({canonical(r) for r in slot_records}) != len(slot_records):
        raise ValueError("Duplicate source role predictions")
    categories=np.asarray(prediction["categories"])
    numeric=np.asarray(prediction["numeric"])
    if categories.shape != (len(slots),len(CATEGORIES)) or numeric.shape != (len(slots),len(NUMERIC_NAMES)):
        raise ValueError("Fixed category/numeric prediction schema required")
    links,domain_links,gaps={},{},{}
    used_links,used_domains,used_gaps=set(),set(),set()
    for n,row in enumerate(np.asarray(prediction["relations"]).tolist()):
        if len(row)!=5 or any(type(v) is not int for v in row):raise ValueError("Discrete ordered role links required")
        owner,target,kind,order,group=row
        if not 0<=owner<len(slots) or not 0<=target<len(slots) or not 0<=kind<len(RELATIONS) or order<0 or group< -1:raise ValueError("Link outside typed source pool")
        links.setdefault((owner,RELATIONS[kind],group),[]).append((order,target,n))
    for n,row in enumerate(np.asarray(prediction["domain_links"]).tolist()):
        if len(row)!=4 or any(type(v) is not int for v in row):raise ValueError("Discrete ordered source-domain links required")
        owner,target,kind,order=row
        if not 0<=owner<len(slots) or target<0 or not 0<=kind<len(DOMAIN_KINDS) or order<0:raise ValueError("Invalid domain link")
        domain_links.setdefault((owner,DOMAIN_KINDS[kind]),[]).append((order,target,n))
    for n,row in enumerate(np.asarray(prediction["gap_classes"]).tolist()):
        if len(row)!=3 or any(type(v) is not int for v in row):raise ValueError("Discrete gap group/class required")
        owner,group,kind=row
        if not 0<=owner<len(slots) or group<0 or not 0<=kind<len(GAP_KINDS) or (owner,group) in gaps:raise ValueError("Duplicate or invalid gap classification")
        gaps[owner,group]=(kind,n)

    def cat(i,name):
        value=categories[i,CATEGORY_NAMES.index(name)].item()
        value=_integer(value)
        if not 0<=value<len(CATEGORIES[name]):raise ValueError("Required category remains unknown: "+name)
        return CATEGORIES[name][value]
    def num(i,name,multiplier=1,integer=False):
        if exact_teacher_scalars is not None:
            if name not in exact_teacher_scalars[i]:raise ValueError("Missing exact teacher scalar")
            value=exact_teacher_scalars[i][name]
        else:value=float(numeric[i,NUMERIC_NAMES.index(name)])*multiplier
        return _integer(value) if integer else value if exact_teacher_scalars is not None else _scalar(value)
    def ordered(i,kind,group=-1,single=False):
        items=sorted(links.get((i,kind,group),[]))
        if [v[0] for v in items]!=list(range(len(items))) or (single and len(items)!=1):raise ValueError("Missing, duplicate or unordered predicted links: "+kind)
        used_links.update(v[2] for v in items)
        values=[dict(slot_records[v[1]]) for v in items]
        return values[0] if single else values
    def domains(i,kind,single=False):
        items=sorted(domain_links.get((i,kind),[]))
        if [v[0] for v in items]!=list(range(len(items))) or (single and len(items)!=1):raise ValueError("Missing, duplicate or unordered domain link")
        used_domains.update(v[2] for v in items)
        values=[dict(domain_record(v[1])) for v in items]
        return values[0] if single else values
    def endpoint(i):
        return dict(span=ordered(i,"curve_endpoint",single=True),endpoint_index=num(i,"analytic_endpoint_index",integer=True))
    def position(i):
        form=cat(i,"position_form")
        if form=="uv_mm":return {form:[num(i,"uv_u_m",1000),num(i,"uv_v_m",1000)]}
        if form=="barycentric":return {form:[num(i,"barycentric_u"),num(i,"barycentric_v")]}
        if form=="curve_endpoint":return {form:endpoint(i)}
        value=dict(kind=cat(i,"occurrence_kind"),ordinal=num(i,"occurrence_ordinal",integer=True))
        if value["kind"]=="EDGE_INTERIOR":value["alpha"]=num(i,"occurrence_alpha")
        return {form:value}
    graph=dict(version=GRAPH_VERSION,coordinates=[],geometry_spans=[],children=[],parents=[])
    for i,ref in enumerate(slot_records):
        row=dict(slot=dict(ref));role=ref["role"]
        if role in ("boundary","geometry_node"):
            row["domain"]=domains(i,"own",True)
            form=cat(i,"coordinate_form")
            if form=="position":row[form]=position(i)
            elif form=="curve_endpoint":row[form]=endpoint(i)
            else:row.update(physical_ref=ordered(i,"physical_ref",single=True),local_position=position(i))
            graph["coordinates"].append(row)
        elif role=="geometry_span":
            row.update(domain=domains(i,"own",True),kind=cat(i,"span_kind"),is_closed=cat(i,"closed"),
                start_boundary=ordered(i,"start_boundary",single=True),end_boundary=ordered(i,"end_boundary",single=True),
                internal_geometry_nodes=ordered(i,"internal_geometry_nodes"))
            if row["kind"]=="arc":
                row["parameters"]=dict(center_mm=[num(i,"center_u_m",1000),num(i,"center_v_m",1000)],radius_mm=num(i,"radius_m",1000),
                    u_axis=[num(i,"u_axis_u"),num(i,"u_axis_v")],v_axis=[num(i,"v_axis_u"),num(i,"v_axis_v")],sweep_rad=num(i,"sweep_rad"),
                    is_closed=cat(i,"arc_prototype_closed"),turn_count=num(i,"arc_prototype_turn_count",integer=True))
                if cat(i,"arc_interval_present"):row["parameter_interval"]=[num(i,"arc_interval_start"),num(i,"arc_interval_end")]
                if cat(i,"arc_authority_present"):row["authority_span"]=ordered(i,"authority_span",single=True)
            elif row["kind"]=="carrier_slice":row["parameters"]=dict(direction=cat(i,"direction"),extra_turns=num(i,"carrier_extra_turns",integer=True))
            graph["geometry_spans"].append(row)
        elif role=="child":
            kind=cat(i,"world_kind")
            row.update(parent=ordered(i,"parent",single=True),order_index=num(i,"child_order_index",integer=True),
                start_boundary=ordered(i,"start_boundary",single=True),end_boundary=ordered(i,"end_boundary",single=True),
                geometry_spans=ordered(i,"geometry_spans"),world_kind=kind,geometry_kind="line" if kind=="line" else "curve",weld_type=cat(i,"weld_type"))
            graph["children"].append(row)
        else:
            row.update(children=ordered(i,"children"),start_boundary=ordered(i,"start_boundary",single=True),end_boundary=ordered(i,"end_boundary",single=True),
                phase_boundary=ordered(i,"phase_boundary",single=True),allowed_domains=domains(i,"parent_allowed"),
                direction=cat(i,"direction"),is_closed=cat(i,"closed"),turn_count=num(i,"parent_turn_count",integer=True))
            kind=cat(i,"frame_kind");frame=dict(kind=kind)
            if kind=="line":frame["axis_world"]=[num(i,"parent_axis_x"),num(i,"parent_axis_y"),num(i,"parent_axis_z")]
            elif kind=="arc":frame["authority_span"]=ordered(i,"frame_authority_span",single=True)
            elif kind=="carrier":frame["domain"]=domains(i,"parent_frame",True)
            else:frame["canonical_span_order"]=ordered(i,"frame_canonical_span_order")
            row.update(frame=frame,gaps=[])
            for group in range(len(row["children"])-1):
                if (i,group) not in gaps:raise ValueError("Every business adjacency requires a predicted gap class")
                gap_class,n=gaps[i,group];used_gaps.add(n)
                row["gaps"].append(dict(relation=GAP_KINDS[gap_class],left_boundary=ordered(i,"gap_left",group,True),
                    right_boundary=ordered(i,"gap_right",group,True),geometry_spans=ordered(i,"gap_span",group)))
            graph["parents"].append(row)
    if len(used_links)!=len(np.asarray(prediction["relations"])) or len(used_domains)!=len(np.asarray(prediction["domain_links"])) or len(used_gaps)!=len(np.asarray(prediction["gap_classes"])):
        raise ValueError("Unconsumed predicted relation/domain/gap fields; no silent deletion")
    return graph


def decode_prediction(prediction,anchor_count,domain_record):
    """No GT, exact teacher scalars or preserved graph templates are accepted."""
    return _decode(prediction,anchor_count,domain_record)


def restore_teacher_representation(pack,domain_record):
    """Loss-schema coverage review only; never a neural result or runtime input."""
    if pack.get("version")!=VERSION or not pack.get("supervision_only"):
        raise ValueError("Explicit loss-only pack required")
    return _decode({key:pack[key] for key in PREDICTION_KEYS},pack["anchor_count"],domain_record,pack["exact_numeric_values"])
