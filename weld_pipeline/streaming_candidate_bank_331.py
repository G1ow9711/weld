"""Same327 proposals, generated/written by phase; no candidate pruning.

Do not retain 100,000 small-face dictionaries while fitting circles. This
offline writer preserves all proposal fields/point ordinals and component
statistics. It does not change the frozen running model or its inference.
"""
from pathlib import Path
import gc
import json
import shutil

import numpy as np
from .robust_surface_planes import surface_candidates
from .small_face_line_carriers_327 import small_face_line_carriers
from .arc_surface_candidates import arc_surface_candidates
from .contract_curve_geometry import CircularArc
from .analytic_carrier_slots_327 import build_carriers
from .pure_prediction_file_guard_329 import NoFileReadsDuringPrediction

VERSION = "SAME327_ALL_PROPOSALS_PHASE_STREAMING_BANK_331_v1"
PROVIDER_VERSION = "FRESH_SOURCE_INDEPENDENT_PURE_XYZ_ANALYTIC_CARRIERS_327_v1"


def attach_original_support(candidate, xyz):
    if "raw_support_rows" not in candidate:
        carrier = build_carriers([candidate])[0]
        if carrier.kind == "line":
            q = (xyz-carrier.origin_mm) @ carrier.direction
            residual = xyz-carrier.origin_mm-q[:,None]*carrier.direction
            mask = (q>=0.) & (q<=carrier.length_mm) & (np.linalg.norm(residual,axis=1)<=.2)
        else:
            radial = xyz-carrier.arc.center_mm
            height = radial @ carrier.arc.plane_normal
            planar = radial-height[:,None]*carrier.arc.plane_normal
            angle = (np.sign(carrier.arc.sweep_rad)*np.arctan2(
                planar @ np.cross(carrier.arc.plane_normal,carrier.arc.start_direction),
                planar @ carrier.arc.start_direction)) % (2*np.pi)
            mask = (np.abs(height)<=.2) & (np.abs(np.linalg.norm(planar,axis=1)-carrier.arc.radius_mm)<=.2) & (angle<=abs(carrier.arc.sweep_rad))
        candidate["raw_support_rows"] = np.flatnonzero(mask).tolist()
    candidate["material_identity"] = "MATERIAL_UNVERIFIED"
    candidate["weld_identity"] = "UNSELECTED_GEOMETRY_HYPOTHESIS"


def json_piece(path, value):
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,separators=(",",":"),allow_nan=False)


def write_streaming_bank(xyz_mm, destination, original_sha, progress):
    xyz = np.asarray(xyz_mm,dtype=np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<8 or not np.isfinite(xyz).all():
        raise ValueError("All finite original XYZ required")
    destination = Path(destination)
    pieces = destination.parent/"pure_component_statistics"
    pieces.mkdir(exist_ok=False)
    partial = destination.with_name(destination.name+".streaming.partial")
    counts,attempts,first = {},[],True
    with partial.open("x",encoding="utf-8") as output:
        output.write('{"candidates":[')
        def emit(group):
            nonlocal first
            for index,candidate in enumerate(group):
                attach_original_support(candidate,xyz)
                if not first:
                    output.write(",")
                json.dump(candidate,output,ensure_ascii=False,separators=(",",":"),allow_nan=False)
                first = False
                # Release every dictionary/support list after it is recorded.
                group[index] = None
        with NoFileReadsDuringPrediction() as guard:
            progress("FULL_ORIGINAL_XYZ_ROBUST_SURFACES",original_XYZ=len(xyz))
            ordinary,geometry = surface_candidates(xyz)
            counts["ordinary_candidates"] = len(ordinary)
            json_piece(pieces/"ordinary.json",geometry)
            emit(ordinary)
            del ordinary
            gc.collect()
            progress("FULL_ORIGINAL_XYZ_MULTISCALE_SMALL_FACES",ordinary_candidates=counts["ordinary_candidates"])
            small,small_stats = small_face_line_carriers(xyz,geometry)
            counts["small_candidates"] = len(small)
            json_piece(pieces/"small.json",small_stats)
            emit(small)
            del small,small_stats,geometry
            gc.collect()
            progress("ALL_SMALL_FACE_DICTIONARIES_WRITTEN_BEFORE_CIRCLE_FIT",small_candidates=counts["small_candidates"])
            arcs,arc_stats = arc_surface_candidates(xyz)
            counts["arc_candidates"] = len(arcs)
            for candidate in arcs:
                parameters = candidate["arc"]
                phase = parameters["start_angle_rad"]
                arc = CircularArc(np.array([*parameters["center_xy_mm"],candidate["height_mm"]]),
                    parameters["radius_mm"],np.array([0.,0.,1.]),np.array([np.cos(phase),np.sin(phase),0.]),
                    parameters["sweep_rad"],False)
                candidate["segments"][0]["arc_parameters"] = arc.to_parameters()
            json_piece(pieces/"arc.json",arc_stats)
            emit(arcs)
            del arcs,arc_stats
            gc.collect()
        attempts.extend(guard.attempts)
        assert not attempts
        output.write('],"statistics":{')
        simple = dict(version=PROVIDER_VERSION,**counts,all_original_XYZ_processed=len(xyz),
            source_identifiers_or_filename_branches=False,target_reads_for_generation=0,
            STEP_reads=0,learned_label_reads=0,raw_support_references_retained=True,
            runtime_requires_fresh_XYZ_reconstruction=True,
            supported_geometry_scope="FINITE_LINE_AND_VERIFIED_HORIZONTAL_OPEN_CIRCLE_ONLY_NOT_ALL_CONTRACT_TYPES",
            full_contract_pass=False)
        first_stat = True
        for key,value in simple.items():
            if not first_stat:output.write(",")
            json.dump(key,output);output.write(":");json.dump(value,output,allow_nan=False)
            first_stat = False
        # Read only our freshly written pure statistics after generation.
        for name,filename in (("ordinary_statistics","ordinary.json"),("small_statistics","small.json"),("arc_statistics","arc.json")):
            output.write(",");json.dump(name,output);output.write(":")
            with (pieces/filename).open("r",encoding="utf-8") as component:
                shutil.copyfileobj(component,output,length=65536)
        output.write('},')
        metadata = dict(original_XYZ=len(xyz),original_array_sha256=original_sha,
            target_reads_for_generation=0,STEP_reads=0,NN_forwards=0,actual_optimizer_updates=0,
            read_guard_attempts=attempts,all_candidates_retained_without_GT_pruning=True,
            material_identity_is_unverified=True,candidate_coverage_is_not_model_precision=True,
            serialization_mode=VERSION,full_contract_pass=False)
        for n,(key,value) in enumerate(metadata.items()):
            if n:output.write(",")
            json.dump(key,output);output.write(":");json.dump(value,output,ensure_ascii=False,allow_nan=False)
        output.write("}\n")
    partial.replace(destination)
    return dict(**counts,total_candidates=sum(counts.values()))
