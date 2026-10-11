"""Keep all original327 hypotheses; append pure XYZ full-face finite domains."""
from pathlib import Path
import gc
import json
import shutil

import numpy as np
from .robust_surface_planes import surface_candidates
from .small_face_line_carriers_327 import small_face_line_carriers
from .arc_surface_candidates import arc_surface_candidates
from .contract_curve_geometry import CircularArc
from .full_support_horizontal_carriers_334 import full_support_horizontal_carriers
from .streaming_candidate_bank_331 import attach_original_support,json_piece
from .pure_prediction_file_guard_329 import NoFileReadsDuringPrediction

VERSION = "FRESH_ALL_BASE327_PLUS_MEASURED_FULL_FACE_DOMAINS_PHASE_STREAMING_334_v1"


def write_streaming_bank(xyz_mm,destination,original_sha,progress):
    xyz = np.asarray(xyz_mm,dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 8 or not np.isfinite(xyz).all():
        raise ValueError("All finite original XYZ required")
    destination = Path(destination)
    pieces = destination.parent / "pure_component_statistics"
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
                group[index] = None
        with NoFileReadsDuringPrediction() as guard:
            progress("FRESH_ALL_ORIGINAL_XYZ_ROBUST_SURFACES",original_XYZ=len(xyz))
            ordinary,geometry = surface_candidates(xyz)
            counts["ordinary_candidates"] = len(ordinary)
            additional,additional_statistics = full_support_horizontal_carriers(ordinary,geometry)
            counts["full_face_support_candidates"] = len(additional)
            json_piece(pieces / "ordinary.json",geometry)
            json_piece(pieces / "full_face_support.json",additional_statistics)
            emit(ordinary)
            del ordinary,additional_statistics
            gc.collect()
            progress("FRESH_ALL_ORIGINAL_XYZ_SMALL_FACES",ordinary_candidates=counts["ordinary_candidates"],
                pure_full_face_candidates_waiting=counts["full_face_support_candidates"])
            small,small_stats = small_face_line_carriers(xyz,geometry)
            counts["small_candidates"] = len(small)
            json_piece(pieces / "small.json",small_stats)
            emit(small)
            del small,small_stats,geometry
            gc.collect()
            progress("ALL_ORIGINAL_SMALL_FACE_DICTIONARIES_WRITTEN_BEFORE_CIRCLE_FIT",small_candidates=counts["small_candidates"])
            arcs,arc_stats = arc_surface_candidates(xyz)
            counts["arc_candidates"] = len(arcs)
            for candidate in arcs:
                parameters = candidate["arc"]
                phase = parameters["start_angle_rad"]
                arc = CircularArc(np.array([*parameters["center_xy_mm"],candidate["height_mm"]]),
                    parameters["radius_mm"],np.array([0.,0.,1.]),np.array([np.cos(phase),np.sin(phase),0.]),
                    parameters["sweep_rad"],False)
                candidate["segments"][0]["arc_parameters"] = arc.to_parameters()
            json_piece(pieces / "arc.json",arc_stats)
            emit(arcs)
            del arcs,arc_stats
            gc.collect()
            progress("ALL_ORIGINAL_BASE_CANDIDATES_WRITTEN_APPEND_ALL_FULL_FACE_FINITE_DOMAINS")
            emit(additional)
            del additional
            gc.collect()
        attempts.extend(guard.attempts)
        assert not attempts
        output.write('],"statistics":{')
        simple = dict(version=VERSION,**counts,all_original_XYZ_processed=len(xyz),
            source_identifiers_or_filename_branches=False,target_reads_for_generation=0,STEP_reads=0,
            learned_label_reads=0,raw_support_references_retained=True,runtime_requires_fresh_XYZ_reconstruction=True,
            all_original_base_algorithm_parameters_unchanged=True,original_base_candidate_records_precede_added_domains=True,
            supported_geometry_scope="FINITE_LINE_AND_HORIZONTAL_OPEN_CIRCLE_OTHER_TYPES_STILL_REQUIRED",full_contract_pass=False)
        first_stat = True
        for key,value in simple.items():
            if not first_stat:
                output.write(",")
            json.dump(key,output);output.write(":");json.dump(value,output,allow_nan=False)
            first_stat = False
        for name,filename in (("ordinary_statistics","ordinary.json"),("small_statistics","small.json"),
                              ("arc_statistics","arc.json"),("full_face_support_statistics","full_face_support.json")):
            output.write(",");json.dump(name,output);output.write(":")
            with (pieces / filename).open("r",encoding="utf-8") as component:
                shutil.copyfileobj(component,output,length=65536)
        output.write("},")
        metadata = dict(original_XYZ=len(xyz),original_array_sha256=original_sha,
            target_reads_for_generation=0,STEP_reads=0,NN_forwards=0,actual_optimizer_updates=0,
            read_guard_attempts=attempts,all_candidates_retained_without_GT_pruning=True,
            original_base_candidate_count=sum(counts[name] for name in ("ordinary_candidates","small_candidates","arc_candidates")),
            additional_full_face_support_candidate_count=counts["full_face_support_candidates"],
            material_identity_is_unverified=True,candidate_coverage_is_not_model_precision=True,
            serialization_mode=VERSION,full_contract_pass=False)
        for index,(key,value) in enumerate(metadata.items()):
            if index:
                output.write(",")
            json.dump(key,output);output.write(":");json.dump(value,output,ensure_ascii=False,allow_nan=False)
        output.write("}\n")
    partial.replace(destination)
    return dict(**counts,total_candidates=sum(counts.values()),
        original_base_candidate_count=sum(counts[name] for name in ("ordinary_candidates","small_candidates","arc_candidates")))
