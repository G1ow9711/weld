"""Inspect native closest-point coordinates/parameters before an offline audit."""
import json
from pathlib import Path
import gmsh
import numpy as np

gmsh.initialize([], readConfigFiles=False); gmsh.option.setNumber('General.Terminal', 0)
try:
    a = gmsh.model.occ.addPoint(0., 0., 0.); b = gmsh.model.occ.addPoint(10., 0., 0.)
    line = gmsh.model.occ.addLine(a, b); gmsh.model.occ.synchronize()
    lower, upper = gmsh.model.getParametrizationBounds(1, line); rows = []
    for point in ([-5., 2., 0.], [15., 2., 0.], [3.25, 2., 0.]):
        closest, parameter = gmsh.model.getClosestPoint(1, line, point)
        clamped = np.clip(parameter, lower, upper)
        value = gmsh.model.getValue(1, line, clamped.tolist())
        from_coordinate = gmsh.model.getParametrization(1, line, closest.tolist())
        rows.append(dict(query=point, raw_closest=closest.tolist(), raw_parameter=parameter.tolist(),
            lower=lower.tolist(), upper=upper.tolist(), clamped_parameter=clamped.tolist(),
            clamped_value=value.tolist(), parameter_from_closest_coordinate=from_coordinate.tolist()))
    surface = gmsh.model.occ.addRectangle(0., 0., 0., 10., 8.); gmsh.model.occ.synchronize()
    surfaces = []
    for point in ([3.25, 2.5, 1.], [15., 2.5, 1.]):
        closest, parameter = gmsh.model.getClosestPoint(2, surface, point)
        recomputed = gmsh.model.getParametrization(2, surface, closest.tolist())
        raw_value = gmsh.model.getValue(2, surface, parameter.tolist())
        recomputed_value = gmsh.model.getValue(2, surface, recomputed.tolist())
        surfaces.append(dict(query=point, closest=closest.tolist(), raw_parameter=parameter.tolist(),
            recomputed_parameter=recomputed.tolist(), raw_reconstruction_error_mm=float(np.linalg.norm(raw_value-closest)),
            recomputed_reconstruction_error_mm=float(np.linalg.norm(recomputed_value-closest)),
            raw_inside=gmsh.model.isInside(2, surface, parameter.tolist(), parametric=True),
            recomputed_inside=gmsh.model.isInside(2, surface, recomputed.tolist(), parametric=True)))
    output = dict(gmsh_version=gmsh.__version__, rows=rows, surface_rows=surfaces,
        closest_API_documentation=gmsh.model.getClosestPoint.__doc__,
        parametrization_API_documentation=gmsh.model.getParametrization.__doc__)
    Path('weld_pipeline/reports/native_edge_projection_preflight_164_20261002/probe_surface.json').write_text(
        json.dumps(output, indent=2), encoding='utf8')
    print(json.dumps(rows, ensure_ascii=False), flush=True)
    print(json.dumps(surfaces, ensure_ascii=False), flush=True)
finally:
    gmsh.finalize()
