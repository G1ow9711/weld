"""Structural exception for curved upright fillets with transverse notch plates.

The finite side-face pair bounds each obstruction. A ray from the longitudinal
wall on the actual free welding side measures the local two-board passage.
The customer 35mm test is never applied to arbitrary nearest-face distances.
No labels, piece names, absolute-coordinate tables or fitted rule head.
"""
import math
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_direction_spans_v247 as prior
from . import curveonly_01a1ac7c_source_selection_v56 as provenance
from . import curveonly_01a1ac7c_material_sections_v124 as section
from . import curveonly_01a1ac7c_finite_shadow_v73 as shadow
from . import curveonly_01a1ac7c_structural_event_cells_v298 as events
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.gp import gp_Lin, gp_Dir
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_IN, TopAbs_ON

NUMERIC = 1e-6
PASSAGE_MM = 35.


def point_tangent(points, station, q):
    i = min(len(points)-2, max(0, np.searchsorted(station, q, side='right')-1))
    tangent = points[i+1]-points[i]
    tangent /= np.linalg.norm(tangent)
    point = np.array([np.interp(q, station, points[:, j]) for j in range(3)])
    return point, tangent


class Geometry:
    def __init__(self, solids, packet, inventory):
        self.solids = solids
        self.T = np.asarray(packet['pose']['transform_STEP_to_XYZ'])
        self.R, self.t = self.T[:3, :3], self.T[:3, 3]
        self.base = np.asarray(packet['base_axis'], float)
        self.base /= np.linalg.norm(self.base)
        self.faces = inventory['faces']
        self.classifiers = [BRepClass3d_SolidClassifier(s) for s in solids]
        self.intersectors = []
        for solid in solids:
            intersector = IntCurvesFace_ShapeIntersector()
            intersector.Load(solid, NUMERIC)
            self.intersectors.append(intersector)
        self.pairs = []
        index = 0
        for mi, solid in enumerate(solids):
            planes = []
            for face in cad.items(solid, cad.TopAbs_FACE):
                row = self.faces[index]
                assert row['material'] == mi
                if row['status'] == 'CANONICAL_PLANE':
                    n = np.asarray(row['normal'], float)
                    world = n@self.R.T
                    if abs(world@self.base) <= NUMERIC:
                        props = GProp_GProps()
                        BRepGProp.SurfaceProperties_s(face, props)
                        planes.append((index, n, float(row['offset_mm']), props.Mass()))
                index += 1
            choices = []
            for i, a in enumerate(planes):
                for b in planes[:i]:
                    sign = float(np.sign(a[1]@b[1]))
                    if np.linalg.norm(a[1]-sign*b[1]) > NUMERIC:
                        continue
                    thickness = abs(a[2]-sign*b[2])
                    if thickness > NUMERIC:
                        choices.append((min(a[3], b[3]), a, b, thickness))
            if choices:
                _, a, b, thickness = max(choices, key=lambda x:x[0])
                self.pairs.append(dict(material=mi, faces=[a[0], b[0]],
                    normal=a[1], limits=sorted((a[2], b[2]*float(np.sign(a[1]@b[1])))),
                    thickness_mm=thickness))

    def inside(self, material, world):
        q = (world-self.t)@self.R
        c = self.classifiers[material]
        c.Perform(cad.gp_Pnt(*map(float, q)), NUMERIC)
        return c.State() in (TopAbs_IN, TopAbs_ON)

    def free_side(self, ci, q, packet):
        c = packet['candidates'][ci]
        points = np.asarray(c['points']); station = cad.arclength(points)
        point, tangent = point_tangent(points, station, q)
        normal = np.cross(tangent, self.base)
        normal /= max(np.linalg.norm(normal), 1e-300)
        owner = shadow.owner_at(ci, q, packet)
        if owner < 0:
            return None
        signs = []
        for sign in (-1., 1.):
            # Same pre-existing 1/8mm cross-section probes: side identity only.
            occupied = self.inside(owner, point+sign*normal+self.base*3.)
            opposite = self.inside(owner, point-sign*normal+self.base*3.)
            below = [mi for mi in range(len(self.solids)) if mi != owner and
                self.inside(mi, point+sign*normal-self.base)]
            if not occupied and opposite and below:
                signs.append((sign*normal, owner, below))
        if len(signs) != 1:
            return None
        return point, tangent, signs[0][0], owner, signs[0][2]

    def passage(self, material, point, outward):
        native_point = (point-self.t)@self.R
        native_direction = outward@self.R
        intersector = self.intersectors[material]
        intersector.Perform(gp_Lin(cad.gp_Pnt(*map(float, native_point)),
            gp_Dir(*map(float, native_direction))), NUMERIC, 1e6)
        if not intersector.IsDone():
            return None
        hits = [float(intersector.WParameter(i)) for i in range(1, intersector.NbPnt()+1)
                if intersector.WParameter(i) >= NUMERIC]
        return min(hits) if hits else None


def crossings(ci, packet, inventory, geometry):
    c = packet['candidates'][ci]
    points = np.asarray(c['points']); station = cad.arclength(points)
    raw = inventory['candidates'][ci]['qualified_geometry_events']
    blocks, review = [], []
    for pair in geometry.pairs:
        roots = []
        for face in pair['faces']:
            values = sorted(set(round(v['q_mm'], 9) for v in raw
                if v['kind'] == 'FACE_CARRIER_ROOT' and v['face'] == face))
            roots.append(values)
        nodes = sorted(set([0., float(station[-1])]+roots[0]+roots[1]))
        for a, b in zip(nodes[:-1], nodes[1:]):
                lo, hi = sorted((a, b))
                if hi-lo <= NUMERIC:
                    continue
                state = geometry.free_side(ci, (lo+hi)/2, packet)
                if state is None:
                    continue
                point, tangent, outward, owner, support = state
                native = (point-geometry.t)@geometry.R
                coordinate = float(native@pair['normal'])
                if not pair['limits'][0]-NUMERIC <= coordinate <= pair['limits'][1]+NUMERIC:
                    continue
                if pair['material'] == owner or pair['material'] in support:
                    continue
                n = pair['normal']@geometry.R.T
                # Its large plate face must cross the longitudinal path,
                # rather than be another longitudinal or horizontal panel.
                if abs(n@tangent) <= abs(n@outward):
                    continue
                # Follow the transverse plate's own middle plane. A normal
                # ray to a curved longitudinal wall can leave an oblique plate
                # slab before reaching its finite notched edge.
                direction = np.cross(n, geometry.base)
                direction /= max(np.linalg.norm(direction), 1e-300)
                direction *= 1. if direction@outward > 0. else -1.
                passage = geometry.passage(pair['material'], point, direction)
                record = dict(material=pair['material'], side_faces=pair['faces'],
                    start_mm=lo, end_mm=hi, measured_two_board_passage_mm=passage,
                    owner=owner, supporting_materials=support,
                    welding_side='FREE_SIDE_ABOVE_VERIFIED_SUPPORT',
                    side_pair_thickness_mm=pair['thickness_mm'],
                    passage_direction='WITHIN_TRANSVERSE_PLATE_MIDPLANE_TOWARD_FREE_WELDING_SIDE',
                    transverse_normal_tangent_dot=abs(float(n@tangent)))
                review.append(record)
                if passage is not None and passage <= PASSAGE_MM:
                    blocks.append(record)
    return blocks, review


def union(ranges):
    result = []
    for a, b in sorted(ranges):
        if result and a <= result[-1][1]+NUMERIC:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def complement(length, blocks):
    result, last = [], 0.
    for a, b in union((v['start_mm'], v['end_mm']) for v in blocks):
        a, b = max(0., a), min(length, b)
        if a-last > NUMERIC:
            result.append((last, a))
        last = max(last, b)
    if length-last > NUMERIC:
        result.append((last, length))
    return result


def apply(raw, features, packet, inventory, solids):
    """Replace only evidence-qualified notched upright source families."""
    rows = prior.candidate_rows(features, packet)
    geometry = Geometry(solids, packet, inventory)
    lengths = [r['length_mm'] for r in rows]
    eligible, audits = [], []
    for ci, (c, row) in enumerate(zip(packet['candidates'], rows)):
        grid = row['grid']
        # This branch is the upright curved panel on a horizontal support.
        if row['closed'] or not (c.get('STEP') and c.get('pose')):
            continue
        if np.min(grid[:, 0]) < math.cos(math.radians(3.)) or np.max(grid[:, 1]) > math.sin(math.radians(3.)):
            continue
        if np.min(grid[:, 21]) < 1. or np.max(np.expm1(grid[:, 22])) > 3.:
            continue
        # A base below and an upright owner above, on opposite local sides.
        probes = [geometry.free_side(ci, q, packet) for q in np.linspace(0., lengths[ci], 5)[1:-1]]
        if any(q is None for q in probes):
            continue
        blocks, review = crossings(ci, packet, inventory, geometry)
        audits.append(dict(candidate=ci, blocks=blocks, crossing_review=review))
        if blocks:
            eligible.append((ci, blocks))
    # Literal child/native provenance: process the maximal qualified carrier.
    footprints = {ci:provenance.primitive_intervals(ci, 0., lengths[ci], packet, lengths)
                  for ci, _ in eligible}
    chosen = []
    for ci, blocks in sorted(eligible, key=lambda q:-lengths[q[0]]):
        if any(all(any(root == oldroot and a >= x-NUMERIC and b <= y+NUMERIC
                       for oldroot, x, y in footprints[old])
                   for root, a, b in footprints[ci]) for old, _ in chosen):
            continue
        chosen.append((ci, blocks))
    replacements, covered = [], []
    for ci, blocks in chosen:
        covered.extend(footprints[ci])
        for a, b in complement(lengths[ci], blocks):
            proposal = prior.codec.proposals(packet, [rows[ci]],
                [prior.previous.intervals_to_codes(rows[ci], [dict(a=a, b=b, confidence=1.)])])[0]
            proposal['structural_rule'] = 'TRANSVERSE_PLATE_LOCAL_PASSAGE_AND_SIDE_PAIR_V338'
            replacements.append(proposal)
    kept, removed = [], []
    for i, p in enumerate(raw):
        q = p['interval']
        spans = provenance.primitive_intervals(p['candidate'], q['start_mm'], q['end_mm'], packet, lengths)
        overlap = any(root == cr and min(b, y)-max(a, x) > provenance.MIN_OVERLAP_MM
                      for root, a, b in spans for cr, x, y in covered)
        (removed if overlap else kept).append(i)
    return [raw[i] for i in kept]+replacements, dict(branch='V338',
        candidate_audits=audits, chosen_candidates=[q[0] for q in chosen],
        replaced_raw_proposals=removed, new_rule_proposals=len(replacements),
        gap_width_from_finite_side_face_pair=True, passage_limit_mm=PASSAGE_MM,
        nominal_STEP_not_actual_geometry_acceptance=True)
