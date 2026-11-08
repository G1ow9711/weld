"""Frozen local windows augmented with finite material volume observations."""
import time
import types
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_local_intervals_v95 as legacy
from . import curveonly_01a1ac7c_structured_intervals_v115 as sequence
from . import curveonly_01a1ac7c_local_runtime_v112 as previous
from . import curveonly_01a1ac7c_material_sections_v124 as material
from OCP.BRepCheck import BRepCheck_Analyzer

RAW_COLUMNS = 34
SECTION_COLUMNS = len(material.NAMES)
INPUT_COLUMNS = RAW_COLUMNS + SECTION_COLUMNS
OBSERVATION_COLUMNS = len(legacy.RAW_COLUMNS) + 12 + SECTION_COLUMNS
WINDOW_MM = legacy.WINDOW_MM
FEATURES = OBSERVATION_COLUMNS * len(WINDOW_MM) + len(WINDOW_MM)
NAMES = tuple(cad.NAMES) + tuple('solid_section_' + q for q in material.NAMES)
assert RAW_COLUMNS == len(cad.NAMES) and SECTION_COLUMNS == 74
assert INPUT_COLUMNS == len(NAMES) == 108 and FEATURES == 756


def augment(features, packet, sections, receipt):
    features = np.asarray(features)
    sections = np.asarray(sections)
    assert features.ndim == 2 and features.shape[1] == RAW_COLUMNS
    assert features.dtype == np.float32 and sections.dtype == np.float32
    assert sections.shape == (len(features), SECTION_COLUMNS)
    assert np.isfinite(features).all() and np.isfinite(sections).all()
    combined = np.column_stack([features, sections]).astype(np.float32)
    assert np.array_equal(combined[:, :RAW_COLUMNS], features)
    metadata = dict(raw_prefix_columns=RAW_COLUMNS, section_columns=SECTION_COLUMNS,
                    section_definition='FINITE_SOLID_SECTION_V124', receipt=receipt,
                    unavailable_is_unknown_not_a_background_target=True)
    return combined, dict(packet, material_sections_v124=metadata)


def observations(features, packet):
    features = np.asarray(features)
    assert features.ndim == 2 and features.shape[1] == INPUT_COLUMNS
    metadata = packet['material_sections_v124']
    assert metadata['raw_prefix_columns'] == RAW_COLUMNS and metadata['section_columns'] == SECTION_COLUMNS
    old = legacy.observations(features[:, :RAW_COLUMNS], packet)
    result = np.column_stack([old, features[:, RAW_COLUMNS:]]).astype(np.float32)
    assert result.shape == (len(features), OBSERVATION_COLUMNS) and np.isfinite(result).all()
    return result


def encode(features, packet):
    namespace = dict(legacy.__dict__, observations=observations, FEATURES=FEATURES)
    return types.FunctionType(legacy.encode.__code__, namespace)(features, packet)


class Raw34Baseline:
    """Give the unchanged owned global model exactly its original 34 columns."""
    def __init__(self, model):
        self.model = model

    @property
    def last_timing(self):
        return self.model.last_timing

    def predict(self, features, packet):
        features = np.asarray(features)
        assert features.ndim == 2 and features.shape[1] == INPUT_COLUMNS
        return self.model.predict(features[:, :RAW_COLUMNS], packet)


class LocalCurveSet(legacy.LocalCurveSet):
    def predict(self, features, packet):
        namespace = dict(legacy.__dict__, encode=encode, FEATURES=FEATURES)
        return types.FunctionType(legacy.LocalCurveSet.predict.__code__, namespace)(self, features, packet)


class StructuredCurveSet(sequence.StructuredCurveSet):
    def predict(self, features, packet):
        pointwise = types.SimpleNamespace(encode=encode, FEATURES=FEATURES, intervals=legacy.intervals)
        namespace = dict(sequence.__dict__, pointwise=pointwise)
        return types.FunctionType(sequence.StructuredCurveSet.predict.__code__, namespace)(self, features, packet)


class LocalRefinement(legacy.LocalRefinement):
    def __init__(self, baseline, local):
        if not isinstance(baseline, Raw34Baseline):
            baseline = Raw34Baseline(baseline)
        super().__init__(baseline, local)


def input_curves(record):
    start = time.perf_counter(); captured = {}
    proxy = types.ModuleType('curveonly_section_material_capture')
    proxy.__dict__.update(cad.__dict__)

    def load(path):
        solids = cad.load_material(path); captured['solids'] = solids
        return solids

    proxy.load_material = load
    read = types.FunctionType(previous.input_curves.__code__, dict(previous.__dict__, cad=proxy))
    features, packet = read(record)
    solids = captured.get('solids')
    validity = [bool(BRepCheck_Analyzer(q, True).IsValid()) for q in solids] if solids else []
    invalid = bool(validity and not all(validity))
    sections, receipt = material.build(packet, None if invalid else solids)
    if invalid:
        receipt.update(reason='INVALID_FINITE_MATERIAL_BREP', invalid_or_missing_material_not_absent=True)
    if not solids:
        receipt.update(reason='NO_STEP_INPUT', invalid_or_missing_material_not_absent=True)
    receipt = dict(receipt, real_solid_validity=validity)
    combined, packet = augment(features, packet, sections, receipt)
    packet['timing'] = dict(packet['timing'], solid_section_seconds=receipt['seconds'],
                            input_to_features=time.perf_counter()-start)
    return combined, packet


def decode(features, packet, model, threshold=.5):
    assert isinstance(model, LocalRefinement)
    return previous.decode(features, packet, model, threshold)


arclength = cad.arclength
