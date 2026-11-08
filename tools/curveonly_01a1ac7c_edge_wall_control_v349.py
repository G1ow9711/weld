"""TRAIN first, complete final outputs for the curved base-edge structural rule."""
import sys
import types
from tools import curveonly_01a1ac7c_transverse_control_v339 as original
from tools import curveonly_01a1ac7c_preserved_gap_control_v343 as preceding
from weld_pipeline import curveonly_01a1ac7c_edge_wall_rules_v348 as rules

e = original.e
DEST, OUTPUT = e.REC/'edge_wall_control_v349', e.OUT/'edge_wall_control_v349'
QUICK = tuple(c['source_id'] for c in e.read(e.REC/'native_coverage_control_v296/INPUTS.json')['cases']
              if c['split'] == 'TRAIN')


def namespace():
    return dict(original.__dict__, __file__=__file__, DEST=DEST, OUTPUT=OUTPUT,
        rules=rules, check=check, one=one, QUICK=QUICK)


def check():
    return types.FunctionType(original.check.__code__, namespace())()


def prepare():
    assert e.read(preceding.DEST/'full/CLOSURE.json')['analysis_closed']
    assert e.read(preceding.DEST/'full/CLOSURE.json')['passed_research_adoption']
    types.FunctionType(original.prepare.__code__, namespace())()
    baseline = e.read(preceding.DEST/'full/RESULT.json')
    e.write(DEST/'BASELINE_V343_SCORES.json', dict(original_cases=baseline['cases'], additional_cases=[],
        source=e.bind(preceding.DEST/'full/RESULT.json'), same_V60=True))
    p = e.read(DEST/'FROZEN_PROTOCOL.json')
    sources = p['sources']+e.read(preceding.DEST/'FROZEN_PROTOCOL.json')['sources']+[
        e.bind(original.__file__), e.bind(rules.previous.__file__),
        e.bind(rules.finite.__file__), e.bind(rules.aliases.__file__)]
    e.write(DEST/'FROZEN_PROTOCOL.json', dict(p, sources=sources,
        baseline_scores=e.bind(DEST/'BASELINE_V343_SCORES.json'),
        source_replay_baseline='R35 executed neural proposals replayed exactly; compared final baseline is adopted V343',
        preceding_targeted_full=e.bind(preceding.DEST/'full/CLOSURE.json'),
        design=e.bind(e.REC/'EDGE_WALL_RULE_DESIGN_V348.md'),
        single_repair='Recover native curved base-edge arcs only with unique base free side, a same finite upright above/below all <=1mm stations, old strict contact observation, and no transverse process gap.',
        contact_distance_is_not_permission_alone=True,
        no_validation_derived_thresholds=True, no_new_fitted_weights=True,
        quick_is_all22TRAIN_only=True, quick_is_exposed_TRAIN_and_iterative_VAL=False))


def one(job, phase):
    return types.FunctionType(original.one.__code__, namespace())(job, phase)


def predict(phase):
    return types.FunctionType(original.predict.__code__, namespace())(phase)


def score(phase):
    return types.FunctionType(original.score.__code__, namespace())(phase)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare()
    else: {'predict':predict, 'score':score}[sys.argv[1]](sys.argv[2])
