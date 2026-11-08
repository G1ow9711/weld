"""Final-output regression for preserving existing structural model gaps."""
import sys
import types
from tools import curveonly_01a1ac7c_transverse_control_v339 as original
from tools import curveonly_01a1ac7c_owner_control_v341 as preceding
from weld_pipeline import curveonly_01a1ac7c_preserved_gaps_v342 as rules

e = original.e
DEST, OUTPUT = e.REC/'preserved_gap_control_v343', e.OUT/'preserved_gap_control_v343'
QUICK = original.QUICK+('curve_baf6eacd4a0bf4d2',)


def namespace():
    return dict(original.__dict__, __file__=__file__, DEST=DEST, OUTPUT=OUTPUT,
        rules=rules, check=check, one=one, QUICK=QUICK)


def check():
    return types.FunctionType(original.check.__code__, namespace())()


def prepare():
    assert e.read(preceding.DEST/'full/CLOSURE.json')['analysis_closed']
    types.FunctionType(original.prepare.__code__, namespace())()
    p = e.read(DEST/'FROZEN_PROTOCOL.json')
    e.write(DEST/'FROZEN_PROTOCOL.json', dict(p, sources=p['sources']+[
        e.bind(original.__file__), e.bind(rules.previous.__file__), e.bind(rules.previous.previous.__file__)],
        preceding_targeted_full=e.bind(preceding.DEST/'full/CLOSURE.json'),
        single_repair='Preserve frozen R35 process-gap decisions at the same finite transverse side-face pair on the welding side; customer35 unchanged.'))


def one(job, phase):
    return types.FunctionType(original.one.__code__, namespace())(job, phase)


def predict(phase):
    return types.FunctionType(original.predict.__code__, namespace())(phase)


def score(phase):
    return types.FunctionType(original.score.__code__, namespace())(phase)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare()
    else: {'predict':predict, 'score':score}[sys.argv[1]](sys.argv[2])
