"""Reject ambiguous states and preserve endpoint identity on a real contact."""
import json
from pathlib import Path

import numpy as np

from weld_pipeline.curved_cache import load_piece
from weld_pipeline.endpoint_process_attributes import child_profiles
from weld_pipeline.prepare_contract_child import child_attributes
from weld_pipeline.train_candidate_tree import mapped_references
from weld_pipeline.train_endpoint_process_attributes import known_label
from weld_pipeline.train_horizontal_patch_tree import context


def main():
    def attr(value):
        return {'truth_available':True,'value':value}
    assert known_label('front_wall',attr(-1)) is None
    assert known_label('start_height',attr(0)) is None
    assert known_label('start_height',attr(-1)) is None
    assert known_label('start_hole_height',attr(0)) == {'state':'no_hole','value':0.}
    assert known_label('start_base_edge',attr(51)) == {'state':'over_50','value':51.}
    parent={'metadata':{'HasEave (Any)':True,'IsSplitedForDualRobot (Any)':True,
        'startHeight':10.,'endHeight':12.},'segments':[{}, {}, {}]}
    labels=child_attributes(parent,1)
    for field in ['has_eave','side_plate_split','start_height','end_height']:
        assert known_label(field,labels[field]) is None
    print('PASS ambiguous sentinels and internal endpoint/Any exclusions')
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    name=split['train'][0];piece=load_piece(split['piece_roots'][name],verify=True)
    structural,mean,covariance=context(split,name)
    mapping=mapped_references(piece);si,refs=next(iter(mapping.items()))
    p=np.asarray(refs[0]['segments'][0]['points'],float)
    a=child_profiles(piece['sequences'][si],p,structural[si],mean,covariance)
    b=child_profiles(piece['sequences'][si],p[::-1],structural[si],mean,covariance)
    for first,last in [('start_height','end_height'),('start_hole_height','end_hole_height'),
        ('start_base_edge','end_base_edge'),('front_wall','back_wall')]:
        np.testing.assert_allclose(a[first],b[last],atol=2e-7,rtol=2e-7)
    np.testing.assert_allclose(a['has_eave'],b['has_eave'],atol=2e-7,rtol=2e-7)
    print('PASS physical endpoint identity and reversal correspondence')


if __name__=='__main__':
    main()
