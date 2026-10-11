"""Candidate delivery schema with resolvable along-line attribute references."""
import argparse
import json
from pathlib import Path

from weld_pipeline.predict_hybrid_with_attributes import predict as base_predict


def predict(pcd,checkpoint,outdir):
    predictions,receipt=base_predict(pcd,checkpoint,outdir)
    out=Path(outdir)
    path=out/'attribute_profiles.json'
    data=json.loads(path.read_text('utf8'))
    data['children']={r['parent_id']+'_'+str(r['child_index']+1):r for r in data['children']}
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    schema_path=out/'output_schema.json'
    schema=json.loads(schema_path.read_text('utf8'))
    schema.update(attribute_profile_reference='attribute_profiles.json#<weld_id>; keyed child dictionary',
        attribute_estimator_version='learned_cad_candidate_plate_attributes_v1',
        candidate_attribute_fields=['thickness_1','thickness_2','normal_1','normal_2'],
        candidate_scalar_definition='CSV values refer to the midpoint; all five along-line sample values remain in the profile.',
        material_direction='Outward material normals, unit vectors. Side order is right handed with the directed child tangent.',
        customer_side_schema_frozen=False,full_contract_output_compliant=False,
        missing_features=['reviewed side schema','reviewed split reasons','remaining attribute estimators','independent attribute acceptance'])
    schema['columns'].append('attribute_profile_reference')
    schema_path.write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf8')
    receipt.update(output_schema_version='candidate_attributes_v2',full_contract_pass=False)
    (out/'inference_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    return predictions,receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pcd',required=True);parser.add_argument('--checkpoint',required=True);parser.add_argument('--outdir',required=True)
    args=parser.parse_args();predict(args.pcd,args.checkpoint,args.outdir)
