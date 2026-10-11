"""Point-only inference with the expanded, still unreviewed plate estimator."""
import argparse
import json
from pathlib import Path

from weld_pipeline.predict_candidate_model import predict as previous_predict


def predict(pcd,checkpoint,outdir):
    predictions,receipt=previous_predict(pcd,checkpoint,outdir)
    path=Path(outdir)/'output_schema.json';schema=json.loads(path.read_text('utf8'))
    schema.update(output_schema_version='expanded_candidate_attributes_v3',
        attribute_training_scope='Nineteen FIT pieces; unreviewed STEP-derived face/thickness reference candidates. Two checks excluded from training.',
        attribute_confidence_definition='Minimum predicted material-direction class confidence over all five samples. This is not calibrated thickness uncertainty.',
        customer_side_schema_frozen=False,full_contract_output_compliant=False)
    path.write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf8')
    receipt.update(output_schema_version='expanded_candidate_attributes_v3',full_contract_pass=False)
    (Path(outdir)/'inference_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    return predictions,receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pcd',required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--outdir',required=True)
    args=parser.parse_args();predict(args.pcd,args.checkpoint,args.outdir)
