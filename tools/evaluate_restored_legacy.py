"""Score frozen reconstructed predictions using the original parent-structure gate."""
import argparse
from pathlib import Path
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents,controls
ROOT=Path(__file__).resolve().parents[1]
BASE=Path(__import__('os').environ.get('WELD_RESTORE_ROOT',str(ROOT/'server_rebuild/best_baseline_restore'))).resolve()
BASE.relative_to(ROOT)

def evaluate(key,model='run379'):
    folder=BASE/'legacy_runtime'/('source_'+key)/model/('test_'+key)
    frozen=read(folder/('final_prediction.json' if model=='run379' else 'prediction_receipt.json'));pred=folder/'final_decoded.json'
    assert digest(pred)==frozen['decoded_sha256']
    labels=ROOT/'server_rebuild/labels_v1'/key;receipt=read(labels/'receipt.json');target=labels/'targets.json'
    assert digest(target)==receipt['targets_sha256']
    input_receipt=read(ROOT/'server_rebuild/inputs_v1'/key/'receipt.json')
    assert receipt['files'][input_receipt['source_file']]==input_receipt['source_sha256']
    runtime=BASE/'legacy_runtime'/('source_'+key)
    case=read(runtime/'run98/local_cohort.json')['cases'][0]
    assert case['input_sha256']==input_receipt['xyz_sha256']
    truth=read(target);assert truth['original_children']==receipt['children']
    control=controls()
    result=score_native_parents(read(pred),truth,[False]*len(truth['parents']))
    result.update(source=key,role='TRAIN',model='restored_'+model,prediction_sha256=digest(pred),target_sha256=digest(target),input_sha256=case['input_sha256'],controls=control,fit_calls=0,independent_acceptance=False,historical_input_parity_proven=False,full_contract_pass=False)
    write(folder/'evaluation.json',result)
    print(key,result['categories']['overall'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);ap.add_argument('--model',choices=['run379','run446'],default='run379');args=ap.parse_args();evaluate(args.key,args.model)
