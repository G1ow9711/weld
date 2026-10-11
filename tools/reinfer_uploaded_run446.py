"""Original run446 model on uploaded frozen inputs, isolated outputs."""
from pathlib import Path
import shutil,runpy,sys
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import OUT,ROOT,read,write,digest
from weld_pipeline.sjg_child_parent_model_1256 import infer

def main():
    dest=ROOT/'server_rebuild/run446_uploaded_input_inference_v1';dest.mkdir(exist_ok=False)
    protocol=read(OUT/'run444/registered_protocol.json')
    for name,expected in protocol['module_sha256'].items():assert digest(ROOT/'weld_pipeline'/name)==expected
    for name in ('models.pkl','training_receipt.json'):
        shutil.copy2(OUT/'run446'/name,dest/name);assert digest(dest/name)==digest(OUT/'run446'/name)
    write(dest/'protocol.json',dict(scope='Original model inference from uploaded frozen upstream inputs. Not raw XYZ recomputation.',original_protocol_sha256=digest(OUT/'run444/registered_protocol.json'),script_sha256=digest(__file__),source_modules_verified=True,protected_DEV_reads=0))
    keys=('21','BM79A_native_local','10','25','36')
    for key in keys:infer(dest,key)
    scorer=ROOT/'.codex-local/evaluate_five_case.py';sys.argv=[str(scorer),str(dest),*keys];runpy.run_path(str(scorer),run_name='__main__')
    cases=read(dest/'evaluation_subset.json')['cases'];tp=sum(c['TP'] for c in cases);fp=sum(c['FP'] for c in cases);g=sum(c['G'] for c in cases)
    write(dest/'receipt.json',dict(status='FROZEN_INPUT_INFERENCE_AND_SCORING_COMPLETE',TP=tp,FP=fp,G=g,rate_percent=100*tp/g,historical_counts_match=(tp,fp,g)==(259,5,269),decoded_bytes_match={key:digest(dest/f'test_{key}/final_decoded.json')==digest(OUT/f'run446/test_{key}/final_decoded.json') for key in keys},scope='Frozen upstream input inference only. Full raw XYZ chain not reproduced by this test.'))
    print('FROZEN_INPUT_RESULT',tp,fp,g,flush=True)

if __name__=='__main__':main()
