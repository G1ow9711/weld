"""Re-score uploaded frozen predictions in separate output; no model inference."""
from pathlib import Path
import runpy,sys,shutil
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
ROOT=Path(__file__).resolve().parents[1]

def main():
    dest=ROOT/'server_rebuild/run446_uploaded_prediction_recheck_v1';dest.mkdir(exist_ok=False)
    keys=('21','BM79A_native_local','10','25','36');records=[]
    for key in keys:
        src=OUT/'run446'/f'test_{key}';folder=dest/f'test_{key}';folder.mkdir()
        for name in ('final_decoded.json','prediction_receipt.json'):
            p=src/name;assert p.is_file();shutil.copy2(p,folder/name)
            assert digest(p)==digest(folder/name)
        receipt=read(folder/'prediction_receipt.json');assert digest(folder/'final_decoded.json')==receipt['decoded_sha256']
        records.append(dict(key=key,decoded_sha256=receipt['decoded_sha256']))
    scorer=ROOT/'.codex-local/evaluate_five_case.py'
    write(dest/'protocol.json',dict(scope='Uploaded frozen prediction re-scoring only; no frozen-input inference or raw XYZ recomputation',cases=records,scorer_sha256=digest(scorer),original_summary_sha256=digest(OUT/'run446/evaluation_summary.json'),protected_DEV_reads=0))
    sys.argv=[str(scorer),str(dest),*keys];runpy.run_path(str(scorer),run_name='__main__')
    cases=read(dest/'evaluation_subset.json')['cases'];tp=sum(c['TP'] for c in cases);fp=sum(c['FP'] for c in cases);g=sum(c['G'] for c in cases);historical=read(OUT/'run446/evaluation_summary.json')['TRAIN'];assert (tp,fp,g)==(historical['TP'],historical['FP'],historical['G'])
    write(dest/'receipt.json',dict(status='UPLOADED_FROZEN_PREDICTION_SCORE_REPRODUCED',TP=tp,FP=fp,G=g,extraction_correct_rate_percent=100*tp/g,scope='Re-scoring only. Not inference reproducibility or formal acceptance.',protocol_sha256=digest(dest/'protocol.json')))
    print('REPRODUCED_FROZEN_SCORE',tp,fp,g,flush=True)

if __name__=='__main__':main()
