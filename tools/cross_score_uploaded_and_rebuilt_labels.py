"""Cross frozen predictions and both label versions to locate score differences."""
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest
from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions
from weld_pipeline.sjg_recall_set_worker_1162 import score_value,target_for

def main():
 rows=[]
 for key in ('21','BM79A_native_local','10','25','36'):
  original,binding=target_for(key);newpath=ROOT/f'server_rebuild/labels_v1/{key}/targets.json';new=read(newpath)
  preds={'uploaded':OUT/f'run446/test_{key}/final_decoded.json','rebuilt':ROOT/f'server_rebuild/best_baseline_restore/legacy_runtime/source_{key}/run446/test_{key}/final_decoded.json'}
  results={}
  for pname,path in preds.items():
   p=read_shared_predictions(path)
   for lname,t in [('original',original),('rebuilt',new)]:
    r=score_value(p,t,key)['report'];results[pname+'__'+lname]=r;print(key,pname,lname,r['TP'],r['FP'],flush=True)
  rows.append(dict(key=key,original_target_sha256=binding['target_sha256'],rebuilt_target_sha256=digest(newpath),results=results))
 dest=ROOT/'server_rebuild/run446_uploaded_vs_rebuilt_audit_v1'
 write(dest/'cross_label_scores.json',dict(cases=rows,scope='Same original scorer, 2x2 frozen prediction/label crossing, five TRAIN only. No threshold or target modification.',script_sha256=digest(__file__)))

if __name__=='__main__':main()
