"""Run an unseen-to-this-checkpoint development assembly with label reads denied."""
import builtins,io,json
from pathlib import Path
from weld_pipeline.predict_contact import predict


def main():
    root=Path.cwd();folder=root/'DataSet/9月20日/224-GR24A1';pcd=next(folder.glob('*AllFaces.pcd')).resolve()
    checkpoint=root/'weld_pipeline/runs/contact_single_piece_v2/best.pt'
    out=root/'weld_pipeline/runs/contact_single_piece_v2/development_validation_GR24A1'
    forbidden=[(root/'DataSet').resolve(),(root/'data').resolve(),(root/'training/data').resolve()]
    reads=[];original_open=builtins.open;original_io=io.open
    def check(file,mode):
        if isinstance(file,(str,Path)) and 'r' in str(mode):
            path=Path(file).resolve()
            if any(path.is_relative_to(d) for d in forbidden):
                if path!=pcd:raise RuntimeError('Reference/derived dataset read forbidden during inference: '+str(path))
                reads.append(str(path))
    def guarded_open(file,mode='r',*a,**k):check(file,mode);return original_open(file,mode,*a,**k)
    def guarded_io(file,mode='r',*a,**k):check(file,mode);return original_io(file,mode,*a,**k)
    builtins.open=guarded_open;io.open=guarded_io
    try:predict(pcd,checkpoint,out)
    finally:builtins.open=original_open;io.open=original_io
    # Reference data become available only AFTER the prediction files exist.
    from weld_pipeline.truth import load_truth
    from weld_pipeline.evaluation import score_welds
    pred=json.loads((out/'predictions.json').read_text());truth,_=load_truth(folder);score=score_welds(pred,truth)
    score['scope']='Different development assembly; this checkpoint was trained only on 161-LB18A. Not a fresh formal blind test.'
    score['inference_input_guard']={'status':'PASS','source_reads':reads,'reference_and_cached_label_reads_denied':True}
    score['by_type']={}
    for typ in ['horizontal','vertical']:
        score['by_type'][typ]=score_welds([p for p in pred if p['type']==typ],[t for t in truth if t['type']==typ])
    (out/'independent_score.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
    print('DEVELOPMENT_VALIDATION',{k:score[k] for k in ['truth_count','prediction_count','valid_welds','valid_extraction_rate','false_or_duplicate_rate_per_truth']})


if __name__=='__main__':main()
