"""Read isolation for a real raw-PCD development validation, then scoring."""
import argparse,builtins,io,json
from pathlib import Path
from weld_pipeline.predict_event_gap import predict


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--outdir',required=True);ap.add_argument('--excel',action='store_true');ap.add_argument('--folder')
    ap.add_argument('--reference-folder',help='Separately audited reference package, read only after inference')
    ap.add_argument('--fast-mixed',action='store_true')
    ap.add_argument('--decoder-config')
    ap.add_argument('--role',choices=['development','training'],default='development');a=ap.parse_args()
    root=Path.cwd();folder=Path(a.folder) if a.folder else root/'DataSet/9月20日'/a.group
    if folder.name!=a.group:raise ValueError('Group and reference folder name differ')
    pcd=next(folder.glob('*AllFaces.pcd')).resolve()
    forbidden=[(root/p).resolve() for p in ['DataSet','data','training/data','weld_pipeline/data',
        'weld_pipeline/reports/rigid_vertical_augmentation','weld_pipeline/reports/short_vertical_augmentation',
        'weld_pipeline/reports/partial_vertical_augmentation']]
    reads=[];original_open=builtins.open;original_io=io.open
    def check(file,mode):
        if isinstance(file,(str,Path)) and ('r' in str(mode) or '+' in str(mode)):
            path=Path(file).resolve()
            if any(path.is_relative_to(d) for d in forbidden):
                if path!=pcd:raise RuntimeError('Reference or cached dataset read forbidden during inference: '+str(path))
                reads.append(str(path))
    def guarded_open(file,mode='r',*args,**kwargs):check(file,mode);return original_open(file,mode,*args,**kwargs)
    def guarded_io(file,mode='r',*args,**kwargs):check(file,mode);return original_io(file,mode,*args,**kwargs)
    builtins.open=guarded_open;io.open=guarded_io
    try:pred,receipt=predict(pcd,a.checkpoint,a.outdir,a.excel,a.fast_mixed,a.decoder_config)
    finally:builtins.open=original_open;io.open=original_io
    # Customer truth becomes accessible only after predictions and CSV exist.
    from weld_pipeline.truth import load_truth
    from weld_pipeline.evaluation import score_welds
    from weld_pipeline.curve_geometry_metrics import check_matched_curves
    from training.labels import sha256
    reference_folder=Path(a.reference_folder) if a.reference_folder else folder
    if reference_folder.name!=a.group:raise ValueError('Reference package group differs')
    if a.reference_folder:
        manifest=json.loads((reference_folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status')!='DERIVED_REFERENCE_FRAME_CHECKED':raise ValueError('Reference package not audited')
        for source,wanted in manifest['outputs'].items():
            if sha256(source)!=wanted:raise ValueError('Derived reference package changed')
    truth,missing=load_truth(reference_folder);score=score_welds(pred,truth)
    score['curve_geometry']=check_matched_curves(pred,truth,score)
    score['reference_hashes']={str(p):sha256(p) for p in [*reference_folder.glob('*.xlsx'),*reference_folder.glob('*welds.json'),*reference_folder.glob('manifest.json')]}
    score['evaluation_source_hashes']={str(p):sha256(p) for p in [Path(__file__),
        Path('weld_pipeline/evaluation.py'),Path('weld_pipeline/truth.py'),
        Path('weld_pipeline/customer_label_corrections.py'),Path('weld_pipeline/curve_geometry_metrics.py')]}
    score['input_guard']={'status':'PASS','source_reads':reads,'derived_and_reference_inputs_denied':True}
    score['by_type']={}
    for typ in ['horizontal','vertical']:
        subset=[w for w in truth if w['type']==typ]
        if subset:score['by_type'][typ]=score_welds([w for w in pred if w['type']==typ],subset)
    score['missing_reference_curves']=missing;score['dataset_role']=a.role
    score['scope']='Training-piece raw inference check. Not generalization or customer acceptance.' if a.role=='training' else 'Development validation. Not a new blinded customer acceptance.'
    (Path(a.outdir)/'independent_score.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
    print('RAW_SCORE',{k:score[k] for k in ['truth_count','prediction_count','valid_welds','valid_extraction_rate','false_or_duplicate_rate_per_truth']},flush=True)


if __name__=='__main__':main()
