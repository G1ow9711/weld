"""Train a single-piece fit diagnostic, not a held-out model accuracy claim."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from training.dataio import read_xyz_pcd
from training.pilot import seed_all
from training.scripts.train_pilot import atomic_save
from training.labels import sha256
from weld_pipeline.contact_sequences import make_sequences,training_targets,FEATURE_VERSION
from weld_pipeline.contact_model import ContactNet,decode
from weld_pipeline.truth import load_truth
from weld_pipeline.evaluation import score_welds


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--init');parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--outdir',default='weld_pipeline/runs/contact_single_piece_v1');args=parser.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    if (out/'epochs.jsonl').exists():raise ValueError('Use a new run directory; do not silently mix attempts')
    snapshots=out/'source_snapshot';snapshots.mkdir(exist_ok=True)
    sources=[Path(__file__),*[Path('weld_pipeline')/n for n in ['contact_model.py','contact_sequences.py','geometry_candidates.py','evaluation.py','truth.py']]]
    source_hashes={str(p):sha256(p) for p in sources}
    for p in sources:shutil.copy2(p,snapshots/p.name)
    seed_all(71);folder=Path('DataSet/9月20日/161-LB18A');pcd=next(folder.glob('*AllFaces.pcd'))
    start=time.perf_counter();xyz=read_xyz_pcd(pcd)*1000
    sequences,meta=make_sequences(xyz)
    # Features are fully computed before loading any reference or Excel flag.
    truth,_=load_truth(folder);targets,mapping=training_targets(sequences,truth)
    oracle=score_welds(decode(sequences,targets),truth)
    diagnostic={'scope':'single training assembly representation and fitting diagnostic; no independent validation',
        'feature_version':FEATURE_VERSION,'feature_channels':sequences[0]['features'].shape[1],
        'sequences':len(sequences),'feature_points':sum(len(s['xyz']) for s in sequences),
        'pcd_sha256':sha256(pcd),'source_hashes':source_hashes,'feature_and_label_seconds':time.perf_counter()-start,'mapping':mapping,
        'representation_oracle':{k:v for k,v in oracle.items() if k not in ['per_truth','unmatched_prediction_indices']},'geometry':meta}
    (out/'data_diagnostic.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    print('DATA',json.dumps(diagnostic),flush=True)
    model=ContactNet(sequences[0]['features'].shape[1]).cuda()
    if args.init:
        with Path(args.init).open('rb') as f:initial=torch.load(f,map_location='cpu')
        model.load_state_dict(initial['model'])
        diagnostic['initial_weights']=str(args.init);diagnostic['initial_weights_sha256']=sha256(args.init)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    diagnostic['arguments']=vars(args)
    (out/'data_diagnostic.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    data=[(torch.from_numpy(s['features'].T).cuda().unsqueeze(0),torch.from_numpy(t.T).cuda().unsqueeze(0)) for s,t in zip(sequences,targets)]
    rng=np.random.default_rng(71);best=None;start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        lr=args.lr*(.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/args.epochs)))
        for group in optimizer.param_groups:group['lr']=lr
        model.train();losses=[]
        for i in rng.permutation(len(data)):
            x,y=data[int(i)];optimizer.zero_grad(set_to_none=True);logits=model(x)
            # Balanced soft heatmaps supervise global endpoints, internal joins,
            # and the explicit work-range endpoint flags independently.
            weight=1+25*y[:,1:]
            binary=F.binary_cross_entropy_with_logits(logits[:,:1],y[:,:1])
            heat=F.binary_cross_entropy_with_logits(logits[:,1:],y[:,1:],reduction='none')
            loss=binary+(heat*weight).mean()*3
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        if epoch==1 or epoch%10==0 or epoch==args.epochs:
            model.eval()
            with torch.no_grad():prob=[model(x).sigmoid()[0].T.cpu().numpy() for x,y in data]
            prediction=decode(sequences,prob);score=score_welds(prediction,truth)
            rec={'epoch':epoch,'loss':float(np.mean(losses)),'elapsed_seconds':time.perf_counter()-start,
                'valid_welds':score['valid_welds'],'truth_count':len(truth),'predictions':len(prediction),
                'valid_extraction_rate':score['valid_extraction_rate'],'false_rate':score['false_or_duplicate_rate_per_truth']}
            with (out/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
            print('FIT_DIAGNOSTIC',json.dumps(rec),flush=True)
            quality=(bool(score['all_gates_pass']),-(len(truth)-score['valid_welds']+score['false_or_duplicate_predictions']),score['valid_welds'])
            checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'features':sequences[0]['features'].shape[1],
                    'feature_version':FEATURE_VERSION,'scope':'one-piece fit diagnostic','seed':71,'initial_weights':args.init,'source_hashes':source_hashes}
            atomic_save(checkpoint,out/'last.pt')
            if best is None or quality>best:
                best=quality;atomic_save(checkpoint,out/'best.pt')
                (out/'predictions.json').write_text(json.dumps(prediction,indent=2),encoding='utf-8')
                (out/'score.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
    (out/'complete.json').write_text(json.dumps({'completed_epochs':args.epochs,'selection_quality':best,'seconds':time.perf_counter()-start,
        'not_independent_acceptance':True},indent=2),encoding='utf-8')


if __name__=='__main__':main()
