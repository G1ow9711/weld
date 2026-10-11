"""Exercise the actual cached batch, masked loss, and both training branches."""
import json
from pathlib import Path

import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.joint_relation_heatmaps import JointRelationHeatmaps
from weld_pipeline.train_joint_relation_crossfit import DATA, load_data, loss_fn, sample_batch, seed_all


def main():
    config=json.loads((DATA/'config.json').read_text('utf8'))
    original=dict(config)
    config['pieces']=config['pieces'][:1]
    data=load_data(config)
    piece=data[config['pieces'][0]]
    x,node,y,known=sample_batch(piece,np.random.default_rng(143))
    seed_all(143);base=JointRelationHeatmaps(False).cuda()
    seed_all(143);graph=JointRelationHeatmaps(True).cuda()
    same_initial=all(torch.equal(value,graph.state_dict()[key]) for key,value in base.state_dict().items())
    cases=[]
    for messages,model in [(False,base),(True,graph)]:
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
        losses=[]
        for _ in range(2):
            logits=model(x,node,*piece['torch_graph'])[:,:,64:64+1024]
            loss=loss_fn(logits,y,known)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            losses.append(float(loss.detach()))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite actual training loss')
            if not losses[:-1]:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
        gradient=sum(float(p.grad.abs().sum()) for layer in model.relations for module in layer
                     for p in module.parameters() if p.grad is not None)
        if messages and not gradient>0.:
            raise ValueError('Actual message training has no gradient')
        cases.append(dict(messages=messages,losses=losses,message_gradient_l1=gradient))
    covered=[name for fold in original['folds'] for name in fold['held_out_real']]
    exclusive=all(not set(fold['train_real'])&set(fold['held_out_real']) for fold in original['folds'])
    passed=(same_initial and exclusive and sorted(covered)==sorted(original['pieces'])
            and np.isfinite(x.cpu().numpy()).all() and bool(known.sum()>0))
    receipt=dict(passed=bool(passed),identical_initial_parameters=same_initial,
                 held_source_coverage_complete=sorted(covered)==sorted(original['pieces']),
                 train_held_sources_disjoint=exclusive,batch_shape=list(x.shape),
                 known_target_elements=int(known.sum()),cases=cases,
                 trainer_sha256=sha256('weld_pipeline/train_joint_relation_crossfit.py'),
                 model_sha256=sha256('weld_pipeline/joint_relation_heatmaps.py'),
                 prepared_data_config_sha256=sha256(DATA/'config.json'),
                 full_contract_pass=False)
    Path('weld_pipeline/reports/joint_relation_training_check_20261001.json').write_text(
        json.dumps(receipt,indent=2),encoding='utf8')
    print('RELATION_TRAINING_CHECK',json.dumps(receipt),flush=True)
    if not passed:
        raise ValueError('Actual training checks failed')


if __name__=='__main__':
    main()
