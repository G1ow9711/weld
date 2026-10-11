"""Verify matched continuation protocol and retain both full-output scores."""
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    modes=['raw','grouped'];protocols=[];models=[];rows=[]
    for mode in modes:
        model=BASE/('curved_group_comparison_'+mode+'_v1');r=read(model/'receipt.json');assert digest(model/'model.pt')==r['model_sha256'];p=read(model/'protocol.json');checkpoint=torch.load(model/'model.pt',map_location='cpu',weights_only=False);assert checkpoint['protocol_sha256']==digest(model/'protocol.json');protocols.append(p);models.append(checkpoint);output=BASE/('curved_group_comparison_'+mode+'_all_candidates_v1');score=read(output/'geometry_diagnostic.json');assert score['conditional_parity_count']==6;rows.append(dict(mode=mode,model_sha256=r['model_sha256'],protocol_sha256=digest(model/'protocol.json'),full_output_receipt_sha256=digest(output/'receipt.json'),score_sha256=digest(output/'geometry_diagnostic.json'),score=score['score']['categories']['overall'],conditional_boundary_carriers_correct=sum(row['all_boundaries_within_label_sets'] for row in read(model/'conditional_predictions.json')['rows'])))
    keys=['steps','seed','lr','lr_end','batch_size','sampling','warm_model_sha256','normalization_frozen','input_dim','module_sha256','base_module_sha256','trainer_sha256']
    for key in keys:assert protocols[0][key]==protocols[1][key],key
    for key in ['mean','std']:assert np.array_equal(models[0][key],models[1][key])
    out=BASE/'curved_group_comparison_audit_v1';out.mkdir(exist_ok=False);write(out/'report.json',dict(matched_protocol_keys=keys,normalization_exactly_equal=True,rows=rows,scope='Same one TRAIN family, shared upstream. Full1920 output fixed-H geometry diagnostic only, not generalization or contract acceptance. No automatic promotion.',DEV_reads=0));print(rows,flush=True)

if __name__=='__main__':main()
