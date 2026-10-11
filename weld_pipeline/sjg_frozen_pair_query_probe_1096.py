"""Read-only comparison after failed run278; never an extraction score."""
import gc
import time
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,reclaim_own_idle_pages,check_resources
from weld_pipeline.sjg_parent_set_network_1075 import ParentSetModel
from weld_pipeline.sjg_roboweldar_pair_support_1093 import PairParentInput
from weld_pipeline.sjg_parent_stream_encoder_1087 import stream_encode
from weld_pipeline.sjg_parent_query_diagnosis_1092 import cosine_summary
from weld_pipeline.sjg_mls_network_control_1079 import parameter_hash


def main():
    dest=OUT/'run279';dest.mkdir(exist_ok=False);tick=time.perf_counter()
    torch.set_num_threads(2);check_resources()
    r=read(OUT/'run277/receipt.json');prediction=read(OUT/'run278/receipt.json')
    sha=digest(OUT/'run277/final.pt');assert sha==r['checkpoint_sha256']==prediction['checkpoint_sha256']
    with (OUT/'run277/final.pt').open('rb') as f:cp=torch.load(f,map_location='cpu')
    model=ParentSetModel(bin_channels=247);model.load_state_dict(cp['state_dict']);model.eval()
    before=parameter_hash(model);assert before==r['final_parameter_sha256']
    del cp;gc.collect();reclaim_own_idle_pages();check_resources()
    with np.load(OUT/'run277/normalization.npz') as z:norm={k:z[k] for k in z.files}
    data=PairParentInput('21',norm);assert data.binding()==prediction['input_binding']
    with torch.no_grad():result=model.forward_encoded(stream_encode(model,data))
    probability=result['existence'].sigmoid()
    assert int((probability>=.5).sum())==prediction['selected_parent_queries']==0
    assert abs(float(probability.max())-prediction['probability_max'])<2e-5
    old=read(OUT/'run272/choices.json');ids=torch.tensor([c['query'] for c in old])
    logits=result['profile_query'][ids,0]@result['profile_key'].T/model.width**.5
    chosen=logits.argmax(-1)
    rows=[dict(query=c['query'],old_probability=c['probability'],new_probability=float(probability[q]),
               old_child_count=c['child_count'],new_count_head=int(result['count'][q].argmax())+1,
               old_first_profile=c['children'][0]['profile'],new_first_profile=int(chosen[j]),
               fraction_change=(result['fraction'][q,0]-torch.tensor(c['children'][0]['fractions'])).tolist())
          for j,(q,c) in enumerate(zip(ids.tolist(),old))]
    # A fixed-weight feature-path intervention, never a new tuned threshold.
    class Masked:
        def __len__(self):return len(data)
        def arrays(self,indices):
            context,bins=data.arrays(indices);bins[:,:,242:]=0
            return context,bins
    with torch.no_grad():masked=model.forward_encoded(stream_encode(model,Masked()))
    report=dict(status='FROZEN_PAIR_QUERY_PROBE_COMPLETE',optimizer_updates=0,GT_reads=0,STEP_reads=0,
        extraction_test=False,official_test_run='run278',checkpoint_sha256=sha,
        parent_latent_cosine_all=cosine_summary(result['parent_latent']),
        old_seven_queries=rows,
        pair_zero_ablation_fixed_weights=dict(
            existence_logit_max_change=float((result['existence']-masked['existence']).abs().max()),
            latent_max_change=float((result['parent_latent']-masked['parent_latent']).abs().max()),
            actual_positive_queries=int((probability>=.5).sum()),
            masked_positive_queries=int((masked['existence'].sigmoid()>=.5).sum()),
            causal_training_effect_not_identified=True),seconds=time.perf_counter()-tick,utc=utc())
    assert parameter_hash(model)==before and digest(OUT/'run277/final.pt')==sha
    write(dest/'receipt.json',report)
    print('FROZEN_QUERY_PROBE_COMPLETE',report['parent_latent_cosine_all'],flush=True)
    print('OLD_SEVEN_PROFILES',[r['new_first_profile'] for r in rows],flush=True)


if __name__=='__main__':main()
