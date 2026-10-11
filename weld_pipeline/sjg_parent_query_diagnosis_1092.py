"""CPU-only inspection of frozen query representations; no optimization/export."""
import gc
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_parent_set_network_1075 import ParentSetModel
from weld_pipeline.sjg_parent_input_1078 import ParentInput
from weld_pipeline.sjg_parent_stream_encoder_1087 import stream_encode
from weld_pipeline.sjg_mls_network_control_1079 import parameter_hash


def cosine_summary(values):
    normalized=torch.nn.functional.normalize(values.double(),dim=-1)
    cosine=normalized@normalized.T
    selected=cosine[torch.triu(torch.ones_like(cosine,dtype=torch.bool),diagonal=1)]
    return dict(minimum=float(selected.min()),mean=float(selected.mean()),maximum=float(selected.max()))


def main():
    dest=OUT/'run273';dest.mkdir(exist_ok=False);start=time.perf_counter()
    try:
        torch.set_num_threads(2);reclaim_own_idle_pages();check_resources()
        receipt=read(OUT/'run271/receipt.json');prediction=read(OUT/'run272/receipt.json')
        before_file=digest(OUT/'run271/final.pt')
        assert before_file==receipt['checkpoint_sha256']==prediction['checkpoint_sha256']
        assert digest(OUT/'run272/choices.json')==prediction['choices_sha256']
        choices=read(OUT/'run272/choices.json')
        with (OUT/'run271/final.pt').open('rb') as f:checkpoint=torch.load(f,map_location='cpu')
        model=ParentSetModel();model.load_state_dict(checkpoint['state_dict']);model.eval()
        before=parameter_hash(model);del checkpoint;gc.collect()
        assert before==receipt['final_parameter_sha256']
        with np.load(OUT/'run271/normalization.npz') as z:normalization={k:z[k] for k in z.files}
        data=ParentInput('21',normalization)
        assert data.binding()==prediction['input_binding']
        with torch.no_grad():result=model.forward_encoded(stream_encode(model,data))
        probabilities=result['existence'].sigmoid()
        selected=torch.nonzero(probabilities>=.5).flatten()
        expected=[c['query'] for c in choices]
        assert selected.tolist()==expected
        probability_error=max(abs(float(probabilities[c['query']])-c['probability']) for c in choices)
        assert probability_error<2e-5
        profile_logits=result['profile_query'][selected,0]@result['profile_key'].T/model.width**.5
        profile_probability=profile_logits.softmax(-1)
        profile_ids=profile_logits.argmax(-1)
        assert profile_ids.tolist()==[c['children'][0]['profile'] for c in choices]
        value=dict(status='FROZEN_QUERY_DIAGNOSIS_COMPLETE',source='21',all_profiles=len(data),parent_queries=256,
            unchanged_checkpoint_sha256=before_file,unchanged_parameter_sha256=before,
            optimizer_updates=0,GT_reads=0,STEP_reads=0,external_transmission=False,
            CPU_GPU_selected_query_ids_identical=True,CPU_GPU_active_probability_max_difference=probability_error,
            existence_probability_quantiles={str(q):float(torch.quantile(probabilities,q)) for q in (0.,.25,.5,.75,1.)},
            sum_existence_probabilities=float(probabilities.sum()),threshold_positive_queries=len(selected),
            sum_is_soft_mass_not_a_validated_predicted_count=True,
            selected_profile_ids=profile_ids.tolist(),selected_profile_probability=profile_probability.max(-1).values.tolist(),
            learned_query_cosine_all=cosine_summary(model.parent_query.detach()),
            encoded_parent_cosine_all=cosine_summary(result['parent_latent']),
            encoded_parent_cosine_selected=cosine_summary(result['parent_latent'][selected]),
            profile_query_cosine_selected=cosine_summary(result['profile_query'][selected,0]),
            cause_not_proven_without_intervention=True,official_run272_score_unchanged=True,
            seconds=time.perf_counter()-start,utc=utc())
        assert parameter_hash(model)==before and digest(OUT/'run271/final.pt')==before_file
        write(dest/'receipt.json',value)
        print('FROZEN_QUERY_DIAGNOSIS',value['threshold_positive_queries'],'SOFT_MASS',value['sum_existence_probabilities'],
            'ACTIVE_PARENT_COSINE',value['encoded_parent_cosine_selected'],flush=True)
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),optimizer_updates=0,utc=utc()));raise


if __name__=='__main__':main()
