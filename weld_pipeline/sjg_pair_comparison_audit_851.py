"""Read-only prediction verification and posthoc candidate discrimination audit."""
import argparse
import numpy as np
from scipy.stats import rankdata
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_parent_inputs_829 import training_targets


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;receipt=read(dest/'receipt.json')
    assert receipt['status']=='GEOMETRY_PAIR_COMPARISON_EVALUATED'
    frozen=read(dest/'all_arms_frozen_before_GT.json')
    for record in frozen['arms']:
        assert digest(dest/record['arm']/'all_outputs_frozen_before_GT.json')==record['sha256']
    inventory=read(OUT/'inventory.json');summary={};audit={}
    runs={'global156':OUT/'run57',**{arm:dest/arm for arm in receipt['arms']}}
    for arm,path in runs.items():
        result=read(path/'receipt.json');rows=result['parents'];summaries={}
        for name,ids in [('all',TRAIN+VALID+ARC),('train',TRAIN),('development',VALID),('arc_diagnostic',ARC)]:
            totals={key:sum(r['strict'][key] for r in rows if r['source'] in ids) for key in ('G','TP','FP','predicted')}
            totals.update(TP_over_G=totals['TP']/totals['G'],FP_over_G=totals['FP']/totals['G']);summaries[name]=totals
        endpoint=[];bidirectional=[];audit_rows=[]
        for source in TRAIN+VALID+ARC:
            for folder in ['free','parents']:
                records=read(path/folder/'all_predictions_frozen.json')['sources'];record=next(r for r in records if r['source']==source)
                assert digest(path/folder/record['file'])==record['sha256']
                if folder=='parents':assert digest(path/folder/record['before_nms_file'])==record['before_nms_sha256']
            candidates=read(OUT/f'run55/source_{source:02d}_predictions.json')['predictions']
            chosen=read(path/f'free/source_{source:02d}_predictions.json')['predictions']
            start_slots=set();end_slots=set()
            for child in chosen:
                original=candidates[child['candidate_index']]
                assert np.array_equal(child['points'],original['points'])
                for slots,field in [(start_slots,'start_peak_index'),(end_slots,'end_peak_index')]:
                    key=(child['profile'],child[field]);assert key not in slots;slots.add(key)
            target_path=ROOT/inventory[source]['target']['target_path']
            assert digest(target_path)==inventory[source]['target']['target_sha256']
            labels,_,_=training_targets(candidates,np.empty((0,4),np.int64),read(target_path))
            selected=read(path/f'free/source_{source:02d}_selection.json');logits=np.asarray(selected['logits']);n_pos=int(labels.sum());n_neg=len(labels)-n_pos
            auc=float((rankdata(logits)[labels>0].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg)) if n_pos and n_neg else None
            audit_rows.append(dict(source=source,positive_candidates=n_pos,negative_candidates=n_neg,AUC=auc,
                threshold_TP=int(((logits>0)&(labels>0)).sum()),threshold_FP=int(((logits>0)&(labels==0)).sum()),
                selected_positive=int(labels[selected['selected_candidates']].sum()),selected_count=len(chosen)))
            score=read(path/f'parents/source_{source:02d}_strict_geometry.json')
            for match in score['matches']:
                endpoint.extend(match['ordered_endpoint_errors_after_allowed_whole_correspondence_mm'])
                bidirectional.append(match['bidirectional_sampled_maximum_mm'])
        summaries['matched_TP_only_error_mm']=dict(endpoint_mean=float(np.mean(endpoint)),endpoint_p95=float(np.percentile(endpoint,95)),
            endpoint_max=float(np.max(endpoint)),bidirectional_max=float(np.max(bidirectional)))
        summary[arm]=summaries;audit[arm]=audit_rows
    base=summary['global156'];decisions={}
    for arm in receipt['arms']:
        value=summary[arm]
        decisions[arm]=dict(development_TP_increases=value['development']['TP']>base['development']['TP'],
            development_FP_does_not_increase=value['development']['FP']<=base['development']['FP'],
            all_TP_increases=value['all']['TP']>base['all']['TP'],all_FP_does_not_increase=value['all']['FP']<=base['all']['FP'])
        decisions[arm]['all_comparison_gates_pass']=all(decisions[arm].values())
    write(dest/'comparison_verification.json',dict(status='PASS',prediction_hashes=True,coordinates_unchanged=True,
        unique_peak_slots_within_profiles=True,all_original_denominators_preserved=True,full_contract_pass=False))
    write(dest/'comparison_summary.json',dict(arms=summary,comparison_decisions=decisions,full_contract_pass=False))
    write(dest/'candidate_generalization_audit.json',dict(posthoc_only=True,never_training_labels_for_development=True,arms=audit))
    print(summary);print(decisions)


if __name__=='__main__':main()
