"""Verify matched continuation and full outputs, then audit endpoint coverage."""
import argparse
import json
import numpy as np
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc
from weld_pipeline.sjg_event_comparison_audit_843 import audit as event_audit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);base=OUT/'run64'
    assert read(base/'receipt.json')['status']=='CONDITION_DROPOUT_COMPARISON_EVALUATED'
    write(dest/'registered_protocol.json',dict(stage='CONDITION_COMPARISON_VERIFICATION_AND_EVENT_AUDIT',
        source_sha256=digest(__file__),evaluation='run64',full_contract_pass=False,start_utc=utc()))
    for arm in ['control','condition_dropout']:
        p=OUT/'run63'/arm;r=read(p/'receipt.json');epochs=[json.loads(s) for s in (p/'epochs.jsonl').read_text().splitlines()]
        assert len(epochs)==2 and [e['epoch'] for e in epochs]==[1,2]
        assert r['actual_optimizer_updates']==1204 and r['completed_full_grid_epochs']==2
        assert all(e['grid_points']==1785106 for e in epochs)
        assert digest(p/'last.pt')==r['checkpoint_sha256']
        assert all((e['masked_points']==0) if arm=='control' else (0<e['masked_points']<e['grid_points']) for e in epochs)
    summary={};coverage={}
    for arm in ['baseline','control','condition_dropout']:
        p=OUT/'run57' if arm=='baseline' else base/arm/'output';result=read(p/'receipt.json');groups={}
        for name,ids in [('all',TRAIN+VALID+ARC),('train',TRAIN),('development',VALID),('arc_diagnostic',ARC)]:
            totals={key:sum(r['strict'][key] for r in result['parents'] if r['source'] in ids) for key in ['G','TP','FP','predicted']}
            totals.update(TP_over_G=totals['TP']/totals['G'],FP_over_G=totals['FP']/totals['G']);groups[name]=totals
        endpoints=[];distances=[]
        for source in TRAIN+VALID+ARC:
            for folder in ['free','parents']:
                binding=next(r for r in read(p/folder/'all_predictions_frozen.json')['sources'] if r['source']==source)
                assert digest(p/folder/binding['file'])==binding['sha256']
                if folder=='parents':assert digest(p/folder/binding['before_nms_file'])==binding['before_nms_sha256']
            candidate_dir=OUT/'run55' if arm=='baseline' else base/arm/'candidates'
            binding=next(r for r in read(candidate_dir/'all_inputs_frozen.json')['sources'] if r['source']==source)
            assert digest(candidate_dir/binding['file'])==binding['sha256']
            candidate_file=candidate_dir/f'source_{source:02d}_predictions.json';assert digest(candidate_file)==binding['candidate_sha256']
            candidates=read(candidate_file)['predictions'];chosen=read(p/f'free/source_{source:02d}_predictions.json')['predictions']
            starts=set();ends=set()
            for child in chosen:
                original=candidates[child['candidate_index']];assert np.array_equal(child['points'],original['points'])
                for slots,key in [(starts,'start_peak_index'),(ends,'end_peak_index')]:
                    slot=(child['profile'],child[key]);assert slot not in slots;slots.add(slot)
            for match in read(p/f'parents/source_{source:02d}_strict_geometry.json')['matches']:
                endpoints.extend(match['ordered_endpoint_errors_after_allowed_whole_correspondence_mm']);distances.append(match['bidirectional_sampled_maximum_mm'])
        groups['matched_TP_only_error_mm']=dict(endpoint_mean=float(np.mean(endpoints)),endpoint_p95=float(np.percentile(endpoints,95)),
            endpoint_max=float(np.max(endpoints)),bidirectional_max=float(np.max(distances)))
        summary[arm]=groups
    for binding in read(base/'all_arms_frozen_before_GT.json')['arms']:
        p=base/binding['arm']
        assert digest(p/'sequence/all_predictions_frozen.json')==binding['sequence_sha256']
        assert digest(p/'candidates/all_inputs_frozen.json')==binding['candidates_sha256']
        assert digest(p/'output/all_outputs_frozen_before_GT.json')==binding['outputs_sha256']
        assert read(p/'candidates/receipt.json')['status']=='ALL_INTERVAL_INPUTS_FROZEN_NO_LABEL_READS'
        coverage[binding['arm']]=event_audit(p/'sequence',dest/(binding['arm']+'_events'))
    a=summary['control'];b=summary['condition_dropout']
    decision=dict(development_TP_increases=b['development']['TP']>a['development']['TP'],
        development_FP_does_not_increase=b['development']['FP']<=a['development']['FP'],
        all_TP_increases=b['all']['TP']>a['all']['TP'],all_FP_does_not_increase=b['all']['FP']<=a['all']['FP'])
    decision['comparison_gates_pass']=all(decision.values())
    write(dest/'receipt.json',dict(status='CONDITION_RESULTS_VERIFIED_AND_AUDITED',summary=summary,event_coverage=coverage,
        decision=decision,hashes_and_coordinates_and_peak_slots_verified=True,original_G=496,full_contract_pass=False,end_utc=utc()))
    print('CONDITION_COMPARISON',summary,decision,flush=True)


if __name__=='__main__':main()
