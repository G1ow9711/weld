"""All-input frozen model-output plus finite-endcap proposal control; no labels."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import gc,subprocess,sys,time
from pathlib import Path
from tools import curveonly_01a1ac7c_representation_control_v464 as control
from tools import curveonly_01a1ac7c_paired_face_control_v409 as guards
from weld_pipeline import curveonly_01a1ac7c_endcap_connections_v467 as rules

e,common=control.e,control.common
DEST=e.REC/'endcap_connection_pilot_v468'


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(e.REC/'end_interface_audit_v466/CLOSURE.json')['analysis_closed']
    assert e.read(control.DEST/'full/CLOSURE.json')['analysis_closed']
    p=control.check();outs={c['source_id']:c for c in e.read(control.DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    jobs=[dict(c,current_prediction=outs[c['source_id']]['prediction']) for c in e.read(p['inputs']['path'])['cases']]
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve()]
    for source in sources:assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(source.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True);e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truth_or_scores=True))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(source_commit=commit,sources=[e.bind(s) for s in sources]+p['sources'],
        inputs=e.bind(DEST/'INPUTS.json'),model_outputs=e.bind(control.DEST/'full/OUTPUTS_FROZEN.json'),
        previous_geometry_audit=e.bind(e.REC/'end_interface_audit_v466/CLOSURE.json'),
        no_published_weld_outputs_or_new_TP_claim=True,no_new_fit_or_VALcalibration=True,
        no_spatial_gap_size_admission_threshold=True,epoch=time.time()))


def observe():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for binding in p['sources']+[p['inputs'],p['model_outputs']]:common.valid(binding)
    assert not (DEST/'OBSERVATIONS.json').exists();jobs=e.read(p['inputs']['path'])['cases']
    resource=common.shared.resources.resources();assert .9*resource['maximum']-resource['memory']>.8*1024**3 and resource['cpu']+1<.8*resource['quota'],resource
    guards.install_fence();rows=[]
    for job in jobs:
        decoded=e.read(common.valid(job['current_prediction'])['path']);packet=e.read(common.valid(job['geometry'])['path'])
        refs=rules.references(decoded,packet)
        if refs:
            prep=e.read(common.valid(job['prepared'])['path']);inventory=e.read(common.valid(prep['inventory'])['path'])
            solids=e.core.load_material(common.valid(job['STEP'])['path'])
            value=rules.propose(decoded,packet,inventory,solids)
            del solids,inventory;gc.collect()
        else:value=dict(references=[],junctions=[],proposals=[],rejected=[],reason='NO_QUALIFIED_OPPOSITE_COMPLETE_MODEL_UNIT')
        rows.append(dict(source_id=job['source_id'],split=job['split'],prediction=job['current_prediction'],observation=value))
        e.write(DEST/'PROGRESS.json',dict(cases=rows))
        print('ENDCAP_CONNECTION_PROPOSAL',job['source_id'],'refs',len(value['references']),'junctions',len(value['junctions']),
              'proposals',len(value['proposals']),'removed_if_future_adopted',[v['own_model_parent_indices'] for v in value['proposals']],flush=True)
    e.write(DEST/'OBSERVATIONS.json',dict(cases=rows,no_truth_or_score_reads=True,
        no_published_weld_outputs_or_new_TP_claim=True,epoch=time.time()))


if __name__=='__main__':{'prepare':prepare,'observe':observe}[sys.argv[1]]()
