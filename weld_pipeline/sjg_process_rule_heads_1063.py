"""Process-rule supervision from original TRAIN export; inference uses predictions.

Censored / absent states are separate from exact millimetres. No STEP, GT side,
thickness or robot work-range configuration is fabricated.
"""
import csv,json,math
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,digest
FIELDS=('front_wall','back_wall','start_height','end_height','start_hole_height','end_hole_height','start_base_edge','end_base_edge')
EXCEL=('FrontWallDistance','BackWallDistance','startHeight','endHeight','StartViaHoleHeight','EndViaHoleHeight','StartBasePlateEdgeDistance','EndBasePlateEdgeDistance')
FLAGS=('HasEave (Any)','IsSplitedForDualRobot (Any)')

def encode_value(field,value):
    if value is None or value=='':return -1,0.,False
    try:v=float(value)
    except (TypeError,ValueError):return -1,0.,False
    if not math.isfinite(v):return -1,0.,False
    if field.endswith('wall') and v==-1:return 1,0.,False
    if field.endswith('base_edge') and v==51:return 1,0.,False
    if field.endswith('hole_height') and v==0:return 1,0.,False
    if v<0:return -1,0.,False
    if (field.endswith('wall') or field.endswith('base_edge')) and v>50+1e-6:return -1,0.,False
    if field in ('start_height','end_height') and v==0:return -1,0.,False
    return 0,v/50.,True

def compile_single_train(source='21'):
    report=ROOT/'weld_pipeline/reports/process_rule_mining_20261008'
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==source)
    assert case['role']=='TRAIN';assert digest(OUT/case['target_file'])==case['target_sha256']
    target=read(OUT/case['target_file']);assert target['original_children']==1 and target['original_parents']==1 and not target['incomplete_source_no_unmatched_query_negative_labels']
    oid=target['parents'][0]['original_parent_id']
    with (report/'all_excel_raw.csv').open(encoding='utf-8-sig') as f:rows=[a for a in csv.DictReader(f) if a['ObjectId']==oid and a['pilot_train_sources']==source]
    assert len(rows)==1;row=rows[0];binding=next(a for a in read(report/'file_inventory.json')['xlsx'] if a['file']==row['source_file']);assert digest(ROOT/binding['file'])==binding['sha256']
    conflicts=read(report/'xlsx_cbf_all_matrix_mismatches.json');blocked={a['field'] for a in conflicts if a['ObjectId']==oid and a['file']==row['source_file']}
    flags=np.array([float(row[k]=='True') for k in FLAGS],np.float32);fm=np.array([k not in blocked and row[k] in ('True','False') for k in FLAGS])
    vals=[encode_value(k,row[e]) for k,e in zip(FIELDS,EXCEL)]
    result=dict(flags=flags,flag_mask=fm,state=np.array([v[0] for v in vals],np.int64),numeric=np.array([v[1] for v in vals],np.float32),numeric_mask=np.array([v[2] for v in vals],bool))
    receipt=dict(source=source,parent_id=oid,original_excel=binding,raw_values={k:row[k] for k in (*FLAGS,*EXCEL)},conflicts_masked=sorted(blocked),fields=list(FIELDS),states={'0':'exact_mm','1':'field_specific_censored_or_absent','-1':'unknown_not_supervised'},millimetre_scale=50,front_wall_minus1_is_combined_state_not_exact_distance=True,passage_35mm_rule_not_activated_without_measured_clearance=True,STEP_reads=0,inference_target_reads=0)
    return result,receipt

def process_loss(out,supervision,seq,target):
    a,b=seq['start'],seq['end'];active=target['occupation'][a:b].any(-1)
    if not active.any():return out['process_code'].sum()*0,{}
    flags=out['process_flags'][a:b][active].mean(0)
    # Start/end endpoint attributes use separate heads pooled over predicted carrier context.
    states=out['process_state'][a:b][active].mean(0);numeric=out['process_numeric'][a:b][active].mean(0)
    fm=supervision['flag_mask'];known=supervision['state']>=0;nm=supervision['numeric_mask']
    flag=F.binary_cross_entropy_with_logits(flags[fm],supervision['flags'][fm]) if fm.any() else flags.sum()*0
    state=F.cross_entropy(states[known],supervision['state'][known]) if known.any() else states.sum()*0
    reg=F.smooth_l1_loss(numeric[nm],supervision['numeric'][nm],beta=1/50) if nm.any() else numeric.sum()*0
    return flag+state+reg,dict(process_flags=float(flag.detach()),process_states=float(state.detach()),process_numeric_scaled50=float(reg.detach()),process_valid_flags=int(fm.sum()),process_valid_states=int(known.sum()),process_exact_numeric_fields=int(nm.sum()))

def controls():
    assert encode_value('front_wall',-1)==(1,0.,False)
    assert encode_value('start_base_edge',51)==(1,0.,False)
    assert encode_value('start_base_edge',50)==(0,1.,True)
    assert encode_value('start_hole_height',0)==(1,0.,False)
    for field,value in [('start_base_edge',-1),('front_wall',83),('start_height',''),('start_hole_height',-1)]:assert encode_value(field,value)[0]==-1
    return dict(censored_states_not_exact_regression=True,hole_zero_absence=True,unknown_and_conflicts_masked=True,no_35mm_surrogate=True)
