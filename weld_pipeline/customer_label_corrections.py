"""Explicit customer corrections. Never modify source workbooks in place."""
CORRECTION_VERSION='customer_corrections_20260928_v1'
CORRECTIONS=(
    {'group':'224-FR71A','id':'14a920a7-9818-4415-b17c-f7204e3f0e74','number':'f1',
     'field':'EndBasePlateEdgeDistance','raw_value':-1,'effective_value':51},
    {'group':'224-FR71A','id':'f1c5176b-c7f1-4a76-80a8-277d2e1dfc50','number':'f2',
     'field':'EndBasePlateEdgeDistance','raw_value':-1,'effective_value':51},
)
EVIDENCE='Customer reply on 2026-09-28: question 2, correct value is 51.'


def apply_label_corrections(group,row):
    effective=dict(row);applied=list(row.get('_label_corrections',[]))
    for item in CORRECTIONS:
        if group!=item['group'] or str(row.get('ObjectId'))!=item['id']:continue
        if row.get('WeldNumber (f/v + Name number)')!=item['number']:
            raise ValueError('Correction record identity differs from source label')
        previous=next((a for a in applied if a['field']==item['field'] and a['id']==item['id']),None)
        if previous is not None:
            if row.get(item['field'])!=item['effective_value']:
                raise ValueError('Previously corrected value has changed')
            continue
        if row.get(item['field'])!=item['raw_value']:
            raise ValueError('Correction expected original -1; source changed and needs review')
        effective[item['field']]=item['effective_value']
        applied.append({**item,'version':CORRECTION_VERSION,'evidence':EVIDENCE})
    if applied:effective['_label_corrections']=applied
    return effective,applied
