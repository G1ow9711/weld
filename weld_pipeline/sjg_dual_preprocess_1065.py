"""Independent input-root preprocessing; original process labels enter bags only."""
import argparse,csv,gc,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,utc
from weld_pipeline.sjg_process_rule_heads_1063 import FIELDS,EXCEL,FLAGS,encode_value

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run243');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);manifest=read(OUT/'run206/training_manifest.json');bindings=[];moments=[]
    start=time.perf_counter()
    # Select coordinates solely from the input domain, never from states/owners.
    for c in manifest['cases']:
        check_resources();key=c['source_key'];p=OUT/'run206'/c['file'];assert digest(p)==c['sha256']
        domainpath=OUT/f'run201/source_{key}_joint_domain.json';domain=read(domainpath)
        static=next(b for b in manifest['static_bindings'] if b['source_key']==key)
        assert digest(OUT/'run206'/static['file'])==static['sha256']
        with np.load(p) as z:
            f=z['node_family'];lo=z['node_lo'];hi=z['node_hi'];interval=z['interval_x']
        lookup={(int(fid),float(a),float(b)):i for i,(fid,a,b) in enumerate(zip(f,lo,hi))}
        assert len(lookup)==len(f)
        pooled=np.zeros((len(domain['families']),interval.shape[1]),np.float32);root_count=0
        for fid,family in enumerate(domain['families']):
            rows=[lookup[(fid,float(a),float(b))] for a,b in family['domains_mm']]
            weights=np.array([b-a for a,b in family['domains_mm']],np.float64)
            assert rows and np.all(weights>0)
            pooled[fid]=np.average(interval[rows],axis=0,weights=weights);root_count+=len(rows)
        with np.load(OUT/'run206'/static['file']) as z:
            x=np.concatenate((z['relation_x'][z['family_relation']],z['family_x'],pooled),1).astype(np.float32)
        assert x.shape==(len(domain['families']),669) and np.isfinite(x).all()
        path=dest/f'source_{key}_features.npy';np.save(path,x)
        moments.append((x.mean(0,dtype=np.float64),np.square(x,dtype=np.float64).mean(0)))
        bindings.append(dict(source=key,file=path.name,sha256=digest(path),families=len(x),roots=root_count,
            domain_sha256=digest(domainpath),static_sha256=static['sha256'],archive_sha256=c['sha256'],
            archive_contains_supervision=True,archive_fields_read=['node_family','node_lo','node_hi','interval_x'],
            selected_only_exact_input_domain_roots=True,target_reads_for_root_selection=0))
        print('PURE_FEATURES_FROZEN',key,len(x),'ROOTS',root_count,flush=True)
        del x,pooled,interval,domain,lookup;gc.collect()
    mean=np.mean([a for a,b in moments],0);scale=np.sqrt(np.maximum(np.mean([b for a,b in moments],0)-mean**2,0));scale[scale<1e-6]=1
    np.savez(dest/'process_normalization.npz',mean=mean.astype(np.float32),scale=scale.astype(np.float32))
    write(dest/'input_freeze.json',dict(cases=bindings,normalization_sha256=digest(dest/'process_normalization.npz'),
        normalization='macro by seven TRAIN pure-input populations',target_reads=0,STEP_reads=0,utc=utc()))
    # Labels are read only after the independent input population is frozen.
    from weld_pipeline.sjg_hierarchical_data_1030 import family_supervision
    report=ROOT/'weld_pipeline/reports/process_rule_mining_20261008'
    with (report/'all_excel_raw.csv').open(encoding='utf-8-sig') as stream:excel=list(csv.DictReader(stream))
    inventory=read(report/'file_inventory.json')['xlsx'];conflicts=read(report/'xlsx_cbf_all_matrix_mismatches.json')
    records=[];coverage=[];blocked_total=0
    cases={c['source_key']:c for c in read(OUT/'run98/local_cohort.json')['cases']}
    for binding in bindings:
        key=binding['source'];case=cases[key];assert digest(OUT/case['target_file'])==case['target_sha256']
        target=read(OUT/case['target_file']);domain=read(OUT/f'run201/source_{key}_joint_domain.json')
        with np.load(OUT/f'run201/source_{key}_joint_mapping.npz') as z:supervised,_=family_supervision(key,domain,z['profile_family'],z['local_to_canonical'],target)
        by_child={c['original_child_id']:[] for p in target['parents'] for c in p['children']}
        for fid,children in supervised.items():
            for child,v in children.items():by_child[child].append([fid,bool(v['forward'])])
        assert all(by_child.values()),'Original child has no witness; do not silently reduce denominator'
        source_rows=[r for r in excel if r['pilot_train_sources']==key]
        for p in target['parents']:
            oid=p['original_parent_id'];rows=[r for r in source_rows if r['ObjectId']==oid];assert len(rows)==1,(key,oid,len(rows))
            row=rows[0];original=next(i for i in inventory if i['file']==row['source_file']);assert digest(ROOT/original['file'])==original['sha256']
            blocked={a['field'] for a in conflicts if a['ObjectId']==oid and a['file']==row['source_file']};blocked_total+=len(blocked)
            encoded=[encode_value(k,row[e]) for k,e in zip(FIELDS,EXCEL)]
            children=[by_child[c['original_child_id']] for c in p['children']]
            records.append(dict(source=key,parent=oid,children=children,flags=[int(row[k]=='True') for k in FLAGS],
                flag_mask=[k not in blocked and row[k] in ('True','False') for k in FLAGS],
                state=[v[0] for v in encoded],numeric=[v[1] for v in encoded],numeric_mask=[v[2] for v in encoded],
                blocked=sorted(blocked),original_excel=original))
        coverage.append(dict(source=key,parents=len(target['parents']),children=len(by_child),all_children_represented=True))
        del domain,supervised;gc.collect()
    assert len(records)==222 and sum(c['children'] for c in coverage)==363 and blocked_total==62
    write(dest/'process_bags.json',records)
    write(dest/'receipt.json',dict(status='DUAL_PREPROCESS_READY',coverage=coverage,parents=222,children=363,
        masked_conflicts=blocked_total,input_freeze_sha256=digest(dest/'input_freeze.json'),bags_sha256=digest(dest/'process_bags.json'),
        process_roles='seven TRAIN only; no DEV or ARC24 labels',parent_any_not_broadcast=True,optimizer_updates=0,seconds=time.perf_counter()-start,utc=utc()))
    print('DUAL_PREPROCESS_READY',len(records),'parents',flush=True)

if __name__=='__main__':main()
