"""Posthoc TRAIN-only diagnosis of frozen material atoms, never model input."""
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources


def atom_union_bound(component,owner):
    """Optimistic best union separately for each owner, without excluding reuse.

    Boundary/unassigned points do not disappear from each owner's denominator.
    This is a reference-assisted diagnostic, not a runnable object prediction.
    """
    known=owner>=0;owners=np.unique(owner[known]);present=known&(component>=0)
    atoms,inverse=np.unique(component[present],return_inverse=True);truth=np.searchsorted(owners,owner[present])
    table=np.bincount(inverse*len(owners)+truth,minlength=len(atoms)*len(owners)).reshape(len(atoms),len(owners))
    total=np.bincount(np.searchsorted(owners,owner[known]),minlength=len(owners));sizes=table.sum(1);rows=[]
    for j,ident in enumerate(owners):
        # The optimum includes the highest purity atoms first. Additional atoms
        # help iff their target-to-nontarget ratio exceeds the current objective.
        purity=table[:,j]/np.maximum(sizes,1);order=np.argsort(-purity,kind='stable')
        hit=np.cumsum(table[order,j]);extra=np.cumsum(sizes[order]-table[order,j])
        iou=hit/np.maximum(1,total[j]+extra);best=int(np.argmax(iou)) if len(iou) else None
        rows.append(dict(reference_owner=int(ident),reference_points=int(total[j]),
            atom_known_points=int(table[:,j].sum()),boundary_or_unassigned_known_points=int(total[j]-table[:,j].sum()),
            pure_atom_points=int(table[sizes==table[:,j],j].sum()),
            optimistic_union_IoU=float(iou[best]) if best is not None else 0.,
            atoms_in_best_union=best+1 if best is not None else 0))
    return dict(rows=rows,atoms_with_known_support=len(atoms),reference_owners=len(owners),
        optimistic_matched_at_half=sum(r['optimistic_union_IoU']>=.5 for r in rows),
        optimistic_at_point95=sum(r['optimistic_union_IoU']>=.95 for r in rows),
        mixed_atoms=int(((table>0).sum(1)>1).sum()),
        mixed_atom_known_points=int(sizes[(table>0).sum(1)>1].sum()),
        reference_assisted_nonexclusive_upper_bound_not_prediction=True)


def support_matrix(source,n):
    folder=OUT/f'run34/source_{source:02d}';binding=read(folder/'input_freeze.json')
    assert digest(folder/'surfaces.json')==binding['files']['surfaces.json']
    faces=read(folder/'surfaces.json');archives={};rr=[];cc=[]
    try:
        for i,face in enumerate(faces):
            path=ROOT/face['support_file'];key=str(path)
            if key not in archives:
                assert digest(path)==face['support_sha256'];archives[key]=np.load(path)
            ids=archives[key][face['point_ids_key']].astype(np.int64)
            assert np.all((ids>=0)&(ids<n));rr.append(ids);cc.append(np.full(len(ids),i,np.int64))
    finally:
        for z in archives.values():z.close()
    rows=np.concatenate(rr);cols=np.concatenate(cc)
    mat=csr_matrix((np.ones(len(rows),np.int32),(rows,cols)),shape=(n,len(faces)))
    mat.sum_duplicates();mat.data[:]=1
    return mat


def pair_finite_support(source,neighbors,distance,boundary,owner,instance):
    n=len(owner);support=support_matrix(source,n);counts={}
    for k in range(neighbors.shape[1]):
        other=neighbors[:,k].clip(0);shared=np.asarray(support.multiply(support[other]).sum(axis=1)).ravel()>0
        valid=(neighbors[:,k]>=0)&np.isfinite(distance[:,k])&(owner>=0)&(owner[other]>=0)
        negative=distance[:,k]>.5;same=owner==owner[other];interior=(boundary<.5)&(boundary[other]<.5)
        surface_known=(instance>=0)&(instance[other]>=0)
        groups={'all':valid,'shared_predicted_finite_support':valid&shared,'no_shared_predicted_support':valid&~shared,
            'both_predicted_interior':valid&interior,'any_predicted_boundary':valid&~interior,
            'same_reference_face_posthoc':valid&surface_known&(instance==instance[other]),
            'different_reference_faces_same_owner_posthoc':valid&same&surface_known&(instance!=instance[other]),
            'same_owner_face_unresolved_posthoc':valid&same&~surface_known}
        for name,mask in groups.items():
            r=counts.setdefault(name,dict(known_reference_pairs=0,true_different=0,false_different=0,predicted_same_true_different=0))
            r['known_reference_pairs']+=int(mask.sum());r['true_different']+=int((mask&negative&~same).sum())
            r['false_different']+=int((mask&negative&same).sum());r['predicted_same_true_different']+=int((mask&~negative&~same).sum())
    for r in counts.values():r['predicted_different_precision']=r['true_different']/max(1,r['true_different']+r['false_different'])
    return counts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json',dict(stage='TRAIN_FROZEN_ATOM_AND_FINITE_SUPPORT_DIAGNOSIS',sources=TRAIN,
        predictions='run75/dense_control',training=False,posthoc_reference_only=True,reference_not_model_input=True,
        optimistic_union_allows_atom_reuse=True,no_development_targets_read=True,source_sha256=digest(__file__),start_utc=utc()))
    teachers={r['source']:r for r in read(OUT/'run22/label_summary.json')};folder=OUT/'run75/dense_control'
    relations={r['source']:r for r in read(folder/'all_predictions_frozen.json')['sources']}
    structures={r['source']:r for r in read(folder/'constrained_predictions_frozen.json')['sources']};rows=[]
    for source in TRAIN:
        check_resources();print('MATERIAL_ATOM_DIAGNOSIS_START',source,flush=True)
        r=relations[source];s=structures[source];teacher=OUT/'run22'/teachers[source]['teacher_file']
        assert digest(folder/r['file'])==r['sha256'] and digest(folder/s['file'])==s['sha256']
        assert digest(teacher)==teachers[source]['teacher_sha256']
        with np.load(folder/r['file']) as p,np.load(folder/s['file']) as q,np.load(teacher) as t:
            owner=t['reference_volume'];atoms=atom_union_bound(q['interior_component'],owner)
            pair=pair_finite_support(source,p['neighbors'],p['material_distance'],p['boundary'],owner,t['instance'])
        row=dict(source=source,atom_union=atoms,pair_groups=pair);rows.append(row)
        write(dest/f'source_{source:02d}_diagnosis.json',row)
        print('MATERIAL_ATOM_DIAGNOSIS_COMPLETE',source,'UNION_BOUND',atoms['optimistic_matched_at_half'],'/',atoms['reference_owners'],
            'MIXED_ATOMS',atoms['mixed_atoms'],'ALL_PAIRS',pair['all'],'COMMON_SUPPORT',pair['shared_predicted_finite_support'],flush=True)
    write(dest/'receipt.json',dict(status='TRAIN_ATOM_FINITE_SUPPORT_DIAGNOSED',rows=rows,
        original_train_G=359,full_contract_G=496,full_contract_pass=False,optimizer_updates=0,end_utc=utc()))


if __name__=='__main__':main()
