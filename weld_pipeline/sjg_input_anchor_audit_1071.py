"""Read-only numerical provenance audit; no feature, label or model modification."""
import collections,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_sparse_contact_domain_1048 import family_cells
from weld_pipeline.sjg_projected_boundary_anchors_1050 import ProjectedBoundaryAnchors
from weld_pipeline.sjg_shared_joint_domain_1023 import intervals

def endpoints(family,profiles,transforms):
    tags=collections.defaultdict(set);circ=family['circumference_mm']
    for lo,hi in family['domains_mm']:
        tags[float(lo)].add('domain');tags[float(hi)].add('domain')
    for p in family['profiles']:
        start,sign,length=transforms[p]
        for lo,hi in intervals(float(start),float(sign),float(length),circ):
            tags[lo].add('carrier');tags[hi].add('carrier')
        for name,kind in [('support_intervals_mm','support'),('unobserved_gaps_mm','gap')]:
            for a,b in profiles[p].get(name,[]):
                for lo,hi in intervals(float(start+sign*a),float(sign),float(b-a),circ):
                    tags[lo].add(kind);tags[hi].add(kind)
    return tags

def main():
    dest=OUT/'run248';dest.mkdir(exist_ok=False);start=time.perf_counter();check_resources()
    data=OUT/'run224';freeze=read(data/'input_freeze.json')
    for name,sha in freeze['files'].items():assert digest(data/name)==sha
    base=OUT/'run185/source_21';binding=read(base/'input_freeze.json')
    for name in ('profiles.json','surfaces.json','surface_support.npz'):assert digest(base/name)==binding['files'][name]
    db=freeze['provenance']['domain'];assert digest(OUT/'run201'/db['json_file'])==db['json_sha256'];assert digest(OUT/'run201'/db['mapping_file'])==db['mapping_sha256']
    domain=read(OUT/'run201'/db['json_file']);profiles=read(base/'profiles.json');faces=read(base/'surfaces.json')
    with np.load(OUT/'run201'/db['mapping_file']) as z:transforms=z['local_to_canonical']
    with np.load(data/'input_domain.npz') as z:original=z['cells'];families=z['family']
    fb=read(OUT/'run166/source_21_freeze.json');assert digest(OUT/'run166'/fb['feature_file'])==fb['feature_sha256']
    assert digest(OUT/fb['input_file'])==fb['input_sha256']==freeze['source_input_sha256']
    xyz=np.load(OUT/fb['input_file'],mmap_mode='r')
    with np.load(OUT/'run166'/fb['feature_file']) as z:boundary=z['boundary']
    thresholds=[1e-10,1e-8,1e-5,.001,.01,1.];rows=[];pairs=collections.Counter();tiny=[];base_widths=[];all_widths=[]
    same_count=0;physical_gap_tiny=0;limits=np.empty(len(domain['families']))
    with np.load(base/'surface_support.npz') as support:
        provider=ProjectedBoundaryAnchors(xyz,boundary,faces,support)
        for fid,family in enumerate(domain['families']):
            extra=provider.family(family,domain['relations'][family['relation']]['ordered_face_indices'])
            full,_,_=family_cells(family,profiles,transforms,extra);basic,_,_=family_cells(family,profiles,transforms)
            saved=original[families==fid];assert np.array_equal(full,saved),(fid,full.shape,saved.shape)
            same_count+=len(full);tags=endpoints(family,profiles,transforms)
            for q in extra:tags[float(q)].add('projected_PT_evidence')
            widths=full[:,1]-full[:,0];bw=basic[:,1]-basic[:,0];all_widths.extend(widths);base_widths.extend(bw)
            # Diagnostic cancellation envelope only; NEVER used to merge cells.
            scale=max(1.,float(np.abs(np.asarray(family['descriptor'])).max()),float(np.abs(full[:,:2]).max()))
            tolerance=64*np.finfo(np.float64).eps*scale;limits[fid]=tolerance
            for i in np.flatnonzero(widths<1):
                left,right=full[i,:2];lt=tags[float(left)];rt=tags[float(right)]
                def group(t):return 'projected_only' if t=={'projected_PT_evidence'} else 'mixed' if 'projected_PT_evidence' in t else 'geometric'
                pairs[group(lt)+'__'+group(rt)]+=1
                if widths[i]<=tolerance:
                    physical_gap_tiny+=int(bool(full[i,3]))
                    tiny.append(dict(family=fid,cell_index=int(np.flatnonzero(families==fid)[i]),lo=float(left),hi=float(right),
                        width_mm=float(widths[i]),diagnostic_fp64_envelope_mm=tolerance,left_sources=sorted(lt),right_sources=sorted(rt),
                        supported=bool(full[i,2]),gap_possible=bool(full[i,3]),profiles=family['profiles']))
            rows.append(dict(family=fid,profiles=len(family['profiles']),cells=len(full),base_cells=len(basic),projected_anchor_count=len(extra),
                counts_below={str(t):int((widths<t).sum()) for t in thresholds},base_counts_below={str(t):int((bw<t).sum()) for t in thresholds},
                diagnostic_fp64_close_cells=int((widths<=tolerance).sum()),fp64_envelope_mm=tolerance))
            if (fid+1)%1000==0:print('INPUT_AUDIT',fid+1,'/',len(domain['families']),flush=True)
        provider_meta=provider.metadata()
    assert same_count==len(original);write(dest/'per_family.json',rows);write(dest/'machine_scale_cells.json',tiny)
    pf=OUT/'run246/source_21';prediction_freeze=read(pf/'prediction_freeze.json');assert digest(pf/'predictions.json')==prediction_freeze['predictions_sha256']
    predictions=read(pf/'predictions.json');lookup={int(fid):original[families==fid] for fid in np.unique(families)};correlation=collections.Counter()
    for p in predictions:
        length=p['hi']-p['lo'];fid=p['family'];cells=lookup[fid];mid=(p['hi']+p['lo'])/2
        j=np.searchsorted(cells[:,0],mid,side='right')-1
        if j<0 or mid>cells[j,1]:correlation['midpoint_outside_family_domain']+=1;continue
        width=cells[j,1]-cells[j,0]
        if length<1:
            correlation['outputs_below_1mm']+=1;correlation['below_1mm_output_midpoint_in_below_1mm_cell']+=int(width<1)
            correlation['below_1mm_output_midpoint_in_fp64_close_cell']+=int(width<=limits[fid])
        if length<.01:
            correlation['outputs_below_001mm']+=1;correlation['below_001mm_output_midpoint_in_below_001mm_cell']+=int(width<.01)
    all_widths=np.asarray(all_widths);base_widths=np.asarray(base_widths)
    summary=dict(status='READ_ONLY_ANCHOR_PROVENANCE_AUDIT_COMPLETE',families=len(rows),all_cells_reproduced_bitwise=True,cells=same_count,
        base_geometry_cells_without_PT_anchor_partition=len(base_widths),additional_cells_from_PT_anchor_partition=len(all_widths)-len(base_widths),
        full_counts_below={str(t):int((all_widths<t).sum()) for t in thresholds},base_counts_below={str(t):int((base_widths<t).sum()) for t in thresholds},
        below_1mm_endpoint_source_pairs=dict(pairs),diagnostic_fp64_close_cells=len(tiny),fp64_close_cells_marked_gap_possible=physical_gap_tiny,
        fp64_envelope_definition='64*float64_eps*max(1,abs(family descriptor),abs(domain coordinates)); diagnostic only, not physical equality proof',
        output_input_resolution_correlation=dict(correlation),provider_metadata=provider_meta,optimizer_updates=0,
        model_or_inputs_or_predictions_modified=False,target_reads=0,STEP_reads=0,full_contract_pass=False,seconds=time.perf_counter()-start,utc=utc())
    write(dest/'receipt.json',summary);print(__import__('json').dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
