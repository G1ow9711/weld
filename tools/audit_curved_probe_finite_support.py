"""Audit existing TRAIN witnesses against preserved section-support intervals."""
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    folder=BASE/'curved_upright_probe_v3/new_fcaf2af1abee';p=folder/'profiles.json';profiles=read(p);binding=read(folder/'receipt.json');assert digest(p)==binding['profiles_sha256'];w=folder/'d3_interval_capacity.json';capacity=read(w);assert capacity['profiles_sha256']==digest(p);rows=[]
    for row in capacity['rows']:
        candidates=[]
        for match in row['matches']:
            profile=profiles[match['profile']];lo,hi=match['interval_mm'];intervals=profile['support_intervals_mm'];clipped=sorted((max(lo,a),min(hi,b)) for a,b in intervals if min(hi,b)>max(lo,a));merged=[]
            for a,b in clipped:
                if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
                else:merged.append([a,b])
            supported=sum(b-a for a,b in merged);gaps=[];end=lo
            for a,b in merged:
                if a>end:gaps.append([end,a])
                end=max(end,b)
            if end<hi:gaps.append([end,hi])
            candidates.append(dict(profile=match['profile'],status=profile['status'],reference_projected_length_mm=hi-lo,section_supported_length_mm=supported,unobserved_length_mm=(hi-lo)-supported,largest_unobserved_gap_mm=max([b-a for a,b in gaps],default=0),entire_interval_in_one_observed_section_run=any(a<=lo and b>=hi for a,b in intervals),base_face=profile['face_indices'],bottom_sample_nearest_max_mm=profile.get('bottom_sample_nearest_max_mm'),support_intervals_are_spacing_grouped=True))
        rows.append(dict(parent=row['parent'],child=row['child'],matches=candidates,any_single_section_run=any(c['entire_interval_in_one_observed_section_run'] for c in candidates)))
    out=BASE/'curved_probe_v3_finite_support_audit_v1';out.mkdir(exist_ok=False);write(out/'report.json',dict(rows=rows,children_with_single_section_run=sum(r['any_single_section_run'] for r in rows),profiles_sha256=digest(p),capacity_sha256=digest(w),scope='TRAIN projected-witness audit only. Section runs themselves allow spacing-based adjacency; they do not establish continuity at base height, contact, or runtime endpoints. No new proposals or fits.',DEV_reads=0))
    for row in rows:print(row['parent'],'single run',row['any_single_section_run'],'best missing mm',min(c['unobserved_length_mm'] for c in row['matches']),flush=True)

if __name__=='__main__':main()
