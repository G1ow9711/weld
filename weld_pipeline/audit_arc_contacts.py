"""Freeze XYZ-only candidates, then measure reference coverage (not accuracy)."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.arc_surface_candidates import arc_surface_candidates


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',required=True);ap.add_argument('--outdir',required=True)
    args=ap.parse_args();folder=Path(args.folder);out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    pcd=next(folder.glob('*AllFaces.pcd'))
    start=time.perf_counter();xyz=read_xyz_pcd(pcd)*1000
    candidates,stats=arc_surface_candidates(xyz);seconds=time.perf_counter()-start
    candidate_file=out/'arc_contact_candidates.json'
    candidate_file.write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    # Labels are read only after the complete candidate file has been written.
    from weld_pipeline.truth import load_truth
    truth,missing=load_truth(folder);records=[]
    trees=[cKDTree(c['segments'][0]['points']) for c in candidates]
    for w in truth:
        for si,s in enumerate(w['segments']):
            if s['kind']!='arc':continue
            q=np.asarray(s['points'])
            errors=[float(t.query(q)[0].max()) for t in trees]
            best=int(np.argmin(errors)) if errors else None
            records.append({'weld_id':w['id'],'segment_index':si,
                            'best_candidate':best,'max_reference_to_candidate_mm':errors[best] if best is not None else None,
                            'covered_at_3mm':best is not None and errors[best]<=3.})
    sources=['arc_surface_candidates.py','local_tangent_circles.py','tangent_circle_candidates.py',
             'circle_candidates.py','point_frame.py','audit_arc_contacts.py']
    receipt={'scope':'Candidate curve coverage only; multiple reference welds may share a candidate. NOT one-to-one weld accuracy.',
             'pcd':str(pcd),'pcd_sha256':sha256(pcd),'candidate_sha256':sha256(candidate_file),
             'source_hashes':{n:sha256(Path('weld_pipeline')/n) for n in sources},
             'reference_hashes':{str(p):sha256(p) for p in [*folder.glob('*.xlsx'),*folder.glob('*welds.json')]},
             'point_only_candidate_generation':True,'read_and_candidates_seconds':seconds,
             'statistics':stats,'missing_reference_curves':missing,'arc_reference_segments':len(records),
             'covered_reference_segments':sum(r['covered_at_3mm'] for r in records),'per_arc':records}
    (out/'coverage.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['statistics','arc_reference_segments','covered_reference_segments','read_and_candidates_seconds']}),flush=True)


if __name__=='__main__':main()
