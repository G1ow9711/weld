"""Supplement the unchanged endpoint/vertex audit with surface distances."""
import argparse,json
from pathlib import Path
import numpy as np
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.recover_single_weld_reference import read_reference_ply
from weld_pipeline.reference_surface_distance import reference_surface_distances


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',required=True);ap.add_argument('--audit',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();folder=Path(args.folder);audit_path=Path(args.audit)
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    for p,h in audit['source_hashes'].items():
        if sha256(p)!=h:raise ValueError('Reference audit source changed: '+p)
    ply=next(folder.glob('*workpiece.ply'));pcd=next(folder.glob('*AllFaces.pcd'))
    dense=read_reference_ply(ply);raw=read_xyz_pcd(pcd)*1000
    checks=[]
    for item in audit['coordinate_checks']:
        matrix=np.asarray(item['exportToWorld']);local=(raw-matrix[:3,3])@matrix[:3,:3]
        errors=reference_surface_distances(dense,local)
        checks.append({'median_mm':float(np.median(errors)),'p95_mm':float(np.quantile(errors,.95)),
                       'p99_mm':float(np.quantile(errors,.99)),'max_mm':float(errors.max()),'over_3_mm':int((errors>3).sum()),
                       'exportToWorld':matrix.tolist()})
        print(json.dumps(checks[-1]),flush=True)
    chosen=int(np.argmin([c['p95_mm'] for c in checks]));accepted=checks[chosen]['max_mm']<=3 and checks[1-chosen]['p95_mm']>checks[chosen]['p95_mm']+5
    report={'status':'FRAME_SUPPORTED' if accepted else 'FRAME_REQUIRES_REVIEW','selected_transform':chosen if accepted else None,
            'scope':'Reference frame audit with bounded local triangles, no extrapolation; not weld extraction accuracy.',
            'source_hashes':{str(p):sha256(p) for p in [Path(__file__),Path('weld_pipeline/reference_surface_distance.py'),audit_path,ply,pcd]},
            'checks':checks,'max_triangle_edge_mm':20.,'neighbors':64,'reference_voxel_mm':2.}
    Path(args.out).write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
