"""Reference-centered geometry diagnostic, never an inference input or new label."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from training.labels import sha256
from training.dataio import read_xyz_pcd


def patches(name,radius):
    folder=Path('weld_pipeline/data/mixed_surface_v1')/name
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    source=Path(manifest['record']['input'])
    if sha256(source)!=manifest['record']['pcd_sha256']:raise ValueError('Raw point cloud changed')
    xyz=read_xyz_pcd(source)*1000
    features=np.load(folder/'features.npy',mmap_mode='r')
    mapping=manifest['record']['mapping']['line_mapping']['weld_to_sequence'];rows=[]
    for i,w in enumerate(manifest['truth']):
        if w['type']!='vertical':continue
        si=mapping[str(i)];s=manifest['sequences'][si];f=features[(s['start']+s['stop'])//2]
        basis=np.stack([f[60:63],f[63:66],np.array([0.,0.,1.])])
        if not np.allclose(basis@basis.T,np.eye(3),atol=1e-4):raise ValueError('Nonorthogonal diagnostic frame')
        start=np.array(w['segments'][0]['points'][0]);end=np.array(w['segments'][-1]['points'][-1]);center=(start+end)/2
        selected=xyz[(np.linalg.norm(xyz[:,:2]-center[:2],axis=1)<=radius)&(np.abs(xyz[:,2]-center[2])<=90)]
        local=(selected-center)@basis.T
        _,keep=np.unique(np.floor(local/2).astype(np.int64),axis=0,return_index=True);local=local[np.sort(keep)]
        rows.append({'number':w['metadata']['WeldNumber (f/v + Name number)'],'segments':len(w['segments']),
                     'side_split':w['metadata']['IsSplitedForDualRobot (Any)'],'points':local})
    return rows,{'pcd':str(source),'pcd_sha256':sha256(source),'manifest_sha256':sha256(folder/'manifest.json')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outdir',default='weld_pipeline/reports/vertical_context_audit');a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);results=[];hashes={};begin=time.perf_counter()
    for radius in [50.,150.]:
        train,hashes['train']=patches('224-GR24A1',radius);dev,hashes['development']=patches('527-FR146A',radius)
        for row in dev:
            q=row['points'];treeq=cKDTree(q);best=None
            for candidate in train:
                p=candidate['points']
                for swap in [False,True]:
                    for sx,sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                        transform=np.eye(3)
                        if swap:transform=transform[[1,0,2]]
                        transform[:2]*=np.array([sx,sy])[:,None]
                        r=p@transform.T
                        distances=np.concatenate([treeq.query(r)[0],cKDTree(r).query(q)[0]])
                        metric=(float(np.quantile(distances,.95)),float(distances.mean()))
                        if best is None or metric<best[0]:
                            best=(metric,{'train_number':candidate['number'],'train_segments':candidate['segments'],
                                'train_side_split':candidate['side_split'],'train_points':len(p),'swap_xy':swap,
                                'signs':[sx,sy],'p95_mm':metric[0],'mean_mm':metric[1],'max_mm':float(distances.max()),
                                'fraction_within_3mm':float((distances<=3).mean())})
            result={'radius_mm':radius,'development_number':row['number'],'development_segments':row['segments'],
                    'development_side_split':row['side_split'],'development_points':len(q),'nearest_training_patch':best[1]}
            results.append(result);print(json.dumps(result),flush=True)
    report={'scope':'Reference-centered local shape comparison only. No training data, inference, or labels changed. Similar local patches do not prove whole-PCD ambiguity.',
            'voxel_mm':2,'vertical_half_window_mm':90,'groups':hashes,'comparisons':results,
            'source_sha256':sha256(Path(__file__)),'elapsed_seconds':time.perf_counter()-begin}
    (out/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
