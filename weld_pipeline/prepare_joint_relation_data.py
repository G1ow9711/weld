"""Prepare label-free graph features and canonical training arrays with hash bounds."""
import json
import shutil
import time
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.joint_relation_heatmaps import make_graph, VERSION, LOCAL_CHANNELS


def main():
    started=time.perf_counter();out=Path('weld_pipeline/data/joint_relation_v1')
    if out.exists():
        raise ValueError('Frozen relation data exists')
    out.mkdir(parents=True)
    split_path=Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split=json.loads(split_path.read_text('utf8'));names=[name for name in split['train'] if '__' not in name]
    family_path=Path('weld_pipeline/reports/source_family_audit_20261001/summary.json')
    families=json.loads(family_path.read_text('utf8'))
    sources=[__file__,'weld_pipeline/joint_relation_heatmaps.py','weld_pipeline/canonical_horizontal_patch_heatmaps.py',
             'weld_pipeline/finite_contact_plate_profiles.py']
    config=dict(version=VERSION,pieces=names,source_sha256={path:sha256(path) for path in sources},
        split_sha256=sha256(split_path),source_folds_sha256=sha256(family_path),
        folds=families['folds'],feature_inputs_are_label_free=True,
        scope='H_straight_supervision_V_geometry_nodes_no_ARC_teacher_no_observed_check',
        materials_independently_verified=False,full_contract_pass=False)
    (out/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in sources:
        shutil.copy2(path,snapshot/Path(path).name)
    manifests={}
    for name in names:
        root=out/name;root.mkdir()
        piece=load_piece(split['piece_roots'][name],verify=True)
        pcd=Path(piece['record']['input'])
        if sha256(pcd)!=piece['record']['pcd_sha256']:
            raise ValueError('Raw source changed')
        xyz=read_xyz_pcd(pcd)*1000.
        with np.load(Path(split['structure_roots'][name])/(name+'.npz'),allow_pickle=False) as context:
            structural,process=context['structural'],context['process']
        graph=make_graph(xyz,piece['sequences'],structural,process,FiniteContactPlates(xyz))
        np.savez(root/'graph.npz',**{key:graph[key] for key in ['nodes','edge_index','edge_type','edge_features','sequence_indices']})
        total=sum(len(piece['sequences'][int(i)]['features']) for i in graph['sequence_indices'])
        features=np.lib.format.open_memmap(root/'features.npy',mode='w+',dtype=np.float32,shape=(total,len(LOCAL_CHANNELS)))
        targets=np.lib.format.open_memmap(root/'targets.npy',mode='w+',dtype=np.float32,shape=(total,4))
        entries=[];row=0
        for node,index in enumerate(graph['sequence_indices']):
            index=int(index);sequence,labels,transform=canonical_frame(piece['sequences'][index],piece['targets'][index])
            stop=row+len(labels)
            features[row:stop]=sequence['features'][:,LOCAL_CHANNELS]
            targets[row:stop]=labels[:,:4]
            events=np.flatnonzero(labels[:,1:4].max(1)>.25)
            foreground=np.flatnonzero(labels[:,0]>.5)
            entries.append(dict(sequence=index,node=node,type=sequence['type'],start=row,stop=stop,transform=transform,
                                event_indices=events.tolist(),foreground_indices=foreground.tolist()))
            row=stop
        features.flush();targets.flush();del features,targets
        manifest=dict(name=name,version=VERSION,entries=entries,source_cache_root=split['piece_roots'][name],
            original_cache_manifest_sha256=piece['cache_manifest_sha256'],pcd_sha256=piece['record']['pcd_sha256'],
            files={file:sha256(root/file) for file in ['features.npy','targets.npy','graph.npz']},
            graph_relation_counts=np.bincount(graph['edge_type'],minlength=3).tolist(),
            graph_uses_reference_labels=False,full_contract_pass=False)
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
        manifests[name]=sha256(root/'manifest.json')
        print('RELATION_DATA',name,'NODES',len(graph['nodes']),'EDGES',len(graph['edge_type']),'ROWS',row,flush=True)
        del graph,piece,xyz
    complete=dict(version=VERSION,manifests=manifests,elapsed_seconds=time.perf_counter()-started,
                  observed_check_read=False,full_contract_pass=False)
    (out/'complete.json').write_text(json.dumps(complete,indent=2,ensure_ascii=False),encoding='utf8')
    print('RELATION_DATA_COMPLETE',len(manifests),flush=True)


if __name__=='__main__':
    main()
