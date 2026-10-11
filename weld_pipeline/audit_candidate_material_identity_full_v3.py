"""Schema-aware full FIT audit; preserve and verify the 15 completed v2 pieces."""
import collections
import json
import shutil
import time
from pathlib import Path

import gmsh
import numpy as np

from training.labels import sha256
from weld_pipeline.audit_candidate_material_identity import classify
from weld_pipeline.audit_step_weld_contacts import contacts
from weld_pipeline.curved_cache import load_piece

OUT = Path('weld_pipeline/reports/candidate_material_identity_full_148v3_20261001')
PARENT = Path('weld_pipeline/reports/candidate_material_identity_full_148v2_20261001')


def main():
    if OUT.exists():
        raise ValueError('Preserve fixed full audit')
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    refs = Path('weld_pipeline/reports/step_contact_continuity_candidates_v1')
    names = [n for n in split['train'] if '__' not in n and (refs/(n+'_attribute_candidates.json')).exists()]
    if len(names) != 19 or set(names) & set(split['validation']):
        raise ValueError('Exactly 19 original CAD-backed FIT pieces required')
    parent_config = json.loads((PARENT/'config.json').read_text('utf8'))
    if parent_config['pieces'] != names or parent_config['split_sha256'] != sha256(split_path):
        raise ValueError('Parent FIT scope changed')
    for source, digest in parent_config['sources'].items():
        if sha256(source) != digest:
            raise ValueError('Original audit algorithm changed')
    OUT.mkdir(parents=True)
    sources = [__file__, 'weld_pipeline/audit_candidate_material_identity.py',
               'weld_pipeline/audit_step_weld_contacts.py', 'weld_pipeline/curved_cache.py']
    config = dict(parent_config, sources={p:sha256(p) for p in sources},
                  parent_config_sha256=sha256(PARENT/'config.json'),
                  schema_repair='Missing STEP path is resolved from the hash-matched CAD description; pose and tolerances unchanged.',
                  cached_parent_samples_verified_against_current_sequences=True)
    (OUT/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    snap = OUT/'source_snapshot'; snap.mkdir()
    for source in sources:
        shutil.copy2(source,snap/Path(source).name)
    gmsh.initialize([],readConfigFiles=False); gmsh.option.setNumber('General.Terminal',0)
    gmsh.option.setString('Geometry.OCCTargetUnit','MM')
    results = {}; started = time.perf_counter()
    try:
        for name in names:
            cad_path = Path('weld_pipeline/reports/step_geometry_20260930')/(name+'.json')
            ref_path = refs/(name+'_attribute_candidates.json')
            cad = json.loads(cad_path.read_text('utf8')); ref = json.loads(ref_path.read_text('utf8'))
            if not ref['pose_verified_for_reference']:
                raise ValueError('Unverified finite-reference pose')
            step = Path(ref.get('step',cad['step']))
            if sha256(step) != ref['step_sha256'] or sha256(step) != cad['sha256']:
                raise ValueError('Original STEP differs')
            piece = load_piece(split['piece_roots'][name],verify=True)
            if piece['record']['pcd_sha256'] != ref['pcd_sha256'] or sha256(piece['record']['input']) != ref['pcd_sha256']:
                raise ValueError('Original PCD differs')
            prior = PARENT/(name+'.json')
            if prior.exists():
                result = json.loads(prior.read_text('utf8'))
                for key, digest in [('input_sha256',ref['pcd_sha256']),('step_sha256',ref['step_sha256']),
                                    ('cad_description_sha256',sha256(cad_path)),('pose_reference_sha256',sha256(ref_path))]:
                    if result[key] != digest:
                        raise ValueError('Parent audit input binding differs')
                indices = [i for i,s in enumerate(piece['sequences']) if s.get('curve_kind') != 'arc']
                if indices != [r['sequence'] for r in result['candidates']]:
                    raise ValueError('Parent candidate ordering differs')
                for row in result['candidates']:
                    seq = piece['sequences'][row['sequence']]; target = piece['targets'][row['sequence']]
                    wanted = np.unique(np.round(np.asarray(config['fractions'])*(len(seq['xyz'])-1)).astype(int))
                    if wanted.tolist() != [s['sample_index'] for s in row['samples']]:
                        raise ValueError('Parent sample ordering differs')
                    for sample in row['samples']:
                        i = sample['sample_index']
                        if not np.array_equal(seq['xyz'][i].astype(float),sample['xyz_mm']):
                            raise ValueError('Parent candidate point changed')
                        expected = bool(target[i,0]>.5) if target[i,0]>=0 else None
                        if sample['reference_foreground_diagnostic_only'] != expected:
                            raise ValueError('Parent diagnostic label changed')
                result['reused_completed_parent_sha256'] = sha256(prior)
            else:
                transform = np.asarray(ref['alignment']['transform'])
                rotation, translation = transform[:3,:3], transform[:3,3]
                gmsh.clear();gmsh.model.occ.importShapes(str(step.resolve()));gmsh.model.occ.synchronize()
                faces = [{**f,'volume':v['volume'],'board_name':v['name'],
                          'thickness':v['supported_thickness_mm'],'outward':dict(normal=None,verified=False)}
                         for v in cad['volumes'] for f in v['planar_faces']]
                rows=[]; counts=collections.Counter(); fg=collections.Counter(); bg=collections.Counter()
                for si,(seq,target) in enumerate(zip(piece['sequences'],piece['targets'])):
                    if seq.get('curve_kind')=='arc':
                        counts['arc_candidates_not_audited']+=1;continue
                    indices=np.unique(np.round(np.asarray(config['fractions'])*(len(seq['xyz'])-1)).astype(int))
                    samples=[]
                    for i in indices:
                        q=seq['xyz'][i].astype(float)
                        found=contacts(q,faces,rotation,translation)
                        hypothesis=classify(found,seq['features'][i,60:66].astype(float).reshape(2,3))
                        known=bool(target[i,0]>=0); weld=bool(target[i,0]>.5) if known else None
                        samples.append(dict(sample_index=int(i),fraction=float(i/(len(seq['xyz'])-1)),xyz_mm=q.tolist(),
                            reference_foreground_diagnostic_only=weld,finite_contact_faces=found,material_hypothesis=hypothesis))
                        counts[hypothesis['state']]+=1
                        if known:(fg if weld else bg)[hypothesis['state']]+=1
                    rows.append(dict(sequence=si,type=seq['type'],has_any_original_weld_supervision=bool((target[:,0]>.5).any()),
                                     samples=samples,identity_role='offline_reference_only_not_a_welding_side_definition'))
                result=dict(piece=name,candidates=rows,total_nonarc_candidates=len(rows),sample_class_counts=dict(counts),
                    foreground_sample_class_counts=dict(fg),background_sample_class_counts=dict(bg),
                    input_sha256=ref['pcd_sha256'],step_sha256=ref['step_sha256'],cad_description_sha256=sha256(cad_path),
                    pose_reference_sha256=sha256(ref_path),sample_count_is_not_child_geometry_denominator=True,
                    hidden_material_and_customer_truth_reviewed=False,full_contract_pass=False)
            result['source_cache_manifest_sha256']=piece['cache_manifest_sha256']
            if sha256(step)!=ref['step_sha256'] or sha256(piece['record']['input'])!=ref['pcd_sha256']:
                raise ValueError('Original audit input changed during execution')
            (OUT/(name+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
            results[name]={k:v for k,v in result.items() if k!='candidates'}
            print('FULL_MATERIAL_AUDIT',name,'FG',result['foreground_sample_class_counts'],
                  'BG',result['background_sample_class_counts'],'REUSED',prior.exists(),flush=True)
    finally:
        gmsh.finalize()
    complete=dict(config=config,pieces=results,elapsed_seconds=time.perf_counter()-started,
                  observed_check_read=False,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete,indent=2,ensure_ascii=False),encoding='utf8')
    print('FULL_MATERIAL_AUDIT_COMPLETE',len(results),flush=True)


if __name__=='__main__':
    main()
