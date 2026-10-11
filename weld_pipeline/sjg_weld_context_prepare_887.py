"""Freeze raw-recomputed continuous material context on existing profile grids."""
import argparse
import gc
import traceback
import shutil
import numpy as np
import torch
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_sequence_model_820 import configure_precision
from weld_pipeline.sjg_weld_material_context_886 import material_fields,MaterialQueryContext,CONTEXT_NAMES


def main(reuse_run=None):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);configure_precision()
    inventory=read(OUT/'inventory.json');frozen=[];rows=[]
    write(dest/'registered_protocol.json',dict(stage='RAW_MATERIAL_WELD_CONTEXT',sources=TRAIN+VALID+ARC,
        features=CONTEXT_NAMES,original_G=496,source_sha256=digest(__file__),
        context_source_sha256=digest(ROOT/'weld_pipeline/sjg_weld_material_context_886.py'),
        profile_selection='unchanged run51 union of TRAIN and input-only inference profiles',
        GT_reads=0,STEP_reads=0,material_ID_unknown=True,full_contract_pass=False,start_utc=utc()))
    try:
        for source in TRAIN+VALID+ARC:
            check_resources();folder=dest/f'source_{source:02d}';folder.mkdir()
            previous=OUT/reuse_run/f'source_{source:02d}' if reuse_run else None
            if previous is not None and (previous/'manifest.json').exists():
                prior_protocol=read(OUT/reuse_run/'registered_protocol.json')
                assert prior_protocol['context_source_sha256']==digest(ROOT/'weld_pipeline/sjg_weld_material_context_886.py')
                prior=read(previous/'manifest.json')
                assert prior['raw_sha256']==digest(ROOT/inventory[source]['raw']['original_pcd_path'])
                assert prior['previous_manifest_sha256']==digest(OUT/f'run51/source_{source:02d}/manifest.json')
                for r in prior['profiles']:
                    assert digest(previous/r['file'])==r['sha256'];shutil.copy2(previous/r['file'],folder/r['file'])
                shutil.copy2(previous/'manifest.json',folder/'manifest.json')
                frozen.append(dict(source=source,manifest_sha256=digest(folder/'manifest.json'),reused_from=reuse_run))
                rows.append(dict(source=source,points=prior['points'],profiles=len(prior['profiles']),grid_points=sum(r['points'] for r in prior['profiles']),reused_from=reuse_run))
                print('VERIFIED_CONTEXT_REUSED',rows[-1],flush=True);continue
            base=OUT/f'run34/source_{source:02d}';binding=read(base/'input_freeze.json')
            for name in ('profiles.json','surfaces.json'):assert digest(base/name)==binding['files'][name]
            raw=ROOT/inventory[source]['raw']['original_pcd_path'];assert digest(raw)==binding['raw_sha256']
            xyz=read_xyz_pcd(raw)*1000.;faces=read(base/'surfaces.json');profiles=read(base/'profiles.json')
            old=OUT/f'run51/source_{source:02d}';manifest=read(old/'manifest.json')
            fields=material_fields(xyz,faces,lambda *a:print('CONTEXT_SOURCE',source,*a,flush=True))
            assert fields['every_original_point_once'];context=MaterialQueryContext(xyz,profiles,fields);records=[]
            for number,r in enumerate(manifest['profiles'],1):
                assert digest(old/r['file'])==r['sha256']
                with np.load(old/r['file']) as z:q=z['q_mm']
                extra=context(r['profile'],q);path=folder/r['file'];np.savez_compressed(path,q_mm=q,extra=extra)
                records.append(dict(profile=r['profile'],file=path.name,sha256=digest(path),points=len(q)))
                if number%100==0:print('CONTEXT_PROFILE',source,number,'/',len(manifest['profiles']),flush=True)
            write(folder/'manifest.json',dict(profiles=records,raw_sha256=digest(raw),points=len(xyz),
                previous_manifest_sha256=digest(old/'manifest.json'),weights=fields['weights'],support_bindings=fields['support_bindings'],
                every_original_point_once=True,raw_XYZ_recomputed=True))
            frozen.append(dict(source=source,manifest_sha256=digest(folder/'manifest.json')))
            rows.append(dict(source=source,points=len(xyz),profiles=len(records),grid_points=sum(r['points'] for r in records)))
            print('RAW_MATERIAL_CONTEXT_FROZEN',rows[-1],flush=True)
            del xyz,faces,profiles,fields,context;gc.collect();torch.cuda.empty_cache()
        write(dest/'all_inputs_frozen.json',dict(sources=frozen,GT_reads=0,STEP_reads=0))
        write(dest/'receipt.json',dict(status='RAW_MATERIAL_WELD_CONTEXT_COMPLETE',rows=rows,optimizer_updates=0,
            full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
