"""Authenticate complete frozen point features before reusing a substage."""
from pathlib import Path

import numpy as np

from .sjg_local_runtime_904 import read,digest

ROOT=Path(__file__).resolve().parents[1]


def verify_complete_component(module,key,model_source):
    component='point_features' if module.__name__.endswith('infer_legacy_point_features') else 'run26_point_features'
    folder=module.RESTORE/component/('source_'+key)
    block_path=module.BLOCKS/('source_'+key)/'manifest.json'
    case=read(block_path);manifest=read(folder/'feature_manifest.json');protocol=read(folder/'protocol.json')
    checkpoint=module.MODEL_DIR/'last.pt';receipt=module.MODEL_DIR/'receipt.json'
    if (manifest['status']!='POINT_FEATURES_COMPLETE' or not manifest['core_coverage_complete']
            or not manifest['original_core_coverage_once'] or manifest['source_key']!=key
            or manifest['original_points']!=case['original_points']
            or manifest['input_sha256']!=case['input_sha256']
            or protocol['input_sha256']!=case['input_sha256']
            or protocol['block_manifest_sha256']!=digest(block_path)
            or protocol['source_sha256']!=digest(module.__file__)
            or protocol['model_source_sha256']!=digest(ROOT/model_source)
            or protocol['checkpoint_receipt_sha256']!=digest(receipt)
            or protocol['checkpoint_sha256']!=read(receipt)['checkpoint_sha256']
            or protocol['checkpoint_sha256']!=digest(checkpoint)
            or protocol['limit_blocks']!=0 or protocol['optimizer_updates']!=0
            or protocol['GT_reads']!=0 or protocol['STEP_reads']!=0
            or len(manifest['blocks'])!=len(case['blocks'])
            or digest(ROOT/case['input_file'])!=case['input_sha256']):
        raise ValueError('Complete point-feature component provenance differs')
    seen=np.zeros(case['original_points'],np.uint8)
    for index,(item,block) in enumerate(zip(manifest['blocks'],case['blocks'])):
        if (item['file']!=f'features_{index:04d}.npz' or item['core_points']!=block['core_count']
                or item['input_block_sha256']!=block['sha256']
                or digest(folder/item['file'])!=item['sha256']
                or digest(module.BLOCKS/block['file'])!=block['sha256']):
            raise ValueError('Frozen feature or input block differs')
        with np.load(module.BLOCKS/block['file'],allow_pickle=False) as data:
            ids=data['original_point_ids'];count=int(data['core_count'])
        with np.load(folder/item['file'],allow_pickle=False) as data:
            result={name:data[name] for name in data.files}
        if count!=block['core_count']:raise ValueError('Original block core count differs')
        module.validate_output(result,ids,count)
        np.add.at(seen,result['original_point_ids'],1)
        if np.any(seen>1):raise ValueError('Complete point-feature cores overlap')
    if not np.all(seen==1):raise ValueError('Complete point-feature cores omit original points')
    return dict(status='COMPLETE_FROZEN_POINT_COMPONENT_AUTHENTICATED_FOR_REUSE',
        component=component,original_points=len(seen),checked_blocks=len(case['blocks']),
        manifest_sha256=digest(folder/'feature_manifest.json'),protocol_sha256=digest(folder/'protocol.json'),
        block_manifest_sha256=digest(block_path),checkpoint_sha256=protocol['checkpoint_sha256'],
        verifier_sha256=digest(__file__),model_updates=0,input_coordinates_changed=False)
