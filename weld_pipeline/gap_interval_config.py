"""Load a checkpoint-bound decoder configuration, never reference curves."""
import json,math
from pathlib import Path
from training.labels import sha256


def load_interval_config(path,checkpoint):
    path=Path(path);config=json.loads(path.read_text(encoding='utf-8'))
    if config.get('version')!='shared_gap_interval_decoder_v1':raise ValueError('Unsupported decoder configuration')
    if config['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('Decoder calibrated for a different checkpoint')
    for source,wanted in config['sources'].items():
        if sha256(source)!=wanted:raise ValueError('Decoder source changed after calibration: '+source)
    parameters=config['parameters']
    if set(parameters)!={'threshold','weld_penalty','mask_weight','max_background_gap_mm'}:raise ValueError('Unexpected decoder parameters')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in parameters.values()):raise ValueError('Nonfinite decoder parameter')
    if not 0<parameters['threshold']<1 or not 0<=parameters['weld_penalty']<=20 or not 0<parameters['mask_weight']<=5 or not 0<=parameters['max_background_gap_mm']<=100:
        raise ValueError('Decoder parameter outside permitted range')
    return parameters,config
