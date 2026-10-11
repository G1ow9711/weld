"""Resume the frozen 36-job protocol by replaying completed model receipts, never refitting them."""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

from training.labels import sha256
import weld_pipeline.train_workpiece_context_crossfit as trial

PRIOR = Path('weld_pipeline/runs/workpiece_context_crossfit_seed153')
OUT = Path('weld_pipeline/runs/workpiece_context_crossfit_seed153_resume_v2')


def main():
    if OUT.exists():
        raise ValueError('Preserve the current resumed run')
    cfg = json.loads((PRIOR/'config.json').read_text('utf8'))
    if (cfg['seeds'] != list(trial.SEEDS) or cfg['representations'] != list(trial.REPRESENTATIONS)
            or cfg['n_estimators'] != 128 or cfg['equivalent_frames'] != 16):
        raise ValueError('Original preregistered protocol changed')
    for path, digest in cfg['sources'].items():
        if sha256(path) != digest:
            raise ValueError('Original frozen training source changed: '+path)
    for path, key in ((trial.DATA/'config.json', 'data_config_sha256'),
                      (trial.DATA/'complete.json', 'data_complete_sha256')):
        if sha256(path) != cfg[key]:
            raise ValueError('Original data configuration changed')
    names = cfg['pieces']; source_rows = cfg['source_rows']; jobs = []
    all_rows = np.arange(max(b for _, b in source_rows.values()))
    for seed in trial.SEEDS:
        for representation in trial.REPRESENTATIONS:
            jobs.append(dict(stem=f'full_fit_seed{seed}_{representation}', seed=seed,
                representation=representation, train=names, evaluation_rows=all_rows))
    for fold in cfg['folds']:
        train = [n for n in fold['train_real'] if n in source_rows]
        held = [n for n in fold['held_out_real'] if n in source_rows]
        rows = np.concatenate([np.arange(*source_rows[n]) for n in held])
        for seed in trial.SEEDS:
            for representation in trial.REPRESENTATIONS:
                jobs.append(dict(stem=f'fold{fold["fold"]}_seed{seed}_{representation}', seed=seed,
                    representation=representation, train=train, held=held, evaluation_rows=rows))
    original_fit = trial.fit_head; call_index = 0; reused = {}; newly_trained = []

    def fixed_fit(x, y, weights, representation, seed):
        nonlocal call_index
        job = jobs[call_index]; call_index += 1
        expected_weights = trial.balanced_weights(y, job['train'], source_rows)
        if job['seed'] != seed or job['representation'] != representation or not np.array_equal(weights, expected_weights):
            raise ValueError('Resumed job sampling/source weights differ from original protocol')
        if call_index == 1:
            if json.loads((OUT/'config.json').read_text('utf8')) != cfg:
                raise ValueError('Resumed complete training configuration differs from the original')
            shutil.copy2(__file__, OUT/'source_snapshot'/Path(__file__).name)
            (OUT/'resume_protocol.json').write_text(json.dumps(dict(prior=str(PRIOR),
                prior_config_sha256=sha256(PRIOR/'config.json'), wrapper_sha256=sha256(__file__),
                frozen_protocol_unchanged=True, completed_models_are_replayed_not_refitted=True,
                old_outputs_are_not_modified=True, observed_check_read=False,
                full_contract_pass=False), indent=2), encoding='utf8')
        path = PRIOR/(job['stem']+'.joblib'); receipt_path = PRIOR/(job['stem']+'.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text('utf8'))
            digest = sha256(path)
            if digest != receipt['checkpoint_sha256']:
                raise ValueError('Completed model checksum changed: '+job['stem'])
            model = joblib.load(path)
            if (list(model.classes_) != [0, 1] or len(model.estimators_) != 128
                    or model.n_features_in_ != x.shape[1] or model.random_state != seed):
                raise ValueError('Completed model architecture/seed changed')
            rows = job['evaluation_rows']
            p = trial.predict_equivalent_views(model, x[rows], representation)
            score = trial.score_probability(p, y[rows], model.classes_)
            if score != receipt['score']:
                raise ValueError('Completed model score replay differs: '+job['stem'])
            if job['stem'].startswith('full_fit'):
                expected = np.load(PRIOR/(job['stem']+'_probabilities.npy'), allow_pickle=False)
                if not np.array_equal(p.astype(np.float32), expected):
                    raise ValueError('Completed model probability replay differs')
            elif receipt['train'] != job['train'] or receipt['held'] != job['held']:
                raise ValueError('Completed model source partition changed')
            reused[job['stem']] = digest
            print('WORKPIECE_CONTEXT_RESUME_REPLAY', job['stem'], 'NO_REFIT', flush=True)
        else:
            # Unfinished jobs restart independently, with their original seed and source weights.
            model = original_fit(x, y, weights, representation, seed)
            newly_trained.append(job['stem'])
            print('WORKPIECE_CONTEXT_RESUME_NEW', job['stem'], flush=True)
        with (OUT/'resume_jobs.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(job=job['stem'], reused=job['stem'] in reused,
                original_sha256=reused.get(job['stem']), prior_receipt=str(receipt_path)
                    if job['stem'] in reused else None))+'\n')
        return model

    trial.OUT = OUT; trial.fit_head = fixed_fit
    trial.main()
    if call_index != 36 or len(reused)+len(newly_trained) != 36:
        raise ValueError('Incomplete original 36-job protocol')
    for stem, digest in reused.items():
        if sha256(PRIOR/(stem+'.joblib')) != digest:
            raise ValueError('Original completed checkpoint changed during resume')
    receipt = dict(prior=str(PRIOR), complete_sha256=sha256(OUT/'complete.json'),
        original_config_sha256=sha256(PRIOR/'config.json'), resumed_config_sha256=sha256(OUT/'config.json'),
        reused_completed_models=len(reused), newly_trained_models=len(newly_trained),
        prior_checkpoint_sha256=reused, new_jobs=newly_trained, original_checkpoints_unchanged=True,
        original_protocol_fully_completed=True, observed_check_read=False, full_contract_pass=False,
        wrapper_sha256=sha256(__file__))
    (OUT/'resume_complete.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf8')
    print('WORKPIECE_CONTEXT_RESUME_COMPLETE', 'REUSED', len(reused), 'NEW', len(newly_trained), flush=True)


if __name__ == '__main__':
    main()
