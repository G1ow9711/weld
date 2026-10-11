"""Read-only TRAIN views containing point-derived vertical candidates only."""


def vertical_views(pieces):
    views, report = [], []
    for piece in pieces:
        indices = [i for i, seq in enumerate(piece['sequences'])
                   if seq['type'] == 'vertical' and (piece['targets'][i][:, 0] >= 0).any()]
        report.append({'group': piece['record']['name'], 'source_sequence_indices': indices,
                       'original_sequence_count': len(piece['sequences'])})
        if not indices:
            continue
        view = dict(piece)
        view['sequences'] = [piece['sequences'][i] for i in indices]
        view['targets'] = [piece['targets'][i] for i in indices]
        view.pop('hard_samples', None)
        view['source_sequence_indices'] = indices
        if any(not (s['features'][:, 66] > .5).all() for s in view['sequences']):
            raise ValueError('Point-derived type and feature flag disagree')
        views.append(view)
    if not views:
        raise ValueError('No known vertical training supervision')
    return views, report
