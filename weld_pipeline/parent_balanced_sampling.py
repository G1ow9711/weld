"""Keep each source family's prior sampling mass when adding TRAIN variants."""
import numpy as np


def family(record):
    if record.get('target_policy') == 'all_channel_minus_one_is_unknown':
        if not record.get('source_group'):
            raise ValueError('Partial augmentation requires its original TRAIN parent')
        return ('partial', record['source_group'])
    return ('complete', record['name'])


class SourceBalancedPool:
    def __init__(self, pieces, reference_records, original_train, partial_repeat=4):
        if not pieces or not 1 <= partial_repeat <= 8:
            raise ValueError('Invalid source-balanced pool')
        self.pieces = list(pieces)
        names = [p['record']['name'] for p in pieces]
        reference_names = [r['name'] for r in reference_records]
        if len(names) != len(set(names)) or len(reference_names) != len(set(reference_names)):
            raise ValueError('Duplicate training entry')
        if not set(reference_names) <= set(names):
            raise ValueError('A previous training view was removed')
        parents = set(original_train)
        mass, counts = {}, {}
        for record in reference_records:
            key = family(record)
            mass[key] = mass.get(key, 0.) + (partial_repeat if key[0] == 'partial' else 1.)
        for piece in pieces:
            key = family(piece['record'])
            if key[0] == 'partial' and key[1] not in parents:
                raise ValueError('Augmentation parent outside original TRAIN')
            if key not in mass:
                raise ValueError('New source family needs an explicitly declared sampling budget')
            counts[key] = counts.get(key, 0) + 1
        self.weights = np.array([mass[family(p['record'])]/counts[family(p['record'])] for p in pieces], dtype=float)
        self.probabilities = self.weights/self.weights.sum()
        self.cumulative = np.cumsum(self.probabilities)
        self.cumulative[-1] = 1.
        self.report = {'policy': 'Prior sampling mass per source family; equal probability within each family.',
                       'unique_views': len(pieces), 'reference_views': len(reference_records),
                       'prior_mass': float(self.weights.sum()),
                       'entries': [{'name': p['record']['name'], 'family': list(family(p['record'])),
                                    'weight': float(w), 'probability': float(prob)}
                                   for p,w,prob in zip(pieces,self.weights,self.probabilities)]}

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, index):
        return self.pieces[index]

    def sample(self, rng):
        return self.pieces[int(np.searchsorted(self.cumulative, rng.random(), side='right'))]


def select_piece(data, rng):
    return data.sample(rng) if isinstance(data, SourceBalancedPool) else data[int(rng.integers(len(data)))]
