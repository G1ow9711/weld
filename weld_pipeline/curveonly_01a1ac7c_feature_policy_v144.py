"""Independent feature and fusion factors for complete-curve family controls."""
import time
import numpy as np


class FeaturePolicy:
    def __init__(self, model, features, policy):
        assert features in ('old238', 'new756')
        assert policy in ('mixed', 'local')
        self.model = model
        self.features = features
        self.policy = policy
        self.last_timing = None

    def native(self, features):
        features = np.asarray(features)
        assert features.ndim == 2 and features.shape[1] == 108
        return features[:, :34] if self.features == 'old238' else features

    def predict(self, features, packet):
        tick = time.perf_counter()
        native = self.native(features)
        target = self.model if self.policy == 'mixed' else self.model.local
        result = target.predict(native, packet)
        self.last_timing = dict(cpu_seconds=time.perf_counter()-tick,
                                feature_factor=self.features, policy_factor=self.policy,
                                model=target.last_timing,
                                source_selection_and_export_unchanged=True)
        return result
