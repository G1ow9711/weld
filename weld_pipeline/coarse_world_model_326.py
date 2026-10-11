"""Pure-XYZ coarse geometry entry point; no local refinement or label inputs.

Keep the complete old parameter layout so an immutable Current65 checkpoint
loads strictly. Unused refinement/material heads receive no geometry gradient.
"""
import torch
from torch import nn

from .world_set_model_295 import LocalWorldModel
from .world_set_model_285 import observed_reference_rows


class CoarseWorldModel326(LocalWorldModel):
    def __init__(self):
        super().__init__(512, 16, 128, 16384)
        self.extra_process_heads = nn.Linear(128, 8)

    def forward(self, source):
        tokens, reference, origin, receipt = self.encode_original_source(source)
        positions = (source.mean_xyz_mm[source.counts > 0] - source.center_mm) / 1000.
        rows = torch.from_numpy(observed_reference_rows(positions, self.parent_queries))
        parents = self.parent_decoder(
            (tokens[rows] + self.parent_queries_embedding.weight
             + self.query_positions(reference))[None], tokens[None])[0]
        output = self._outputs(parents, reference, origin, dict(receipt))
        output["input_receipt"].update(
            local_refinement_stages=0,
            own_prediction_observation_positions=0,
            local_GT_or_STEP_reads=0,
            coarse_entry_point_version="PURE_XYZ_ORIGINAL_COARSE_326_v1")
        return output
