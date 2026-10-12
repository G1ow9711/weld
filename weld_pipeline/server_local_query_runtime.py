"""Score the same complete input queries with original raw point neighborhoods."""
import numpy as np
import torch

from .server_full_prediction_runtime import CompleteQueryProposals
from .server_full_training_samples import query_frame, geometry_descriptor
from .server_query_point_neighborhood import RawQueryNeighborhood


class LocalCompleteQueryProposals(CompleteQueryProposals):
    def __init__(self, model, mean, std, xyz, **kwargs):
        super().__init__(model, mean, std, **kwargs)
        self.local_points = RawQueryNeighborhood(xyz)

    @torch.inference_mode()
    def score_queries(self, inp, pid, positions, tokens):
        positions = np.asarray(positions, float)
        frames = [query_frame(inp, pid, float(q)) for q in positions]
        curve = inp.curve(pid)
        geometry = np.array([geometry_descriptor(curve, point, basis, known)
                             for point, basis, known in frames])
        raw = inp.features_at(pid, positions)
        local = self.local_points.features(inp, pid, positions)
        tensor = lambda value: torch.tensor(value, device=self.device, dtype=torch.float32)
        hidden = self.model.encode_queries(torch.clamp((tensor(raw) - self.mean) / self.std, -20, 20),
            torch.full((len(raw),), pid, device=self.device, dtype=torch.long),
            {pid: tokens}, self.normal_tokens, tensor(geometry), tensor(local))
        output = self.model.predict(hidden, tensor(raw[:, 456]))
        result = {name:value.detach().cpu().numpy() for name,value in output.items()}
        result['reference_probability'] = torch.sigmoid(output['reference_score']).cpu().numpy()
        return result, frames
