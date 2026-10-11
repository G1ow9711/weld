"""Shared full graph model plus every contract attribute; pure source inputs."""
import torch
from .fully_typed_graph_heads_281 import FullyTypedGraphHeads
from .contract_attribute_heads_283 import ContractAttributeHeads
from .typed_prediction_codec_281 import SLOT_COLUMNS


class ContractGraphHeads(FullyTypedGraphHeads):
    def __init__(self, hidden=64):
        super().__init__(hidden)
        self.contract_attributes = ContractAttributeHeads(hidden)
        self.child_role_column = next(i for i, value in enumerate(SLOT_COLUMNS) if value[0] == "child")

    def predict_child_attributes(self, output, child_anchor_rows, sample_fractions, predicted_xyz_m, predicted_tangents):
        if (child_anchor_rows.ndim != 1 or child_anchor_rows.dtype != torch.long
            or ((child_anchor_rows < 0) | (child_anchor_rows >= len(output["anchor_embeddings"]))).any()):
            raise ValueError("Attribute queries must reference actual predicted child anchor contexts")
        child_context = output["anchor_embeddings"][child_anchor_rows] + self.role_embedding.weight[self.child_role_column]
        return self.contract_attributes(child_context, sample_fractions, predicted_xyz_m, predicted_tangents)
