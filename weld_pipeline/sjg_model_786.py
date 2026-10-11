"""Local-frame graph model retaining relative orientation and displacement."""
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_model_777 import SurfaceJointNet777


class SurfaceJointNet786(SurfaceJointNet777):
    def __init__(self, node_dim=26, width=96, message_passing=True):
        super().__init__(node_dim, width, message_passing)
        self.messages = nn.ModuleList([nn.Sequential(nn.Linear(width*2+8, width),
            nn.SiLU(), nn.Linear(width, width)) for _ in range(2)])

    def encode_graph(self, nodes, edges, types):
        h = self.node((nodes-self.node_mean)/self.node_scale)
        if self.message_passing and edges.shape[1]:
            src, dst = edges
            degree = torch.zeros(len(h), device=h.device).index_add_(0, dst,
                torch.ones(len(dst), device=h.device)).clamp_min(1)
            for i, (layer, norm) in enumerate(zip(self.messages, self.norms)):
                msg = layer(torch.cat([h[src], h[dst], types], dim=1))
                agg = torch.zeros_like(h).index_add_(0, dst, msg)/degree[:, None]
                h = h+self.message_gate[i].tanh()*F.silu(norm(agg))
        return h
