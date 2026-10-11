"""All legacy/carrier/RawTriangle domains, with a full streamed denominator.

Chunking changes memory use. It never removes candidates or uses truth to build
initial inputs. This module does not claim a trained model or complete decoder.
"""
import math
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .pure_pcd_graph_heads_275_revision2 import SourceFeatures, SourceGraphHeads, CATEGORIES
from .raw_triangle_neural_features_279 import CHANNELS, RawTriangleCatalog

VERSION="PURE_PCD_ALL_ORIGINAL_MIXED_DOMAINS_AND_COMPLETE_STREAMED_POINTER_LOSS_279_v1"

def readonly_tensor(array):
    """A read-only mmap tensor view. Neither forward nor replay mutates it."""
    array=np.asarray(array)
    if array.dtype!=np.float32:
        raise ValueError("Frozen float32 neural input required")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="The given NumPy array is not writable")
        return torch.from_numpy(array)

class MixedSourceFeatures(SourceFeatures):
    def __init__(self,root,entry,*,verify_artifact_hashes=True):
        super().__init__(root,entry,verify_artifact_hashes=verify_artifact_hashes)
        self.raw_triangles=RawTriangleCatalog(root,entry)
        if self.raw_triangles.receipt["initial_source_input_receipt_sha256"]!=entry["producer_receipt_sha256"]:
            raise ValueError("Mixed features must bind the same original initial input source")
        self.legacy_domain_count=len(self.domain_records)
        self.total_domain_count=self.legacy_domain_count+self.raw_triangles.count
        self.initial_input_contract.update(all_original_raw_triangle_occurrences=self.raw_triangles.count,
            full_mixed_domain_count=self.total_domain_count, teacher_selected_triangle_reads=0,
            raw_triangles_add_role_capacity=False)

    def batch(self,anchor_rows,device="cpu"):
        if str(device)!="cpu":
            raise ValueError("This frozen, measured replay implementation currently requires CPU")
        result=super().batch(anchor_rows,device)
        result["raw_triangle_context"]=torch.from_numpy(np.array(self.raw_triangles.context,dtype=np.float32,copy=True))
        return result

    def mixed_domain_index(self,domain):
        if domain.get("type")=="RAW_TRIANGLE":
            return self.legacy_domain_count+self.raw_triangles.index(domain)
        return super().domain_index(domain)

    def mixed_domain_record(self,index):
        if type(index) is not int or not 0<=index<self.total_domain_count:
            raise ValueError("Pointer outside unchanged complete mixed domain pool")
        return (dict(self.domain_records[index]) if index<self.legacy_domain_count else
                self.raw_triangles.record(index-self.legacy_domain_count))

class MixedSourceGraphHeads(SourceGraphHeads):
    def __init__(self,hidden=64):
        super().__init__(hidden)
        def encoder(channels):
            return nn.Sequential(nn.Linear(channels,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.raw_triangle_encoder=encoder(len(CHANNELS))
        self.raw_triangle_context_encoder=encoder(len(CHANNELS))
        self.raw_triangle_type=nn.Parameter(torch.zeros(hidden))
        self.raw_context_fusion=nn.Sequential(nn.Linear(hidden*2,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())

    def forward(self,inputs,*,observed_domain_mask=None):
        out=super().forward(inputs,observed_domain_mask=observed_domain_mask)
        if inputs["raw_triangle_context"].shape!=(len(CHANNELS),):
            raise ValueError("Context from every original RawTriangle is required")
        context=self.raw_triangle_context_encoder(inputs["raw_triangle_context"]).expand(len(out["anchor_embeddings"]),-1)
        hidden=self.raw_context_fusion(torch.cat((out["anchor_embeddings"],context),dim=-1))
        out.update(presence_logits=self.presence(hidden),numeric=self.numeric(hidden).reshape(-1,6,12),
            categories={name:layer(hidden).reshape(-1,6,CATEGORIES[name]) for name,layer in self.category.items()},
            domain_queries=self.domain_query(hidden).reshape(-1,6,self.hidden),anchor_embeddings=hidden,
            mixed_domain_version=VERSION)
        return out

    def raw_embeddings(self,features):
        return self.raw_triangle_encoder(features)+self.raw_triangle_type

    def _raw_log_partition(self,queries,features):
        logits=queries@self.raw_embeddings(features).T/math.sqrt(self.hidden)
        # Accumulate the full million-candidate partition in float64. Scores
        # and learned parameters remain float32; geometry authority is separate.
        return torch.logsumexp(logits.to(torch.float64),dim=1)

    def mixed_domain_cross_entropy(self,queries,legacy_embeddings,targets,catalog,*,chunk_size=1024,replay=True):
        """Exact full candidate softmax, replaying each chunk during backward.

        All original rows enter the denominator. Only the positive numerator
        reads explicit supervised pointers, as a loss input, never a feature.
        Use ordinary loss.backward(); the installed checkpoint API forbids
        autograd.grad and backward(inputs=...) for reentrant replay.
        """
        if queries.device.type!="cpu" or legacy_embeddings.device.type!="cpu":
            raise ValueError("Frozen measured CPU replay is required")
        if queries.ndim!=2 or queries.shape[1]!=self.hidden or targets.shape!=(len(queries),):
            raise ValueError("All explicit positive domain query rows and targets required")
        legacy_count=len(legacy_embeddings)
        count=legacy_count+catalog.count
        if targets.dtype!=torch.long or ((targets<0)|(targets>=count)).any():
            raise ValueError("Known target outside the unchanged complete mixed domain pool")
        if not len(queries):
            return queries.sum()*0,dict(candidate_count=count,queries=0,raw_rows_consumed=0,unknown_targets=True)
        legacy_logits=queries@legacy_embeddings.T/math.sqrt(self.hidden)
        partition=torch.logsumexp(legacy_logits.to(torch.float64),dim=1)
        visited=0
        for start,stop,features in catalog.blocks(chunk_size):
            block=readonly_tensor(features)
            part=(checkpoint(self._raw_log_partition,queries,block,preserve_rng_state=False)
                  if replay and torch.is_grad_enabled() and queries.requires_grad
                  else self._raw_log_partition(queries,block))
            partition=torch.logaddexp(partition,part)
            visited+=stop-start
        if visited!=catalog.count:
            raise ValueError("Full raw domain denominator was not consumed")
        is_legacy=targets<legacy_count
        positive=torch.zeros_like(partition,dtype=legacy_logits.dtype)
        if is_legacy.any():
            positive[is_legacy]=legacy_logits[is_legacy,targets[is_legacy]]
        if (~is_legacy).any():
            raw_rows=(targets[~is_legacy]-legacy_count).detach().numpy()
            selected=torch.from_numpy(np.array(catalog.features[raw_rows],dtype=np.float32,copy=True))
            positive[~is_legacy]=(queries[~is_legacy]*self.raw_embeddings(selected)).sum(dim=1)/math.sqrt(self.hidden)
        loss=(partition-positive).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite mixed domain loss; preserve failure rather than drop targets")
        return loss,dict(candidate_count=count,queries=len(queries),raw_rows_consumed=visited,
            legacy_and_carrier_rows_consumed=legacy_count,full_denominator=True,chunk_size=chunk_size,
            backward_replays_all_raw_candidate_chunks=bool(replay),teacher_selected_initial_triangles=False)

    @torch.no_grad()
    def full_mixed_domain_argmax(self,queries,legacy_embeddings,catalog,*,chunk_size=1024):
        """Runtime selection sees every observed domain; no teacher pointer input."""
        if queries.device.type!="cpu" or queries.ndim!=2 or queries.shape[1]!=self.hidden:
            raise ValueError("Frozen CPU mixed domain query schema required")
        logits=queries@legacy_embeddings.T/math.sqrt(self.hidden)
        if not len(legacy_embeddings) or not torch.isfinite(logits).all():
            raise ValueError("Finite complete legacy/carrier pool required")
        values,indices=logits.max(dim=1)
        visited=0
        for start,stop,features in catalog.blocks(chunk_size):
            raw_logits=queries@self.raw_embeddings(readonly_tensor(features)).T/math.sqrt(self.hidden)
            if not torch.isfinite(raw_logits).all():
                raise ValueError("Nonfinite original raw candidate scores cannot be dropped")
            local_values,local_indices=raw_logits.max(dim=1)
            better=local_values>values
            values=torch.where(better,local_values,values)
            indices=torch.where(better,local_indices+len(legacy_embeddings)+start,indices)
            visited+=stop-start
        if visited!=catalog.count:
            raise ValueError("Incomplete runtime mixed domain pool")
        return indices,values,dict(all_initial_domains=len(legacy_embeddings)+visited,
             raw_rows_consumed=visited,GT_STEP_or_teacher_selected_domains_read=False)
