"""Pure source reader supporting separately retained independent review attempts."""
import numpy as np
import torch
from .pure_pcd_graph_heads_275_revision2 import SourceFeatures
from .raw_triangle_neural_features_281 import RawTriangleCatalog

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

