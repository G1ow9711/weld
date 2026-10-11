"""Point-input parent features and incremental set context; no target access."""
import numpy as np
from weld_pipeline.world_set_evaluation_285 import distance_to_primitive

RELATION_FIELDS = ('curve_a_to_b_log10mm', 'curve_b_to_a_log10mm',
    'whole_endpoint_difference_log10mm', 'log_length_ratio_abs', 'child_count_difference_log',
    'gap_count_difference_log', 'closed_difference', 'line_fraction_difference',
    'arc_fraction_difference', 'existence_mean_difference', 'observed_face_jaccard',
    'both_observed_faces_available', 'ordered_observed_face_pair_jaccard',
    'both_physical_sides_known', 'physical_sides_equal_when_known', 'whole_topology_equal')


def selected_chains(selection):
    successors = {a:(b,k) for a,b,k in selection['selected_links']}
    predecessors = {b:a for a,b,k in selection['selected_links']}
    visited=set(); paths=[]
    for i in selection['selected_nodes']:
        if i in predecessors: continue
        chain=[]; edges=[]; j=i
        while j not in visited:
            visited.add(j); chain.append(j)
            if j not in successors: break
            j,kind=successors[j]; edges.append(kind)
        paths.append(chain)
    for i in selection['selected_nodes']:
        if i in visited: continue
        chain=[]; edges=[]; j=i
        while j not in visited:
            visited.add(j); chain.append(j)
            if j not in successors: raise ValueError('Unclosed cycle')
            j,kind=successors[j]; edges.append(kind)
        if j!=i: raise ValueError('Cycle joins occupied path')
        boundary=next((k for k,e in enumerate(edges) if e!=1),None)
        if boundary is not None:
            shift=(boundary+1)%len(chain); chain=chain[shift:]+chain[:shift]
        paths.append(chain)
    assert visited==set(selection['selected_nodes'])
    return paths


def unary_features(decoded, nodes, selection, case):
    paths=selected_chains(selection); vectors=[]; metadata=[]
    for parent in decoded['parents']:
        chain=paths[parent['predicted_parent_query']]
        lengths=np.array([nodes[i]['hi']-nodes[i]['lo'] for i in chain])
        assert np.all(lengths>0)
        probabilities=np.array([nodes[i]['probability'] for i in chain])
        x=np.array(case['x'][[nodes[i]['object_row'] for i in chain]],dtype=np.float64)
        mean=(x*lengths[:,None]).sum(0)/lengths.sum()
        sides=[s.get('predicted_side','UNKNOWN') for s in parent['segments']]
        known=all(v not in (None,'UNKNOWN') for v in sides)
        length=sum(s['length_mm'] for s in parent['segments'])
        probability=float(lengths@probabilities/lengths.sum())
        extra=[float(probabilities.min()),probability,float(probabilities.max()),
               parent['child_count'],np.log1p(length),float(parent['is_closed']),len(parent['gaps']),float(known)]
        vectors.append(np.r_[mean,extra])
        face_pairs=[]
        for i in chain:
            node=nodes[i]; pid=case['graph'].profile(node['fragment'])
            faces=case['graph'].profiles[pid].get('face_indices',[])[:2]
            face_pairs.append(list(faces[::-1] if node['side'] else faces))
        metadata.append(dict(graph_nodes=chain,face_pairs=face_pairs,physical_sides=sides,side_known=known,
            probability_mean=probability,length_mm=length,
            line_fraction=sum(s['kind']=='line' for s in parent['segments'])/len(parent['segments'])))
    result=np.asarray(vectors,dtype=np.float64).reshape(-1,160)
    assert np.isfinite(result).all()
    return result,metadata


class ParentRelations:
    def __init__(self, parents, metadata):
        assert len(parents)==len(metadata)
        self.parents=parents; self.metadata=metadata
        self.points=[np.vstack([s['points'] for s in p['segments']]) for p in parents]
        self.faces=[{f for pair in m['face_pairs'] for f in pair if f>=0} for m in metadata]
        self.face_pairs=[{tuple(pair) for pair in m['face_pairs'] if len(pair)==2 and min(pair)>=0} for m in metadata]
        self.total=np.zeros((len(parents),16)); self.maximum=self.total.copy(); self.selected=[]

    def pair(self, a, b):
        pa,pb=self.parents[a],self.parents[b]; ma,mb=self.metadata[a],self.metadata[b]
        def directed(points, parent):
            distances=np.full(len(points),np.inf)
            for s in parent['segments']:
                distances=np.minimum(distances,distance_to_primitive(points,s))
            return float(distances.max())
        da=directed(self.points[a],pb); db=directed(self.points[b],pa)
        ea=self.points[a][[0,-1]]; eb=self.points[b][[0,-1]]
        end=min(float(np.linalg.norm(ea-eb,axis=1).max()),float(np.linalg.norm(ea-eb[::-1],axis=1).max()))
        fa,fb=self.faces[a],self.faces[b]; qa,qb=self.face_pairs[a],self.face_pairs[b]
        both=ma['side_known'] and mb['side_known']
        ta=tuple((s['kind'],s['weld_type']) for s in pa['segments'])
        tb=tuple((s['kind'],s['weld_type']) for s in pb['segments'])
        topology=(pa['is_closed']==pb['is_closed'] and
                  ((ta==tb and pa['gaps']==pb['gaps']) or (ta==tb[::-1] and pa['gaps']==pb['gaps'][::-1])))
        v=np.array([np.log1p(da/10),np.log1p(db/10),np.log1p(end/10),
            abs(np.log(ma['length_mm']/mb['length_mm'])),np.log1p(abs(pa['child_count']-pb['child_count'])),
            np.log1p(abs(len(pa['gaps'])-len(pb['gaps']))),float(pa['is_closed']!=pb['is_closed']),
            abs(ma['line_fraction']-mb['line_fraction']),abs((1-ma['line_fraction'])-(1-mb['line_fraction'])),
            abs(ma['probability_mean']-mb['probability_mean']),len(fa&fb)/max(1,len(fa|fb)),float(bool(fa and fb)),
            len(qa&qb)/max(1,len(qa|qb)),float(both),float(both and ma['physical_sides']==mb['physical_sides']),float(topology)])
        assert v.shape==(16,) and np.isfinite(v).all()
        return v

    def add(self, index):
        if index in self.selected: raise ValueError('Repeated candidate')
        self.selected.append(index); occupied=set(self.selected)
        for i in range(len(self.parents)):
            if i in occupied: continue
            v=self.pair(i,index); self.total[i]+=v; self.maximum[i]=np.maximum(self.maximum[i],v)

    def features(self):
        if not self.selected: return np.zeros((len(self.parents),32))
        return np.concatenate((self.maximum,self.total/len(self.selected)),axis=1)
