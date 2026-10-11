# Derived from 691 SHA256 392fb4646080aac55313413d33f2fef2ae08251e42bea2f7953901e3a69ae821; only child-maximum training risk changed.
"""One TRAIN objective for D3 parent matching, fixed-pair trials and backward.

Truth conditions training geometry only. Free inference retains its own
count/gap/margin decisions. Old query assignments never constrain matching.
"""
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from weld_pipeline.count_conditioned_event_loss_581 import resolved_views
from weld_pipeline.legacy_incremental_loss_396 import world_positions
from weld_pipeline.legacy_legal_allocation_model_518 import margin_logits_for_counts

VERSION = 'BOUNDED639_CHILD_MAX_SOFTPLUS_COMPLETE_PAIR_MINUS_BACKGROUND_V1'
SOFT_KEEP_WEIGHT = .2


def child_endpoint_risk(endpoint_errors_mm, original_G):
    """One risk per real child; a shared bad endpoint affects both children."""
    assert endpoint_errors_mm.ndim == 3 and endpoint_errors_mm.shape[-1] == 2
    assert bool(torch.isfinite(endpoint_errors_mm).all())
    return F.softplus((endpoint_errors_mm.amax(-1)-3.)/1.).sum(-1)/max(1,original_G)


def minimum_assignment(positive_by_parent, background):
    """Each tensor is [query, allowed whole-parent view]."""
    if not positive_by_parent:
        return []
    costs, choices = [], []
    for positive in positive_by_parent:
        matrix = positive.detach().double().cpu().numpy()
        assert matrix.ndim == 2 and np.isfinite(matrix).all()
        view = matrix.argmin(axis=1)
        costs.append(matrix[np.arange(len(matrix)),view]-background.detach().double().cpu().numpy())
        choices.append(view)
    cost = np.stack(costs,axis=1)
    assert cost.shape[0] >= cost.shape[1]
    queries, parents = linear_sum_assignment(cost)
    return [(int(q),int(p),int(choices[p][q])) for q,p in zip(queries,parents)]


def selected_objective(positive_by_parent, background, pairs):
    assert len(pairs) == len(positive_by_parent)
    assert len({q for q,p,v in pairs}) == len(pairs)
    assert {p for q,p,v in pairs} == set(range(len(positive_by_parent)))
    return background.sum()+sum((positive_by_parent[p][q,v]-background[q] for q,p,v in pairs),background.sum()*0.)


def conditioned_nodes(output, candidate, view, events, cache):
    n = int(view['original_child_count'])
    gaps = tuple(s == 'POSITIVE_GAP' for s in view['canonical_gap_status'])
    target = np.asarray(view['canonical_positions_mm'])
    domain = output['allocation_domain_mm'][candidate]
    margins = (bool(target[0,0] > float(domain[0])), bool(target[-1,1] < float(domain[1])))
    key = (candidate,n,gaps,margins)
    if key not in cache:
        batch = len(output['presence_logits'])
        part = {name:value[candidate:candidate+1] for name,value in output.items()
            if isinstance(value,torch.Tensor) and value.ndim and len(value)==batch}
        q = part['presence_logits'].shape[1]
        gap_logits = torch.full_like(part['gap_logits'],-2.)
        if n > 1:
            gap_logits[:,:,:n-1] = gap_logits.new_tensor([2. if value else -2. for value in gaps])
        part['gap_logits'] = gap_logits
        part['margin_increment_logits'] = torch.zeros_like(part['margin_increment_logits'])
        requested = torch.full_like(part['frozen325_counts'],n)
        prior_margin = margin_logits_for_counts(part,requested)
        desired_margin = prior_margin.new_tensor([2. if v else -2. for v in margins]).reshape(1,1,2).expand(1,q,2)
        part['margin_increment_logits'] = desired_margin-prior_margin
        cache[key] = events.positions_for_requested_counts(part,n)[0,:,:n]
    return cache[key], margins


def pair_components(output, candidate, view, events, cache, G, keep_G):
    presence = output['presence_logits'][candidate]
    count_logits = output['count_logits'][candidate]
    q = len(presence); n = int(view['original_child_count'])
    nodes,margins = conditioned_nodes(output,candidate,view,events,cache)
    got = world_positions(nodes,view['candidate_geometry'])
    wanted = got.new_tensor(view['resolved_original_world_endpoints_mm'])
    error = torch.linalg.vector_norm(got-wanted[None],dim=-1)
    weight = n/max(1,G)
    components = {
        'positive_presence': F.softplus(-presence).double()*weight,
        'count': F.cross_entropy(count_logits,torch.full((q,),n,device=presence.device,dtype=torch.long),reduction='none').double()*weight,
        'geometry': 2.*F.smooth_l1_loss(error/3.,torch.zeros_like(error),reduction='none',beta=1.).mean((-1,-2))*weight,
        'maximum': child_endpoint_risk(error, G),
    }
    relation = F.binary_cross_entropy_with_logits(output['closed_logits'][candidate],
        presence.new_full((q,),float(view['parent_is_closed'])),reduction='none').double()
    if n>1:
        gaps = presence.new_tensor([s=='POSITIVE_GAP' for s in view['canonical_gap_status']]).expand(q,-1)
        relation = relation+F.binary_cross_entropy_with_logits(output['gap_logits'][candidate,:,:n-1],gaps,reduction='none').double().mean(-1)
    if view.get('direction_semantics_confirmed') is True:
        relation = relation+F.binary_cross_entropy_with_logits(output['direction_logits'][candidate],
            presence.new_full((q,),float(not view['negative_output_chart'])),reduction='none').double()
    requested = torch.full_like(output['frozen325_counts'][candidate:candidate+1],n)
    batch = len(output['presence_logits'])
    part = {name:value[candidate:candidate+1] for name,value in output.items()
        if isinstance(value,torch.Tensor) and value.ndim and len(value)==batch}
    margin_logits = margin_logits_for_counts(part,requested)[0]
    relation = relation+F.binary_cross_entropy_with_logits(margin_logits,margin_logits.new_tensor(margins).expand(q,-1),reduction='none').mean(-1)
    components['relation'] = relation*weight
    domain = output['allocation_domain_mm'][candidate]; span = domain[1]-domain[0]
    physical = F.relu(domain[0]-nodes).mean((-1,-2))+F.relu(nodes-domain[1]).mean((-1,-2))
    physical = physical+F.relu(nodes[:,:,0]-nodes[:,:,1]).mean(-1)
    if n>1:
        physical = physical+F.relu(nodes[:,:-1,1]-nodes[:,1:,0]).mean(-1)
    components['physical'] = .1*physical/span*weight
    keep = presence.double()*0.
    mask = np.asarray(view.get('protected_keep_child_mask',np.zeros(n,bool)),bool)
    replaced = 0
    if mask.any():
        keep_mask = torch.as_tensor(mask,device=got.device)
        reference = got.new_tensor(view['protected_keep_world_endpoints_mm'])
        assert bool(torch.isfinite(reference[keep_mask]).all())
        compatible = torch.linalg.vector_norm(reference-wanted,dim=-1).amax(-1)<=3.
        replaced = int((keep_mask&~compatible).sum())
        # A conflicting teacher never replaces the original truth.
        reference = torch.where(compatible[:,None,None],reference,wanted)
        keep_error = torch.linalg.vector_norm(got[:,keep_mask]-reference[None,keep_mask],dim=-1)
        keep = F.smooth_l1_loss(keep_error/3.,torch.zeros_like(keep_error),reduction='none',beta=1.).mean((-1,-2))
        competitors = torch.cat((count_logits[:,:n],count_logits[:,n+1:]),-1)
        keep = keep+F.relu(presence.double().new_tensor(math.log(4.))-presence.double())
        keep = keep+F.relu(2.-(count_logits[:,n]-competitors.max(-1).values)).double()
        keep = keep*SOFT_KEEP_WEIGHT*int(mask.sum())/max(1,keep_G)
    components['soft_keep'] = keep
    total = sum(components.values(),presence.double()*0.)
    assert bool(torch.isfinite(total).all())
    return total,components,dict(teacher_conflicts_replaced_by_original_GT=replaced)


def event_loss(output,parents_by_candidate,base,background_known,group_ids,
               hard_negative_weights,original_G,original_G_by_group,*,events,transaction_plan=None):
    presence,counts = output['presence_logits'],output['count_logits']
    b,q = presence.shape
    assert len(parents_by_candidate)==b and sum(original_G_by_group.values())==original_G>0
    assert bool(torch.all(hard_negative_weights==1.))
    assert background_known.shape==(b,)
    keep_counts = {g:sum(int(np.asarray(r.get('protected_keep_child_mask',[]),bool).sum())
        for i,rows in enumerate(parents_by_candidate) if int(group_ids[i])==g for r in rows) for g in (0,1)}
    domains = tuple((i,tuple((r['original_parent_index'],tuple(r['original_child_ids'])) for r in rows))
        for i,rows in enumerate(parents_by_candidate) if int(group_ids[i]) in (0,1) and rows)
    if transaction_plan is not None:
        assert transaction_plan['domains']==domains and transaction_plan['original_G']==original_G
    plan = dict(domains=domains,original_G=original_G,pairs={})
    zero = presence.double().sum()*0.
    group_loss = {0:zero,1:zero}; group_terms = {0:{},1:{}}
    assignments,cache,ignored_anchors = [],{},0
    expected_count = (counts.softmax(-1)*counts.new_tensor([max(1,k) for k in range(counts.shape[-1])])).sum(-1)
    for i,records in enumerate(parents_by_candidate):
        group = int(group_ids[i])
        if group==2:
            continue
        assert group in (0,1) and len(records)<=q
        G = int(original_G_by_group[group])
        background = F.softplus(presence[i]).double()*expected_count[i].double()/max(1,G)
        if not bool(background_known[i]):
            background = background*0.
        options = [list(resolved_views(record)) for record in records]
        all_positive,all_components = [],[]
        for views in options:
            assert views
            values = [pair_components(output,i,view,events,cache,G,keep_counts[group]) for view in views]
            all_positive.append(torch.stack([row[0] for row in values],dim=1))
            all_components.append(values)
        if transaction_plan is None:
            pairs = minimum_assignment(all_positive,background)
        else:
            pairs = transaction_plan['pairs'].get(i,[])
        plan['pairs'][i] = pairs
        group_loss[group] = group_loss[group]+selected_objective(all_positive,background,pairs)
        selected_queries = {row[0] for row in pairs}
        negative = sum((background[j] for j in range(q) if j not in selected_queries),zero)
        group_terms[group]['background'] = group_terms[group].get('background',zero)+negative
        for query,parent,view_index in pairs:
            view = options[parent][view_index]
            values,components,facts = all_components[parent][view_index]
            for name,term in components.items():
                group_terms[group][name] = group_terms[group].get(name,zero)+term[query]
            assignments.append(dict(candidate=i,predicted_parent_query=query,
                original_parent_index=view['original_parent_index'],original_child_count=view['original_child_count'],
                canonical_original_child_ids=view['canonical_original_child_ids'],resolved_view_index=view['resolved_view_index'],
                canonical_permutation=view['canonical_permutation'],negative_output_chart=view['negative_output_chart'],
                canonical_gap_status=view['canonical_gap_status'],reserved_train_anchor=False,**facts))
        ignored_anchors += sum(record.get('fixed_train_query') is not None for record in records)
    for group,branch in ((0,'H'),(1,'V')):
        parameters = list(getattr(events,branch).parameters())
        regularization = .001*sum((p.double().square().sum() for p in parameters),zero)/sum(p.numel() for p in parameters)
        group_loss[group] = group_loss[group]+regularization
        group_terms[group]['regularization'] = regularization
    total = (group_loss[0]+group_loss[1])/2.
    assert bool(torch.isfinite(total))
    return total,dict(version=VERSION,loss_by_group=group_loss,group_components=group_terms,
        transaction_plan_TRAIN_only=plan,assignments=assignments,
        original_G=original_G,original_G_by_group=original_G_by_group,
        geometry_supervised_children=sum(r['original_child_count'] for r in assignments),
        old_query_anchors_ignored=ignored_anchors,reserved_train_anchor_count=0,
        matching_and_backward_use_same_full_pair_minus_background=True,
        background_expected_count_has_gradient=True,geometry_conditioned_on_GT_count_gap_margin=True,
        regularization_is_assignment_and_argmax_independent=True,soft_keep_weight=SOFT_KEEP_WEIGHT,
        free_inference_or_contract_acceptance_not_measured=True)
