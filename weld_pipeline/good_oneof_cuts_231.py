"""Good-only valid cuts, proved against original hard rows and product rows.

This returns additional rows. It does not change objectives, candidates or the
original constraints, and is not installed in the live recovery solver.
"""
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.sparse import coo_matrix

VERSION='ORIGINAL_ROW_PROVED_GOOD_ONEOF_PRODUCT_CUTS_231_v1'


@dataclass(frozen=True)
class ProvedCuts:
    constraint: LinearConstraint
    fixed_zero_product_columns: tuple
    audit: dict


def _columns(value,count,label):
    values=tuple(value)
    if any(isinstance(item,(bool,np.bool_)) or not isinstance(item,(int,np.integer)) for item in values):
        raise ValueError('Explicit integer columns required: '+label)
    values=tuple(int(item) for item in values)
    if len(set(values))!=len(values) or any(item<0 or item>=count for item in values):
        raise ValueError('Repeated or out-of-range columns: '+label)
    return values


def prove_good_oneof_cuts(*,mode,constraints,integrality,bounds,node_columns,groups,products):
    """Every cut uses an actual original one-of row and McCormick premises.

    products contains (first_z_column, second_z_column, t_column), with canonical
    first < second. Missing products are never imputed. A group lacking a hard
    sum(z)=1 row produces no cuts. Invalid definitions for a used product fail.
    """
    if mode!='good':
        raise ValueError('Training Good mode required; Runtime/Bad cannot use these groups')
    A=constraints.A.tocsr(copy=True)
    A.sum_duplicates();A.eliminate_zeros();A.sort_indices()
    rows,count=A.shape
    lower=np.broadcast_to(np.asarray(constraints.lb,dtype=float),(rows,))
    upper=np.broadcast_to(np.asarray(constraints.ub,dtype=float),(rows,))
    integer=np.asarray(integrality)
    lo=np.broadcast_to(np.asarray(bounds.lb,dtype=float),(count,))
    hi=np.broadcast_to(np.asarray(bounds.ub,dtype=float),(count,))
    if integer.shape!=(count,) or np.isnan(A.data).any() or np.isnan(lower).any() or np.isnan(upper).any():
        raise ValueError('Complete finite-coefficient original matrix and integrality required')
    if not np.isfinite(A.data).all() or np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo>hi):
        raise ValueError('Invalid original coefficients or bounds')
    nodes=tuple(sorted(_columns(node_columns,count,'node_columns')))
    node_set=set(nodes)
    if any(integer[column]!=1 or lo[column]<0 or hi[column]>1 for column in nodes):
        raise ValueError('Original binary node columns in [0,1] required')
    groups=tuple(tuple(sorted(_columns(group,count,'one_of_group'))) for group in groups)
    if any(not group or not set(group)<=node_set for group in groups) or len(set(groups))!=len(groups):
        raise ValueError('Nonempty distinct groups of original binary node columns required')
    product_columns={}
    used_columns=set()
    for value in products:
        value=tuple(value)
        if len(value)!=3:
            raise ValueError('Canonical product triple required')
        first,second,column=_columns(value,count,'product_definition')
        pair=(first,second)
        if (first>=second or first not in node_set or second not in node_set or column in node_set
                or pair in product_columns or column in used_columns or lo[column]<0 or hi[column]>1):
            raise ValueError('Noncanonical, repeated or invalid original product definition')
        product_columns[pair]=column;used_columns.add(column)
    C=A.tocsc()

    def key(row):
        begin,end=A.indptr[row:row+2]
        return tuple(A.indices[begin:end].tolist()),tuple(A.data[begin:end].tolist())

    def candidate_rows(column,size):
        begin,end=C.indptr[column:column+2]
        return (int(row) for row in C.indices[begin:end] if A.indptr[row+1]-A.indptr[row]==size)

    def hard_one_of(group):
        for row in candidate_rows(group[0],len(group)):
            cols,values=key(row)
            if cols==group:
                if values==(1.,)*len(group) and lower[row]==upper[row]==1.:
                    return row
                if values==(-1.,)*len(group) and lower[row]==upper[row]==-1.:
                    return row
        return None

    verified={}
    def proved_product(first,second):
        pair=(min(first,second),max(first,second))
        column=product_columns[pair]
        if pair in verified:return column
        expected=[({column:1.,pair[0]:-1.},'upper',0.),
                  ({column:1.,pair[1]:-1.},'upper',0.),
                  ({column:1.,pair[0]:-1.,pair[1]:-1.},'lower',-1.)]
        matches=[]
        for values,side,rhs in expected:
            cols=tuple(sorted(values));coeff=tuple(values[item] for item in cols)
            found=None
            for row in candidate_rows(column,len(cols)):
                actual_cols,actual_coeff=key(row)
                if actual_cols!=cols:continue
                if actual_coeff==coeff and (upper[row] if side=='upper' else lower[row])==rhs:
                    found=row;break
                if actual_coeff==tuple(-value for value in coeff) and (lower[row] if side=='upper' else upper[row])==-rhs:
                    found=row;break
            if found is None:
                raise ValueError('Original McCormick premise missing for product '+str(pair))
            matches.append(found)
        verified[pair]=tuple(matches)
        return column

    extra=[];zero_columns={};audit_groups=[]
    seen_external=set()
    for number,group in enumerate(groups):
        original_row=hard_one_of(group)
        record=dict(group=number,members=list(group),original_hard_row=original_row,
                    group_zero_products=0,external_complete_rows=0,external_missing_product_nodes=[])
        if original_row is None:
            record['status']='SKIPPED_NO_ORIGINAL_HARD_ONE_OF_EQUALITY'
            audit_groups.append(record);continue
        record['status']='PROVED_FROM_ORIGINAL_HARD_EQUALITY'
        for first,second in itertools.combinations(group,2):
            pair=(first,second)
            if pair not in product_columns:continue
            column=proved_product(*pair)
            record['group_zero_products']+=1
            if column not in zero_columns:
                zero_columns[column]=dict(group=number,original_hard_row=original_row)
                extra.append(((column,),(1.,),0.,dict(kind='INTRA_GROUP_PRODUCT_ZERO',group=number)))
        for other in nodes:
            if other in group:continue
            pairs=tuple((min(member,other),max(member,other)) for member in group)
            if any(pair not in product_columns for pair in pairs):
                record['external_missing_product_nodes'].append(other)
                continue
            columns=tuple(proved_product(*pair) for pair in pairs)
            values={column:1. for column in columns};values[other]=-1.
            cols=tuple(sorted(values));coeff=tuple(values[column] for column in cols)
            if (cols,coeff) not in seen_external:
                seen_external.add((cols,coeff))
                extra.append((cols,coeff,0.,dict(kind='COMPLETE_GROUP_EXTERNAL_PRODUCT_SUM',group=number,node=other)))
                record['external_complete_rows']+=1
        audit_groups.append(record)
    rr=[];cc=[];vv=[]
    for row,(cols,values,_,_) in enumerate(extra):
        rr.extend([row]*len(cols));cc.extend(cols);vv.extend(values)
    matrix=coo_matrix((np.asarray(vv,float),(np.asarray(rr,np.int64),np.asarray(cc,np.int64))),
                      shape=(len(extra),count)).tocsr()
    rhs=np.asarray([value[2] for value in extra],float)
    digest=sha256()
    for values in (A.indptr,A.indices,A.data,lower,upper,integer,lo,hi):
        array=np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    premise=dict(groups=audit_groups,verified_products=[dict(pair=list(pair),column=product_columns[pair],
        original_McCormick_rows=list(proof)) for pair,proof in sorted(verified.items())],
        rows=[value[3] for value in extra])
    audit=dict(version=VERSION,mode=mode,original_matrix_premise_sha256=digest.hexdigest(),
        premise_scope='CANONICAL_CSR_ROWS_BOUNDS_AND_INTEGRALITY_NOT_225_NATIVE_HASH',
        original_rows=rows,original_columns=count,added_rows=len(extra),added_nonzeros=int(matrix.nnz),
        fixed_zero_product_count=len(zero_columns),verified_product_count=len(verified),
        groups=audit_groups,premises_sha256=sha256(json.dumps(premise,sort_keys=True).encode()).hexdigest(),
        original_candidates_and_objective_unmodified=True,unavailable_products_never_imputed=True,
        integer_validity='BINARY_ONE_OF_TIMES_BINARY_NODE_WITH_ORIGINAL_PRODUCT_PREMISES',
        training_release=False,installed_in_live_227=False,full_contract_pass=False)
    return ProvedCuts(LinearConstraint(matrix,rhs,rhs),tuple(sorted(zero_columns)),audit)
