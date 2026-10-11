"""Label-free finite arclength proposals, distinct from observed surface events.

This supplies representation coverage only. It neither establishes contact nor
predicts welds. Consumers must retain is_grid provenance and train appropriately.
"""
import numpy as np

def finite_carrier_grid(length_mm, original_positions_mm, spacing_mm=1., max_grid_events=1_000_001):
 length=float(length_mm);spacing=float(spacing_mm);old=np.asarray(original_positions_mm,dtype=np.float64)
 if not np.isfinite(length) or length<=0 or not np.isfinite(spacing) or spacing<=0:raise ValueError('Positive finite length and spacing required')
 if old.ndim!=1 or not np.isfinite(old).all() or np.any(old<0) or np.any(old>length):raise ValueError('Original events must lie on finite carrier')
 intervals=int(np.ceil(length/spacing))
 if intervals+1>max_grid_events:raise ValueError('Grid exceeds fixed resource budget; do not silently truncate')
 grid=np.linspace(0.,length,intervals+1,dtype=np.float64)
 positions=np.unique(np.concatenate((old,grid)))
 original_ids=np.searchsorted(positions,old);grid_ids=np.searchsorted(positions,grid)
 observed=np.zeros(len(positions),bool);observed[original_ids]=True
 is_grid=np.zeros(len(positions),bool);is_grid[grid_ids]=True
 assert np.array_equal(positions[original_ids],old)
 return dict(positions_mm=positions,original_to_merged=original_ids,is_existing_event=observed,is_grid=is_grid,grid_ids=grid_ids,grid_spacing_mm=length/intervals)
