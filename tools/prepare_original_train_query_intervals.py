"""Reuse the same positive-query protocol on original registered TRAIN families."""
from pathlib import Path
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
import tools.diagnose_expansion_carrier_coverage as coverage
import tools.prepare_expansion_positive_intervals as intervals
import tools.prepare_positive_query_interval_training as queries
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'server_rebuild/expansion_v1/original_train_query_inputs_v1';KEYS=('21','BM79A_native_local','10','25','36')
def main():
 DEST.mkdir(exist_ok=False)
 for name,path in [('frozen_features','server_rebuild/best_baseline_restore'),('inputs','server_rebuild/inputs_v1'),('labels','server_rebuild/labels_v1')]:
  (DEST/name).symlink_to(ROOT/path,target_is_directory=True)
 write(DEST/'coverage_cohort_protocol.json',dict(cases=[dict(key=k,role='ORIGINAL_REGISTERED_TRAIN',labels_receipt_sha256=digest(DEST/'labels'/k/'receipt.json')) for k in KEYS],source='Original restored five TRAIN roles; does not admit missing4/42 or DEV64',fit_calls=0,DEV_reads=0))
 coverage.BASE=DEST;intervals.BASE=DEST;queries.BASE=DEST
 for key in KEYS:
  (DEST/'native_capacity'/('source_'+key)).mkdir(parents=True)
  coverage.run(key);intervals.run(key);queries.run(key);print('ORIGINAL_POSITIVE_QUERY_READY',key,flush=True)
 write(DEST/'receipt.json',dict(status='ALL_FIVE_ORIGINAL_TRAIN_POSITIVE_QUERY_TABLES_COMPLETE',keys=KEYS,query_data_root=str(queries.DEST.relative_to(ROOT)),fit_calls=0,DEV_reads=0,source_sha256=digest(__file__)))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
