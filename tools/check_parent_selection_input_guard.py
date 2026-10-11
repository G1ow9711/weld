"""Replay input-only selection with Python reads of targets/supervision denied."""
import builtins,io,os
from pathlib import Path
import tools.infer_unified_parent_competition_support as runtime
from weld_pipeline.sjg_local_runtime_904 import write,digest
BASE=runtime.BASE
real_open=builtins.open;real_io=io.open;opened=[]
def guarded(original):
 def call(file,mode='r',*args,**kwargs):
  if isinstance(file,(str,bytes,os.PathLike)):
   path=os.fsdecode(file)
   if 'r' in mode:
    denied=('/labels_v1/' in path or 'targets.json' in path or 'd3_supervision' in path or 'child_ownership' in path or 'evaluation.json' in path or 'compatibility.json' in path)
    if denied:raise AssertionError('Forbidden inference read: '+path)
    opened.append(path)
  return original(file,mode,*args,**kwargs)
 return call
if __name__=='__main__':
 builtins.open=guarded(real_open);io.open=guarded(real_io)
 try:runtime.run('25','input_guard_control_v2')
 finally:builtins.open=real_open;io.open=real_io
 actual=BASE/'input_guard_control_v2/source_25/prediction.json';expected=BASE/'set_competition_support_v2/source_25/prediction.json';same=digest(actual)==digest(expected)
 write(BASE/'input_guard_control_v2/control.json',dict(PASS=same,prediction_byte_identity=same,actual_sha256=digest(actual),reference_sha256=digest(expected),read_paths=sorted(set(opened)),scope='Python builtins.open/io.open access guard and single-case frozen input replay; not native OS read isolation or full XYZ replay',no_denied_reads=True))
 assert same,'Guarded replay changed output'
 print('INPUT_GUARD_PASS',flush=True)
