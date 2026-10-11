"""Stream parent arrays and intern identical points without dropping outputs."""
import hashlib,json
import numpy as np


def read_shared_predictions(path):
    cache={}
    def hook(value):
        if 'points' in value and 'kind' in value:
            points=np.asarray(value['points'],np.float64)
            signature=(points.shape,hashlib.sha256(points.tobytes()).digest())
            if signature not in cache:points.setflags(write=False);cache[signature]=points
            else:assert np.array_equal(cache[signature],points)
            value['points']=cache[signature]
        return value
    decoder=json.JSONDecoder(object_hook=hook)
    with path.open('r',encoding='utf-8-sig') as stream:
        buffer='';pos=0;eof=False
        def refill():
            nonlocal buffer,pos,eof
            chunk=stream.read(1024*1024);buffer=buffer[pos:]+chunk;pos=0;eof=not chunk
        def whitespace():
            nonlocal pos
            while True:
                while pos<len(buffer) and buffer[pos].isspace():pos+=1
                if pos<len(buffer) or eof:return
                refill()
        def token(expected):
            nonlocal pos
            whitespace();assert buffer[pos:pos+1]==expected;pos+=1
        def value():
            nonlocal pos
            whitespace()
            while True:
                try:
                    v,end=decoder.raw_decode(buffer,pos)
                    if end==len(buffer) and not eof:refill();continue
                    pos=end;return v
                except json.JSONDecodeError:
                    if eof:raise
                    refill()
        token('{');result={};first=True
        while True:
            whitespace()
            if buffer[pos:pos+1]=='}':pos+=1;break
            if not first:token(',')
            key=value();token(':')
            if key=='parents':
                token('[');parents=[];first_parent=True
                while True:
                    whitespace()
                    if buffer[pos:pos+1]==']':pos+=1;break
                    if not first_parent:token(',')
                    parents.append(value());first_parent=False
                    if pos>1024*1024:buffer=buffer[pos:];pos=0
                result[key]=parents
            else:result[key]=value()
            first=False
        whitespace();assert pos==len(buffer) and eof
    return result
