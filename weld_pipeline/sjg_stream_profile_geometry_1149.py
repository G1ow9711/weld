"""Read only feature-factory geometry from a JSON array without whole-file copies."""
import json


def read_profile_geometry(path,retain_fields=()):
    decoder=json.JSONDecoder();result=[]
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
        whitespace();assert buffer[pos]=='[';pos+=1;first=True
        while True:
            whitespace()
            if pos<len(buffer) and buffer[pos]==']':pos+=1;break
            if not first:
                assert pos<len(buffer) and buffer[pos]==',';pos+=1;whitespace()
            while True:
                try:item,end=decoder.raw_decode(buffer,pos);break
                except json.JSONDecodeError:
                    if eof:raise
                    refill()
            assert isinstance(item,dict) and 'representation' in item
            # These are the only profile fields accessed by the relative feature factories.
            profile=dict(representation=item['representation'],face_indices=item.get('face_indices',[]))
            profile.update({name:item[name] for name in retain_fields if name in item})
            result.append(profile)
            pos=end;first=False
            if pos>1024*1024:buffer=buffer[pos:];pos=0
        whitespace();assert pos==len(buffer) and eof,'Unexpected trailing JSON data'
    return result
