"""Same167 numerical network with the original prediction function's complete globals."""
import types
from . import curveonly_01a1ac7c_normal_spans_v274 as previous

NAMES,OBSERVATIONS,CHANNELS=previous.NAMES,previous.OBSERVATIONS,previous.CHANNELS
PERMUTATION,codec,base=previous.PERMUTATION,previous.codec,previous.base
normalize,arclength,sequence=previous.normalize,previous.arclength,previous.sequence
normals=previous.normals
candidate_rows,input_curves,pack=previous.candidate_rows,previous.input_curves,previous.pack
OrderedSpanNetwork=previous.OrderedSpanNetwork
decode=previous.decode
core=previous.previous.previous
# V247's predict is already a cloned V235 function. Its module dictionary
# does not contain V235's reverse_ranges / intervals_to_codes helper names.
predict=types.FunctionType(previous.previous.predict.__code__,
    dict(previous.previous.predict.__globals__,pack=pack))


class OrderedSpanSet(core.OrderedSpanSet):
    def predict(self,features,packet):
        return types.FunctionType(core.OrderedSpanSet.predict.__code__,
            dict(core.__dict__,candidate_rows=candidate_rows,predict=predict))(self,features,packet)
