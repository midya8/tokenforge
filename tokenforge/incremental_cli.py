"""Bounded byte-row audit. Rows are already split, not a new regex contract."""
import argparse,json,sys
from .incremental import train_rows,integer


def parse(text):
    if len(text.encode('utf-8'))>131072:raise ValueError('input exceeds128KiB')
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    try:v=json.loads(text,object_pairs_hook=pairs)
    except(json.JSONDecodeError,RecursionError)as e:raise ValueError('invalid JSON')from e
    if type(v)is not dict or set(v)!={'rows','vocab_size','min_pair_count'}:raise ValueError('expected exact rows/vocab_size/min_pair_count keys')
    if type(v['rows'])is not list or len(v['rows'])>64 or any(type(r)is not list for r in v['rows']):raise ValueError('at most64 explicit byte rows required')
    if sum(map(len,v['rows']))>1024:raise ValueError('at most1024 input bytes')
    for row in v['rows']:
        for x in row:integer(x,0,255)
    integer(v['vocab_size'],256,288);integer(v['min_pair_count'],1,1024)
    return v


def audit(v):
    v=parse(json.dumps(v));r=train_rows(**v,trace=True)
    # Heap and indexed edge-work are returned separately from semantic trace.
    return {'schema':'tokenforge-indexed/1','input':v,'semantic':{k:r[k]for k in ['merges','rows','steps','stop_reason']},'indexed_work':r['work']}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args(argv)
    try:
        with open(a.input,encoding='utf-8')as f:v=parse(f.read(131073))
        print(json.dumps(audit(v),indent=2,allow_nan=False));return 0
    except(ValueError,OverflowError,OSError)as e:print('ERROR: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
