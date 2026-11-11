"""Occurrence-indexed BPE training. Original scan trainer remains the default."""
import heapq
from .tokenizer import BPETokenizer


def integer(value,low,high=None):
    if type(value)is not int or value<low or(high is not None and value>high):
        raise ValueError('integer outside supported bounds')
    return value


def train_rows(rows,vocab_size=512,min_pair_count=2,*,trace=False):
    """Train from byte rows without crossing row boundaries.

    Count overlapping adjacent occurrences, choose (-count,a,b), then replace
    leftmost nonoverlapping matches. Updates touch only edges incident to a
    replacement. Trace copies all live rows/counts and is not a timing claim.
    """
    integer(vocab_size,256);integer(min_pair_count,1)
    rows=[list(row)for row in rows]
    for row in rows:
        for value in row:integer(value,0,255)
    values=[];previous=[];following=[];alive=[];starts=[]
    for row in rows:
        start=len(values);starts.append(start if row else -1)
        for j,value in enumerate(row):
            i=len(values);values.append(value);previous.append(i-1 if j else -1)
            following.append(i+1 if j+1<len(row)else -1);alive.append(True)
    occurrences={};heap=[]
    work={'initial_edges':0,'edge_removals':0,'edge_insertions':0,'posting_positions_visited':0,'pair_replacements':0,'heap_pushes':0,'heap_pops':0,'stale_heap_pops':0,'scan_reference_adjacencies':0,'trace_pair_entries':0,'trace_token_entries':0}
    def edge(i):
        return (values[i],values[following[i]])if i>=0 and alive[i] and following[i]>=0 else None
    def remove(i,changed):
        pair=edge(i)
        if pair is not None:
            occurrences[pair].remove(i);work['edge_removals']+=1;changed.add(pair)
    def insert(i,changed,initial=False):
        pair=edge(i)
        if pair is not None:
            occurrences.setdefault(pair,set()).add(i);work['initial_edges' if initial else 'edge_insertions']+=1;changed.add(pair)
    def push(pair):
        count=len(occurrences[pair])
        if count:heapq.heappush(heap,(-count,*pair));work['heap_pushes']+=1
        else:del occurrences[pair]
    def current_rows():
        result=[]
        for i in starts:
            row=[]
            while i>=0:row.append(values[i]);i=following[i]
            result.append(row)
        return result
    changed=set()
    for i in range(len(values)):insert(i,changed,True)
    for pair in sorted(changed):push(pair)
    history=[];merges=[];tokens=sum(map(len,rows));nonempty=sum(bool(r)for r in rows)
    reason='vocab-cap'
    while 256+len(merges)<vocab_size:
        best=None
        while heap:
            neg,a,b=heapq.heappop(heap);work['heap_pops']+=1;pair=(a,b)
            if -neg!=len(occurrences.get(pair,())):work['stale_heap_pops']+=1;continue
            best=pair;count=-neg;break
        # Scan baseline must recount even the final no-pair/min-count iteration.
        work['scan_reference_adjacencies']+=tokens-nonempty
        if best is None:reason='no-pairs';break
        if count<min_pair_count:reason='min-count';break
        new_id=256+len(merges);changed=set();positions=sorted(occurrences[best]);before=None;pairs=None
        if trace:
            before=current_rows();pairs=[{'pair':list(p),'count':len(v)}for p,v in sorted(occurrences.items())]
            work['trace_pair_entries']+=len(pairs);work['trace_token_entries']+=tokens
        replaced=[]
        for i in positions:
            work['posting_positions_visited']+=1
            if edge(i)!=best:continue
            j=following[i];left=previous[i];right=following[j]
            remove(left,changed);remove(i,changed);remove(j,changed)
            values[i]=new_id;following[i]=right;alive[j]=False;following[j]=-1
            if right>=0:previous[right]=i
            insert(left,changed);insert(i,changed);replaced.append(i);tokens-=1
        for pair in sorted(changed):push(pair)
        work['pair_replacements']+=len(replaced);merges.append(best)
        if trace:
            after=current_rows();work['trace_token_entries']+=tokens
            history.append({'rank':len(merges)-1,'new_id':new_id,'pair':list(best),'frequency':count,'replacements':len(replaced),'position_ids':replaced,'pair_counts':pairs,'before':before,'after':after})
    return {'merges':[list(p)for p in merges],'rows':current_rows(),'steps':history,'stop_reason':reason,'work':work}


def train_incremental(text,vocab_size=512,min_pair_count=2,pretokenizer=None,*,trace=False):
    from .pretokenize import pretokenize
    split=pretokenizer or pretokenize
    rows=[list(piece.encode('utf-8'))for piece in split(text)if piece]
    report=train_rows(rows,vocab_size,min_pair_count,trace=trace)
    return BPETokenizer([tuple(p)for p in report['merges']]),report
