from pathlib import Path
import json,random,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tokenforge.incremental_cli import audit
from tokenforge.pretokenize import pretokenize

def build():
    inputs=[('overlap-five',[[97]*5],260,1),('overlap-odd-rows',[[97]*9,[97]*7,[]],270,1),('lexicographic-tie',[[98,97,98,97],[97,98,97,98]],262,1),('separate-singletons',[[97],[97],[],[98]],270,1),('min-count-stop',[[97,98,99]],270,2),('cap-zero',[[97]*12],256,1),('shared-pair-neighbors',[[97,98,97,98,97,99]*3],270,1),('byte-extremes',[[0,255,0,255,0,255],[0,0,255]],270,1),('empty-rows',[[],[]],260,1),('utf8-byte-rows',[list(p.encode())for p in pretokenize('Hi 👋 café سڵاو 中文 123!!')],280,1),('original-first-twelve',[list(p.encode())for p in pretokenize((ROOT/'examples/data/corpus.txt').read_text())[:12]],280,1)]
    rng=random.Random(907)
    for i in range(16):inputs.append((f'seeded-{i}',[[rng.randrange(6)for _ in range(rng.randrange(1,20))]for _ in range(rng.randrange(1,6))],rng.randrange(257,280),rng.randrange(1,4)))
    return [{'name':name,'input':{'rows':rows,'vocab_size':cap,'min_pair_count':minimum},'report':audit({'rows':rows,'vocab_size':cap,'min_pair_count':minimum})}for name,rows,cap,minimum in inputs]
if __name__=='__main__':
    from tokenforge import train_incremental
    text=(ROOT/'examples/data/corpus.txt').read_text();tokenizer,report=train_incremental(text,300,2,trace=True)
    summary={'source':'examples/data/corpus.txt','utf8_bytes':len(text.encode()),'vocab_size':300,'min_pair_count':2,'merges':report['merges'],'final_rows':report['rows'],'work':report['work']}
    (ROOT/'examples/indexed-corpus.json').write_text(json.dumps(summary,indent=2)+'\n')
    ws=build();(ROOT/'examples/indexed.json').write_text(json.dumps(ws,separators=(',',':'))+'\n');(ROOT/'docs/indexed-data.js').write_text('globalThis.INDEXED_INPUTS='+json.dumps([{'name':w['name'],'input':w['input']}for w in ws],separators=(',',':'))+';\n');print('reports',len(ws),'merge steps',sum(len(w['report']['semantic']['steps'])for w in ws))
