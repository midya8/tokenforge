import itertools,json,random,tempfile,unittest
from pathlib import Path
from tokenforge.tokenizer import BPETokenizer
from tokenforge.incremental import train_rows,train_incremental


def oracle(rows,vocab,min_count):
    rows=[r[:]for r in rows];merges=[];steps=[]
    while 256+len(merges)<vocab:
        counts={}
        for row in rows:
            for a,b in zip(row,row[1:]):counts[(a,b)]=counts.get((a,b),0)+1
        if not counts:reason='no-pairs';break
        pair=min(counts,key=lambda p:(-counts[p],p))
        if counts[pair]<min_count:reason='min-count';break
        before=[r[:]for r in rows];out=[];n=0
        for row in rows:
            result=[];i=0
            while i<len(row):
                if tuple(row[i:i+2])==pair:result.append(256+len(merges));i+=2;n+=1
                else:result.append(row[i]);i+=1
            out.append(result)
        rows=out;steps.append({'pair':list(pair),'frequency':counts[pair],'replacements':n,'before':before,'after':rows,'pair_counts':[{'pair':list(p),'count':counts[p]}for p in sorted(counts)]});merges.append(list(pair))
    else:reason='vocab-cap'
    return rows,merges,steps,reason


class IncrementalTests(unittest.TestCase):
    def compare(self,rows,vocab,min_count):
        original=[x[:]for x in rows];a=train_rows(rows,vocab,min_count,trace=True);out,merges,steps,reason=oracle(rows,vocab,min_count)
        self.assertEqual(rows,original);self.assertEqual(a['rows'],out);self.assertEqual(a['merges'],merges);self.assertEqual(a['stop_reason'],reason)
        for got,expected in zip(a['steps'],steps):
            for k,v in expected.items():self.assertEqual(got[k],v,k)
        self.assertEqual(len(a['steps']),len(steps));self.assertEqual(a['work']['pair_replacements'],sum(s['replacements']for s in steps));self.assertEqual(sum(map(len,original))-sum(map(len,out)),a['work']['pair_replacements'])
    def test_complete_binary_rows_and_caps(self):
        n=0
        for length in range(9):
            for xs in itertools.product((97,98),repeat=length):
                for cap,min_count in [(256,1),(257,1),(270,1),(270,2),(260,3)]:self.compare([list(xs)],cap,min_count);n+=1
        print('independent exhaustive row/cap/count comparisons',n)
    def test_random_multirow_unequal_lengths_ties(self):
        rng=random.Random(281)
        for _ in range(250):self.compare([[rng.randrange(6)for _ in range(rng.randrange(35))]for _ in range(rng.randrange(9))],rng.randrange(256,285),rng.randrange(1,6))
    def test_overlapping_runs_and_boundaries(self):
        for n in range(1,40):self.compare([[97]*n,[],[97,98,97]*n],290,1)
        r=train_rows([[97]*5],257,1,trace=True);self.assertEqual(r['steps'][0]['frequency'],4);self.assertEqual(r['steps'][0]['replacements'],2);self.assertEqual(r['steps'][0]['position_ids'],[0,2]);self.assertEqual(r['rows'],[[256,256,97]])
        self.assertEqual(train_rows([[97],[97]],260,1)['merges'],[])
    def test_original_text_training_artifacts_and_roundtrips(self):
        texts=['','a'*21,'aaaa bbbb aaaa bbbb','the tokenizer decides what a model sees. '*8,'Hi 👋 café\nسڵاو 中文 123!!','a\t\nb',Path(__file__).resolve().parents[1].joinpath('examples/data/corpus.txt').read_text()]
        for text in texts:
            for cap in [256,257,270,300]:
                a,r=train_incremental(text,cap,1);b=BPETokenizer.train(text,cap,1);self.assertEqual(a.merges,b.merges);self.assertEqual(a.encode(text),b.encode(text));self.assertEqual(a.decode(a.encode(text)),text)
                for i in range(a.vocab_size):self.assertEqual(a.token_bytes(i),b.token_bytes(i))
                with tempfile.TemporaryDirectory()as t:
                    x=Path(t)/'a.json';y=Path(t)/'b.json';a.save(x);b.save(y);self.assertEqual(x.read_bytes(),y.read_bytes())
    def test_no_trace_same_engine_and_detached_outputs(self):
        rows=[[1,2]*20];a=train_rows(rows,280,1);b=train_rows(rows,280,1,trace=True);self.assertEqual(a['merges'],b['merges']);self.assertEqual(a['rows'],b['rows']);self.assertEqual(a['steps'],[]);b['rows'][0].clear();self.assertEqual(rows,[[1,2]*20])
    def test_validation(self):
        for rows,cap,count in [([[True]],260,1),([[-1]],260,1),([[256]],260,1),([[1]],True,1),([[1]],255,1),([[1]],260,True),([[1]],260,0)]:
            with self.assertRaises(ValueError):train_rows(rows,cap,count)

    def test_all_full_reports_and_work_conservation(self):
        from tokenforge.incremental_cli import audit,parse
        root=Path(__file__).resolve().parents[1]
        for w in json.loads((root/'examples/indexed.json').read_text()):
            r=audit(w['input']);self.assertEqual(r,w['report']);self.compare(w['input']['rows'],w['input']['vocab_size'],w['input']['min_pair_count']);work=r['indexed_work'];final_edges=sum(max(0,len(x)-1)for x in r['semantic']['rows']);self.assertEqual(work['initial_edges']+work['edge_insertions']-work['edge_removals'],final_edges);self.assertLessEqual(work['heap_pops'],work['heap_pushes']);self.assertEqual(work['heap_pops']-work['stale_heap_pops'],len(r['semantic']['merges'])+(r['semantic']['stop_reason']=='min-count'))
        for text in ['{"rows":[],"rows":[]}','['*1000,' '.ljust(131073)]:
            with self.assertRaises(ValueError):parse(text)
        v={'rows':[[97]*5],'vocab_size':260,'min_pair_count':1}
        for change in [{'rows':[[True]]},{'rows':[[256]]},{'rows':[[]]*65},{'rows':[[1]*1025]},{'vocab_size':289},{'extra':1}]:
            with self.assertRaises(ValueError):audit({**v,**change})
    def test_train_and_bounded_cli(self):
        import subprocess,sys
        from tokenforge.incremental_cli import audit
        with tempfile.TemporaryDirectory()as t:
            t=Path(t);source=t/'text.txt';source.write_text('aaaa aaaa 👋 café');a=t/'a.json';b=t/'b.json';report=t/'r.json'
            subprocess.run([sys.executable,'-m','tokenforge','train','--input',str(source),'--output',str(a),'--vocab-size','280'],check=True,capture_output=True)
            subprocess.run([sys.executable,'-m','tokenforge','train','--input',str(source),'--output',str(b),'--vocab-size','280','--trainer','indexed','--training-report',str(report)],check=True,capture_output=True)
            self.assertEqual(a.read_bytes(),b.read_bytes());self.assertTrue(json.loads(report.read_text())['steps'])
            v={'rows':[[97]*5],'vocab_size':260,'min_pair_count':1};p=t/'input.json';p.write_text(json.dumps(v));r=subprocess.run([sys.executable,'-m','tokenforge.incremental_cli','--input',str(p)],capture_output=True,text=True,check=True);self.assertEqual(json.loads(r.stdout),audit(v));p.write_text('{}');r=subprocess.run([sys.executable,'-m','tokenforge.incremental_cli','--input',str(p)],capture_output=True,text=True);self.assertEqual(r.returncode,2)

    def test_original_corpus_scan_count_and_every_merge(self):
        import tokenforge.tokenizer as original
        from tokenforge.pretokenize import pretokenize
        text=Path(__file__).resolve().parents[1].joinpath('examples/data/corpus.txt').read_text();calls=[];function=original._pair_counts
        def counted(rows):
            calls.append(sum(max(0,len(r)-1)for r in rows));return function(rows)
        original._pair_counts=counted
        try:scan=BPETokenizer.train(text,300,2)
        finally:original._pair_counts=function
        indexed,r=train_incremental(text,300,2,trace=True);self.assertEqual(indexed.merges,scan.merges);self.assertEqual(sum(calls),r['work']['scan_reference_adjacencies']);self.compare([list(p.encode())for p in pretokenize(text)],300,2)
        self.assertEqual((len(text.encode()),len(indexed.merges),sum(calls)),(1227,44,30664));self.assertEqual(r['work']['initial_edges'],985);self.assertEqual(r['work']['edge_removals'],1014);self.assertEqual(r['work']['edge_insertions'],568);self.assertEqual(r['work']['posting_positions_visited'],446);self.assertEqual(r['work']['heap_pushes'],652);self.assertEqual(r['work']['heap_pops'],117);self.assertEqual(r['work']['stale_heap_pops'],73)
