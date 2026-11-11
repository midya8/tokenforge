"""Installed-only imports and every original/new CLI outside source checkout."""
import importlib,json,subprocess,sys,tempfile
from pathlib import Path
root=Path(sys.argv[1]).resolve()
for name in ['tokenforge','tokenforge.tokenizer','tokenforge.pretokenize','tokenforge.stats','tokenforge.cli','tokenforge.incremental','tokenforge.incremental_cli']:
    m=importlib.import_module(name);assert root not in Path(m.__file__).resolve().parents,name
from tokenforge import BPETokenizer,train_incremental
from tokenforge.incremental_cli import audit
ws=json.loads((root/'examples/indexed.json').read_text())
for w in ws:assert audit(w['input'])==w['report']
text=(root/'examples/data/corpus.txt').read_text();a,r=train_incremental(text,300,2,trace=True);b=BPETokenizer.train(text,300,2);assert a.merges==b.merges
assert a.decode(a.encode(text))==text
with tempfile.TemporaryDirectory()as t:
    t=Path(t);source=t/'text.txt';source.write_text(text);scan=t/'scan.json';indexed=t/'indexed.json';report=t/'trace.json'
    def run(args):return subprocess.run(args,capture_output=True,text=True,check=True).stdout
    base=[sys.executable,'-m','tokenforge']
    run(base+['train','--input',str(source),'--vocab-size','300','--output',str(scan)])
    run(base+['train','--input',str(source),'--vocab-size','300','--output',str(indexed),'--trainer','indexed','--training-report',str(report)])
    assert scan.read_bytes()==indexed.read_bytes();assert json.loads(report.read_text())==r
    for vocab in [scan,indexed]:
        ids=run(base+['encode','--vocab',str(vocab),'--text','café 👋 tokenizer']).strip()
        assert run(base+['decode','--vocab',str(vocab),'--ids',ids]).strip()=='café 👋 tokenizer'
        stats=json.loads(run(base+['stats','--vocab',str(vocab),'--input',str(source),'--json',str(t/'stats.json')])and(t/'stats.json').read_text());assert stats
    p=t/'bytes.json';p.write_text(json.dumps(ws[0]['input']));x=run([sys.executable,'-m','tokenforge.incremental_cli','--input',str(p)]);y=run([str(Path(sys.executable).parent/'tokenforge-indexed'),'--input',str(p)]);assert json.loads(x)==json.loads(y)==ws[0]['report']
print('Installed:7importorigins/8packagefiles;27completeaudits;originalcorpus44merges;all4originalCLImodes/newindexedtraining/bothboundedentrypoints')
