#!/usr/bin/env python3
"""Freeze natural graph pairs and run same-host short-budget classical controls."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.torch_version import TorchVersion
from torch_geometric.datasets import TUDataset
from rdkit import rdBase
from nema.classical_followups import McSplit, fmcs, score_mapping
from nema.data import load_pair, pair_paths
from nema.graph import LabeledGraph, GraphPair
from nema.models.enpda import ENPDAModel
from nema.enpda_solver import ENPDASolver

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/enpda_submission_followups.json'
OUTPUT=ROOT/'results/submission_followups'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def graph_payload(data):
    edges=sorted({tuple(sorted((int(u),int(v)))) for u,v in data.edge_index.t().tolist() if u!=v})
    return {'nodes':int(data.num_nodes),'edges':[list(e) for e in edges]}
def from_payload(payload):
    edges=torch.tensor(payload['edges'],dtype=torch.long).reshape(-1,2).t().contiguous()
    return LabeledGraph(torch.ones(payload['nodes'],dtype=torch.long),edges,torch.zeros(edges.shape[1],dtype=torch.long))
def prepare(config):
    path=OUTPUT/'natural_manifest.json'
    if path.exists():
        value=json.loads(path.read_text());assert value['config_sha256']==sha(CONFIG);return value
    dataset=TUDataset(root=str(ROOT/'data/nonmolecular'),name=config['natural_dataset'])
    rows=[];used=set()
    for lo,hi,count in config['natural_bins']:
        candidates=[i for i,g in enumerate(dataset) if lo<=int(g.num_nodes)<hi]
        candidates.sort(key=lambda i:hashlib.sha256(f"{config['selection_seed']}:{lo}:{hi}:{i}".encode()).hexdigest())
        assert len(candidates)>=2*count
        for k in range(count):
            a,b=candidates[2*k:2*k+2];assert a not in used and b not in used;used.update((a,b))
            rows.append({'key':f'IMDB-natural-{lo}-{k:03d}','bin':[lo,hi],'indices':[a,b],
                         'left':graph_payload(dataset[a]),'right':graph_payload(dataset[b])})
    value={'created_at_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':sha(CONFIG),
           'raw_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'data/nonmolecular/IMDB-BINARY/raw').glob('*.txt'))},'pairs':rows}
    write(path,value);return value
def natural_pairs(config):
    return [(GraphPair(from_payload(r['left']),from_payload(r['right']),key=r['key']),r['key']) for r in prepare(config)['pairs']]
def native_pairs(config):
    rows=[]
    for ds in config['native_datasets']:
        for path in pair_paths(ROOT/'data/official',ds):
            pair=load_pair(path)
            if pair.metadata is None: rows.append((pair,str(path.relative_to(ROOT))))
    assert len(rows)==291
    return rows
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',choices=['native','natural','prepare'],required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--limit',type=int);args=parser.parse_args()
    os.chdir(ROOT);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=json.loads(CONFIG.read_text());prepare(config)
    if args.suite=='prepare':return
    pairs=(native_pairs(config) if args.suite=='native' else natural_pairs(config))
    if args.limit:pairs=pairs[:args.limit]
    outdir=OUTPUT/(args.suite+('_smoke' if args.limit else '')+'_'+args.device)
    outdir.mkdir(parents=True,exist_ok=True)
    hardware={'platform':platform.platform(),'processor':platform.processor(),'torch':str(torch.__version__),
              'rdkit':rdBase.rdkitVersion,'device':args.device,'cuda':torch.version.cuda,
              'cpu_model':next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown'),
              'torch_threads':torch.get_num_threads(),'created_at_utc':datetime.now(timezone.utc).isoformat()}
    codefiles=['src/nema/classical_followups.py','scripts/build_mcsplit_followup.py','scripts/run_enpda_submission_followups.py',
               'src/nema/enpda_solver.py','src/nema/models/enpda.py','src/nema/association.py','src/nema/rounding.py','src/nema/features.py']
    provenance={'config_sha256':sha(CONFIG),'code_sha256':{p:sha(ROOT/p) for p in codefiles},
                'checkpoint_sha256':{str(s):sha(ROOT/f'checkpoints/enpda_formal/seed{s}.best.pt') for s in config['checkpoint_seeds']},
                'library_sha256':sha(ROOT/'artifacts/mcsplit-adapter/libmcsplit.so')}
    if (outdir/'provenance.json').exists(): assert json.loads((outdir/'provenance.json').read_text())==provenance
    else: write(outdir/'provenance.json',provenance);write(outdir/'hardware.json',hardware)
    models={}
    torch.serialization.add_safe_globals([TorchVersion])
    for seed in config['checkpoint_seeds']:
        payload=torch.load(ROOT/f'checkpoints/enpda_formal/seed{seed}.best.pt',weights_only=True,map_location='cpu')
        model=ENPDAModel(**payload['model_config']);model.load_state_dict(payload['model'])
        models[seed]=model
    # Warm neural runtime and both classical adapters outside timed evaluations.
    classic=McSplit(ROOT/'artifacts/mcsplit-adapter/libmcsplit.so')
    warm=GraphPair(from_payload({'nodes':3,'edges':[[0,1],[1,2]]}),from_payload({'nodes':3,'edges':[[0,1],[1,2]]}))
    for model in models.values(): ENPDASolver(model,device=args.device).solve(warm)
    fmcs(warm,.01);classic(warm,.01)
    output=outdir/'records.jsonl';done=set()
    if output.exists():
        for line in output.read_text().splitlines():
            row=json.loads(line);ident=(row['source'],row['method'],row.get('seed'),row.get('budget_seconds'))
            assert ident not in done;done.add(ident)
    with output.open('a') as stream:
        for index,(pair,source) in enumerate(pairs):
            arms=[('ENPDA-Core',seed,None) for seed in models]+[('Analytic-Core',0,None)]
            classical_methods=('FMCS','McSplit') if args.suite=='native' else ('McSplit',)
            arms += [(method,None,budget) for budget in config['classical_budgets_seconds'] for method in classical_methods]
            # Deterministic per-pair random ordering prevents systematic thermal/order bias.
            np.random.default_rng(config['selection_seed']+index).shuffle(arms)
            for method,seed,budget in arms:
                if (source,method,seed,budget) in done:continue
                if method.endswith('Core'):
                    solver=ENPDASolver(models[seed],trajectory='analytic' if method=='Analytic-Core' else 'learned',device=args.device)
                    if args.device=='cuda':torch.cuda.synchronize()
                    start=time.perf_counter();result=solver.solve(pair)
                    if args.device=='cuda':torch.cuda.synchronize()
                    row=result.to_dict();assert score_mapping(pair,row['mapping'])==row['common_edges']
                    row['runtime_seconds']=time.perf_counter()-start
                else:row=(fmcs if method=='FMCS' else classic)(pair,budget)
                assert score_mapping(pair,row['mapping'])==row['common_edges']
                row.update({'method':method,'seed':seed,'source':source,'key':pair.key,'true_edges':pair.true_edges,
                            'budget_seconds':budget,'nodes':[pair.left.num_nodes,pair.right.num_nodes],
                            'edges':[pair.left.num_edges,pair.right.num_edges]})
                stream.write(json.dumps(row)+'\n');stream.flush()
            print(f'{args.suite} {index+1}/{len(pairs)} {source}',flush=True)
    assert provenance['code_sha256']=={p:sha(ROOT/p) for p in codefiles}
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows)==len(pairs)*(10 if args.suite=='native' else 7)
    write(outdir/'COMPLETE.json',{'pairs':len(pairs),'records':len(rows),'output_sha256':sha(output),'provenance':provenance})
if __name__=='__main__': main()
