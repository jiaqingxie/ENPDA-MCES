"""Rebuild globally graph-disjoint pools, label them, and evaluate frozen methods."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from nema.graph import LabeledGraph
from nema.data import load_pair, pair_paths
from nema.reconstructed import collect_graph_bank, iter_pickle_graphs
import nema.hard_retrieval as hard
from nema.retrieval_baselines import build_retrieval_baseline
from run_enpda_pilot import atomic_json, sha256
import build_hard_retrieval as builder
import run_unified_retrieval_baseline_gpu as baseline
from audit_hard_retrieval_metric_bounds import method_bounds

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/enpda_retrieval_disjoint_v2.json'
WORK = ROOT


class ExactIdentity:
    def __init__(self):
        self.cache = {}
        self.buckets = defaultdict(list)
        self.vf2_checks = 0

    def __call__(self, data):
        graph = data if isinstance(data, LabeledGraph) else LabeledGraph.from_pyg(data)
        value = {'nodes':graph.node_labels.tolist(),
                 'edges':sorted((int(graph.edge_index[0,k]),int(graph.edge_index[1,k]),int(graph.edge_labels[k])) for k in range(graph.num_edges))}
        ordered = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        if ordered in self.cache:
            return self.cache[ordered]
        g=nx.Graph()
        g.add_nodes_from((i,{'label':v}) for i,v in enumerate(value['nodes']))
        g.add_edges_from((u,v,{'label':label}) for u,v,label in value['edges'])
        key=(g.number_of_nodes(),g.number_of_edges(),nx.weisfeiler_lehman_graph_hash(g,node_attr='label',edge_attr='label',iterations=4))
        for ident,other in self.buckets[key]:
            self.vf2_checks+=1
            if nx.is_isomorphic(g,other,node_match=nx.algorithms.isomorphism.categorical_node_match('label',None),
                                edge_match=nx.algorithms.isomorphism.categorical_edge_match('label',None)):
                self.cache[ordered]=ident
                return ident
        self.buckets[key].append((ordered,g))
        self.cache[ordered]=ordered
        return ordered


def prepare(cfg, dataset, seed, folder):
    identity=ExactIdentity()
    training=set()
    source_hashes={}
    for name in ('AIDS','MOLHIV','MCF-7'):
        for path in sorted((ROOT/f'data/official/MCES/{name}-train/raw').glob('graphs_*.pkl')):
            source_hashes[str(path.relative_to(ROOT))]=sha256(path)
            for _,data,_ in iter_pickle_graphs(path): training.add(identity(data))
    # Independently establish that the broader exclusion bank covers the new train/validation tensors.
    for split in ('train','validation'):
        rows=torch.load(ROOT/f'data/enpda_graph_disjoint_v1/{split}.pt',map_location='cpu',weights_only=False)
        assert all(identity(g) in training for r in rows for g in (r['pair'].left,r['pair'].right))
    bank,_=collect_graph_bank(ROOT/'data/official',dataset)
    unique={}
    for entry in bank:
        ident=identity(entry.data)
        if ident in training or ident in unique:continue
        entry.fingerprint=ident
        unique[ident]=entry
        source_hashes[str(Path(entry.source_path).relative_to(ROOT))]=sha256(Path(entry.source_path))
    bank=[unique[k] for k in sorted(unique)]
    hard.graph_fingerprint=identity
    queries,pools,library,audit=hard.build_hard_protocol(bank,training,cfg['queries'],cfg['candidates'],cfg['controlled_count'],
                                                      seed,cfg['parent_pool_size'],cfg['derived_per_parent'])
    query_ids={identity(q.data) for q in queries}
    candidate_ids={identity(e.candidate.graph.data) for pool in pools for e in pool}
    parent_ids={e.parent_fingerprint for e in library}
    assert not (query_ids|candidate_ids|parent_ids)&training
    assert not query_ids&parent_ids
    assert len(query_ids)==cfg['queries'] and all(len({identity(e.candidate.graph.data) for e in pool})==cfg['candidates'] for pool in pools)
    manifest={'protocol_version':cfg['protocol_version'],'config_sha256':sha256(CONFIG),'dataset':dataset,'seed':seed,
              'queries':cfg['queries'],'candidates':cfg['candidates'],'training_graph_classes':len(training),
              'globally_filtered_bank_classes':len(bank),'identity_vf2_checks':identity.vf2_checks,
              'graph_overlap':0,'selection_audit':audit,'source_sha256':source_hashes,
              'query_graphs':[q.manifest_record() for q in queries],
              'query_pools':[[e.manifest_record() for e in pool] for pool in pools],
              'candidate_library':[e.manifest_record() for e in library],
              'controlled_candidates':[[e.candidate.manifest_record() for e in pool if e.candidate.candidate_kind.startswith('controlled_')] for pool in pools]}
    path=folder/'manifest.json'
    if path.exists(): assert json.loads(path.read_text())==manifest,'Frozen pool changed'
    else: atomic_json(path,manifest)
    print('Pool frozen:',dataset,seed,'heldout bank',len(bank),'global training classes',len(training),'overlap 0',flush=True)
    return queries,pools,manifest,identity,training


def audit_materialized(rows, identity, training, out):
    hashes={}
    for row in rows:
        path=Path(row['source_path'])
        pair=load_pair(path)
        left,right=identity(pair.left),identity(pair.right)
        assert left==row['query_fingerprint'] and right==row['candidate_fingerprint']
        assert left not in training and right not in training
        hashes[str(path.relative_to(ROOT))]=sha256(path)
    atomic_json(out/'materialized_input_audit.json',{'pairs':len(rows),'graph_overlap':0,'sha256':hashes})


def rescore_mappings(path):
    count=0
    for line in path.read_text().splitlines():
        row=json.loads(line)
        pair=load_pair(row['source_path'])
        left,right=pair.left,pair.right
        assert row['metadata']['mapping_encoding']=='source_to_target; -1 denotes an unmatched source node'
        mapping=row['mapping']
        assert len(mapping)==left.num_nodes and all(-1<=v<right.num_nodes for v in mapping)
        used=[v for v in mapping if v>=0]
        assert len(used)==len(set(used))
        assert all(int(left.node_labels[u])==int(right.node_labels[v]) for u,v in enumerate(mapping) if v>=0)
        target={(min(u,v),max(u,v)):label for (u,v),label in zip(right.edge_index.t().tolist(),right.edge_labels.tolist())}
        edges=0;incident=set()
        for (u,v),label in zip(left.edge_index.t().tolist(),left.edge_labels.tolist()):
            a,b=mapping[u],mapping[v]
            if a>=0 and b>=0 and target.get((min(a,b),max(a,b)))==label:
                edges+=1;incident.update((u,v))
        assert (edges,len(incident))==(row['common_edges'],row['common_nodes'])
        count+=1
    assert count==10000
    return count


def oracle(cfg, dataset, seed, folder, queries, pools):
    raw=folder/'raw/test';raw.mkdir(parents=True,exist_ok=True)
    builder._QUERIES=queries;builder._POOLS=pools;builder._RAW=raw
    builder._CANDIDATE_COUNT=cfg['candidates'];builder._RELEVANCE_THRESHOLD=.5
    builder._RASCAL_TIMEOUT_SECONDS=cfg['oracle']['seconds_per_pair'];builder._PROTOCOL_SEED=seed
    path=folder/'oracle.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done={int(r['key'])-1 for r in rows};assert len(done)==len(rows)
    pending=[i for i in range(cfg['queries']*cfg['candidates']) if i not in done]
    with path.open('a') as stream:
        with ProcessPoolExecutor(max_workers=cfg['oracle']['workers'],mp_context=mp.get_context('fork')) as pool:
            futures=[pool.submit(builder._build_index,i) for i in pending]
            for index,future in enumerate(as_completed(futures),1):
                row=future.result();rows.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if index%250==0 or index==len(pending):print(f'oracle {len(rows)}/10000',flush=True)
    assert len(rows)==10000 and {int(r['key']) for r in rows}==set(range(1,10001))
    atomic_json(folder/'oracle_summary.json',builder._summarize(rows,cfg['candidates']))
    return sorted(rows,key=lambda r:int(r['key']))


def evaluate(cfg, dataset, seed, model_seed, folder, rows, out):
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(0)
    assert str(torch.version.cuda).startswith('12.8')
    outputs={}
    for method,column in [('size','size_similarity'),('wl','wl_similarity')]:
        path=out/f'{method}.jsonl'
        with path.open('w') as stream:
            for r in rows:stream.write(json.dumps({'key':str(r['key']),'similarity':r[column],'source_path':r['source_path']})+'\n')
        outputs[method]=path
    cache=baseline.load_test_cache(rows,torch.device('cuda'))
    checkpoint_hashes={}
    for method in ('simgnn','gmn','neuromatch'):
        cp=ROOT/f'checkpoints/unified_retrieval/{dataset}/{method}/seed{model_seed}.best.pt'
        checkpoint_hashes[method]=sha256(cp)
        payload=torch.load(cp,map_location='cpu',weights_only=False)
        assert payload['dataset']==dataset and payload['method']==method and payload['model_seed']==model_seed
        model=build_retrieval_baseline(method,hidden_dim=payload['config']['hidden_dim'],layers=payload['config']['message_passing_layers']).cuda()
        model.load_state_dict(payload['model_state_dict'],strict=True)
        path=out/f'{method}.jsonl'
        _,seconds=baseline.evaluate(model,rows,cache,path,method)
        outputs[method]=path
        atomic_json(out/f'{method}.timing.json',{'seconds':seconds,'checkpoint_sha256':checkpoint_hashes[method]})
        del model
    del cache
    torch.cuda.empty_cache()
    # Reuse the original Fast evaluator and budget; change only its input pool and destinations.
    sys.path.insert(0,str(WORK/'scripts'))
    import eval_enpda_retrieval_gpu as enpda
    enpda.pair_paths=lambda ignored_root,name,retrieval:pair_paths(ROOT/cfg['data_root']/f'seed-{seed}',name,retrieval=True)
    cp=ROOT/f'checkpoints/enpda_graph_disjoint_v1/formal/seed{model_seed}/full.best.pt'
    checkpoint_hashes['enpda']=sha256(cp)
    outputs['enpda']=out/'enpda_fast.jsonl'
    sys.argv=['eval_enpda_retrieval_gpu.py','--dataset',dataset,'--protocol-seed',str(seed),'--training-seed',str(model_seed),
              '--checkpoint',str(cp),'--output',str(outputs['enpda']),'--attestation',str(out/'enpda_COMPLETE.json'),'--workers','8']
    enpda.main()
    rescored=rescore_mappings(outputs['enpda'])
    classification={}
    for r in rows:
        if r.get('rascal_timed_out'):
            pair=load_pair(r['source_path'])
            upper=hard.mces_similarity_upper_bound(pair.left,pair.right)
            assert upper+2e-6>=r['true_similarity']
            classification[int(r['key'])]={'rascal_lower_bound':r['true_similarity'],
                                         'relaxation_upper_bound':max(upper,r['true_similarity'])}
    results={method:{scope:method_bounds(path,rows,classification,.5,hard_only=(scope=='hard_only'))
                     for scope in ('full','hard_only')} for method,path in outputs.items()}
    atomic_json(out/'summary.json',{'status':'complete','dataset':dataset,'seed':seed,'model_seed':model_seed,
                                 'methods':results,'checkpoint_sha256':checkpoint_hashes,
                                 'oracle_timeouts':len(classification),'graph_overlap':0,
                                 'independently_rescored_mappings':rescored,
                                 'latex_updated':False})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['MOLHIV','MCF-7'],required=True)
    parser.add_argument('--model-seed',type=int,choices=[0,1,2],required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--oracle-only', action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT);torch.set_num_threads(1)
    cfg=json.loads(CONFIG.read_text());seed=cfg['protocol_seeds'][args.model_seed]
    folder=ROOT/cfg['data_root']/f'seed-{seed}/retrieval'/args.dataset
    out=ROOT/cfg['output_root']/f'seed-{seed}'/args.dataset
    out.mkdir(parents=True,exist_ok=True)
    code=[Path(__file__),ROOT/'src/nema/hard_retrieval.py',ROOT/'scripts/build_hard_retrieval.py',
          ROOT/'src/nema/reconstructed.py',ROOT/'scripts/run_unified_retrieval_baseline_gpu.py',
          ROOT/'scripts/audit_hard_retrieval_metric_bounds.py',ROOT/'src/nema/retrieval_bounds.py',
          ROOT/'src/nema/models/enpda.py',ROOT/'src/nema/enpda_solver.py',WORK/'scripts/eval_enpda_retrieval_gpu.py']
    provenance={'config_sha256':sha256(CONFIG),'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in code},
                'dataset':args.dataset,'seed':seed,'model_seed':args.model_seed}
    frozen=out/('preparation_provenance.json' if args.prepare_only else 'provenance.json')
    if frozen.exists():assert json.loads(frozen.read_text())==provenance
    else:atomic_json(frozen,provenance)
    lock = out / ('oracle_writer.lock' if args.oracle_only else 'writer.lock')
    if not args.prepare_only:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    record={'status':'running','stage':'pool_preparation','started_at_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(out/'status.json',record)
    try:
        queries,pools,manifest,identity,training=prepare(cfg,args.dataset,seed,folder)
        if args.prepare_only:
            atomic_json(out/'PREPARED.json',{'status':'complete','manifest_sha256':sha256(folder/'manifest.json'),'graph_overlap':0})
            return
        record['stage']='oracle';atomic_json(out/'status.json',record)
        rows=oracle(cfg,args.dataset,seed,folder,queries,pools)
        record['stage']='materialized_input_audit';atomic_json(out/'status.json',record)
        audit_materialized(rows,identity,training,out)
        if args.oracle_only:
            atomic_json(out/'ORACLE_COMPLETE.json', {'pairs': len(rows), 'oracle_sha256': sha256(folder/'oracle.jsonl')})
            return
        record['stage']='evaluation';atomic_json(out/'status.json',record)
        evaluate(cfg,args.dataset,seed,args.model_seed,folder,rows,out)
        assert provenance['code_sha256']=={str(p.relative_to(ROOT)):sha256(p) for p in code}
        assert provenance['config_sha256']==sha256(CONFIG)
        record.update(status='complete',stage='verified_metric_intervals',finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      manifest_sha256=sha256(folder/'manifest.json'),oracle_sha256=sha256(folder/'oracle.jsonl'),summary_sha256=sha256(out/'summary.json'))
        atomic_json(out/'COMPLETE.json',record);atomic_json(out/'status.json',record)
    except BaseException as error:
        record.update(status='failed',error=repr(error));atomic_json(out/'status.json',record);raise
    finally:
        if not args.prepare_only:
            lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
