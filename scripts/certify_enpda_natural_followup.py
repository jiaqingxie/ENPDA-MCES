#!/usr/bin/env python3
"""Independent unrestricted MCES oracle for every preselected natural graph pair."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse,json,time,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
import numpy as np
from nema.graph import GraphPair
from nema.classical_followups import score_mapping, _fmcs_molecule
from nema.association import AssociationGraph
from nema.certificate import solve_lifted_milp
from nema.solvers import source_to_target_mapping
from run_enpda_submission_followups import ROOT,OUTPUT,CONFIG,prepare,from_payload,sha,write

def worker(payload):
    row,budget=payload;torch.set_num_threads(1)
    pair=GraphPair(from_payload(row['left']),from_payload(row['right']),key=row['key'])
    start=time.perf_counter()
    degree0=sorted(pair.left.degrees.tolist(),reverse=True)
    degree1=sorted(pair.right.degrees.tolist(),reverse=True)
    cap=min(pair.left.num_edges,pair.right.num_edges,math.floor(sum(min(a,b) for a,b in zip(degree0,degree1))/2))
    order0=np.argsort(-pair.left.degrees.numpy(),kind='stable').tolist()
    order1=np.argsort(-pair.right.degrees.numpy(),kind='stable').tolist()
    mapping=[-1]*pair.left.num_nodes
    for u,v in zip(order0,order1):mapping[u]=v
    edges=score_mapping(pair,mapping)
    result=None
    if edges<cap:
        left,right,swapped=pair.oriented()
        result=solve_lifted_milp(AssociationGraph.build(left,right),time_limit=budget)
        if result.mapping is not None:
            other=source_to_target_mapping(pair,result.mapping,swapped).tolist()
            score=score_mapping(pair,other)
            if score>edges:mapping,edges=other,score
        if result.upper_bound is not None:cap=min(cap,math.floor(result.upper_bound))
    assert edges<=cap
    return {'key':row['key'],'mapping':mapping,'lower_bound':edges,'upper_bound':cap,
            'certified_optimal':edges==cap,'timed_out':result is not None and result.status==1,
            'milp_status':None if result is None else result.status,
            'milp_message':None if result is None else result.message,
            'milp_raw_upper':None if result is None else result.raw_upper_bound,
            'runtime_seconds':time.perf_counter()-start,
            'method':'lifted MILP + structural caps','budget_seconds':budget}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    config=json.loads(CONFIG.read_text());manifest=prepare(config)
    oracle_config=ROOT/'configs/enpda_natural_oracle_v2.json'
    budget=json.loads(oracle_config.read_text())['milp_seconds']
    output=OUTPUT/'natural_oracle.jsonl';done={}
    if output.exists():
        for line in output.read_text().splitlines():
            row=json.loads(line);assert row['key'] not in done;done[row['key']]=row
    provenance={'config_sha256':sha(CONFIG),'oracle_config_sha256':sha(oracle_config),'manifest_sha256':sha(OUTPUT/'natural_manifest.json'),
                'script_sha256':sha(__file__),'rdkit_version':__import__('rdkit').__version__,
                'certificate_code_sha256':sha(ROOT/'src/nema/certificate.py'),'milp_seconds':budget}
    att=OUTPUT/'natural_oracle_provenance.json'
    if att.exists():assert json.loads(att.read_text())==provenance
    else:write(att,provenance)
    tasks=[(r,budget) for r in manifest['pairs'] if r['key'] not in done]
    with output.open('a') as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,p) for p in tasks]):
            row=future.result();done[row['key']]=row;stream.write(json.dumps(row)+'\n');stream.flush()
            print(f"oracle {len(done)}/{len(manifest['pairs'])} {row['key']} [{row['lower_bound']},{row['upper_bound']}]",flush=True)
    assert set(done)=={r['key'] for r in manifest['pairs']}
    write(OUTPUT/'natural_oracle_COMPLETE.json',{'pairs':len(done),'certified':sum(r['certified_optimal'] for r in done.values()),'output_sha256':sha(output),'provenance':provenance})
if __name__=='__main__':main()
