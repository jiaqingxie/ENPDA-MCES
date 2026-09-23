#!/usr/bin/env python3
"""Classical references allotted each pair's mean measured frozen-Core latency."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json
from pathlib import Path
import numpy as np
import torch
from nema.classical_followups import McSplit,fmcs,score_mapping
from run_enpda_submission_followups import ROOT,OUTPUT,CONFIG,native_pairs,natural_pairs,write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['native','natural'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=p.parse_args()
    os.chdir(ROOT);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=json.loads(CONFIG.read_text());folder=OUTPUT/f'{args.suite}_{args.device}'
    marker=json.loads((folder/'COMPLETE.json').read_text());assert marker['output_sha256']==sha(folder/'records.jsonl')
    records=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines()]
    pairs=native_pairs(config) if args.suite=='native' else natural_pairs(config)
    times={}
    for row in records:
        if row['method']=='ENPDA-Core':times.setdefault(row['source'],[]).append(row['runtime_seconds'])
    assert len(times)==len(pairs) and all(len(x)==3 for x in times.values())
    provenance={'config_sha256':sha(ROOT/'configs/enpda_submission_latency_match.json'),
                'base_records_sha256':sha(folder/'records.jsonl'),'script_sha256':sha(__file__),
                'adapter_sha256':sha(ROOT/'src/nema/classical_followups.py')}
    output=folder/'latency_matched.jsonl';done=set()
    if output.exists():
        assert json.loads((folder/'latency_matched_provenance.json').read_text())==provenance
        for line in output.read_text().splitlines():
            r=json.loads(line);assert (r['source'],r['method']) not in done;done.add((r['source'],r['method']))
    write(folder/'latency_matched_provenance.json',provenance)
    solver=McSplit(ROOT/'artifacts/mcsplit-adapter/libmcsplit.so')
    with output.open('a') as stream:
        for i,(pair,source) in enumerate(pairs):
            budget=float(np.mean(times[source]))
            arms=[('FMCS',fmcs),('McSplit',solver)] if args.suite=='native' else [('McSplit',solver)]
            if i%2:arms.reverse()
            for method,method_fn in arms:
                if (source,method) in done:continue
                row=method_fn(pair,budget);assert row['common_edges']==score_mapping(pair,row['mapping'])
                row.update({'source':source,'key':pair.key,'method':method,'seed':None,'true_edges':pair.true_edges,'budget_rule':f'same-pair mean Core {args.device} latency'})
                stream.write(json.dumps(row)+'\n');stream.flush()
            print(f'matched {args.suite} {i+1}/{len(pairs)}',flush=True)
    rows=[json.loads(line) for line in output.read_text().splitlines()];assert len(rows)==(2 if args.suite=='native' else 1)*len(pairs)
    write(folder/'latency_matched_COMPLETE.json',{'pairs':len(pairs),'records':len(rows),'output_sha256':sha(output),'provenance':provenance})
if __name__=='__main__':main()
