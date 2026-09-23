#!/usr/bin/env python3
"""Validate all maps, retain unresolved instances, and generate paired reports."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from nema.classical_followups import score_mapping
from run_enpda_submission_followups import ROOT,OUTPUT,CONFIG,native_pairs,natural_pairs,prepare,write,sha

def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
def check_marker(folder,name,filename):
    marker=json.loads((folder/name).read_text());assert marker['output_sha256']==sha(folder/filename)
def bootstrap_effect(delta,lower,upper,seed=20260916):
    # delta shape: independent model seeds x paired graphs. Denominator interval
    # gives conservative endpoints for every resampled effect, including negative effects.
    low=np.minimum(delta/lower,delta/upper)*100
    high=np.maximum(delta/lower,delta/upper)*100
    rng=np.random.default_rng(seed);n=delta.shape[1];count=20000
    lows=np.empty(count);highs=np.empty(count)
    for start in range(0,count,500):
        b=min(500,count-start);indices=rng.integers(n,size=(b,n));seeds=rng.integers(delta.shape[0],size=(b,delta.shape[0]))
        lows[start:start+b]=low[seeds[:,:,None],indices[:,None,:]].mean(axis=(1,2))
        highs[start:start+b]=high[seeds[:,:,None],indices[:,None,:]].mean(axis=(1,2))
    return {'mean_interval':[float(low.mean()),float(high.mean())],
            'paired_95_ci':[float(np.quantile(lows,.025)),float(np.quantile(highs,.975))]}
def summarize_suite(suite,config,device='cpu'):
    folder=OUTPUT/f'{suite}_{device}'
    check_marker(folder,'COMPLETE.json','records.jsonl')
    check_marker(folder,'latency_matched_COMPLETE.json','latency_matched.jsonl')
    pairs=native_pairs(config) if suite=='native' else natural_pairs(config)
    pair_lookup={s:p for p,s in pairs}
    raw=read(folder/'records.jsonl');matched=read(folder/'latency_matched.jsonl')
    lookup={}
    for row in raw+matched:
        assert row['common_edges']==score_mapping(pair_lookup[row['source']],row['mapping'])
        arm=row['method']
        if not arm.endswith('Core'):
            arm+='-matched' if 'budget_rule' in row else f"-{row['budget_seconds']:.2f}s"
        identity=(row['source'],arm,row.get('seed'));assert identity not in lookup
        lookup[identity]=row
    oracle={}
    if suite=='natural':
        check_marker(OUTPUT,'natural_oracle_combined_COMPLETE.json','natural_oracle_combined.jsonl')
        oracle={r['key']:r for r in read(OUTPUT/'natural_oracle_combined.jsonl')}
        assert set(oracle)==set(pair_lookup)
        for source,pair in pair_lookup.items(): assert score_mapping(pair,oracle[source]['mapping'])==oracle[source]['lower_bound']
    results={}
    subsets={ds:[s for p,s in pairs if f'/{ds}-test/' in s] for ds in config['native_datasets']} if suite=='native' else {'IMDB-natural':list(pair_lookup)}
    for ds,sources in subsets.items():
        if suite=='native':lower=upper=np.array([pair_lookup[s].true_edges for s in sources],dtype=float)
        else:
            lower=np.array([max(oracle[s]['lower_bound'],max(r['common_edges'] for r in raw+matched if r['source']==s)) for s in sources],dtype=float)
            upper=np.array([oracle[s]['upper_bound'] for s in sources],dtype=float)
            assert np.all(lower<=upper) and np.all(lower>0)
        methods={};edge_arrays={}
        arms=sorted({a for s,a,seed in lookup})
        for arm in arms:
            seeds=[0,1,2] if arm=='ENPDA-Core' else [0] if arm=='Analytic-Core' else [None]
            rows=[[lookup[s,arm,seed] for s in sources] for seed in seeds]
            edges=np.array([[r['common_edges'] for r in sr] for sr in rows],dtype=float)
            times=np.array([[r['runtime_seconds'] for r in sr] for sr in rows]);edge_arrays[arm]=edges
            entry={'pairs':len(sources),'records':int(edges.size),
                   'accuracy_interval':[float((edges/upper*100).mean()),float((edges/lower*100).mean())],
                   'mean_seconds':float(times.mean()),'median_seconds':float(np.median(times)),
                   'p95_seconds':float(np.quantile(times,.95)),
                   'fraction_within_0_16_seconds':float(np.mean(times<=.16))}
            if not arm.endswith('Core'):
                budgets=np.array([[r['budget_seconds'] for r in sr] for sr in rows]);over=np.maximum(0,times-budgets)
                entry.update({'mean_budget_seconds':float(budgets.mean()),'mean_overshoot_seconds':float(over.mean()),'max_overshoot_seconds':float(over.max()),
                              'fraction_over_budget_plus_1ms':float(np.mean(times>budgets+.001)),
                              'timeout_fraction':float(np.mean([r['timed_out'] for sr in rows for r in sr]))})
            methods[arm]=entry
        effects={}
        for arm in arms:
            if arm!='ENPDA-Core': effects[arm]=bootstrap_effect(edge_arrays['ENPDA-Core']-edge_arrays[arm],lower,upper)
        result={'pairs':len(sources),'methods':methods,'enpda_minus':effects,'device':device,
                'hardware':json.loads((folder/'hardware.json').read_text())}
        if suite=='natural':
            result.update({'oracle_certified':sum(oracle[s]['certified_optimal'] for s in sources),
                           'oracle_timeouts':sum(oracle[s]['timed_out'] for s in sources),
                           'certified_after_valid_incumbent_caps':int(np.sum(lower==upper)),
                           'unresolved':int(np.sum(lower<upper)),
                           'mean_oracle_bound_gap_percent':float(((upper-lower)/upper*100).mean())})
        results[ds]=result
    return results
def interval(v):return f'{v[0]:.2f}' if abs(v[0]-v[1])<1e-9 else f'[{v[0]:.2f}, {v[1]:.2f}]'
def generate_tables(summary):
    lines=['# ENPDA submission follow-ups','',
           'New timings use serial evaluations on the same host within each device experiment, with one torch/BLAS/search thread. Historical H100 timings are not mixed with new measurements. Classical caps are cooperative; actual latency and overshoot are retained.',
           '', 'FMCS maximizes connected common bonds; McSplit maximizes induced common vertices. Their legal returned maps are scored by the same unrestricted MCES edge count. These are practical reference baselines with different search objectives, not interchangeable exact MCES solvers.', '']
    native=summary['native'];datasets=list(native)
    device=native[datasets[0]]['device'];host='H100 + host CPU' if device=='cuda' else 'CPU'
    tex=[r'\begin{table}[t]',r'\centering\small',
         r'\caption{Same-host '+host+r' short-budget references on all 291 native pairs. Entries are MCES accuracy (\%) / measured mean milliseconds. Fixed caps are nominal cooperative caps; ``matched'' grants each pair the mean latency of its three Core seeds. FMCS optimizes connected bonds; McSplit optimizes induced vertices. All maps are independently scored by unrestricted MCES.}',
         r'\label{tab:short-classical}',r'\begin{tabular}{lrrr}',r'\toprule Method & AIDS & MOLHIV & MCF-7 \\',r'\midrule']
    arm_order=['Analytic-Core','ENPDA-Core','FMCS-0.10s','FMCS-0.16s','FMCS-0.60s','McSplit-0.10s','McSplit-0.16s','McSplit-0.60s','FMCS-matched','McSplit-matched']
    for ds in datasets:
        lines.extend([f'## {ds} ({native[ds]["pairs"]} pairs)','', '| Method | MCES accuracy % | Mean ms | P95 ms | ENPDA minus method, paired 95% CI (points) |','|---|---:|---:|---:|---|'])
        for arm in arm_order:
            row=native[ds]['methods'][arm];effect=native[ds]['enpda_minus'].get(arm)
            ci='—' if effect is None else str([round(v,2) for v in effect['paired_95_ci']])
            lines.append(f"| {arm} | {interval(row['accuracy_interval'])} | {1000*row['mean_seconds']:.1f} | {1000*row['p95_seconds']:.1f} | {ci} |")
        lines.append('')
    for arm in arm_order:
        values=[f"{native[ds]['methods'][arm]['accuracy_interval'][0]:.2f} / {1000*native[ds]['methods'][arm]['mean_seconds']:.1f}" for ds in datasets]
        tex.append(arm+' & '+' & '.join(values)+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (OUTPUT/'short_classical.tex').write_text('\n'.join(tex)+'\n')
    natural=summary['natural']['IMDB-natural']
    lines.extend(['## Natural IMDB-BINARY (100 unedited graph pairs)','',
                  f"Independent oracle certificates: {natural['oracle_certified']}/100; raw oracle timeouts: {natural['oracle_timeouts']}/100. After intersecting structural upper bounds with every validated legal incumbent, {natural['unresolved']} pairs remain unresolved. Every preselected pair is retained.",
                  '', '| Method | All-pair accuracy interval % | Mean ms | ENPDA minus method, conservative paired 95% CI |','|---|---:|---:|---|'])
    ntex=[r'\begin{table}[t]',r'\centering\small',r'\caption{Frozen transfer to 100 unedited, index-disjoint natural IMDB-BINARY pairs (10--39 nodes). Accuracy ranges retain all unresolved optima; confidence intervals combine paired graph/seed resampling with conservative denominator bounds. Timings are same-host CPU milliseconds.}',
          r'\label{tab:natural-imdb-accuracy}',r'\begin{tabular}{lrrr}',r'\toprule Method & Accuracy (\%) & Time (ms) & ENPDA gain, 95\% CI \\',r'\midrule']
    for arm in arm_order:
        if arm not in natural['methods']:continue
        row=natural['methods'][arm];effect=natural['enpda_minus'].get(arm)
        ci='---' if effect is None else '['+','.join(f'{v:.2f}' for v in effect['paired_95_ci'])+']'
        lines.append(f"| {arm} | {interval(row['accuracy_interval'])} | {1000*row['mean_seconds']:.1f} | {ci} |")
        if arm in ['Analytic-Core','ENPDA-Core','FMCS-0.16s','McSplit-0.16s','FMCS-matched','McSplit-matched']:
            ntex.append(f"{arm} & {interval(row['accuracy_interval'])} & {1000*row['mean_seconds']:.1f} & {ci}"+r' \\')
    ntex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (OUTPUT/'natural_imdb.tex').write_text('\n'.join(ntex)+'\n')
    lines.extend(['','## Provenance and limitations','',
                  'The original protocol was fixed before the full experiment. The latency-match supplement was declared after the first 38 native CPU pairs had partial outputs, before any latency-matched baseline was run; it deterministically covers every pair.',
                  '', 'The natural set has 50 pairs from each half-open node-size bin [10,20) and [20,40). Within each bin, graph indices are sorted by SHA256(20260916:lo:hi:index), and consecutive entries among the first 100 are paired. No edits or class/outcome conditioning are used. Distinct dataset indices do not guarantee non-isomorphic topologies.',
                  '', 'Model checkpoints are unchanged. All maps, raw times, timeout flags, input hashes, software versions and code hashes are retained alongside the report. The 75-pair certificate selection audit independently verifies every file hash against deterministic systematic positions.',
                  '', 'The attempted RASCAL natural oracle returned a non-injective map and was rejected. The final oracle combines the sparse lifted integer program, independent degree/edge caps, and exact complete-graph reductions. FMCS was omitted from the entire natural suite after failing to respond to its fractional-time callback for several minutes on the third pair; all native FMCS results remain. No graph pair was removed or replaced.',
                  '', 'Sources: [official McSplit code](https://github.com/jamestrimble/ijcai2017-partitioning-common-subgraph), [RDKit FMCS API](https://rdkit.org/docs/source/rdkit.Chem.rdFMCS.html), [RDKit RASCAL API](https://rdkit.org/docs/source/rdkit.Chem.rdRascalMCES.html), [TU datasets](https://chrsmrrs.github.io/datasets/docs/datasets/).'])
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
def main():
    config=json.loads(CONFIG.read_text())
    device='cuda' if (OUTPUT/'native_cuda/latency_matched_COMPLETE.json').exists() else 'cpu'
    summary={'native':summarize_suite('native',config,device),'natural':summarize_suite('natural',config)}
    if device=='cuda':summary['native_cpu']=summarize_suite('native',config)
    write(OUTPUT/'summary.json',summary);generate_tables(summary)
    write(OUTPUT/'COMPLETE.json',{'summary_sha256':sha(OUTPUT/'summary.json'),'report_sha256':sha(OUTPUT/'report.md'),
                                'native_pairs':291,'natural_pairs':100,'bootstrap_replicates':20000})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
