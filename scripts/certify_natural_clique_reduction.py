#!/usr/bin/env python3
"""Exact MCES when one input is complete, by exhaustive vertex deletion.

For K_k versus G, MCES is the largest edge count of any min(k,|G|)-vertex
induced subgraph of G. This is an exact reduction for universal graph labels.
Apply the same fixed enumeration-size limit to every frozen natural pair.
"""
from __future__ import annotations
import ctypes,itertools,json,math,subprocess,time
from pathlib import Path
import numpy as np
from run_enpda_submission_followups import ROOT,OUTPUT,CONFIG,prepare,from_payload,sha,write
from nema.graph import GraphPair
from nema.classical_followups import score_mapping

CODE=r'''
#include <vector>
#include <algorithm>
static int n, removed_count, best, nedges;
static const int* adjacency;
static std::vector<int> degrees, chosen, solution;
static void visit(int next, int cost) {
    if(cost>=best) return;
    if((int)chosen.size()==removed_count) {best=cost;solution=chosen;return;}
    int need=removed_count-chosen.size();
    for(int v=next;v<=n-need;++v) {
        int increment=degrees[v];
        for(int u:chosen) increment-=adjacency[u*n+v];
        chosen.push_back(v);visit(v+1,cost+increment);chosen.pop_back();
    }
}
extern "C" int densest(int nodes, int keep, const int* adj, int* retained) {
    n=nodes;removed_count=n-keep;adjacency=adj;nedges=0;
    degrees.assign(n,0);chosen.clear();solution.clear();
    for(int u=0;u<n;++u)for(int v=0;v<n;++v)degrees[u]+=adj[u*n+v];
    for(int d:degrees)nedges+=d;nedges/=2;best=nedges+1;
    visit(0,0);
    for(int i=0;i<n;++i)retained[i]=1;
    for(int i:solution)retained[i]=0;
    return nedges-best;
}
'''
def library():
    folder=ROOT/'artifacts/clique-reduction';folder.mkdir(exist_ok=True)
    source=folder/'exact.cpp';source.write_text(CODE)
    binary=folder/'exact.so';subprocess.run(['g++','-O3','-shared','-fPIC',str(source),'-o',str(binary)],check=True)
    fn=ctypes.CDLL(str(binary)).densest;ptr=ctypes.POINTER(ctypes.c_int)
    fn.argtypes=[ctypes.c_int,ctypes.c_int,ptr,ptr];fn.restype=ctypes.c_int
    return fn
def dense(fn,g,k):
    n=g['nodes'];adj=np.zeros((n,n),dtype=np.int32)
    for u,v in g['edges']:adj[u,v]=adj[v,u]=1
    retained=np.zeros(n,dtype=np.int32);ptr=lambda x:x.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    score=fn(n,k,ptr(adj),ptr(retained));return score,np.flatnonzero(retained).tolist()
def self_check(fn):
    rng=np.random.default_rng(719)
    for n in range(3,9):
        edges=[(u,v) for u in range(n) for v in range(u+1,n) if rng.random()<.45]
        for k in range(1,n+1):
            expected=max(sum(u in s and v in s for u,v in edges) for s in map(set,itertools.combinations(range(n),k)))
            actual,_=dense(fn,{'nodes':n,'edges':edges},k);assert actual==expected
def main():
    fn=library();self_check(fn)
    manifest=prepare(json.loads(CONFIG.read_text()));results=[]
    for row in manifest['pairs']:
        start=time.perf_counter();a,b=row['left'],row['right']
        ac=len(a['edges'])==a['nodes']*(a['nodes']-1)//2
        bc=len(b['edges'])==b['nodes']*(b['nodes']-1)//2
        if not ac and not bc:continue
        complete,other,swap=(a,b,False) if ac else (b,a,True)
        k=min(complete['nodes'],other['nodes']);combinations=math.comb(other['nodes'],k)
        if combinations>5000000:continue
        score,kept=dense(fn,other,k)
        mapping=[-1]*a['nodes']
        for i,v in enumerate(kept):
            if swap:mapping[v]=i
            else:mapping[i]=v
        pair=GraphPair(from_payload(a),from_payload(b));assert score_mapping(pair,mapping)==score
        results.append({'key':row['key'],'mapping':mapping,'lower_bound':score,'upper_bound':score,
                        'certified_optimal':True,'timed_out':False,'runtime_seconds':time.perf_counter()-start,
                        'method':'exact complete-graph reduction; exhaustive vertex deletion',
                        'enumeration_size':combinations,'budget_seconds':None})
    path=OUTPUT/'natural_clique_certificates.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in results))
    write(OUTPUT/'natural_clique_COMPLETE.json',{'certified_pairs':len(results),'max_enumeration_size':5000000,
          'output_sha256':sha(path),'script_sha256':sha(__file__),'manifest_sha256':sha(OUTPUT/'natural_manifest.json'),'tiny_exhaustive_tests':33})
    print('Certified',len(results),'natural pairs by exact complete-graph reduction',flush=True)
if __name__=='__main__':main()
