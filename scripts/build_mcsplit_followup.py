#!/usr/bin/env python3
"""Build the pinned upstream McSplit search with an in-memory, fractional-time adapter."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'artifacts/mcsplit-upstream'
COMMIT = 'a1f3e596ee8482ad332ff3de4166051338b5adaf'

def main():
    UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
    if not UPSTREAM.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', 'https://github.com/jamestrimble/ijcai2017-partitioning-common-subgraph.git', str(UPSTREAM)], check=True)
        subprocess.run(['git', '-C', str(UPSTREAM), 'sparse-checkout', 'set', 'code/james-cpp'], check=True)
        subprocess.run(['git', '-C', str(UPSTREAM), 'checkout', COMMIT], check=True)
    assert subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    source = UPSTREAM / 'code/james-cpp'
    build = ROOT / 'artifacts/mcsplit-adapter'
    build.mkdir(exist_ok=True)
    text = (source / 'mcsp.c').read_text()
    text = text.replace('static std::atomic<bool> abort_due_to_timeout;', 'static std::atomic<bool> abort_due_to_timeout;\nstatic std::chrono::steady_clock::time_point fractional_deadline;')
    text = text.replace('if (abort_due_to_timeout)\n', 'if (abort_due_to_timeout || std::chrono::steady_clock::now() >= fractional_deadline)\n')
    (build / 'mcsp_fractional.c').write_text(text)
    bridge = r'''
#define main unused_upstream_main
#include "mcsp_fractional.c"
#undef main
extern "C" int run_mcsplit(int n0, const int* lab0, int e0, const int* edges0,
                           int n1, const int* lab1, int e1, const int* edges1,
                           double seconds, int* output) {
    fractional_deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
    set_default_arguments(); arguments.quiet = true;
    arguments.heuristic = min_max; arguments.vertex_labelled = true; arguments.edge_labelled = true;
    abort_due_to_timeout.store(false);
    Graph g0(n0), g1(n1);
    for (int i=0;i<n0;++i) {g0.label[i]=lab0[i];output[i]=-1;}
    for (int i=0;i<n1;++i) g1.label[i]=lab1[i];
    for (int i=0;i<e0;++i) g0.adjmat[edges0[3*i]][edges0[3*i+1]]=g0.adjmat[edges0[3*i+1]][edges0[3*i]]=edges0[3*i+2];
    for (int i=0;i<e1;++i) g1.adjmat[edges1[3*i]][edges1[3*i+1]]=g1.adjmat[edges1[3*i+1]][edges1[3*i]]=edges1[3*i+2];
    auto d0=calculate_degrees(g0), d1=calculate_degrees(g1);
    vector<int> v0(n0),v1(n1); std::iota(v0.begin(),v0.end(),0); std::iota(v1.begin(),v1.end(),0);
    std::stable_sort(v0.begin(),v0.end(),[&](int a,int b){return d0[a]>d0[b];});
    std::stable_sort(v1.begin(),v1.end(),[&](int a,int b){return d1[a]>d1[b];});
    auto s0=induced_subgraph(g0,v0),s1=induced_subgraph(g1,v1);
    auto solution=mcs(s0,s1);
    for(auto p:solution) output[v0[p.v]]=v1[p.w];
    return std::chrono::steady_clock::now() >= fractional_deadline;
}
'''
    (build / 'bridge.cpp').write_text(bridge)
    subprocess.run(['g++','-O3','-std=c++11','-shared','-fPIC','-pthread','-I',str(source),str(build/'bridge.cpp'),str(source/'graph.c'),'-o',str(build/'libmcsplit.so')],check=True)
    print(build / 'libmcsplit.so')

if __name__ == '__main__': main()
