#!/usr/bin/env python3
"""Instrument accepted upstream McSplit incumbents without changing its objective."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'artifacts/mcsplit-upstream'
COMMIT = 'a1f3e596ee8482ad332ff3de4166051338b5adaf'
BUILD = ROOT / 'artifacts/strict60_20260919/mcsplit'


def main():
    assert subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    source = UPSTREAM / 'code/james-cpp'
    BUILD.mkdir(parents=True, exist_ok=True)
    text = (source / 'mcsp.c').read_text()
    text = text.replace('static std::atomic<bool> abort_due_to_timeout;', '''
static std::atomic<bool> abort_due_to_timeout;
static std::chrono::steady_clock::time_point fractional_deadline;
static void (*trace_callback)(int, const int*) = nullptr;
static std::vector<int> trace_left, trace_right;
''')
    assert text.count('if (abort_due_to_timeout)\n') == 1
    text = text.replace('if (abort_due_to_timeout)\n',
        'if (abort_due_to_timeout || std::chrono::steady_clock::now() >= fractional_deadline)\n')
    assert text.count('incumbent = current;') == 1
    text = text.replace('incumbent = current;', '''incumbent = current;
        if (trace_callback) {
            std::vector<int> mapping(trace_left.size(), -1);
            for (auto p : incumbent) mapping[trace_left[p.v]] = trace_right[p.w];
            trace_callback(mapping.size(), mapping.data());
        }''')
    (BUILD / 'mcsp_trace.c').write_text(text)
    bridge = r'''
#define main unused_upstream_main
#include "mcsp_trace.c"
#undef main
extern "C" int run_mcsplit_trace(int n0, const int* lab0, int e0, const int* edges0,
    int n1, const int* lab1, int e1, const int* edges1,
    double seconds, void (*callback)(int, const int*)) {
    fractional_deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
    set_default_arguments(); arguments.quiet = true;
    arguments.heuristic = min_max; arguments.vertex_labelled = true; arguments.edge_labelled = true;
    abort_due_to_timeout.store(false);
    Graph g0(n0), g1(n1);
    for (int i=0;i<n0;++i) g0.label[i]=lab0[i];
    for (int i=0;i<n1;++i) g1.label[i]=lab1[i];
    for (int i=0;i<e0;++i) g0.adjmat[edges0[3*i]][edges0[3*i+1]]=g0.adjmat[edges0[3*i+1]][edges0[3*i]]=edges0[3*i+2];
    for (int i=0;i<e1;++i) g1.adjmat[edges1[3*i]][edges1[3*i+1]]=g1.adjmat[edges1[3*i+1]][edges1[3*i]]=edges1[3*i+2];
    auto d0=calculate_degrees(g0),d1=calculate_degrees(g1);
    vector<int> v0(n0),v1(n1); std::iota(v0.begin(),v0.end(),0);std::iota(v1.begin(),v1.end(),0);
    std::stable_sort(v0.begin(),v0.end(),[&](int a,int b){return d0[a]>d0[b];});
    std::stable_sort(v1.begin(),v1.end(),[&](int a,int b){return d1[a]>d1[b];});
    auto s0=induced_subgraph(g0,v0),s1=induced_subgraph(g1,v1);
    trace_left=v0;trace_right=v1;trace_callback=callback;
    auto solution=mcs(s0,s1);
    trace_callback=nullptr;
    return std::chrono::steady_clock::now() >= fractional_deadline;
}
'''
    (BUILD / 'bridge.cpp').write_text(bridge)
    subprocess.run(['g++', '-O3', '-std=c++11', '-shared', '-fPIC', '-pthread', '-I', str(source),
        str(BUILD / 'bridge.cpp'), str(source / 'graph.c'), '-o', str(BUILD / 'libmcsplit_trace.so')], check=True)
    print(BUILD / 'libmcsplit_trace.so')


if __name__ == '__main__':
    main()
