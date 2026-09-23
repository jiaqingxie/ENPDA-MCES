#!/usr/bin/env python3
"""Score the best verified RASCAL incumbent actually discovered before 60 seconds."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

import rdkit
from reproduce_nga_rascal import molecule

HERE=Path(__file__).resolve().parent
ROOT=HERE
while not (ROOT/"pyproject.toml").exists():
    if ROOT == ROOT.parent:
        raise RuntimeError("Cannot locate the ENPDA checkout")
    ROOT = ROOT.parent
ARTIFACT=ROOT/"artifacts/rascal_nga_reproduction_20260919"
OUT=ROOT/"results/rascal_nga_reproduction_20260919/incumbent60"
ARM={"name":"v2024_smiles_incumbent60","version":"2024.03.5","representation":"smiles","complete_rings":False}
LIBRARY=str(next((Path(rdkit.__file__).parent.parent/"rdkit.libs").glob("*RascalMCES*")))
BUDGET=60.


def decode_witness(pair,event):
    """Orient an already fixed bond correspondence; no edge selection/search."""
    left,_=molecule(pair["left"],"smiles");right,_=molecule(pair["right"],"smiles")
    bp=event["bond_pairs"]
    if len(pair["left"]["nodes"])>len(pair["right"]["nodes"]):bp=[[b,a] for a,b in bp]
    witness=[]
    for a,b in bp:
        if a>=left.GetNumBonds() or b>=right.GetNumBonds():raise ValueError("bond index outside graph")
        x,y=left.GetBondWithIdx(a),right.GetBondWithIdx(b)
        if x.GetBondType()!=y.GetBondType():raise ValueError("bond label mismatch")
        witness.append([x.GetBeginAtomIdx(),x.GetEndAtomIdx(),y.GetBeginAtomIdx(),y.GetEndAtomIdx()])
    if len({tuple(sorted((u,v))) for u,v,s,t in witness})!=len(witness):raise ValueError("duplicate source edge")
    if len({tuple(sorted((s,t))) for u,v,s,t in witness})!=len(witness):raise ValueError("duplicate target edge")
    def orient(remaining,mapping,inverse):
        if not remaining:return mapping
        index=max(range(len(remaining)),key=lambda i:sum(v in mapping for v in remaining[i][:2]))
        u,v,s,t=remaining[index]
        rest=remaining[:index]+remaining[index+1:]
        for a,b in ((s,t),(t,s)):
            if pair["left"]["nodes"][u]!=pair["right"]["nodes"][a] or pair["left"]["nodes"][v]!=pair["right"]["nodes"][b]:continue
            if any(x in mapping and mapping[x]!=y or y in inverse and inverse[y]!=x for x,y in ((u,a),(v,b))):continue
            result=orient(rest,{**mapping,u:a,v:b},{**inverse,a:u,b:v})
            if result is not None:return result
        return None
    mapping=orient(witness,{},{} )
    if mapping is None:raise ValueError("bond correspondence has no injective labelled vertex witness")
    re={tuple(sorted((u,v))):lab for u,v,lab in pair["right"]["edges"]}
    implied=sum(u in mapping and v in mapping and re.get(tuple(sorted((mapping[u],mapping[v]))))==lab for u,v,lab in pair["left"]["edges"])
    return {**{k:pair[k] for k in ("dataset","key","source_path","source_sha256","true_edges")},
            "status":"ok","common_edges":len(bp),"common_nodes":len(mapping),"accuracy":len(bp)/pair["true_edges"],
            "atom_matches":sorted(mapping.items()),"bond_matches":bp,"bond_witness":witness,"mapping_common_edges":implied,
            "mapping_injective":True,"mapping_node_labels_valid":True,"witness_valid":True}


def run_pair(pair):
    base={k:pair[k] for k in ("dataset","key","source_path","source_sha256","true_edges")}
    with tempfile.TemporaryDirectory(prefix="rascal-incumbent-") as tmp:
        trace=Path(tmp)/"trace.jsonl"
        rd,wr=os.pipe();env=os.environ.copy()
        env.update({"RASCAL_REQUEST_START_FD":str(wr),"NGA_RASCAL_LIBRARY":LIBRARY,"NGA_RASCAL_TRACE":str(trace),
                    "LD_PRELOAD":str(HERE/"rascal_incumbent_trace.so")+(":"+env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")})
        proc=subprocess.Popen([sys.executable,str(HERE/"reproduce_nga_rascal.py"),"--single-pair"],
                              stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,pass_fds=(wr,))
        os.close(wr)
        try:
            proc.stdin.write(json.dumps([pair,ARM,60]));proc.stdin.close();proc.stdin=None
            with selectors.DefaultSelector() as selector:
                selector.register(rd,selectors.EVENT_READ)
                if not selector.select(timeout=30):raise RuntimeError("Worker initialization timeout")
            signal=os.read(rd,256).strip()
            if not signal:raise RuntimeError("Worker exited before timing start")
            started=float(signal);deadline=started+BUDGET
            try:
                stdout,stderr=proc.communicate(timeout=max(0.,deadline-time.perf_counter()))
                elapsed=time.perf_counter()-started
                status="returned" if proc.returncode==0 else "native_failure"
                if elapsed>BUDGET:status="late_return"
            except subprocess.TimeoutExpired:
                elapsed=time.perf_counter()-started;proc.kill();stdout,stderr=proc.communicate();status="hard_timeout"
            events=[];invalid=[]
            for line in trace.read_text().splitlines() if trace.exists() else []:
                try:event=json.loads(line)
                except json.JSONDecodeError:
                    invalid.append({"error":"incomplete trace record at termination"});continue
                event["elapsed_seconds"]=event["monotonic_ns"]/1e9-started
                events.append(event)
            eligible=[e for e in events if 0.<=e["elapsed_seconds"]<=BUDGET]
            best=None;best_time=None
            for event in eligible:
                try:witness=decode_witness(pair,event)
                except Exception as exc:
                    invalid.append({"event":event,"error":str(exc)});continue
                if best is None or witness["common_edges"]>best["common_edges"]:
                    best=witness;best_time=event["elapsed_seconds"]
            api=None
            if status=="returned":
                marker="RASCAL_RECORD_JSON="
                encoded=[line[len(marker):] for line in stdout.splitlines() if line.startswith(marker)]
                if len(encoded)==1:
                    api=json.loads(encoded[0])
                    if api.get("status")=="ok" and api.get("witness_valid") and (best is None or api["common_edges"]>best["common_edges"]):
                        best=api;best_time=elapsed
            return {**base,"status":status,"score":best["accuracy"] if best else 0.,"best_result":best,
                    "best_discovered_seconds":best_time,"supervisor_elapsed_seconds":elapsed,"returncode":proc.returncode,
                    "trace_events":events,"eligible_events":len(eligible),"invalid_trace_events":invalid,
                    "post_deadline_events_excluded":len(events)-len(eligible),"api_result":api,"stderr":stderr}
        finally:
            os.close(rd)
            if proc.poll() is None:proc.kill();proc.wait()


def main():
    assert rdkit.__version__==ARM["version"]
    pairs=json.loads((ARTIFACT/"pairs.json").read_text());OUT.mkdir(parents=True,exist_ok=True)
    protocol={"arm":ARM,"pairs":len(pairs),"workers":8,"budget_seconds":BUDGET,
              "rule":"Best verified fixed bond correspondence emitted by the unchanged RDKit incumbent-update routine before the external deadline. Partial solutions count. After-deadline discoveries never count. Zero only if no valid answer is available before the deadline.",
              "implementation":"Observer delegates to the original pinned updateMaxClique, then copies only improving accepted cliques with CLOCK_MONOTONIC timestamp. No modification to original search state, pruning, search order or scoring options.",
              "decoding":"After collection, orient fixed bond correspondences into a labelled injective vertex witness; no edges may be added or removed from a logged candidate.",
              "timing_start":"Immediately before molecular conversion and solver invocation; interpreter/import initialization excluded.",
              "input_sha256":hashlib.sha256((ARTIFACT/"pairs.json").read_bytes()).hexdigest(),
              "rdkit":rdkit.__version__,"cpu":subprocess.run(["lscpu"],capture_output=True,text=True).stdout,
              "started_utc":datetime.now(timezone.utc).isoformat(),
              "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),HERE/"reproduce_nga_rascal.py",HERE/"rascal_incumbent_trace.so")}}
    (OUT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n")
    records=[]
    with (OUT/"records.jsonl").open("x") as f,ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(run_pair,p) for p in pairs]):
            r=future.result();records.append(r);f.write(json.dumps(r)+"\n");f.flush()
            if len(records)%25==0 or len(records)==len(pairs):print(json.dumps({"completed":len(records),"total":len(pairs)}),flush=True)
    summary={}
    for d in ("AIDS","MOLHIV","MCF-7"):
        a=[r for r in records if r["dataset"]==d]
        summary[d]={"pairs":len(a),"accuracy_percent":100*sum(r["score"] for r in a)/len(a),
                    "statuses":dict(Counter(r["status"] for r in a)),"partial_solutions_at_deadline":sum(r["status"]!="returned" and r["score"]>0 for r in a),
                    "invalid_trace_events":sum(len(r["invalid_trace_events"]) for r in a)}
    final={"datasets":summary,"protocol":protocol,"finished_utc":datetime.now(timezone.utc).isoformat(),"records_sha256":hashlib.sha256((OUT/"records.jsonl").read_bytes()).hexdigest()}
    (OUT/"COMPLETE.json").write_text(json.dumps(final,indent=2)+"\n");print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
