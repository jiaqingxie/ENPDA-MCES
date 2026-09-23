#!/usr/bin/env python3
"""Combine valid MILP bounds and independent exact complete-graph reductions."""
import json
from pathlib import Path
from run_enpda_submission_followups import ROOT,OUTPUT,CONFIG,prepare,from_payload,write,sha
from nema.graph import GraphPair
from nema.classical_followups import score_mapping

def main():
    manifest=prepare(json.loads(CONFIG.read_text()))
    files=['natural_oracle.jsonl','natural_clique_certificates.jsonl']
    sources={f:{r['key']:r for r in map(json.loads,(OUTPUT/f).read_text().splitlines())} for f in files}
    merged=[]
    for item in manifest['pairs']:
        pair=GraphPair(from_payload(item['left']),from_payload(item['right']))
        records=[value[item['key']] for value in sources.values() if item['key'] in value]
        assert records
        for row in records:assert score_mapping(pair,row['mapping'])==row['lower_bound']
        best=max(records,key=lambda r:r['lower_bound']);lower=best['lower_bound'];upper=min(r['upper_bound'] for r in records)
        assert lower<=upper
        merged.append({'key':item['key'],'mapping':best['mapping'],'lower_bound':lower,'upper_bound':upper,
                       'certified_optimal':lower==upper,'timed_out':any(r['timed_out'] for r in records),
                       'methods':[r['method'] for r in records],'runtime_seconds':sum(r['runtime_seconds'] for r in records)})
    output=OUTPUT/'natural_oracle_combined.jsonl';output.write_text(''.join(json.dumps(r)+'\n' for r in merged))
    write(OUTPUT/'natural_oracle_combined_COMPLETE.json',{'pairs':len(merged),'certified_pairs':sum(r['certified_optimal'] for r in merged),
          'output_sha256':sha(output),'source_sha256':{f:sha(OUTPUT/f) for f in files},'manifest_sha256':sha(OUTPUT/'natural_manifest.json'),
          'script_sha256':sha(__file__)})
    print('Natural oracle complete:',len(merged),'pairs;',sum(r['certified_optimal'] for r in merged),'certified')
if __name__=='__main__':main()
