"""Post-audit, full-scope deterministic sensitivity; never replace frozen results."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from nema.association import AssociationGraph
from run_enpda_graph_disjoint import ControlledModel
from consolidate_enpda_disjoint_results import sha,score

ROOT=Path(__file__).resolve().parents[1]


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(0,1,2),required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert 'H100' in torch.cuda.get_device_name(0) and str(torch.version.cuda).startswith('12.8')
    root=ROOT/f'results/enpda_graph_disjoint_v1/consolidated/deterministic_core_sensitivity/seed{args.seed}'
    root.mkdir(parents=True,exist_ok=True)
    if (root/'COMPLETE.json').exists():raise RuntimeError('Refusing to overwrite completed sensitivity experiment')
    config=json.loads((ROOT/'configs/enpda_graph_disjoint_v1.json').read_text())
    data=torch.load(ROOT/'data/enpda_graph_disjoint_v1/test.pt',weights_only=False,map_location='cpu')
    checkpoints={a:ROOT/f'checkpoints/enpda_graph_disjoint_v1/formal/seed{args.seed}/{a}.best.pt' for a in ('full','trained_no_price')}
    models={}
    for a,path in checkpoints.items():
        payload=torch.load(path,weights_only=False,map_location='cpu')
        m=ControlledModel(**payload['model_config']);m.load_state_dict(payload['model']);models[a]=m.cuda().eval()
    provenance={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'seed':args.seed,
                'protocol':'post-audit deterministic arithmetic sensitivity, all 291 pairs and all 5 original arms; same frozen weights; no result selection',
                'code_sha256':{p:sha(ROOT/p) for p in ('scripts/audit_enpda_deterministic_core.py','scripts/run_enpda_graph_disjoint.py','src/nema/models/enpda.py','src/nema/association.py','src/nema/rounding.py')},
                'checkpoint_sha256':{a:sha(p) for a,p in checkpoints.items()},'test_sha256':sha(ROOT/'data/enpda_graph_disjoint_v1/test.pt'),
                'config_sha256':sha(ROOT/'configs/enpda_graph_disjoint_v1.json'),
                'hardware':{'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'cuda':torch.version.cuda},
                'deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'timing':'loading excluded; ACG build, synchronized model trajectory, Hungarian, hard scoring included'}
    write(root/'status.json',provenance)
    # One untimed warmup per arm, without a data-dependent choice.
    left,right,_=data[0]['pair'].oriented();warm=AssociationGraph.build(left,right)
    with torch.inference_mode():
        for model in models.values():model(warm).hard_mapping()
        with (root/'native.jsonl').open('x') as stream:
            for index,rec in enumerate(data):
                pair=rec['pair']
                for arm in config['core_evaluation_arms']:
                    model=models['trained_no_price'] if arm=='trained_no_price' else models['full']
                    model.price_enabled=arm not in ('trained_no_price','full_prices_removed')
                    model.analytic_price_enabled=arm!='analytic_no_price'
                    mode='analytic' if arm.startswith('analytic') else 'learned'
                    torch.cuda.synchronize();start=time.perf_counter()
                    left,right,swapped=pair.oriented();association=AssociationGraph.build(left,right)
                    o=model(association,mode=mode);mapping=o.hard_mapping().tolist()
                    edges,nodes=association.hard_statistics(torch.tensor(mapping))
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                    if swapped:
                        inv=[-1]*pair.left.num_nodes
                        for i,j in enumerate(mapping):
                            if j>=0:inv[j]=i
                        mapping=inv
                    assert score(pair,mapping)[:2]==(edges,nodes)
                    if arm in ('trained_no_price','full_prices_removed','analytic_no_price'):assert torch.count_nonzero(o.prices)==0
                    row={'dataset':rec['dataset'],'source_path':rec['path'],'identities':rec['identities'],'seed':args.seed,'arm':arm,
                         'mapping':mapping,'common_edges':edges,'common_nodes':nodes,'true_edges':pair.true_edges,'accuracy':edges/pair.true_edges,
                         'runtime_seconds':seconds,'prices':o.prices.cpu().tolist()}
                    stream.write(json.dumps(row)+'\n');stream.flush()
                if (index+1)%25==0:print(args.seed,index+1,len(data),flush=True)
    for p,h in provenance['code_sha256'].items():assert sha(ROOT/p)==h
    provenance.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),records=291*5,output_sha256=sha(root/'native.jsonl'))
    write(root/'COMPLETE.json',provenance);write(root/'status.json',provenance)
    print('complete',args.seed,flush=True)


if __name__=='__main__':main()
