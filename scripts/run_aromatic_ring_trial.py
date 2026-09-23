#!/usr/bin/env python3
"""Frozen-checkpoint, complete-cycle projection pilot with an external deadline."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_policy(pair, rule):
    if rule == 'unrestricted':
        from nema.unrestricted_projection import UnrestrictedProjection
        return UnrestrictedProjection(pair)
    assert rule == 'complete_aromatic_cycles', rule
    from nema.aromatic_projection import AromaticProjection
    return AromaticProjection(pair)


class RingJournal:
    def __init__(self, pair, path, started, budget, rule='complete_aromatic_cycles'):
        self.started, self.budget = started, budget
        self.policy = output_policy(pair, rule)
        self.stream = Path(path).open('a', buffering=1)
        self.best = (-1, -1)
        self.events = self.late_candidates = 0

    def elapsed(self):
        return time.perf_counter() - self.started

    def expired(self):
        return self.elapsed() >= self.budget

    def publish(self, mapping, source, allowed_edges=None):
        if hasattr(mapping, 'detach'):
            mapping = mapping.detach().cpu().long().tolist()
        result = self.policy.project(mapping, allowed_edges)
        completed = self.elapsed()
        if completed > self.budget:
            self.late_candidates += 1
            return False
        stats = (result['common_edges'], result['common_nodes'])
        if stats > self.best:
            self.stream.write(json.dumps({**result, 'source': source,
                'completed_at_seconds': completed}) + '\n')
            self.stream.flush()
            self.best = stats; self.events += 1
        return True


def native_child():
    # Library/import startup excluded; wait for a timed request from the parent.
    from reproduce_nga_rascal import solve
    print('RING_NATIVE_READY', flush=True)
    payload = json.loads(sys.stdin.readline())
    print('RING_NATIVE_RESULT=' + json.dumps(solve(payload)), flush=True)


def worker(payload):
    import pickle
    import random
    import numpy as np
    import rdkit
    import torch
    from torch.torch_version import TorchVersion
    from nema.aromatic_projection import AromaticProjection
    from nema.deadline_benchmark import enpda, nga
    from nema.official_nga import _official_modules
    from nema.data import load_pair
    from nema.models.enpda import ENPDAModel
    from reproduce_nga_rascal_incumbent60 import decode_witness
    assert rdkit.__version__ == '2024.03.5', rdkit.__version__
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    seed = payload['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    raw = payload['pair']; method = payload['method']
    child = None
    if method.startswith('nga'):
        _official_modules(payload['vendor_root'])
        with (Path(payload['root']) / raw['source_path']).open('rb') as stream:
            left, right = pickle.load(stream)
        assert len(left) == len(right) == 1
        data_s, data_t = left[0], right[0]
        assert torch.cuda.is_available()
        torch.zeros(1, device='cuda').add_(1); torch.cuda.synchronize()
    elif method == 'enpda':
        pair = load_pair(Path(payload['root']) / raw['source_path'])
        assert torch.cuda.is_available()
        torch.serialization.add_safe_globals([TorchVersion])
        checkpoint = torch.load(payload['checkpoint'], map_location='cpu', weights_only=True)
        model = ENPDAModel(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model'], strict=True)
        model = model.cuda().eval()
        torch.zeros(1, device='cuda').add_(1); torch.cuda.synchronize()
    else:
        native_trace = Path(payload['trace_path']).with_suffix('.native.jsonl')
        env = os.environ.copy()
        lib = next((Path(rdkit.__file__).parent.parent / 'rdkit.libs').glob('*RascalMCES*'))
        env.update(NGA_RASCAL_LIBRARY=str(lib), NGA_RASCAL_TRACE=str(native_trace),
                   LD_PRELOAD=str(Path(__file__).parent / 'rascal_incumbent_trace.so'))
        # The supervisor kills this entire process group at the deadline.
        child = subprocess.Popen([sys.executable, __file__, '--native-child'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env)
        assert child.stdout.readline().strip() == 'RING_NATIVE_READY'
    started = time.perf_counter()
    os.write(int(os.environ['RING_READY_FD']), repr(started).encode())
    os.close(int(os.environ['RING_READY_FD']))
    journal = RingJournal(raw, payload['trace_path'], started, payload['budget'],
                          payload.get('output_rule', 'complete_aromatic_cycles'))
    journal.publish([-1] * len(raw['left']['nodes']), 'empty')
    if method.startswith('nga'):
        details = nga(data_s, data_t, journal, seed,
                      decode_every_epoch=method == 'nga_anytime',
                      vendor_root=payload['vendor_root'])
    elif method == 'enpda':
        details = enpda(pair, model, journal, seed + 10007 * payload['pair_index'],
                       'enpda_solver', rotation_seed=seed)
    else:
        arm = {'name': method + '_online_cycle_projection', 'representation': 'smiles',
               'complete_rings': method == 'rascal', 'version': '2024.03.5'}
        child.stdin.write(json.dumps([raw, arm, 60]) + '\n')
        child.stdin.close(); child.stdin = None
        offset = 0; buffered = ''; count = 0
        def consume():
            nonlocal offset, buffered, count
            if not native_trace.exists():
                return
            with native_trace.open() as stream:
                stream.seek(offset); buffered += stream.read(); offset = stream.tell()
            lines = buffered.split('\n'); buffered = lines.pop()
            for line in lines:
                if not line or journal.expired():
                    continue
                event = json.loads(line)
                if event['monotonic_ns'] / 1e9 - started > payload['budget']:
                    continue
                fixed = decode_witness(raw, event)
                mapping = [-1] * len(raw['left']['nodes'])
                for u, v in fixed['atom_matches']:
                    mapping[u] = v
                allowed = {tuple(sorted((u, v))) for u, v, a, b in fixed['bond_witness']}
                journal.publish(mapping, 'rascal_accepted_clique', allowed)
                count += 1
        while not journal.expired():
            consume()
            if child.poll() is not None:
                stdout, stderr = child.communicate()
                assert child.returncode == 0, stderr
                consume()
                for line in stdout.splitlines():
                    if line.startswith('RING_NATIVE_RESULT='):
                        result = json.loads(line.split('=', 1)[1])
                        assert result['status'] in ('ok', 'empty'), result
                        if result['status'] == 'ok':
                            mapping = [-1] * len(raw['left']['nodes'])
                            for u, v in result['atom_matches']:
                                mapping[u] = v
                            allowed = {tuple(sorted((u, v))) for u, v, a, b in result['bond_witness']}
                            journal.publish(mapping, 'rascal_api_return', allowed)
                break
            time.sleep(.005)
        if child.poll() is None:
            child.kill(); child.communicate()
        details = {'native_candidates_decoded': count}
    print('RING_RESULT=' + json.dumps({'details': details, 'elapsed': journal.elapsed()}), flush=True)


def run_pair(pair, payload):
    policy = output_policy(pair, payload.get('output_rule', 'complete_aromatic_cycles'))
    with tempfile.TemporaryDirectory(prefix='enpda-ring-trial-') as tmp:
        trace = Path(tmp) / 'events.jsonl'
        rd, wr = os.pipe()
        env = os.environ.copy(); env['RING_READY_FD'] = str(wr)
        proc = subprocess.Popen([sys.executable, __file__, '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, pass_fds=(wr,), start_new_session=True)
        os.close(wr)
        try:
            proc.stdin.write(json.dumps({**payload, 'pair': pair, 'trace_path': str(trace)}))
            proc.stdin.close(); proc.stdin = None
            with selectors.DefaultSelector() as selector:
                selector.register(rd, selectors.EVENT_READ)
                assert selector.select(timeout=120), 'worker initialization timeout'
            ready = os.read(rd, 256)
            if not ready:
                stdout, stderr = proc.communicate()
                raise RuntimeError('worker failed before start: ' + stderr[-6000:])
            started = float(ready)
            try:
                stdout, stderr = proc.communicate(timeout=max(0., started + payload['budget'] - time.perf_counter()))
                status = 'returned' if proc.returncode == 0 else 'worker_failure'
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate(); status = 'hard_timeout'
            elapsed = time.perf_counter() - started
            events = []; truncated = 0
            for line in trace.read_text().splitlines() if trace.exists() else []:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    truncated += 1; continue
                policy.verify(event)
                if 0 <= event['completed_at_seconds'] <= payload['budget']:
                    events.append(event)
            best = max(events, key=lambda e: (e['common_edges'], e['common_nodes']),
                default={**policy.project([-1] * len(pair['left']['nodes'])),
                         'completed_at_seconds': 0., 'source': 'empty'})
            return {**{k: pair[k] for k in ('dataset', 'key', 'source_path', 'source_sha256', 'true_edges')},
                'method': payload['method'], 'seed': payload['seed'], 'status': status,
                'budget_seconds': payload['budget'], 'supervisor_elapsed_seconds': elapsed,
                'incumbent': best, 'events': events, 'truncated_trace_lines': truncated,
                'checkpoint_sha256': payload['checkpoint_sha256'], 'protocol_sha256': payload['protocol_sha256'],
                'stderr': stderr[-6000:], 'stdout': stdout[-6000:]}
        finally:
            os.close(rd)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-child', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--budget', type=float, default=60.)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--methods', nargs='+',
                        choices=('enpda', 'rascal', 'rascal_unfiltered', 'nga', 'nga_anytime'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.native_child:
        native_child(); return
    if args.worker:
        worker(json.load(sys.stdin)); return
    artifact = Path(__file__).resolve().parent
    config = json.loads((artifact / 'protocol.json').read_text())
    methods = args.methods or config.get('methods', ['enpda', 'rascal', 'rascal_unfiltered'])
    if any(method.startswith('nga') for method in methods):
        assert args.workers == 1, 'NGA receives an unshared GPU, one query at a time'
        assert args.seed in config['nga']['seeds']
    for name, expected in config['code_sha256'].items():
        assert sha(artifact / name) == expected, name
    assert sha(artifact / 'pairs.json') == config['pairs_sha256']
    pairs = json.loads((artifact / 'pairs.json').read_text())
    assert 0 <= args.shard_index < args.num_shards
    indexed_pairs = [(i, p) for i, p in enumerate(pairs) if i % args.num_shards == args.shard_index]
    if args.limit:
        indexed_pairs = indexed_pairs[:args.limit]
    pairs = [p for i, p in indexed_pairs]
    checkpoint_seed = args.seed if isinstance(config['checkpoint_sha256'], dict) else 0
    checkpoint = args.root / f'checkpoints/enpda_graph_disjoint_v1/formal/seed{checkpoint_seed}/full.best.pt'
    expected_checkpoint = (config['checkpoint_sha256'][str(args.seed)]
                           if isinstance(config['checkpoint_sha256'], dict) else config['checkpoint_sha256'])
    assert sha(checkpoint) == expected_checkpoint
    if config.get('output_rule') == 'unrestricted':
        assert set(methods) <= {'enpda', 'rascal_unfiltered'}
        assert args.workers == 1 and args.budget == 60
    args.output.mkdir(parents=True, exist_ok=True)
    import rdkit, torch
    fingerprint = {'created_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
        'host': os.uname().nodename, 'cpu': subprocess.check_output(['lscpu'], text=True),
        'gpu': subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout,
        'rdkit': rdkit.__version__, 'torch': torch.__version__, 'workers': args.workers,
        'protocol_sha256': sha(artifact / 'protocol.json')}
    (args.output / 'execution.json').write_text(json.dumps(fingerprint, indent=2) + '\n')
    for method in methods:
        target = args.output / (method + '.jsonl')
        rows = [json.loads(s) for s in target.read_text().splitlines()] if target.exists() else []
        seen = {r['source_path'] for r in rows}
        assert len(seen) == len(rows)
        payload = {'root': str(args.root), 'method': method, 'seed': args.seed, 'budget': args.budget,
            'output_rule': config.get('output_rule', 'complete_aromatic_cycles'),
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
            'vendor_root': str(artifact / 'vendor/nga-official'),
            'protocol_sha256': sha(artifact / 'protocol.json')}
        assert all(r['protocol_sha256'] == payload['protocol_sha256'] and
                   r['budget_seconds'] == args.budget and r['seed'] == args.seed and
                   r['method'] == method for r in rows)
        with target.open('a', buffering=1) as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_pair, pair, {**payload, 'pair_index': index})
                       for index, pair in indexed_pairs if pair['source_path'] not in seen]
            for future in as_completed(futures):
                row = future.result(); rows.append(row)
                stream.write(json.dumps(row) + '\n'); stream.flush()
                print(json.dumps({'method': method, 'completed': len(rows), 'total': len(pairs),
                    'dataset': row['dataset'], 'key': row['key'], 'status': row['status'],
                    'edges': row['incumbent']['common_edges']}), flush=True)
                assert row['status'] != 'worker_failure', row['stderr']
        assert len(rows) == len(pairs)
        assert {r['source_path'] for r in rows} == {p['source_path'] for p in pairs}
        (args.output / (method + '.COMPLETE.json')).write_text(json.dumps({
            'pairs': len(rows), 'records_sha256': sha(target), 'protocol_sha256': payload['protocol_sha256']}, indent=2) + '\n')


if __name__ == '__main__':
    main()
