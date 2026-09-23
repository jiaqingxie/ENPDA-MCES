"""Run the existing Sinkhorn control on isolated graph-disjoint tensors."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import train_once_sinkhorn as original
from run_enpda_graph_disjoint import prepare_items
from run_enpda_pilot import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/train_once_sinkhorn.json'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    manifest = json.loads(Path(cfg['graph_disjoint_manifest']).read_text())
    assert all(v == 0 for v in manifest['graph_intersections'].values())
    rows = {}
    for split in ('train', 'validation'):
        spec = manifest['materialized_files'][split]
        path = Path(spec['path'])
        assert sha256(path) == spec['sha256']
        rows[split] = torch.load(path, map_location='cpu', weights_only=False)
        assert all(r['pair'].true_edges is None and r['pair'].true_nodes is None
                   and r['pair'].true_similarity is None for r in rows[split])

    def isolated_items(files, device, limit, selection_seed, teachers=None):
        assert len(files) == 1
        spec = files[0]
        values = [r for r in rows[spec['materialized_split']] if r['dataset'] == spec['dataset']]
        if args.smoke and teachers is not None:
            values = [r for r in values if r['source_key'] in teachers]
        assert len(values) >= limit
        return prepare_items(values[:limit], device, teachers)

    original.load_dataset_items = isolated_items
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    started = time.perf_counter()
    original.train(args.config, args.seed, args.device, args.smoke)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    suffix = '.smoke' if args.smoke else ''
    dest = Path(cfg['outputs']['training_result_pattern'].format(seed=args.seed) + suffix)
    atomic_json(dest.with_suffix(dest.suffix + '.timing.json'), {
        'stage': 'sinkhorn_training_including_validation_and_input_preparation',
        'elapsed_seconds': time.perf_counter() - started,
        'manifest_sha256': sha256(Path(cfg['graph_disjoint_manifest'])),
        'training_pairs': len(rows['train']), 'validation_pairs': len(rows['validation']),
        'test_inputs_loaded': False, 'smoke': args.smoke,
    })


if __name__ == '__main__':
    main()
