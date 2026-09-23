"""Portable command matrix for the graph-disjoint evaluation protocol."""

def evaluation_tasks():
    tasks = []
    def add(key, deps, commands, outputs, seeds=(), hours=6, description=''):
        tasks.append(dict(key=key, dependencies=list(deps), commands=commands, expected_outputs=outputs,
                          seeds=list(seeds), max_hours=hours, description=description,
                          job_name=key))
    def cmd(script, *args):
        return ['scripts/' + script, *map(str, args)]
    dslist = [('AIDS','aids'), ('MOLHIV','mol'), ('MCF-7','mcf')]
    for seed in range(3):
        dep = [f'training:{seed}']
        checkpoint = f'checkpoints/enpda_formal/seed{seed}.best.pt'
        commands = [cmd('train_enpda_disjoint_sinkhorn.py', '--seed', seed)]
        outputs = [f'checkpoints/train_once_sinkhorn/seed{seed}.best.pt', f'results/train_once_sinkhorn/training/seed{seed}.json']
        for ds, slug in dslist:
            folder = f'results/train_once_sinkhorn/native/seed{seed}'
            commands.append(cmd('eval_train_once_sinkhorn_gpu.py', '--dataset', ds, '--training-seed', seed,
                                '--checkpoint', outputs[0], '--output', f'{folder}/{ds}.jsonl',
                                '--completion', f'{folder}/complete_{ds}.json'))
            outputs += [f'{folder}/{ds}.jsonl', f'{folder}/complete_{ds}.json']
        add(f'sink-s{seed}', dep, commands, outputs, [seed], description='Fresh Sinkhorn affinity; deterministic and Gumbel evaluations share its checkpoint')
        for ds, slug in dslist:
            prefix = f'results/enpda_formal/native/seed{seed}'
            commands, outputs = [], []
            for arm in ('core','analytic_core','solver'):
                output = f'{prefix}/enpda_{arm}_{ds}.jsonl'
                marker = f'{prefix}/complete_{arm}_{ds}.json'
                commands.append(cmd('eval_enpda_native_gpu.py', '--dataset', ds, '--training-seed', seed,
                                    '--arm', arm, '--checkpoint', checkpoint, '--workers', 4,
                                    '--output', output, '--attestation', marker))
                outputs += [output, marker]
            nativekey = f'native-{slug}-s{seed}'
            add(nativekey, dep, commands, outputs, [seed], description='Native Core, analytic Core and full Solver; accuracy, MSE, time and stream choices')
            commands, outputs = [], []
            for arm in ('analytic', 'full_no_price', 'initializer_only', 'dynamics_only', 'full'):
                folder = f'results/enpda_component_ablation_v2/seed{seed}'
                output, marker = f'{folder}/{arm}_{ds}.jsonl', f'{folder}/complete_{arm}_{ds}.json'
                commands.append(cmd('eval_enpda_component_ablation_gpu.py', '--dataset', ds, '--seed', seed,
                                    '--arm', arm, '--checkpoint', checkpoint, '--config', 'configs/enpda_component_ablation_v2.json',
                                    '--output', output, '--marker', marker))
                outputs += [output, marker]
            add(f'comp-{slug}-s{seed}', dep, commands, outputs, [seed], description='All inference component ablations at fixed four-round budget')
            commands, outputs = [], []
            for arm in ('learned','analytic'):
                folder = f'results/enpda_search_frontier/seed{seed}'
                output, marker = f'{folder}/{arm}_{ds}.jsonl', f'{folder}/complete_{arm}_{ds}.json'
                commands.append(cmd('eval_enpda_search_frontier_gpu.py', '--dataset', ds, '--seed', seed,
                                    '--arm', arm, '--checkpoint', checkpoint, '--workers', 4,
                                    '--output', output, '--marker', marker))
                outputs += [output, marker]
            add(f'search-{slug}-s{seed}', dep, commands, outputs, [seed], description='Matched search-budget frontiers for neural and analytic proposals')
            folder = f'results/enpda_rascal_60s/seed{seed}'
            add(f'cap60-{slug}-s{seed}', dep,
                [cmd('eval_enpda_rascal_60s_gpu.py', '--dataset', ds, '--seed', seed, '--checkpoint', checkpoint,
                     '--workers', 4, '--output', f'{folder}/{ds}.jsonl', '--marker', f'{folder}/complete_{ds}.json')],
                [f'{folder}/{ds}.jsonl', f'{folder}/complete_{ds}.json'], [seed], description='ENPDA under the unchanged nominal 60-second cap')
            solver = f'{prefix}/enpda_solver_{ds}.jsonl'
            folder = f'results/enpda_price_source_ablation/seed{seed}'
            commands = [cmd('eval_enpda_price_source_ablation_gpu.py', '--dataset', ds, '--seed', seed,
                            '--checkpoint', checkpoint, '--solver-records', solver,
                            '--output', f'{folder}/{ds}.jsonl', '--marker', f'{folder}/complete_{ds}.json')]
            outputs = [f'{folder}/{ds}.jsonl', f'{folder}/complete_{ds}.json']
            if seed == 0:
                folder2 = 'results/enpda_followups/price_certificate'
                commands.append(cmd('eval_enpda_price_certificate_gpu.py', '--dataset', ds, '--checkpoint', checkpoint,
                                    '--solver-records', solver, '--output', f'{folder2}/{ds}.jsonl',
                                    '--attestation', f'{folder2}/complete_{ds}.json'))
                outputs += [f'{folder2}/{ds}.jsonl', f'{folder2}/complete_{ds}.json']
            add(f'price-{slug}-s{seed}', [nativekey], commands, outputs, [seed], description='Price-source/repair comparison against the new frozen incumbent')
    allseeds = [f'training:{s}' for s in range(3)]
    add('rounds', allseeds, [cmd('eval_enpda_round_ablation_gpu.py')], ['results/enpda_round_ablation/COMPLETE.json'], range(3))
    add('round-time', allseeds, [cmd('profile_enpda_round_latency_gpu.py')], ['results/enpda_round_latency/COMPLETE.json'], range(3))
    add('imdb-ood', allseeds, [cmd('eval_enpda_nonmolecular_ood_gpu.py')], ['results/enpda_nonmolecular_ood/COMPLETE.json'], range(3))
    add('protein-ood', allseeds, [cmd('run_enpda_protein_ood.py', 'evaluate'), cmd('run_enpda_protein_ood.py', 'finalize')],
        ['results/enpda_protein_ood/summary.json'], range(3))
    add('dd-scale', [*allseeds, 'sink-s0'], [cmd('run_enpda_dd_scaling.py', 'evaluate'), cmd('run_enpda_dd_scaling.py', 'finalize')],
        ['results/enpda_dd_scaling/COMPLETE.json'], range(3))
    for suite in ('native','natural'):
        folder = f'results/submission_followups/{suite}_cuda'
        add(f'short-{suite}', allseeds,
            [cmd('run_enpda_submission_followups.py', '--suite', suite, '--device', 'cuda'),
             cmd('run_enpda_latency_matched_classical.py', '--suite', suite, '--device', 'cuda')],
            [f'{folder}/COMPLETE.json', f'{folder}/latency_matched_COMPLETE.json'], range(3),
            description='Same-host Core and classical controls, fixed and remeasured latency-matched budgets')
    return tasks
