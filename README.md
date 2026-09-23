# ENPDA

Equivariant Neural Primal-Dual Assignment for maximum common edge subgraph (MCES) matching of node- and edge-labeled graphs.

This repository contains source code, experiment configurations, tests, and commands that regenerate inputs, train models, and evaluate them. It contains **no experimental results, trained checkpoints, datasets, manuscript files, or original development history**. All generated files stay in ignored local directories.

The Python package keeps the name **nema** for checkpoint/import compatibility. The current ENPDA training entry point is **reproduce.py**, using the graph-disjoint protocol. Earlier NEMA routines are used for reference pretraining and analytic controls.

## Quick start: CPU verification

Use Python 3.12 on Linux. The optional native baselines require Git and a C++ compiler (g++).

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install --no-deps -e .
python scripts/bootstrap.py --mcsplit
python -m pytest
python reproduce.py prepare
python reproduce.py smoke --seeds 0
~~~

Preparation downloads the checksum-pinned public NGA archive. It enumerates the original input bank, checks exact label-preserving graph isomorphism, builds the final disjoint split, removes optimum labels from training/validation tensors, and independently audits the materialized tensors. It does not require a saved prediction or training run. Pickled data are loaded only from this trusted source; do not substitute untrusted pickle files.

The smoke run uses one training/validation/test pair per dataset, one reference-training epoch, and one student-training epoch. It covers teacher generation, both the full and separately trained no-price arms, validation selection, checkpoint reload, and native evaluation. It is a pipeline check, not a paper experiment.

## Formal GPU reproduction

The recorded training environment used an NVIDIA H100 and the NVIDIA PyTorch 25.02 stack (Torch 2.7.0a0+ecf3bae40a.nv25.02, CUDA 12.8). Formal entry points enforce H100/H200 and CUDA 12.8; some timing/retrieval routines specifically require H100. Use H100 for the full matrix. These checks intentionally remain enabled.

~~~bash
docker build -t enpda .
docker run --rm --gpus all --ipc=host -v "$PWD:/work" -w /work enpda bash
# Inside the container:
python -m pip install --no-deps -e .
python reproduce.py all --dry-run
python reproduce.py all
~~~

The full matrix is a substantial GPU/CPU experiment. Inspect the printed commands first. It trains seeds 0, 1, and 2 from scratch and regenerates auxiliary inputs, supervised retrieval baselines, oracle labels, and certificates. No old experimental output is downloaded. The strict 60-second and aromatic experiments use a separate RDKit profile described below; they are not included in the default all stage.

Stages can also run separately, in the order listed:

| Stage | Work and prerequisites |
|---|---|
| bootstrap | Fetch pinned NGA and McSplit source; build McSplit adapters. |
| prepare | Download inputs and generate/audit the graph-disjoint split. |
| train | Fresh reference pretraining, teacher generation, full/no-price training; all seeds by default. |
| native | Core, analytic Core and full Solver using the new checkpoints. |
| controls | Train Sinkhorn; deterministic/Gumbel evaluation, component ablations, matched search frontiers, price-source/certificate checks. Requires native outputs for price comparisons. |
| rounds | Round-prefix accuracy and latency; requires all three trained seeds. |
| baselines | Official NGA three-fit protocol and the analytic search portfolio. |
| ood | Generate IMDB/PROTEINS/ENZYMES planted inputs, natural IMDB and DD inputs; compute natural certificates, then evaluate frozen models and classical controls. Requires controls and bootstrap. |
| retrieval | Six disjoint 20-query, 500-candidate pools, fresh oracles, fresh supervised baselines, ENPDA evaluation and interval metrics. |
| certificates | Fresh 10/300/1800-second lifted MILP runs on a fixed 75-pair sample; join independent bounds with seed-0 Solver outputs. |
| summarize | Aggregate complete native outputs and compute paired component/seed bootstrap intervals. |

Each stage accepts --dry-run. Per-seed stages accept --seeds 0 (or other subsets). A successful command is recorded locally; --resume skips previously successful commands if source/configuration hashes agree. Keep their generated outputs intact. Failed commands stop the run; do not bypass provenance checks or silently mix configurations. Training and input freezing refuse to overwrite existing nonempty run directories. Use a separate checkout for an independent repeat.

## Strict deadlines and aromatic constraints

These experiments observe native RASCAL incumbent updates through a symbol-specific C++ adapter. They require the **RDKit 2024.03.5** Linux wheel, g++, nm, and a GPU. The other experiment profile pins RDKit 2026.03.5. Do not interchange these profiles or pool their timings.

~~~bash
docker build --build-arg RDKIT_VERSION=2024.3.5 -t enpda-deadline .
docker run --rm --gpus all --ipc=host -v "$PWD:/work" -w /work enpda-deadline bash
# Reuse locally trained checkpoints, not saved predictions:
python -m pip install --no-deps -e .
python reproduce.py deadlines --dry-run
python reproduce.py deadlines
python reproduce.py aromatic
~~~

The preparation script checks the pinned native symbol, builds the observer, exports all 291 sanitized inputs directly from the raw files, and fingerprints source/checkpoints. Each arm runs one query at a time with an external process-group deadline. Only feasible witnesses scored before the deadline are retained. The aromatic arm additionally applies the same complete-cycle projection to every emitted witness; it does not retrain ENPDA.

## Files and generated outputs

- src/nema/models/enpda.py: equivariant primal-dual model.
- src/nema/enpda_solver.py: inference and search portfolio.
- src/nema/association.py and certificate.py: association graph and sparse lifted bounds.
- scripts/: training, evaluation, input generation, baseline adapters and aggregation.
- configs/: fixed experiment settings; enpda_sota.json is a byte-identical compatibility alias of enpda_graph_disjoint_v1.json.
- tests/: model, feasibility, certificates, equivariance, rounding, baselines and protocol tests.
- data/, checkpoints/, results/, artifacts/, vendor/: generated locally and excluded from Git.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for protocol details and validation limits, and [THIRD_PARTY.md](THIRD_PARTY.md) for upstream sources. No author identity, contact address, pretrained model, or result table is embedded in this release.

## License

The original code in this repository is available under the [MIT License](LICENSE).
Third-party software and datasets retain their upstream licenses and notices; see
[THIRD_PARTY.md](THIRD_PARTY.md).
