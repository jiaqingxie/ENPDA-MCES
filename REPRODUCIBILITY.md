# Reproduction protocol and validation

## Inputs and split

The molecular input archive is pinned to NGA commit e4a8f1f9ec9e31f79f3fbd648717dfbb9fe113fc and SHA-256 2eba0cca01584ef6cda6cb685a1f2274d98233d70ed3813979aef36827d2634a. The downloader validates the archive on first acquisition. Existing extracted inputs are hashed in the generated inventory; retain that inventory and the original archive when reporting a run.

The native test scope is AIDS 100, MOLHIV 91 and MCF-7 100. MOLHIV files 23, 46, 48, 54, 61, 64 and 76 contain empty released tensors and are excluded by the predeclared sanitation rule. SMILES recovery remains available to other protocols, but these seven pairs are not reintroduced into the native evaluation.

Graph identity means exact node- and edge-label-preserving isomorphism of model input graphs. Ordered tensor hashes cache repeated inputs; Weisfeiler-Lehman (WL) hashes only select candidates for the exact VF2 isomorphism check. The rule globally excludes all test identities before selecting training and validation pairs. It deduplicates unordered pairs, reserves deterministic disjoint validation anchors, and discards pairs crossing that boundary. The resulting pair counts are 1,343 training, 172 validation and 291 test. No optimum label or model score affects this selection. Labels describing the optimum are removed from training and validation tensors.

TU out-of-distribution (OOD) data are downloaded through PyTorch Geometric. Protocols store raw-file hashes, selected source graph identities, seeds, and generated input manifests. Retrieval excludes every exact labeled isomorphism class in the union of the three released MCES training banks, including validation inputs. It regenerates all pools and RASCAL oracle labels locally.

## Learning and evaluation

For each training seed, the graph-disjoint runner fits a fresh reference model, generates training-only teacher maps, and trains full ENPDA plus a separate no-price control. Validation hard edge gain selects checkpoints at the declared epochs and round budgets. Test tensors are first loaded after checkpoint selection. The separately trained Sinkhorn control uses the same materialized split and newly generated teachers.

Core uses four rounds and one Hungarian projection. The full Solver includes the declared neural/analytic proposal streams and discrete refinement. Inference ablations and a separately trained no-price model answer different questions; retain their distinct names. The analytic portfolio has no learned parameter dependence. Retrieval's SimGNN/GMN/NeuroMatch references are supervised, independently trained controls rather than correspondence-free ENPDA variants.

The release rebuilds the 75-pair certificate sample from numeric input ordering. It uses the empty mapping as the initial independent valid lower bound instead of reusing a historical NEMA incumbent. This leaves the sample, MILP formulation and requested budgets unchanged, but historical certificate closure counts are not assumed. Fresh legal ENPDA incumbents are joined with independently valid upper bounds after solving. Unresolved instances retain bounds; a timeout does not establish an optimum.

The native aggregator requires every expected pair for all three seeds. Its paired confidence interval resamples training seeds and connected components of the test-pair graph (pairs sharing an exact graph identity are joined transitively). Released reference labels are not silently promoted to independently proved optima; reference-normalized accuracy is not clipped at 100 percent.

## Environment and repeatability

Use the provided CPU requirements for tests/smoke checks, and the GPU Dockerfile for formal experiments. The exact recorded Torch/CUDA stack is specified in README.md. The non-Torch package pins define a reproducible release environment; they are not evidence that every historical auxiliary run used exactly the same package build. The strict native observer additionally requires the RDKit 2024.03.5 profile. All other stages use the default profile.

Seeds, input/code/configuration hashes, checkpoint selection and mappings are recorded by the experiment runners. Sparse CUDA reductions can vary numerically across launches. Hardware, driver, solver and library versions affect timings, search trajectories and some discrete ties; this release does not promise bit-identical GPU results. Record the actual environment alongside a new run.

## Release checks

Before publication, the split was regenerated without saved predictions and its ordered training, validation and test membership was checked against the current graph-disjoint protocol. CPU tests cover model equivariance, legal partial assignments, gradients, Hungarian/search behavior, tiny exact certificates, baseline adapters, aromatic and unrestricted projection, and bootstrap grouping. A seed-0 CPU smoke run executed reference training, teachers, both student arms, validation selection, checkpoint reload and test evaluation. Native McSplit adapters were rebuilt from the pinned source for the integration tests. The Sinkhorn training control also passed a CPU smoke run on the regenerated split and fresh smoke teachers.

Full three-seed GPU training, all retrieval pools, long certificates, the Docker image build and the RDKit-2024 deadline matrix were not rerun as part of packaging. The source release supplies their regeneration commands; CPU smoke success alone does not validate their final numerical results. Runtime provenance and fresh execution are required to establish a full reproduction.

## Output policy

Only source, fixed settings, tests and documentation are versioned. Datasets, generated manifests, checkpoint weights, predictions, summary tables, plots, logs, native binaries and third-party checkouts are ignored. Nothing in the repository should be replaced by a historical result to make a check pass. Use git status before publishing any local changes.
