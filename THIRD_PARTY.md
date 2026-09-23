# Third-party sources

Third-party source is fetched into ignored local directories. Preserve each upstream repository's license and notices; this release does not relicense those projects.

| Dependency | Source and pin | Purpose |
|---|---|---|
| NGA | https://github.com/LOGO-CUHKSZ/NGA, commit e4a8f1f9ec9e31f79f3fbd648717dfbb9fe113fc | Official implementation and public molecular pair archive. |
| McSplit | https://github.com/jamestrimble/ijcai2017-partitioning-common-subgraph, commit a1f3e596ee8482ad332ff3de4166051338b5adaf | Classical labeled common-subgraph baseline; adapters are built locally. |
| RDKit / RASCAL | https://github.com/rdkit/rdkit | Molecular graph handling and RASCAL oracle/baseline; version-specific profiles in README.md. |
| TU datasets | https://chrsmrrs.github.io/datasets/ | IMDB-BINARY, PROTEINS, ENZYMES and DD inputs, downloaded through PyTorch Geometric. |
| PyTorch Geometric | https://github.com/pyg-team/pytorch_geometric | Graph data objects and TU download utilities. |

Additional Python dependencies and exact release pins are listed in requirements-common.txt and requirements-cpu.txt. Upstream project references and third-party attribution are retained; personal author/contact information from the development workspace is not included.
