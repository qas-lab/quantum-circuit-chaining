# qdev-laura

qdev-laura is a framework for benchmarking quantum circuit optimizers and studying whether chaining heterogeneous optimizers can improve over single-optimizer runs.

This repository accompanies the paper "OptChain: Enhancing Quantum Circuit Optimization through Optimizer Chaining".

## Note

The most mature, paper-facing parts of this repository are:

- single-step benchmarking of five optimizers on the project circuit suite
- a small chaining pilot with semantic equivalence verification
- GUOQ IBM-native benchmark support
- figure-generation and analysis scripts for the current paper results

Offline RL code is also present in the repository, but it remains exploratory infrastructure and future-work material rather than the main validated artifact claim.

## Supported optimizers

The main benchmarking workflow evaluates these five optimizers:

- `qiskit_ai`
- `qiskit_standard`
- `tket`
- `wisq_rules`
- `wisq_bqskit`

## Dependencies

Required:

- Python 3.12
- `uv`
- Java 21 for WISQ / GUOQ

Also needed for full cross-tool reproduction:

- git submodules initialized
- a separate `.venv-tket` environment, because PyTKET conflicts with the IBM AI transpiler dependency stack

Python requirements and tool configuration live in `pyproject.toml`.

## Installation

Clone the repository and initialize submodules:

```bash
git submodule update --init --recursive
```

Install the main Python environment:

```bash
uv sync
```

Install Java 21 for WISQ / GUOQ:

```bash
bash scripts/install_jdk21.sh
export JAVA_HOME="$HOME/.local/share/java/jdk-21.0.5+11"
export PATH="$JAVA_HOME/bin:$PATH"
java --version
```

Create the isolated TKET environment:

```bash
bash scripts/setup_tket_env.sh
```

Additional setup notes:

- WISQ / GUOQ: `docs/wisq_setup.md`
- TKET environment: `docs/tket_environment.md`
- TKET quickstart: `docs/tket_quickstart.md`

## Usage

### Quick reviewer path: regenerate figures from the released data bundle

If you already have the external data bundle extracted into `data/`, regenerate the paper figures with:

```bash
uv run python scripts/generate_paper_figures.py
```

Re-run the chain-pilot equivalence check with:

```bash
uv run python scripts/verify_chain_pilot_equivalence.py
```

Outputs of interest:

- figures: `paper/figures/`
- chain equivalence summary: `reports/chain_experiment/equivalence_check_results.json`

### Run the clean confirmatory rerun

This is the main paper-facing rerun path for optimization-only evidence on original inputs.

```bash
./scripts/run_confirmatory_rerun.sh 1
```

This workflow:

1. creates a fresh database under `data/confirmatory/`
2. imports GUOQ IBM-native circuits
3. imports Benchpress circuits after IBM-native preprocessing
4. runs the five optimizers on the original inputs
5. writes logs under `logs/confirmatory/`

After the rerun completes, regenerate the figures:

```bash
uv run python scripts/generate_paper_figures.py
```

### Run the single-step workflow manually

Import GUOQ circuits:

```bash
uv run python scripts/import_guoq_circuits.py \
  --source benchmarks/ai_transpile/qasm/guoq_ibmnew \
  --database data/trajectories.db \
  --category guoq_ibmnew
```

Import Benchpress circuits with IBM-native preprocessing:

```bash
uv run python scripts/import_benchpress_preprocessed.py \
  --database data/trajectories.db \
  --output-dir benchmarks/ai_transpile/qasm/benchpress_ibmn
```

Run the single-step optimizer sweep:

```bash
uv run python scripts/run_single_step_grid_search.py --resume
```

A paper-style clean rerun uses a narrower invocation internally:

```bash
uv run python scripts/run_single_step_grid_search.py \
  --database data/confirmatory/full_unmapped_r1.db \
  --sources guoq benchpress \
  --resume \
  --no-artifacts \
  --max-concurrent-fast 4 \
  --max-concurrent-wisq-rules 2 \
  --max-concurrent-wisq-bqskit 1
```

Synthesize trajectory rewards for downstream RL experiments:

```bash
uv run python scripts/synthesize_trajectories.py --database data/trajectories.db
```

### Other common commands

Run tests:

```bash
uv run pytest tests/
```

Run the CI-safe RL subset:

```bash
uv run pytest tests/test_rl_algorithms.py tests/test_rl_dataset.py \
  tests/test_rl_evaluation.py tests/test_rl_networks.py -v --tb=short
```

Run lint and type checks:

```bash
uv run ruff check .
uv run ty check
```

## Data

Small code and documentation live in git. Large experimental outputs are handled through a separate data bundle plus a committed checksum manifest.

Canonical input families:

- Benchpress-style repo inputs: `benchmarks/ai_transpile/qasm/`
- GUOQ IBM-native inputs: `benchmarks/ai_transpile/qasm/guoq_ibmnew/`
- Benchpress IBM-native preprocessed family: `benchmarks/ai_transpile/qasm/benchpress_ibmn/`

Primary database roles:

- paper-safe confirmatory rerun snapshot: `data/confirmatory/`
- active operational pipeline DB: `data/trajectories.db`
- derived offline-RL/training DB: `data/trajectories_combined.db`

For the full source-of-truth policy and release boundary, see:

- `docs/data_sources_of_truth.md`
- `docs/artifact_scope.md`
- `docs/dataverse_release_checklist.md`

## External data bundle

Large artifacts are intended to be distributed separately from git.

Bundle integrity is pinned in:

- `ARTIFACT_MANIFEST.sha256`

The manifest covers:

- `trajectories.db`
- `trajectories_step2.db`
- `trajectories_step3.db`
- `trajectories_combined.db`
- `data/artifacts/` transpiled-QASM tree
- `data/benchpress_circuits/` benchmark-corpus materialization

After extracting the bundle into `data/`, verify integrity with:

```bash
cd data
sha256sum -c ../ARTIFACT_MANIFEST.sha256
```

## Key scripts

Primary entry points:

- `scripts/run_confirmatory_rerun.sh` — paper-facing clean confirmatory rerun
- `scripts/run_single_step_grid_search.py` — main single-step optimizer sweep
- `scripts/import_guoq_circuits.py` — imports GUOQ inputs
- `scripts/import_benchpress_preprocessed.py` — preprocesses/imports Benchpress inputs to the IBM native gate set
- `scripts/synthesize_trajectories.py` — converts optimization runs into RL trajectories
- `scripts/generate_paper_figures.py` — regenerates paper figures from the confirmatory DB
- `scripts/verify_chain_pilot_equivalence.py` — verifies semantic equivalence for cited chain-pilot outputs
- `scripts/compare_circuits.py` — lower-level circuit comparison helper

## Repository layout

- `benchmarks/ai_transpile/` — optimizer wrappers, trajectory schema, state/reward code
- `configs/` — training and evaluation configs
- `scripts/` — import, benchmark, synthesis, analysis, and verification entry points
- `docs/` — setup docs, provenance notes, and artifact-boundary notes
- `paper/` — manuscript source and generated figures
- `reports/` — generated analysis outputs and chain-pilot results
- `data/` — local data workspace; large contents are not expected to live in git

## Known limitations

Current tool-specific limitations reflected in the repository and artifact outputs:

- `qiskit_ai` can fail on some circuits with a deterministic Rust/Python interop error (`Already borrowed`)
- some circuits using `if_else` are incompatible with `qasm2.dumps`
- WISQ does not support some custom composite gates used by several benchmark circuits
- larger circuits can trigger upstream `qiskit-ibm-transpiler` scalar/type failures

Reproducibility caveats:

- the confirmatory DB family is the paper-safe source for original-circuit optimizer evidence, but current snapshots may be partial rather than benchmark-complete
- `data/trajectories.db` is an operational DB and mixes original circuits with generated artifact circuits
- `data/trajectories_combined.db` is a derived training dataset, not the authoritative record of raw optimizer executions
- the chain pilot under `reports/chain_experiment/` is a separate experiment from the confirmatory rerun

For paper figures specifically, see `paper/figures/FIGURES_README.md`.

## Citation and contact

If you use this artifact, please cite the Harvard Dataverse dataset (`10.7910/DVN/M46YZK`) and the accompanying paper. See `CITATION.cff`.

Contact:

- Laura Baird — `lbaird@uccs.edu`

## Release notes

For the current public-release checklist, see `docs/dataverse_release_checklist.md`.

Current artifact endpoints:

- Harvard Dataverse DOI: `10.7910/DVN/M46YZK` (current record: draft)
- Draft dataset URL: `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/M46YZK&version=DRAFT`
- Public artifact repository: `https://github.com/qas-lab/quantum-circuit-chaining`

Still pending before a final public freeze:

- the final SHA-256 of the packaged archive file(s)
- the final paper citation block once the Dataverse record is published

The source artifact is licensed under the MIT License; see `LICENSE`.
