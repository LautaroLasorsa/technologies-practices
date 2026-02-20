# Practice 068 — DVC: Data & Pipeline Versioning

## Technologies

- **DVC** (v3.55+) — Data Version Control: Git-like versioning for data, models, and ML pipelines
- **scikit-learn** — Simple ML model (LogisticRegression) to demonstrate pipeline stages
- **pandas** — Data loading and manipulation
- **PyYAML** — Reading params.yaml configuration files
- **joblib** — Model serialization (bundled with scikit-learn)

## Stack

Python 3.11+ (uv). No Docker needed -- DVC works entirely locally.

## Theoretical Context

### What DVC Is & The Problem It Solves

Git is designed for source code: small text files where diffs are meaningful. But ML projects involve **large binary files** (datasets, model weights, embeddings) that break Git in several ways:

1. **Repository bloat**: A 500MB dataset committed to Git lives in `.git/` forever, even if deleted. Cloning becomes slow.
2. **No meaningful diffs**: `git diff` on a binary file is useless.
3. **No pipeline reproducibility**: Running `train.py` after `preprocess.py` with specific parameters is a manual, error-prone process.
4. **No experiment tracking**: "Which hyperparameters produced which accuracy?" requires manual notes.

**DVC solves all four problems** by sitting alongside Git as a complementary version control system:

- **Data versioning**: Large files are stored in a separate cache/remote, while tiny `.dvc` pointer files are committed to Git.
- **Pipeline definition**: `dvc.yaml` declares stages with deps/outs/params, forming a reproducible DAG.
- **Smart caching**: `dvc repro` only re-runs stages whose dependencies actually changed.
- **Experiment tracking**: `dvc metrics diff` and `dvc params diff` compare results across commits/branches.

### How DVC Works Internally

#### Content-Addressable Storage

DVC uses the same concept as Git's object store: files are identified by their **hash** (MD5 by default), not their name. When you run `dvc add data.csv`:

1. DVC computes the MD5 hash of `data.csv` (e.g., `ec1d2935f811b77cc49b031b999cbf17`)
2. The file is copied into the **DVC cache** at `.dvc/cache/files/md5/ec/1d2935f811b77cc49b031b999cbf17` (first 2 chars = directory, rest = filename)
3. A small **pointer file** `data.csv.dvc` is created containing the hash, file size, and path
4. `data.csv` is added to `.gitignore` (so Git ignores the large file)
5. You commit `data.csv.dvc` + `.gitignore` to Git

The pointer file is ~100 bytes regardless of whether the actual data is 1KB or 10GB. Git tracks the pointer; DVC manages the actual data.

#### File Linking Strategies

When DVC needs to place a cached file in your workspace, it uses one of these strategies (in preference order):

| Strategy | How it works | Disk usage | Availability |
|----------|-------------|------------|--------------|
| **Reflink** (default) | Copy-on-write: file appears copied but shares disk blocks until modified | 1x | Btrfs, XFS, APFS only |
| **Hardlink** | Two directory entries pointing to same inode | 1x | Same partition only |
| **Symlink** | Symbolic link to cache | 1x | Most systems |
| **Copy** | Full duplicate | 2x | All systems (fallback) |

On Windows with NTFS, DVC typically uses **copies** as the fallback. This is fine for small-to-medium datasets.

#### .dvc Pointer Files

A `.dvc` file looks like this:

```yaml
outs:
- md5: ec1d2935f811b77cc49b031b999cbf17
  size: 51820
  hash: md5
  path: customers.csv
```

This is all Git needs to track. The actual 50KB CSV lives in the DVC cache and remote.

For directories, DVC creates a `.dir` manifest (JSON array mapping each file to its hash), and the `.dvc` file points to this manifest.

### How DVC Integrates with Git

DVC is designed as a **Git companion**, not a replacement:

```
Git tracks:                    DVC manages:
  - Source code (.py)            - Large data files (.csv, .parquet)
  - dvc.yaml (pipeline def)     - Model artifacts (.pkl, .pt)
  - dvc.lock (hash lockfile)    - DVC cache (.dvc/cache/)
  - params.yaml (config)        - DVC remote (cloud/local storage)
  - metrics/*.json (results)
  - *.dvc (pointer files)
  - .dvcignore
```

The workflow is always: `dvc add/repro` first, then `git add` + `git commit`. Every Git commit captures a snapshot of both code AND data (via pointer files).

### Key Concepts

| Concept | Description |
|---------|-------------|
| **`.dvc` file** | Pointer file containing the hash of a tracked data file/directory. Committed to Git; maps to actual data in the DVC cache. |
| **DVC cache** | Local content-addressable store (`.dvc/cache/`). Holds all versions of all tracked files, identified by hash. |
| **DVC remote** | External storage (S3, GCS, Azure, SSH, or local directory) for sharing/backing up cached data. Like Git's `origin` but for data. |
| **`dvc.yaml`** | Pipeline definition file. Declares stages with `cmd`, `deps`, `outs`, `params`, and `metrics`. Forms a DAG. |
| **`dvc.lock`** | Auto-generated lockfile recording the exact hashes of all deps/outs/params for each stage. Committed to Git for reproducibility. |
| **`params.yaml`** | Default parameters file. Stages can depend on specific keys; DVC detects when values change. |
| **Stage** | A single step in the pipeline: a command with declared dependencies and outputs. |
| **Pipeline DAG** | Directed acyclic graph of stages connected by dependency chains (output of stage A is input of stage B). |
| **`dvc repro`** | Reproduce the pipeline. Runs only stages whose deps/params/code changed since the last run. |
| **`dvc checkout`** | Sync workspace data files with the `.dvc` pointer files in the current Git commit. The data equivalent of `git checkout`. |
| **`.dvcignore`** | Like `.gitignore` but for DVC. Tells DVC to skip certain files when scanning directories. |

### Pipeline DAG: dvc.yaml Stages

A `dvc.yaml` file defines the pipeline as a set of stages:

```yaml
stages:
  prepare:
    cmd: python src/01_prepare.py
    deps:
      - data/raw/customers.csv
      - src/01_prepare.py
    params:
      - prepare.test_ratio
      - prepare.seed
    outs:
      - data/prepared/train.csv
      - data/prepared/test.csv

  featurize:
    cmd: python src/02_featurize.py
    deps:
      - data/prepared/train.csv
      - data/prepared/test.csv
      - src/02_featurize.py
    params:
      - featurize.features
    outs:
      - data/prepared/train_featurized.csv
      - data/prepared/test_featurized.csv

  train:
    cmd: python src/03_train.py
    deps:
      - data/prepared/train_featurized.csv
      - src/03_train.py
    params:
      - train.C
      - train.max_iter
      - train.solver
    outs:
      - models/model.pkl

  evaluate:
    cmd: python src/04_evaluate.py
    deps:
      - models/model.pkl
      - data/prepared/test_featurized.csv
      - src/04_evaluate.py
    metrics:
      - metrics/scores.json:
          cache: false
```

DVC builds a DAG from these declarations:

```
prepare --> featurize --> train --> evaluate
                    \                /
                     +----(test)---+
```

Each stage only re-runs if its `deps`, `params`, or source `cmd` changed. This is computed by comparing hashes in `dvc.lock`.

### dvc repro and Smart Caching

When you run `dvc repro`, DVC:

1. Reads `dvc.yaml` and builds the DAG
2. For each stage (topological order), compares the current hash of every dep/param/cmd against `dvc.lock`
3. If ALL hashes match -> **skip** (cached, no work needed)
4. If ANY hash differs -> **re-run** the stage and all downstream stages

This means:
- Change `prepare.test_ratio` in `params.yaml` -> re-runs: prepare, featurize, train, evaluate (full cascade)
- Change `train.C` in `params.yaml` -> re-runs: train, evaluate only (prepare + featurize cached)
- Change nothing -> `dvc repro` completes instantly ("Pipeline is up to date")

The `--force` flag (`dvc repro -f`) bypasses caching and re-runs everything.

### Local Remote vs Cloud Remote

A DVC remote is just a storage location for cached data. It can be:

- **Local directory**: `dvc remote add -d myremote /path/to/storage` (used in this practice)
- **AWS S3**: `dvc remote add -d myremote s3://mybucket/dvc`
- **Google Cloud Storage**: `dvc remote add -d myremote gs://mybucket/dvc`
- **Azure Blob**: `dvc remote add -d myremote azure://mycontainer/dvc`
- **SSH**: `dvc remote add -d myremote ssh://user@host/path`

A local remote is perfect for learning -- it's just another directory on your filesystem that acts as the "cloud." The `-d` flag makes it the default remote for `dvc push` and `dvc pull`.

### Ecosystem Context

| Tool | Approach | Best for |
|------|----------|----------|
| **DVC** | Git companion, pointer files, CLI pipelines | Individual/small-team ML projects, reproducible pipelines |
| **Git LFS** | Git extension, tracks large files in Git workflow | Large files in non-ML projects (game assets, media) |
| **lakeFS** | Git-like branching for data lakes (S3/GCS) | Enterprise data platforms, multi-TB datasets |
| **MLflow** | Experiment tracking, model registry, serving | Experiment logging, model deployment |
| **Weights & Biases** | Cloud experiment tracking, visualization | Collaborative ML research, hyperparameter sweeps |

**DVC vs Git LFS**: Git LFS extends Git to handle large files but offers no pipeline management, experiment tracking, or selective downloads. DVC was purpose-built for ML workflows. DVC pointer files are also more flexible -- they work with any storage backend, while LFS requires an LFS server.

**DVC + MLflow/W&B**: DVC handles data/pipeline versioning; MLflow/W&B handle experiment visualization and model serving. They're complementary -- many teams use DVC for reproducibility + MLflow for the experiment UI.

**November 2025 acquisition**: lakeFS (by Treeverse) acquired the DVC open-source project from Iterative.ai. DVC remains open-source with its own brand and community. lakeFS serves enterprise-scale data versioning while DVC continues to serve individual and small-team data science projects. The acquisition signals consolidation in the data version control space.

## Description

Build a complete DVC-managed ML pipeline from scratch. You will generate a synthetic dataset, track it with DVC, define a 4-stage pipeline (prepare -> featurize -> train -> evaluate), run experiments by changing hyperparameters, and compare results across Git commits using DVC's metrics/params diff tools.

### What you'll learn

1. **DVC initialization and data tracking** -- `dvc init`, `dvc add`, understanding pointer files
2. **Remote configuration** -- setting up a local remote, `dvc push`, `dvc pull`
3. **Pipeline definition** -- writing `dvc.yaml` with stages, deps, outs, params, metrics
4. **Reproducibility** -- `dvc repro` and smart caching (only changed stages re-run)
5. **Version switching** -- `git checkout` + `dvc checkout` to restore previous data/model versions
6. **Experiment comparison** -- `dvc metrics diff`, `dvc params diff` across commits

## Instructions

### Phase 1: Setup & Data Tracking (~15 min)

This phase covers DVC initialization, data generation, tracking a file with DVC, and configuring a local remote. These are the foundational operations -- everything else builds on them.

1. Install dependencies: `uv sync`

2. **Initialize DVC** in the practice directory:
   ```
   cd practice_068_dvc
   dvc init
   ```
   This creates the `.dvc/` directory (config, cache, tmp). DVC is now active in this folder. Note: DVC requires a Git repository. Since the parent `technologies_practices` folder is already a Git repo, `dvc init` works from this subfolder.

3. **Generate the synthetic dataset**:
   ```
   uv run python src/00_generate_data.py
   ```
   This creates `data/raw/customers.csv` (1000 rows, 6 columns).

4. **Track the dataset with DVC**:
   ```
   dvc add data/raw/customers.csv
   ```
   Observe what happens:
   - `data/raw/customers.csv.dvc` is created (pointer file)
   - `data/raw/.gitignore` is created/updated (tells Git to ignore `customers.csv`)
   - The actual CSV is copied into `.dvc/cache/files/md5/XX/...`

   Open `data/raw/customers.csv.dvc` and inspect its contents -- you'll see the MD5 hash, file size, and path. This is what Git tracks instead of the large file.

5. **Configure a local DVC remote** (simulates cloud storage):
   ```
   dvc remote add -d local_storage tmp_remote_storage
   ```
   The `-d` flag makes this the default remote. The path is relative to the practice directory.

6. **Push data to the remote**:
   ```
   dvc push
   ```
   Inspect `tmp_remote_storage/` -- you'll see the same content-addressable structure as `.dvc/cache/`.

7. **Commit everything to Git**:
   ```
   git add data/raw/customers.csv.dvc data/raw/.gitignore .dvc/ .dvcignore
   git commit -m "Track raw dataset with DVC"
   ```
   Note: you commit the `.dvc` pointer file, NOT the actual CSV.

8. **Test the pull workflow** -- delete the local cache and data, then restore:
   ```
   rm -rf .dvc/cache
   rm data/raw/customers.csv
   dvc pull
   ```
   The CSV is restored from the remote. This proves the remote is working and you can always recover your data.

### Phase 2: Pipeline Definition (~20 min)

This phase builds the DVC pipeline. You will create `params.yaml`, write `dvc.yaml`, and implement the four pipeline stages. The pipeline scripts are in `src/` -- they are plain Python scripts that know nothing about DVC.

1. **Create `params.yaml`** at the practice root:
   ```yaml
   prepare:
     test_ratio: 0.2
     seed: 42

   featurize:
     features:
       - age
       - tenure_months
       - monthly_charge
       - support_calls
       - usage_hours

   train:
     C: 1.0
     max_iter: 100
     solver: lbfgs
   ```
   This file is the single source of truth for all hyperparameters. DVC stages declare which params they depend on.

2. **Create `dvc.yaml`** at the practice root. Define four stages: prepare, featurize, train, evaluate. Each stage has:
   - `cmd`: the Python command to run
   - `deps`: input files the stage reads (including the source script itself!)
   - `params`: keys from params.yaml this stage depends on
   - `outs`: output files the stage produces
   - `metrics` (evaluate only): metric files with `cache: false`

   See the Theoretical Context section for the exact `dvc.yaml` structure. Type it out yourself -- understanding the dependency declarations is the core learning here.

3. **Implement the pipeline stages** (the TODO(human) blocks):
   - `src/01_prepare.py` -- split raw data into train/test
   - `src/02_featurize.py` -- scale features with StandardScaler
   - `src/03_train.py` -- train LogisticRegression, save model.pkl
   - `src/04_evaluate.py` -- compute metrics, write scores.json

4. **Run the full pipeline**:
   ```
   dvc repro
   ```
   DVC reads `dvc.yaml`, builds the DAG, and runs all four stages in order. Watch the output -- it shows each stage executing.

5. **Commit the pipeline and results**:
   ```
   git add dvc.yaml dvc.lock params.yaml metrics/scores.json src/
   git commit -m "Add ML pipeline with baseline params"
   git tag v1-baseline
   ```
   Tag this as the baseline for later comparison.

### Phase 3: Reproducibility & Smart Caching (~15 min)

This phase demonstrates DVC's killer feature: only re-running what changed.

1. **Run `dvc repro` again** without changing anything:
   ```
   dvc repro
   ```
   Expected output: "Stage 'prepare' didn't change, skipping" (and same for all stages). The entire pipeline is cached.

2. **Change a downstream parameter** -- edit `params.yaml`:
   ```yaml
   train:
     C: 10.0        # was 1.0
     max_iter: 200   # was 100
     solver: lbfgs
   ```

3. **Re-run the pipeline**:
   ```
   dvc repro
   ```
   Observe: **prepare and featurize are SKIPPED** (their deps/params didn't change). Only **train and evaluate re-run**. This is smart caching.

4. **Compare metrics against the baseline**:
   ```
   dvc metrics diff
   ```
   This shows how each metric changed compared to the last committed version.

5. **Compare params**:
   ```
   dvc params diff
   ```
   Shows exactly which parameters you changed.

6. **Commit the experiment**:
   ```
   git add dvc.lock params.yaml metrics/scores.json
   git commit -m "Experiment: increase C to 10.0, max_iter to 200"
   git tag v2-high-C
   ```

7. **Push updated data to remote**:
   ```
   dvc push
   ```

### Phase 4: Version Switching (~15 min)

This phase shows how `git checkout` + `dvc checkout` lets you switch between complete experiment snapshots (code + data + model).

1. **Check current state**:
   ```
   dvc metrics show
   cat params.yaml
   ```
   You should see the v2-high-C metrics.

2. **Switch to the baseline**:
   ```
   git checkout v1-baseline -- dvc.lock params.yaml metrics/scores.json
   dvc checkout
   ```
   The first command restores the Git-tracked files (lockfile, params, metrics) from the v1 tag. The second command restores the DVC-tracked files (data, model) by looking up the hashes in the restored `dvc.lock`.

3. **Verify the rollback**:
   ```
   dvc metrics show
   cat params.yaml
   ```
   You should see the v1-baseline metrics and params (C=1.0).

4. **Return to latest**:
   ```
   git checkout master -- dvc.lock params.yaml metrics/scores.json
   dvc checkout
   ```
   (Or `git checkout HEAD -- ...` if on master already)

5. **Full branch experiment** -- create a branch, change the data, see complete divergence:
   ```
   git checkout -b experiment/more-data
   ```
   Edit `src/00_generate_data.py` and change `n_samples=1000` to `n_samples=2000`, then:
   ```
   uv run python src/00_generate_data.py
   dvc add data/raw/customers.csv
   dvc repro
   dvc push
   git add -A
   git commit -m "Experiment: double dataset to 2000 samples"
   git tag v3-more-data
   ```
   Now switch back: `git checkout master && dvc checkout`. The workspace is restored to the 1000-sample version instantly (from cache).

### Phase 5: Experiment Comparison (~15 min)

This phase uses DVC's comparison tools to analyze results across all experiments.

1. **Compare metrics between tags**:
   ```
   dvc metrics diff v1-baseline v2-high-C
   ```
   This compares the two experiments side by side.

2. **Compare params between tags**:
   ```
   dvc params diff v1-baseline v2-high-C
   ```

3. **Show metrics across all tags**:
   ```
   dvc metrics show --all-tags
   ```
   This displays metrics for every tagged commit.

4. **Try the Python API** -- open a Python shell or create a quick script:
   ```python
   import dvc.api

   # Read params from a specific Git revision
   params = dvc.api.params_show(rev="v1-baseline")
   print(params)

   # Read metrics from a specific revision
   metrics = dvc.api.metrics_show(rev="v2-high-C")
   print(metrics)
   ```
   The Python API lets you programmatically access DVC-tracked params and metrics from any Git revision without checking out the code.

5. **Generate a markdown diff** (useful for PR descriptions):
   ```
   dvc metrics diff v1-baseline v2-high-C --md
   dvc params diff v1-baseline v2-high-C --md
   ```

6. **Inspect the DAG**:
   ```
   dvc dag
   ```
   This prints the pipeline DAG in ASCII art, showing the stage execution order.

## Motivation

- **ML reproducibility gap**: Most ML projects lack reproducible pipelines. "It worked on my machine" is the default. DVC solves this by making pipelines declarative and data versioned.
- **Industry standard**: DVC has 14k+ GitHub stars and is used by teams at Intel, Microsoft, and many ML startups. It's the most popular open-source data versioning tool.
- **Complements Git skills**: If you know Git, DVC is a natural extension. The mental model (track, commit, push, pull, diff) is identical -- just for data instead of code.
- **Lightweight**: Unlike MLflow or W&B, DVC requires no server, no database, no cloud account. It works entirely with local files and any storage backend. Perfect for starting ML engineering practices.
- **lakeFS ecosystem**: With the Nov 2025 acquisition by Treeverse/lakeFS, DVC is now part of the broader data version control ecosystem, with long-term maintenance guaranteed.

## Commands

All commands run from `practice_068_dvc/`.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies (dvc, pandas, scikit-learn, pyyaml) |
| `dvc init` | Initialize DVC in the practice directory (creates `.dvc/`) |
| `dvc version` | Check installed DVC version |

### Phase 1: Data Tracking

| Command | Description |
|---------|-------------|
| `uv run python src/00_generate_data.py` | Generate synthetic dataset (`data/raw/customers.csv`) |
| `dvc add data/raw/customers.csv` | Track dataset with DVC (creates `.dvc` pointer file) |
| `dvc remote add -d local_storage tmp_remote_storage` | Configure local directory as default DVC remote |
| `dvc push` | Push cached data to the remote |
| `dvc pull` | Pull data from the remote (restores tracked files) |

### Phase 2: Pipeline

| Command | Description |
|---------|-------------|
| `dvc repro` | Run the full pipeline (only executes stages with changed deps/params) |
| `dvc repro --force` | Force re-run all stages regardless of cache |
| `dvc repro train` | Run pipeline up to and including the `train` stage |
| `dvc dag` | Print the pipeline DAG in ASCII art |

### Phase 3: Metrics & Params

| Command | Description |
|---------|-------------|
| `dvc metrics show` | Display current metric values |
| `dvc metrics diff` | Compare metrics between workspace and last commit |
| `dvc metrics diff v1-baseline v2-high-C` | Compare metrics between two Git tags |
| `dvc metrics diff --md` | Output metrics diff as a markdown table |
| `dvc params diff` | Compare params between workspace and last commit |
| `dvc params diff v1-baseline v2-high-C` | Compare params between two Git tags |
| `dvc metrics show --all-tags` | Show metrics for all tagged commits |

### Phase 4: Version Switching

| Command | Description |
|---------|-------------|
| `git checkout v1-baseline -- dvc.lock params.yaml metrics/scores.json` | Restore Git-tracked files from a tag |
| `dvc checkout` | Restore DVC-tracked files (data, models) to match current `.dvc`/`dvc.lock` |
| `git checkout master -- dvc.lock params.yaml metrics/scores.json` | Return to latest committed version |

### Phase 5: Python API

| Command | Description |
|---------|-------------|
| `uv run python -c "import dvc.api; print(dvc.api.params_show())"` | Show current params via Python API |
| `uv run python -c "import dvc.api; print(dvc.api.params_show(rev='v1-baseline'))"` | Show params from a specific revision |

## References

- [DVC Official Documentation](https://dvc.org/doc)
- [DVC Get Started Guide](https://dvc.org/doc/start)
- [DVC Pipeline Definition](https://dvc.org/doc/user-guide/pipelines/defining-pipelines)
- [dvc.yaml File Reference](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files)
- [DVC Metrics, Plots, and Parameters](https://dvc.org/doc/start/data-management/metrics-parameters-plots)
- [DVC Python API Reference](https://dvc.org/doc/api-reference)
- [DVC Internal Files](https://dvc.org/doc/user-guide/project-structure/internal-files)
- [DVC vs Git LFS Comparison (lakeFS blog)](https://lakefs.io/blog/dvc-vs-git-vs-dolt-vs-lakefs/)
- [DVC Joins lakeFS: Your Questions Answered](https://dvc.org/blog/dvc-joins-lakefs-your-questions-answered/)
- [DVC on PyPI](https://pypi.org/project/dvc/)
- [Tutorial: Data and Model Versioning](https://dvc.org/doc/use-cases/versioning-data-and-models/tutorial)

## State

`not-started`
