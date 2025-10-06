# Sharded Build/Search Workflow

This guide explains how to construct and evaluate sharded DiskANN indexes using the new `compare_build` and `compare_search` utilities.

## Prerequisites

- A C++17 toolchain with CMake 3.15+, OpenMP, Boost (program_options), MKL, and tcmalloc (see the main `README.md`).
- Base vector data stored in DiskANN binary format (`.bin`).
- Optional: query vectors (`.bin`) and ground-truth results (`.bin`/`.truth`).

## Build the tools

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target compare_build compare_search -j
```

Both executables are emitted into `build/apps/`.

## Stage 1: Shard construction with `compare_build`

`compare_build` randomly shuffles the base data, partitions it into the requested number of shards, writes an index per shard, and records metadata plus ID mappings in a `manifest.json`.

### Required parameters

- `--data_type {float|uint8|int8}` — element type of the base vectors.
- `--dist_fn {l2|cosine|mips|fast_l2}` — metric used for indexing.
- `--data_path <path>` — DiskANN binary containing the base dataset.
- `--output_root <dir>` — folder where shard directories will be created.
- `--shard_count <N>` — how many random shards to produce.

### Optional knobs

- `--Lbuild <int>` — graph build complexity (default `100`).
- `--max_degree <int>` — maximum graph degree (default `64`).
- `--alpha <float>` — graph construction alpha (default `1.2`).
- `--build_threads <int>` — worker threads (default: auto-detect).
- `--random_seed <int>` — deterministic shuffling.

### Example

```bash
./build/apps/compare_build \
  --data_type float \
  --dist_fn l2 \
  --data_path /data/base.100M.fbin \
  --output_root /data/experiments \
  --shard_count 8 \
  --Lbuild 120 \
  --max_degree 72 \
  --build_threads 32
```

Output directory layout:

```
/data/experiments/
└── S_8/
    ├── shard_000.index
    ├── shard_000.map
    ├── ...
    └── manifest.json
```

Each `.map` file maintains the mapping from shard-local IDs back to global dataset indices.

## Stage 2: Evaluation with `compare_search`

`compare_search` loads a particular shard configuration, runs multi-shard search, merges per-shard candidates, and reports average comparisons and recall (when ground truth is provided).

### Required parameters

- `--data_type {float|uint8|int8}` — must match the manifest.
- `--dist_fn {l2|cosine|mips|fast_l2}` — must match the manifest.
- `--query_file <path>` — aligned DiskANN binary containing query vectors.
- `--recall_at <K>` — number of final neighbors to retain per query.
- `--search_list <L1 L2 ...>` — one or more search `L` values (space-separated).
- `--shard_counts <S1 S2 ...>` — shard directories (named `S_<count>`) to evaluate.
- `--index_root <dir>` — parent folder containing the shard directories.

### Optional knobs

- `--gt_file <path>` — ground-truth binary for recall calculations.
- `--per_shard_k <int>` — candidates to pull per shard (defaults to `recall_at`).
- `--search_threads <int>` — number of OpenMP workers (default: auto).

> Note: ensure `per_shard_k * shard_count ≥ recall_at`; the tool enforces this to guarantee enough global candidates when merging shard results.
- `--output_dir <path>` — if set, writes per-shard CSV summaries (`search_list,recall,total_comparisons`).

### Example

```bash
./build/apps/compare_search \
  --data_type float \
  --dist_fn l2 \
  --query_file /data/query.10K.fbin \
  --recall_at 10 \
  --search_list 64 96 128 \
  --shard_counts 4 8 \
  --index_root /data/experiments \
  --gt_file /data/base.10K-gt100 \
  --output_dir /index/test_sift_shard/output
```

Sample output:

```
Evaluating shards for S=4 from /data/experiments/S_4
S=4 L=64 Recall@10: 0.923 Avg Comparisons/query: 185.2 Total Comparisons: 1852000
S=4 L=96 Recall@10: 0.945 Avg Comparisons/query: 210.7 Total Comparisons: 2107000
...
```

## Tips

-  Make sure the smallest `search_list` value is ≥ `recall_at` (and ≥ `per_shard_k`).
-  Ground-truth file is optional; without it, recall is reported as `NaN`.
-  Reuse the manifest to track build statistics and sharding seeds for reproducibility.

## Troubleshooting

- `compare_build`: ensure your dataset is large enough for the requested shard count.
- `compare_search`: if a manifest is missing, confirm `S_<count>/manifest.json` exists under `--index_root`.
- For deterministic experiments, specify `--random_seed` during build and reuse the same seed when rebuilding shards.
