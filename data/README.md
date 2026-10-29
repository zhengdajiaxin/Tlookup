# TLookup Experimental Data

## Overview

This dataset contains the experimental results used in the TLookup paper, including the CQ baseline and TLookup CPU and GPU benchmarks. The CSV files provide the experimental configurations, execution times, and proof-related sizes.

## Dataset Files

| File | Contents | Records |
| --- | --- | ---: |
| [cq_results.csv](cq_results.csv) | CQ proving time, verification time, proof size, and reported total time. | 46 |
| [cq_index_generation.csv](cq_index_generation.csv) | CQ index generation time and index size, recorded separately from proving. | 4 |
| [tlookup_cpu_results.csv](tlookup_cpu_results.csv) | TLookup CPU preprocessing, proving, verification, and size measurements. | 91 |
| [tlookup_gpu_results.csv](tlookup_gpu_results.csv) | TLookup GPU preprocessing, proving, verification, and size measurements. | 91 |

Files use UTF-8 encoding and comma-separated columns. CPU and GPU records are independent; comparisons should use their explicit configuration fields rather than row positions.

## Experimental Parameters

| Field | Meaning |
| --- | --- |
| `table_size` | Number of entries in the public lookup table. |
| `witness_size` | Number of lookup values in a CQ witness. |
| `query_size` | Number of lookup values in a TLookup query. |
| `blowup` | Dimensionless low-degree extension expansion factor for TLookup. |
| `run` | Run label within a TLookup configuration block; not a repetition count. |
| `preprocess_metric` | Preprocessing label: `preprocess_ms` or `table_preprocess_ms`; the latter identifies table preprocessing. |

TLookup table sizes range from 32,768 to 2,097,152 in powers of two, with query sizes from 64 through the table size and `blowup=4`. CQ table sizes range from 32,768 to 262,144. Columns beginning with `source_` and the `corrected_fields` column contain record metadata and are not performance measurements.

## Performance Metrics

| Field | Unit | Meaning |
| --- | --- | --- |
| `preprocess_ms`, `preprocess_s` | Milliseconds, seconds | TLookup preprocessing time, with its scope identified by `preprocess_metric`. |
| `prove_ms`, `prove_s` | Milliseconds, seconds | Proof generation time. |
| `verify_ms` | Milliseconds | Proof verification time. |
| `proof_bytes` | Bytes | Proof size. |
| `preprocessing_bytes` | Bytes | TLookup preprocessing/public-parameter size, not peak memory consumption. |
| `index_gen_ms` | Milliseconds | CQ index generation time. |
| `index_bytes` | Bytes | CQ index size. |
| `total_ms`, `total_s` | Milliseconds, seconds | Reported combined preprocessing/index generation and proving time, excluding verification. |

CQ index generation and TLookup preprocessing are distinct operations. The seconds columns express the corresponding time metrics in seconds. The files do not specify hardware models, repetition counts, or statistical aggregation methods.

## Reproducibility

The TLookup benchmark entry point is [examples/scale_experiment.rs](../examples/scale_experiment.rs). From the repository root, run a CPU case with:

```bash
cargo run --release --no-default-features --example scale_experiment -- \
  --table-size 32768 --query-size 64 --blowup 4
```

The example also accepts `--query-sizes 64,128,256` to reuse table preprocessing across different query sizes, and `--runs` to control the number of iterations. In this mode, table preprocessing is measured once. Online verification timing excludes the separately reported `Tvalidate_ms` preprocessing-validation step.

For GPU execution, install the project's CUDA/ICICLE prerequisites and enable `icicle-rust` in the experiment invocation. For example, in a POSIX shell:

```bash
env FRI_LOOKUP_ICICLE_DEVICE=CUDA \
    FRI_LOOKUP_ICICLE_DEVICE_ID=0 \
    FRI_LOOKUP_LDE_BACKEND=gpu \
    FRI_LOOKUP_MERKLE_BACKEND=gpu \
    FRI_LOOKUP_GPU_RESIDENT_FRI_BACKEND=gpu \
    FRI_LOOKUP_BATCH_MERKLE_BACKEND=gpu \
    FRI_LOOKUP_BATCH_COMBINE_BACKEND=gpu \
    FRI_LOOKUP_FOLD_BACKEND=gpu \
    cargo run --release --features icicle-rust --example scale_experiment -- \
      --table-size 32768 --query-size 64 --blowup 4
```

These commands run the current implementation; timing depends on the execution environment. GPU execution accelerates selected computational stages while verification remains on the CPU. See the [project instructions](../README.md) and [CPU/GPU execution map](../CPU_GPU_EXECUTION_MAP.md) for build requirements and backend behavior. CQ is an independent baseline and is not executed by this TLookup example.
