# fri-lookup

Transparent FRI-style lookup protocol prototype based on `FRI查找表.docx`.

The crate is intentionally shaped like `D:\zkProjects\cq-main\cq-main` so both projects can later be driven with the same table/query instances:

- table size and query size must be powers of two;
- public tables reject duplicate values;
- query witnesses are vectors whose entries are sampled from the public table;
- the main smoke test is a `roundtrip` prover/verifier flow.

## Implemented modules

- `table`: public lookup table with value-to-index mapping, matching the CQ project style.
- `merkle`: row-wise batched Merkle commitments.
- `preprocessing`: reusable domain, public table, and fixed protocol structure preprocessing.
- `fri`: LDE evaluation, Merkle commitment, low-degree proof interface, and batch-FRI combination.
- `usc`: FRI-backed univariate sumcheck for claimed sums over multiplicative subgroups.
- `lookup`: T-cq/log-derivative lookup flow with preprocessing, proving, and verification.
- `transcript`: Fiat-Shamir challenges using Keccak256.

## Protocol mapping

The implemented lookup identity follows the document's logarithmic-derivative reduction. For challenge `beta`, the prover computes

```text
U[i] = 1 / (T[i] + beta)
A[i] = M[i] * U[i]
V[j] = 1 / (F[j] + beta)
```

and the lookup relation reduces to

```text
sum_i A[i] == sum_j V[j].
```

Before `beta`, the prover computes `M|L_T` and `F|L_F` and commits to them as separate Merkle roots `C_M` and `C_F`. These roots are absorbed before deriving `beta`.

After `beta`, the prover computes `U, A, V` and their LDE evaluations. The post-beta commitments follow the row-wise batched Merkle layout from the notes:

```text
C_{U,A} = Merkle.BatchCommit_{L_T}(U|L_T, A|L_T)
C_V     = Merkle.Commit_{L_F}(V|L_F)
```

The table-side Batch FRI proves the low-degree validity of `(M, U, A)` while keeping the commitment groups `(M)` and `(U,A)`. The query-side Batch FRI proves `(F, V)` while keeping the commitment groups `(F)` and `(V)`.

The query-side `B` column from the notes is treated as an alias for `V`:

```text
B(Y) := V(Y)
```

Because of this, the main query-side Batch FRI commits only `(F, V)`. There is no separate `B` interpolation, no `B|L_F` LDE, no committed `B` column, and no local `B - V = 0` consistency check. The sum relation is proved directly over `V` through the query-side USC instance.

Local consistency is folded into two error columns using Fiat-Shamir batching challenges:

```text
E_T = theta_T * (eta_T,1 * (U * (beta + T) - 1)
      + eta_T,2 * (A - M * U))
E_F = theta_F * eta_F * (V * (beta + F) - 1)
```

The four sum claims

```text
sum E_T = 0,  sum A = s
sum E_F = 0,  sum V = s
```

are merged into two USC instances:

```text
phi_T = E_T + delta_T * A,   sum phi_T = delta_T * s
phi_F = E_F + delta_F * V,   sum phi_F = delta_F * s
```

During verification, the USC query points are reused to open the main commitments and reconstruct `phi_T` from `T, M, U, A` and `phi_F` from `F, V`.

Low-degree claims are routed through the `fri` module using sampled FRI openings and folding checks. Batch FRI follows the degree-normalized design: for columns `f_j` with declared bounds `d_j`, it sets `D = max_j d_j` and proves one combined polynomial

```text
B(X) = sum_j rho_j * X^(D - d_j) * f_j(X).
```

The same normalization factor is used when verifier checks opened batch rows against the opened value of `B`.

## Preprocessing

`FriLookup::preprocess(table, query_size, blowup)` now separates the reusable work described in the design notes:

- domain preprocessing for `H_T`, `H_F`, `L_T`, and `L_F`, including domain elements, roots of unity, FFT/IFFT metadata, bit-reversal permutations, coset shift, vanishing polynomials, vanishing evaluations on LDE domains, and FRI layer pairing maps;
- public table preprocessing, including distinctness checks through `Table::new`, the canonical `table_id = H(T, H_T, L_T, N)`, interpolation coefficients for `T(X)`, `T` evaluated on `L_T`, the full Merkle tree for `T|L_T`, the table value-to-index map held by `Table`, and the reusable FRI low-degree proof for `T`;
- fixed protocol structure preprocessing, including row-wise batched Merkle layouts, column orders for table/query/USC commitments, Merkle separators, opening/index rules, degree-normalization metadata, LDE normalization factors, and transcript/challenge ordering.

For repeated lookups against the same fixed table, use the two-stage API so table-side preprocessing is done once:

```rust
let reusable = FriLookup::preprocess_table(&table, blowup)?;

for query_values in queries {
    let witness = LookupWitness::new(&query_values)?;
    let statement = LookupStatement {
        query_size: query_values.len(),
    };
    let params = FriLookup::parameters_for_query(&reusable, statement.query_size)?;
    let proof = FriLookup::prove(&params, &witness, &statement)?;
    FriLookup::verify(&params, &statement, &proof)?;
}
```

`preprocess_table` computes and stores the table interpolation, `T|L_T` Merkle tree, table FRI proof, table lookup index, table-side `H_T/L_T` domain preprocessing, and table-side degree normalizers once. `parameters_for_query` only derives the query-size-dependent `H_F/L_F` domains and protocol layout for each `query_size`.

On the first verification for a reusable table, the verifier recomputes the canonical table domains, `table_id`, `T(X)`, `T|L_T`, and the Merkle root from the public table, then verifies the reusable table FRI proof. A successful validation is cached and shared by all query-size-specific parameters derived from the same reusable preprocessing; changing the table identifier, root, domain, blowup, or FRI proof invalidates the cache.

## Run

```powershell
cd D:\zkProjects\fri-lookup
cargo test
```

Run scale experiments from the command line:

```powershell
# One case.
cargo run --example scale_experiment -- --table-size 1024 --query-size 128 --blowup 4

# Reuse one fixed table preprocessing and run multiple queries.
cargo run --example scale_experiment -- --table-size 1024 --runs 10 --blowup 4

# Reuse one fixed table preprocessing and keep every query the same size.
cargo run --example scale_experiment -- --table-size 1024 --query-size 128 --runs 10 --blowup 4

# Reuse one fixed table preprocessing and run fixed different query sizes once each.
cargo run --example scale_experiment -- --table-size 1024 --query-sizes 8,16,32,64 --blowup 4

# Reuse one fixed table preprocessing and cycle fixed different query sizes.
cargo run --example scale_experiment -- --table-size 1024 --query-sizes 8,16,32,64 --runs 10 --blowup 4
```

In repeated-query mode, table preprocessing is measured once and the CSV `preprocessing_ms` column reuses that same value for every output row. If `--query-size` is omitted, the example cycles query sizes through powers of two up to the table size. If `--query-sizes` is provided, the example uses that fixed list; with `--runs`, it cycles through the list.

The separate `Tvalidate_ms` record measures only the deterministic validation performed the first time the reusable preprocessing parameters are used. The online lookup-proof `verification_ms` timer starts after that validation has succeeded and been cached. In average mode, `verification_ms` is the mean of all five online verification runs.

On this machine, Rust currently uses the `x86_64-pc-windows-msvc` target. If `cargo test` fails before compiling this crate with `link.exe` or missing Windows SDK libraries, install Visual Studio Build Tools with the C++ workload or switch to an installed GNU Rust target.

## Optional ICICLE BN254 NTT backend

The dense LDE path has a backend dispatch layer. By default it uses arkworks CPU FFT. The ICICLE BN254 NTT backend is behind the `icicle-rust` feature and only applies when the field type is `ark_bn254::Fr`; other fields continue to use CPU.

The ICICLE source is expected at `third_party/icicle` and is wired as a local path dependency. On a GPU server, make sure CMake, CUDA, and the ICICLE backend/runtime requirements are installed, then run:

```powershell
cargo test --features icicle-rust
$env:FRI_LOOKUP_LDE_BACKEND = "gpu"
$env:FRI_LOOKUP_ICICLE_DEVICE = "CUDA"
$env:FRI_LOOKUP_ICICLE_DEVICE_ID = "0"
cargo run --release --example scale_experiment -- --table-size 65536 --query-size 8192 --blowup 4
```

Useful knobs:

- `FRI_LOOKUP_LDE_BACKEND=cpu|gpu|auto`
- `FRI_LOOKUP_GPU_LDE_MIN_SIZE=65536`
- `FRI_LOOKUP_GPU_RESIDENT_FRI_BACKEND=cpu|gpu|auto`
- `FRI_LOOKUP_GPU_RESIDENT_FRI_MIN_SIZE=65536`
- `FRI_LOOKUP_BATCH_MERKLE_BACKEND=cpu|gpu|auto`
- `FRI_LOOKUP_BATCH_COMBINE_BACKEND=cpu|gpu|auto`
- `FRI_LOOKUP_FOLD_BACKEND=cpu|gpu|auto`
- `FRI_LOOKUP_PARALLEL_PROVE=1|0`
- `FRI_SCALE_DIAGNOSTICS=1` prints warmup/stage timings for `scale_experiment`
- `FRI_LOOKUP_GPU_DIAGNOSTICS=1` prints GPU kernel diagnostics
- `FRI_LOOKUP_ICICLE_DEVICE=CUDA`
- `FRI_LOOKUP_ICICLE_DEVICE_ID=0`

If `FRI_LOOKUP_LDE_BACKEND` is omitted, the code uses `auto`: ICICLE is selected only for `ark_bn254::Fr` dense LDE sizes at or above `FRI_LOOKUP_GPU_LDE_MIN_SIZE`. The scale experiment CSV reports `peak_memory_bytes` from Linux `VmHWM` when available.

## Experimental Data

The experimental performance data supplied for the TLookup paper are available in [data/README.md](data/README.md), which documents the original workbook, metrics, configurations, source-cell provenance, reproduction commands, and known data limitations.

- [CQ baseline results](data/cq_results.csv)
- [CQ index generation time and size](data/cq_index_generation.csv)
- [TLookup CPU results](data/tlookup_cpu_results.csv)
- [TLookup GPU results](data/tlookup_gpu_results.csv)
- [Original experimental workbook](data/performance_results_128.xlsx)
- [Author-confirmed corrections](data/author_corrections.csv)
- [Data validation report](data/validation_report.md)

The datasets preserve the workbook's stored numeric precision and apply only the explicitly documented author corrections. Both original workbooks remain unchanged; corrected CSV fields are identified per record. CQ index generation is recorded separately from TLookup preprocessing, and CPU/GPU records retain independent source locations. Regenerate the CSVs with `python3 scripts/export_benchmark_data.py`; verify existing exports without writing with `python3 scripts/export_benchmark_data.py --check`.
