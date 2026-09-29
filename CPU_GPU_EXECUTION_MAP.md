# fri-lookup CPU/GPU 执行映射表

本文按当前源码实现说明 CPU-only 构建与 `icicle-rust` GPU 构建中的实际执行位置。这里的“GPU 版本”不是整条协议都在 GPU 上：Rust 协议控制、Fiat–Shamir transcript、挑战生成、证明结构组装和验证器仍在 CPU 上，只有满足字段、feature、后端开关和规模阈值的计算密集阶段会进入 GPU。

## 1. 结论表

| 协议阶段 | CPU-only 版本 | GPU 版本中仍由 CPU 执行 | GPU 版本中可由 GPU 执行 |
|---|---|---|---|
| 构建与后端初始化 | `cargo` 编译 Rust；不编译 CUDA | `build.rs` 调用 `nvcc`、加载 ICICLE backend、选择 device、创建/调度 GPU worker | 无协议计算；CUDA runtime 初始化发生在设备后端 |
| 协议/域预处理 | `preprocess_domain_side`、`domain_info`、`fft_plan`、`fri_layers`、degree normalizer 全部在 CPU 生成 | 同 CPU-only；当前没有 GPU 预处理实现 | 无 |
| 固定表 `T` 的预处理 | `fri::interpolate` CPU IFFT；`evaluate_lde_from_coeffs` CPU FFT；CPU Merkle；CPU FRI | `FriLookup::preprocess_table` 的分派、检查、root 对比、结果封装；GPU FRI 中的 transcript/alpha/query index 生成 | GPU-resident 路径可执行 `T` 的 inverse NTT、LDE forward NTT、Merkle Keccak、FRI folding 和 opening gather |
| 乘数列 `M` 构造 | 查询值到表索引的计数循环、support 搜索、插值均为 CPU；稀疏 LDE 为 CPU | `M` 的计数和 support 搜索始终为 CPU；`polynomial_coefficients` 始终 CPU | 当没有走稀疏公式而转为 dense LDE 时，LDE 可用 GPU NTT；`M` 的 Batch Merkle commitment 可用 GPU |
| 查询列 `F` 构造 | CPU IFFT、CPU LDE、CPU Merkle | `F` 系数插值 (`polynomial_coefficients`) 仍是 CPU；后端选择和结果封装在 CPU | dense LDE 可用 ICICLE NTT；Batch Merkle commitment 可用 GPU |
| Lookup transcript 与 `beta`/`eta`/`rho`/`zeta` 等挑战 | `Transcript` 中的 Keccak256 和 rejection sampling 全部 CPU | 全部 CPU；生成的标量按需传到显存 | 无 |
| `U=1/(T+beta)`、`A=M*U`、`V=1/(F+beta)` | `batch_reciprocals`、逐点乘法、系数 IFFT、LDE 和 commitment 均为 CPU | 输入合法性检查、backend dispatch；GPU 结果中的 `U/A/V` 基域值和系数会复制回 CPU | GPU-resident auxiliary 路径执行 `scalar_add`、`inv_scalars`、`mul_scalars`、inverse/forward NTT，并在显存上构造 `C_{U,A}`、`C_V` |
| 约束在基域上的精确性检查 | `check_table_constraint_on_domain`、`check_query_constraint_on_domain` 的逐点循环 | **始终 CPU**，即使随后约束商走 GPU | 无 |
| 约束商 `Q_T`、`Q_F` | CPU FFT 多项式乘法、取高次系数、基域 FFT、LDE FFT、Merkle | dispatch、trim、错误检查和输出结构组装；GPU 产出的系数/基域值按需回传 | GPU 路径执行 forward/inverse NTT、逐点乘、标量乘、加减、高次系数截取、商 LDE 和 Merkle commitment |
| 主 Batch FRI 列组合 | CPU 逐点 degree-normalized 线性组合 | batching challenge 生成、backend dispatch、proof struct 组装 | Batch-combine GPU 或 GPU-resident FRI 使用 `mul_scalars`/`scalar_mul`/`add_scalars` 组合列 |
| Merkle 叶子和树 | CPU Keccak；大规模时用 CPU threads | 普通 `merkle.rs` GPU 路径仍在 CPU 拼接 leaf/node preimage；root/path/proof 管理在 CPU | ICICLE Keccak 可计算 hash；Batch/GPU-resident 路径还用原生 CUDA kernel 在显存中拼 leaf/node preimage |
| FRI folding | `fold_layer_cpu_with_tree`，CPU threads 做 field arithmetic 和 leaf hash | 每轮 `alpha` 和 transcript 吸收、循环控制、最终 proof 组装 | `gpu_fold_bn254` 或 GPU-resident FRI 执行 add/sub/mul folding；达到阈值时同时构造 GPU Merkle tree |
| FRI query/opening | CPU 采样 query index、读取层值、生成 Merkle path | query index 始终由 CPU transcript 采样；最终 opening 对象在 CPU 组装 | GPU-resident 路径用 gather kernels 只取被查询的 field/digest，再 D2H 回传 |
| OOD claimed value | CPU Horner 求值 `evaluate_coeffs`，CPU transcript 生成 `zeta` | **始终 CPU** | 无 |
| OOD opening quotient 与 OOD Batch FRI | CPU synthetic division、基域 FFT、LDE、Merkle、Batch FRI | claimed values、`zeta`、系数和证明组装在 CPU | GPU-resident 路径可在显存逐点构造 opening quotient、构造 Merkle、组合、fold 和 gather openings |
| 公共和 `s=sum(A)` | CPU iterator 求和 | **始终 CPU** | 无 |
| Specialized USC (`p_A`、`p_V`) | CPU 校验 sum claim、构造商系数、FFT/LDE/Merkle/FRI | sum claim 校验、`average=s/n` 标量计算、挑战和 proof 组装仍为 CPU | GPU-resident 路径执行 `(column-average)*inverse_weight`、Merkle、Batch FRI 和 opening gather |
| Prover 的 table/query 并行调度 | CPU threads，可由 `FRI_LOOKUP_PARALLEL_PROVE` 控制 | 同左；所有 GPU job 还会通过 `fri-lookup-icicle-worker` 串行调度 | 被 worker 提交的 ICICLE/CUDA kernels 在 GPU 执行 |
| Verifier | FRI 验证、Merkle path 验证、OOD identity、USC identity、transcript 重放全部 CPU；FRI queries 可用 CPU threads 并行 | **与 CPU-only 相同，当前没有 GPU verifier 路径** | 无 |
| 证明序列化 | 当前源码没有独立序列化后端；`LookupProof` 是 CPU 内存中的 Rust struct | 同 CPU-only | 无 |
| Host-device 数据传输 | 无 | Rust/arkworks ↔ ICICLE 字段格式转换、H2D/D2H 发起与 stream 同步均由 CPU 控制 | `DeviceVec` 驻留数据；CUDA gather/memset 等处理显存数据 |

## 2. 按主证明流程映射到函数

主入口是 `FriLookup::prove_with_timings`（`src/lookup.rs:486`），实际顺序如下。

| 顺序 | 过程 | CPU 函数/工作 | GPU 函数/工作 |
|---:|---|---|---|
| 1 | 构造并提交 `M`、`F` | `construct_initial_witness_precommits`、`build_multiplicity_precommit`、`build_initial_query_precommit`；计数、support、插值在 CPU | dense LDE 经 `lde::evaluate_dense_lde_*` 可进 ICICLE；`prepare_batch_commitment_group` 可选择 GPU Batch Merkle |
| 2 | 生成 `beta` | `lookup_transcript`、`Transcript::challenge` | 无 |
| 3 | 构造 `U,A,V` | CPU fallback：`batch_reciprocals`、`build_table_auxiliary_precommit`、`build_query_auxiliary_precommit` | `try_prepare_lookup_auxiliary_gpu_resident` → `gpu_batch_merkle_bn254::try_prepare_lookup_auxiliary`：GPU inverse、pointwise mul、NTT/LDE、Merkle |
| 4 | 生成 `eta_t1, eta_t2` | `Transcript::challenge_nonzero` | 无 |
| 5 | 检查约束并构造 `Q_T,Q_F` | `check_*_constraint_on_domain` 始终 CPU；fallback 为 `product_high_coeffs_cpu`/`poly_mul`、`build_constraint_quotient_and_commitment` | `try_prepare_table_constraint_quotient_gpu_resident`、`try_prepare_query_constraint_quotient_gpu_resident`；较窄的 partial path 还可用 `try_product_high_coefficients_gpu` |
| 6 | 主 Batch FRI | CPU fallback：`prepare_batch_low_degree_with_commitments_and_lde`、`combine_lde_columns_normalized_cpu_with_tree`、`fold_layer_cpu_with_tree` | GPU store：`prepare_batch_low_degree_with_commitments_and_gpu_lde_store`；resident combine/fold/Merkle/gather，或分别启用 GPU combine、GPU fold、GPU Batch Merkle |
| 7 | 生成 `zeta_t,zeta_f` 和 OOD claims | `challenge_outside_lde_domain`、`evaluate_coeffs` | 无 |
| 8 | OOD proofs | CPU fallback：`build_ood_precommit_column`、`opening_quotient_coeffs`、CPU/可分派 LDE、Batch FRI | `prepare_ood_opening_batch_from_gpu_lde_store` 或 `try_prepare_ood_opening_batch_gpu_resident`，之后 `PreparedGpuOodOpeningBatch::finish` |
| 9 | 计算公共和 `s` | `table_accumulator.iter().copied().sum()` | 无 |
| 10 | Specialized USC | CPU fallback：`specialized_usc_quotient_coeffs`、LDE、Merkle、Batch FRI | `prepare_specialized_usc_batch_from_gpu_lde_store` 或 `try_prepare_specialized_usc_batch_gpu_resident`，之后 `PreparedGpuSpecializedUscBatch::finish` |
| 11 | 组装证明 | `LookupProof` 字段填充、timings 统计 | 无 |

## 3. 各 GPU 后端的边界

| 后端 | 选择位置 | 自动启用默认阈值 | GPU 上的核心操作 | CPU 回传/保留工作 |
|---|---|---:|---|---|
| Dense LDE | `src/lde.rs:196` `select_dense_lde_backend` | LDE size `>= 2^16` | ICICLE inverse NTT + forward NTT | 从系数入口调用时先在 CPU 计算 base-domain evaluations；结果返回 host `Vec` |
| 普通 Merkle | `src/merkle.rs:100` `select_merkle_backend` | leaves `>= 2^16` | ICICLE Keccak | leaf/node preimage 在 CPU 拼接，整层 host buffer 往返 |
| Batch Merkle | `src/fri.rs:2503` | leaves `>= 2^16` | CUDA leaf/node preimage kernels + ICICLE Keccak | root/path 管理；非 resident 输入需要 H2D |
| GPU-resident FRI | `src/fri.rs:3219` | LDE size `>= 2^16` | 显存列组合、Merkle、所有 FRI folds、gather openings | transcript、alpha/query index、root 吸收、proof struct |
| Batch combine | `src/fri.rs:3443` | LDE size `>= 2^20` | ICICLE `mul_scalars`/`scalar_mul`/`add_scalars` | normalized weight 列目前在 CPU 生成并传入；非 resident 结果 D2H |
| Standalone FRI fold | `src/fri.rs:4107` | 当前 layer size `>= 2^16` | add/sub/mul folding；可选 GPU Merkle | folding coefficients 在 CPU 生成；非 resident folded vector D2H |
| Fold 后 Merkle | `src/fri.rs:10083` | folded leaves `>= 2^16` | CUDA preimage + ICICLE Keccak | 小层会回退 CPU tree |
| Auxiliary `U/A/V` | 由 Dense LDE 的两个 plan 决定 | table 或 query 至少一个 plan 选 GPU | denominator、batch inverse、`A=M*U`、IFFT/LDE、commitment | `U/A/V` 基域值和系数会复制回 CPU供后续 claims/检查使用 |
| `Q_T/Q_F` | 由 product plan 和 LDE plan 共同决定 | 两个 plan 均需选 GPU | NTT convolution、vector ops、LDE、commitment | 基域约束检查始终 CPU；商系数和 base eval 回传 |

## 4. 原生 CUDA 文件究竟做什么

`native/fri_lookup_cuda_kernels.cu` 中的 kernel 不负责 NTT，也不直接实现 Keccak。它提供的是显存数据整理和稀疏读取：

| CUDA kernel | 用途 |
|---|---|
| `field_leaf_preimages_kernel` | 把一个 BN254 field 编码为 `"leaf" || field` |
| `row2_leaf_preimages_kernel` / `row3_leaf_preimages_kernel` / `row5_leaf_preimages_kernel` | 把 2/3/5 列的一行编码为 Merkle leaf preimage |
| `node_preimages_kernel` | 把两个 digest 编码为 `"node" || left || right` |
| `gather_fields_kernel` | 按 query indices 从一个显存列抽取 field |
| `gather_row_fields_kernel` | 按 query indices 从最多 5 个显存列抽取行 |
| `gather_digests_kernel` | 按 indices 抽取 Merkle sibling digests |
| `fri_lookup_cuda_memset_zero` | 异步清零显存 buffer |

真正的 GPU NTT、field vector ops 和 Keccak 分别来自 ICICLE 的 `ntt`、`vec_ops` 和 `IcicleKeccak256`。

## 5. GPU 版本成立的条件

1. 必须使用 `--features icicle-rust`。`gpu-fft` feature 本身只引入 `ark-bn254`，并不会编译/启用 ICICLE 或 CUDA kernel。
2. 当前 GPU 快路径只支持字段 `ark_bn254::Fr`。其他 `FftField` 即使是 GPU 构建也走 CPU。
3. 需要可用的 CUDA/ICICLE backend；`build.rs` 只在 `icicle-rust` 下用 `nvcc` 编译 `native/fri_lookup_cuda_kernels.cu`。
4. `auto` 模式还必须达到对应规模阈值。显式 `cpu` 会禁用对应 GPU 子后端；显式 `gpu` 通常在不可用时返回错误（普通 Dense LDE 的选择器是例外：不支持字段/feature 时会选择 CPU）。
5. 多个开关彼此独立。设置 `FRI_LOOKUP_LDE_BACKEND=gpu` 不等于强制 Batch Merkle、Batch combine、FRI fold 或 GPU-resident FRI 全部使用 GPU。

CPU-only 示例：

```bash
cargo test
```

完整 GPU 倾向示例：

```bash
cargo test --features icicle-rust

export FRI_LOOKUP_ICICLE_DEVICE=CUDA
export FRI_LOOKUP_ICICLE_DEVICE_ID=0
export FRI_LOOKUP_LDE_BACKEND=gpu
export FRI_LOOKUP_GPU_RESIDENT_FRI_BACKEND=gpu
export FRI_LOOKUP_BATCH_MERKLE_BACKEND=gpu
export FRI_LOOKUP_BATCH_COMBINE_BACKEND=gpu
export FRI_LOOKUP_FOLD_BACKEND=gpu
```

## 6. 有效环境变量与默认阈值

| 环境变量 | 值 | 默认/作用 |
|---|---|---|
| `FRI_LOOKUP_LDE_BACKEND` | `cpu/gpu/auto` | 默认 `auto` |
| `FRI_LOOKUP_GPU_LDE_MIN_SIZE` | 正整数 | `2^16` |
| `FRI_LOOKUP_MERKLE_BACKEND` | `cpu/gpu/auto` | 普通 `MerkleTree` 后端，默认 `auto` |
| `FRI_LOOKUP_GPU_MERKLE_MIN_LEAVES` | 正整数 | 普通 Merkle 默认 `2^16` |
| `FRI_LOOKUP_BATCH_MERKLE_BACKEND` | `cpu/gpu/auto` | 默认 `auto` |
| `FRI_LOOKUP_GPU_BATCH_MERKLE_MIN_LEAVES` | 正整数 | `2^16` |
| `FRI_LOOKUP_BATCH_COMBINE_BACKEND` | `cpu/gpu/auto` | 默认 `auto` |
| `FRI_LOOKUP_GPU_BATCH_COMBINE_MIN_SIZE` | 正整数 | `2^20` |
| `FRI_LOOKUP_FOLD_BACKEND` | `cpu/gpu/auto` | 默认 `auto` |
| `FRI_LOOKUP_GPU_FOLD_MIN_SIZE` | 正整数 | `2^16` |
| `FRI_LOOKUP_GPU_FOLD_MERKLE_MIN_LEAVES` | 正整数 | `2^16` |
| `FRI_LOOKUP_GPU_RESIDENT_FRI_BACKEND` | `cpu/gpu/auto` | 默认 `auto` |
| `FRI_LOOKUP_GPU_RESIDENT_FRI_MIN_SIZE` | 正整数 | `2^16` |
| `FRI_LOOKUP_PARALLEL_PROVE` | `1/0` 等 | 默认开启 CPU table/query 并行 |
| `FRI_LOOKUP_ICICLE_DEVICE` | device type | 默认 `CUDA` |
| `FRI_LOOKUP_ICICLE_DEVICE_ID` | device id | 默认 `0` |

`DEFAULT_GPU_INV_MIN_SIZE`（`src/lookup.rs:29`）和 `DEFAULT_GPU_USC_MIN_SIZE`（`src/usc.rs:10`）当前只有常量定义，没有被选择逻辑引用，因此它们目前不会单独控制 inverse 或 USC 的 GPU 阈值。二者实际随 Dense LDE/GPU-resident FRI 的选择进入 GPU。

## 7. 数据传输图

```text
CPU: witness/table + transcript challenges
        |
        | H2D：字段列、系数、beta/eta/rho/zeta、query indices
        v
GPU: NTT / inverse / pointwise ops / combine / Merkle / FRI fold
        |
        | D2H：roots、U/A/V 与部分系数、final evals、query values/paths
        v
CPU: transcript 吸收、claims、proof 组装、完整 verifier
```

GPU-resident FRI 的主要收益是避免每一层 folding 后把整列复制回 CPU；通常只回传 root、最终小层和被抽样的 openings。普通独立 GPU combine/fold 路径仍会回传完整结果向量，再由后续 CPU/另一后端继续处理。

## 8. 关键源码索引

- 构建 feature 与 CUDA 编译：`Cargo.toml:7-16`，`build.rs:3-42`
- 主 prover/verifier：`src/lookup.rs:478-1265`
- CPU/GPU Dense LDE：`src/lde.rs:78-272`
- 普通 CPU/GPU Merkle：`src/merkle.rs:86-535`
- Batch Merkle、combine、resident FRI、fold 的调度：`src/fri.rs:2503-3659`、`src/fri.rs:4107-4327`
- GPU-resident FRI：`src/fri.rs:4330-7965`
- GPU Batch Merkle/auxiliary/quotient：`src/fri.rs:7967-9601`
- GPU Batch combine：`src/fri.rs:9603-9907`
- GPU fold：`src/fri.rs:9909-10543`
- 原生 CUDA preimage/gather kernels：`native/fri_lookup_cuda_kernels.cu:86-478`
- CPU transcript：`src/transcript.rs:1-77`
- CPU verifier/USC verifier：`src/lookup.rs:1058-1265`，`src/usc.rs:61-126`
