use std::{collections::BTreeMap, env, process, time::Instant};

use ark_bn254::Fr;
use ark_ff::{FftField, Field, PrimeField, ToBytes};
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use fri_lookup::{
    fri::{
        BatchFriProof, BatchLdeRowOpening, FriLayerOpening, FriProof, FriQuery, OodOpeningProof,
        OodRowOpening,
    },
    lookup::{FriLookup, LookupProof, LookupStatement, LookupWitness, PublicParameters},
    merkle::{Digest32, MerklePath},
    preprocessing::{
        ColumnLabel, DegreeNormalizationPreprocessing, DegreeNormalizersOnLde, DomainInfo,
        DomainPreprocessing, DomainSidePreprocessing, FftPlanPreprocessing, FriLayerPreprocessing,
        MerkleLeafLayout, ProtocolStructurePreprocessing, VanishingPolynomial,
    },
    table::Table,
    usc::{SpecializedUscOpening, SpecializedUscProof},
};

const USIZE_BYTES: usize = 8;
const VEC_LEN_BYTES: usize = 8;
const RNG_SEED: u64 = 42;
const DEFAULT_BLOWUP: usize = 4;

fn main() {
    let experiment = parse_experiment().unwrap_or_else(|message| {
        eprintln!("{message}");
        print_usage();
        process::exit(2);
    });

    println!("table_size,query_size,blowup,preprocessing_ms,proving_ms,verification_ms,preprocessing_proof_size_bytes,proof_size_bytes");
    match experiment {
        Experiment::Cases(cases) => {
            for (table_size, query_size, blowup) in cases {
                match run_case(table_size, query_size, blowup) {
                    Ok((row, tvalidate_ms)) => {
                        println!("Tvalidate_ms,{tvalidate_ms:.3}");
                        print_row(table_size, query_size, blowup, &row);
                    }
                    Err(error) => eprintln!(
                        "case failed: table_size={}, query_size={}, blowup={}, error={:?}",
                        table_size, query_size, blowup, error
                    ),
                }
            }
        }
        Experiment::Repeated {
            table_size,
            blowup,
            runs,
            query_size,
            query_sizes,
            aggregation,
        } => {
            if let Err(error) = run_repeated_table(
                table_size,
                blowup,
                runs,
                query_size,
                query_sizes,
                aggregation,
            ) {
                eprintln!(
                    "repeated experiment failed: table_size={}, blowup={}, runs={}, error={:?}",
                    table_size, blowup, runs, error
                );
            }
        }
    }
}

enum Experiment {
    Cases(Vec<(usize, usize, usize)>),
    Repeated {
        table_size: usize,
        blowup: usize,
        runs: usize,
        query_size: Option<usize>,
        query_sizes: Option<Vec<usize>>,
        aggregation: Aggregation,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Aggregation {
    None,
    Median,
    Average,
}

impl Aggregation {
    fn is_enabled(self) -> bool {
        self != Self::None
    }
}

fn parse_experiment() -> Result<Experiment, String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Ok(Experiment::Cases(parse_case_list(
            &env::var("FRI_SCALE_CASES").unwrap_or_else(|_| {
                "32:8:4,64:8:4,128:16:4,256:32:4,512:64:4,1024:128:4".to_string()
            }),
        )?));
    }

    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_usage();
        process::exit(0);
    }

    if args.len() == 3 && args.iter().all(|arg| !arg.starts_with('-')) {
        return Ok(Experiment::Cases(vec![(
            parse_usize("table_size", &args[0])?,
            parse_usize("query_size", &args[1])?,
            parse_usize("blowup", &args[2])?,
        )]));
    }

    let mut cases = Vec::new();
    let mut table_size = None;
    let mut query_size = None;
    let mut query_sizes = None;
    let mut blowup = None;
    let mut runs = None;
    let mut aggregation = Aggregation::None;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--case" => {
                let value = next_arg(&args, index, "--case")?;
                cases.push(parse_case(value)?);
                index += 2;
            }
            "--cases" => {
                let value = next_arg(&args, index, "--cases")?;
                cases.extend(parse_case_list(value)?);
                index += 2;
            }
            "--table-size" => {
                table_size = Some(parse_usize(
                    "--table-size",
                    next_arg(&args, index, "--table-size")?,
                )?);
                index += 2;
            }
            "--query-size" => {
                query_size = Some(parse_usize(
                    "--query-size",
                    next_arg(&args, index, "--query-size")?,
                )?);
                index += 2;
            }
            "--query-sizes" => {
                query_sizes = Some(parse_usize_list(
                    "--query-sizes",
                    next_arg(&args, index, "--query-sizes")?,
                )?);
                index += 2;
            }
            "--blowup" => {
                blowup = Some(parse_usize(
                    "--blowup",
                    next_arg(&args, index, "--blowup")?,
                )?);
                index += 2;
            }
            "--runs" => {
                runs = Some(parse_usize("--runs", next_arg(&args, index, "--runs")?)?);
                index += 2;
            }
            "--median" => {
                if aggregation != Aggregation::None {
                    return Err("only one aggregation mode can be selected".to_string());
                }
                aggregation = Aggregation::Median;
                index += 1;
            }
            "--average" | "--mean" => {
                if aggregation != Aggregation::None {
                    return Err("only one aggregation mode can be selected".to_string());
                }
                aggregation = Aggregation::Average;
                index += 1;
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    if query_size.is_some() && query_sizes.is_some() {
        return Err("--query-size and --query-sizes cannot be used together".to_string());
    }

    if runs.is_some() || query_sizes.is_some() || aggregation.is_enabled() {
        let table_size = table_size.ok_or("--runs mode requires --table-size".to_string())?;
        let blowup = blowup.unwrap_or(DEFAULT_BLOWUP);
        let runs = runs.unwrap_or_else(|| match (&query_sizes, aggregation.is_enabled()) {
            (Some(sizes), true) => sizes.len() * 5,
            (Some(sizes), false) => sizes.len(),
            (None, true) => 5,
            (None, false) => 1,
        });
        if runs == 0 {
            return Err("--runs must be greater than zero".to_string());
        }
        if query_sizes.as_ref().is_some_and(|sizes| sizes.is_empty()) {
            return Err("--query-sizes must not be empty".to_string());
        }
        return Ok(Experiment::Repeated {
            table_size,
            blowup,
            runs,
            query_size,
            query_sizes,
            aggregation,
        });
    }

    match (table_size, query_size, blowup) {
        (Some(table_size), Some(query_size), Some(blowup)) => {
            cases.push((table_size, query_size, blowup));
        }
        (None, None, None) => {}
        _ => {
            return Err(
                "--table-size, --query-size, and --blowup must be provided together".to_string(),
            )
        }
    }

    if cases.is_empty() {
        return Err("no experiment cases provided".to_string());
    }
    Ok(Experiment::Cases(cases))
}

fn parse_case_list(raw_cases: &str) -> Result<Vec<(usize, usize, usize)>, String> {
    raw_cases
        .split(',')
        .filter(|case| !case.trim().is_empty())
        .map(parse_case)
        .collect()
}

fn parse_case(case: &str) -> Result<(usize, usize, usize), String> {
    let parts = case.split(':').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!(
            "invalid case '{case}', expected table_size:query_size:blowup"
        ));
    }
    Ok((
        parse_usize("table_size", parts[0])?,
        parse_usize("query_size", parts[1])?,
        parse_usize("blowup", parts[2])?,
    ))
}

fn parse_usize(name: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {name}: {value}"))
}

fn parse_usize_list(name: &str, value: &str) -> Result<Vec<usize>, String> {
    let values = value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .map(|item| parse_usize(name, item.trim()))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(format!("{name} must contain at least one value"));
    }
    Ok(values)
}

fn next_arg<'a>(args: &'a [String], index: usize, name: &str) -> Result<&'a str, String> {
    args.get(index + 1)
        .map(|value| value.as_str())
        .ok_or_else(|| format!("missing value for {name}"))
}

fn print_usage() {
    eprintln!(
        "Usage:\n  cargo run --example scale_experiment\n  cargo run --example scale_experiment -- --table-size 1024 --query-size 128 --blowup 4\n  cargo run --example scale_experiment -- --table-size 1024 --runs 10 --blowup 4\n  cargo run --example scale_experiment -- --table-size 1024 --query-size 128 --runs 10 --median --blowup 4\n  cargo run --example scale_experiment -- --table-size 1024 --query-sizes 8,16,32,64 --blowup 4\n  cargo run --example scale_experiment -- --table-size 1024 --query-sizes 8,16,32,64 --runs 20 --median --blowup 4\n  cargo run --example scale_experiment -- --case 256:32:4 --case 512:64:4\n  cargo run --example scale_experiment -- --cases 256:32:4,512:64:4\n  cargo run --example scale_experiment -- 1024 128 4"
    );
}

#[derive(Clone)]
struct ExperimentRow {
    table_preprocess_ms: f64,
    total_prover_ms: f64,
    total_verifier_ms: f64,
    preprocessing_bytes: usize,
    proof_bytes: usize,
}

fn row_from_timings(
    table_preprocess_ms: f64,
    timings: fri_lookup::lookup::LookupProveTimings,
    total_verifier_ms: f64,
    preprocessing_bytes: usize,
    proof_bytes: usize,
) -> ExperimentRow {
    ExperimentRow {
        table_preprocess_ms,
        total_prover_ms: timings.total_prover_ms,
        total_verifier_ms,
        preprocessing_bytes,
        proof_bytes,
    }
}

fn print_row(table_size: usize, query_size: usize, blowup: usize, row: &ExperimentRow) {
    println!(
        "{},{},{},{:.3},{:.3},{:.3},{},{}",
        table_size,
        query_size,
        blowup,
        row.table_preprocess_ms,
        row.total_prover_ms,
        row.total_verifier_ms,
        row.preprocessing_bytes,
        row.proof_bytes,
    );
}

fn run_stage<T>(
    name: &str,
    op: impl FnOnce() -> Result<T, fri_lookup::error::Error>,
) -> Result<T, fri_lookup::error::Error> {
    let diagnostics = scale_diagnostics_enabled();
    let start = Instant::now();
    if diagnostics {
        eprintln!("stage start: {name}");
    }
    match op() {
        Ok(value) => {
            if diagnostics {
                eprintln!(
                    "stage ok: {name}: {:.3} ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
            Ok(value)
        }
        Err(error) => {
            eprintln!(
                "stage failed: {name}: {:.3} ms: {error:?}",
                start.elapsed().as_secs_f64() * 1000.0
            );
            Err(error)
        }
    }
}

fn scale_diagnostics_enabled() -> bool {
    matches!(
        env::var("FRI_SCALE_DIAGNOSTICS")
            .ok()
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn run_case(
    table_size: usize,
    query_size: usize,
    blowup: usize,
) -> Result<(ExperimentRow, f64), fri_lookup::error::Error> {
    let table_values = (0..table_size)
        .map(|index| Fr::from((index as u64) + 1))
        .collect::<Vec<_>>();
    let table = Table::new(&table_values)?;

    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let query_values = (0..query_size)
        .map(|_| table_values[rng.gen_range(0..table_size)])
        .collect::<Vec<_>>();
    let witness = LookupWitness::new(&query_values)?;
    let statement = LookupStatement { query_size };

    warm_up_gpu_backends(table_size, query_size, blowup)?;

    let start = Instant::now();
    let params = FriLookup::preprocess(&table, query_size, blowup)?;
    let table_preprocess_ms = start.elapsed().as_secs_f64() * 1000.0;
    let preprocessing_bytes = public_parameters_size(&params);

    let (proof, timings) = FriLookup::prove_with_timings(&params, &witness, &statement)?;

    let proof_bytes = lookup_proof_size(&proof);

    let start = Instant::now();
    FriLookup::validate_preprocessing_parameters(&params)?;
    let tvalidate_ms = start.elapsed().as_secs_f64() * 1000.0;

    let start = Instant::now();
    FriLookup::verify(&params, &statement, &proof)?;
    let verify_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok((
        row_from_timings(
            table_preprocess_ms,
            timings,
            verify_ms,
            preprocessing_bytes,
            proof_bytes,
        ),
        tvalidate_ms,
    ))
}

fn warm_up_gpu_backends(
    table_size: usize,
    query_size: usize,
    blowup: usize,
) -> Result<(), fri_lookup::error::Error> {
    run_stage("warmup merkle", || {
        warm_up_gpu_merkle(table_size);
        Ok(())
    })?;

    let plan = fri_lookup::lde::DenseLdePlan::<Fr>::new(table_size, blowup)?;
    if fri_lookup::lde::select_dense_lde_backend::<Fr>(&plan)
        == fri_lookup::lde::DenseLdeBackendKind::Gpu
    {
        run_stage("warmup lde", || {
            let warmup_values = vec![Fr::from(0u64); table_size];
            fri_lookup::lde::evaluate_dense_lde_with_plan(&warmup_values, &plan)
        })?;
    }

    run_stage("warmup batch merkle", || {
        fri_lookup::fri::warm_up_gpu_batch_merkle::<Fr>(plan.lde_size, 2)
    })?;
    run_stage("warmup batch combine", || {
        fri_lookup::fri::warm_up_gpu_batch_combine::<Fr>(plan.lde_size, 3)
    })?;
    run_stage("warmup fold", || {
        fri_lookup::fri::warm_up_gpu_fold::<Fr>(plan.lde_size)
    })?;
    run_stage("warmup gpu resident fri", || {
        fri_lookup::fri::warm_up_gpu_resident_fri::<Fr>(plan.lde_size, 3, blowup)
    })?;
    run_stage("warmup usc", || {
        fri_lookup::usc::warm_up_gpu_usc::<Fr>(table_size.max(query_size), blowup)
    })?;
    run_stage("warmup batch reciprocals", || {
        fri_lookup::lookup::warm_up_gpu_batch_reciprocals::<Fr>(table_size + query_size)
    })
}

fn warm_up_gpu_merkle(table_size: usize) {
    let zero = Fr::from(0u64);
    let mut row_bytes = Vec::new();
    zero.write(&mut row_bytes)
        .expect("writing to Vec cannot fail");
    fri_lookup::merkle::warm_up_gpu(table_size, row_bytes.len());
}

fn run_repeated_table(
    table_size: usize,
    blowup: usize,
    runs: usize,
    fixed_query_size: Option<usize>,
    fixed_query_sizes: Option<Vec<usize>>,
    aggregation: Aggregation,
) -> Result<(), fri_lookup::error::Error> {
    let table_values = (0..table_size)
        .map(|index| Fr::from((index as u64) + 1))
        .collect::<Vec<_>>();
    let table = Table::new(&table_values)?;
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let warmup_query_size = fixed_query_sizes
        .as_ref()
        .and_then(|sizes| sizes.iter().copied().max())
        .or(fixed_query_size)
        .unwrap_or(table_size);
    warm_up_gpu_backends(table_size, warmup_query_size, blowup)?;

    let start = Instant::now();
    let reusable = run_stage("preprocess table", || {
        FriLookup::preprocess_table(&table, blowup)
    })?;
    let table_preprocess_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut summary_rows = BTreeMap::<usize, Vec<ExperimentRow>>::new();
    let mut tvalidate_ms = None;

    for run in 0..runs {
        let query_size = query_size_for_run(
            table_size,
            run,
            fixed_query_size,
            fixed_query_sizes.as_deref(),
        );
        let query_values = (0..query_size)
            .map(|_| table_values[rng.gen_range(0..table_size)])
            .collect::<Vec<_>>();
        let witness = LookupWitness::new(&query_values)?;
        let statement = LookupStatement { query_size };

        let params = FriLookup::parameters_for_query(&reusable, query_size).map_err(|error| {
            eprintln!(
                "stage failed: parameters_for_query run={run} query_size={query_size}: {error:?}"
            );
            error
        })?;
        let preprocessing_bytes = public_parameters_size(&params);

        let (proof, timings) = FriLookup::prove_with_timings(&params, &witness, &statement)
            .map_err(|error| {
                eprintln!("stage failed: prove run={run} query_size={query_size}: {error:?}");
                error
            })?;

        let proof_bytes = lookup_proof_size(&proof);

        if tvalidate_ms.is_none() {
            let start = Instant::now();
            FriLookup::validate_preprocessing_parameters(&params).map_err(|error| {
                eprintln!(
                    "stage failed: validate preprocessing parameters run={run} query_size={query_size}: {error:?}"
                );
                error
            })?;
            tvalidate_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
        }

        let start = Instant::now();
        FriLookup::verify(&params, &statement, &proof).map_err(|error| {
            eprintln!("stage failed: verify run={run} query_size={query_size}: {error:?}");
            error
        })?;
        let verify_ms = start.elapsed().as_secs_f64() * 1000.0;

        let row = row_from_timings(
            table_preprocess_ms,
            timings,
            verify_ms,
            preprocessing_bytes,
            proof_bytes,
        );
        print_row(table_size, query_size, blowup, &row);
        if aggregation.is_enabled() {
            summary_rows.entry(query_size).or_default().push(row);
        }
    }

    println!(
        "Tvalidate_ms,{:.3}",
        tvalidate_ms.expect("repeated experiments always run at least once")
    );

    match aggregation {
        Aggregation::None => {}
        Aggregation::Median => {
            for (query_size, rows) in summary_rows {
                let row = median_row(&rows);
                print_row(table_size, query_size, blowup, &row);
            }
        }
        Aggregation::Average => {
            for (query_size, rows) in summary_rows {
                let row = average_row(&rows);
                print_row(table_size, query_size, blowup, &row);
            }
        }
    }

    Ok(())
}

fn median_row(rows: &[ExperimentRow]) -> ExperimentRow {
    let mut sorted_rows = rows.iter().collect::<Vec<_>>();
    sorted_rows.sort_by(|left, right| {
        left.total_prover_ms
            .partial_cmp(&right.total_prover_ms)
            .expect("benchmark timings should not be NaN")
    });
    sorted_rows[sorted_rows.len() / 2].clone()
}

fn average_row(rows: &[ExperimentRow]) -> ExperimentRow {
    assert!(!rows.is_empty());
    let count = rows.len() as f64;
    let average_usize = |value: fn(&ExperimentRow) -> usize| -> usize {
        ((rows.iter().map(value).sum::<usize>() as f64) / count).round() as usize
    };
    ExperimentRow {
        table_preprocess_ms: rows.iter().map(|row| row.table_preprocess_ms).sum::<f64>() / count,
        total_prover_ms: rows.iter().map(|row| row.total_prover_ms).sum::<f64>() / count,
        total_verifier_ms: rows.iter().map(|row| row.total_verifier_ms).sum::<f64>() / count,
        preprocessing_bytes: average_usize(|row| row.preprocessing_bytes),
        proof_bytes: average_usize(|row| row.proof_bytes),
    }
}

fn query_size_for_run(
    table_size: usize,
    run: usize,
    fixed_query_size: Option<usize>,
    fixed_query_sizes: Option<&[usize]>,
) -> usize {
    if let Some(query_size) = fixed_query_size {
        return query_size;
    }
    if let Some(query_sizes) = fixed_query_sizes {
        return query_sizes[run % query_sizes.len()];
    }
    generated_query_size(table_size, run)
}

fn generated_query_size(table_size: usize, run: usize) -> usize {
    let min_query_size = 8.min(table_size).max(1);
    let max_exponent = table_size.trailing_zeros() as usize;
    let min_exponent = min_query_size.next_power_of_two().trailing_zeros() as usize;
    let span = max_exponent.saturating_sub(min_exponent) + 1;
    1usize << (min_exponent + run % span)
}

fn lookup_proof_size<F: PrimeField + FftField>(proof: &LookupProof<F>) -> usize {
    USIZE_BYTES
        + digest_size(&proof.query_root)
        + digest_size(&proof.multiplicity_root)
        + digest_size(&proof.c_qt)
        + digest_size(&proof.c_qf)
        + batch_fri_proof_size(&proof.table_batch)
        + batch_fri_proof_size(&proof.query_batch)
        + vec_size(&proof.table_ood_claimed_values, field_size)
        + vec_size(&proof.query_ood_claimed_values, field_size)
        + ood_opening_proof_size(&proof.table_ood_opening_proof)
        + ood_opening_proof_size(&proof.query_ood_opening_proof)
        + field_size(&proof.common_sum_s)
        + digest_size(&proof.c_p_a)
        + digest_size(&proof.c_p_v)
        + specialized_usc_proof_size(&proof.specialized_usc_table_proof)
        + specialized_usc_proof_size(&proof.specialized_usc_query_proof)
}

fn public_parameters_size<F: PrimeField + FftField>(params: &PublicParameters<F>) -> usize {
    table_size(&params.table)
        + USIZE_BYTES
        + USIZE_BYTES
        + domain_preprocessing_size(&params.domains)
        + protocol_structure_preprocessing_size(&params.protocol)
        + public_table_preprocessing_size(&params.table_preprocessing.polynomial_coefficients)
        + digest_size(&params.table_id)
        + digest_size(&params.table_lde_root)
        + fri_proof_size(&params.table_low_degree)
}

fn table_size<F: PrimeField + FftField>(table: &Table<F>) -> usize {
    USIZE_BYTES + vec_size(table.values(), field_size) + vec_size(table.values(), table_entry_size)
}

fn table_entry_size<F: Field + ToBytes>(value: &F) -> usize {
    vec_bytes_size(field_size(value)) + USIZE_BYTES
}

fn public_table_preprocessing_size<F: Field + ToBytes>(polynomial_coefficients: &[F]) -> usize {
    vec_size(polynomial_coefficients, field_size)
}

fn domain_preprocessing_size<F: PrimeField + FftField>(domains: &DomainPreprocessing<F>) -> usize {
    domain_side_preprocessing_size(&domains.table) + domain_side_preprocessing_size(&domains.query)
}

fn domain_side_preprocessing_size<F: PrimeField + FftField>(
    domain: &DomainSidePreprocessing<F>,
) -> usize {
    domain_info_size(&domain.base)
        + domain_info_size(&domain.lde)
        + fft_plan_size(&domain.fft_plan)
        + fft_plan_size(&domain.ifft_plan)
        + fft_plan_size(&domain.lde_fft_plan)
        + field_size(&domain.coset_shift)
        + vanishing_polynomial_size(&domain.vanishing_polynomial)
        + vec_size(&domain.vanishing_on_lde, field_size)
        + vec_size(&domain.fri_layers, fri_layer_preprocessing_size)
}

fn domain_info_size<F: PrimeField + FftField>(domain: &DomainInfo<F>) -> usize {
    USIZE_BYTES + vec_size(&domain.elements, field_size) + field_size(&domain.root_of_unity)
}

fn fft_plan_size<F: PrimeField + FftField>(plan: &FftPlanPreprocessing<F>) -> usize {
    USIZE_BYTES
        + field_size(&plan.size_inverse)
        + field_size(&plan.root_of_unity)
        + field_size(&plan.inverse_root_of_unity)
        + vec_size(&plan.bit_reversal, usize_size)
}

fn vanishing_polynomial_size<F: PrimeField + FftField>(
    polynomial: &VanishingPolynomial<F>,
) -> usize {
    USIZE_BYTES + field_size(&polynomial.constant) + field_size(&polynomial.leading)
}

fn fri_layer_preprocessing_size<F: PrimeField + FftField>(
    layer: &FriLayerPreprocessing<F>,
) -> usize {
    USIZE_BYTES
        + domain_info_size(&layer.domain)
        + vec_size(&layer.pair_indices, pair_usize_size)
        + vec_size(&layer.next_indices, usize_size)
}

fn protocol_structure_preprocessing_size<F: PrimeField + FftField>(
    protocol: &ProtocolStructurePreprocessing<F>,
) -> usize {
    vec_size(&protocol.merkle_leaf_layouts, merkle_leaf_layout_size)
        + vec_size(&protocol.table_batch_columns, column_label_size)
        + vec_size(&protocol.query_batch_columns, column_label_size)
        + vec_size(&protocol.usc_columns, column_label_size)
        + bytes_size(protocol.merkle_domain_separator)
        + bytes_size(protocol.merkle_node_separator)
        + bytes_size(protocol.opening_index_rule.as_bytes())
        + bytes_size(protocol.fri_query_to_merkle_rule.as_bytes())
        + degree_normalization_size(&protocol.batch_degree_normalization)
        + degree_normalizers_on_lde_size(&protocol.table_degree_normalizers_on_lde)
        + degree_normalizers_on_lde_size(&protocol.query_degree_normalizers_on_lde)
        + vec_size(&protocol.transcript_order, str_ref_size)
        + vec_size(&protocol.verifier_challenge_order, str_ref_size)
}

fn degree_normalizers_on_lde_size<F: FftField + Field + ToBytes>(
    normalizers: &DegreeNormalizersOnLde<F>,
) -> usize {
    USIZE_BYTES
        + vec_size(&normalizers.exponents, usize_size)
        + normalizers
            .materialized()
            .map_or(0, |values| vec_size(values, vec_field_size))
}

fn merkle_leaf_layout_size(layout: &MerkleLeafLayout) -> usize {
    bytes_size(layout.name.as_bytes())
        + bytes_size(layout.tag)
        + bytes_size(layout.round)
        + bytes_size(layout.domain)
        + vec_size(&layout.columns, column_label_size)
}

fn column_label_size(label: &ColumnLabel) -> usize {
    bytes_size(label.label) + USIZE_BYTES
}

fn degree_normalization_size<F: FftField>(
    normalization: &DegreeNormalizationPreprocessing<F>,
) -> usize {
    USIZE_BYTES
        + USIZE_BYTES
        + vec_size(&normalization.table_exponents, usize_size)
        + vec_size(&normalization.query_exponents, usize_size)
}

fn bytes_size(bytes: &[u8]) -> usize {
    VEC_LEN_BYTES + bytes.len()
}

fn vec_bytes_size(byte_count: usize) -> usize {
    VEC_LEN_BYTES + byte_count
}

fn vec_field_size<F: Field + ToBytes>(values: &Vec<F>) -> usize {
    vec_size(values, field_size)
}

fn pair_usize_size(_: &(usize, usize)) -> usize {
    USIZE_BYTES + USIZE_BYTES
}

fn str_ref_size(value: &&'static str) -> usize {
    bytes_size(value.as_bytes())
}

fn ood_opening_proof_size<F: PrimeField + FftField>(proof: &OodOpeningProof<F>) -> usize {
    batch_fri_proof_size(&proof.quotient_batch)
        + vec_size(&proof.row_openings, ood_row_opening_size)
}

fn ood_row_opening_size<F: PrimeField + FftField>(opening: &OodRowOpening<F>) -> usize {
    USIZE_BYTES
        + vec_size(&opening.original_row, field_size)
        + vec_size(&opening.original_group_paths, merkle_path_size)
        + batch_lde_row_opening_size(&opening.quotient_opening)
}

fn specialized_usc_proof_size<F: PrimeField + FftField>(proof: &SpecializedUscProof<F>) -> usize {
    batch_fri_proof_size(&proof.batch) + vec_size(&proof.openings, specialized_usc_opening_size)
}

fn specialized_usc_opening_size<F: PrimeField + FftField>(
    opening: &SpecializedUscOpening<F>,
) -> usize {
    USIZE_BYTES
        + vec_size(&opening.existing_group_row, field_size)
        + merkle_path_size(&opening.existing_group_path)
        + batch_lde_row_opening_size(&opening.p_opening)
}

fn batch_lde_row_opening_size<F: PrimeField + FftField>(opening: &BatchLdeRowOpening<F>) -> usize {
    USIZE_BYTES
        + vec_size(&opening.row, field_size)
        + vec_size(&opening.group_paths, merkle_path_size)
        + field_size(&opening.combined_lde_value)
        + merkle_path_size(&opening.combined_lde_path)
}

fn batch_fri_proof_size<F: PrimeField + FftField>(proof: &BatchFriProof<F>) -> usize {
    digest_size(&proof.row_root)
        + USIZE_BYTES
        + USIZE_BYTES
        + vec_size(&proof.column_degree_bounds, usize_size)
        + vec_size(&proof.commitment_groups, batch_commitment_group_size)
        + vec_size(&proof.row_openings, batch_row_opening_size)
        + vec_size(&proof.batching_coefficients, field_size)
        + fri_proof_size(&proof.combined)
}

fn fri_proof_size<F: PrimeField + FftField>(proof: &FriProof<F>) -> usize {
    digest_size(&proof.root)
        + vec_size(&proof.roots, digest_size)
        + vec_size(&proof.final_evals, field_size)
        + vec_size(&proof.queries, fri_query_size)
        + USIZE_BYTES
        + USIZE_BYTES
        + USIZE_BYTES
}

fn fri_query_size<F: PrimeField + FftField>(query: &FriQuery<F>) -> usize {
    USIZE_BYTES + vec_size(&query.openings, fri_layer_opening_size)
}

fn fri_layer_opening_size<F: PrimeField + FftField>(opening: &FriLayerOpening<F>) -> usize {
    USIZE_BYTES
        + field_size(&opening.value)
        + field_size(&opening.sibling_value)
        + merkle_path_size(&opening.path)
        + merkle_path_size(&opening.sibling_path)
}

fn batch_row_opening_size<F: PrimeField + FftField>(
    opening: &fri_lookup::fri::BatchRowOpening<F>,
) -> usize {
    USIZE_BYTES
        + vec_size(&opening.row, field_size)
        + merkle_path_size(&opening.row_path)
        + vec_size(&opening.group_paths, merkle_path_size)
        + field_size(&opening.combined_lde_value)
        + merkle_path_size(&opening.combined_lde_path)
}

fn batch_commitment_group_size(group: &fri_lookup::fri::BatchCommitmentGroup) -> usize {
    vec_size(&group.columns, usize_size) + digest_size(&group.root)
}

fn merkle_path_size(path: &MerklePath) -> usize {
    USIZE_BYTES + vec_size(&path.siblings, digest_size)
}

fn digest_size(_: &Digest32) -> usize {
    32
}

fn usize_size(_: &usize) -> usize {
    USIZE_BYTES
}

fn field_size<F: Field + ToBytes>(value: &F) -> usize {
    let mut bytes = Vec::new();
    value.write(&mut bytes).expect("writing to Vec cannot fail");
    bytes.len()
}

fn vec_size<T>(values: &[T], item_size: fn(&T) -> usize) -> usize {
    VEC_LEN_BYTES + values.iter().map(item_size).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::{average_row, ExperimentRow};

    #[test]
    fn average_row_uses_all_verification_timings() {
        let rows = (1..=5)
            .map(|run| ExperimentRow {
                table_preprocess_ms: run as f64,
                total_prover_ms: (run * 10) as f64,
                total_verifier_ms: (run * 100) as f64,
                preprocessing_bytes: run,
                proof_bytes: run * 10,
            })
            .collect::<Vec<_>>();

        let average = average_row(&rows);

        assert_eq!(average.table_preprocess_ms, 3.0);
        assert_eq!(average.total_prover_ms, 30.0);
        assert_eq!(average.total_verifier_ms, 300.0);
        assert_eq!(average.preprocessing_bytes, 3);
        assert_eq!(average.proof_bytes, 30);
    }
}
