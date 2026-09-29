use ark_ff::{batch_inversion, FftField, PrimeField, ToBytes};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use sha3::{Digest, Keccak256};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{
    error::Error,
    fri::{
        self, BatchCommitmentGroup, BatchFriProof, FriProof, OodOpeningProof, OodRowOpening,
        PreparedBatchCommitments, PreparedBatchFriProof,
    },
    merkle::{Digest32, MerklePath, MerkleTree},
    preprocessing::{
        preprocess_protocol_structure_with_shared_table_normalizers, preprocess_query_domain,
        preprocess_table_degree_normalizers_on_lde, preprocess_table_domain,
        DegreeNormalizersOnLde, DomainPreprocessing, DomainSidePreprocessing,
        ProtocolStructurePreprocessing,
    },
    table::Table,
    transcript::Transcript,
    usc::{self, SpecializedUscOpening, SpecializedUscProof},
    utils::is_power_of_two,
    PROTOCOL_NAME,
};

pub const LOOKUP_PROOF_VERSION: u32 = 3;
pub const DEFAULT_GPU_INV_MIN_SIZE: usize = 1 << 20;

const PARALLEL_PROVE_ENV: &str = "FRI_LOOKUP_PARALLEL_PROVE";
const PROVE_DIAGNOSTICS_ENV: &str = "FRI_LOOKUP_PROVE_DIAGNOSTICS";
const SCALE_DIAGNOSTICS_ENV: &str = "FRI_SCALE_DIAGNOSTICS";

#[derive(Clone, Debug)]
pub struct LookupWitness<F: FftField> {
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct LookupStatement {
    pub query_size: usize,
}

#[derive(Clone, Debug)]
pub struct PublicParameters<F: FftField> {
    pub table: Arc<Table<F>>,
    pub blowup: usize,
    pub query_size: usize,
    pub domains: DomainPreprocessing<F>,
    pub protocol: ProtocolStructurePreprocessing<F>,
    pub table_preprocessing: Arc<PublicTablePreprocessing<F>>,
    pub table_id: Digest32,
    pub table_lde_root: Digest32,
    pub table_low_degree: Arc<FriProof<F>>,
    table_validation_cache: Arc<Mutex<Option<ValidatedTablePreprocessing<F>>>>,
}

#[derive(Clone, Debug)]
pub struct ReusableTablePreprocessing<F: FftField> {
    pub table: Arc<Table<F>>,
    pub blowup: usize,
    pub table_domain: Arc<DomainSidePreprocessing<F>>,
    pub table_preprocessing: Arc<PublicTablePreprocessing<F>>,
    pub table_id: Digest32,
    pub table_lde_root: Digest32,
    pub table_low_degree: Arc<FriProof<F>>,
    pub table_degree_normalizers_on_lde: Arc<DegreeNormalizersOnLde<F>>,
    table_validation_cache: Arc<Mutex<Option<ValidatedTablePreprocessing<F>>>>,
}

#[derive(Debug)]
struct ValidatedTablePreprocessing<F: FftField> {
    table: Arc<Table<F>>,
    table_domain: Arc<DomainSidePreprocessing<F>>,
    table_id: Digest32,
    table_lde_root: Digest32,
    table_low_degree: Arc<FriProof<F>>,
    blowup: usize,
}

impl<F: FftField> ValidatedTablePreprocessing<F> {
    fn matches(&self, params: &PublicParameters<F>) -> bool {
        Arc::ptr_eq(&self.table, &params.table)
            && Arc::ptr_eq(&self.table_domain, &params.domains.table)
            && self.table_id == params.table_id
            && self.table_lde_root == params.table_lde_root
            && Arc::ptr_eq(&self.table_low_degree, &params.table_low_degree)
            && self.blowup == params.blowup
    }
}

#[derive(Clone, Debug)]
pub struct PublicTablePreprocessing<F: FftField> {
    pub polynomial_coefficients: Vec<F>,
    lde: PublicTableLde<F>,
}

#[derive(Clone, Debug)]
struct PublicTableLde<F: FftField> {
    root: Digest32,
    cpu: Arc<Mutex<Option<Arc<MaterializedTableLde<F>>>>>,
    #[cfg(feature = "icicle-rust")]
    gpu: Option<fri::PreparedGpuTableLde<F>>,
}

#[derive(Clone, Debug)]
pub struct MaterializedTableLde<F: FftField> {
    pub values: Vec<F>,
    pub tree: MerkleTree,
}

impl<F: PrimeField + FftField + Send + Sync> PublicTablePreprocessing<F> {
    fn from_prepared_lde(
        polynomial_coefficients: Vec<F>,
        lde: fri::PreparedTableLde<F>,
    ) -> Result<Self, Error> {
        let root = lde.root();
        match lde {
            fri::PreparedTableLde::Cpu { values, tree } => {
                if tree.root() != root {
                    return Err(Error::MerkleRootMismatch);
                }
                Ok(Self {
                    polynomial_coefficients,
                    lde: PublicTableLde {
                        root,
                        cpu: Arc::new(Mutex::new(Some(Arc::new(MaterializedTableLde {
                            values,
                            tree,
                        })))),
                        #[cfg(feature = "icicle-rust")]
                        gpu: None,
                    },
                })
            }
            #[cfg(feature = "icicle-rust")]
            fri::PreparedTableLde::Gpu(gpu) => Ok(Self {
                polynomial_coefficients,
                lde: PublicTableLde {
                    root,
                    cpu: Arc::new(Mutex::new(None)),
                    gpu: Some(gpu),
                },
            }),
        }
    }

    pub fn lde_root(&self) -> Digest32 {
        self.lde.root
    }

    pub fn lde_len(&self, table_size: usize, blowup: usize) -> usize {
        table_size * blowup
    }

    pub fn materialized_lde(
        &self,
        table_size: usize,
        blowup: usize,
    ) -> Result<Arc<MaterializedTableLde<F>>, Error> {
        let mut guard = self
            .lde
            .cpu
            .lock()
            .map_err(|_| Error::LowDegreeCheckFailed)?;
        if let Some(materialized) = guard.as_ref() {
            return Ok(Arc::clone(materialized));
        }

        let materialized = if let Some((values, tree)) = self.try_materialize_gpu_lde()? {
            MaterializedTableLde { values, tree }
        } else {
            let values =
                fri::evaluate_lde_from_coeffs(&self.polynomial_coefficients, table_size, blowup)?;
            let tree = tree_for_column(&values);
            MaterializedTableLde { values, tree }
        };
        if materialized.values.len() != table_size * blowup {
            return Err(Error::InvalidDomainSize(materialized.values.len()));
        }
        if materialized.tree.root() != self.lde.root {
            return Err(Error::MerkleRootMismatch);
        }
        let materialized = Arc::new(materialized);
        *guard = Some(Arc::clone(&materialized));
        Ok(materialized)
    }

    pub fn open_lde_many(
        &self,
        indices: &[usize],
        table_size: usize,
        blowup: usize,
    ) -> Result<Vec<(F, MerklePath)>, Error> {
        #[cfg(feature = "icicle-rust")]
        if let Some(gpu) = self.lde.gpu.as_ref() {
            return gpu.open_many(indices);
        }

        let materialized = self.materialized_lde(table_size, blowup)?;
        indices
            .iter()
            .copied()
            .map(|index| {
                let value = materialized
                    .values
                    .get(index)
                    .copied()
                    .ok_or(Error::InvalidDomainSize(index))?;
                Ok((value, materialized.tree.open(index)))
            })
            .collect()
    }

    #[cfg(feature = "icicle-rust")]
    fn gpu_lde(&self) -> Option<&fri::PreparedGpuTableLde<F>> {
        self.lde.gpu.as_ref()
    }

    fn try_materialize_gpu_lde(&self) -> Result<Option<(Vec<F>, MerkleTree)>, Error> {
        #[cfg(feature = "icicle-rust")]
        {
            if let Some(gpu) = self.lde.gpu.as_ref() {
                return gpu.materialize().map(Some);
            }
        }
        Ok(None)
    }
}

#[derive(Clone, Debug)]
pub struct LookupProof<F: FftField> {
    pub proof_version: u32,
    pub query_root: Digest32,
    pub multiplicity_root: Digest32,
    pub c_qt: Digest32,
    pub c_qf: Digest32,
    pub table_batch: BatchFriProof<F>,
    pub query_batch: BatchFriProof<F>,
    pub table_ood_claimed_values: Vec<F>,
    pub query_ood_claimed_values: Vec<F>,
    pub table_ood_opening_proof: OodOpeningProof<F>,
    pub query_ood_opening_proof: OodOpeningProof<F>,
    pub common_sum_s: F,
    pub c_p_a: Digest32,
    pub c_p_v: Digest32,
    pub specialized_usc_table_proof: SpecializedUscProof<F>,
    pub specialized_usc_query_proof: SpecializedUscProof<F>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LookupProveTimings {
    pub commit_f_m_ms: f64,
    pub construct_u_a_v_ms: f64,
    pub construct_qt_qf_ms: f64,
    pub main_batch_fri_ms: f64,
    pub ood_opening_table_ms: f64,
    pub ood_opening_query_ms: f64,
    pub specialized_usc_table_ms: f64,
    pub specialized_usc_query_ms: f64,
    pub total_prover_ms: f64,
}

impl LookupProveTimings {
    pub fn measured_stage_ms(&self) -> f64 {
        self.commit_f_m_ms
            + self.construct_u_a_v_ms
            + self.construct_qt_qf_ms
            + self.main_batch_fri_ms
            + self.ood_opening_table_ms
            + self.ood_opening_query_ms
            + self.specialized_usc_table_ms
            + self.specialized_usc_query_ms
    }

    pub fn slowest_stage(&self) -> (&'static str, f64) {
        let stages = [
            ("commit_f_m", self.commit_f_m_ms),
            ("construct_u_a_v", self.construct_u_a_v_ms),
            ("construct_qt_qf", self.construct_qt_qf_ms),
            ("main_batch_fri", self.main_batch_fri_ms),
            ("ood_opening_table", self.ood_opening_table_ms),
            ("ood_opening_query", self.ood_opening_query_ms),
            ("specialized_usc_table", self.specialized_usc_table_ms),
            ("specialized_usc_query", self.specialized_usc_query_ms),
        ];
        let mut slowest = stages[0];
        for stage in stages.iter().copied().skip(1) {
            if stage.1 > slowest.1 {
                slowest = stage;
            }
        }
        slowest
    }
}

#[derive(Clone, Debug)]
pub struct ColumnOpening<F: FftField> {
    pub index: usize,
    pub value: F,
    pub path: MerklePath,
}

#[derive(Clone, Debug)]
pub struct ConstraintQuotient<F: FftField> {
    pub coeffs: Vec<F>,
    pub base_evals: Vec<F>,
    pub lde_evals: Vec<F>,
}

#[derive(Clone, Debug)]
struct InitialWitnessPrecommit<F: FftField> {
    lde: Vec<F>,
    coeffs: Vec<F>,
    commitments: PreparedBatchCommitments,
}

#[derive(Clone, Debug)]
struct MultiplicityPrecommit<F: FftField> {
    values: Vec<F>,
    support: Vec<usize>,
    lde: Vec<F>,
    coeffs: Vec<F>,
    commitments: PreparedBatchCommitments,
}

#[derive(Clone, Debug)]
struct TableAuxiliaryPrecommit<F: FftField> {
    reciprocal: Vec<F>,
    reciprocal_lde: Vec<F>,
    reciprocal_coeffs: Vec<F>,
    accumulator: Vec<F>,
    accumulator_lde: Vec<F>,
    accumulator_coeffs: Vec<F>,
    commitments: PreparedBatchCommitments,
}

#[derive(Clone, Debug)]
struct QueryAuxiliaryPrecommit<F: FftField> {
    reciprocal: Vec<F>,
    reciprocal_lde: Vec<F>,
    reciprocal_coeffs: Vec<F>,
    commitments: PreparedBatchCommitments,
}

enum OodPrecommit<F: FftField> {
    Cpu {
        quotient_columns: Vec<Vec<F>>,
        quotient_lde_columns: Vec<Vec<F>>,
        commitments: PreparedBatchCommitments,
        quotient_degree_bounds: Vec<usize>,
    },
    Gpu(fri::PreparedGpuOodOpeningBatch<F>),
}

impl<F: PrimeField + FftField + Send + Sync> OodPrecommit<F> {
    fn root(&self) -> Digest32 {
        match self {
            Self::Cpu { commitments, .. } => commitments.commitment_groups()[0].root,
            Self::Gpu(precommit) => precommit.root(),
        }
    }
}

enum SpecializedUscPrecommit<F: FftField> {
    Cpu {
        p_values: Vec<F>,
        p_lde: Vec<F>,
        commitments: PreparedBatchCommitments,
    },
    Gpu(fri::PreparedGpuSpecializedUscBatch<F>),
}

impl<F: PrimeField + FftField + Send + Sync> SpecializedUscPrecommit<F> {
    fn root(&self) -> Digest32 {
        match self {
            Self::Cpu { commitments, .. } => commitments.commitment_groups()[0].root,
            Self::Gpu(precommit) => precommit.root(),
        }
    }
}

pub struct FriLookup;

impl<F: FftField> LookupWitness<F> {
    pub fn new(values: &[F]) -> Result<Self, Error> {
        if !is_power_of_two(values.len()) {
            return Err(Error::WitnessSizeNotPow2(values.len()));
        }
        Ok(Self {
            values: values.to_vec(),
        })
    }
}

impl FriLookup {
    pub fn preprocess<F: PrimeField + FftField + Send + Sync>(
        table: &Table<F>,
        query_size: usize,
        blowup: usize,
    ) -> Result<PublicParameters<F>, Error> {
        let table_preprocessing = Self::preprocess_table(table, blowup)?;
        Self::parameters_for_query(&table_preprocessing, query_size)
    }

    pub fn preprocess_table<F: PrimeField + FftField + Send + Sync>(
        table: &Table<F>,
        blowup: usize,
    ) -> Result<ReusableTablePreprocessing<F>, Error> {
        let table_domain = Arc::new(preprocess_table_domain(table.size(), blowup)?);
        let table_id = compute_table_identifier(table, &table_domain);
        let table_degree_normalizers_on_lde = Arc::new(preprocess_table_degree_normalizers_on_lde(
            table.size(),
            blowup,
        )?);
        let (table_polynomial_coefficients, table_lde, table_low_degree) = if let Some(prepared) =
            fri::try_prepare_table_low_degree_gpu_resident(table.values(), table.size(), blowup)?
        {
            (
                prepared.polynomial_coefficients,
                prepared.lde,
                prepared.proof,
            )
        } else {
            let table_polynomial = fri::interpolate(table.values())?;
            if let Some(prepared) = fri::try_prepare_low_degree_from_coeffs_gpu_resident(
                &table_polynomial.coeffs,
                table.size(),
                blowup,
            )? {
                (
                    table_polynomial.coeffs,
                    fri::PreparedTableLde::Cpu {
                        values: prepared.lde_values,
                        tree: prepared.lde_tree,
                    },
                    prepared.proof,
                )
            } else {
                let table_lde_values =
                    fri::evaluate_lde_from_coeffs(&table_polynomial.coeffs, table.size(), blowup)?;
                let table_lde_tree = tree_for_column(&table_lde_values);
                let table_low_degree =
                    fri::prove_low_degree_from_lde(&table_lde_values, table.size(), blowup)?;
                (
                    table_polynomial.coeffs,
                    fri::PreparedTableLde::Cpu {
                        values: table_lde_values,
                        tree: table_lde_tree,
                    },
                    table_low_degree,
                )
            }
        };
        let table_lde_root = table_lde.root();
        if table_lde_root != table_low_degree.root {
            return Err(Error::MerkleRootMismatch);
        }
        let table_preprocessing =
            PublicTablePreprocessing::from_prepared_lde(table_polynomial_coefficients, table_lde)?;

        Ok(ReusableTablePreprocessing {
            table: Arc::new(table.clone()),
            blowup,
            table_domain,
            table_preprocessing: Arc::new(table_preprocessing),
            table_id,
            table_lde_root,
            table_low_degree: Arc::new(table_low_degree),
            table_degree_normalizers_on_lde,
            table_validation_cache: Arc::new(Mutex::new(None)),
        })
    }

    pub fn parameters_for_query<F: FftField>(
        table_preprocessing: &ReusableTablePreprocessing<F>,
        query_size: usize,
    ) -> Result<PublicParameters<F>, Error> {
        if !is_power_of_two(query_size) {
            return Err(Error::WitnessSizeNotPow2(query_size));
        }
        let domains = DomainPreprocessing {
            table: Arc::clone(&table_preprocessing.table_domain),
            query: preprocess_query_domain(query_size, table_preprocessing.blowup)?,
        };
        let protocol = preprocess_protocol_structure_with_shared_table_normalizers(
            table_preprocessing.table.size(),
            query_size,
            table_preprocessing.blowup,
            Arc::clone(&table_preprocessing.table_degree_normalizers_on_lde),
        )?;

        Ok(PublicParameters {
            table: Arc::clone(&table_preprocessing.table),
            blowup: table_preprocessing.blowup,
            query_size,
            domains,
            protocol,
            table_preprocessing: Arc::clone(&table_preprocessing.table_preprocessing),
            table_id: table_preprocessing.table_id,
            table_lde_root: table_preprocessing.table_lde_root,
            table_low_degree: Arc::clone(&table_preprocessing.table_low_degree),
            table_validation_cache: Arc::clone(&table_preprocessing.table_validation_cache),
        })
    }

    pub fn prove<F: PrimeField + FftField + Send + Sync>(
        params: &PublicParameters<F>,
        witness: &LookupWitness<F>,
        statement: &LookupStatement,
    ) -> Result<LookupProof<F>, Error> {
        Self::prove_with_timings(params, witness, statement).map(|(proof, _)| proof)
    }

    pub fn prove_with_timings<F: PrimeField + FftField + Send + Sync>(
        params: &PublicParameters<F>,
        witness: &LookupWitness<F>,
        statement: &LookupStatement,
    ) -> Result<(LookupProof<F>, LookupProveTimings), Error> {
        let total_start = Instant::now();
        let mut timings = LookupProveTimings::default();

        if witness.values.len() != statement.query_size || params.query_size != statement.query_size
        {
            return Err(Error::StatementMismatch);
        }

        let stage_start = Instant::now();
        let (multiplicity_precommit, query_precommit) =
            construct_initial_witness_precommits(params, witness)?;
        let MultiplicityPrecommit {
            values: multiplicity,
            support: multiplicity_support,
            lde: multiplicity_lde,
            coeffs: multiplicity_coeffs,
            commitments: table_m_commitments,
        } = multiplicity_precommit;
        let InitialWitnessPrecommit {
            lde: query_lde,
            coeffs: query_value_coeffs,
            commitments: query_f_commitments,
        } = query_precommit;
        let multiplicity_root = table_m_commitments.commitment_groups()[0].root;
        let query_root = query_f_commitments.commitment_groups()[0].root;
        timings.commit_f_m_ms = elapsed_ms(stage_start.elapsed());

        let mut transcript = lookup_transcript(params, statement);
        transcript.absorb_bytes(b"tlookup/query-root", &query_root);
        transcript.absorb_bytes(b"tlookup/multiplicity-root", &multiplicity_root);
        let beta = transcript.challenge::<F>(b"tlookup/beta");

        let stage_start = Instant::now();
        let (table_auxiliary_precommit, query_auxiliary_precommit) =
            prove_diag_stage("prove.construct_auxiliary_precommits", || {
                construct_auxiliary_precommits(
                    params.table.values(),
                    &witness.values,
                    &multiplicity,
                    &multiplicity_support,
                    beta,
                    params.table.size(),
                    statement.query_size,
                    params.blowup,
                )
            })?;
        let TableAuxiliaryPrecommit {
            reciprocal: table_reciprocal,
            reciprocal_lde: mut table_reciprocal_lde,
            reciprocal_coeffs: table_reciprocal_coeffs,
            accumulator: table_accumulator,
            accumulator_lde: mut table_accumulator_lde,
            accumulator_coeffs: table_accumulator_coeffs,
            commitments: table_auxiliary_commitments,
        } = table_auxiliary_precommit;
        let QueryAuxiliaryPrecommit {
            reciprocal: query_reciprocal,
            reciprocal_lde: mut query_reciprocal_lde,
            reciprocal_coeffs: query_reciprocal_coeffs,
            commitments: query_auxiliary_commitments,
        } = query_auxiliary_precommit;
        let table_auxiliary_root = table_auxiliary_commitments.commitment_groups()[0].root;
        let query_auxiliary_root = query_auxiliary_commitments.commitment_groups()[0].root;
        transcript.absorb_bytes(b"tlookup/commitment/table-U-A", &table_auxiliary_root);
        transcript.absorb_bytes(b"tlookup/commitment/query-V", &query_auxiliary_root);
        timings.construct_u_a_v_ms = elapsed_ms(stage_start.elapsed());

        let eta_t1 = transcript.challenge_nonzero::<F>(b"tlookup/table-constraint-batching/eta1");
        let eta_t2 = transcript.challenge_nonzero::<F>(b"tlookup/table-constraint-batching/eta2");

        let stage_start = Instant::now();
        let (
            mut table_quotient,
            table_quotient_commitments,
            mut query_quotient,
            query_quotient_commitments,
        ) = prove_diag_stage("prove.construct_quotients_and_commitments", || {
            construct_quotients_and_commitments(
                params,
                statement,
                &multiplicity,
                &table_reciprocal,
                &table_accumulator,
                &witness.values,
                &query_reciprocal,
                &multiplicity_coeffs,
                &table_reciprocal_coeffs,
                &table_accumulator_coeffs,
                &query_value_coeffs,
                &query_reciprocal_coeffs,
                beta,
                eta_t1,
                eta_t2,
            )
        })?;
        let c_qt = table_quotient_commitments.commitment_groups()[0].root;
        let c_qf = query_quotient_commitments.commitment_groups()[0].root;
        transcript.absorb_bytes(b"tlookup/commitment/Q_T", &c_qt);
        transcript.absorb_bytes(b"tlookup/commitment/Q_F", &c_qf);
        timings.construct_qt_qf_ms = elapsed_ms(stage_start.elapsed());

        let table_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/main-batch-fri/table", 4);
        let query_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/main-batch-fri/query", 3);

        let table_main_commitments = fri::concatenate_batch_commitments(vec![
            table_m_commitments,
            table_auxiliary_commitments,
            table_quotient_commitments,
        ])?;
        let query_main_commitments = fri::concatenate_batch_commitments(vec![
            query_f_commitments,
            query_auxiliary_commitments,
            query_quotient_commitments,
        ])?;

        let stage_start = Instant::now();
        let mut table_lde_materialized: Option<Arc<MaterializedTableLde<F>>> = None;
        #[cfg(feature = "icicle-rust")]
        let mut table_gpu_lde_store = if let Some(table_lde) = params.table_preprocessing.gpu_lde()
        {
            prove_diag_stage("prove.prepare_table_gpu_lde_store_from_table", || {
                fri::try_prepare_gpu_lde_column_store_from_table_lde_and_commitments(
                    table_lde,
                    &table_main_commitments,
                    params.table.size(),
                    params.blowup,
                )
            })?
        } else {
            None
        };
        #[cfg(not(feature = "icicle-rust"))]
        let mut table_gpu_lde_store = None;
        if table_gpu_lde_store.is_none() {
            let materialized = params
                .table_preprocessing
                .materialized_lde(params.table.size(), params.blowup)?;
            table_gpu_lde_store =
                prove_diag_stage("prove.prepare_table_gpu_lde_store_from_commitments", || {
                    fri::try_prepare_gpu_lde_column_store_from_commitments(
                        &[materialized.values.as_slice()],
                        &table_main_commitments,
                        params.table.size(),
                        params.blowup,
                    )
                })?;
            table_lde_materialized = Some(materialized);
        }
        let mut query_gpu_lde_store =
            prove_diag_stage("prove.prepare_query_gpu_lde_store_from_commitments", || {
                fri::try_prepare_gpu_lde_column_store_from_commitments(
                    &[],
                    &query_main_commitments,
                    statement.query_size,
                    params.blowup,
                )
            })?;

        if table_gpu_lde_store.is_none() {
            ensure_quotient_lde(&mut table_quotient, params.table.size(), params.blowup)?;
        }
        if query_gpu_lde_store.is_none() {
            ensure_quotient_lde(&mut query_quotient, statement.query_size, params.blowup)?;
        }
        if table_gpu_lde_store.is_none() {
            if table_reciprocal_lde.is_empty() {
                table_reciprocal_lde = fri::evaluate_lde_from_coeffs(
                    &table_reciprocal_coeffs,
                    params.table.size(),
                    params.blowup,
                )?;
            }
            if table_accumulator_lde.is_empty() {
                table_accumulator_lde = fri::evaluate_auto_sparse_lde_with_support(
                    &table_accumulator,
                    params.blowup,
                    &multiplicity_support,
                )?;
            }
        }
        if query_gpu_lde_store.is_none() && query_reciprocal_lde.is_empty() {
            query_reciprocal_lde = fri::evaluate_lde_from_coeffs(
                &query_reciprocal_coeffs,
                statement.query_size,
                params.blowup,
            )?;
        }

        let table_columns = [
            multiplicity.as_slice(),
            table_reciprocal.as_slice(),
            table_accumulator.as_slice(),
            table_quotient.base_evals.as_slice(),
        ];
        let query_columns = [
            witness.values.as_slice(),
            query_reciprocal.as_slice(),
            query_quotient.base_evals.as_slice(),
        ];
        let table_lde_columns = [
            multiplicity_lde.as_slice(),
            table_reciprocal_lde.as_slice(),
            table_accumulator_lde.as_slice(),
            table_quotient.lde_evals.as_slice(),
        ];
        let query_lde_columns = [
            query_lde.as_slice(),
            query_reciprocal_lde.as_slice(),
            query_quotient.lde_evals.as_slice(),
        ];
        if table_gpu_lde_store.is_none() && table_lde_materialized.is_none() {
            table_lde_materialized = Some(
                params
                    .table_preprocessing
                    .materialized_lde(params.table.size(), params.blowup)?,
            );
        }
        let table_lde_values_for_cpu = table_lde_materialized
            .as_ref()
            .map(|materialized| materialized.values.as_slice())
            .unwrap_or(&[]);
        let table_ood_lde_columns = [
            table_lde_values_for_cpu,
            multiplicity_lde.as_slice(),
            table_reciprocal_lde.as_slice(),
            table_accumulator_lde.as_slice(),
            table_quotient.lde_evals.as_slice(),
        ];
        let query_ood_lde_columns = [
            query_lde.as_slice(),
            query_reciprocal_lde.as_slice(),
            query_quotient.lde_evals.as_slice(),
        ];

        if table_gpu_lde_store.is_none() {
            table_gpu_lde_store =
                prove_diag_stage("prove.prepare_table_gpu_lde_store_from_ood_columns", || {
                    fri::try_prepare_gpu_lde_column_store(
                        &table_ood_lde_columns,
                        params.table.size(),
                        params.blowup,
                    )
                })?;
        }
        if query_gpu_lde_store.is_none() {
            query_gpu_lde_store =
                prove_diag_stage("prove.prepare_query_gpu_lde_store_from_ood_columns", || {
                    fri::try_prepare_gpu_lde_column_store(
                        &query_ood_lde_columns,
                        statement.query_size,
                        params.blowup,
                    )
                })?;
        }
        let (table_batch_prepared, query_batch_prepared) =
            prove_diag_stage("prove.prepare_main_batches", || {
                prepare_main_batches(
                    params,
                    statement,
                    &table_columns,
                    &table_lde_columns,
                    &table_coefficients,
                    table_main_commitments,
                    table_gpu_lde_store.as_ref(),
                    &query_columns,
                    &query_lde_columns,
                    &query_coefficients,
                    query_main_commitments,
                    query_gpu_lde_store.as_ref(),
                )
            })?;
        let table_batch = table_batch_prepared.proof_with_openings(&[])?;
        let query_batch = query_batch_prepared.proof_with_openings(&[])?;
        if table_batch.commitment_groups[1].root != table_auxiliary_root
            || query_batch.commitment_groups[1].root != query_auxiliary_root
            || table_batch.commitment_groups[2].root != c_qt
            || query_batch.commitment_groups[2].root != c_qf
        {
            return Err(Error::MerkleRootMismatch);
        }
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/main-batch-fri/table",
            &table_batch,
        );
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/main-batch-fri/query",
            &query_batch,
        );
        timings.main_batch_fri_ms = elapsed_ms(stage_start.elapsed());

        let zeta_t = challenge_outside_lde_domain::<F>(
            &mut transcript,
            b"tlookup/ood-point/table",
            params.table.size() * params.blowup,
        );
        let zeta_f = challenge_outside_lde_domain::<F>(
            &mut transcript,
            b"tlookup/ood-point/query",
            statement.query_size * params.blowup,
        );

        let table_coeffs = vec![
            params.table_preprocessing.polynomial_coefficients.clone(),
            multiplicity_coeffs.clone(),
            table_reciprocal_coeffs.clone(),
            table_accumulator_coeffs.clone(),
            table_quotient.coeffs.clone(),
        ];
        let query_coeffs = vec![
            query_value_coeffs.clone(),
            query_reciprocal_coeffs.clone(),
            query_quotient.coeffs.clone(),
        ];
        let table_ood_claimed_values = table_coeffs
            .iter()
            .map(|coeffs| evaluate_coeffs(coeffs, zeta_t))
            .collect::<Vec<_>>();
        let query_ood_claimed_values = query_coeffs
            .iter()
            .map(|coeffs| evaluate_coeffs(coeffs, zeta_f))
            .collect::<Vec<_>>();
        absorb_field_vec(
            &mut transcript,
            b"tlookup/ood-claims/table",
            &table_ood_claimed_values,
        );
        absorb_field_vec(
            &mut transcript,
            b"tlookup/ood-claims/query",
            &query_ood_claimed_values,
        );

        let (table_ood_precommit, query_ood_precommit) =
            prove_diag_stage("prove.construct_ood_precommits", || {
                construct_ood_precommits(
                    &table_coeffs,
                    &table_ood_lde_columns,
                    &table_ood_claimed_values,
                    zeta_t,
                    params.table.size(),
                    table_gpu_lde_store.as_ref(),
                    &query_coeffs,
                    &query_ood_lde_columns,
                    &query_ood_claimed_values,
                    zeta_f,
                    statement.query_size,
                    query_gpu_lde_store.as_ref(),
                    params.blowup,
                )
            })?;
        transcript.absorb_bytes(
            b"tlookup/ood-quotient-commitment/table",
            &table_ood_precommit.root(),
        );
        transcript.absorb_bytes(
            b"tlookup/ood-quotient-commitment/query",
            &query_ood_precommit.root(),
        );
        let table_ood_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/ood-batching/table", 5);
        let query_ood_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/ood-batching/query", 3);

        let (table_ood_opening_proof, table_ood_ms, query_ood_opening_proof, query_ood_ms) =
            if should_parallelize_lookup_prove() {
                std::thread::scope(|scope| {
                    let table_handle = scope.spawn(|| {
                        let stage_start = Instant::now();
                        let proof = prove_diag_stage("prove.finish_table_ood_proof", || {
                            finish_table_ood_proof(
                                params,
                                &table_batch_prepared,
                                table_ood_precommit,
                                &table_ood_coefficients,
                            )
                        })?;
                        Ok((proof, elapsed_ms(stage_start.elapsed())))
                    });
                    let query_handle = scope.spawn(|| {
                        let stage_start = Instant::now();
                        let proof = prove_diag_stage("prove.finish_query_ood_proof", || {
                            finish_query_ood_proof(
                                &query_batch_prepared,
                                query_ood_precommit,
                                &query_ood_coefficients,
                                statement.query_size,
                                params.blowup,
                            )
                        })?;
                        Ok((proof, elapsed_ms(stage_start.elapsed())))
                    });
                    let (table_proof, table_ms) = join_prover_thread(table_handle)?;
                    let (query_proof, query_ms) = join_prover_thread(query_handle)?;
                    Ok((table_proof, table_ms, query_proof, query_ms))
                })?
            } else {
                let stage_start = Instant::now();
                let table_proof = prove_diag_stage("prove.finish_table_ood_proof", || {
                    finish_table_ood_proof(
                        params,
                        &table_batch_prepared,
                        table_ood_precommit,
                        &table_ood_coefficients,
                    )
                })?;
                let table_ms = elapsed_ms(stage_start.elapsed());
                let stage_start = Instant::now();
                let query_proof = prove_diag_stage("prove.finish_query_ood_proof", || {
                    finish_query_ood_proof(
                        &query_batch_prepared,
                        query_ood_precommit,
                        &query_ood_coefficients,
                        statement.query_size,
                        params.blowup,
                    )
                })?;
                let query_ms = elapsed_ms(stage_start.elapsed());
                (table_proof, table_ms, query_proof, query_ms)
            };
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/ood-batch-fri/table",
            &table_ood_opening_proof.quotient_batch,
        );
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/ood-batch-fri/query",
            &query_ood_opening_proof.quotient_batch,
        );
        timings.ood_opening_table_ms = table_ood_ms;
        timings.ood_opening_query_ms = query_ood_ms;

        let common_sum_s = table_accumulator.iter().copied().sum::<F>();
        transcript.absorb(b"tlookup/common-sum-s", &common_sum_s);

        let (table_usc_precommit, query_usc_precommit) =
            prove_diag_stage("prove.construct_specialized_usc_precommits", || {
                construct_specialized_usc_precommits(
                    &table_accumulator_coeffs,
                    table_accumulator_lde.as_slice(),
                    params.table.size(),
                    table_gpu_lde_store.as_ref(),
                    &query_reciprocal_coeffs,
                    query_reciprocal_lde.as_slice(),
                    statement.query_size,
                    query_gpu_lde_store.as_ref(),
                    common_sum_s,
                    params.blowup,
                )
            })?;
        let c_p_a = table_usc_precommit.root();
        let c_p_v = query_usc_precommit.root();
        transcript.absorb_bytes(b"tlookup/commitment/p_A", &c_p_a);
        transcript.absorb_bytes(b"tlookup/commitment/p_V", &c_p_v);
        let table_usc_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/usc/table", 1);
        let query_usc_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/usc/query", 1);

        let (specialized_usc_table_proof, table_usc_ms, specialized_usc_query_proof, query_usc_ms) =
            if should_parallelize_lookup_prove() {
                std::thread::scope(|scope| {
                    let table_handle = scope.spawn(|| {
                        let stage_start = Instant::now();
                        let proof = prove_specialized_usc(
                            table_usc_precommit,
                            &table_usc_coefficients,
                            params.table.size(),
                            params.blowup,
                            |indices| {
                                table_existing_group_openings_for_column(
                                    &table_batch_prepared,
                                    indices,
                                    1,
                                )
                            },
                        )?;
                        Ok((proof, elapsed_ms(stage_start.elapsed())))
                    });
                    let query_handle = scope.spawn(|| {
                        let stage_start = Instant::now();
                        let proof = prove_specialized_usc(
                            query_usc_precommit,
                            &query_usc_coefficients,
                            statement.query_size,
                            params.blowup,
                            |indices| {
                                query_existing_group_openings_for_column(
                                    &query_batch_prepared,
                                    indices,
                                    1,
                                )
                            },
                        )?;
                        Ok((proof, elapsed_ms(stage_start.elapsed())))
                    });
                    let (table_proof, table_ms) = join_prover_thread(table_handle)?;
                    let (query_proof, query_ms) = join_prover_thread(query_handle)?;
                    Ok((table_proof, table_ms, query_proof, query_ms))
                })?
            } else {
                let stage_start = Instant::now();
                let table_proof = prove_specialized_usc(
                    table_usc_precommit,
                    &table_usc_coefficients,
                    params.table.size(),
                    params.blowup,
                    |indices| {
                        table_existing_group_openings_for_column(&table_batch_prepared, indices, 1)
                    },
                )?;
                let table_ms = elapsed_ms(stage_start.elapsed());
                let stage_start = Instant::now();
                let query_proof = prove_specialized_usc(
                    query_usc_precommit,
                    &query_usc_coefficients,
                    statement.query_size,
                    params.blowup,
                    |indices| {
                        query_existing_group_openings_for_column(&query_batch_prepared, indices, 1)
                    },
                )?;
                let query_ms = elapsed_ms(stage_start.elapsed());
                (table_proof, table_ms, query_proof, query_ms)
            };
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/usc-batch-fri/table",
            &specialized_usc_table_proof.batch,
        );
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/usc-batch-fri/query",
            &specialized_usc_query_proof.batch,
        );
        timings.specialized_usc_table_ms = table_usc_ms;
        timings.specialized_usc_query_ms = query_usc_ms;

        timings.total_prover_ms = elapsed_ms(total_start.elapsed());

        Ok((
            LookupProof {
                proof_version: LOOKUP_PROOF_VERSION,
                query_root,
                multiplicity_root,
                c_qt,
                c_qf,
                table_batch,
                query_batch,
                table_ood_claimed_values,
                query_ood_claimed_values,
                table_ood_opening_proof,
                query_ood_opening_proof,
                common_sum_s,
                c_p_a,
                c_p_v,
                specialized_usc_table_proof,
                specialized_usc_query_proof,
            },
            timings,
        ))
    }

    pub fn verify<F: PrimeField + FftField + Send + Sync>(
        params: &PublicParameters<F>,
        statement: &LookupStatement,
        proof: &LookupProof<F>,
    ) -> Result<(), Error> {
        if proof.proof_version != LOOKUP_PROOF_VERSION {
            return Err(Error::InvalidProofVersion);
        }
        verify_proof_shape(params, statement, proof)?;
        Self::validate_preprocessing_parameters(params)?;

        let mut transcript = lookup_transcript(params, statement);
        transcript.absorb_bytes(b"tlookup/query-root", &proof.query_root);
        transcript.absorb_bytes(b"tlookup/multiplicity-root", &proof.multiplicity_root);
        let beta = transcript.challenge::<F>(b"tlookup/beta");

        transcript.absorb_bytes(
            b"tlookup/commitment/table-U-A",
            &proof.table_batch.commitment_groups[1].root,
        );
        transcript.absorb_bytes(
            b"tlookup/commitment/query-V",
            &proof.query_batch.commitment_groups[1].root,
        );
        let eta_t1 = transcript.challenge_nonzero::<F>(b"tlookup/table-constraint-batching/eta1");
        let eta_t2 = transcript.challenge_nonzero::<F>(b"tlookup/table-constraint-batching/eta2");

        transcript.absorb_bytes(b"tlookup/commitment/Q_T", &proof.c_qt);
        transcript.absorb_bytes(b"tlookup/commitment/Q_F", &proof.c_qf);
        let table_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/main-batch-fri/table", 4);
        let query_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/main-batch-fri/query", 3);
        if proof.table_batch.batching_coefficients != table_coefficients
            || proof.query_batch.batching_coefficients != query_coefficients
        {
            return Err(Error::CommitmentMismatch("main batching coefficients"));
        }

        fri::verify_batch_low_degree(&proof.table_batch, params.table.size(), params.blowup)?;
        fri::verify_batch_low_degree(&proof.query_batch, statement.query_size, params.blowup)?;
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/main-batch-fri/table",
            &proof.table_batch,
        );
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/main-batch-fri/query",
            &proof.query_batch,
        );

        let zeta_t = challenge_outside_lde_domain::<F>(
            &mut transcript,
            b"tlookup/ood-point/table",
            params.table.size() * params.blowup,
        );
        let zeta_f = challenge_outside_lde_domain::<F>(
            &mut transcript,
            b"tlookup/ood-point/query",
            statement.query_size * params.blowup,
        );
        if proof.table_ood_claimed_values.len() != 5 || proof.query_ood_claimed_values.len() != 3 {
            return Err(Error::InvalidOutOfDomainOpening);
        }
        absorb_field_vec(
            &mut transcript,
            b"tlookup/ood-claims/table",
            &proof.table_ood_claimed_values,
        );
        absorb_field_vec(
            &mut transcript,
            b"tlookup/ood-claims/query",
            &proof.query_ood_claimed_values,
        );
        transcript.absorb_bytes(
            b"tlookup/ood-quotient-commitment/table",
            &proof
                .table_ood_opening_proof
                .quotient_batch
                .commitment_groups[0]
                .root,
        );
        transcript.absorb_bytes(
            b"tlookup/ood-quotient-commitment/query",
            &proof
                .query_ood_opening_proof
                .quotient_batch
                .commitment_groups[0]
                .root,
        );
        let table_ood_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/ood-batching/table", 5);
        let query_ood_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/ood-batching/query", 3);
        if proof
            .table_ood_opening_proof
            .quotient_batch
            .batching_coefficients
            != table_ood_coefficients
            || proof
                .query_ood_opening_proof
                .quotient_batch
                .batching_coefficients
                != query_ood_coefficients
        {
            return Err(Error::CommitmentMismatch("ood batching coefficients"));
        }

        let table_original_groups = table_ood_original_groups(params, proof);
        let query_original_groups = query_ood_original_groups(proof);
        let table_quotient_degree_bounds =
            vec![usc::specialized_usc_column_degree_bound(params.table.size()); 5];
        let query_quotient_degree_bounds =
            vec![usc::specialized_usc_column_degree_bound(statement.query_size); 3];
        fri::fri_batch_verify_out_of_domain(
            &table_original_groups,
            &proof.table_ood_claimed_values,
            &table_quotient_degree_bounds,
            params.table.size(),
            params.blowup,
            zeta_t,
            &proof.table_ood_opening_proof,
        )?;
        fri::fri_batch_verify_out_of_domain(
            &query_original_groups,
            &proof.query_ood_claimed_values,
            &query_quotient_degree_bounds,
            statement.query_size,
            params.blowup,
            zeta_f,
            &proof.query_ood_opening_proof,
        )?;
        verify_table_quotient_identity(
            &proof.table_ood_claimed_values,
            zeta_t,
            params.table.size(),
            beta,
            eta_t1,
            eta_t2,
        )?;
        verify_query_quotient_identity(
            &proof.query_ood_claimed_values,
            zeta_f,
            statement.query_size,
            beta,
        )?;
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/ood-batch-fri/table",
            &proof.table_ood_opening_proof.quotient_batch,
        );
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/ood-batch-fri/query",
            &proof.query_ood_opening_proof.quotient_batch,
        );

        transcript.absorb(b"tlookup/common-sum-s", &proof.common_sum_s);
        transcript.absorb_bytes(b"tlookup/commitment/p_A", &proof.c_p_a);
        transcript.absorb_bytes(b"tlookup/commitment/p_V", &proof.c_p_v);
        let table_usc_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/usc/table", 1);
        let query_usc_coefficients =
            challenge_vector::<F>(&mut transcript, b"tlookup/usc/query", 1);
        if proof
            .specialized_usc_table_proof
            .batch
            .batching_coefficients
            != table_usc_coefficients
            || proof
                .specialized_usc_query_proof
                .batch
                .batching_coefficients
                != query_usc_coefficients
        {
            return Err(Error::CommitmentMismatch("usc batching coefficients"));
        }
        usc::specialized_usc_verify(
            proof.table_batch.commitment_groups[1].root,
            1,
            proof.common_sum_s,
            params.table.size(),
            params.blowup,
            &table_usc_coefficients,
            &proof.specialized_usc_table_proof,
        )?;
        usc::specialized_usc_verify(
            proof.query_batch.commitment_groups[1].root,
            0,
            proof.common_sum_s,
            statement.query_size,
            params.blowup,
            &query_usc_coefficients,
            &proof.specialized_usc_query_proof,
        )?;
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/usc-batch-fri/table",
            &proof.specialized_usc_table_proof.batch,
        );
        absorb_batch_fri_transcript(
            &mut transcript,
            b"tlookup/usc-batch-fri/query",
            &proof.specialized_usc_query_proof.batch,
        );

        Ok(())
    }

    pub fn validate_preprocessing_parameters<F: PrimeField + FftField + Send + Sync>(
        params: &PublicParameters<F>,
    ) -> Result<(), Error> {
        validate_public_table_preprocessing(params)
    }
}

fn construct_initial_witness_precommits<F: PrimeField + FftField + Send + Sync>(
    params: &PublicParameters<F>,
    witness: &LookupWitness<F>,
) -> Result<(MultiplicityPrecommit<F>, InitialWitnessPrecommit<F>), Error> {
    let build_multiplicity = || build_multiplicity_precommit(params, &witness.values);
    let build_query = || build_initial_query_precommit(&witness.values, params.blowup);

    if should_parallelize_lookup_prove() {
        std::thread::scope(|scope| {
            let multiplicity_handle = scope.spawn(build_multiplicity);
            let query_handle = scope.spawn(build_query);
            let multiplicity_precommit = join_prover_thread(multiplicity_handle)?;
            let query_precommit = join_prover_thread(query_handle)?;
            Ok((multiplicity_precommit, query_precommit))
        })
    } else {
        Ok((build_multiplicity()?, build_query()?))
    }
}

fn build_multiplicity_precommit<F: PrimeField + FftField>(
    params: &PublicParameters<F>,
    query_values: &[F],
) -> Result<MultiplicityPrecommit<F>, Error> {
    let mut multiplicity = vec![F::zero(); params.table.size()];
    for value in query_values {
        let index = params
            .table
            .index_of(value)
            .ok_or_else(|| Error::ValueNotInTable(format!("{}", value)))?;
        multiplicity[index] += F::one();
    }
    let support = fri::non_zero_indices(&multiplicity);
    let lde = fri::evaluate_auto_sparse_lde_with_support(&multiplicity, params.blowup, &support)?;
    let coeffs = polynomial_coefficients(&multiplicity)?;
    let commitments = fri::prepare_batch_commitment_group(&[lde.as_slice()], vec![0])?;
    Ok(MultiplicityPrecommit {
        values: multiplicity,
        support,
        lde,
        coeffs,
        commitments,
    })
}

fn build_initial_query_precommit<F: PrimeField + FftField>(
    query_values: &[F],
    blowup: usize,
) -> Result<InitialWitnessPrecommit<F>, Error> {
    let coeffs = polynomial_coefficients(query_values)?;
    let lde = fri::evaluate_lde_from_coeffs(&coeffs, query_values.len(), blowup)?;
    let commitments = fri::prepare_batch_commitment_group(&[lde.as_slice()], vec![0])?;
    Ok(InitialWitnessPrecommit {
        lde,
        coeffs,
        commitments,
    })
}

#[allow(clippy::too_many_arguments)]
fn construct_auxiliary_precommits<F: PrimeField + FftField + Send + Sync>(
    table_values: &[F],
    query_values: &[F],
    multiplicity: &[F],
    multiplicity_support: &[usize],
    beta: F,
    table_row_count: usize,
    query_row_count: usize,
    blowup: usize,
) -> Result<(TableAuxiliaryPrecommit<F>, QueryAuxiliaryPrecommit<F>), Error> {
    if let Some(gpu_precommit) = prove_diag_stage("prove.construct_auxiliary.gpu_resident", || {
        fri::try_prepare_lookup_auxiliary_gpu_resident(
            table_values,
            multiplicity,
            query_values,
            beta,
            table_row_count,
            query_row_count,
            blowup,
        )
    })? {
        let table_precommit = TableAuxiliaryPrecommit {
            reciprocal: gpu_precommit.table_reciprocal,
            reciprocal_lde: Vec::new(),
            reciprocal_coeffs: gpu_precommit.table_reciprocal_coeffs,
            accumulator: gpu_precommit.table_accumulator,
            accumulator_lde: Vec::new(),
            accumulator_coeffs: gpu_precommit.table_accumulator_coeffs,
            commitments: gpu_precommit.table_commitments,
        };
        let query_precommit = QueryAuxiliaryPrecommit {
            reciprocal: gpu_precommit.query_reciprocal,
            reciprocal_lde: Vec::new(),
            reciprocal_coeffs: gpu_precommit.query_reciprocal_coeffs,
            commitments: gpu_precommit.query_commitments,
        };
        return Ok((table_precommit, query_precommit));
    }

    let (table_reciprocal, query_reciprocal) = batch_reciprocals(table_values, query_values, beta)?;
    let build_table = || {
        build_table_auxiliary_precommit(
            multiplicity,
            multiplicity_support,
            table_reciprocal,
            table_row_count,
            blowup,
        )
    };
    let build_query = || build_query_auxiliary_precommit(query_reciprocal, query_row_count, blowup);

    if should_parallelize_lookup_prove() {
        std::thread::scope(|scope| {
            let table_handle = scope.spawn(build_table);
            let query_handle = scope.spawn(build_query);
            let table_precommit = join_prover_thread(table_handle)?;
            let query_precommit = join_prover_thread(query_handle)?;
            Ok((table_precommit, query_precommit))
        })
    } else {
        Ok((build_table()?, build_query()?))
    }
}

fn build_table_auxiliary_precommit<F: PrimeField + FftField>(
    multiplicity: &[F],
    multiplicity_support: &[usize],
    table_reciprocal: Vec<F>,
    row_count: usize,
    blowup: usize,
) -> Result<TableAuxiliaryPrecommit<F>, Error> {
    if multiplicity.len() != row_count || table_reciprocal.len() != row_count {
        return Err(Error::InvalidDomainSize(row_count));
    }
    let table_accumulator = multiplicity
        .iter()
        .zip(&table_reciprocal)
        .map(|(mult, reciprocal)| *mult * *reciprocal)
        .collect::<Vec<_>>();
    let table_reciprocal_coeffs = polynomial_coefficients(&table_reciprocal)?;
    let table_reciprocal_lde =
        fri::evaluate_lde_from_coeffs(&table_reciprocal_coeffs, row_count, blowup)?;
    let table_accumulator_lde = fri::evaluate_auto_sparse_lde_with_support(
        &table_accumulator,
        blowup,
        multiplicity_support,
    )?;
    let table_accumulator_coeffs = polynomial_coefficients(&table_accumulator)?;
    let commitments = fri::prepare_batch_commitment_group(
        &[
            table_reciprocal_lde.as_slice(),
            table_accumulator_lde.as_slice(),
        ],
        vec![1, 2],
    )?;
    Ok(TableAuxiliaryPrecommit {
        reciprocal: table_reciprocal,
        reciprocal_lde: table_reciprocal_lde,
        reciprocal_coeffs: table_reciprocal_coeffs,
        accumulator: table_accumulator,
        accumulator_lde: table_accumulator_lde,
        accumulator_coeffs: table_accumulator_coeffs,
        commitments,
    })
}

fn build_query_auxiliary_precommit<F: PrimeField + FftField>(
    query_reciprocal: Vec<F>,
    row_count: usize,
    blowup: usize,
) -> Result<QueryAuxiliaryPrecommit<F>, Error> {
    if query_reciprocal.len() != row_count {
        return Err(Error::InvalidDomainSize(row_count));
    }
    let query_reciprocal_coeffs = polynomial_coefficients(&query_reciprocal)?;
    let query_reciprocal_lde =
        fri::evaluate_lde_from_coeffs(&query_reciprocal_coeffs, row_count, blowup)?;
    let commitments =
        fri::prepare_batch_commitment_group(&[query_reciprocal_lde.as_slice()], vec![1])?;
    Ok(QueryAuxiliaryPrecommit {
        reciprocal: query_reciprocal,
        reciprocal_lde: query_reciprocal_lde,
        reciprocal_coeffs: query_reciprocal_coeffs,
        commitments,
    })
}

#[allow(clippy::too_many_arguments)]
fn construct_quotients_and_commitments<F: PrimeField + FftField + Send + Sync>(
    params: &PublicParameters<F>,
    statement: &LookupStatement,
    multiplicity: &[F],
    table_reciprocal: &[F],
    table_accumulator: &[F],
    query_values: &[F],
    query_reciprocal: &[F],
    multiplicity_coeffs: &[F],
    table_reciprocal_coeffs: &[F],
    table_accumulator_coeffs: &[F],
    query_value_coeffs: &[F],
    query_reciprocal_coeffs: &[F],
    beta: F,
    eta_t1: F,
    eta_t2: F,
) -> Result<
    (
        ConstraintQuotient<F>,
        PreparedBatchCommitments,
        ConstraintQuotient<F>,
        PreparedBatchCommitments,
    ),
    Error,
> {
    let build_table = || {
        check_table_constraint_on_domain(
            params.table.values(),
            multiplicity,
            table_reciprocal,
            table_accumulator,
            beta,
            eta_t1,
            eta_t2,
        )?;
        if let Some(prepared) = prove_diag_stage("prove.construct_quotients.table.gpu", || {
            fri::try_prepare_table_constraint_quotient_gpu_resident(
                &params.table_preprocessing.polynomial_coefficients,
                multiplicity_coeffs,
                table_reciprocal_coeffs,
                params.table.size(),
                params.blowup,
                eta_t1,
                eta_t2,
                vec![3],
            )
        })? {
            return Ok(constraint_quotient_from_prepared(prepared));
        }

        let coeffs = table_constraint_quotient_coeffs_from_coeffs(
            &params.table_preprocessing.polynomial_coefficients,
            multiplicity_coeffs,
            table_reciprocal_coeffs,
            table_accumulator_coeffs,
            params.table.size(),
            eta_t1,
            eta_t2,
        )?;
        build_constraint_quotient_and_commitment(
            coeffs,
            params.table.size(),
            params.blowup,
            vec![3],
        )
    };

    let build_query = || {
        check_query_constraint_on_domain(query_values, query_reciprocal, beta)?;
        if let Some(prepared) = prove_diag_stage("prove.construct_quotients.query.gpu", || {
            fri::try_prepare_query_constraint_quotient_gpu_resident(
                query_value_coeffs,
                query_reciprocal_coeffs,
                statement.query_size,
                params.blowup,
                vec![2],
            )
        })? {
            return Ok(constraint_quotient_from_prepared(prepared));
        }

        let coeffs = query_constraint_quotient_coeffs_from_coeffs(
            query_value_coeffs,
            query_reciprocal_coeffs,
            statement.query_size,
        )?;
        build_constraint_quotient_and_commitment(
            coeffs,
            statement.query_size,
            params.blowup,
            vec![2],
        )
    };

    if should_parallelize_lookup_prove() {
        std::thread::scope(|scope| {
            let table_handle = scope.spawn(build_table);
            let query_handle = scope.spawn(build_query);
            let (table_quotient, table_commitments) = join_prover_thread(table_handle)?;
            let (query_quotient, query_commitments) = join_prover_thread(query_handle)?;
            Ok((
                table_quotient,
                table_commitments,
                query_quotient,
                query_commitments,
            ))
        })
    } else {
        let (table_quotient, table_commitments) = build_table()?;
        let (query_quotient, query_commitments) = build_query()?;
        Ok((
            table_quotient,
            table_commitments,
            query_quotient,
            query_commitments,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_main_batches<'a, F: PrimeField + FftField + Send + Sync>(
    params: &PublicParameters<F>,
    statement: &LookupStatement,
    table_columns: &[&'a [F]],
    table_lde_columns: &[&'a [F]],
    table_coefficients: &[F],
    table_commitments: PreparedBatchCommitments,
    table_gpu_lde_store: Option<&'a fri::PreparedGpuLdeColumnStore<F>>,
    query_columns: &[&'a [F]],
    query_lde_columns: &[&'a [F]],
    query_coefficients: &[F],
    query_commitments: PreparedBatchCommitments,
    query_gpu_lde_store: Option<&'a fri::PreparedGpuLdeColumnStore<F>>,
) -> Result<(PreparedBatchFriProof<'a, F>, PreparedBatchFriProof<'a, F>), Error> {
    let prepare_table = move || {
        prove_diag_stage("prove.prepare_main_batches.table", || {
            if let Some(store) = table_gpu_lde_store {
                fri::prepare_batch_low_degree_with_commitments_and_gpu_lde_store(
                    table_columns,
                    table_lde_columns,
                    table_coefficients,
                    &[params.table.size(); 4],
                    params.table.size(),
                    params.blowup,
                    table_commitments,
                    store,
                    &[1, 2, 3, 4],
                )
            } else {
                fri::prepare_batch_low_degree_with_commitments_and_lde(
                    table_columns,
                    table_lde_columns,
                    table_coefficients,
                    &[params.table.size(); 4],
                    params.table.size(),
                    params.blowup,
                    table_commitments,
                )
            }
        })
    };
    let prepare_query = move || {
        prove_diag_stage("prove.prepare_main_batches.query", || {
            if let Some(store) = query_gpu_lde_store {
                fri::prepare_batch_low_degree_with_commitments_and_gpu_lde_store(
                    query_columns,
                    query_lde_columns,
                    query_coefficients,
                    &[statement.query_size; 3],
                    statement.query_size,
                    params.blowup,
                    query_commitments,
                    store,
                    &[0, 1, 2],
                )
            } else {
                fri::prepare_batch_low_degree_with_commitments_and_lde(
                    query_columns,
                    query_lde_columns,
                    query_coefficients,
                    &[statement.query_size; 3],
                    statement.query_size,
                    params.blowup,
                    query_commitments,
                )
            }
        })
    };

    if should_parallelize_lookup_prove() {
        std::thread::scope(|scope| {
            let table_handle = scope.spawn(prepare_table);
            let query_handle = scope.spawn(prepare_query);
            let table_batch_prepared = join_prover_thread(table_handle)?;
            let query_batch_prepared = join_prover_thread(query_handle)?;
            Ok((table_batch_prepared, query_batch_prepared))
        })
    } else {
        Ok((prepare_table()?, prepare_query()?))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_table_constraint_quotient<F: PrimeField + FftField>(
    table_values: &[F],
    multiplicity: &[F],
    table_reciprocal: &[F],
    table_accumulator: &[F],
    beta: F,
    eta_t1: F,
    eta_t2: F,
    blowup: usize,
) -> Result<ConstraintQuotient<F>, Error> {
    let n = table_values.len();
    if multiplicity.len() != n || table_reciprocal.len() != n || table_accumulator.len() != n {
        return Err(Error::InvalidDomainSize(n));
    }
    check_table_constraint_on_domain(
        table_values,
        multiplicity,
        table_reciprocal,
        table_accumulator,
        beta,
        eta_t1,
        eta_t2,
    )?;
    let t = polynomial_coefficients(table_values)?;
    let m = polynomial_coefficients(multiplicity)?;
    let u = polynomial_coefficients(table_reciprocal)?;
    let a = polynomial_coefficients(table_accumulator)?;
    build_table_constraint_quotient_from_coeffs(&t, &m, &u, &a, n, eta_t1, eta_t2, blowup)
}

#[allow(clippy::too_many_arguments)]
fn build_table_constraint_quotient_from_coeffs<F: PrimeField + FftField + Send + Sync>(
    t: &[F],
    m: &[F],
    u: &[F],
    a: &[F],
    domain_size: usize,
    eta_t1: F,
    eta_t2: F,
    blowup: usize,
) -> Result<ConstraintQuotient<F>, Error> {
    let coeffs =
        table_constraint_quotient_coeffs_from_coeffs(t, m, u, a, domain_size, eta_t1, eta_t2)?;
    quotient_from_coeffs(coeffs, domain_size, blowup)
}

#[allow(clippy::too_many_arguments)]
fn table_constraint_quotient_coeffs_from_coeffs<F: PrimeField + FftField + Send + Sync>(
    t: &[F],
    m: &[F],
    u: &[F],
    _a: &[F],
    domain_size: usize,
    eta_t1: F,
    eta_t2: F,
) -> Result<Vec<F>, Error> {
    let u_t_high = product_high_coeffs(u, t, domain_size)?;
    let m_u_high = product_high_coeffs(m, u, domain_size)?;
    let len = u_t_high.len().max(m_u_high.len());
    let mut quotient = vec![F::zero(); len];
    for (index, target) in quotient.iter_mut().enumerate() {
        if let Some(value) = u_t_high.get(index) {
            *target += *value * eta_t1;
        }
        if let Some(value) = m_u_high.get(index) {
            *target -= *value * eta_t2;
        }
    }
    trim_trailing_zeros(&mut quotient);
    if quotient.is_empty() {
        quotient.push(F::zero());
    }
    Ok(quotient)
}

pub fn build_query_constraint_quotient<F: PrimeField + FftField>(
    query_values: &[F],
    query_reciprocal: &[F],
    beta: F,
    blowup: usize,
) -> Result<ConstraintQuotient<F>, Error> {
    let n = query_values.len();
    if query_reciprocal.len() != n {
        return Err(Error::InvalidDomainSize(n));
    }
    check_query_constraint_on_domain(query_values, query_reciprocal, beta)?;
    let f = polynomial_coefficients(query_values)?;
    let v = polynomial_coefficients(query_reciprocal)?;
    build_query_constraint_quotient_from_coeffs(&f, &v, n, blowup)
}

fn build_query_constraint_quotient_from_coeffs<F: PrimeField + FftField + Send + Sync>(
    f: &[F],
    v: &[F],
    domain_size: usize,
    blowup: usize,
) -> Result<ConstraintQuotient<F>, Error> {
    let coeffs = query_constraint_quotient_coeffs_from_coeffs(f, v, domain_size)?;
    quotient_from_coeffs(coeffs, domain_size, blowup)
}

fn query_constraint_quotient_coeffs_from_coeffs<F: PrimeField + FftField + Send + Sync>(
    f: &[F],
    v: &[F],
    domain_size: usize,
) -> Result<Vec<F>, Error> {
    let mut quotient = product_high_coeffs(v, f, domain_size)?;
    trim_trailing_zeros(&mut quotient);
    if quotient.is_empty() {
        quotient.push(F::zero());
    }
    Ok(quotient)
}

#[allow(clippy::too_many_arguments)]
fn check_table_constraint_on_domain<F: FftField>(
    table_values: &[F],
    multiplicity: &[F],
    table_reciprocal: &[F],
    table_accumulator: &[F],
    beta: F,
    eta_t1: F,
    eta_t2: F,
) -> Result<(), Error> {
    let n = table_values.len();
    if multiplicity.len() != n || table_reciprocal.len() != n || table_accumulator.len() != n {
        return Err(Error::InvalidDomainSize(n));
    }
    for (((table_value, multiplicity), reciprocal), accumulator) in table_values
        .iter()
        .zip(multiplicity)
        .zip(table_reciprocal)
        .zip(table_accumulator)
    {
        let first = *reciprocal * (beta + *table_value) - F::one();
        let second = *accumulator - (*multiplicity * *reciprocal);
        if eta_t1 * first + eta_t2 * second != F::zero() {
            return Err(Error::NonExactTableConstraintDivision);
        }
    }
    Ok(())
}

fn check_query_constraint_on_domain<F: FftField>(
    query_values: &[F],
    query_reciprocal: &[F],
    beta: F,
) -> Result<(), Error> {
    let n = query_values.len();
    if query_reciprocal.len() != n {
        return Err(Error::InvalidDomainSize(n));
    }
    for (query_value, reciprocal) in query_values.iter().zip(query_reciprocal) {
        if *reciprocal * (beta + *query_value) - F::one() != F::zero() {
            return Err(Error::NonExactQueryConstraintDivision);
        }
    }
    Ok(())
}

fn constraint_quotient_from_prepared<F: FftField>(
    prepared: fri::PreparedConstraintQuotient<F>,
) -> (ConstraintQuotient<F>, PreparedBatchCommitments) {
    let fri::PreparedConstraintQuotient {
        mut coeffs,
        base_evals,
        commitments,
    } = prepared;
    trim_trailing_zeros(&mut coeffs);
    if coeffs.is_empty() {
        coeffs.push(F::zero());
    }
    (
        ConstraintQuotient {
            coeffs,
            base_evals,
            lde_evals: Vec::new(),
        },
        commitments,
    )
}

fn build_constraint_quotient_and_commitment<F: PrimeField + FftField + Send + Sync>(
    mut coeffs: Vec<F>,
    domain_size: usize,
    blowup: usize,
    commitment_columns: Vec<usize>,
) -> Result<(ConstraintQuotient<F>, PreparedBatchCommitments), Error> {
    trim_trailing_zeros(&mut coeffs);
    if coeffs.is_empty() {
        coeffs.push(F::zero());
    }
    let base_evals = quotient_base_evals_from_coeffs(&coeffs, domain_size)?;

    if let Some(commitments) = fri::try_prepare_batch_commitment_group_from_coeffs_gpu_lde(
        &[coeffs.as_slice()],
        domain_size,
        blowup,
        commitment_columns.clone(),
    )? {
        return Ok((
            ConstraintQuotient {
                coeffs,
                base_evals,
                lde_evals: Vec::new(),
            },
            commitments,
        ));
    }

    let lde_evals = fri::evaluate_lde_from_coeffs(&coeffs, domain_size, blowup)?;
    let commitments =
        fri::prepare_batch_commitment_group(&[lde_evals.as_slice()], commitment_columns)?;
    Ok((
        ConstraintQuotient {
            coeffs,
            base_evals,
            lde_evals,
        },
        commitments,
    ))
}

fn quotient_base_evals_from_coeffs<F: PrimeField + FftField>(
    coeffs: &[F],
    domain_size: usize,
) -> Result<Vec<F>, Error> {
    if coeffs.len() > domain_size {
        return Err(Error::LowDegreeCheckFailed);
    }
    let domain = GeneralEvaluationDomain::<F>::new(domain_size)
        .ok_or(Error::InvalidDomainSize(domain_size))?;
    let mut padded = coeffs.to_vec();
    padded.resize(domain_size, F::zero());
    Ok(domain.fft(&padded))
}

fn ensure_quotient_lde<F: PrimeField + FftField>(
    quotient: &mut ConstraintQuotient<F>,
    domain_size: usize,
    blowup: usize,
) -> Result<(), Error> {
    if quotient.base_evals.is_empty() {
        quotient.base_evals = quotient_base_evals_from_coeffs(&quotient.coeffs, domain_size)?;
    }
    if quotient.lde_evals.is_empty() {
        quotient.lde_evals = fri::evaluate_lde_from_coeffs(&quotient.coeffs, domain_size, blowup)?;
    }
    Ok(())
}

fn quotient_from_coeffs<F: PrimeField + FftField>(
    mut coeffs: Vec<F>,
    domain_size: usize,
    blowup: usize,
) -> Result<ConstraintQuotient<F>, Error> {
    trim_trailing_zeros(&mut coeffs);
    if coeffs.is_empty() {
        coeffs.push(F::zero());
    }
    let base_evals = quotient_base_evals_from_coeffs(&coeffs, domain_size)?;
    let lde_evals = fri::evaluate_lde_from_coeffs(&coeffs, domain_size, blowup)?;
    Ok(ConstraintQuotient {
        coeffs,
        base_evals,
        lde_evals,
    })
}

#[allow(clippy::too_many_arguments)]
fn construct_ood_precommits<F: PrimeField + FftField + Send + Sync>(
    table_coeffs: &[Vec<F>],
    table_lde_columns: &[&[F]],
    table_claimed_values: &[F],
    zeta_t: F,
    table_row_count: usize,
    table_gpu_lde_store: Option<&fri::PreparedGpuLdeColumnStore<F>>,
    query_coeffs: &[Vec<F>],
    query_lde_columns: &[&[F]],
    query_claimed_values: &[F],
    zeta_f: F,
    query_row_count: usize,
    query_gpu_lde_store: Option<&fri::PreparedGpuLdeColumnStore<F>>,
    blowup: usize,
) -> Result<(OodPrecommit<F>, OodPrecommit<F>), Error> {
    let build_table = || {
        prove_diag_stage("prove.construct_ood_precommits.table", || {
            build_ood_precommit(
                table_coeffs,
                table_lde_columns,
                table_claimed_values,
                zeta_t,
                table_row_count,
                blowup,
                table_gpu_lde_store,
                &[0, 1, 2, 3, 4],
            )
        })
    };
    let build_query = || {
        prove_diag_stage("prove.construct_ood_precommits.query", || {
            build_ood_precommit(
                query_coeffs,
                query_lde_columns,
                query_claimed_values,
                zeta_f,
                query_row_count,
                blowup,
                query_gpu_lde_store,
                &[0, 1, 2],
            )
        })
    };

    if should_parallelize_lookup_prove() {
        std::thread::scope(|scope| {
            let table_handle = scope.spawn(build_table);
            let query_handle = scope.spawn(build_query);
            let table_precommit = join_prover_thread(table_handle)?;
            let query_precommit = join_prover_thread(query_handle)?;
            Ok((table_precommit, query_precommit))
        })
    } else {
        Ok((build_table()?, build_query()?))
    }
}

#[allow(clippy::too_many_arguments)]
fn build_ood_precommit<F: PrimeField + FftField + Send + Sync>(
    coeffs: &[Vec<F>],
    original_lde_columns: &[&[F]],
    claimed_values: &[F],
    zeta: F,
    row_count: usize,
    blowup: usize,
    gpu_lde_store: Option<&fri::PreparedGpuLdeColumnStore<F>>,
    gpu_column_indices: &[usize],
) -> Result<OodPrecommit<F>, Error> {
    if coeffs.len() != claimed_values.len() || coeffs.len() != original_lde_columns.len() {
        return Err(Error::InvalidOutOfDomainOpening);
    }
    let quotient_degree_bounds =
        vec![usc::specialized_usc_column_degree_bound(row_count); coeffs.len()];

    if let Some(store) = gpu_lde_store {
        let gpu_precommit = fri::prepare_ood_opening_batch_from_gpu_lde_store(
            store,
            gpu_column_indices,
            claimed_values,
            zeta,
            row_count,
            blowup,
            &quotient_degree_bounds,
        )?;
        return Ok(OodPrecommit::Gpu(gpu_precommit));
    }

    if let Some(gpu_precommit) = fri::try_prepare_ood_opening_batch_gpu_resident(
        original_lde_columns,
        claimed_values,
        zeta,
        row_count,
        blowup,
        &quotient_degree_bounds,
    )? {
        return Ok(OodPrecommit::Gpu(gpu_precommit));
    }

    let column_results = if should_parallelize_lookup_prove() && coeffs.len() > 1 {
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(coeffs.len());
            for (index, (poly, claimed)) in coeffs.iter().zip(claimed_values).enumerate() {
                handles.push(scope.spawn(move || {
                    let (column, lde) =
                        build_ood_precommit_column(poly, *claimed, zeta, row_count, blowup)?;
                    Ok((index, column, lde))
                }));
            }

            let mut results = Vec::with_capacity(handles.len());
            for handle in handles {
                results.push(join_prover_thread(handle)?);
            }
            results.sort_by_key(|(index, _, _)| *index);
            Ok(results)
        })?
    } else {
        let mut results = Vec::with_capacity(coeffs.len());
        for (index, (poly, claimed)) in coeffs.iter().zip(claimed_values).enumerate() {
            let (column, lde) =
                build_ood_precommit_column(poly, *claimed, zeta, row_count, blowup)?;
            results.push((index, column, lde));
        }
        results
    };

    let mut quotient_columns = Vec::with_capacity(column_results.len());
    let mut quotient_lde_columns = Vec::with_capacity(column_results.len());
    for (_, column, lde) in column_results {
        quotient_columns.push(column);
        quotient_lde_columns.push(lde);
    }

    let quotient_refs = quotient_lde_columns
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let commitments = fri::prepare_batch_commitment_group(
        &quotient_refs,
        (0..quotient_refs.len()).collect::<Vec<_>>(),
    )?;
    Ok(OodPrecommit::Cpu {
        quotient_columns,
        quotient_lde_columns,
        commitments,
        quotient_degree_bounds,
    })
}

fn build_ood_precommit_column<F: PrimeField + FftField>(
    coeffs: &[F],
    claimed_value: F,
    zeta: F,
    row_count: usize,
    blowup: usize,
) -> Result<(Vec<F>, Vec<F>), Error> {
    let q_coeffs = opening_quotient_coeffs(coeffs, claimed_value, zeta)?;
    let column = quotient_base_evals_from_coeffs(&q_coeffs, row_count)?;
    let lde = fri::evaluate_lde_from_coeffs(&q_coeffs, row_count, blowup)?;
    Ok((column, lde))
}

fn finish_table_ood_proof<F: PrimeField + FftField + Send + Sync>(
    params: &PublicParameters<F>,
    table_batch_prepared: &PreparedBatchFriProof<'_, F>,
    precommit: OodPrecommit<F>,
    coefficients: &[F],
) -> Result<OodOpeningProof<F>, Error> {
    let (quotient_batch, quotient_openings) = match precommit {
        OodPrecommit::Cpu {
            quotient_columns,
            quotient_lde_columns,
            commitments,
            quotient_degree_bounds,
        } => {
            let columns = quotient_columns
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>();
            let lde_columns = quotient_lde_columns
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>();
            let quotient_prepared = fri::prepare_batch_low_degree_with_commitments_and_lde(
                &columns,
                &lde_columns,
                coefficients,
                &quotient_degree_bounds,
                params.table.size(),
                params.blowup,
                commitments,
            )?;
            let quotient_batch = quotient_prepared.proof_with_openings(&[])?;
            let lde_indices = quotient_batch
                .combined
                .queries
                .iter()
                .map(|query| query.index)
                .collect::<Vec<_>>();
            let quotient_openings = quotient_prepared.lde_row_openings(&lde_indices)?;
            (quotient_batch, quotient_openings)
        }
        OodPrecommit::Gpu(precommit) => {
            prove_diag_stage("prove.finish_table_ood.gpu_precommit_finish", || {
                precommit.finish(coefficients)
            })?
        }
    };
    let lde_indices = quotient_batch
        .combined
        .queries
        .iter()
        .map(|query| query.index)
        .collect::<Vec<_>>();
    let main_openings = prove_diag_stage("prove.finish_table_ood.main_batch_openings", || {
        table_batch_prepared.lde_row_openings(&lde_indices)
    })?;
    let table_lde_openings = prove_diag_stage("prove.finish_table_ood.table_lde_openings", || {
        params
            .table_preprocessing
            .open_lde_many(&lde_indices, params.table.size(), params.blowup)
    })?;
    let row_openings = lde_indices
        .iter()
        .copied()
        .zip(quotient_openings)
        .zip(main_openings)
        .zip(table_lde_openings)
        .map(
            |(((lde_index, quotient_opening), main_opening), (table_value, table_path))| {
                let mut original_row = Vec::with_capacity(5);
                original_row.push(table_value);
                original_row.extend_from_slice(&main_opening.row);
                let mut original_group_paths = Vec::with_capacity(4);
                original_group_paths.push(table_path);
                original_group_paths.extend(main_opening.group_paths);
                OodRowOpening {
                    lde_index,
                    original_row,
                    original_group_paths,
                    quotient_opening,
                }
            },
        )
        .collect();
    Ok(OodOpeningProof {
        quotient_batch,
        row_openings,
    })
}

fn finish_query_ood_proof<F: PrimeField + FftField + Send + Sync>(
    query_batch_prepared: &PreparedBatchFriProof<'_, F>,
    precommit: OodPrecommit<F>,
    coefficients: &[F],
    row_count: usize,
    blowup: usize,
) -> Result<OodOpeningProof<F>, Error> {
    let (quotient_batch, quotient_openings) = match precommit {
        OodPrecommit::Cpu {
            quotient_columns,
            quotient_lde_columns,
            commitments,
            quotient_degree_bounds,
        } => {
            let columns = quotient_columns
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>();
            let lde_columns = quotient_lde_columns
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>();
            let quotient_prepared = fri::prepare_batch_low_degree_with_commitments_and_lde(
                &columns,
                &lde_columns,
                coefficients,
                &quotient_degree_bounds,
                row_count,
                blowup,
                commitments,
            )?;
            let quotient_batch = quotient_prepared.proof_with_openings(&[])?;
            let lde_indices = quotient_batch
                .combined
                .queries
                .iter()
                .map(|query| query.index)
                .collect::<Vec<_>>();
            let quotient_openings = quotient_prepared.lde_row_openings(&lde_indices)?;
            (quotient_batch, quotient_openings)
        }
        OodPrecommit::Gpu(precommit) => {
            prove_diag_stage("prove.finish_query_ood.gpu_precommit_finish", || {
                precommit.finish(coefficients)
            })?
        }
    };
    let lde_indices = quotient_batch
        .combined
        .queries
        .iter()
        .map(|query| query.index)
        .collect::<Vec<_>>();
    let main_openings = prove_diag_stage("prove.finish_query_ood.main_batch_openings", || {
        query_batch_prepared.lde_row_openings(&lde_indices)
    })?;
    let row_openings = lde_indices
        .iter()
        .copied()
        .zip(quotient_openings)
        .zip(main_openings)
        .map(
            |((lde_index, quotient_opening), main_opening)| OodRowOpening {
                lde_index,
                original_row: main_opening.row,
                original_group_paths: main_opening.group_paths,
                quotient_opening,
            },
        )
        .collect();
    Ok(OodOpeningProof {
        quotient_batch,
        row_openings,
    })
}

fn construct_specialized_usc_precommits<F: PrimeField + FftField + Send + Sync>(
    table_accumulator_coeffs: &[F],
    table_accumulator_lde: &[F],
    table_row_count: usize,
    table_gpu_lde_store: Option<&fri::PreparedGpuLdeColumnStore<F>>,
    query_reciprocal_coeffs: &[F],
    query_reciprocal_lde: &[F],
    query_row_count: usize,
    query_gpu_lde_store: Option<&fri::PreparedGpuLdeColumnStore<F>>,
    common_sum_s: F,
    blowup: usize,
) -> Result<(SpecializedUscPrecommit<F>, SpecializedUscPrecommit<F>), Error> {
    let build_table = || {
        prove_diag_stage("prove.construct_specialized_usc_precommits.table", || {
            build_specialized_usc_precommit(
                table_accumulator_coeffs,
                table_accumulator_lde,
                common_sum_s,
                table_row_count,
                blowup,
                table_gpu_lde_store,
                3,
            )
        })
    };
    let build_query = || {
        prove_diag_stage("prove.construct_specialized_usc_precommits.query", || {
            build_specialized_usc_precommit(
                query_reciprocal_coeffs,
                query_reciprocal_lde,
                common_sum_s,
                query_row_count,
                blowup,
                query_gpu_lde_store,
                1,
            )
        })
    };

    if should_parallelize_lookup_prove() {
        std::thread::scope(|scope| {
            let table_handle = scope.spawn(build_table);
            let query_handle = scope.spawn(build_query);
            let table_precommit = join_prover_thread(table_handle)?;
            let query_precommit = join_prover_thread(query_handle)?;
            Ok((table_precommit, query_precommit))
        })
    } else {
        Ok((build_table()?, build_query()?))
    }
}

#[allow(clippy::too_many_arguments)]
fn build_specialized_usc_precommit<F: PrimeField + FftField + Send + Sync>(
    coeffs: &[F],
    existing_lde_column: &[F],
    common_sum_s: F,
    row_count: usize,
    blowup: usize,
    gpu_lde_store: Option<&fri::PreparedGpuLdeColumnStore<F>>,
    gpu_column_index: usize,
) -> Result<SpecializedUscPrecommit<F>, Error> {
    validate_specialized_usc_sum_claim(coeffs, common_sum_s, row_count)?;
    let p_degree_bound = usc::specialized_usc_column_degree_bound(row_count);

    if let Some(store) = gpu_lde_store {
        let gpu_precommit = fri::prepare_specialized_usc_batch_from_gpu_lde_store(
            store,
            gpu_column_index,
            common_sum_s,
            row_count,
            blowup,
            p_degree_bound,
        )?;
        return Ok(SpecializedUscPrecommit::Gpu(gpu_precommit));
    }

    if let Some(gpu_precommit) = fri::try_prepare_specialized_usc_batch_gpu_resident(
        existing_lde_column,
        common_sum_s,
        row_count,
        blowup,
        p_degree_bound,
    )? {
        return Ok(SpecializedUscPrecommit::Gpu(gpu_precommit));
    }

    let p_coeffs = specialized_usc_quotient_coeffs(coeffs, common_sum_s, row_count)?;
    let p_values = quotient_base_evals_from_coeffs(&p_coeffs, row_count)?;
    let p_lde = fri::evaluate_lde_from_coeffs(&p_coeffs, row_count, blowup)?;
    let commitments = fri::prepare_batch_commitment_group(&[p_lde.as_slice()], vec![0])?;
    Ok(SpecializedUscPrecommit::Cpu {
        p_values,
        p_lde,
        commitments,
    })
}

fn prove_specialized_usc<F, G>(
    precommit: SpecializedUscPrecommit<F>,
    coefficients: &[F],
    row_count: usize,
    blowup: usize,
    existing_openings: G,
) -> Result<SpecializedUscProof<F>, Error>
where
    F: PrimeField + FftField + Send + Sync,
    G: FnOnce(&[usize]) -> Result<Vec<(Vec<F>, MerklePath)>, Error>,
{
    let (batch, p_openings) = match precommit {
        SpecializedUscPrecommit::Cpu {
            p_values,
            p_lde,
            commitments,
        } => {
            let columns = [p_values.as_slice()];
            let lde_columns = [p_lde.as_slice()];
            let p_degree_bound = usc::specialized_usc_column_degree_bound(row_count);
            let prepared = fri::prepare_batch_low_degree_with_commitments_and_lde(
                &columns,
                &lde_columns,
                coefficients,
                &[p_degree_bound],
                row_count,
                blowup,
                commitments,
            )?;
            let batch = prepared.proof_with_openings(&[])?;
            let lde_indices = batch
                .combined
                .queries
                .iter()
                .map(|query| query.index)
                .collect::<Vec<_>>();
            let p_openings = prepared.lde_row_openings(&lde_indices)?;
            (batch, p_openings)
        }
        SpecializedUscPrecommit::Gpu(precommit) => precommit.finish(coefficients)?,
    };
    let lde_indices = batch
        .combined
        .queries
        .iter()
        .map(|query| query.index)
        .collect::<Vec<_>>();
    let existing = existing_openings(&lde_indices)?;
    let openings = lde_indices
        .iter()
        .copied()
        .zip(existing)
        .zip(p_openings)
        .map(
            |((lde_index, (existing_group_row, existing_group_path)), p_opening)| {
                SpecializedUscOpening {
                    lde_index,
                    existing_group_row,
                    existing_group_path,
                    p_opening,
                }
            },
        )
        .collect();
    Ok(SpecializedUscProof { batch, openings })
}

fn table_existing_group_openings_for_column<F: PrimeField + FftField + Send + Sync>(
    table_batch_prepared: &PreparedBatchFriProof<'_, F>,
    lde_indices: &[usize],
    group_index: usize,
) -> Result<Vec<(Vec<F>, MerklePath)>, Error> {
    table_batch_prepared
        .lde_row_openings(lde_indices)?
        .into_iter()
        .map(|opening| {
            if group_index >= opening.group_paths.len() || opening.row.len() < 3 {
                return Err(Error::InvalidSpecializedUscIdentity);
            }
            Ok((
                vec![opening.row[1], opening.row[2]],
                opening.group_paths[group_index].clone(),
            ))
        })
        .collect()
}

fn query_existing_group_openings_for_column<F: PrimeField + FftField + Send + Sync>(
    query_batch_prepared: &PreparedBatchFriProof<'_, F>,
    lde_indices: &[usize],
    group_index: usize,
) -> Result<Vec<(Vec<F>, MerklePath)>, Error> {
    query_batch_prepared
        .lde_row_openings(lde_indices)?
        .into_iter()
        .map(|opening| {
            if group_index >= opening.group_paths.len() || opening.row.len() < 2 {
                return Err(Error::InvalidSpecializedUscIdentity);
            }
            Ok((
                vec![opening.row[1]],
                opening.group_paths[group_index].clone(),
            ))
        })
        .collect()
}

fn verify_proof_shape<F: PrimeField + FftField>(
    params: &PublicParameters<F>,
    statement: &LookupStatement,
    proof: &LookupProof<F>,
) -> Result<(), Error> {
    if params.query_size != statement.query_size
        || proof.table_batch.row_count != params.table.size()
        || proof.query_batch.row_count != statement.query_size
        || proof.table_batch.column_count != 4
        || proof.query_batch.column_count != 3
        || proof.table_batch.commitment_groups.len() != 3
        || proof.query_batch.commitment_groups.len() != 3
        || proof.table_batch.commitment_groups[0].columns != [0]
        || proof.table_batch.commitment_groups[1].columns != [1, 2]
        || proof.table_batch.commitment_groups[2].columns != [3]
        || proof.query_batch.commitment_groups[0].columns != [0]
        || proof.query_batch.commitment_groups[1].columns != [1]
        || proof.query_batch.commitment_groups[2].columns != [2]
    {
        return Err(Error::StatementMismatch);
    }
    if proof.table_batch.commitment_groups[0].root != proof.multiplicity_root
        || proof.query_batch.commitment_groups[0].root != proof.query_root
        || proof.table_batch.commitment_groups[2].root != proof.c_qt
        || proof.query_batch.commitment_groups[2].root != proof.c_qf
        || proof.specialized_usc_table_proof.batch.commitment_groups[0].root != proof.c_p_a
        || proof.specialized_usc_query_proof.batch.commitment_groups[0].root != proof.c_p_v
    {
        return Err(Error::MerkleRootMismatch);
    }
    Ok(())
}

fn table_ood_original_groups<F: FftField>(
    params: &PublicParameters<F>,
    proof: &LookupProof<F>,
) -> Vec<BatchCommitmentGroup> {
    vec![
        BatchCommitmentGroup {
            columns: vec![0],
            root: params.table_lde_root,
        },
        BatchCommitmentGroup {
            columns: vec![1],
            root: proof.table_batch.commitment_groups[0].root,
        },
        BatchCommitmentGroup {
            columns: vec![2, 3],
            root: proof.table_batch.commitment_groups[1].root,
        },
        BatchCommitmentGroup {
            columns: vec![4],
            root: proof.table_batch.commitment_groups[2].root,
        },
    ]
}

fn query_ood_original_groups<F: FftField>(proof: &LookupProof<F>) -> Vec<BatchCommitmentGroup> {
    vec![
        proof.query_batch.commitment_groups[0].clone(),
        proof.query_batch.commitment_groups[1].clone(),
        proof.query_batch.commitment_groups[2].clone(),
    ]
}

#[allow(clippy::too_many_arguments)]
fn verify_table_quotient_identity<F: PrimeField + FftField>(
    values: &[F],
    zeta: F,
    table_size: usize,
    beta: F,
    eta_t1: F,
    eta_t2: F,
) -> Result<(), Error> {
    let t = values[0];
    let m = values[1];
    let u = values[2];
    let a = values[3];
    let q_t = values[4];
    let r_t = eta_t1 * (u * (beta + t) - F::one()) + eta_t2 * (a - m * u);
    let z_h = zeta.pow(&[table_size as u64]) - F::one();
    if z_h.is_zero() {
        return Err(Error::InvalidOutOfDomainPoint);
    }
    if r_t != z_h * q_t {
        return Err(Error::InvalidTableQuotientIdentity);
    }
    Ok(())
}

fn verify_query_quotient_identity<F: PrimeField + FftField>(
    values: &[F],
    zeta: F,
    query_size: usize,
    beta: F,
) -> Result<(), Error> {
    let f = values[0];
    let v = values[1];
    let q_f = values[2];
    let r_f = v * (beta + f) - F::one();
    let z_h = zeta.pow(&[query_size as u64]) - F::one();
    if z_h.is_zero() {
        return Err(Error::InvalidOutOfDomainPoint);
    }
    if r_f != z_h * q_f {
        return Err(Error::InvalidQueryQuotientIdentity);
    }
    Ok(())
}

fn compute_table_identifier<F: PrimeField + FftField>(
    table: &Table<F>,
    table_domain: &DomainSidePreprocessing<F>,
) -> Digest32 {
    let mut hasher = Keccak256::new();
    hasher.update(b"TLookup.TableIdentifier.v1");
    hash_usize(&mut hasher, table.size());
    hash_field_slice(&mut hasher, b"T", table.values());
    hash_domain_info(&mut hasher, b"H_T", &table_domain.base);
    hash_domain_info(&mut hasher, b"L_T", &table_domain.lde);
    hasher.finalize().into()
}

fn hash_domain_info<F: FftField + ToBytes>(
    hasher: &mut Keccak256,
    label: &[u8],
    domain: &crate::preprocessing::DomainInfo<F>,
) {
    hash_bytes(hasher, label);
    hash_usize(hasher, domain.size);
    hash_field(hasher, &domain.root_of_unity);
    hash_field_slice(hasher, b"elements", &domain.elements);
}

fn hash_field_slice<F: ToBytes>(hasher: &mut Keccak256, label: &[u8], values: &[F]) {
    hash_bytes(hasher, label);
    hash_usize(hasher, values.len());
    for value in values {
        hash_field(hasher, value);
    }
}

fn hash_field<F: ToBytes>(hasher: &mut Keccak256, value: &F) {
    let mut bytes = Vec::new();
    value
        .write(&mut bytes)
        .expect("writing field element to Vec cannot fail");
    hash_bytes(hasher, &bytes);
}

fn hash_usize(hasher: &mut Keccak256, value: usize) {
    hash_bytes(hasher, &(value as u64).to_le_bytes());
}

fn hash_bytes(hasher: &mut Keccak256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn table_domains_match<F: FftField>(
    left: &DomainSidePreprocessing<F>,
    right: &DomainSidePreprocessing<F>,
) -> bool {
    left.base.size == right.base.size
        && left.base.root_of_unity == right.base.root_of_unity
        && left.base.elements == right.base.elements
        && left.lde.size == right.lde.size
        && left.lde.root_of_unity == right.lde.root_of_unity
        && left.lde.elements == right.lde.elements
}

fn validate_public_table_preprocessing<F: PrimeField + FftField + Send + Sync>(
    params: &PublicParameters<F>,
) -> Result<(), Error> {
    let mut cache = params
        .table_validation_cache
        .lock()
        .map_err(|_| Error::LowDegreeCheckFailed)?;
    if cache
        .as_ref()
        .is_some_and(|validated| validated.matches(params))
    {
        return Ok(());
    }

    let expected_domain = preprocess_table_domain(params.table.size(), params.blowup)?;
    if !table_domains_match(&params.domains.table, &expected_domain) {
        return Err(Error::TableIdentifierMismatch);
    }
    let expected_id = compute_table_identifier(&params.table, &expected_domain);
    if expected_id != params.table_id {
        return Err(Error::TableIdentifierMismatch);
    }

    let table_polynomial = fri::interpolate(params.table.values())?;
    let table_lde = fri::evaluate_lde_from_coeffs(
        &table_polynomial.coeffs,
        params.table.size(),
        params.blowup,
    )?;
    let recomputed_root = tree_for_column(&table_lde).root();
    if recomputed_root != params.table_lde_root {
        return Err(Error::MerkleRootMismatch);
    }
    fri::verify_low_degree(&params.table_low_degree, params.table_lde_root)?;

    *cache = Some(ValidatedTablePreprocessing {
        table: Arc::clone(&params.table),
        table_domain: Arc::clone(&params.domains.table),
        table_id: params.table_id,
        table_lde_root: params.table_lde_root,
        table_low_degree: Arc::clone(&params.table_low_degree),
        blowup: params.blowup,
    });
    Ok(())
}

fn lookup_transcript<F: PrimeField + FftField>(
    params: &PublicParameters<F>,
    statement: &LookupStatement,
) -> Transcript {
    let mut transcript = Transcript::new(PROTOCOL_NAME);
    transcript.absorb_bytes(
        b"tlookup/proof-version",
        &LOOKUP_PROOF_VERSION.to_le_bytes(),
    );
    transcript.absorb(b"tlookup/table-size", &F::from(params.table.size() as u64));
    transcript.absorb(b"tlookup/query-size", &F::from(statement.query_size as u64));
    transcript.absorb(b"tlookup/blowup", &F::from(params.blowup as u64));
    transcript.absorb_bytes(b"tlookup/table-id", &params.table_id);
    transcript.absorb_bytes(b"tlookup/table-commitment", &params.table_lde_root);
    transcript
}

fn absorb_batch_fri_transcript<F: FftField>(
    transcript: &mut Transcript,
    label: &'static [u8],
    batch: &BatchFriProof<F>,
) {
    transcript.absorb_bytes(label, b"v1");
    transcript.absorb_bytes(b"batch-row-root", &batch.row_root);
    for group in &batch.commitment_groups {
        transcript.absorb_bytes(b"batch-group-root", &group.root);
    }
    for root in &batch.combined.roots {
        transcript.absorb_bytes(b"batch-fri-root", root);
    }
}

fn absorb_field_vec<F: PrimeField>(
    transcript: &mut Transcript,
    label: &'static [u8],
    values: &[F],
) {
    transcript.absorb_bytes(label, b"v1");
    for value in values {
        transcript.absorb(label, value);
    }
}

fn challenge_vector<F: PrimeField>(
    transcript: &mut Transcript,
    label: &'static [u8],
    len: usize,
) -> Vec<F> {
    transcript.absorb_bytes(label, b"v1");
    (0..len)
        .map(|_| transcript.challenge_nonzero::<F>(b"tlookup/batching-coefficient"))
        .collect()
}

fn challenge_outside_lde_domain<F: PrimeField + FftField>(
    transcript: &mut Transcript,
    label: &'static [u8],
    lde_size: usize,
) -> F {
    transcript.challenge_with_rejection(label, |value: &F| {
        !value.is_zero() && value.pow(&[lde_size as u64]) != F::one()
    })
}

fn polynomial_coefficients<F: FftField>(values: &[F]) -> Result<Vec<F>, Error> {
    let poly = fri::interpolate(values)?;
    Ok(poly.coeffs)
}

fn validate_specialized_usc_sum_claim<F: PrimeField + FftField>(
    coeffs: &[F],
    claimed_sum: F,
    row_count: usize,
) -> Result<(), Error> {
    let size = F::from(row_count as u64);
    let size_inverse = size.inverse().ok_or(Error::InvalidDomainSize(row_count))?;
    let expected_constant = claimed_sum * size_inverse;
    let actual_constant = coeffs.first().copied().unwrap_or_else(F::zero);
    if actual_constant != expected_constant {
        return Err(Error::InvalidSumClaim);
    }
    Ok(())
}

fn specialized_usc_quotient_coeffs<F: PrimeField + FftField>(
    coeffs: &[F],
    claimed_sum: F,
    row_count: usize,
) -> Result<Vec<F>, Error> {
    validate_specialized_usc_sum_claim(coeffs, claimed_sum, row_count)?;
    let quotient_coeffs = if row_count == 1 {
        vec![F::zero()]
    } else if coeffs.len() > 1 {
        coeffs[1..].to_vec()
    } else {
        vec![F::zero()]
    };
    if quotient_coeffs.len() > row_count {
        return Err(Error::LowDegreeCheckFailed);
    }
    Ok(quotient_coeffs)
}

fn opening_quotient_coeffs<F: FftField>(
    coeffs: &[F],
    claimed_value: F,
    zeta: F,
) -> Result<Vec<F>, Error> {
    let mut numerator = coeffs.to_vec();
    if numerator.is_empty() {
        numerator.push(-claimed_value);
    } else {
        numerator[0] -= claimed_value;
    }
    trim_trailing_zeros(&mut numerator);
    if numerator.is_empty() {
        return Ok(vec![F::zero()]);
    }
    let mut quotient_rev = Vec::with_capacity(numerator.len().saturating_sub(1));
    let mut carry = *numerator.last().unwrap();
    quotient_rev.push(carry);
    for coeff in numerator
        .iter()
        .rev()
        .skip(1)
        .take(numerator.len().saturating_sub(2))
    {
        carry = *coeff + zeta * carry;
        quotient_rev.push(carry);
    }
    let remainder = numerator[0] + zeta * carry;
    if !remainder.is_zero() {
        return Err(Error::InvalidOutOfDomainOpening);
    }
    quotient_rev.reverse();
    if quotient_rev.is_empty() {
        quotient_rev.push(F::zero());
    }
    Ok(quotient_rev)
}

fn product_high_coeffs<F: PrimeField + FftField + Send + Sync>(
    left: &[F],
    right: &[F],
    domain_size: usize,
) -> Result<Vec<F>, Error> {
    if let Some(high_coeffs) = fri::try_product_high_coefficients_gpu(left, right, domain_size)? {
        return Ok(high_coeffs);
    }
    product_high_coeffs_cpu(left, right, domain_size)
}

fn product_high_coeffs_cpu<F: FftField>(
    left: &[F],
    right: &[F],
    domain_size: usize,
) -> Result<Vec<F>, Error> {
    if domain_size == 0 {
        return Err(Error::InvalidDomainSize(domain_size));
    }
    if left.len() > domain_size || right.len() > domain_size {
        return Err(Error::LowDegreeCheckFailed);
    }

    let quotient_len = domain_size.saturating_sub(1).max(1);
    let mut high_coeffs = vec![F::zero(); quotient_len];
    let product = poly_mul(left, right)?;
    for (index, target) in high_coeffs.iter_mut().enumerate() {
        if let Some(value) = product.get(domain_size + index) {
            *target = *value;
        }
    }
    Ok(high_coeffs)
}

fn poly_mul<F: FftField>(left: &[F], right: &[F]) -> Result<Vec<F>, Error> {
    if left.is_empty() || right.is_empty() {
        return Ok(Vec::new());
    }
    let result_len = left.len() + right.len() - 1;
    let fft_size = result_len.next_power_of_two();
    let domain =
        GeneralEvaluationDomain::<F>::new(fft_size).ok_or(Error::InvalidDomainSize(fft_size))?;
    let mut left_pad = left.to_vec();
    let mut right_pad = right.to_vec();
    left_pad.resize(fft_size, F::zero());
    right_pad.resize(fft_size, F::zero());
    let mut left_evals = domain.fft(&left_pad);
    let right_evals = domain.fft(&right_pad);
    for (left_value, right_value) in left_evals.iter_mut().zip(right_evals) {
        *left_value *= right_value;
    }
    let mut coeffs = domain.ifft(&left_evals);
    coeffs.truncate(result_len);
    trim_trailing_zeros(&mut coeffs);
    Ok(coeffs)
}

fn evaluate_coeffs<F: FftField>(coeffs: &[F], point: F) -> F {
    coeffs
        .iter()
        .rev()
        .fold(F::zero(), |acc, coeff| acc * point + *coeff)
}

fn trim_trailing_zeros<F: FftField>(coeffs: &mut Vec<F>) {
    while coeffs.last().is_some_and(|value| value.is_zero()) {
        coeffs.pop();
    }
}

fn batch_reciprocals<F: FftField + PrimeField>(
    table_values: &[F],
    query_values: &[F],
    beta: F,
) -> Result<(Vec<F>, Vec<F>), Error> {
    let mut denominators = Vec::with_capacity(table_values.len() + query_values.len());
    denominators.extend(table_values.iter().map(|value| *value + beta));
    denominators.extend(query_values.iter().map(|value| *value + beta));
    if denominators.iter().any(|value| value.is_zero()) {
        return Err(Error::InvalidBetaDenominator);
    }
    batch_inversion(&mut denominators);
    let query_reciprocal = denominators.split_off(table_values.len());
    Ok((denominators, query_reciprocal))
}

fn tree_for_column<F: FftField>(column: &[F]) -> MerkleTree {
    MerkleTree::from_column(column)
}

fn join_prover_thread<T>(
    handle: std::thread::ScopedJoinHandle<'_, Result<T, Error>>,
) -> Result<T, Error> {
    handle.join().map_err(|_| Error::LowDegreeCheckFailed)?
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn prove_diag_stage<T>(name: &str, op: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
    if !prove_diagnostics_enabled() {
        return op();
    }

    let start = Instant::now();
    eprintln!("prove stage start: {name}");
    match op() {
        Ok(value) => {
            eprintln!(
                "prove stage ok: {name}: {:.3} ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
            Ok(value)
        }
        Err(error) => {
            eprintln!(
                "prove stage failed: {name}: {:.3} ms: {error:?}",
                start.elapsed().as_secs_f64() * 1000.0
            );
            Err(error)
        }
    }
}

fn prove_diagnostics_enabled() -> bool {
    env_flag(PROVE_DIAGNOSTICS_ENV) || env_flag(SCALE_DIAGNOSTICS_ENV)
}

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .ok()
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn should_parallelize_lookup_prove() -> bool {
    match std::env::var(PARALLEL_PROVE_ENV)
        .ok()
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("1" | "true" | "yes" | "on" | "parallel") => true,
        Some("0" | "false" | "no" | "off" | "serial") => false,
        _ => true,
    }
}

pub fn warm_up_gpu_batch_reciprocals<F: FftField + PrimeField + Send + Sync>(
    element_count: usize,
) -> Result<(), Error> {
    if element_count == 0 {
        return Ok(());
    }

    let table_row_count = element_count
        .max(2)
        .checked_next_power_of_two()
        .ok_or(Error::InvalidDomainSize(element_count))?;
    let warmup_blowup = 2usize;
    let plan = crate::lde::DenseLdePlan::<F>::new(table_row_count, warmup_blowup)?;
    if crate::lde::select_dense_lde_backend::<F>(&plan) != crate::lde::DenseLdeBackendKind::Gpu {
        return Ok(());
    }

    let query_row_count = 1usize;
    let table_values = (0..table_row_count)
        .map(|index| F::from((index as u64) + 2))
        .collect::<Vec<_>>();
    let multiplicity = vec![F::from(0u64); table_row_count];
    let query_values = vec![F::from(3u64); query_row_count];

    let _ = fri::try_prepare_lookup_auxiliary_gpu_resident(
        &table_values,
        &multiplicity,
        &query_values,
        F::from(1u64),
        table_row_count,
        query_row_count,
        warmup_blowup,
    )?;
    Ok(())
}
