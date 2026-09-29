use ark_ff::{batch_inversion, FftField, PrimeField};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
#[cfg(feature = "icicle-rust")]
use std::sync::Arc;
use std::{env, fmt, thread};

use crate::{
    error::Error,
    lde,
    merkle::{row_to_leaf, Digest32, MerklePath, MerkleTree},
    transcript::Transcript,
    utils::is_power_of_two,
};

const FRI_PROTOCOL_LABEL: &[u8] = b"FRI-SAMPLED-0.1";
const DEFAULT_QUERY_COUNT: usize = 64;
const SPARSE_LDE_LOG_FACTOR: usize = 2;
const SPARSE_LDE_TARGET_CHUNK_DENOMINATORS: usize = 1 << 15;
const CPU_PARALLEL_MIN_ITEMS: usize = 1 << 14;

pub const DEFAULT_GPU_FOLD_MIN_SIZE: usize = 1 << 16;
pub const DEFAULT_GPU_BATCH_COMBINE_MIN_SIZE: usize = 1 << 20;
pub const DEFAULT_GPU_BATCH_MERKLE_MIN_LEAVES: usize = 1 << 16;
pub const DEFAULT_GPU_RESIDENT_FRI_MIN_SIZE: usize = 1 << 16;

const FOLD_BACKEND_ENV: &str = "FRI_LOOKUP_FOLD_BACKEND";
const GPU_FOLD_MIN_SIZE_ENV: &str = "FRI_LOOKUP_GPU_FOLD_MIN_SIZE";
const BATCH_COMBINE_BACKEND_ENV: &str = "FRI_LOOKUP_BATCH_COMBINE_BACKEND";
const GPU_BATCH_COMBINE_MIN_SIZE_ENV: &str = "FRI_LOOKUP_GPU_BATCH_COMBINE_MIN_SIZE";
const BATCH_MERKLE_BACKEND_ENV: &str = "FRI_LOOKUP_BATCH_MERKLE_BACKEND";
const GPU_BATCH_MERKLE_MIN_LEAVES_ENV: &str = "FRI_LOOKUP_GPU_BATCH_MERKLE_MIN_LEAVES";
const GPU_RESIDENT_FRI_BACKEND_ENV: &str = "FRI_LOOKUP_GPU_RESIDENT_FRI_BACKEND";
const GPU_RESIDENT_FRI_MIN_SIZE_ENV: &str = "FRI_LOOKUP_GPU_RESIDENT_FRI_MIN_SIZE";

#[derive(Clone, Debug)]
pub struct FriProof<F: FftField> {
    pub root: Digest32,
    pub roots: Vec<Digest32>,
    pub final_evals: Vec<F>,
    pub queries: Vec<FriQuery<F>>,
    pub degree_bound: usize,
    pub blowup: usize,
    pub domain_size: usize,
}

#[derive(Clone, Debug)]
pub struct FriQuery<F: FftField> {
    pub index: usize,
    pub openings: Vec<FriLayerOpening<F>>,
}

#[derive(Clone, Debug)]
pub struct FriLayerOpening<F: FftField> {
    pub index: usize,
    pub value: F,
    pub sibling_value: F,
    pub path: MerklePath,
    pub sibling_path: MerklePath,
}

#[derive(Clone, Debug)]
pub struct BatchFriProof<F: FftField> {
    pub row_root: Digest32,
    pub row_count: usize,
    pub column_count: usize,
    pub column_degree_bounds: Vec<usize>,
    pub commitment_groups: Vec<BatchCommitmentGroup>,
    pub row_openings: Vec<BatchRowOpening<F>>,
    pub batching_coefficients: Vec<F>,
    pub combined: FriProof<F>,
}

#[derive(Clone, Debug)]
pub struct BatchCommitmentGroup {
    pub columns: Vec<usize>,
    pub root: Digest32,
}

#[derive(Clone, Debug)]
pub struct PreparedBatchCommitments {
    group_trees: Vec<GroupTreeBackend>,
    commitment_groups: Vec<BatchCommitmentGroup>,
    #[cfg(feature = "icicle-rust")]
    gpu_lde_columns: Option<gpu_batch_merkle_bn254::GpuCommittedLdeColumns>,
}

#[derive(Clone, Debug)]
pub struct PreparedConstraintQuotient<F: FftField> {
    pub coeffs: Vec<F>,
    pub base_evals: Vec<F>,
    pub commitments: PreparedBatchCommitments,
}

impl PreparedBatchCommitments {
    pub fn commitment_groups(&self) -> &[BatchCommitmentGroup] {
        &self.commitment_groups
    }
}

#[cfg(feature = "icicle-rust")]
type MaybeGpuCommittedLdeColumns = gpu_batch_merkle_bn254::GpuCommittedLdeColumns;
#[cfg(not(feature = "icicle-rust"))]
type MaybeGpuCommittedLdeColumns = ();

#[derive(Clone, Debug)]
pub struct BatchRowOpening<F: FftField> {
    pub index: usize,
    pub row: Vec<F>,
    pub row_path: MerklePath,
    pub group_paths: Vec<MerklePath>,
    pub combined_lde_value: F,
    pub combined_lde_path: MerklePath,
}

#[derive(Clone, Debug)]
pub struct BatchLdeRowOpening<F: FftField> {
    pub lde_index: usize,
    pub row: Vec<F>,
    pub group_paths: Vec<MerklePath>,
    pub combined_lde_value: F,
    pub combined_lde_path: MerklePath,
}

#[derive(Clone, Debug)]
pub struct OodOpeningProof<F: FftField> {
    pub quotient_batch: BatchFriProof<F>,
    pub row_openings: Vec<OodRowOpening<F>>,
}

#[derive(Clone, Debug)]
pub struct OodRowOpening<F: FftField> {
    pub lde_index: usize,
    pub original_row: Vec<F>,
    pub original_group_paths: Vec<MerklePath>,
    pub quotient_opening: BatchLdeRowOpening<F>,
}

pub struct PreparedGpuLdeColumnStore<F: FftField> {
    #[cfg(feature = "icicle-rust")]
    inner: gpu_resident_fri_bn254::PreparedLdeColumnStore<F>,
    #[cfg(not(feature = "icicle-rust"))]
    _field: std::marker::PhantomData<F>,
}

unsafe impl<F: FftField> Send for PreparedGpuLdeColumnStore<F> {}
unsafe impl<F: FftField> Sync for PreparedGpuLdeColumnStore<F> {}

impl<F: PrimeField + FftField + Send + Sync> PreparedGpuLdeColumnStore<F> {
    fn lde_rows(
        &self,
        column_indices: &[usize],
        lde_indices: &[usize],
    ) -> Result<Vec<Vec<F>>, Error> {
        #[cfg(feature = "icicle-rust")]
        {
            return crate::gpu_lock::with_gpu_lock(|| {
                gpu_resident_fri_bn254::lde_rows_from_store(
                    &self.inner,
                    column_indices,
                    lde_indices,
                )
            });
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            let _ = (column_indices, lde_indices);
            Err(Error::GpuBackendUnavailable)
        }
    }

    fn combined_openings(
        &self,
        column_indices: &[usize],
        batching_coefficients: &[F],
        column_degree_bounds: &[usize],
        degree_bound: usize,
        blowup: usize,
        lde_indices: &[usize],
        expected_root: Digest32,
    ) -> Result<Vec<(F, MerklePath)>, Error> {
        #[cfg(feature = "icicle-rust")]
        {
            return crate::gpu_lock::with_gpu_lock(|| {
                gpu_resident_fri_bn254::combined_openings_from_store(
                    &self.inner,
                    column_indices,
                    batching_coefficients,
                    column_degree_bounds,
                    degree_bound,
                    blowup,
                    lde_indices,
                    expected_root,
                )
            });
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            let _ = (
                column_indices,
                batching_coefficients,
                column_degree_bounds,
                degree_bound,
                blowup,
                lde_indices,
                expected_root,
            );
            Err(Error::GpuBackendUnavailable)
        }
    }

    fn group_paths(
        &self,
        commitment_groups: &[BatchCommitmentGroup],
        lde_indices: &[usize],
    ) -> Result<Vec<Vec<MerklePath>>, Error> {
        #[cfg(feature = "icicle-rust")]
        {
            return crate::gpu_lock::with_gpu_lock(|| {
                gpu_resident_fri_bn254::group_paths_from_store(
                    &self.inner,
                    commitment_groups,
                    lde_indices,
                )
            });
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            let _ = (commitment_groups, lde_indices);
            Err(Error::GpuBackendUnavailable)
        }
    }
}

pub struct PreparedGpuOodOpeningBatch<F: FftField> {
    #[cfg(feature = "icicle-rust")]
    inner: gpu_resident_fri_bn254::PreparedOodQuotientBatch<F>,
    #[cfg(not(feature = "icicle-rust"))]
    _field: std::marker::PhantomData<F>,
}

impl<F: PrimeField + FftField + Send + Sync> PreparedGpuOodOpeningBatch<F> {
    pub fn root(&self) -> Digest32 {
        #[cfg(feature = "icicle-rust")]
        {
            self.inner.root()
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            [0u8; 32]
        }
    }

    pub fn finish(
        self,
        batching_coefficients: &[F],
    ) -> Result<(BatchFriProof<F>, Vec<BatchLdeRowOpening<F>>), Error> {
        #[cfg(feature = "icicle-rust")]
        {
            return crate::gpu_lock::with_gpu_lock(|| self.inner.finish(batching_coefficients));
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            let _ = batching_coefficients;
            Err(Error::GpuBackendUnavailable)
        }
    }
}

pub struct PreparedGpuSpecializedUscBatch<F: FftField> {
    #[cfg(feature = "icicle-rust")]
    inner: gpu_resident_fri_bn254::PreparedSpecializedUscBatch<F>,
    #[cfg(not(feature = "icicle-rust"))]
    _field: std::marker::PhantomData<F>,
}

impl<F: PrimeField + FftField + Send + Sync> PreparedGpuSpecializedUscBatch<F> {
    pub fn root(&self) -> Digest32 {
        #[cfg(feature = "icicle-rust")]
        {
            self.inner.root()
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            [0u8; 32]
        }
    }

    pub fn finish(
        self,
        batching_coefficients: &[F],
    ) -> Result<(BatchFriProof<F>, Vec<BatchLdeRowOpening<F>>), Error> {
        #[cfg(feature = "icicle-rust")]
        {
            return crate::gpu_lock::with_gpu_lock(|| self.inner.finish(batching_coefficients));
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            let _ = batching_coefficients;
            Err(Error::GpuBackendUnavailable)
        }
    }
}

struct GpuCombinedFriResult<F: FftField> {
    proof: FriProof<F>,
    row_openings: Vec<(F, MerklePath)>,
    combined_lde: CombinedLdeBackend<F>,
}

pub(crate) struct PreparedLowDegreeFromCoeffs<F: FftField> {
    pub lde_values: Vec<F>,
    pub lde_tree: MerkleTree,
    pub proof: FriProof<F>,
}

pub(crate) struct PreparedTableLowDegree<F: FftField> {
    pub polynomial_coefficients: Vec<F>,
    pub lde: PreparedTableLde<F>,
    pub proof: FriProof<F>,
}

#[derive(Clone, Debug)]
pub(crate) enum PreparedTableLde<F: FftField> {
    Cpu {
        values: Vec<F>,
        tree: MerkleTree,
    },
    #[cfg(feature = "icicle-rust")]
    Gpu(PreparedGpuTableLde<F>),
}

impl<F: FftField> PreparedTableLde<F> {
    pub(crate) fn root(&self) -> Digest32 {
        match self {
            Self::Cpu { tree, .. } => tree.root(),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(lde) => lde.root(),
        }
    }
}

#[cfg(feature = "icicle-rust")]
#[derive(Clone)]
pub(crate) struct PreparedGpuTableLde<F: FftField> {
    inner: Arc<gpu_resident_fri_bn254::GpuPreparedCombinedLde<F>>,
}

#[cfg(feature = "icicle-rust")]
impl<F: FftField> fmt::Debug for PreparedGpuTableLde<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedGpuTableLde")
            .field("len", &self.len())
            .field("root", &self.root())
            .finish()
    }
}

#[cfg(feature = "icicle-rust")]
impl<F: FftField> PreparedGpuTableLde<F> {
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    pub(crate) fn root(&self) -> Digest32 {
        self.inner.root()
    }
}

#[cfg(feature = "icicle-rust")]
impl<F: PrimeField + FftField + Send> PreparedGpuTableLde<F> {
    pub(crate) fn open_many(&self, indices: &[usize]) -> Result<Vec<(F, MerklePath)>, Error> {
        self.inner.open_many(indices)
    }

    pub(crate) fn materialize(&self) -> Result<(Vec<F>, MerkleTree), Error> {
        gpu_resident_fri_bn254::materialize_prepared_combined_lde(&self.inner)
    }
}

#[derive(Clone)]
enum CombinedLdeBackend<F: FftField> {
    Cpu {
        values: Vec<F>,
        tree: MerkleTree,
    },
    #[cfg(feature = "icicle-rust")]
    Gpu(Arc<gpu_resident_fri_bn254::GpuPreparedCombinedLde<F>>),
}

impl<F: FftField> CombinedLdeBackend<F> {
    fn cpu(values: Vec<F>, tree: MerkleTree) -> Self {
        Self::Cpu { values, tree }
    }

    #[cfg(feature = "icicle-rust")]
    fn gpu(state: gpu_resident_fri_bn254::GpuPreparedCombinedLde<F>) -> Self {
        Self::Gpu(Arc::new(state))
    }

    fn root(&self) -> Digest32 {
        match self {
            Self::Cpu { tree, .. } => tree.root(),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(state) => state.root(),
        }
    }
}

impl<F: FftField + Send> CombinedLdeBackend<F> {
    fn prefers_batch_opening(&self) -> bool {
        match self {
            Self::Cpu { .. } => false,
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(_) => true,
        }
    }

    #[allow(dead_code)]
    fn open(&self, index: usize) -> Result<(F, MerklePath), Error> {
        match self {
            Self::Cpu { values, tree } => {
                let value = values
                    .get(index)
                    .copied()
                    .ok_or(Error::InvalidDomainSize(index))?;
                Ok((value, tree.open(index)))
            }
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(state) => state.open(index),
        }
    }

    fn open_many(&self, indices: &[usize]) -> Result<Vec<(F, MerklePath)>, Error> {
        match self {
            Self::Cpu { values, tree } => indices
                .iter()
                .copied()
                .map(|index| {
                    let value = values
                        .get(index)
                        .copied()
                        .ok_or(Error::InvalidDomainSize(index))?;
                    Ok((value, tree.open(index)))
                })
                .collect(),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(state) => state.open_many(indices),
        }
    }
}

impl<F: FftField> fmt::Debug for CombinedLdeBackend<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu { values, tree } => f
                .debug_struct("CombinedLdeBackend::Cpu")
                .field("len", &values.len())
                .field("root", &tree.root())
                .finish(),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(state) => f
                .debug_struct("CombinedLdeBackend::Gpu")
                .field("len", &state.len())
                .field("root", &state.root())
                .finish(),
        }
    }
}

#[derive(Clone)]
enum GroupTreeBackend {
    Cpu(MerkleTree),
    #[cfg(feature = "icicle-rust")]
    Gpu(Arc<gpu_batch_merkle_bn254::GpuPreparedGroupTree>),
}

impl GroupTreeBackend {
    fn cpu(tree: MerkleTree) -> Self {
        Self::Cpu(tree)
    }

    #[cfg(feature = "icicle-rust")]
    fn gpu(tree: gpu_batch_merkle_bn254::GpuPreparedGroupTree) -> Self {
        Self::Gpu(Arc::new(tree))
    }

    fn prefers_batch_opening(&self) -> bool {
        match self {
            Self::Cpu(_) => false,
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(_) => true,
        }
    }

    fn root(&self) -> Digest32 {
        match self {
            Self::Cpu(tree) => tree.root(),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(tree) => tree.root(),
        }
    }

    #[allow(dead_code)]
    fn open(&self, index: usize) -> Result<MerklePath, Error> {
        match self {
            Self::Cpu(tree) => Ok(tree.open(index)),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(tree) => tree.open(index),
        }
    }

    fn open_many(&self, indices: &[usize]) -> Result<Vec<MerklePath>, Error> {
        match self {
            Self::Cpu(tree) => Ok(indices.iter().map(|&index| tree.open(index)).collect()),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(tree) => tree.open_many(indices),
        }
    }
}

impl fmt::Debug for GroupTreeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu(tree) => f
                .debug_struct("GroupTreeBackend::Cpu")
                .field("root", &tree.root())
                .finish(),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(tree) => f
                .debug_struct("GroupTreeBackend::Gpu")
                .field("root", &tree.root())
                .finish(),
        }
    }
}

enum FriTreeBackend {
    Cpu(MerkleTree),
    #[cfg(feature = "icicle-rust")]
    Gpu(Arc<gpu_fold_bn254::GpuPreparedMerkleTree>),
}

impl FriTreeBackend {
    fn cpu(tree: MerkleTree) -> Self {
        Self::Cpu(tree)
    }

    #[cfg(feature = "icicle-rust")]
    fn gpu(tree: gpu_fold_bn254::GpuPreparedMerkleTree) -> Self {
        Self::Gpu(Arc::new(tree))
    }

    fn root(&self) -> Digest32 {
        match self {
            Self::Cpu(tree) => tree.root(),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(tree) => tree.root(),
        }
    }

    fn open_many(&self, indices: &[usize]) -> Result<Vec<MerklePath>, Error> {
        match self {
            Self::Cpu(tree) => Ok(indices.iter().map(|&index| tree.open(index)).collect()),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(tree) => tree.open_many(indices),
        }
    }
}

impl fmt::Debug for FriTreeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu(tree) => f
                .debug_struct("FriTreeBackend::Cpu")
                .field("root", &tree.root())
                .finish(),
            #[cfg(feature = "icicle-rust")]
            Self::Gpu(tree) => f
                .debug_struct("FriTreeBackend::Gpu")
                .field("root", &tree.root())
                .finish(),
        }
    }
}

#[derive(Clone)]
pub struct PreparedBatchFriProof<'a, F: FftField> {
    lde_columns: Vec<&'a [F]>,
    gpu_lde_store: Option<&'a PreparedGpuLdeColumnStore<F>>,
    gpu_column_indices: Vec<usize>,
    group_trees: Vec<GroupTreeBackend>,
    combined_lde: Option<CombinedLdeBackend<F>>,
    row_root: Digest32,
    row_count: usize,
    column_count: usize,
    column_degree_bounds: Vec<usize>,
    commitment_groups: Vec<BatchCommitmentGroup>,
    batching_coefficients: Vec<F>,
    combined: FriProof<F>,
    blowup: usize,
}

impl<'a, F: PrimeField + FftField + Send + Sync> PreparedBatchFriProof<'a, F> {
    pub fn proof_with_openings(&self, opened_indices: &[usize]) -> Result<BatchFriProof<F>, Error> {
        let row_openings = self.build_row_openings(opened_indices)?;

        Ok(BatchFriProof {
            row_root: self.row_root,
            row_count: self.row_count,
            column_count: self.column_count,
            column_degree_bounds: self.column_degree_bounds.clone(),
            commitment_groups: self.commitment_groups.clone(),
            row_openings,
            batching_coefficients: self.batching_coefficients.clone(),
            combined: self.combined.clone(),
        })
    }

    fn build_row_openings(
        &self,
        opened_indices: &[usize],
    ) -> Result<Vec<BatchRowOpening<F>>, Error> {
        if self.prefers_batched_openings() {
            self.build_row_openings_batched(opened_indices)
        } else {
            self.build_row_openings_parallel(opened_indices)
        }
    }

    fn prefers_batched_openings(&self) -> bool {
        self.gpu_lde_store.is_some()
            || self
                .combined_lde
                .as_ref()
                .is_some_and(CombinedLdeBackend::prefers_batch_opening)
            || self
                .group_trees
                .iter()
                .any(GroupTreeBackend::prefers_batch_opening)
    }

    fn build_row_openings_batched(
        &self,
        opened_indices: &[usize],
    ) -> Result<Vec<BatchRowOpening<F>>, Error> {
        if opened_indices.is_empty() {
            return Ok(Vec::new());
        }

        let mut lde_indices = Vec::with_capacity(opened_indices.len());
        for &index in opened_indices {
            if index >= self.row_count {
                return Err(Error::InvalidDomainSize(index));
            }
            lde_indices.push(index * self.blowup);
        }
        let rows = self.lde_rows(&lde_indices)?;

        let group_paths_by_group = self.group_paths_by_group(&lde_indices)?;
        let combined_openings = self.combined_openings(&lde_indices)?;

        opened_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(position, index)| {
                let group_paths = group_paths_by_group
                    .iter()
                    .map(|paths| paths[position].clone())
                    .collect::<Vec<_>>();
                let (combined_lde_value, combined_lde_path) = &combined_openings[position];
                Ok(BatchRowOpening {
                    index,
                    row: rows[position].clone(),
                    row_path: group_paths[0].clone(),
                    group_paths,
                    combined_lde_value: *combined_lde_value,
                    combined_lde_path: combined_lde_path.clone(),
                })
            })
            .collect()
    }

    fn build_row_openings_parallel(
        &self,
        opened_indices: &[usize],
    ) -> Result<Vec<BatchRowOpening<F>>, Error> {
        let worker_count = parallel_worker_count(opened_indices.len());
        if worker_count <= 1 || opened_indices.len() < 2 {
            return opened_indices
                .iter()
                .copied()
                .map(|index| self.build_row_opening(index))
                .collect();
        }

        let chunk_size = (opened_indices.len() + worker_count - 1) / worker_count;
        let mut chunk_results = thread::scope(|scope| {
            let mut handles = Vec::new();
            for (chunk_index, chunk) in opened_indices.chunks(chunk_size).enumerate() {
                handles.push(scope.spawn(move || {
                    let openings = chunk
                        .iter()
                        .copied()
                        .map(|index| self.build_row_opening(index))
                        .collect::<Result<Vec<_>, Error>>()?;
                    Ok((chunk_index, openings))
                }));
            }

            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| Error::LowDegreeCheckFailed)?)
                .collect::<Result<Vec<_>, Error>>()
        })?;

        chunk_results.sort_by_key(|(chunk_index, _)| *chunk_index);
        Ok(chunk_results
            .into_iter()
            .flat_map(|(_, openings)| openings)
            .collect())
    }

    fn build_row_opening(&self, index: usize) -> Result<BatchRowOpening<F>, Error> {
        if index >= self.row_count {
            return Err(Error::InvalidDomainSize(index));
        }
        let lde_index = index * self.blowup;
        let opening = self.build_lde_row_opening(lde_index)?;
        Ok(BatchRowOpening {
            index,
            row: opening.row,
            row_path: opening.group_paths[0].clone(),
            group_paths: opening.group_paths,
            combined_lde_value: opening.combined_lde_value,
            combined_lde_path: opening.combined_lde_path,
        })
    }

    pub fn lde_row_openings(
        &self,
        lde_indices: &[usize],
    ) -> Result<Vec<BatchLdeRowOpening<F>>, Error> {
        if self.prefers_batched_openings() {
            self.build_lde_row_openings_batched(lde_indices)
        } else {
            lde_indices
                .iter()
                .copied()
                .map(|index| self.build_lde_row_opening(index))
                .collect()
        }
    }

    fn lde_rows(&self, lde_indices: &[usize]) -> Result<Vec<Vec<F>>, Error> {
        let lde_size = self.row_count * self.blowup;
        if lde_indices.iter().any(|index| *index >= lde_size) {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        if lde_indices.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(store) = self.gpu_lde_store {
            return store.lde_rows(&self.gpu_column_indices, lde_indices);
        }
        Ok(lde_indices
            .iter()
            .map(|&index| {
                self.lde_columns
                    .iter()
                    .map(|column| column[index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>())
    }

    fn group_paths_by_group(&self, lde_indices: &[usize]) -> Result<Vec<Vec<MerklePath>>, Error> {
        if !self.group_trees.is_empty() {
            return self
                .group_trees
                .iter()
                .map(|tree| tree.open_many(lde_indices))
                .collect::<Result<Vec<_>, Error>>();
        }
        if let Some(store) = self.gpu_lde_store {
            let mapped_groups = self
                .commitment_groups
                .iter()
                .map(|group| {
                    let columns = group
                        .columns
                        .iter()
                        .copied()
                        .map(|column| {
                            self.gpu_column_indices
                                .get(column)
                                .copied()
                                .ok_or(Error::LowDegreeCheckFailed)
                        })
                        .collect::<Result<Vec<_>, Error>>()?;
                    Ok(BatchCommitmentGroup {
                        columns,
                        root: group.root,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            return store.group_paths(&mapped_groups, lde_indices);
        }
        Err(Error::GpuBackendFailed)
    }

    fn combined_openings(&self, lde_indices: &[usize]) -> Result<Vec<(F, MerklePath)>, Error> {
        if let Some(combined_lde) = &self.combined_lde {
            return combined_lde.open_many(lde_indices);
        }
        if let Some(store) = self.gpu_lde_store {
            return store.combined_openings(
                &self.gpu_column_indices,
                &self.batching_coefficients,
                &self.column_degree_bounds,
                self.row_count,
                self.blowup,
                lde_indices,
                self.combined.root,
            );
        }
        Err(Error::GpuBackendFailed)
    }

    fn build_lde_row_openings_batched(
        &self,
        lde_indices: &[usize],
    ) -> Result<Vec<BatchLdeRowOpening<F>>, Error> {
        if lde_indices.is_empty() {
            return Ok(Vec::new());
        }
        let lde_size = self.row_count * self.blowup;
        if lde_indices.iter().any(|index| *index >= lde_size) {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        let rows = self.lde_rows(lde_indices)?;
        let group_paths_by_group = self.group_paths_by_group(lde_indices)?;
        let combined_openings = self.combined_openings(lde_indices)?;

        lde_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(position, lde_index)| {
                let group_paths = group_paths_by_group
                    .iter()
                    .map(|paths| paths[position].clone())
                    .collect::<Vec<_>>();
                let (combined_lde_value, combined_lde_path) = &combined_openings[position];
                Ok(BatchLdeRowOpening {
                    lde_index,
                    row: rows[position].clone(),
                    group_paths,
                    combined_lde_value: *combined_lde_value,
                    combined_lde_path: combined_lde_path.clone(),
                })
            })
            .collect()
    }

    fn build_lde_row_opening(&self, lde_index: usize) -> Result<BatchLdeRowOpening<F>, Error> {
        let mut openings = self.build_lde_row_openings_batched(&[lde_index])?;
        openings.pop().ok_or(Error::LowDegreeCheckFailed)
    }
}

pub fn interpolate<F: FftField>(base_evals: &[F]) -> Result<DensePolynomial<F>, Error> {
    lde::interpolate(base_evals)
}

pub fn evaluate_lde<F: FftField>(base_evals: &[F], blowup: usize) -> Result<Vec<F>, Error> {
    lde::evaluate_dense_lde(base_evals, blowup)
}

pub fn evaluate_lde_from_coeffs<F: FftField>(
    coeffs: &[F],
    degree_bound: usize,
    blowup: usize,
) -> Result<Vec<F>, Error> {
    lde::evaluate_dense_lde_from_coeffs(coeffs, degree_bound, blowup)
}

#[derive(Clone, Debug)]
struct SparseLdeSupport<F: FftField> {
    index: usize,
    point: F,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FriFoldBackendKind {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchCombineBackendKind {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchMerkleBackendKind {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuResidentFriBackendKind {
    Cpu,
    Gpu,
}

pub fn evaluate_auto_sparse_lde<F: FftField + Send + Sync>(
    base_evals: &[F],
    blowup: usize,
) -> Result<Vec<F>, Error> {
    if !is_power_of_two(base_evals.len()) {
        return Err(Error::InvalidDomainSize(base_evals.len()));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }

    let support_indices = non_zero_indices(base_evals);
    if should_use_sparse_lde(support_indices.len(), base_evals.len(), blowup) {
        evaluate_sparse_lde_with_support(base_evals, blowup, &support_indices)
    } else {
        evaluate_lde(base_evals, blowup)
    }
}

pub fn evaluate_auto_sparse_lde_with_support<F: FftField + Send + Sync>(
    base_evals: &[F],
    blowup: usize,
    support_indices: &[usize],
) -> Result<Vec<F>, Error> {
    validate_sparse_support(base_evals, support_indices)?;
    if should_use_sparse_lde(support_indices.len(), base_evals.len(), blowup) {
        evaluate_sparse_lde_with_support(base_evals, blowup, support_indices)
    } else {
        evaluate_lde(base_evals, blowup)
    }
}

pub fn evaluate_sparse_lde<F: FftField + Send + Sync>(
    base_evals: &[F],
    blowup: usize,
) -> Result<Vec<F>, Error> {
    let support_indices = non_zero_indices(base_evals);
    evaluate_sparse_lde_with_support(base_evals, blowup, &support_indices)
}

pub fn evaluate_sparse_lde_with_support<F: FftField + Send + Sync>(
    base_evals: &[F],
    blowup: usize,
    support_indices: &[usize],
) -> Result<Vec<F>, Error> {
    if !is_power_of_two(base_evals.len()) {
        return Err(Error::InvalidDomainSize(base_evals.len()));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }

    validate_sparse_support(base_evals, support_indices)?;
    Ok(evaluate_sparse_columns_with_support(&[base_evals], blowup, support_indices)?.remove(0))
}

pub fn should_use_sparse_lde(non_zero_count: usize, base_len: usize, blowup: usize) -> bool {
    if non_zero_count == 0 {
        return true;
    }
    if !is_power_of_two(base_len) || !is_power_of_two(blowup) {
        return false;
    }

    let lde_size = base_len * blowup;
    let sparse_threshold = (lde_size.trailing_zeros() as usize * SPARSE_LDE_LOG_FACTOR).max(1);
    non_zero_count <= sparse_threshold
}

pub fn non_zero_indices<F: FftField>(base_evals: &[F]) -> Vec<usize> {
    base_evals
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (!value.is_zero()).then_some(index))
        .collect()
}

fn validate_sparse_support<F: FftField>(
    base_evals: &[F],
    support_indices: &[usize],
) -> Result<(), Error> {
    let mut previous = None;
    for &index in support_indices {
        if index >= base_evals.len() || previous.is_some_and(|prev| prev >= index) {
            return Err(Error::InvalidDomainSize(index));
        }
        previous = Some(index);
    }

    let mut support_cursor = 0;
    for (index, value) in base_evals.iter().enumerate() {
        if support_cursor < support_indices.len() && support_indices[support_cursor] == index {
            support_cursor += 1;
        } else if !value.is_zero() {
            return Err(Error::CommitmentMismatch("sparse LDE support"));
        }
    }
    Ok(())
}

fn evaluate_sparse_columns_with_support<F: FftField + Send + Sync>(
    base_columns: &[&[F]],
    blowup: usize,
    support_indices: &[usize],
) -> Result<Vec<Vec<F>>, Error> {
    if base_columns.is_empty() {
        return Err(Error::LowDegreeCheckFailed);
    }
    let base_len = base_columns[0].len();
    if !is_power_of_two(base_len) {
        return Err(Error::InvalidDomainSize(base_len));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    if base_columns.iter().any(|column| column.len() != base_len) {
        return Err(Error::InvalidDomainSize(base_len));
    }
    for column in base_columns {
        validate_sparse_support(column, support_indices)?;
    }

    let lde_size = base_len * blowup;
    if support_indices.is_empty() {
        return Ok(vec![vec![F::zero(); lde_size]; base_columns.len()]);
    }

    let base_domain =
        GeneralEvaluationDomain::<F>::new(base_len).ok_or(Error::InvalidDomainSize(base_len))?;
    let lde_domain =
        GeneralEvaluationDomain::<F>::new(lde_size).ok_or(Error::InvalidDomainSize(lde_size))?;
    let support = support_indices
        .iter()
        .map(|&index| SparseLdeSupport {
            index,
            point: base_domain.element(index),
        })
        .collect::<Vec<_>>();
    let lde_points = (0..lde_size)
        .map(|index| lde_domain.element(index))
        .collect::<Vec<_>>();
    let base_size_inv = F::from(base_len as u64)
        .inverse()
        .ok_or(Error::LowDegreeCheckFailed)?;
    let chunk_len = sparse_lde_chunk_len(support.len(), lde_size);
    let chunks = (0..lde_size)
        .step_by(chunk_len)
        .map(|start| (start, (start + chunk_len).min(lde_size)))
        .collect::<Vec<_>>();

    let mut chunk_results = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunks.len());
        for &(start, end) in &chunks {
            let support = &support;
            let lde_points = &lde_points;
            handles.push(scope.spawn(move || {
                evaluate_sparse_columns_chunk(
                    base_columns,
                    blowup,
                    base_len,
                    base_size_inv,
                    support,
                    lde_points,
                    start,
                    end,
                )
            }));
        }

        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| Error::LowDegreeCheckFailed))
            .collect::<Result<Vec<_>, Error>>()
    })?;

    chunk_results.sort_by_key(|(start, _)| *start);
    let mut columns = vec![vec![F::zero(); lde_size]; base_columns.len()];
    for (start, chunk_columns) in chunk_results {
        for (column, chunk_column) in columns.iter_mut().zip(chunk_columns) {
            column[start..start + chunk_column.len()].copy_from_slice(&chunk_column);
        }
    }
    Ok(columns)
}

fn sparse_lde_chunk_len(support_len: usize, lde_size: usize) -> usize {
    let denominators_per_point = support_len.max(1);
    let target_points = (SPARSE_LDE_TARGET_CHUNK_DENOMINATORS / denominators_per_point).max(1);
    target_points.min(lde_size).max(1)
}

fn evaluate_sparse_columns_chunk<F: FftField>(
    base_columns: &[&[F]],
    blowup: usize,
    base_len: usize,
    base_size_inv: F,
    support: &[SparseLdeSupport<F>],
    lde_points: &[F],
    start: usize,
    end: usize,
) -> (usize, Vec<Vec<F>>) {
    let non_base_indices = (start..end)
        .filter(|index| index % blowup != 0)
        .collect::<Vec<_>>();
    let mut denominators = Vec::with_capacity(non_base_indices.len() * support.len());
    for &lde_index in &non_base_indices {
        let point = lde_points[lde_index];
        denominators.extend(support.iter().map(|entry| point - entry.point));
    }
    batch_inversion(&mut denominators);

    let mut inverse_offset = 0;
    let mut columns = vec![Vec::with_capacity(end - start); base_columns.len()];
    for lde_index in start..end {
        if lde_index % blowup == 0 {
            let base_index = lde_index / blowup;
            for (column, base_column) in columns.iter_mut().zip(base_columns) {
                column.push(base_column[base_index]);
            }
            continue;
        }

        let point = lde_points[lde_index];
        let vanishing_over_size = (point.pow(&[base_len as u64]) - F::one()) * base_size_inv;
        let inverse_slice = &denominators[inverse_offset..inverse_offset + support.len()];
        inverse_offset += support.len();

        for (column, base_column) in columns.iter_mut().zip(base_columns) {
            let value =
                support
                    .iter()
                    .zip(inverse_slice)
                    .fold(F::zero(), |acc, (entry, inverse)| {
                        acc + vanishing_over_size
                            * entry.point
                            * *inverse
                            * base_column[entry.index]
                    });
            column.push(value);
        }
    }

    (start, columns)
}

pub fn degree_from_evals<F: FftField>(evals: &[F]) -> Result<usize, Error> {
    if !is_power_of_two(evals.len()) {
        return Err(Error::InvalidDomainSize(evals.len()));
    }
    let domain = GeneralEvaluationDomain::<F>::new(evals.len()).unwrap();
    let coeffs = domain.ifft(evals);
    Ok(coeffs
        .iter()
        .rposition(|coeff| !coeff.is_zero())
        .unwrap_or(0))
}

pub fn commit_single_column<F: FftField + Sync>(evals: &[F]) -> Digest32 {
    MerkleTree::from_column(evals).root()
}

pub fn prove_low_degree<F: PrimeField + FftField + Send + Sync>(
    base_evals: &[F],
    degree_bound: usize,
    blowup: usize,
) -> Result<FriProof<F>, Error> {
    let lde_evals = evaluate_lde(base_evals, blowup)?;
    prove_low_degree_from_lde(&lde_evals, degree_bound, blowup)
}

pub fn prove_low_degree_from_lde<F: PrimeField + FftField + Send + Sync>(
    lde_evals: &[F],
    degree_bound: usize,
    blowup: usize,
) -> Result<FriProof<F>, Error> {
    validate_low_degree_lde(lde_evals, degree_bound, blowup)?;
    if let Some(proof) = prove_low_degree_from_lde_gpu_resident(lde_evals, degree_bound, blowup)? {
        return Ok(proof);
    }
    let initial_tree = MerkleTree::from_column(lde_evals);
    prove_low_degree_from_lde_with_initial_tree(lde_evals, degree_bound, blowup, initial_tree)
}

fn prove_low_degree_from_lde_gpu_resident<F: PrimeField + FftField>(
    lde_evals: &[F],
    degree_bound: usize,
    blowup: usize,
) -> Result<Option<FriProof<F>>, Error> {
    let lde_size = lde_evals.len();
    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Cpu) => Ok(None),
        Some(GpuResidentFriBackendKind::Gpu) => {
            try_prove_low_degree_from_lde_gpu_resident(lde_evals, degree_bound, blowup)?
                .map(Some)
                .ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if requested_fold_backend() == Some(FriFoldBackendKind::Cpu) {
                return Ok(None);
            }
            let gpu_fold_would_run = match requested_fold_backend() {
                Some(FriFoldBackendKind::Gpu) => true,
                Some(FriFoldBackendKind::Cpu) => false,
                None => should_auto_use_gpu_fold::<F>(lde_size),
            };
            if (should_auto_use_gpu_resident_fri::<F>(lde_size) || gpu_fold_would_run)
                && gpu_resident_fri_supports_field::<F>()
            {
                if let Ok(Some(proof)) =
                    try_prove_low_degree_from_lde_gpu_resident(lde_evals, degree_bound, blowup)
                {
                    return Ok(Some(proof));
                }
            }
            Ok(None)
        }
    }
}

fn validate_low_degree_lde<F: FftField>(
    lde_evals: &[F],
    degree_bound: usize,
    blowup: usize,
) -> Result<(), Error> {
    if degree_bound == 0 || !is_power_of_two(degree_bound) {
        return Err(Error::InvalidDomainSize(degree_bound));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    if lde_evals.len() != degree_bound * blowup || !is_power_of_two(lde_evals.len()) {
        return Err(Error::InvalidDomainSize(lde_evals.len()));
    }
    Ok(())
}

fn prove_low_degree_from_lde_with_initial_tree<F: PrimeField + FftField + Send + Sync>(
    lde_evals: &[F],
    degree_bound: usize,
    blowup: usize,
    initial_tree: MerkleTree,
) -> Result<FriProof<F>, Error> {
    validate_low_degree_lde(lde_evals, degree_bound, blowup)?;

    let mut layers = vec![lde_evals.to_vec()];
    let mut trees = vec![FriTreeBackend::cpu(initial_tree)];
    let mut roots = vec![trees[0].root()];

    let mut transcript = fri_transcript::<F>(roots[0], degree_bound, blowup, layers[0].len());
    let mut current_degree_bound = degree_bound;

    while current_degree_bound > 1 {
        let alpha = transcript.challenge::<F>(b"fold-alpha");
        let (next, tree) = fold_layer_with_tree(layers.last().unwrap(), alpha)?;
        let root = tree.root();
        transcript.absorb_bytes(b"fold-root", &root);

        layers.push(next);
        trees.push(tree);
        roots.push(root);
        current_degree_bound = (current_degree_bound + 1) / 2;
    }

    let final_evals = layers.last().unwrap().clone();
    let queries = sample_queries(&mut transcript, &layers, &trees)?;

    Ok(FriProof {
        root: roots[0],
        roots,
        final_evals,
        queries,
        degree_bound,
        blowup,
        domain_size: layers[0].len(),
    })
}

pub fn verify_low_degree<F: PrimeField + FftField + Send + Sync>(
    proof: &FriProof<F>,
    expected_root: Digest32,
) -> Result<(), Error> {
    if proof.root != expected_root || proof.roots.first() != Some(&expected_root) {
        return Err(Error::MerkleRootMismatch);
    }
    if proof.degree_bound == 0 || !is_power_of_two(proof.degree_bound) {
        return Err(Error::InvalidDomainSize(proof.degree_bound));
    }
    if !is_power_of_two(proof.blowup) {
        return Err(Error::InvalidBlowup(proof.blowup));
    }
    if proof.domain_size != proof.degree_bound * proof.blowup || !is_power_of_two(proof.domain_size)
    {
        return Err(Error::InvalidDomainSize(proof.domain_size));
    }

    let expected_rounds = proof.degree_bound.trailing_zeros() as usize;
    if proof.roots.len() != expected_rounds + 1 {
        return Err(Error::LowDegreeCheckFailed);
    }
    if proof.final_evals.len() != proof.blowup {
        return Err(Error::LowDegreeCheckFailed);
    }
    if proof
        .final_evals
        .iter()
        .any(|value| *value != proof.final_evals[0])
    {
        return Err(Error::LowDegreeCheckFailed);
    }
    if commit_single_column(&proof.final_evals) != *proof.roots.last().unwrap() {
        return Err(Error::MerkleRootMismatch);
    }

    let mut transcript = fri_transcript::<F>(
        proof.root,
        proof.degree_bound,
        proof.blowup,
        proof.domain_size,
    );
    let mut alphas = Vec::with_capacity(expected_rounds);
    for root in proof.roots.iter().skip(1) {
        alphas.push(transcript.challenge::<F>(b"fold-alpha"));
        transcript.absorb_bytes(b"fold-root", root);
    }

    if proof.queries.len() != query_count(proof.domain_size) {
        return Err(Error::LowDegreeCheckFailed);
    }

    for query in &proof.queries {
        let sampled_index = transcript.challenge_index(b"fri-query", proof.domain_size);
        if query.index != sampled_index {
            return Err(Error::LowDegreeCheckFailed);
        }
    }
    verify_queries_parallel(proof, &alphas)?;

    Ok(())
}

pub fn prove_batch_low_degree<F: PrimeField + FftField + Send + Sync>(
    columns: &[Vec<F>],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    opened_indices: &[usize],
) -> Result<BatchFriProof<F>, Error> {
    let groups = vec![(0..columns.len()).collect::<Vec<_>>()];
    prove_batch_low_degree_with_groups(
        columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
        opened_indices,
        &groups,
    )
}

pub fn prove_batch_low_degree_with_groups<F: PrimeField + FftField + Send + Sync>(
    columns: &[Vec<F>],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    opened_indices: &[usize],
    commitment_groups: &[Vec<usize>],
) -> Result<BatchFriProof<F>, Error> {
    let lde_columns = columns
        .iter()
        .map(|column| evaluate_lde(column, blowup))
        .collect::<Result<Vec<_>, Error>>()?;
    prove_batch_low_degree_with_groups_and_lde(
        columns,
        &lde_columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
        opened_indices,
        commitment_groups,
    )
}

fn column_slices<F>(columns: &[Vec<F>]) -> Vec<&[F]> {
    columns.iter().map(Vec::as_slice).collect()
}

pub fn prove_batch_low_degree_with_groups_and_lde<F: PrimeField + FftField + Send + Sync>(
    columns: &[Vec<F>],
    lde_columns: &[Vec<F>],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    opened_indices: &[usize],
    commitment_groups: &[Vec<usize>],
) -> Result<BatchFriProof<F>, Error> {
    let column_refs = column_slices(columns);
    let lde_column_refs = column_slices(lde_columns);
    prepare_batch_low_degree_with_groups_and_lde(
        &column_refs,
        &lde_column_refs,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
        commitment_groups,
    )?
    .proof_with_openings(opened_indices)
}

pub fn prepare_batch_commitment_group<F: PrimeField + FftField + Send + Sync>(
    lde_columns: &[&[F]],
    columns: Vec<usize>,
) -> Result<PreparedBatchCommitments, Error> {
    if lde_columns.is_empty() || lde_columns.len() != columns.len() {
        return Err(Error::LowDegreeCheckFailed);
    }
    if lde_columns
        .iter()
        .any(|column| column.len() != lde_columns[0].len())
    {
        return Err(Error::InvalidDomainSize(lde_columns[0].len()));
    }
    let local_group = (0..lde_columns.len()).collect::<Vec<_>>();
    let (mut group_trees, gpu_lde_columns) =
        build_batch_group_trees_with_committed_columns(lde_columns, &[local_group])?;
    let tree = group_trees.pop().ok_or(Error::LowDegreeCheckFailed)?;
    let root = tree.root();
    #[cfg(not(feature = "icicle-rust"))]
    let _ = gpu_lde_columns;
    Ok(PreparedBatchCommitments {
        group_trees: vec![tree],
        commitment_groups: vec![BatchCommitmentGroup { columns, root }],
        #[cfg(feature = "icicle-rust")]
        gpu_lde_columns,
    })
}

pub fn try_prepare_batch_commitment_group_from_coeffs_gpu_lde<
    F: PrimeField + FftField + Send + Sync,
>(
    coeff_columns: &[&[F]],
    degree_bound: usize,
    blowup: usize,
    columns: Vec<usize>,
) -> Result<Option<PreparedBatchCommitments>, Error> {
    if coeff_columns.is_empty() || coeff_columns.len() != columns.len() {
        return Err(Error::LowDegreeCheckFailed);
    }
    if degree_bound == 0 || !is_power_of_two(degree_bound) {
        return Err(Error::InvalidDomainSize(degree_bound));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    if coeff_columns
        .iter()
        .any(|column| column.len() > degree_bound)
    {
        return Err(Error::LowDegreeCheckFailed);
    }

    let plan = lde::DenseLdePlan::<F>::new(degree_bound, blowup)?;
    if lde::select_dense_lde_backend::<F>(&plan) != lde::DenseLdeBackendKind::Gpu {
        return Ok(None);
    }

    #[cfg(feature = "icicle-rust")]
    {
        let prepared = crate::gpu_lock::with_gpu_lock(|| {
            gpu_batch_merkle_bn254::try_commit_coeff_columns_as_lde::<F>(
                coeff_columns,
                degree_bound,
                blowup,
            )
        })?;
        let Some((tree, gpu_lde_columns)) = prepared else {
            return Ok(None);
        };
        let tree = GroupTreeBackend::gpu(tree);
        let root = tree.root();
        let commitments = PreparedBatchCommitments {
            group_trees: vec![tree],
            commitment_groups: vec![BatchCommitmentGroup { columns, root }],
            gpu_lde_columns: Some(gpu_lde_columns),
        };
        Ok(Some(commitments))
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (coeff_columns, degree_bound, blowup, columns);
        Ok(None)
    }
}

#[derive(Clone, Debug)]
pub struct GpuLookupAuxiliaryPrecommit<F: FftField> {
    pub table_reciprocal: Vec<F>,
    pub table_reciprocal_coeffs: Vec<F>,
    pub table_accumulator: Vec<F>,
    pub table_accumulator_coeffs: Vec<F>,
    pub table_commitments: PreparedBatchCommitments,
    pub query_reciprocal: Vec<F>,
    pub query_reciprocal_coeffs: Vec<F>,
    pub query_commitments: PreparedBatchCommitments,
}

#[allow(clippy::too_many_arguments)]
pub fn try_prepare_lookup_auxiliary_gpu_resident<F: PrimeField + FftField + Send + Sync>(
    table_values: &[F],
    multiplicity: &[F],
    query_values: &[F],
    beta: F,
    table_row_count: usize,
    query_row_count: usize,
    blowup: usize,
) -> Result<Option<GpuLookupAuxiliaryPrecommit<F>>, Error> {
    if table_values.len() != table_row_count
        || multiplicity.len() != table_row_count
        || query_values.len() != query_row_count
    {
        return Err(Error::InvalidDomainSize(table_row_count));
    }
    if table_values
        .iter()
        .chain(query_values)
        .any(|value| (*value + beta).is_zero())
    {
        return Err(Error::InvalidBetaDenominator);
    }
    if table_row_count == 0
        || query_row_count == 0
        || !is_power_of_two(table_row_count)
        || !is_power_of_two(query_row_count)
    {
        return Err(Error::InvalidDomainSize(table_row_count));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }

    let table_plan = lde::DenseLdePlan::<F>::new(table_row_count, blowup)?;
    let query_plan = lde::DenseLdePlan::<F>::new(query_row_count, blowup)?;
    let table_uses_gpu =
        lde::select_dense_lde_backend::<F>(&table_plan) == lde::DenseLdeBackendKind::Gpu;
    let query_uses_gpu =
        lde::select_dense_lde_backend::<F>(&query_plan) == lde::DenseLdeBackendKind::Gpu;
    if !table_uses_gpu && !query_uses_gpu {
        return Ok(None);
    }

    #[cfg(feature = "icicle-rust")]
    {
        let prepared = crate::gpu_lock::with_gpu_lock(|| {
            gpu_batch_merkle_bn254::try_prepare_lookup_auxiliary::<F>(
                table_values,
                multiplicity,
                query_values,
                beta,
                table_row_count,
                query_row_count,
                blowup,
            )
        })?;
        let Some(prepared) = prepared else {
            return Ok(None);
        };

        let table_tree = GroupTreeBackend::gpu(prepared.table_tree);
        let table_root = table_tree.root();
        let query_tree = GroupTreeBackend::gpu(prepared.query_tree);
        let query_root = query_tree.root();

        Ok(Some(GpuLookupAuxiliaryPrecommit {
            table_reciprocal: prepared.table_reciprocal,
            table_reciprocal_coeffs: prepared.table_reciprocal_coeffs,
            table_accumulator: prepared.table_accumulator,
            table_accumulator_coeffs: prepared.table_accumulator_coeffs,
            table_commitments: PreparedBatchCommitments {
                group_trees: vec![table_tree],
                commitment_groups: vec![BatchCommitmentGroup {
                    columns: vec![1, 2],
                    root: table_root,
                }],
                gpu_lde_columns: Some(prepared.table_lde_columns),
            },
            query_reciprocal: prepared.query_reciprocal,
            query_reciprocal_coeffs: prepared.query_reciprocal_coeffs,
            query_commitments: PreparedBatchCommitments {
                group_trees: vec![query_tree],
                commitment_groups: vec![BatchCommitmentGroup {
                    columns: vec![1],
                    root: query_root,
                }],
                gpu_lde_columns: Some(prepared.query_lde_columns),
            },
        }))
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (
            table_values,
            multiplicity,
            query_values,
            beta,
            table_row_count,
            query_row_count,
            blowup,
        );
        Ok(None)
    }
}

pub fn try_product_high_coefficients_gpu<F: PrimeField + FftField + Send + Sync>(
    left: &[F],
    right: &[F],
    domain_size: usize,
) -> Result<Option<Vec<F>>, Error> {
    if domain_size == 0 || !is_power_of_two(domain_size) {
        return Err(Error::InvalidDomainSize(domain_size));
    }
    if left.len() > domain_size || right.len() > domain_size {
        return Err(Error::LowDegreeCheckFailed);
    }

    let plan = lde::DenseLdePlan::<F>::new(domain_size, 2)?;
    if lde::select_dense_lde_backend::<F>(&plan) != lde::DenseLdeBackendKind::Gpu {
        return Ok(None);
    }

    #[cfg(feature = "icicle-rust")]
    {
        crate::gpu_lock::with_gpu_lock(|| {
            gpu_batch_merkle_bn254::try_product_high_coefficients::<F>(left, right, domain_size)
        })
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (left, right, domain_size);
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn try_prepare_table_constraint_quotient_gpu_resident<
    F: PrimeField + FftField + Send + Sync,
>(
    table_coeffs: &[F],
    multiplicity_coeffs: &[F],
    reciprocal_coeffs: &[F],
    domain_size: usize,
    blowup: usize,
    eta_t1: F,
    eta_t2: F,
    columns: Vec<usize>,
) -> Result<Option<PreparedConstraintQuotient<F>>, Error> {
    if columns.len() != 1 {
        return Err(Error::LowDegreeCheckFailed);
    }
    if domain_size == 0 || !is_power_of_two(domain_size) {
        return Err(Error::InvalidDomainSize(domain_size));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    if table_coeffs.len() > domain_size
        || multiplicity_coeffs.len() > domain_size
        || reciprocal_coeffs.len() > domain_size
    {
        return Err(Error::LowDegreeCheckFailed);
    }

    let product_plan = lde::DenseLdePlan::<F>::new(domain_size, 2)?;
    let lde_plan = lde::DenseLdePlan::<F>::new(domain_size, blowup)?;
    if lde::select_dense_lde_backend::<F>(&product_plan) != lde::DenseLdeBackendKind::Gpu
        || lde::select_dense_lde_backend::<F>(&lde_plan) != lde::DenseLdeBackendKind::Gpu
    {
        return Ok(None);
    }

    #[cfg(feature = "icicle-rust")]
    {
        let prepared = crate::gpu_lock::with_gpu_lock(|| {
            gpu_batch_merkle_bn254::try_prepare_table_constraint_quotient::<F>(
                table_coeffs,
                multiplicity_coeffs,
                reciprocal_coeffs,
                domain_size,
                blowup,
                eta_t1,
                eta_t2,
            )
        })?;
        let Some(output) = prepared else {
            return Ok(None);
        };
        let tree = GroupTreeBackend::gpu(output.tree);
        let root = tree.root();
        Ok(Some(PreparedConstraintQuotient {
            coeffs: output.coeffs,
            base_evals: output.base_evals,
            commitments: PreparedBatchCommitments {
                group_trees: vec![tree],
                commitment_groups: vec![BatchCommitmentGroup { columns, root }],
                gpu_lde_columns: Some(output.lde_columns),
            },
        }))
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (
            table_coeffs,
            multiplicity_coeffs,
            reciprocal_coeffs,
            domain_size,
            blowup,
            eta_t1,
            eta_t2,
            columns,
        );
        Ok(None)
    }
}

pub fn try_prepare_query_constraint_quotient_gpu_resident<
    F: PrimeField + FftField + Send + Sync,
>(
    query_coeffs: &[F],
    reciprocal_coeffs: &[F],
    domain_size: usize,
    blowup: usize,
    columns: Vec<usize>,
) -> Result<Option<PreparedConstraintQuotient<F>>, Error> {
    if columns.len() != 1 {
        return Err(Error::LowDegreeCheckFailed);
    }
    if domain_size == 0 || !is_power_of_two(domain_size) {
        return Err(Error::InvalidDomainSize(domain_size));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    if query_coeffs.len() > domain_size || reciprocal_coeffs.len() > domain_size {
        return Err(Error::LowDegreeCheckFailed);
    }

    let product_plan = lde::DenseLdePlan::<F>::new(domain_size, 2)?;
    let lde_plan = lde::DenseLdePlan::<F>::new(domain_size, blowup)?;
    if lde::select_dense_lde_backend::<F>(&product_plan) != lde::DenseLdeBackendKind::Gpu
        || lde::select_dense_lde_backend::<F>(&lde_plan) != lde::DenseLdeBackendKind::Gpu
    {
        return Ok(None);
    }

    #[cfg(feature = "icicle-rust")]
    {
        let prepared = crate::gpu_lock::with_gpu_lock(|| {
            gpu_batch_merkle_bn254::try_prepare_query_constraint_quotient::<F>(
                query_coeffs,
                reciprocal_coeffs,
                domain_size,
                blowup,
            )
        })?;
        let Some(output) = prepared else {
            return Ok(None);
        };
        let tree = GroupTreeBackend::gpu(output.tree);
        let root = tree.root();
        Ok(Some(PreparedConstraintQuotient {
            coeffs: output.coeffs,
            base_evals: output.base_evals,
            commitments: PreparedBatchCommitments {
                group_trees: vec![tree],
                commitment_groups: vec![BatchCommitmentGroup { columns, root }],
                gpu_lde_columns: Some(output.lde_columns),
            },
        }))
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (
            query_coeffs,
            reciprocal_coeffs,
            domain_size,
            blowup,
            columns,
        );
        Ok(None)
    }
}

pub fn concatenate_batch_commitments(
    parts: Vec<PreparedBatchCommitments>,
) -> Result<PreparedBatchCommitments, Error> {
    let mut group_trees = Vec::new();
    let mut commitment_groups = Vec::new();
    #[cfg(feature = "icicle-rust")]
    let mut gpu_lde_column_parts = Vec::new();
    #[cfg(feature = "icicle-rust")]
    let mut all_parts_have_gpu_columns = true;

    for mut part in parts {
        #[cfg(feature = "icicle-rust")]
        {
            if let Some(columns) = part.gpu_lde_columns.take() {
                gpu_lde_column_parts.push(columns);
            } else {
                all_parts_have_gpu_columns = false;
            }
        }
        group_trees.append(&mut part.group_trees);
        commitment_groups.append(&mut part.commitment_groups);
    }
    if group_trees.is_empty() || group_trees.len() != commitment_groups.len() {
        return Err(Error::LowDegreeCheckFailed);
    }
    #[cfg(feature = "icicle-rust")]
    let gpu_lde_columns = if all_parts_have_gpu_columns {
        Some(gpu_batch_merkle_bn254::concatenate_committed_lde_columns(
            gpu_lde_column_parts,
        )?)
    } else {
        None
    };

    Ok(PreparedBatchCommitments {
        group_trees,
        commitment_groups,
        #[cfg(feature = "icicle-rust")]
        gpu_lde_columns,
    })
}

pub fn prepare_batch_low_degree_with_commitments_and_lde<
    'a,
    F: PrimeField + FftField + Send + Sync,
>(
    columns: &[&'a [F]],
    lde_columns: &[&'a [F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    commitments: PreparedBatchCommitments,
) -> Result<PreparedBatchFriProof<'a, F>, Error> {
    let commitment_groups = commitments
        .commitment_groups
        .iter()
        .map(|group| group.columns.clone())
        .collect::<Vec<_>>();
    validate_batch_low_degree_inputs(
        columns,
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
        &commitment_groups,
    )?;

    let (combined, combined_lde) = prepare_combined_batch_fri_with_optional_gpu_resident(
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
    )?;

    prepared_batch_from_parts(
        columns,
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        blowup,
        &commitment_groups,
        commitments.group_trees,
        combined,
        combined_lde,
        None,
        Vec::new(),
    )
}

pub fn prepare_batch_low_degree_with_commitments_and_gpu_lde_store<
    'a,
    F: PrimeField + FftField + Send + Sync,
>(
    columns: &[&'a [F]],
    lde_columns: &[&'a [F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    commitments: PreparedBatchCommitments,
    gpu_lde_store: &'a PreparedGpuLdeColumnStore<F>,
    gpu_column_indices: &[usize],
) -> Result<PreparedBatchFriProof<'a, F>, Error> {
    let commitment_groups = commitments
        .commitment_groups
        .iter()
        .map(|group| group.columns.clone())
        .collect::<Vec<_>>();
    validate_batch_low_degree_inputs_allow_missing_lde(
        columns,
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
        &commitment_groups,
    )?;
    if gpu_column_indices.len() != lde_columns.len() {
        return Err(Error::LowDegreeCheckFailed);
    }

    let result = prove_batch_combined_low_degree_from_gpu_lde_store(
        gpu_lde_store,
        gpu_column_indices,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
        &[],
    )?;
    let (combined, combined_lde) = combined_batch_from_gpu_result(result)?;
    prepared_batch_from_parts(
        columns,
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        blowup,
        &commitment_groups,
        commitments.group_trees,
        combined,
        combined_lde,
        Some(gpu_lde_store),
        gpu_column_indices.to_vec(),
    )
}

pub fn prepare_batch_low_degree_with_groups_and_lde<'a, F: PrimeField + FftField + Send + Sync>(
    columns: &[&'a [F]],
    lde_columns: &[&'a [F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    commitment_groups: &[Vec<usize>],
) -> Result<PreparedBatchFriProof<'a, F>, Error> {
    validate_batch_low_degree_inputs(
        columns,
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
        commitment_groups,
    )?;

    let group_trees = build_batch_group_trees(lde_columns, commitment_groups)?;
    let (combined, combined_lde) = prepare_combined_batch_fri_with_optional_gpu_resident(
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
    )?;

    prepared_batch_from_parts(
        columns,
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        blowup,
        commitment_groups,
        group_trees,
        combined,
        combined_lde,
        None,
        Vec::new(),
    )
}

pub fn try_prepare_batch_low_degree_with_groups_and_lde_gpu_resident<
    'a,
    F: PrimeField + FftField + Send + Sync,
>(
    columns: &[&'a [F]],
    lde_columns: &[&'a [F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    commitment_groups: &[Vec<usize>],
) -> Result<Option<PreparedBatchFriProof<'a, F>>, Error> {
    validate_batch_low_degree_inputs(
        columns,
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
        commitment_groups,
    )?;

    let Some(result) = prove_batch_combined_low_degree_gpu_resident(
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
        &[],
    )?
    else {
        return Ok(None);
    };

    let (combined, combined_lde) = combined_batch_from_gpu_result(result)?;
    let group_trees = build_batch_group_trees(lde_columns, commitment_groups)?;
    prepared_batch_from_parts(
        columns,
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        blowup,
        commitment_groups,
        group_trees,
        combined,
        combined_lde,
        None,
        Vec::new(),
    )
    .map(Some)
}

fn validate_batch_low_degree_inputs<F: FftField>(
    columns: &[&[F]],
    lde_columns: &[&[F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    commitment_groups: &[Vec<usize>],
) -> Result<(), Error> {
    if columns.is_empty()
        || columns.len() != batching_coefficients.len()
        || columns.len() != column_degree_bounds.len()
        || columns.len() != lde_columns.len()
    {
        return Err(Error::LowDegreeCheckFailed);
    }
    if columns
        .iter()
        .any(|column| column.len() != columns[0].len())
    {
        return Err(Error::InvalidDomainSize(columns[0].len()));
    }
    if column_degree_bounds
        .iter()
        .any(|bound| *bound == 0 || *bound > degree_bound)
    {
        return Err(Error::LowDegreeCheckFailed);
    }
    if column_degree_bounds.iter().copied().max().unwrap_or(0) > degree_bound {
        return Err(Error::LowDegreeCheckFailed);
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    let lde_size = columns[0].len() * blowup;
    if lde_columns.iter().any(|column| column.len() != lde_size) {
        return Err(Error::InvalidDomainSize(lde_size));
    }
    validate_commitment_groups(commitment_groups, columns.len())
}

fn validate_batch_low_degree_inputs_allow_missing_lde<F: FftField>(
    columns: &[&[F]],
    lde_columns: &[&[F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    commitment_groups: &[Vec<usize>],
) -> Result<(), Error> {
    if columns.is_empty()
        || columns.len() != batching_coefficients.len()
        || columns.len() != column_degree_bounds.len()
        || columns.len() != lde_columns.len()
    {
        return Err(Error::LowDegreeCheckFailed);
    }
    let row_count = columns
        .iter()
        .find_map(|column| (!column.is_empty()).then_some(column.len()))
        .ok_or(Error::LowDegreeCheckFailed)?;
    if columns
        .iter()
        .any(|column| !column.is_empty() && column.len() != row_count)
    {
        return Err(Error::InvalidDomainSize(row_count));
    }
    if column_degree_bounds
        .iter()
        .any(|bound| *bound == 0 || *bound > degree_bound)
    {
        return Err(Error::LowDegreeCheckFailed);
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    let lde_size = row_count * blowup;
    if lde_columns
        .iter()
        .any(|column| !column.is_empty() && column.len() != lde_size)
    {
        return Err(Error::InvalidDomainSize(lde_size));
    }
    validate_commitment_groups(commitment_groups, columns.len())
}

fn prepare_combined_batch_fri_with_optional_gpu_resident<F: PrimeField + FftField + Send + Sync>(
    lde_columns: &[&[F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
) -> Result<(FriProof<F>, CombinedLdeBackend<F>), Error> {
    if let Some(result) = prove_batch_combined_low_degree_gpu_resident(
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
        blowup,
        &[],
    )? {
        return combined_batch_from_gpu_result(result);
    }

    let (combined_lde, combined_lde_tree) = combine_lde_columns_normalized_with_tree(
        lde_columns,
        batching_coefficients,
        column_degree_bounds,
        degree_bound,
    )?;
    let combined = prove_low_degree_from_lde_with_initial_tree(
        &combined_lde,
        degree_bound,
        blowup,
        combined_lde_tree.clone(),
    )?;
    Ok((
        combined,
        CombinedLdeBackend::cpu(combined_lde, combined_lde_tree),
    ))
}

fn combined_batch_from_gpu_result<F: FftField>(
    result: GpuCombinedFriResult<F>,
) -> Result<(FriProof<F>, CombinedLdeBackend<F>), Error> {
    let GpuCombinedFriResult {
        proof,
        row_openings,
        combined_lde,
    } = result;
    if !row_openings.is_empty() {
        return Err(Error::LowDegreeCheckFailed);
    }
    if proof.root != combined_lde.root() {
        return Err(Error::MerkleRootMismatch);
    }
    Ok((proof, combined_lde))
}

#[allow(clippy::too_many_arguments)]
fn prepared_batch_from_parts<'a, F: FftField + Send>(
    columns: &[&'a [F]],
    lde_columns: &[&'a [F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    blowup: usize,
    commitment_groups: &[Vec<usize>],
    group_trees: Vec<GroupTreeBackend>,
    combined: FriProof<F>,
    combined_lde: CombinedLdeBackend<F>,
    gpu_lde_store: Option<&'a PreparedGpuLdeColumnStore<F>>,
    gpu_column_indices: Vec<usize>,
) -> Result<PreparedBatchFriProof<'a, F>, Error> {
    let groups = commitment_groups
        .iter()
        .zip(&group_trees)
        .map(|(columns, tree)| BatchCommitmentGroup {
            columns: columns.clone(),
            root: tree.root(),
        })
        .collect::<Vec<_>>();
    let row_root = groups.first().ok_or(Error::LowDegreeCheckFailed)?.root;

    let row_count = columns
        .iter()
        .find_map(|column| (!column.is_empty()).then_some(column.len()))
        .unwrap_or_else(|| combined.domain_size / blowup);

    let lazy_gpu_openings = gpu_lde_store.is_some() && combined_lde.prefers_batch_opening();
    let (group_trees, combined_lde) = if lazy_gpu_openings {
        (Vec::new(), None)
    } else {
        (group_trees, Some(combined_lde))
    };

    Ok(PreparedBatchFriProof {
        lde_columns: lde_columns.to_vec(),
        gpu_lde_store,
        gpu_column_indices,
        group_trees,
        combined_lde,
        row_root,
        row_count,
        column_count: columns.len(),
        column_degree_bounds: column_degree_bounds.to_vec(),
        commitment_groups: groups,
        batching_coefficients: batching_coefficients.to_vec(),
        combined,
        blowup,
    })
}

pub fn verify_batch_low_degree<F: PrimeField + FftField + Send + Sync>(
    proof: &BatchFriProof<F>,
    degree_bound: usize,
    blowup: usize,
) -> Result<(), Error> {
    if proof.combined.degree_bound != degree_bound || proof.combined.blowup != blowup {
        return Err(Error::LowDegreeCheckFailed);
    }
    if proof.column_count == 0 || proof.column_count != proof.batching_coefficients.len() {
        return Err(Error::LowDegreeCheckFailed);
    }
    if proof.column_count != proof.column_degree_bounds.len() {
        return Err(Error::LowDegreeCheckFailed);
    }
    validate_commitment_groups(
        &proof
            .commitment_groups
            .iter()
            .map(|group| group.columns.clone())
            .collect::<Vec<_>>(),
        proof.column_count,
    )?;
    if proof.row_root != proof.commitment_groups[0].root {
        return Err(Error::MerkleRootMismatch);
    }
    if proof
        .column_degree_bounds
        .iter()
        .any(|bound| *bound == 0 || *bound > degree_bound)
    {
        return Err(Error::LowDegreeCheckFailed);
    }
    if proof
        .column_degree_bounds
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        > degree_bound
    {
        return Err(Error::LowDegreeCheckFailed);
    }
    verify_low_degree(&proof.combined, proof.combined.root)?;

    for opening in &proof.row_openings {
        if opening.index >= proof.row_count || opening.row.len() != proof.column_count {
            return Err(Error::LowDegreeCheckFailed);
        }
        let lde_index = opening.index * blowup;
        if opening.group_paths.len() != proof.commitment_groups.len() {
            return Err(Error::LowDegreeCheckFailed);
        }
        for (group, path) in proof.commitment_groups.iter().zip(&opening.group_paths) {
            let projected = group
                .columns
                .iter()
                .map(|column| opening.row[*column])
                .collect::<Vec<_>>();
            MerkleTree::verify(group.root, &projected, path)?;
            if path.index != lde_index {
                return Err(Error::LowDegreeCheckFailed);
            }
        }

        let expected_combined = normalized_row_combination(
            proof.row_count * blowup,
            degree_bound,
            lde_index,
            &opening.row,
            &proof.batching_coefficients,
            &proof.column_degree_bounds,
        )?;
        if expected_combined != opening.combined_lde_value {
            return Err(Error::CommitmentMismatch("batch row combination"));
        }
        MerkleTree::verify(
            proof.combined.root,
            &[opening.combined_lde_value],
            &opening.combined_lde_path,
        )?;
        if opening.combined_lde_path.index != lde_index {
            return Err(Error::LowDegreeCheckFailed);
        }
    }

    Ok(())
}

pub fn verify_batch_lde_row_opening<F: PrimeField + FftField>(
    proof: &BatchFriProof<F>,
    opening: &BatchLdeRowOpening<F>,
    degree_bound: usize,
    blowup: usize,
) -> Result<(), Error> {
    if opening.row.len() != proof.column_count
        || opening.group_paths.len() != proof.commitment_groups.len()
    {
        return Err(Error::LowDegreeCheckFailed);
    }
    if opening.lde_index >= proof.row_count * blowup {
        return Err(Error::InvalidDomainSize(opening.lde_index));
    }
    for (group, path) in proof.commitment_groups.iter().zip(&opening.group_paths) {
        let projected = group
            .columns
            .iter()
            .map(|column| opening.row[*column])
            .collect::<Vec<_>>();
        MerkleTree::verify(group.root, &projected, path)?;
        if path.index != opening.lde_index {
            return Err(Error::LowDegreeCheckFailed);
        }
    }

    let expected_combined = normalized_row_combination(
        proof.row_count * blowup,
        degree_bound,
        opening.lde_index,
        &opening.row,
        &proof.batching_coefficients,
        &proof.column_degree_bounds,
    )?;
    if expected_combined != opening.combined_lde_value {
        return Err(Error::CommitmentMismatch("batch lde row combination"));
    }
    MerkleTree::verify(
        proof.combined.root,
        &[opening.combined_lde_value],
        &opening.combined_lde_path,
    )?;
    if opening.combined_lde_path.index != opening.lde_index {
        return Err(Error::LowDegreeCheckFailed);
    }
    Ok(())
}

pub fn fri_batch_verify_out_of_domain<F: PrimeField + FftField + Send + Sync>(
    original_commitment_groups: &[BatchCommitmentGroup],
    claimed_values: &[F],
    quotient_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    zeta: F,
    proof: &OodOpeningProof<F>,
) -> Result<(), Error> {
    let lde_size = degree_bound * blowup;
    if lde_size == 0 || zeta.pow(&[lde_size as u64]) == F::one() {
        return Err(Error::InvalidOutOfDomainPoint);
    }
    if proof.quotient_batch.column_count != claimed_values.len()
        || proof.quotient_batch.column_degree_bounds != quotient_degree_bounds
        || proof.quotient_batch.row_count != degree_bound
    {
        return Err(Error::InvalidOutOfDomainOpening);
    }
    validate_commitment_groups(
        &original_commitment_groups
            .iter()
            .map(|group| group.columns.clone())
            .collect::<Vec<_>>(),
        claimed_values.len(),
    )?;
    verify_batch_low_degree(&proof.quotient_batch, degree_bound, blowup)?;

    let queries = &proof.quotient_batch.combined.queries;
    if proof.row_openings.len() != queries.len() {
        return Err(Error::InvalidOutOfDomainOpening);
    }
    let domain =
        GeneralEvaluationDomain::<F>::new(lde_size).ok_or(Error::InvalidDomainSize(lde_size))?;

    for (query, opening) in queries.iter().zip(&proof.row_openings) {
        if opening.lde_index != query.index || opening.quotient_opening.lde_index != query.index {
            return Err(Error::InvalidOutOfDomainOpening);
        }
        if opening.original_row.len() != claimed_values.len()
            || opening.original_group_paths.len() != original_commitment_groups.len()
        {
            return Err(Error::InvalidOutOfDomainOpening);
        }
        for (group, path) in original_commitment_groups
            .iter()
            .zip(&opening.original_group_paths)
        {
            let projected = group
                .columns
                .iter()
                .map(|column| opening.original_row[*column])
                .collect::<Vec<_>>();
            MerkleTree::verify(group.root, &projected, path)?;
            if path.index != opening.lde_index {
                return Err(Error::InvalidOutOfDomainOpening);
            }
        }
        verify_batch_lde_row_opening(
            &proof.quotient_batch,
            &opening.quotient_opening,
            degree_bound,
            blowup,
        )?;

        let point = domain.element(opening.lde_index);
        if point == zeta {
            return Err(Error::InvalidOutOfDomainPoint);
        }
        for ((original_value, claimed_value), quotient_value) in opening
            .original_row
            .iter()
            .zip(claimed_values)
            .zip(&opening.quotient_opening.row)
        {
            if *original_value - *claimed_value != (point - zeta) * *quotient_value {
                return Err(Error::InvalidOutOfDomainOpening);
            }
        }
    }

    Ok(())
}

fn validate_commitment_groups(groups: &[Vec<usize>], column_count: usize) -> Result<(), Error> {
    if groups.is_empty() {
        return Err(Error::LowDegreeCheckFailed);
    }
    let mut seen = vec![false; column_count];
    for group in groups {
        if group.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        for &column in group {
            if column >= column_count || seen[column] {
                return Err(Error::LowDegreeCheckFailed);
            }
            seen[column] = true;
        }
    }
    if seen.iter().any(|value| !*value) {
        return Err(Error::LowDegreeCheckFailed);
    }
    Ok(())
}

fn build_batch_group_trees_with_committed_columns<F: PrimeField + FftField + Sync>(
    lde_columns: &[&[F]],
    commitment_groups: &[Vec<usize>],
) -> Result<(Vec<GroupTreeBackend>, Option<MaybeGpuCommittedLdeColumns>), Error> {
    let leaf_count = lde_columns[0].len();
    match requested_batch_merkle_backend() {
        Some(BatchMerkleBackendKind::Cpu) => Ok((
            build_batch_group_trees_cpu(lde_columns, commitment_groups),
            None,
        )),
        Some(BatchMerkleBackendKind::Gpu) => {
            let (trees, columns) =
                try_build_batch_group_trees_gpu_with_columns(lde_columns, commitment_groups)?
                    .ok_or(Error::GpuBackendUnavailable)?;
            Ok((trees, Some(columns)))
        }
        None => {
            if should_prefer_gpu_committed_oracle::<F>(leaf_count) {
                if let Ok(Some((trees, columns))) =
                    try_build_batch_group_trees_gpu_with_columns(lde_columns, commitment_groups)
                {
                    return Ok((trees, Some(columns)));
                }
            }
            Ok((
                build_batch_group_trees_cpu(lde_columns, commitment_groups),
                None,
            ))
        }
    }
}

fn build_batch_group_trees<F: PrimeField + FftField + Sync>(
    lde_columns: &[&[F]],
    commitment_groups: &[Vec<usize>],
) -> Result<Vec<GroupTreeBackend>, Error> {
    let leaf_count = lde_columns[0].len();
    match requested_batch_merkle_backend() {
        Some(BatchMerkleBackendKind::Cpu) => {
            Ok(build_batch_group_trees_cpu(lde_columns, commitment_groups))
        }
        Some(BatchMerkleBackendKind::Gpu) => {
            try_build_batch_group_trees_gpu(lde_columns, commitment_groups)?
                .ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if should_prefer_gpu_committed_oracle::<F>(leaf_count) {
                if let Ok(Some(trees)) =
                    try_build_batch_group_trees_gpu(lde_columns, commitment_groups)
                {
                    return Ok(trees);
                }
            }
            Ok(build_batch_group_trees_cpu(lde_columns, commitment_groups))
        }
    }
}

fn build_batch_group_trees_cpu<F: FftField + Sync>(
    lde_columns: &[&[F]],
    commitment_groups: &[Vec<usize>],
) -> Vec<GroupTreeBackend> {
    let leaf_count = lde_columns[0].len();
    if commitment_groups.len() < 2 || leaf_count < CPU_PARALLEL_MIN_ITEMS {
        return commitment_groups
            .iter()
            .map(|group| GroupTreeBackend::cpu(MerkleTree::from_columns(lde_columns, group)))
            .collect::<Vec<_>>();
    }

    let worker_count = parallel_worker_count(commitment_groups.len());
    let chunk_size = (commitment_groups.len() + worker_count - 1) / worker_count;
    let mut chunk_results = thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, chunk) in commitment_groups.chunks(chunk_size).enumerate() {
            let base_index = chunk_index * chunk_size;
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .enumerate()
                    .map(|(offset, group)| {
                        (
                            base_index + offset,
                            GroupTreeBackend::cpu(MerkleTree::from_columns(lde_columns, group)),
                        )
                    })
                    .collect::<Vec<_>>()
            }));
        }

        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| Error::LowDegreeCheckFailed))
            .collect::<Result<Vec<_>, Error>>()
    })
    .expect("parallel batch group Merkle tree construction failed");

    chunk_results.sort_by_key(|chunk| chunk.first().map(|(index, _)| *index).unwrap_or(usize::MAX));
    chunk_results
        .into_iter()
        .flat_map(|chunk| chunk.into_iter().map(|(_, tree)| tree))
        .collect()
}

fn requested_batch_merkle_backend() -> Option<BatchMerkleBackendKind> {
    match env::var(BATCH_MERKLE_BACKEND_ENV)
        .ok()
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("cpu") => Some(BatchMerkleBackendKind::Cpu),
        Some("gpu") => Some(BatchMerkleBackendKind::Gpu),
        Some("auto") | None => None,
        _ => None,
    }
}

fn should_auto_use_gpu_batch_merkle<F: FftField>(leaf_count: usize) -> bool {
    leaf_count >= gpu_batch_merkle_min_leaves() && gpu_batch_merkle_supports_field::<F>()
}

fn should_prefer_gpu_committed_oracle<F: FftField>(leaf_count: usize) -> bool {
    if !gpu_batch_merkle_supports_field::<F>() {
        return false;
    }
    if should_auto_use_gpu_batch_merkle::<F>(leaf_count) {
        return true;
    }
    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Gpu) => return true,
        Some(GpuResidentFriBackendKind::Cpu) => {}
        None => {
            if should_auto_use_gpu_resident_fri::<F>(leaf_count) {
                return true;
            }
        }
    }
    match requested_batch_combine_backend() {
        Some(BatchCombineBackendKind::Gpu) => true,
        Some(BatchCombineBackendKind::Cpu) => false,
        None => should_auto_use_gpu_batch_combine::<F>(leaf_count),
    }
}

fn gpu_batch_merkle_min_leaves() -> usize {
    env::var(GPU_BATCH_MERKLE_MIN_LEAVES_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_GPU_BATCH_MERKLE_MIN_LEAVES)
}

fn gpu_batch_merkle_supports_field<F: FftField>() -> bool {
    #[cfg(feature = "icicle-rust")]
    {
        gpu_batch_merkle_bn254::supports_field::<F>()
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        false
    }
}

fn try_build_batch_group_trees_gpu_with_columns<F: PrimeField + FftField>(
    lde_columns: &[&[F]],
    commitment_groups: &[Vec<usize>],
) -> Result<Option<(Vec<GroupTreeBackend>, MaybeGpuCommittedLdeColumns)>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_batch_merkle_bn254::try_build_group_trees_with_columns(
                lde_columns,
                commitment_groups,
            )
            .map(|prepared| {
                prepared.map(|(trees, columns)| {
                    (
                        trees
                            .into_iter()
                            .map(GroupTreeBackend::gpu)
                            .collect::<Vec<_>>(),
                        columns,
                    )
                })
            })
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (lde_columns, commitment_groups);
        Ok(None)
    }
}

fn try_build_batch_group_trees_gpu<F: PrimeField + FftField>(
    lde_columns: &[&[F]],
    commitment_groups: &[Vec<usize>],
) -> Result<Option<Vec<GroupTreeBackend>>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_batch_merkle_bn254::try_build_group_trees(lde_columns, commitment_groups).map(
                |trees| {
                    trees.map(|trees| {
                        trees
                            .into_iter()
                            .map(GroupTreeBackend::gpu)
                            .collect::<Vec<_>>()
                    })
                },
            )
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (lde_columns, commitment_groups);
        Ok(None)
    }
}

pub fn combine_columns<F: FftField>(columns: &[Vec<F>], coefficients: &[F]) -> Vec<F> {
    let width = columns[0].len();
    let mut combined = vec![F::zero(); width];
    for (column, coefficient) in columns.iter().zip(coefficients) {
        for (target, value) in combined.iter_mut().zip(column.iter()) {
            *target += *coefficient * value;
        }
    }
    combined
}

pub fn combine_columns_normalized<F: FftField>(
    columns: &[Vec<F>],
    coefficients: &[F],
    degree_bounds: &[usize],
) -> Result<Vec<F>, Error> {
    let degree_bound = degree_bounds
        .iter()
        .copied()
        .max()
        .ok_or(Error::LowDegreeCheckFailed)?;
    let width = columns[0].len();
    (0..width)
        .map(|index| {
            let row = columns
                .iter()
                .map(|column| column[index])
                .collect::<Vec<_>>();
            normalized_row_combination(
                width,
                degree_bound,
                index,
                &row,
                coefficients,
                degree_bounds,
            )
        })
        .collect()
}

fn prove_batch_combined_low_degree_gpu_resident<F: PrimeField + FftField>(
    lde_columns: &[&[F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    opened_indices: &[usize],
) -> Result<Option<GpuCombinedFriResult<F>>, Error> {
    let lde_size = lde_columns[0].len();
    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Cpu) => Ok(None),
        Some(GpuResidentFriBackendKind::Gpu) => try_prove_batch_combined_low_degree_gpu_resident(
            lde_columns,
            batching_coefficients,
            column_degree_bounds,
            degree_bound,
            blowup,
            opened_indices,
        )?
        .map(Some)
        .ok_or(Error::GpuBackendUnavailable),
        None => {
            let gpu_batch_combine_would_run = match requested_batch_combine_backend() {
                Some(BatchCombineBackendKind::Gpu) => true,
                Some(BatchCombineBackendKind::Cpu) => false,
                None => should_auto_use_gpu_batch_combine::<F>(lde_size),
            };
            if (should_auto_use_gpu_resident_fri::<F>(lde_size) || gpu_batch_combine_would_run)
                && gpu_resident_fri_supports_field::<F>()
            {
                if let Ok(Some(result)) = try_prove_batch_combined_low_degree_gpu_resident(
                    lde_columns,
                    batching_coefficients,
                    column_degree_bounds,
                    degree_bound,
                    blowup,
                    opened_indices,
                ) {
                    return Ok(Some(result));
                }
            }
            Ok(None)
        }
    }
}

pub fn should_try_gpu_resident_batch_fri<F: FftField>(lde_size: usize) -> bool {
    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Cpu) => false,
        Some(GpuResidentFriBackendKind::Gpu) => gpu_resident_fri_supports_field::<F>(),
        None => should_auto_use_gpu_resident_fri::<F>(lde_size),
    }
}

pub fn try_prepare_gpu_lde_column_store<F: PrimeField + FftField + Send + Sync>(
    lde_columns: &[&[F]],
    degree_bound: usize,
    blowup: usize,
) -> Result<Option<PreparedGpuLdeColumnStore<F>>, Error> {
    if lde_columns.is_empty() {
        return Err(Error::LowDegreeCheckFailed);
    }
    if degree_bound == 0 || !is_power_of_two(degree_bound) {
        return Err(Error::InvalidDomainSize(degree_bound));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    let lde_size = degree_bound * blowup;
    if lde_columns.iter().any(|column| column.len() != lde_size) {
        return Err(Error::InvalidDomainSize(lde_size));
    }

    let prepare = || {
        #[cfg(feature = "icicle-rust")]
        {
            return crate::gpu_lock::with_gpu_lock(|| {
                gpu_resident_fri_bn254::try_prepare_lde_column_store(
                    lde_columns,
                    degree_bound,
                    blowup,
                )
                .map(|prepared| prepared.map(|inner| PreparedGpuLdeColumnStore { inner }))
            });
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            let _ = (lde_columns, degree_bound, blowup);
            Ok(None)
        }
    };

    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Cpu) => Ok(None),
        Some(GpuResidentFriBackendKind::Gpu) => {
            prepare()?.map(Some).ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if !should_auto_use_gpu_resident_fri::<F>(lde_size) {
                return Ok(None);
            }
            match prepare() {
                Ok(prepared) => Ok(prepared),
                Err(Error::GpuBackendFailed | Error::GpuBackendUnavailable) => Ok(None),
                Err(error) => Err(error),
            }
        }
    }
}

pub fn try_prepare_gpu_lde_column_store_from_commitments<F: PrimeField + FftField + Send + Sync>(
    prefix_lde_columns: &[&[F]],
    commitments: &PreparedBatchCommitments,
    degree_bound: usize,
    blowup: usize,
) -> Result<Option<PreparedGpuLdeColumnStore<F>>, Error> {
    if degree_bound == 0 || !is_power_of_two(degree_bound) {
        return Err(Error::InvalidDomainSize(degree_bound));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    let lde_size = degree_bound * blowup;
    if prefix_lde_columns
        .iter()
        .any(|column| column.len() != lde_size)
    {
        return Err(Error::InvalidDomainSize(lde_size));
    }

    #[cfg(feature = "icicle-rust")]
    let prepare = || {
        let Some(committed_columns) = commitments.gpu_lde_columns.as_ref() else {
            return Ok(None);
        };
        crate::gpu_lock::with_gpu_lock(|| {
            gpu_resident_fri_bn254::try_prepare_lde_column_store_from_committed_columns(
                prefix_lde_columns,
                committed_columns,
                degree_bound,
                blowup,
            )
            .map(|prepared| prepared.map(|inner| PreparedGpuLdeColumnStore { inner }))
        })
    };

    #[cfg(not(feature = "icicle-rust"))]
    let prepare = || {
        let _ = (prefix_lde_columns, commitments, degree_bound, blowup);
        Ok(None)
    };

    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Cpu) => Ok(None),
        Some(GpuResidentFriBackendKind::Gpu) => {
            prepare()?.map(Some).ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if !should_auto_use_gpu_resident_fri::<F>(lde_size) {
                return Ok(None);
            }
            match prepare() {
                Ok(prepared) => Ok(prepared),
                Err(Error::GpuBackendFailed | Error::GpuBackendUnavailable) => Ok(None),
                Err(error) => Err(error),
            }
        }
    }
}

fn prove_batch_combined_low_degree_from_gpu_lde_store<F: PrimeField + FftField + Send + Sync>(
    gpu_lde_store: &PreparedGpuLdeColumnStore<F>,
    column_indices: &[usize],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    opened_indices: &[usize],
) -> Result<GpuCombinedFriResult<F>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_resident_fri_bn254::try_prove_batch_combined_low_degree_from_store(
                &gpu_lde_store.inner,
                column_indices,
                batching_coefficients,
                column_degree_bounds,
                degree_bound,
                blowup,
                opened_indices,
            )
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (
            gpu_lde_store,
            column_indices,
            batching_coefficients,
            column_degree_bounds,
            degree_bound,
            blowup,
            opened_indices,
        );
        Err(Error::GpuBackendUnavailable)
    }
}

#[cfg(feature = "icicle-rust")]
pub(crate) fn try_prepare_gpu_lde_column_store_from_table_lde_and_commitments<
    F: PrimeField + FftField + Send + Sync,
>(
    table_lde: &PreparedGpuTableLde<F>,
    commitments: &PreparedBatchCommitments,
    degree_bound: usize,
    blowup: usize,
) -> Result<Option<PreparedGpuLdeColumnStore<F>>, Error> {
    if matches!(
        requested_gpu_resident_fri_backend(),
        Some(GpuResidentFriBackendKind::Cpu)
    ) {
        return Ok(None);
    }
    if degree_bound == 0 || !is_power_of_two(degree_bound) {
        return Err(Error::InvalidDomainSize(degree_bound));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    let Some(committed_columns) = commitments.gpu_lde_columns.as_ref() else {
        return Ok(None);
    };
    crate::gpu_lock::with_gpu_lock(|| {
        gpu_resident_fri_bn254::try_prepare_lde_column_store_from_table_and_committed_columns(
            &table_lde.inner,
            committed_columns,
            degree_bound,
            blowup,
        )
        .map(|prepared| prepared.map(|inner| PreparedGpuLdeColumnStore { inner }))
    })
}

pub fn prepare_ood_opening_batch_from_gpu_lde_store<F: PrimeField + FftField + Send + Sync>(
    gpu_lde_store: &PreparedGpuLdeColumnStore<F>,
    column_indices: &[usize],
    claimed_values: &[F],
    zeta: F,
    degree_bound: usize,
    blowup: usize,
    column_degree_bounds: &[usize],
) -> Result<PreparedGpuOodOpeningBatch<F>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_resident_fri_bn254::try_prepare_ood_quotient_batch_from_store(
                &gpu_lde_store.inner,
                column_indices,
                claimed_values,
                zeta,
                degree_bound,
                blowup,
                column_degree_bounds,
            )
            .map(|inner| PreparedGpuOodOpeningBatch { inner })
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (
            gpu_lde_store,
            column_indices,
            claimed_values,
            zeta,
            degree_bound,
            blowup,
            column_degree_bounds,
        );
        Err(Error::GpuBackendUnavailable)
    }
}

pub fn prepare_specialized_usc_batch_from_gpu_lde_store<F: PrimeField + FftField + Send + Sync>(
    gpu_lde_store: &PreparedGpuLdeColumnStore<F>,
    column_index: usize,
    claimed_sum: F,
    degree_bound: usize,
    blowup: usize,
    p_degree_bound: usize,
) -> Result<PreparedGpuSpecializedUscBatch<F>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_resident_fri_bn254::try_prepare_specialized_usc_batch_from_store(
                &gpu_lde_store.inner,
                column_index,
                claimed_sum,
                degree_bound,
                blowup,
                p_degree_bound,
            )
            .map(|inner| PreparedGpuSpecializedUscBatch { inner })
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (
            gpu_lde_store,
            column_index,
            claimed_sum,
            degree_bound,
            blowup,
            p_degree_bound,
        );
        Err(Error::GpuBackendUnavailable)
    }
}

pub fn try_prepare_ood_opening_batch_gpu_resident<F: PrimeField + FftField + Send + Sync>(
    original_lde_columns: &[&[F]],
    claimed_values: &[F],
    zeta: F,
    degree_bound: usize,
    blowup: usize,
    column_degree_bounds: &[usize],
) -> Result<Option<PreparedGpuOodOpeningBatch<F>>, Error> {
    if degree_bound == 0 || !is_power_of_two(degree_bound) {
        return Err(Error::InvalidDomainSize(degree_bound));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    let lde_size = degree_bound * blowup;

    let prepare = || {
        #[cfg(feature = "icicle-rust")]
        {
            return crate::gpu_lock::with_gpu_lock(|| {
                gpu_resident_fri_bn254::try_prepare_ood_quotient_batch(
                    original_lde_columns,
                    claimed_values,
                    zeta,
                    degree_bound,
                    blowup,
                    column_degree_bounds,
                )
                .map(|prepared| prepared.map(|inner| PreparedGpuOodOpeningBatch { inner }))
            });
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            let _ = (
                original_lde_columns,
                claimed_values,
                zeta,
                degree_bound,
                blowup,
                column_degree_bounds,
            );
            Ok(None)
        }
    };

    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Cpu) => Ok(None),
        Some(GpuResidentFriBackendKind::Gpu) => {
            prepare()?.map(Some).ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if !should_auto_use_gpu_resident_fri::<F>(lde_size) {
                return Ok(None);
            }
            match prepare() {
                Ok(prepared) => Ok(prepared),
                Err(Error::GpuBackendFailed | Error::GpuBackendUnavailable) => Ok(None),
                Err(error) => Err(error),
            }
        }
    }
}

pub fn try_prepare_specialized_usc_batch_gpu_resident<F: PrimeField + FftField + Send + Sync>(
    existing_lde_column: &[F],
    claimed_sum: F,
    degree_bound: usize,
    blowup: usize,
    p_degree_bound: usize,
) -> Result<Option<PreparedGpuSpecializedUscBatch<F>>, Error> {
    if degree_bound == 0 || !is_power_of_two(degree_bound) {
        return Err(Error::InvalidDomainSize(degree_bound));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    if p_degree_bound == 0 || p_degree_bound > degree_bound {
        return Err(Error::LowDegreeCheckFailed);
    }
    let lde_size = degree_bound * blowup;
    if existing_lde_column.len() != lde_size {
        return Err(Error::InvalidDomainSize(existing_lde_column.len()));
    }

    let prepare = || {
        #[cfg(feature = "icicle-rust")]
        {
            return crate::gpu_lock::with_gpu_lock(|| {
                gpu_resident_fri_bn254::try_prepare_specialized_usc_batch(
                    existing_lde_column,
                    claimed_sum,
                    degree_bound,
                    blowup,
                    p_degree_bound,
                )
                .map(|prepared| prepared.map(|inner| PreparedGpuSpecializedUscBatch { inner }))
            });
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            let _ = (
                existing_lde_column,
                claimed_sum,
                degree_bound,
                blowup,
                p_degree_bound,
            );
            Ok(None)
        }
    };

    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Cpu) => Ok(None),
        Some(GpuResidentFriBackendKind::Gpu) => {
            prepare()?.map(Some).ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if !should_auto_use_gpu_resident_fri::<F>(lde_size) {
                return Ok(None);
            }
            match prepare() {
                Ok(prepared) => Ok(prepared),
                Err(Error::GpuBackendFailed | Error::GpuBackendUnavailable) => Ok(None),
                Err(error) => Err(error),
            }
        }
    }
}

fn requested_gpu_resident_fri_backend() -> Option<GpuResidentFriBackendKind> {
    match env::var(GPU_RESIDENT_FRI_BACKEND_ENV)
        .ok()
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("cpu") => Some(GpuResidentFriBackendKind::Cpu),
        Some("gpu") => Some(GpuResidentFriBackendKind::Gpu),
        Some("auto") | None => None,
        _ => None,
    }
}

fn should_auto_use_gpu_resident_fri<F: FftField>(lde_size: usize) -> bool {
    lde_size >= gpu_resident_fri_min_size() && gpu_resident_fri_supports_field::<F>()
}

fn gpu_resident_fri_min_size() -> usize {
    env::var(GPU_RESIDENT_FRI_MIN_SIZE_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_GPU_RESIDENT_FRI_MIN_SIZE)
}

fn gpu_resident_fri_supports_field<F: FftField>() -> bool {
    #[cfg(feature = "icicle-rust")]
    {
        gpu_resident_fri_bn254::supports_field::<F>()
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        false
    }
}

fn try_prove_batch_combined_low_degree_gpu_resident<F: PrimeField + FftField>(
    lde_columns: &[&[F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    blowup: usize,
    opened_indices: &[usize],
) -> Result<Option<GpuCombinedFriResult<F>>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_resident_fri_bn254::try_prove_batch_combined_low_degree(
                lde_columns,
                batching_coefficients,
                column_degree_bounds,
                degree_bound,
                blowup,
                opened_indices,
            )
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (
            lde_columns,
            batching_coefficients,
            column_degree_bounds,
            degree_bound,
            blowup,
            opened_indices,
        );
        Ok(None)
    }
}

pub(crate) fn try_prepare_table_low_degree_gpu_resident<F: PrimeField + FftField + Send + Sync>(
    table_values: &[F],
    degree_bound: usize,
    blowup: usize,
) -> Result<Option<PreparedTableLowDegree<F>>, Error> {
    if table_values.len() != degree_bound {
        return Err(Error::InvalidDomainSize(table_values.len()));
    }
    let plan = lde::DenseLdePlan::<F>::new(degree_bound, blowup)?;
    if lde::select_dense_lde_backend::<F>(&plan) != lde::DenseLdeBackendKind::Gpu {
        return Ok(None);
    }

    let lde_size = plan.lde_size;
    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Cpu) => Ok(None),
        Some(GpuResidentFriBackendKind::Gpu) => {
            try_prepare_table_low_degree_gpu_resident_inner(table_values, degree_bound, blowup)?
                .map(Some)
                .ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if requested_fold_backend() == Some(FriFoldBackendKind::Cpu) {
                return Ok(None);
            }
            let gpu_fold_would_run = match requested_fold_backend() {
                Some(FriFoldBackendKind::Gpu) => true,
                Some(FriFoldBackendKind::Cpu) => false,
                None => should_auto_use_gpu_fold::<F>(lde_size),
            };
            if (should_auto_use_gpu_resident_fri::<F>(lde_size) || gpu_fold_would_run)
                && gpu_resident_fri_supports_field::<F>()
            {
                if let Ok(Some(prepared)) = try_prepare_table_low_degree_gpu_resident_inner(
                    table_values,
                    degree_bound,
                    blowup,
                ) {
                    return Ok(Some(prepared));
                }
            }
            Ok(None)
        }
    }
}

fn try_prepare_table_low_degree_gpu_resident_inner<F: PrimeField + FftField>(
    table_values: &[F],
    degree_bound: usize,
    blowup: usize,
) -> Result<Option<PreparedTableLowDegree<F>>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_resident_fri_bn254::try_prepare_table_low_degree(table_values, degree_bound, blowup)
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (table_values, degree_bound, blowup);
        Ok(None)
    }
}

pub(crate) fn try_prepare_low_degree_from_coeffs_gpu_resident<
    F: PrimeField + FftField + Send + Sync,
>(
    coeffs: &[F],
    degree_bound: usize,
    blowup: usize,
) -> Result<Option<PreparedLowDegreeFromCoeffs<F>>, Error> {
    if coeffs.len() > degree_bound {
        return Err(Error::LowDegreeCheckFailed);
    }
    let plan = lde::DenseLdePlan::<F>::new(degree_bound, blowup)?;
    if lde::select_dense_lde_backend::<F>(&plan) != lde::DenseLdeBackendKind::Gpu {
        return Ok(None);
    }

    let lde_size = plan.lde_size;
    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Cpu) => Ok(None),
        Some(GpuResidentFriBackendKind::Gpu) => {
            try_prepare_low_degree_from_coeffs_gpu_resident_inner(coeffs, degree_bound, blowup)?
                .map(Some)
                .ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if requested_fold_backend() == Some(FriFoldBackendKind::Cpu) {
                return Ok(None);
            }
            let gpu_fold_would_run = match requested_fold_backend() {
                Some(FriFoldBackendKind::Gpu) => true,
                Some(FriFoldBackendKind::Cpu) => false,
                None => should_auto_use_gpu_fold::<F>(lde_size),
            };
            if (should_auto_use_gpu_resident_fri::<F>(lde_size) || gpu_fold_would_run)
                && gpu_resident_fri_supports_field::<F>()
            {
                if let Ok(Some(prepared)) = try_prepare_low_degree_from_coeffs_gpu_resident_inner(
                    coeffs,
                    degree_bound,
                    blowup,
                ) {
                    return Ok(Some(prepared));
                }
            }
            Ok(None)
        }
    }
}

fn try_prepare_low_degree_from_coeffs_gpu_resident_inner<F: PrimeField + FftField>(
    coeffs: &[F],
    degree_bound: usize,
    blowup: usize,
) -> Result<Option<PreparedLowDegreeFromCoeffs<F>>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_resident_fri_bn254::try_prepare_low_degree_from_coeffs(coeffs, degree_bound, blowup)
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (coeffs, degree_bound, blowup);
        Ok(None)
    }
}

fn try_prove_low_degree_from_lde_gpu_resident<F: PrimeField + FftField>(
    lde_evals: &[F],
    degree_bound: usize,
    blowup: usize,
) -> Result<Option<FriProof<F>>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_resident_fri_bn254::try_prove_low_degree_from_lde(lde_evals, degree_bound, blowup)
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (lde_evals, degree_bound, blowup);
        Ok(None)
    }
}

fn combine_lde_columns_normalized_with_tree<F: PrimeField + FftField + Send + Sync>(
    lde_columns: &[&[F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
) -> Result<(Vec<F>, MerkleTree), Error> {
    let lde_size = lde_columns[0].len();
    match requested_batch_combine_backend() {
        Some(BatchCombineBackendKind::Cpu) => combine_lde_columns_normalized_cpu_with_tree(
            lde_columns,
            batching_coefficients,
            column_degree_bounds,
            degree_bound,
        ),
        Some(BatchCombineBackendKind::Gpu) => {
            let combined = try_combine_lde_columns_gpu(
                lde_columns,
                batching_coefficients,
                column_degree_bounds,
                degree_bound,
            )?
            .ok_or(Error::GpuBackendUnavailable)?;
            let tree = MerkleTree::from_column(&combined);
            Ok((combined, tree))
        }
        None => {
            if should_auto_use_gpu_batch_combine::<F>(lde_size) {
                if let Ok(Some(combined)) = try_combine_lde_columns_gpu(
                    lde_columns,
                    batching_coefficients,
                    column_degree_bounds,
                    degree_bound,
                ) {
                    let tree = MerkleTree::from_column(&combined);
                    return Ok((combined, tree));
                }
            }
            combine_lde_columns_normalized_cpu_with_tree(
                lde_columns,
                batching_coefficients,
                column_degree_bounds,
                degree_bound,
            )
        }
    }
}

fn combine_lde_columns_normalized_cpu_with_tree<F: FftField + Send + Sync>(
    lde_columns: &[&[F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
) -> Result<(Vec<F>, MerkleTree), Error> {
    let lde_size = lde_columns[0].len();
    let domain =
        GeneralEvaluationDomain::<F>::new(lde_size).ok_or(Error::InvalidDomainSize(lde_size))?;
    let root = domain.element(1);
    let worker_count = cpu_parallel_worker_count(lde_size);
    if worker_count <= 1 {
        let (combined, leaf_digests) = combine_lde_columns_normalized_cpu_range_with_leaf_digests(
            lde_columns,
            batching_coefficients,
            column_degree_bounds,
            degree_bound,
            root,
            0,
            lde_size,
        );
        return Ok((combined, MerkleTree::from_leaf_digests(leaf_digests)));
    }

    let chunk_size = (lde_size + worker_count - 1) / worker_count;
    let mut chunk_results = thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, start) in (0..lde_size).step_by(chunk_size).enumerate() {
            let end = (start + chunk_size).min(lde_size);
            handles.push(scope.spawn(move || {
                let (combined, leaf_digests) =
                    combine_lde_columns_normalized_cpu_range_with_leaf_digests(
                        lde_columns,
                        batching_coefficients,
                        column_degree_bounds,
                        degree_bound,
                        root,
                        start,
                        end,
                    );
                (chunk_index, combined, leaf_digests)
            }));
        }

        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| Error::LowDegreeCheckFailed))
            .collect::<Result<Vec<_>, Error>>()
    })?;

    chunk_results.sort_by_key(|(chunk_index, _, _)| *chunk_index);
    let mut combined = Vec::with_capacity(lde_size);
    let mut leaf_digests = Vec::with_capacity(lde_size);
    for (_, chunk_combined, chunk_leaf_digests) in chunk_results {
        combined.extend(chunk_combined);
        leaf_digests.extend(chunk_leaf_digests);
    }

    Ok((combined, MerkleTree::from_leaf_digests(leaf_digests)))
}

struct CombineColumnState<F: FftField> {
    coefficient: F,
    step: F,
    power: F,
    constant_weight: bool,
}

fn combine_lde_columns_normalized_cpu_range_with_leaf_digests<F: FftField>(
    lde_columns: &[&[F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
    root: F,
    start: usize,
    end: usize,
) -> (Vec<F>, Vec<Digest32>) {
    let mut states = batching_coefficients
        .iter()
        .zip(column_degree_bounds)
        .map(|(coefficient, column_bound)| {
            let exponent = degree_bound - *column_bound;
            if exponent == 0 {
                CombineColumnState {
                    coefficient: *coefficient,
                    step: F::one(),
                    power: F::one(),
                    constant_weight: true,
                }
            } else {
                let step = root.pow(&[exponent as u64]);
                CombineColumnState {
                    coefficient: *coefficient,
                    step,
                    power: step.pow(&[start as u64]),
                    constant_weight: false,
                }
            }
        })
        .collect::<Vec<_>>();

    let mut combined = Vec::with_capacity(end - start);
    let mut leaf_digests = Vec::with_capacity(end - start);
    for index in start..end {
        let mut combined_value = F::zero();
        for (column, state) in lde_columns.iter().zip(&mut states) {
            let weight = if state.constant_weight {
                state.coefficient
            } else {
                let weight = state.coefficient * state.power;
                state.power *= state.step;
                weight
            };
            combined_value += weight * column[index];
        }
        leaf_digests.push(row_to_leaf(std::slice::from_ref(&combined_value)));
        combined.push(combined_value);
    }

    (combined, leaf_digests)
}

fn requested_batch_combine_backend() -> Option<BatchCombineBackendKind> {
    match env::var(BATCH_COMBINE_BACKEND_ENV)
        .ok()
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("cpu") => Some(BatchCombineBackendKind::Cpu),
        Some("gpu") => Some(BatchCombineBackendKind::Gpu),
        Some("auto") | None => None,
        _ => None,
    }
}

fn should_auto_use_gpu_batch_combine<F: FftField>(lde_size: usize) -> bool {
    lde_size >= gpu_batch_combine_min_size() && gpu_batch_combine_supports_field::<F>()
}

fn gpu_batch_combine_min_size() -> usize {
    env::var(GPU_BATCH_COMBINE_MIN_SIZE_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_GPU_BATCH_COMBINE_MIN_SIZE)
}

fn gpu_batch_combine_supports_field<F: FftField>() -> bool {
    #[cfg(feature = "icicle-rust")]
    {
        gpu_batch_combine_bn254::supports_field::<F>()
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        false
    }
}

fn try_combine_lde_columns_gpu<F: PrimeField + FftField>(
    lde_columns: &[&[F]],
    batching_coefficients: &[F],
    column_degree_bounds: &[usize],
    degree_bound: usize,
) -> Result<Option<Vec<F>>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_batch_combine_bn254::try_combine_lde_columns(
                lde_columns,
                batching_coefficients,
                column_degree_bounds,
                degree_bound,
            )
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (
            lde_columns,
            batching_coefficients,
            column_degree_bounds,
            degree_bound,
        );
        Ok(None)
    }
}

pub fn normalized_row_combination<F: FftField>(
    domain_size: usize,
    degree_bound: usize,
    index: usize,
    row: &[F],
    coefficients: &[F],
    degree_bounds: &[usize],
) -> Result<F, Error> {
    if row.len() != coefficients.len() || row.len() != degree_bounds.len() {
        return Err(Error::LowDegreeCheckFailed);
    }
    let domain = GeneralEvaluationDomain::<F>::new(domain_size)
        .ok_or(Error::InvalidDomainSize(domain_size))?;
    let point = domain.element(index);
    Ok(row.iter().zip(coefficients).zip(degree_bounds).fold(
        F::zero(),
        |acc, ((value, coefficient), column_bound)| {
            let exponent = degree_bound - *column_bound;
            acc + (*coefficient * point.pow(&[exponent as u64]) * *value)
        },
    ))
}

pub fn transpose<F: Copy>(columns: &[&[F]]) -> Vec<Vec<F>> {
    let height = columns[0].len();
    (0..height)
        .map(|row| columns.iter().map(|column| column[row]).collect())
        .collect()
}

fn sample_queries<F: PrimeField + FftField + Send + Sync>(
    transcript: &mut Transcript,
    layers: &[Vec<F>],
    trees: &[FriTreeBackend],
) -> Result<Vec<FriQuery<F>>, Error> {
    let indices = (0..query_count(layers[0].len()))
        .map(|_| transcript.challenge_index(b"fri-query", layers[0].len()))
        .collect::<Vec<_>>();
    build_query_openings_batched(&indices, layers, trees)
}

fn build_query_openings_batched<F: PrimeField + FftField>(
    indices: &[usize],
    layers: &[Vec<F>],
    trees: &[FriTreeBackend],
) -> Result<Vec<FriQuery<F>>, Error> {
    if layers.is_empty() {
        return Err(Error::LowDegreeCheckFailed);
    }

    let round_count = layers.len() - 1;
    if trees.len() < round_count {
        return Err(Error::LowDegreeCheckFailed);
    }
    let query_count = indices.len();
    let mut openings = (0..query_count)
        .map(|_| Vec::with_capacity(round_count))
        .collect::<Vec<_>>();
    let mut current_indices = indices.to_vec();

    for (layer, tree) in layers.iter().zip(trees).take(round_count) {
        if layer.len() < 2 || layer.len() % 2 != 0 {
            return Err(Error::InvalidDomainSize(layer.len()));
        }

        let half = layer.len() / 2;
        let mut sibling_indices = Vec::with_capacity(query_count);
        let mut next_indices = Vec::with_capacity(query_count);
        for &current_index in &current_indices {
            if current_index >= layer.len() {
                return Err(Error::InvalidDomainSize(current_index));
            }
            sibling_indices.push(if current_index < half {
                current_index + half
            } else {
                current_index - half
            });
            next_indices.push(current_index % half);
        }

        let mut path_indices = Vec::with_capacity(query_count * 2);
        path_indices.extend_from_slice(&current_indices);
        path_indices.extend_from_slice(&sibling_indices);
        let mut paths = tree.open_many(&path_indices)?;
        if paths.len() != query_count * 2 {
            return Err(Error::LowDegreeCheckFailed);
        }
        let sibling_paths = paths.split_off(query_count);

        for position in 0..query_count {
            let current_index = current_indices[position];
            let sibling_index = sibling_indices[position];
            openings[position].push(FriLayerOpening {
                index: current_index,
                value: layer[current_index],
                sibling_value: layer[sibling_index],
                path: paths[position].clone(),
                sibling_path: truncate_top_sibling(sibling_paths[position].clone())?,
            });
        }

        current_indices = next_indices;
    }

    Ok(indices
        .iter()
        .copied()
        .zip(openings)
        .map(|(index, openings)| FriQuery { index, openings })
        .collect())
}

fn truncate_top_sibling(mut path: MerklePath) -> Result<MerklePath, Error> {
    if path.siblings.pop().is_none() {
        return Err(Error::LowDegreeCheckFailed);
    }
    Ok(path)
}

fn verify_queries_parallel<F: PrimeField + FftField + Send + Sync>(
    proof: &FriProof<F>,
    alphas: &[F],
) -> Result<(), Error> {
    let worker_count = parallel_worker_count(proof.queries.len());
    if worker_count <= 1 || proof.queries.len() < 2 {
        for query in &proof.queries {
            verify_query(proof, query, alphas)?;
        }
        return Ok(());
    }

    let chunk_size = (proof.queries.len() + worker_count - 1) / worker_count;
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in proof.queries.chunks(chunk_size) {
            handles.push(scope.spawn(move || {
                for query in chunk {
                    verify_query(proof, query, alphas)?;
                }
                Ok(())
            }));
        }

        handles.into_iter().try_for_each(|handle| {
            handle.join().map_err(|_| Error::LowDegreeCheckFailed)??;
            Ok(())
        })
    })
}

fn parallel_worker_count(item_count: usize) -> usize {
    if item_count == 0 {
        return 1;
    }
    thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(item_count)
        .max(1)
}

fn cpu_parallel_worker_count(item_count: usize) -> usize {
    if item_count < CPU_PARALLEL_MIN_ITEMS {
        return 1;
    }
    parallel_worker_count(item_count)
}

fn verify_query<F: PrimeField + FftField>(
    proof: &FriProof<F>,
    query: &FriQuery<F>,
    alphas: &[F],
) -> Result<(), Error> {
    if query.openings.len() != alphas.len() {
        return Err(Error::LowDegreeCheckFailed);
    }

    let mut current_index = query.index;
    let mut current_len = proof.domain_size;

    for (round, ((opening, alpha), root)) in query
        .openings
        .iter()
        .zip(alphas)
        .zip(&proof.roots)
        .enumerate()
    {
        if opening.index != current_index || opening.index >= current_len {
            return Err(Error::LowDegreeCheckFailed);
        }

        let half = current_len / 2;
        let pair_index = current_index % half;
        let sibling_index = if current_index < half {
            current_index + half
        } else {
            current_index - half
        };

        MerkleTree::verify_pair(
            *root,
            &[opening.value],
            &[opening.sibling_value],
            &opening.path,
            &opening.sibling_path,
        )?;
        if opening.path.index != current_index || opening.sibling_path.index != sibling_index {
            return Err(Error::LowDegreeCheckFailed);
        }

        let (left, right) = if current_index < half {
            (opening.value, opening.sibling_value)
        } else {
            (opening.sibling_value, opening.value)
        };
        let folded = fold_pair(current_len, pair_index, left, right, *alpha)?;

        current_index = pair_index;
        current_len = half;

        let next_value = query
            .openings
            .get(round + 1)
            .map(|next| next.value)
            .unwrap_or_else(|| proof.final_evals[current_index]);
        if folded != next_value {
            return Err(Error::LowDegreeCheckFailed);
        }
    }

    if proof.final_evals[current_index] != proof.final_evals[0] {
        return Err(Error::LowDegreeCheckFailed);
    }
    Ok(())
}

pub fn warm_up_gpu_batch_merkle<F: PrimeField + FftField>(
    leaf_count: usize,
    max_group_width: usize,
) -> Result<(), Error> {
    if leaf_count == 0 || max_group_width == 0 {
        return Ok(());
    }
    if !is_power_of_two(leaf_count) {
        return Err(Error::InvalidDomainSize(leaf_count));
    }

    match requested_batch_merkle_backend() {
        Some(BatchMerkleBackendKind::Cpu) => Ok(()),
        Some(BatchMerkleBackendKind::Gpu) => {
            let (columns, groups) = batch_merkle_warmup_inputs::<F>(leaf_count, max_group_width);
            let column_refs = column_slices(&columns);
            try_build_batch_group_trees_gpu(&column_refs, &groups)?
                .map(|_| ())
                .ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if should_auto_use_gpu_batch_merkle::<F>(leaf_count) {
                let (columns, groups) =
                    batch_merkle_warmup_inputs::<F>(leaf_count, max_group_width);
                let column_refs = column_slices(&columns);
                let _ = try_build_batch_group_trees_gpu(&column_refs, &groups);
            }
            Ok(())
        }
    }
}

fn batch_merkle_warmup_inputs<F: FftField>(
    leaf_count: usize,
    max_group_width: usize,
) -> (Vec<Vec<F>>, Vec<Vec<usize>>) {
    let columns = vec![vec![F::zero(); leaf_count]; max_group_width];
    let groups = vec![(0..max_group_width).collect::<Vec<_>>()];
    (columns, groups)
}

pub fn warm_up_gpu_batch_combine<F: PrimeField + FftField>(
    lde_size: usize,
    column_count: usize,
) -> Result<(), Error> {
    if lde_size == 0 || column_count == 0 {
        return Ok(());
    }
    if !is_power_of_two(lde_size) {
        return Err(Error::InvalidDomainSize(lde_size));
    }

    match requested_batch_combine_backend() {
        Some(BatchCombineBackendKind::Cpu) => Ok(()),
        Some(BatchCombineBackendKind::Gpu) => {
            let (columns, coefficients, degree_bounds, degree_bound) =
                batch_combine_warmup_inputs::<F>(lde_size, column_count);
            let column_refs = column_slices(&columns);
            try_combine_lde_columns_gpu(&column_refs, &coefficients, &degree_bounds, degree_bound)?
                .map(|_| ())
                .ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if should_auto_use_gpu_batch_combine::<F>(lde_size) {
                let (columns, coefficients, degree_bounds, degree_bound) =
                    batch_combine_warmup_inputs::<F>(lde_size, column_count);
                let column_refs = column_slices(&columns);
                let _ = try_combine_lde_columns_gpu(
                    &column_refs,
                    &coefficients,
                    &degree_bounds,
                    degree_bound,
                );
            }
            Ok(())
        }
    }
}

fn batch_combine_warmup_inputs<F: FftField>(
    lde_size: usize,
    column_count: usize,
) -> (Vec<Vec<F>>, Vec<F>, Vec<usize>, usize) {
    let degree_bound = lde_size;
    let alternate_bound = (degree_bound / 2).max(1);
    let columns = vec![vec![F::one(); lde_size]; column_count];
    let coefficients = vec![F::one(); column_count];
    let degree_bounds = (0..column_count)
        .map(|index| {
            if index % 2 == 0 {
                degree_bound
            } else {
                alternate_bound
            }
        })
        .collect::<Vec<_>>();
    (columns, coefficients, degree_bounds, degree_bound)
}

pub fn warm_up_gpu_resident_fri<F: PrimeField + FftField>(
    lde_size: usize,
    column_count: usize,
    blowup: usize,
) -> Result<(), Error> {
    if lde_size == 0 || column_count == 0 {
        return Ok(());
    }
    if !is_power_of_two(lde_size) {
        return Err(Error::InvalidDomainSize(lde_size));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }

    match requested_gpu_resident_fri_backend() {
        Some(GpuResidentFriBackendKind::Cpu) => Ok(()),
        Some(GpuResidentFriBackendKind::Gpu) => {
            let (columns, coefficients, degree_bounds, degree_bound, opened_indices) =
                gpu_resident_fri_warmup_inputs::<F>(column_count, blowup);
            let column_refs = column_slices(&columns);
            try_prove_batch_combined_low_degree_gpu_resident(
                &column_refs,
                &coefficients,
                &degree_bounds,
                degree_bound,
                blowup,
                &opened_indices,
            )?
            .map(|_| ())
            .ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if should_auto_use_gpu_resident_fri::<F>(lde_size) {
                let (columns, coefficients, degree_bounds, degree_bound, opened_indices) =
                    gpu_resident_fri_warmup_inputs::<F>(column_count, blowup);
                let column_refs = column_slices(&columns);
                let _ = try_prove_batch_combined_low_degree_gpu_resident(
                    &column_refs,
                    &coefficients,
                    &degree_bounds,
                    degree_bound,
                    blowup,
                    &opened_indices,
                );
            }
            Ok(())
        }
    }
}

fn gpu_resident_fri_warmup_inputs<F: FftField>(
    column_count: usize,
    blowup: usize,
) -> (Vec<Vec<F>>, Vec<F>, Vec<usize>, usize, Vec<usize>) {
    let degree_bound = 2;
    let lde_size = degree_bound * blowup;
    let alternate_bound = 1;
    let columns = vec![vec![F::one(); lde_size]; column_count];
    let coefficients = vec![F::one(); column_count];
    let degree_bounds = (0..column_count)
        .map(|index| {
            if index % 2 == 0 {
                degree_bound
            } else {
                alternate_bound
            }
        })
        .collect::<Vec<_>>();
    (columns, coefficients, degree_bounds, degree_bound, vec![0])
}

pub fn warm_up_gpu_fold<F: FftField>(domain_size: usize) -> Result<(), Error> {
    validate_fold_domain_size(domain_size)?;

    match requested_fold_backend() {
        Some(FriFoldBackendKind::Cpu) => Ok(()),
        Some(FriFoldBackendKind::Gpu) => {
            let warmup_values = vec![F::zero(); domain_size];
            try_fold_layer_gpu(&warmup_values, F::one())?
                .map(|_| ())
                .ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if should_auto_use_gpu_fold::<F>(domain_size) {
                let warmup_values = vec![F::zero(); domain_size];
                let _ = try_fold_layer_gpu(&warmup_values, F::one());
            }
            Ok(())
        }
    }
}

fn fold_layer_with_tree<F: PrimeField + FftField + Send + Sync>(
    evals: &[F],
    alpha: F,
) -> Result<(Vec<F>, FriTreeBackend), Error> {
    validate_fold_domain_size(evals.len())?;

    match requested_fold_backend() {
        Some(FriFoldBackendKind::Cpu) => fold_layer_with_cpu_tree(evals, alpha),
        Some(FriFoldBackendKind::Gpu) => {
            try_fold_layer_gpu_with_tree(evals, alpha)?.ok_or(Error::GpuBackendUnavailable)
        }
        None => {
            if should_auto_use_gpu_fold::<F>(evals.len()) {
                if let Ok(Some(result)) = try_fold_layer_gpu_with_tree(evals, alpha) {
                    return Ok(result);
                }
            }
            fold_layer_with_cpu_tree(evals, alpha)
        }
    }
}

fn fold_layer_with_cpu_tree<F: PrimeField + FftField + Send + Sync>(
    evals: &[F],
    alpha: F,
) -> Result<(Vec<F>, FriTreeBackend), Error> {
    let (next, tree) = fold_layer_cpu_with_tree(evals, alpha)?;
    Ok((next, FriTreeBackend::cpu(tree)))
}

fn try_fold_layer_gpu_with_tree<F: PrimeField + FftField + Send + Sync>(
    evals: &[F],
    alpha: F,
) -> Result<Option<(Vec<F>, FriTreeBackend)>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| {
            gpu_fold_bn254::try_fold_layer_with_tree(evals, alpha).map(|result| {
                result.map(|(folded, gpu_tree)| {
                    let tree = if let Some(gpu_tree) = gpu_tree {
                        FriTreeBackend::gpu(gpu_tree)
                    } else {
                        FriTreeBackend::cpu(MerkleTree::from_column(&folded))
                    };
                    (folded, tree)
                })
            })
        });
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (evals, alpha);
        Ok(None)
    }
}

fn validate_fold_domain_size(domain_size: usize) -> Result<(), Error> {
    if !is_power_of_two(domain_size) || domain_size < 2 {
        return Err(Error::InvalidDomainSize(domain_size));
    }
    Ok(())
}

fn fold_layer_cpu_with_tree<F: FftField + Send + Sync>(
    evals: &[F],
    alpha: F,
) -> Result<(Vec<F>, MerkleTree), Error> {
    validate_fold_domain_size(evals.len())?;

    let domain = GeneralEvaluationDomain::<F>::new(evals.len())
        .ok_or(Error::InvalidDomainSize(evals.len()))?;
    let two_inv = two_inverse::<F>()?;
    let omega_inv = domain
        .element(1)
        .inverse()
        .ok_or(Error::LowDegreeCheckFailed)?;

    let half = evals.len() / 2;
    let worker_count = cpu_parallel_worker_count(half);
    if worker_count <= 1 {
        let (folded, leaf_digests) =
            fold_layer_cpu_range_with_leaf_digests(evals, alpha, two_inv, omega_inv, 0, half);
        return Ok((folded, MerkleTree::from_leaf_digests(leaf_digests)));
    }

    let chunk_size = (half + worker_count - 1) / worker_count;
    let mut chunk_results = thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, start) in (0..half).step_by(chunk_size).enumerate() {
            let end = (start + chunk_size).min(half);
            handles.push(scope.spawn(move || {
                let (folded, leaf_digests) = fold_layer_cpu_range_with_leaf_digests(
                    evals, alpha, two_inv, omega_inv, start, end,
                );
                (chunk_index, folded, leaf_digests)
            }));
        }

        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| Error::LowDegreeCheckFailed))
            .collect::<Result<Vec<_>, Error>>()
    })?;

    chunk_results.sort_by_key(|(chunk_index, _, _)| *chunk_index);
    let mut folded = Vec::with_capacity(half);
    let mut leaf_digests = Vec::with_capacity(half);
    for (_, chunk_folded, chunk_leaf_digests) in chunk_results {
        folded.extend(chunk_folded);
        leaf_digests.extend(chunk_leaf_digests);
    }

    Ok((folded, MerkleTree::from_leaf_digests(leaf_digests)))
}

fn fold_layer_cpu_range_with_leaf_digests<F: FftField>(
    evals: &[F],
    alpha: F,
    two_inv: F,
    omega_inv: F,
    start: usize,
    end: usize,
) -> (Vec<F>, Vec<Digest32>) {
    let half = evals.len() / 2;
    let mut x_inv = omega_inv.pow(&[start as u64]);
    let mut folded = Vec::with_capacity(end - start);
    let mut leaf_digests = Vec::with_capacity(end - start);
    for index in start..end {
        let folded_value =
            fold_pair_with_inverse(evals[index], evals[index + half], alpha, two_inv, x_inv);
        leaf_digests.push(row_to_leaf(std::slice::from_ref(&folded_value)));
        folded.push(folded_value);
        x_inv *= omega_inv;
    }
    (folded, leaf_digests)
}

fn fold_pair<F: FftField>(
    domain_size: usize,
    index: usize,
    left: F,
    right: F,
    alpha: F,
) -> Result<F, Error> {
    let domain = GeneralEvaluationDomain::<F>::new(domain_size)
        .ok_or(Error::InvalidDomainSize(domain_size))?;
    let x_inv = domain
        .element(index)
        .inverse()
        .ok_or(Error::LowDegreeCheckFailed)?;
    Ok(fold_pair_with_inverse(
        left,
        right,
        alpha,
        two_inverse::<F>()?,
        x_inv,
    ))
}

fn fold_pair_with_inverse<F: FftField>(left: F, right: F, alpha: F, two_inv: F, x_inv: F) -> F {
    let even = (left + right) * two_inv;
    let odd = (left - right) * two_inv * x_inv;
    even + alpha * odd
}

fn two_inverse<F: FftField>() -> Result<F, Error> {
    (F::one() + F::one())
        .inverse()
        .ok_or(Error::LowDegreeCheckFailed)
}

fn requested_fold_backend() -> Option<FriFoldBackendKind> {
    match env::var(FOLD_BACKEND_ENV)
        .ok()?
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" => Some(FriFoldBackendKind::Cpu),
        "gpu" => Some(FriFoldBackendKind::Gpu),
        "auto" => None,
        _ => None,
    }
}

fn should_auto_use_gpu_fold<F: FftField>(domain_size: usize) -> bool {
    domain_size >= gpu_fold_min_size() && gpu_fold_supports_field::<F>()
}

fn gpu_fold_min_size() -> usize {
    env::var(GPU_FOLD_MIN_SIZE_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_GPU_FOLD_MIN_SIZE)
}

fn gpu_fold_supports_field<F: FftField>() -> bool {
    #[cfg(feature = "icicle-rust")]
    {
        gpu_fold_bn254::supports_field::<F>()
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        false
    }
}

fn try_fold_layer_gpu<F: FftField>(evals: &[F], alpha: F) -> Result<Option<Vec<F>>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        return crate::gpu_lock::with_gpu_lock(|| gpu_fold_bn254::try_fold_layer(evals, alpha));
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (evals, alpha);
        Ok(None)
    }
}

#[cfg(feature = "icicle-rust")]
mod gpu_resident_fri_bn254 {
    use super::{
        CombinedLdeBackend, FriLayerOpening, FriProof, FriQuery, GpuCombinedFriResult,
        PreparedLowDegreeFromCoeffs, PreparedTableLde, PreparedTableLowDegree,
    };
    use crate::{
        error::Error,
        merkle::{Digest32, MerklePath, MerkleTree},
    };
    use ark_bn254::Fr as ArkBn254Fr;
    use ark_ff::{BigInteger, FftField, Field, PrimeField};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use icicle_bn254::curve::ScalarField as IcicleBn254Fr;
    use icicle_core::{
        bignum::BigNum,
        hash::HashConfig,
        ntt::{get_root_of_unity, initialize_domain, ntt, NTTConfig, NTTDir, NTTInitDomainConfig},
        vec_ops::{
            add_scalars, inv_scalars, mul_scalars, scalar_add, scalar_mul, sub_scalars,
            VecOpsConfig,
        },
    };
    use icicle_hash::keccak::Keccak256 as IcicleKeccak256;
    use icicle_runtime::{
        memory::{DeviceVec, HostOrDeviceSlice, IntoIcicleSlice, IntoIcicleSliceMut},
        runtime::{load_backend_from_env_or_default, set_device},
        stream::IcicleStream,
        Device,
    };
    use std::{
        cell::RefCell,
        env,
        ffi::c_void,
        marker::PhantomData,
        sync::{Arc, Mutex, OnceLock},
    };

    const ICICLE_DEVICE_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE";
    const ICICLE_DEVICE_ID_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE_ID";
    const DIGEST_BYTES: usize = 32;
    const LEAF_PREIMAGE_BYTES: usize = 36;
    const ROW2_LEAF_PREIMAGE_BYTES: usize = 68;
    const ROW3_LEAF_PREIMAGE_BYTES: usize = 100;
    const ROW5_LEAF_PREIMAGE_BYTES: usize = 164;
    const NODE_PREIMAGE_BYTES: usize = 68;

    static RUNTIME_INIT: OnceLock<Result<(), Error>> = OnceLock::new();
    static NTT_DOMAIN_SIZE: OnceLock<Mutex<usize>> = OnceLock::new();

    extern "C" {
        fn fri_lookup_bn254_leaf_preimages_cuda(
            fields: *const c_void,
            output: *mut u8,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_cuda_memset_zero(
            output: *mut c_void,
            byte_count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_bn254_row2_leaf_preimages_cuda(
            column0: *const c_void,
            column1: *const c_void,
            output: *mut u8,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_bn254_row3_leaf_preimages_cuda(
            column0: *const c_void,
            column1: *const c_void,
            column2: *const c_void,
            output: *mut u8,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_bn254_row5_leaf_preimages_cuda(
            column0: *const c_void,
            column1: *const c_void,
            column2: *const c_void,
            column3: *const c_void,
            column4: *const c_void,
            output: *mut u8,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_merkle_node_preimages_cuda(
            digests: *const u8,
            output: *mut u8,
            pair_count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_bn254_gather_fields_cuda(
            fields: *const c_void,
            indices: *const usize,
            output: *mut c_void,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_bn254_gather_row_fields_cuda(
            column0: *const c_void,
            column1: *const c_void,
            column2: *const c_void,
            column3: *const c_void,
            column4: *const c_void,
            indices: *const usize,
            output: *mut c_void,
            count: usize,
            width: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_gather_digests_cuda(
            digests: *const u8,
            indices: *const usize,
            output: *mut u8,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
    }

    thread_local! {
        static GPU_RESIDENT_FRI_CONTEXT: RefCell<Option<GpuResidentFriContext>> = RefCell::new(None);
    }

    pub(super) struct GpuResidentFriContext {
        pub(super) stream: IcicleStream,
        workspace: Option<GpuResidentFriWorkspace>,
        cached_unshifted_inverse_weights: Option<CachedDeviceFields>,
        pub(super) batch_combine_workspace:
            Option<super::gpu_batch_combine_bn254::BatchCombineWorkspace>,
        pub(super) fold_workspace: Option<super::gpu_fold_bn254::FoldWorkspace>,
    }

    struct CachedDeviceFields {
        len: usize,
        values: DeviceVec<IcicleBn254Fr>,
    }

    struct GpuResidentFriWorkspace {
        capacity: usize,
        column: DeviceVec<IcicleBn254Fr>,
        weights: DeviceVec<IcicleBn254Fr>,
        coefficient: DeviceVec<IcicleBn254Fr>,
        term: DeviceVec<IcicleBn254Fr>,
        coeffs: DeviceVec<IcicleBn254Fr>,
        two_inv: DeviceVec<IcicleBn254Fr>,
        sum: DeviceVec<IcicleBn254Fr>,
        diff: DeviceVec<IcicleBn254Fr>,
        scaled_sum: DeviceVec<IcicleBn254Fr>,
        scaled_diff: DeviceVec<IcicleBn254Fr>,
        leaf_preimages: DeviceVec<u8>,
        node_preimages: DeviceVec<u8>,
    }

    struct GpuFriLayer {
        values: DeviceVec<IcicleBn254Fr>,
        len: usize,
        tree: GpuMerkleTree,
    }

    struct GpuMerkleTree {
        layers: Vec<DeviceVec<u8>>,
        counts: Vec<usize>,
        root: Digest32,
    }

    pub(super) struct PreparedLdeColumnStore<F: FftField> {
        columns: Vec<Arc<DeviceVec<IcicleBn254Fr>>>,
        len: usize,
        _field: PhantomData<F>,
    }

    unsafe impl<F: FftField> Send for PreparedLdeColumnStore<F> {}
    unsafe impl<F: FftField> Sync for PreparedLdeColumnStore<F> {}

    pub fn try_prepare_lde_column_store<F: PrimeField + FftField>(
        lde_columns: &[&[F]],
        degree_bound: usize,
        blowup: usize,
    ) -> Result<Option<PreparedLdeColumnStore<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        if lde_columns.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        let lde_size = degree_bound * blowup;
        if lde_columns.iter().any(|column| column.len() != lde_size) {
            return Err(Error::InvalidDomainSize(lde_size));
        }

        init_runtime()?;
        let columns = lde_columns
            .iter()
            .map(|column| unsafe { cast_ark_bn254_slice(column) })
            .collect::<Vec<_>>();
        let prepared =
            with_context(|context| prepare_lde_column_store_bn254(&columns, lde_size, context))?;
        Ok(Some(prepared))
    }

    pub fn try_prepare_lde_column_store_from_committed_columns<F: PrimeField + FftField>(
        prefix_lde_columns: &[&[F]],
        committed_columns: &super::gpu_batch_merkle_bn254::GpuCommittedLdeColumns,
        degree_bound: usize,
        blowup: usize,
    ) -> Result<Option<PreparedLdeColumnStore<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        let lde_size = degree_bound * blowup;
        if committed_columns.len() != lde_size {
            return Err(Error::InvalidDomainSize(committed_columns.len()));
        }
        if prefix_lde_columns
            .iter()
            .any(|column| column.len() != lde_size)
        {
            return Err(Error::InvalidDomainSize(lde_size));
        }

        init_runtime()?;
        let prefix_columns = prefix_lde_columns
            .iter()
            .map(|column| unsafe { cast_ark_bn254_slice(column) })
            .collect::<Vec<_>>();
        let prepared = with_context(|context| {
            prepare_lde_column_store_from_committed_columns_bn254(
                &prefix_columns,
                committed_columns,
                lde_size,
                context,
            )
        })?;
        Ok(Some(prepared))
    }

    pub fn try_prove_batch_combined_low_degree_from_store<F: PrimeField + FftField>(
        store: &PreparedLdeColumnStore<F>,
        column_indices: &[usize],
        batching_coefficients: &[F],
        column_degree_bounds: &[usize],
        degree_bound: usize,
        blowup: usize,
        opened_indices: &[usize],
    ) -> Result<GpuCombinedFriResult<F>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Err(Error::GpuBackendUnavailable);
        }
        if !opened_indices.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        let lde_size = degree_bound * blowup;
        validate_store_selection(store, column_indices, lde_size)?;
        if column_indices.len() != batching_coefficients.len()
            || column_indices.len() != column_degree_bounds.len()
        {
            return Err(Error::LowDegreeCheckFailed);
        }
        if column_degree_bounds
            .iter()
            .any(|bound| *bound == 0 || *bound > degree_bound)
        {
            return Err(Error::LowDegreeCheckFailed);
        }

        init_runtime()?;
        let coefficients = unsafe { cast_ark_bn254_slice(batching_coefficients) };
        with_context(|context| {
            let columns = store_column_refs(store, column_indices, lde_size)?;
            prove_device_batch_combined_low_degree_bn254::<F>(
                &columns,
                coefficients,
                column_degree_bounds,
                degree_bound,
                blowup,
                context,
            )
        })
    }

    pub fn try_prepare_ood_quotient_batch_from_store<F: PrimeField + FftField>(
        store: &PreparedLdeColumnStore<F>,
        column_indices: &[usize],
        claimed_values: &[F],
        zeta: F,
        degree_bound: usize,
        blowup: usize,
        column_degree_bounds: &[usize],
    ) -> Result<PreparedOodQuotientBatch<F>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Err(Error::GpuBackendUnavailable);
        }
        if column_indices.len() != 3 && column_indices.len() != 5 {
            return Err(Error::InvalidOutOfDomainOpening);
        }
        if column_indices.is_empty()
            || column_indices.len() != claimed_values.len()
            || column_indices.len() != column_degree_bounds.len()
        {
            return Err(Error::InvalidOutOfDomainOpening);
        }
        let lde_size = degree_bound * blowup;
        validate_store_selection(store, column_indices, lde_size)?;
        if column_degree_bounds
            .iter()
            .any(|bound| *bound == 0 || *bound > degree_bound)
        {
            return Err(Error::LowDegreeCheckFailed);
        }

        init_runtime()?;
        let claims = claimed_values
            .iter()
            .map(|value| unsafe { cast_ark_bn254_value(value) })
            .collect::<Vec<_>>();
        let zeta = unsafe { cast_ark_bn254_value(&zeta) };
        with_context(|context| {
            let columns = store_column_refs(store, column_indices, lde_size)?;
            prepare_ood_quotient_batch_from_device_columns_bn254::<F>(
                &columns,
                &claims,
                zeta,
                degree_bound,
                blowup,
                column_degree_bounds,
                context,
            )
        })
    }

    pub fn try_prepare_specialized_usc_batch_from_store<F: PrimeField + FftField>(
        store: &PreparedLdeColumnStore<F>,
        column_index: usize,
        claimed_sum: F,
        degree_bound: usize,
        blowup: usize,
        p_degree_bound: usize,
    ) -> Result<PreparedSpecializedUscBatch<F>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Err(Error::GpuBackendUnavailable);
        }
        let lde_size = degree_bound * blowup;
        validate_store_selection(store, &[column_index], lde_size)?;
        if p_degree_bound == 0 || p_degree_bound > degree_bound {
            return Err(Error::LowDegreeCheckFailed);
        }

        init_runtime()?;
        let claimed_sum = unsafe { cast_ark_bn254_value(&claimed_sum) };
        with_context(|context| {
            let columns = store_column_refs(store, &[column_index], lde_size)?;
            prepare_specialized_usc_batch_from_device_column_bn254::<F>(
                columns[0],
                claimed_sum,
                degree_bound,
                blowup,
                p_degree_bound,
                context,
            )
        })
    }

    pub fn lde_rows_from_store<F: PrimeField + FftField + Send>(
        store: &PreparedLdeColumnStore<F>,
        column_indices: &[usize],
        lde_indices: &[usize],
    ) -> Result<Vec<Vec<F>>, Error> {
        if column_indices.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        if lde_indices.is_empty() {
            return Ok(Vec::new());
        }
        if lde_indices.iter().any(|index| *index >= store.len) {
            return Err(Error::InvalidDomainSize(store.len));
        }
        init_runtime()?;
        with_context(|context| {
            let columns = store_column_refs(store, column_indices, store.len)?;
            let column_values =
                copy_device_field_columns_indices(&columns, lde_indices, &context.stream)?;
            let rows = (0..lde_indices.len())
                .map(|position| {
                    column_values
                        .iter()
                        .map(|values| unsafe { cast_bn254_value_to_f(values[position]) })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            Ok(rows)
        })
    }

    fn prepare_lde_column_store_bn254<F: FftField>(
        lde_columns: &[&[ArkBn254Fr]],
        lde_size: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<PreparedLdeColumnStore<F>, Error> {
        let stream = &context.stream;
        let mut columns = Vec::with_capacity(lde_columns.len());
        for column in lde_columns {
            if column.len() != lde_size {
                return Err(Error::InvalidDomainSize(column.len()));
            }
            let column = column
                .iter()
                .map(ark_to_icicle)
                .collect::<Result<Vec<_>, _>>()?;
            let mut device_column = allocate_device_vec(lde_size)?;
            device_column[..lde_size]
                .copy_from_host_async(column.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
            columns.push(Arc::new(device_column));
        }
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        Ok(PreparedLdeColumnStore {
            columns,
            len: lde_size,
            _field: PhantomData,
        })
    }

    fn prepare_lde_column_store_from_committed_columns_bn254<F: FftField>(
        prefix_lde_columns: &[&[ArkBn254Fr]],
        committed_columns: &super::gpu_batch_merkle_bn254::GpuCommittedLdeColumns,
        lde_size: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<PreparedLdeColumnStore<F>, Error> {
        let stream = &context.stream;
        let mut columns = Vec::with_capacity(prefix_lde_columns.len() + committed_columns.width());
        for column in prefix_lde_columns {
            if column.len() != lde_size {
                return Err(Error::InvalidDomainSize(column.len()));
            }
            let column = column
                .iter()
                .map(ark_to_icicle)
                .collect::<Result<Vec<_>, _>>()?;
            let mut device_column = allocate_device_vec(lde_size)?;
            device_column[..lde_size]
                .copy_from_host_async(column.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
            columns.push(Arc::new(device_column));
        }
        columns.extend(committed_columns.columns().iter().cloned());
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        Ok(PreparedLdeColumnStore {
            columns,
            len: lde_size,
            _field: PhantomData,
        })
    }

    fn validate_store_selection<F: FftField>(
        store: &PreparedLdeColumnStore<F>,
        column_indices: &[usize],
        lde_size: usize,
    ) -> Result<(), Error> {
        if store.len != lde_size {
            return Err(Error::InvalidDomainSize(store.len));
        }
        for &index in column_indices {
            let column = store
                .columns
                .get(index)
                .ok_or(Error::LowDegreeCheckFailed)?;
            if column.len() < lde_size {
                return Err(Error::InvalidDomainSize(column.len()));
            }
        }
        Ok(())
    }

    fn store_column_refs<'a, F: FftField>(
        store: &'a PreparedLdeColumnStore<F>,
        column_indices: &[usize],
        lde_size: usize,
    ) -> Result<Vec<&'a DeviceVec<IcicleBn254Fr>>, Error> {
        validate_store_selection(store, column_indices, lde_size)?;
        column_indices
            .iter()
            .map(|&index| {
                store
                    .columns
                    .get(index)
                    .map(Arc::as_ref)
                    .ok_or(Error::LowDegreeCheckFailed)
            })
            .collect()
    }

    pub(super) struct GpuPreparedCombinedLde<F: FftField> {
        values: DeviceVec<IcicleBn254Fr>,
        len: usize,
        tree: GpuMerkleTree,
        _field: PhantomData<F>,
    }

    unsafe impl<F: FftField> Send for GpuPreparedCombinedLde<F> {}
    unsafe impl<F: FftField> Sync for GpuPreparedCombinedLde<F> {}

    impl<F: FftField> GpuPreparedCombinedLde<F> {
        fn new(values: DeviceVec<IcicleBn254Fr>, len: usize, tree: GpuMerkleTree) -> Self {
            Self {
                values,
                len,
                tree,
                _field: PhantomData,
            }
        }

        pub(super) fn len(&self) -> usize {
            self.len
        }

        pub(super) fn root(&self) -> Digest32 {
            self.tree.root
        }
    }

    impl<F: FftField + Send> GpuPreparedCombinedLde<F> {
        #[allow(dead_code)]
        pub(super) fn open(&self, index: usize) -> Result<(F, MerklePath), Error> {
            Ok(self.open_many(&[index])?.remove(0))
        }

        pub(super) fn open_many(&self, indices: &[usize]) -> Result<Vec<(F, MerklePath)>, Error> {
            crate::gpu_lock::with_gpu_lock(|| {
                init_runtime()?;
                with_context(|context| self.open_many_with_stream(indices, &context.stream))
            })
        }

        fn open_many_with_stream(
            &self,
            indices: &[usize],
            stream: &IcicleStream,
        ) -> Result<Vec<(F, MerklePath)>, Error> {
            if indices.is_empty() {
                return Ok(Vec::new());
            }
            for &index in indices {
                if index >= self.len {
                    return Err(Error::InvalidDomainSize(index));
                }
            }

            let values = copy_device_field_indices(&self.values, indices, stream)?;
            let paths = self.tree.open_many(indices, stream)?;
            Ok(values
                .into_iter()
                .zip(paths)
                .map(|(value, path)| (unsafe { cast_bn254_value_to_f(value) }, path))
                .collect())
        }
    }

    pub(super) fn materialize_prepared_combined_lde<F: PrimeField + FftField + Send>(
        prepared: &Arc<GpuPreparedCombinedLde<F>>,
    ) -> Result<(Vec<F>, MerkleTree), Error> {
        crate::gpu_lock::with_gpu_lock(|| {
            init_runtime()?;
            with_context(|context| {
                let values = unsafe {
                    cast_bn254_vec(copy_device_fields(
                        &prepared.values,
                        prepared.len,
                        &context.stream,
                    )?)
                };
                let tree = copy_gpu_merkle_tree_to_cpu(&prepared.tree, &context.stream)?;
                Ok((values, tree))
            })
        })
    }

    pub(super) fn try_prepare_lde_column_store_from_table_and_committed_columns<
        F: PrimeField + FftField,
    >(
        table_lde: &Arc<GpuPreparedCombinedLde<F>>,
        committed_columns: &super::gpu_batch_merkle_bn254::GpuCommittedLdeColumns,
        degree_bound: usize,
        blowup: usize,
    ) -> Result<Option<PreparedLdeColumnStore<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        let lde_size = degree_bound * blowup;
        if lde_size == 0 || !lde_size.is_power_of_two() || table_lde.len != lde_size {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        if committed_columns.len() != lde_size {
            return Err(Error::InvalidDomainSize(committed_columns.len()));
        }

        init_runtime()?;
        with_context(|context| {
            prepare_lde_column_store_from_table_and_committed_columns_bn254(
                table_lde.as_ref(),
                committed_columns,
                lde_size,
                context,
            )
        })
        .map(Some)
    }

    fn prepare_lde_column_store_from_table_and_committed_columns_bn254<F: FftField>(
        table_lde: &GpuPreparedCombinedLde<F>,
        committed_columns: &super::gpu_batch_merkle_bn254::GpuCommittedLdeColumns,
        lde_size: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<PreparedLdeColumnStore<F>, Error> {
        let stream = &context.stream;
        let mut table_column = allocate_device_vec(lde_size)?;
        table_column[..lde_size]
            .copy(&table_lde.values[..lde_size])
            .map_err(|_| Error::GpuBackendFailed)?;
        let mut columns = Vec::with_capacity(1 + committed_columns.width());
        columns.push(Arc::new(table_column));
        columns.extend(committed_columns.columns().iter().cloned());
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        Ok(PreparedLdeColumnStore {
            columns,
            len: lde_size,
            _field: PhantomData,
        })
    }

    pub(super) struct PreparedOodQuotientBatch<F: FftField> {
        quotient_columns: Vec<DeviceVec<IcicleBn254Fr>>,
        group_root: Digest32,
        row_count: usize,
        blowup: usize,
        quotient_degree_bounds: Vec<usize>,
        _field: PhantomData<F>,
    }

    unsafe impl<F: FftField> Send for PreparedOodQuotientBatch<F> {}
    unsafe impl<F: FftField> Sync for PreparedOodQuotientBatch<F> {}

    impl<F: FftField> PreparedOodQuotientBatch<F> {
        pub(super) fn root(&self) -> Digest32 {
            self.group_root
        }
    }

    impl<F: PrimeField + FftField + Send> PreparedOodQuotientBatch<F> {
        pub(super) fn finish(
            self,
            batching_coefficients: &[F],
        ) -> Result<(super::BatchFriProof<F>, Vec<super::BatchLdeRowOpening<F>>), Error> {
            if batching_coefficients.len() != self.quotient_columns.len()
                || batching_coefficients.len() != self.quotient_degree_bounds.len()
            {
                return Err(Error::LowDegreeCheckFailed);
            }

            init_runtime()?;
            let coefficients_f = batching_coefficients.to_vec();
            let coefficients = unsafe { cast_ark_bn254_slice(batching_coefficients) };
            with_context(|context| {
                finish_ood_quotient_batch_bn254(self, coefficients, coefficients_f, context)
            })
        }
    }

    pub(super) struct PreparedSpecializedUscBatch<F: FftField> {
        p_column: DeviceVec<IcicleBn254Fr>,
        p_tree: GpuMerkleTree,
        row_count: usize,
        blowup: usize,
        p_degree_bound: usize,
        _field: PhantomData<F>,
    }

    unsafe impl<F: FftField> Send for PreparedSpecializedUscBatch<F> {}
    unsafe impl<F: FftField> Sync for PreparedSpecializedUscBatch<F> {}

    impl<F: FftField> PreparedSpecializedUscBatch<F> {
        pub(super) fn root(&self) -> Digest32 {
            self.p_tree.root
        }
    }

    impl<F: PrimeField + FftField + Send> PreparedSpecializedUscBatch<F> {
        pub(super) fn finish(
            self,
            batching_coefficients: &[F],
        ) -> Result<(super::BatchFriProof<F>, Vec<super::BatchLdeRowOpening<F>>), Error> {
            if batching_coefficients.len() != 1 {
                return Err(Error::LowDegreeCheckFailed);
            }

            init_runtime()?;
            let coefficients_f = batching_coefficients.to_vec();
            let coefficients = unsafe { cast_ark_bn254_slice(batching_coefficients) };
            with_context(|context| {
                finish_specialized_usc_batch_bn254(self, coefficients, coefficients_f, context)
            })
        }
    }

    impl Drop for GpuResidentFriContext {
        fn drop(&mut self) {
            let _ = self.stream.synchronize();
            self.cached_unshifted_inverse_weights = None;
            self.workspace = None;
            self.batch_combine_workspace = None;
            self.fold_workspace = None;
            let _ = self.stream.destroy();
        }
    }

    impl GpuResidentFriWorkspace {
        fn new(capacity: usize) -> Result<Self, Error> {
            Ok(Self {
                capacity,
                column: allocate_device_vec(capacity)?,
                weights: allocate_device_vec(capacity)?,
                coefficient: allocate_device_vec(1)?,
                term: allocate_device_vec(capacity)?,
                coeffs: allocate_device_vec(capacity)?,
                two_inv: allocate_device_vec(1)?,
                sum: allocate_device_vec(capacity)?,
                diff: allocate_device_vec(capacity)?,
                scaled_sum: allocate_device_vec(capacity)?,
                scaled_diff: allocate_device_vec(capacity)?,
                leaf_preimages: allocate_device_bytes(capacity * LEAF_PREIMAGE_BYTES)?,
                node_preimages: allocate_device_bytes((capacity / 2).max(1) * NODE_PREIMAGE_BYTES)?,
            })
        }
    }

    impl GpuMerkleTree {
        fn open_many(
            &self,
            indices: &[usize],
            stream: &IcicleStream,
        ) -> Result<Vec<MerklePath>, Error> {
            if indices.is_empty() {
                return Ok(Vec::new());
            }
            for &index in indices {
                if index >= self.counts[0] {
                    return Err(Error::InvalidDomainSize(index));
                }
            }

            let mut paths = indices
                .iter()
                .copied()
                .map(|index| MerklePath {
                    index,
                    siblings: Vec::with_capacity(self.layers.len() - 1),
                })
                .collect::<Vec<_>>();
            let mut current_indices = indices.to_vec();

            for (layer, &count) in self
                .layers
                .iter()
                .zip(&self.counts)
                .take(self.layers.len() - 1)
            {
                let sibling_indices = current_indices
                    .iter()
                    .copied()
                    .map(|index| {
                        if index >= count {
                            return Err(Error::InvalidDomainSize(index));
                        }
                        Ok(if index % 2 == 0 { index + 1 } else { index - 1 })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let siblings = copy_digest_indices(layer, &sibling_indices, stream)?;
                for (path, sibling) in paths.iter_mut().zip(siblings) {
                    path.siblings.push(sibling);
                }
                for index in &mut current_indices {
                    *index /= 2;
                }
            }

            Ok(paths)
        }
    }

    pub fn supports_field<F: FftField>() -> bool {
        is_ark_bn254_fr::<F>()
    }

    pub fn try_prepare_table_low_degree<F: PrimeField + FftField>(
        table_values: &[F],
        degree_bound: usize,
        blowup: usize,
    ) -> Result<Option<PreparedTableLowDegree<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        if table_values.len() != degree_bound {
            return Err(Error::InvalidDomainSize(table_values.len()));
        }
        if degree_bound == 0 || !degree_bound.is_power_of_two() {
            return Err(Error::InvalidDomainSize(degree_bound));
        }
        if !blowup.is_power_of_two() {
            return Err(Error::InvalidBlowup(blowup));
        }
        let lde_size = degree_bound * blowup;
        if !lde_size.is_power_of_two() {
            return Err(Error::InvalidDomainSize(lde_size));
        }

        init_runtime()?;
        init_ntt_domain(lde_size)?;
        let table_values = unsafe { cast_ark_bn254_slice(table_values) };
        let prepared = with_context(|context| {
            prepare_table_low_degree_bn254::<F>(table_values, degree_bound, blowup, context)
        })?;
        Ok(Some(prepared))
    }

    fn prepare_table_low_degree_bn254<F: PrimeField + FftField>(
        table_values: &[ArkBn254Fr],
        degree_bound: usize,
        blowup: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<PreparedTableLowDegree<F>, Error> {
        let domain_size = degree_bound * blowup;
        let values = device_fields_from_host(table_values, context)?;
        let coeffs_device = interpolate_device_values_bn254(&values, degree_bound, context)?;
        let polynomial_coefficients = unsafe {
            cast_bn254_vec(copy_device_fields(
                &coeffs_device,
                degree_bound,
                &context.stream,
            )?)
        };
        let initial_values =
            lde_from_device_coeffs_bn254(&coeffs_device, degree_bound, domain_size, context)?;
        let (lde, proof) =
            finish_table_low_degree_from_device_lde(initial_values, degree_bound, blowup, context)?;
        Ok(PreparedTableLowDegree {
            polynomial_coefficients,
            lde: PreparedTableLde::Gpu(lde),
            proof,
        })
    }

    pub fn try_prepare_low_degree_from_coeffs<F: PrimeField + FftField>(
        coeffs: &[F],
        degree_bound: usize,
        blowup: usize,
    ) -> Result<Option<PreparedLowDegreeFromCoeffs<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        if coeffs.len() > degree_bound {
            return Err(Error::LowDegreeCheckFailed);
        }
        if degree_bound == 0 || !degree_bound.is_power_of_two() {
            return Err(Error::InvalidDomainSize(degree_bound));
        }
        if !blowup.is_power_of_two() {
            return Err(Error::InvalidBlowup(blowup));
        }
        let lde_size = degree_bound * blowup;
        if !lde_size.is_power_of_two() {
            return Err(Error::InvalidDomainSize(lde_size));
        }

        init_runtime()?;
        init_ntt_domain(lde_size)?;
        let coeffs = unsafe { cast_ark_bn254_slice(coeffs) };
        let prepared = with_context(|context| {
            prepare_low_degree_from_coeffs_bn254::<F>(coeffs, degree_bound, blowup, context)
        })?;
        Ok(Some(prepared))
    }

    fn prepare_low_degree_from_coeffs_bn254<F: PrimeField + FftField>(
        coeffs: &[ArkBn254Fr],
        degree_bound: usize,
        blowup: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<PreparedLowDegreeFromCoeffs<F>, Error> {
        let domain_size = degree_bound * blowup;
        let initial_values = lde_from_coeffs_bn254(coeffs, degree_bound, domain_size, context)?;
        let (lde_values, lde_tree, proof) =
            finish_low_degree_from_device_lde(initial_values, degree_bound, blowup, context)?;
        Ok(PreparedLowDegreeFromCoeffs {
            lde_values,
            lde_tree,
            proof,
        })
    }

    fn finish_low_degree_from_device_lde<F: PrimeField + FftField>(
        initial_values: DeviceVec<IcicleBn254Fr>,
        degree_bound: usize,
        blowup: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<(Vec<F>, MerkleTree, FriProof<F>), Error> {
        let domain_size = degree_bound * blowup;
        gpu_resident_diag(|| {
            format!("device FRI: build initial Merkle tree start len={domain_size}")
        });
        let initial_tree = build_merkle_tree_from_device_fields(
            &initial_values,
            domain_size,
            context.workspace.as_mut().unwrap(),
            &context.stream,
        )
        .map_err(|error| {
            gpu_resident_diag(|| {
                format!("device FRI: build initial Merkle tree failed: {error:?}")
            });
            error
        })?;
        gpu_resident_diag(|| "device FRI: build initial Merkle tree ok".to_string());
        let lde_values = unsafe {
            cast_bn254_vec(copy_device_fields(
                &initial_values,
                domain_size,
                &context.stream,
            )?)
        };
        let lde_tree = copy_gpu_merkle_tree_to_cpu(&initial_tree, &context.stream)?;
        let result = prove_combined_device_low_degree_bn254_with_initial_tree::<F>(
            initial_values,
            domain_size,
            degree_bound,
            blowup,
            &[],
            initial_tree,
            context,
        )?;
        let GpuCombinedFriResult {
            proof,
            row_openings,
            combined_lde,
        } = result;
        if !row_openings.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        if proof.root != combined_lde.root() || proof.root != lde_tree.root() {
            return Err(Error::MerkleRootMismatch);
        }
        Ok((lde_values, lde_tree, proof))
    }

    fn finish_table_low_degree_from_device_lde<F: PrimeField + FftField>(
        initial_values: DeviceVec<IcicleBn254Fr>,
        degree_bound: usize,
        blowup: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<(super::PreparedGpuTableLde<F>, FriProof<F>), Error> {
        let domain_size = degree_bound * blowup;
        let initial_tree = build_merkle_tree_from_device_fields(
            &initial_values,
            domain_size,
            context.workspace.as_mut().unwrap(),
            &context.stream,
        )?;
        let result = prove_combined_device_low_degree_bn254_with_initial_tree::<F>(
            initial_values,
            domain_size,
            degree_bound,
            blowup,
            &[],
            initial_tree,
            context,
        )?;
        let GpuCombinedFriResult {
            proof,
            row_openings,
            combined_lde,
        } = result;
        if !row_openings.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        let CombinedLdeBackend::Gpu(inner) = combined_lde else {
            return Err(Error::GpuBackendFailed);
        };
        if proof.root != inner.root() {
            return Err(Error::MerkleRootMismatch);
        }
        Ok((super::PreparedGpuTableLde { inner }, proof))
    }

    pub fn try_prove_batch_combined_low_degree<F: PrimeField + FftField>(
        lde_columns: &[&[F]],
        batching_coefficients: &[F],
        column_degree_bounds: &[usize],
        degree_bound: usize,
        blowup: usize,
        opened_indices: &[usize],
    ) -> Result<Option<GpuCombinedFriResult<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        super::validate_low_degree_lde(&lde_columns[0], degree_bound, blowup)?;
        if lde_columns
            .iter()
            .any(|column| column.len() != lde_columns[0].len())
        {
            return Err(Error::InvalidDomainSize(lde_columns[0].len()));
        }
        if lde_columns.len() != batching_coefficients.len()
            || lde_columns.len() != column_degree_bounds.len()
        {
            return Err(Error::LowDegreeCheckFailed);
        }

        init_runtime()?;
        let columns = lde_columns
            .iter()
            .map(|column| unsafe { cast_ark_bn254_slice(column) })
            .collect::<Vec<_>>();
        let coefficients = unsafe { cast_ark_bn254_slice(batching_coefficients) };
        let result = with_context(|context| {
            prove_batch_combined_low_degree_bn254::<F>(
                &columns,
                coefficients,
                column_degree_bounds,
                degree_bound,
                blowup,
                opened_indices,
                context,
            )
        })?;
        Ok(Some(result))
    }

    pub fn try_prove_low_degree_from_lde<F: PrimeField + FftField>(
        lde_evals: &[F],
        degree_bound: usize,
        blowup: usize,
    ) -> Result<Option<FriProof<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        super::validate_low_degree_lde(lde_evals, degree_bound, blowup)?;

        init_runtime()?;
        let lde_evals = unsafe { cast_ark_bn254_slice(lde_evals) };
        let proof = with_context(|context| {
            prove_low_degree_from_lde_bn254::<F>(lde_evals, degree_bound, blowup, context)
        })?;
        Ok(Some(proof))
    }

    fn prove_low_degree_from_lde_bn254<F: PrimeField + FftField>(
        lde_evals: &[ArkBn254Fr],
        degree_bound: usize,
        blowup: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<FriProof<F>, Error> {
        let domain_size = degree_bound * blowup;
        if lde_evals.len() != domain_size {
            return Err(Error::InvalidDomainSize(lde_evals.len()));
        }
        ensure_workspace_capacity(&mut context.workspace, domain_size)?;
        let stream = &context.stream;

        let lde_values = lde_evals
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        let mut initial_values = allocate_device_vec(domain_size)?;
        initial_values[..domain_size]
            .copy_from_host_async(lde_values.into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;

        let result = prove_combined_device_low_degree_bn254::<F>(
            initial_values,
            domain_size,
            degree_bound,
            blowup,
            &[],
            context,
        )?;
        let GpuCombinedFriResult {
            proof,
            row_openings,
            combined_lde,
        } = result;
        if !row_openings.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        if proof.root != combined_lde.root() {
            return Err(Error::MerkleRootMismatch);
        }
        Ok(proof)
    }

    fn prove_batch_combined_low_degree_bn254<F: PrimeField + FftField>(
        lde_columns: &[&[ArkBn254Fr]],
        batching_coefficients: &[ArkBn254Fr],
        column_degree_bounds: &[usize],
        degree_bound: usize,
        blowup: usize,
        opened_indices: &[usize],
        context: &mut GpuResidentFriContext,
    ) -> Result<GpuCombinedFriResult<F>, Error> {
        let domain_size = lde_columns[0].len();
        ensure_workspace_capacity(&mut context.workspace, domain_size)?;

        let initial_values = combine_lde_columns_to_device(
            lde_columns,
            batching_coefficients,
            column_degree_bounds,
            degree_bound,
            context,
        )?;
        let initial_tree = build_merkle_tree_from_device_fields(
            &initial_values,
            domain_size,
            context.workspace.as_mut().unwrap(),
            &context.stream,
        )?;
        let mut roots = vec![initial_tree.root];
        let mut transcript =
            super::fri_transcript::<F>(roots[0], degree_bound, blowup, domain_size);
        let mut layers = vec![GpuFriLayer {
            values: initial_values,
            len: domain_size,
            tree: initial_tree,
        }];

        let mut current_degree_bound = degree_bound;
        while current_degree_bound > 1 {
            let alpha = transcript.challenge::<F>(b"fold-alpha");
            let alpha = unsafe { cast_ark_bn254_value(&alpha) };
            let (next_values, next_len) = {
                let current = layers.last().unwrap();
                gpu_resident_diag(|| {
                    format!(
                        "device FRI: fold layer start len={}, degree_bound={current_degree_bound}",
                        current.len
                    )
                });
                let next_values = fold_layer_on_device(&current.values, current.len, alpha, context)
                    .map_err(|error| {
                        gpu_resident_diag(|| {
                            format!(
                                "device FRI: fold layer failed len={}, degree_bound={current_degree_bound}: {error:?}",
                                current.len
                            )
                        });
                        error
                    })?;
                gpu_resident_diag(|| {
                    format!(
                        "device FRI: fold layer ok len={}, next_len={}",
                        current.len,
                        current.len / 2
                    )
                });
                (next_values, current.len / 2)
            };
            gpu_resident_diag(|| {
                format!("device FRI: build folded Merkle tree start len={next_len}")
            });
            let next_tree = build_merkle_tree_from_device_fields(
                &next_values,
                next_len,
                context.workspace.as_mut().unwrap(),
                &context.stream,
            )
            .map_err(|error| {
                gpu_resident_diag(|| {
                    format!("device FRI: build folded Merkle tree failed len={next_len}: {error:?}")
                });
                error
            })?;
            gpu_resident_diag(|| format!("device FRI: build folded Merkle tree ok len={next_len}"));
            let root = next_tree.root;
            transcript.absorb_bytes(b"fold-root", &root);
            roots.push(root);
            layers.push(GpuFriLayer {
                values: next_values,
                len: next_len,
                tree: next_tree,
            });
            current_degree_bound = (current_degree_bound + 1) / 2;
        }

        let final_layer = layers.last().unwrap();
        gpu_resident_diag(|| format!("device FRI: copy final evals start len={}", final_layer.len));
        let final_evals = unsafe {
            cast_bn254_vec(
                copy_device_fields(&final_layer.values, final_layer.len, &context.stream).map_err(
                    |error| {
                        gpu_resident_diag(|| {
                            format!(
                                "device FRI: copy final evals failed len={}: {error:?}",
                                final_layer.len
                            )
                        });
                        error
                    },
                )?,
            )
        };
        gpu_resident_diag(|| "device FRI: copy final evals ok".to_string());
        let row_openings =
            open_initial_rows::<F>(&layers[0], blowup, opened_indices, &context.stream).map_err(
                |error| {
                    gpu_resident_diag(|| {
                        format!("device FRI: open initial rows failed: {error:?}")
                    });
                    error
                },
            )?;
        let queries = sample_gpu_queries::<F>(&mut transcript, &layers, &context.stream).map_err(
            |error| {
                gpu_resident_diag(|| format!("device FRI: sample queries failed: {error:?}"));
                error
            },
        )?;
        let first_layer = layers.remove(0);
        let combined_lde = CombinedLdeBackend::gpu(GpuPreparedCombinedLde::new(
            first_layer.values,
            first_layer.len,
            first_layer.tree,
        ));

        Ok(GpuCombinedFriResult {
            proof: FriProof {
                root: roots[0],
                roots,
                final_evals,
                queries,
                degree_bound,
                blowup,
                domain_size,
            },
            row_openings,
            combined_lde,
        })
    }

    pub fn try_prepare_ood_quotient_batch<F: PrimeField + FftField>(
        original_lde_columns: &[&[F]],
        claimed_values: &[F],
        zeta: F,
        degree_bound: usize,
        blowup: usize,
        column_degree_bounds: &[usize],
    ) -> Result<Option<PreparedOodQuotientBatch<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        if original_lde_columns.len() != 3 && original_lde_columns.len() != 5 {
            return Ok(None);
        }
        if original_lde_columns.is_empty()
            || original_lde_columns.len() != claimed_values.len()
            || original_lde_columns.len() != column_degree_bounds.len()
        {
            return Err(Error::InvalidOutOfDomainOpening);
        }
        super::validate_low_degree_lde(&original_lde_columns[0], degree_bound, blowup)?;
        if original_lde_columns
            .iter()
            .any(|column| column.len() != original_lde_columns[0].len())
        {
            return Err(Error::InvalidDomainSize(original_lde_columns[0].len()));
        }
        if column_degree_bounds
            .iter()
            .any(|bound| *bound == 0 || *bound > degree_bound)
        {
            return Err(Error::LowDegreeCheckFailed);
        }

        init_runtime()?;
        let columns = original_lde_columns
            .iter()
            .map(|column| unsafe { cast_ark_bn254_slice(column) })
            .collect::<Vec<_>>();
        let claims = claimed_values
            .iter()
            .map(|value| unsafe { cast_ark_bn254_value(value) })
            .collect::<Vec<_>>();
        let zeta = unsafe { cast_ark_bn254_value(&zeta) };
        let prepared = with_context(|context| {
            prepare_ood_quotient_batch_bn254::<F>(
                &columns,
                &claims,
                zeta,
                degree_bound,
                blowup,
                column_degree_bounds,
                context,
            )
        })?;
        Ok(Some(prepared))
    }

    fn prepare_domain_inverse_weights_bn254(
        lde_size: usize,
        shift: Option<ArkBn254Fr>,
        context: &mut GpuResidentFriContext,
    ) -> Result<(), Error> {
        if lde_size == 0 || !lde_size.is_power_of_two() {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        if let Some(shift) = shift {
            if shift.pow(&[lde_size as u64]) == ArkBn254Fr::from(1u64) {
                return Err(Error::InvalidOutOfDomainPoint);
            }
        }

        ensure_workspace_capacity(&mut context.workspace, lde_size)?;
        if shift.is_none() {
            if let Some(cache) = context
                .cached_unshifted_inverse_weights
                .as_ref()
                .filter(|cache| cache.len == lde_size)
            {
                let workspace = context.workspace.as_mut().unwrap();
                workspace.weights[..lde_size]
                    .copy(&cache.values[..lde_size])
                    .map_err(|_| Error::GpuBackendFailed)?;
                return Ok(());
            }
        }

        init_ntt_domain(lde_size)?;
        let stream = &context.stream;
        let cached_weights = {
            let workspace = context.workspace.as_mut().unwrap();
            launch_memset_zero_fields(&mut workspace.coeffs[..lde_size], lde_size, stream)?;
            let one = ark_to_icicle(&ArkBn254Fr::from(1u64))?;
            let coefficient_index = usize::from(lde_size > 1);
            workspace.coeffs[coefficient_index..coefficient_index + 1]
                .copy_from_host_async((&[one]).into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
            ntt(
                &workspace.coeffs[..lde_size],
                NTTDir::kForward,
                &device_ntt_config(stream),
                &mut workspace.weights[..lde_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let mut cfg = VecOpsConfig::default();
            cfg.stream_handle = **stream;
            cfg.is_async = true;
            if let Some(shift) = shift {
                let neg_shift = ark_to_icicle(&(-shift))?;
                workspace.coefficient[..1]
                    .copy_from_host_async((&[neg_shift]).into_slice(), stream)
                    .map_err(|_| Error::GpuBackendFailed)?;
                scalar_add(
                    &workspace.coefficient[..1],
                    &workspace.weights[..lde_size],
                    &mut workspace.term[..lde_size],
                    &cfg,
                )
                .map_err(|_| Error::GpuBackendFailed)?;
                std::mem::swap(&mut workspace.weights, &mut workspace.term);
            }
            inv_scalars(
                &workspace.weights[..lde_size],
                &mut workspace.term[..lde_size],
                &cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            std::mem::swap(&mut workspace.weights, &mut workspace.term);

            if shift.is_none() {
                let mut cached = allocate_device_vec(lde_size)?;
                cached[..lde_size]
                    .copy(&workspace.weights[..lde_size])
                    .map_err(|_| Error::GpuBackendFailed)?;
                Some(cached)
            } else {
                None
            }
        };

        if let Some(values) = cached_weights {
            context.cached_unshifted_inverse_weights = Some(CachedDeviceFields {
                len: lde_size,
                values,
            });
        }
        Ok(())
    }

    fn prepare_ood_quotient_batch_bn254<F: PrimeField + FftField>(
        original_lde_columns: &[&[ArkBn254Fr]],
        claimed_values: &[ArkBn254Fr],
        zeta: ArkBn254Fr,
        degree_bound: usize,
        blowup: usize,
        column_degree_bounds: &[usize],
        context: &mut GpuResidentFriContext,
    ) -> Result<PreparedOodQuotientBatch<F>, Error> {
        let lde_size = degree_bound * blowup;
        prepare_domain_inverse_weights_bn254(lde_size, Some(zeta), context)?;
        let workspace = context.workspace.as_mut().unwrap();
        let stream = &context.stream;

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        let mut quotient_columns = Vec::with_capacity(original_lde_columns.len());
        for (column, claimed_value) in original_lde_columns.iter().zip(claimed_values) {
            let column = column
                .iter()
                .map(ark_to_icicle)
                .collect::<Result<Vec<_>, _>>()?;
            workspace.column[..lde_size]
                .copy_from_host_async(column.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;

            let neg_claimed_value = ark_to_icicle(&(-*claimed_value))?;
            workspace.coefficient[..1]
                .copy_from_host_async((&[neg_claimed_value]).into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;

            let mut quotient = allocate_device_vec(lde_size)?;
            scalar_add(
                &workspace.coefficient[..1],
                &workspace.column[..lde_size],
                &mut workspace.coeffs[..lde_size],
                &cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            mul_scalars(
                &workspace.coeffs[..lde_size],
                &workspace.weights[..lde_size],
                &mut quotient[..lde_size],
                &cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            quotient_columns.push(quotient);
        }

        let group_root = {
            let group_tree = build_ood_group_tree_from_device_columns(
                &quotient_columns,
                lde_size,
                workspace,
                stream,
            )?;
            group_tree.root
        };
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;

        Ok(PreparedOodQuotientBatch {
            quotient_columns,
            group_root,
            row_count: degree_bound,
            blowup,
            quotient_degree_bounds: column_degree_bounds.to_vec(),
            _field: PhantomData,
        })
    }

    pub fn try_prepare_specialized_usc_batch<F: PrimeField + FftField>(
        existing_lde_column: &[F],
        claimed_sum: F,
        degree_bound: usize,
        blowup: usize,
        p_degree_bound: usize,
    ) -> Result<Option<PreparedSpecializedUscBatch<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        super::validate_low_degree_lde(existing_lde_column, degree_bound, blowup)?;
        if p_degree_bound == 0 || p_degree_bound > degree_bound {
            return Err(Error::LowDegreeCheckFailed);
        }

        init_runtime()?;
        let existing_lde_column = unsafe { cast_ark_bn254_slice(existing_lde_column) };
        let claimed_sum = unsafe { cast_ark_bn254_value(&claimed_sum) };
        let prepared = with_context(|context| {
            prepare_specialized_usc_batch_bn254::<F>(
                existing_lde_column,
                claimed_sum,
                degree_bound,
                blowup,
                p_degree_bound,
                context,
            )
        })?;
        Ok(Some(prepared))
    }

    fn prepare_ood_quotient_batch_from_device_columns_bn254<F: PrimeField + FftField>(
        original_lde_columns: &[&DeviceVec<IcicleBn254Fr>],
        claimed_values: &[ArkBn254Fr],
        zeta: ArkBn254Fr,
        degree_bound: usize,
        blowup: usize,
        column_degree_bounds: &[usize],
        context: &mut GpuResidentFriContext,
    ) -> Result<PreparedOodQuotientBatch<F>, Error> {
        let lde_size = degree_bound * blowup;
        prepare_domain_inverse_weights_bn254(lde_size, Some(zeta), context)?;
        let workspace = context.workspace.as_mut().unwrap();
        let stream = &context.stream;

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        let mut quotient_columns = Vec::with_capacity(original_lde_columns.len());
        for (column, claimed_value) in original_lde_columns.iter().zip(claimed_values) {
            let neg_claimed_value = ark_to_icicle(&(-*claimed_value))?;
            workspace.coefficient[..1]
                .copy_from_host_async((&[neg_claimed_value]).into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;

            let mut quotient = allocate_device_vec(lde_size)?;
            scalar_add(
                &workspace.coefficient[..1],
                &column[..lde_size],
                &mut workspace.coeffs[..lde_size],
                &cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            mul_scalars(
                &workspace.coeffs[..lde_size],
                &workspace.weights[..lde_size],
                &mut quotient[..lde_size],
                &cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            quotient_columns.push(quotient);
        }

        let group_root = {
            let group_tree = build_ood_group_tree_from_device_columns(
                &quotient_columns,
                lde_size,
                workspace,
                stream,
            )?;
            group_tree.root
        };
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;

        Ok(PreparedOodQuotientBatch {
            quotient_columns,
            group_root,
            row_count: degree_bound,
            blowup,
            quotient_degree_bounds: column_degree_bounds.to_vec(),
            _field: PhantomData,
        })
    }

    fn prepare_specialized_usc_batch_bn254<F: PrimeField + FftField>(
        existing_lde_column: &[ArkBn254Fr],
        claimed_sum: ArkBn254Fr,
        degree_bound: usize,
        blowup: usize,
        p_degree_bound: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<PreparedSpecializedUscBatch<F>, Error> {
        let lde_size = degree_bound * blowup;
        let size = ArkBn254Fr::from(degree_bound as u64);
        let size_inverse = size
            .inverse()
            .ok_or(Error::InvalidDomainSize(degree_bound))?;
        let average = claimed_sum * size_inverse;

        prepare_domain_inverse_weights_bn254(lde_size, None, context)?;
        let workspace = context.workspace.as_mut().unwrap();
        let stream = &context.stream;

        let column = existing_lde_column
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        workspace.column[..lde_size]
            .copy_from_host_async(column.into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;

        let neg_average = ark_to_icicle(&(-average))?;
        workspace.coefficient[..1]
            .copy_from_host_async((&[neg_average]).into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        let mut p_column = allocate_device_vec(lde_size)?;
        scalar_add(
            &workspace.coefficient[..1],
            &workspace.column[..lde_size],
            &mut workspace.coeffs[..lde_size],
            &cfg,
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        mul_scalars(
            &workspace.coeffs[..lde_size],
            &workspace.weights[..lde_size],
            &mut p_column[..lde_size],
            &cfg,
        )
        .map_err(|_| Error::GpuBackendFailed)?;

        let p_tree = build_merkle_tree_from_device_fields(&p_column, lde_size, workspace, stream)?;
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;

        Ok(PreparedSpecializedUscBatch {
            p_column,
            p_tree,
            row_count: degree_bound,
            blowup,
            p_degree_bound,
            _field: PhantomData,
        })
    }

    fn prepare_specialized_usc_batch_from_device_column_bn254<F: PrimeField + FftField>(
        existing_lde_column: &DeviceVec<IcicleBn254Fr>,
        claimed_sum: ArkBn254Fr,
        degree_bound: usize,
        blowup: usize,
        p_degree_bound: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<PreparedSpecializedUscBatch<F>, Error> {
        let lde_size = degree_bound * blowup;
        let size = ArkBn254Fr::from(degree_bound as u64);
        let size_inverse = size
            .inverse()
            .ok_or(Error::InvalidDomainSize(degree_bound))?;
        let average = claimed_sum * size_inverse;

        prepare_domain_inverse_weights_bn254(lde_size, None, context)?;
        let workspace = context.workspace.as_mut().unwrap();
        let stream = &context.stream;

        let neg_average = ark_to_icicle(&(-average))?;
        workspace.coefficient[..1]
            .copy_from_host_async((&[neg_average]).into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        let mut p_column = allocate_device_vec(lde_size)?;
        scalar_add(
            &workspace.coefficient[..1],
            &existing_lde_column[..lde_size],
            &mut workspace.coeffs[..lde_size],
            &cfg,
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        mul_scalars(
            &workspace.coeffs[..lde_size],
            &workspace.weights[..lde_size],
            &mut p_column[..lde_size],
            &cfg,
        )
        .map_err(|_| Error::GpuBackendFailed)?;

        let p_tree = build_merkle_tree_from_device_fields(&p_column, lde_size, workspace, stream)?;
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;

        Ok(PreparedSpecializedUscBatch {
            p_column,
            p_tree,
            row_count: degree_bound,
            blowup,
            p_degree_bound,
            _field: PhantomData,
        })
    }

    fn finish_specialized_usc_batch_bn254<F: PrimeField + FftField + Send>(
        batch: PreparedSpecializedUscBatch<F>,
        batching_coefficients: &[ArkBn254Fr],
        batching_coefficients_f: Vec<F>,
        context: &mut GpuResidentFriContext,
    ) -> Result<(super::BatchFriProof<F>, Vec<super::BatchLdeRowOpening<F>>), Error> {
        let PreparedSpecializedUscBatch {
            p_column,
            p_tree,
            row_count,
            blowup,
            p_degree_bound,
            _field,
        } = batch;
        let p_columns = vec![p_column];
        let p_column_refs = p_columns.iter().collect::<Vec<_>>();
        let degree_bounds = vec![p_degree_bound];
        let result = prove_device_batch_combined_low_degree_bn254::<F>(
            &p_column_refs,
            batching_coefficients,
            &degree_bounds,
            row_count,
            blowup,
            context,
        )?;
        let GpuCombinedFriResult {
            proof: combined,
            row_openings,
            combined_lde,
        } = result;
        if !row_openings.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        if combined.root != combined_lde.root() {
            return Err(Error::MerkleRootMismatch);
        }

        let lde_indices = combined
            .queries
            .iter()
            .map(|query| query.index)
            .collect::<Vec<_>>();
        let p_openings = ood_quotient_lde_row_openings_from_device_columns(
            &p_columns,
            &p_tree,
            &combined_lde,
            &lde_indices,
            &context.stream,
        )?;
        let root = p_tree.root;
        let batch = super::BatchFriProof {
            row_root: root,
            row_count,
            column_count: 1,
            column_degree_bounds: degree_bounds,
            commitment_groups: vec![super::BatchCommitmentGroup {
                columns: vec![0],
                root,
            }],
            row_openings: Vec::new(),
            batching_coefficients: batching_coefficients_f,
            combined,
        };
        Ok((batch, p_openings))
    }

    fn finish_ood_quotient_batch_bn254<F: PrimeField + FftField + Send>(
        batch: PreparedOodQuotientBatch<F>,
        batching_coefficients: &[ArkBn254Fr],
        batching_coefficients_f: Vec<F>,
        context: &mut GpuResidentFriContext,
    ) -> Result<(super::BatchFriProof<F>, Vec<super::BatchLdeRowOpening<F>>), Error> {
        let PreparedOodQuotientBatch {
            quotient_columns,
            group_root,
            row_count,
            blowup,
            quotient_degree_bounds,
            _field,
        } = batch;
        let column_count = quotient_columns.len();
        let degree_bound = row_count;
        let quotient_column_refs = quotient_columns.iter().collect::<Vec<_>>();
        gpu_resident_diag(|| {
            format!(
                "finish OOD quotient batch: prove combined low degree start columns={column_count}, row_count={row_count}, blowup={blowup}"
            )
        });
        let result = prove_device_batch_combined_low_degree_bn254::<F>(
            &quotient_column_refs,
            batching_coefficients,
            &quotient_degree_bounds,
            degree_bound,
            blowup,
            context,
        )
        .map_err(|error| {
            gpu_resident_diag(|| {
                format!("finish OOD quotient batch: prove combined low degree failed: {error:?}")
            });
            error
        })?;
        gpu_resident_diag(|| "finish OOD quotient batch: prove combined low degree ok".to_string());
        let GpuCombinedFriResult {
            proof: combined,
            row_openings,
            combined_lde,
        } = result;
        if !row_openings.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        if combined.root != combined_lde.root() {
            return Err(Error::MerkleRootMismatch);
        }

        let lde_indices = combined
            .queries
            .iter()
            .map(|query| query.index)
            .collect::<Vec<_>>();
        gpu_resident_diag(|| {
            format!(
                "finish OOD quotient batch: quotient row openings start queries={}",
                lde_indices.len()
            )
        });
        let combined_openings = match &combined_lde {
            CombinedLdeBackend::Gpu(state) => state
                .open_many_with_stream(&lde_indices, &context.stream)
                .map_err(|error| {
                    gpu_resident_diag(|| {
                        format!("finish OOD quotient batch: combined openings failed: {error:?}")
                    });
                    error
                })?,
            _ => return Err(Error::GpuBackendFailed),
        };
        drop(combined_lde);
        gpu_resident_diag(|| "finish OOD quotient batch: rebuild group tree start".to_string());
        let group_tree = build_ood_group_tree_from_device_columns(
            &quotient_columns,
            degree_bound * blowup,
            context.workspace.as_mut().ok_or(Error::GpuBackendFailed)?,
            &context.stream,
        )
        .map_err(|error| {
            gpu_resident_diag(|| {
                format!("finish OOD quotient batch: rebuild group tree failed: {error:?}")
            });
            error
        })?;
        gpu_resident_diag(|| "finish OOD quotient batch: rebuild group tree ok".to_string());
        if group_tree.root != group_root {
            return Err(Error::MerkleRootMismatch);
        }
        let quotient_openings =
            ood_quotient_lde_row_openings_from_device_columns_with_combined_openings(
                &quotient_columns,
                &group_tree,
                combined_openings,
                &lde_indices,
                &context.stream,
            )
            .map_err(|error| {
                gpu_resident_diag(|| {
                    format!("finish OOD quotient batch: quotient row openings failed: {error:?}")
                });
                error
            })?;
        gpu_resident_diag(|| "finish OOD quotient batch: quotient row openings ok".to_string());
        let root = group_root;
        let quotient_batch = super::BatchFriProof {
            row_root: root,
            row_count,
            column_count,
            column_degree_bounds: quotient_degree_bounds,
            commitment_groups: vec![super::BatchCommitmentGroup {
                columns: (0..column_count).collect(),
                root,
            }],
            row_openings: Vec::new(),
            batching_coefficients: batching_coefficients_f,
            combined,
        };
        Ok((quotient_batch, quotient_openings))
    }

    fn prove_device_batch_combined_low_degree_bn254<F: PrimeField + FftField>(
        lde_columns: &[&DeviceVec<IcicleBn254Fr>],
        batching_coefficients: &[ArkBn254Fr],
        column_degree_bounds: &[usize],
        degree_bound: usize,
        blowup: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<GpuCombinedFriResult<F>, Error> {
        let domain_size = degree_bound * blowup;
        gpu_resident_diag(|| {
            format!(
                "device batch combined LDE: combine columns start columns={}, domain_size={domain_size}, degree_bound={degree_bound}",
                lde_columns.len()
            )
        });
        let initial_values = combine_device_lde_columns_to_device(
            lde_columns,
            domain_size,
            batching_coefficients,
            column_degree_bounds,
            degree_bound,
            context,
        )
        .map_err(|error| {
            gpu_resident_diag(|| {
                format!("device batch combined LDE: combine columns failed: {error:?}")
            });
            error
        })?;
        gpu_resident_diag(|| "device batch combined LDE: combine columns ok".to_string());
        prove_combined_device_low_degree_bn254(
            initial_values,
            domain_size,
            degree_bound,
            blowup,
            &[],
            context,
        )
    }

    fn prove_combined_device_low_degree_bn254<F: PrimeField + FftField>(
        initial_values: DeviceVec<IcicleBn254Fr>,
        domain_size: usize,
        degree_bound: usize,
        blowup: usize,
        opened_indices: &[usize],
        context: &mut GpuResidentFriContext,
    ) -> Result<GpuCombinedFriResult<F>, Error> {
        let initial_tree = build_merkle_tree_from_device_fields(
            &initial_values,
            domain_size,
            context.workspace.as_mut().unwrap(),
            &context.stream,
        )?;
        prove_combined_device_low_degree_bn254_with_initial_tree(
            initial_values,
            domain_size,
            degree_bound,
            blowup,
            opened_indices,
            initial_tree,
            context,
        )
    }

    fn prove_combined_device_low_degree_bn254_with_initial_tree<F: PrimeField + FftField>(
        initial_values: DeviceVec<IcicleBn254Fr>,
        domain_size: usize,
        degree_bound: usize,
        blowup: usize,
        opened_indices: &[usize],
        initial_tree: GpuMerkleTree,
        context: &mut GpuResidentFriContext,
    ) -> Result<GpuCombinedFriResult<F>, Error> {
        let mut roots = vec![initial_tree.root];
        let mut transcript =
            super::fri_transcript::<F>(roots[0], degree_bound, blowup, domain_size);
        let mut layers = vec![GpuFriLayer {
            values: initial_values,
            len: domain_size,
            tree: initial_tree,
        }];

        let mut current_degree_bound = degree_bound;
        while current_degree_bound > 1 {
            let alpha = transcript.challenge::<F>(b"fold-alpha");
            let alpha = unsafe { cast_ark_bn254_value(&alpha) };
            let (next_values, next_len) = {
                let current = layers.last().unwrap();
                (
                    fold_layer_on_device(&current.values, current.len, alpha, context)?,
                    current.len / 2,
                )
            };
            let next_tree = build_merkle_tree_from_device_fields(
                &next_values,
                next_len,
                context.workspace.as_mut().unwrap(),
                &context.stream,
            )?;
            let root = next_tree.root;
            transcript.absorb_bytes(b"fold-root", &root);
            roots.push(root);
            layers.push(GpuFriLayer {
                values: next_values,
                len: next_len,
                tree: next_tree,
            });
            current_degree_bound = (current_degree_bound + 1) / 2;
        }

        let final_layer = layers.last().unwrap();
        let final_evals = unsafe {
            cast_bn254_vec(copy_device_fields(
                &final_layer.values,
                final_layer.len,
                &context.stream,
            )?)
        };
        let row_openings =
            open_initial_rows::<F>(&layers[0], blowup, opened_indices, &context.stream)?;
        let queries = sample_gpu_queries::<F>(&mut transcript, &layers, &context.stream)?;
        let first_layer = layers.remove(0);
        let combined_lde = CombinedLdeBackend::gpu(GpuPreparedCombinedLde::new(
            first_layer.values,
            first_layer.len,
            first_layer.tree,
        ));

        Ok(GpuCombinedFriResult {
            proof: FriProof {
                root: roots[0],
                roots,
                final_evals,
                queries,
                degree_bound,
                blowup,
                domain_size,
            },
            row_openings,
            combined_lde,
        })
    }

    fn combine_device_lde_columns_to_device(
        lde_columns: &[&DeviceVec<IcicleBn254Fr>],
        count: usize,
        batching_coefficients: &[ArkBn254Fr],
        column_degree_bounds: &[usize],
        degree_bound: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        ensure_workspace_capacity(&mut context.workspace, count)?;
        let workspace = context.workspace.as_mut().unwrap();
        let stream = &context.stream;

        let coefficients = batching_coefficients
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        let weight_columns = normalized_weight_columns(
            count,
            degree_bound,
            batching_coefficients,
            column_degree_bounds,
        )?;

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        let mut combined = allocate_device_vec(count)?;
        let mut next = allocate_device_vec(count)?;

        for (column_index, column) in lde_columns.iter().enumerate() {
            let target_is_combined = column_index == 0;
            if let Some(weights) = &weight_columns[column_index] {
                workspace.weights[..count]
                    .copy_from_host_async(weights.into_slice(), stream)
                    .map_err(|_| Error::GpuBackendFailed)?;
                if target_is_combined {
                    mul_scalars(
                        &column[..count],
                        &workspace.weights[..count],
                        &mut combined[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                } else {
                    mul_scalars(
                        &column[..count],
                        &workspace.weights[..count],
                        &mut workspace.term[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                }
            } else {
                workspace.coefficient[..1]
                    .copy_from_host_async(
                        (&coefficients[column_index..column_index + 1]).into_slice(),
                        stream,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                if target_is_combined {
                    scalar_mul(
                        &workspace.coefficient[..1],
                        &column[..count],
                        &mut combined[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                } else {
                    scalar_mul(
                        &workspace.coefficient[..1],
                        &column[..count],
                        &mut workspace.term[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                }
            }

            if !target_is_combined {
                add_scalars(
                    &combined[..count],
                    &workspace.term[..count],
                    &mut next[..count],
                    &cfg,
                )
                .map_err(|_| Error::GpuBackendFailed)?;
                std::mem::swap(&mut combined, &mut next);
            }
        }

        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        Ok(combined)
    }

    fn ood_quotient_lde_row_openings_from_device_columns<F: FftField + Send>(
        quotient_columns: &[DeviceVec<IcicleBn254Fr>],
        group_tree: &GpuMerkleTree,
        combined_lde: &CombinedLdeBackend<F>,
        lde_indices: &[usize],
        stream: &IcicleStream,
    ) -> Result<Vec<super::BatchLdeRowOpening<F>>, Error> {
        if lde_indices.is_empty() {
            return Ok(Vec::new());
        }
        let lde_size = group_tree.counts[0];
        if lde_indices.iter().any(|index| *index >= lde_size) {
            return Err(Error::InvalidDomainSize(lde_size));
        }

        let quotient_column_refs = quotient_columns.iter().collect::<Vec<_>>();
        let column_values =
            copy_device_field_columns_indices(&quotient_column_refs, lde_indices, stream)?;
        let group_paths = group_tree.open_many(lde_indices, stream)?;
        let combined_openings = match combined_lde {
            CombinedLdeBackend::Gpu(state) => state.open_many_with_stream(lde_indices, stream)?,
            _ => return Err(Error::GpuBackendFailed),
        };

        lde_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(position, lde_index)| {
                let row = column_values
                    .iter()
                    .map(|values| unsafe { cast_bn254_value_to_f(values[position]) })
                    .collect::<Vec<_>>();
                let (combined_lde_value, combined_lde_path) = &combined_openings[position];
                Ok(super::BatchLdeRowOpening {
                    lde_index,
                    row,
                    group_paths: vec![group_paths[position].clone()],
                    combined_lde_value: *combined_lde_value,
                    combined_lde_path: combined_lde_path.clone(),
                })
            })
            .collect()
    }

    fn ood_quotient_lde_row_openings_from_device_columns_with_combined_openings<
        F: FftField + Send,
    >(
        quotient_columns: &[DeviceVec<IcicleBn254Fr>],
        group_tree: &GpuMerkleTree,
        combined_openings: Vec<(F, MerklePath)>,
        lde_indices: &[usize],
        stream: &IcicleStream,
    ) -> Result<Vec<super::BatchLdeRowOpening<F>>, Error> {
        if lde_indices.is_empty() {
            return Ok(Vec::new());
        }
        let lde_size = group_tree.counts[0];
        if lde_indices.iter().any(|index| *index >= lde_size) {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        if combined_openings.len() != lde_indices.len() {
            return Err(Error::LowDegreeCheckFailed);
        }

        let quotient_column_refs = quotient_columns.iter().collect::<Vec<_>>();
        let column_values =
            copy_device_field_columns_indices(&quotient_column_refs, lde_indices, stream)?;
        let group_paths = group_tree.open_many(lde_indices, stream)?;

        lde_indices
            .iter()
            .copied()
            .enumerate()
            .zip(combined_openings)
            .map(
                |((position, lde_index), (combined_lde_value, combined_lde_path))| {
                    let row = column_values
                        .iter()
                        .map(|values| unsafe { cast_bn254_value_to_f(values[position]) })
                        .collect::<Vec<_>>();
                    Ok(super::BatchLdeRowOpening {
                        lde_index,
                        row,
                        group_paths: vec![group_paths[position].clone()],
                        combined_lde_value,
                        combined_lde_path,
                    })
                },
            )
            .collect()
    }

    fn build_ood_group_tree_from_device_columns(
        columns: &[DeviceVec<IcicleBn254Fr>],
        leaf_count: usize,
        workspace: &mut GpuResidentFriWorkspace,
        stream: &IcicleStream,
    ) -> Result<GpuMerkleTree, Error> {
        if leaf_count == 0 || !leaf_count.is_power_of_two() {
            return Err(Error::InvalidDomainSize(leaf_count));
        }
        ensure_workspace_has_capacity(workspace, leaf_count)?;
        let leaf_preimage_bytes = match columns.len() {
            3 => ROW3_LEAF_PREIMAGE_BYTES,
            5 => ROW5_LEAF_PREIMAGE_BYTES,
            _ => return Err(Error::LowDegreeCheckFailed),
        };
        let leaf_preimage_len = leaf_count * leaf_preimage_bytes;
        let mut leaf_preimages = allocate_device_bytes(leaf_preimage_len)?;
        match columns.len() {
            3 => launch_row3_leaf_preimages(
                columns,
                &mut leaf_preimages[..leaf_preimage_len],
                leaf_count,
                stream,
            )?,
            5 => launch_row5_leaf_preimages(
                columns,
                &mut leaf_preimages[..leaf_preimage_len],
                leaf_count,
                stream,
            )?,
            _ => return Err(Error::LowDegreeCheckFailed),
        }
        let leaf_hashes = hash_device_chunks(
            &leaf_preimages,
            leaf_preimage_len,
            leaf_preimage_bytes,
            stream,
        )?;
        build_merkle_tree_from_leaf_hashes(leaf_hashes, leaf_count, workspace, stream)
    }

    fn build_merkle_tree_from_leaf_hashes(
        leaf_hashes: DeviceVec<u8>,
        leaf_count: usize,
        workspace: &mut GpuResidentFriWorkspace,
        stream: &IcicleStream,
    ) -> Result<GpuMerkleTree, Error> {
        let mut layers = vec![leaf_hashes];
        let mut counts = vec![leaf_count];
        let mut current_count = leaf_count;

        while current_count > 1 {
            let pair_count = current_count / 2;
            let next_hashes = {
                let previous = layers.last().unwrap();
                launch_node_preimages(
                    &previous[..current_count * DIGEST_BYTES],
                    &mut workspace.node_preimages[..pair_count * NODE_PREIMAGE_BYTES],
                    pair_count,
                    stream,
                )?;
                hash_device_chunks(
                    &workspace.node_preimages,
                    pair_count * NODE_PREIMAGE_BYTES,
                    NODE_PREIMAGE_BYTES,
                    stream,
                )?
            };
            layers.push(next_hashes);
            counts.push(pair_count);
            current_count = pair_count;
        }

        let root = copy_digest_at(layers.last().unwrap(), 0, stream)?;
        Ok(GpuMerkleTree {
            layers,
            counts,
            root,
        })
    }

    fn hash_device_chunks(
        input: &DeviceVec<u8>,
        input_len: usize,
        preimage_bytes: usize,
        stream: &IcicleStream,
    ) -> Result<DeviceVec<u8>, Error> {
        if preimage_bytes == 0 || input_len % preimage_bytes != 0 {
            return Err(Error::GpuBackendFailed);
        }
        let output_len = (input_len / preimage_bytes) * DIGEST_BYTES;
        let mut output = allocate_device_bytes(output_len)?;
        let mut cfg = HashConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;
        let hasher =
            IcicleKeccak256::new(preimage_bytes as u64).map_err(|_| Error::GpuBackendFailed)?;
        hasher
            .hash(&input[..input_len], &cfg, &mut output[..output_len])
            .map_err(|_| Error::GpuBackendFailed)?;
        Ok(output)
    }

    fn build_group_tree_from_device_column_refs(
        columns: &[&DeviceVec<IcicleBn254Fr>],
        leaf_count: usize,
        workspace: &mut GpuResidentFriWorkspace,
        stream: &IcicleStream,
    ) -> Result<GpuMerkleTree, Error> {
        if columns.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        if leaf_count == 0 || !leaf_count.is_power_of_two() {
            return Err(Error::InvalidDomainSize(leaf_count));
        }
        if columns.iter().any(|column| column.len() < leaf_count) {
            return Err(Error::InvalidDomainSize(leaf_count));
        }
        if columns.len() == 1 {
            return build_merkle_tree_from_device_fields(columns[0], leaf_count, workspace, stream);
        }

        ensure_workspace_has_capacity(workspace, leaf_count)?;
        let leaf_preimage_bytes = match columns.len() {
            2 => ROW2_LEAF_PREIMAGE_BYTES,
            3 => ROW3_LEAF_PREIMAGE_BYTES,
            5 => ROW5_LEAF_PREIMAGE_BYTES,
            _ => return Err(Error::LowDegreeCheckFailed),
        };
        let leaf_preimage_len = leaf_count * leaf_preimage_bytes;
        let mut leaf_preimages = allocate_device_bytes(leaf_preimage_len)?;
        match columns.len() {
            2 => launch_row2_leaf_preimages_refs(
                columns,
                &mut leaf_preimages[..leaf_preimage_len],
                leaf_count,
                stream,
            )?,
            3 => launch_row3_leaf_preimages_refs(
                columns,
                &mut leaf_preimages[..leaf_preimage_len],
                leaf_count,
                stream,
            )?,
            5 => launch_row5_leaf_preimages_refs(
                columns,
                &mut leaf_preimages[..leaf_preimage_len],
                leaf_count,
                stream,
            )?,
            _ => return Err(Error::LowDegreeCheckFailed),
        }
        let leaf_hashes = hash_device_chunks(
            &leaf_preimages,
            leaf_preimage_len,
            leaf_preimage_bytes,
            stream,
        )?;
        build_merkle_tree_from_leaf_hashes(leaf_hashes, leaf_count, workspace, stream)
    }

    fn launch_row2_leaf_preimages_refs(
        columns: &[&DeviceVec<IcicleBn254Fr>],
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if columns.len() != 2 || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let column0 = &columns[0][..count];
        let column1 = &columns[1][..count];
        if !column0.is_on_device() || !column1.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_row2_leaf_preimages_cuda(
                column0.as_ptr() as *const c_void,
                column1.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_row3_leaf_preimages_refs(
        columns: &[&DeviceVec<IcicleBn254Fr>],
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if columns.len() != 3 || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let column0 = &columns[0][..count];
        let column1 = &columns[1][..count];
        let column2 = &columns[2][..count];
        if !column0.is_on_device() || !column1.is_on_device() || !column2.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_row3_leaf_preimages_cuda(
                column0.as_ptr() as *const c_void,
                column1.as_ptr() as *const c_void,
                column2.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_row5_leaf_preimages_refs(
        columns: &[&DeviceVec<IcicleBn254Fr>],
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if columns.len() != 5 || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let column0 = &columns[0][..count];
        let column1 = &columns[1][..count];
        let column2 = &columns[2][..count];
        let column3 = &columns[3][..count];
        let column4 = &columns[4][..count];
        if !column0.is_on_device()
            || !column1.is_on_device()
            || !column2.is_on_device()
            || !column3.is_on_device()
            || !column4.is_on_device()
        {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_row5_leaf_preimages_cuda(
                column0.as_ptr() as *const c_void,
                column1.as_ptr() as *const c_void,
                column2.as_ptr() as *const c_void,
                column3.as_ptr() as *const c_void,
                column4.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_row3_leaf_preimages(
        columns: &[DeviceVec<IcicleBn254Fr>],
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if columns.len() != 3 || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let column0 = &columns[0][..count];
        let column1 = &columns[1][..count];
        let column2 = &columns[2][..count];
        if !column0.is_on_device() || !column1.is_on_device() || !column2.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_row3_leaf_preimages_cuda(
                column0.as_ptr() as *const c_void,
                column1.as_ptr() as *const c_void,
                column2.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_row5_leaf_preimages(
        columns: &[DeviceVec<IcicleBn254Fr>],
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if columns.len() != 5 || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let column0 = &columns[0][..count];
        let column1 = &columns[1][..count];
        let column2 = &columns[2][..count];
        let column3 = &columns[3][..count];
        let column4 = &columns[4][..count];
        if !column0.is_on_device()
            || !column1.is_on_device()
            || !column2.is_on_device()
            || !column3.is_on_device()
            || !column4.is_on_device()
        {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_row5_leaf_preimages_cuda(
                column0.as_ptr() as *const c_void,
                column1.as_ptr() as *const c_void,
                column2.as_ptr() as *const c_void,
                column3.as_ptr() as *const c_void,
                column4.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    pub(super) fn with_context<T>(
        f: impl FnOnce(&mut GpuResidentFriContext) -> Result<T, Error>,
    ) -> Result<T, Error> {
        GPU_RESIDENT_FRI_CONTEXT.with(|cell| {
            let mut context = cell.try_borrow_mut().map_err(|_| Error::GpuBackendFailed)?;
            if context.is_none() {
                *context = Some(GpuResidentFriContext {
                    stream: IcicleStream::create().map_err(|_| Error::GpuBackendFailed)?,
                    workspace: None,
                    cached_unshifted_inverse_weights: None,
                    batch_combine_workspace: None,
                    fold_workspace: None,
                });
            }
            f(context.as_mut().unwrap())
        })
    }

    fn gpu_resident_diag(message: impl FnOnce() -> String) {
        if env_flag("FRI_LOOKUP_PROVE_DIAGNOSTICS") || env_flag("FRI_SCALE_DIAGNOSTICS") {
            eprintln!("[gpu-resident] {}", message());
        }
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

    pub fn combined_openings_from_store<F: PrimeField + FftField + Send>(
        store: &PreparedLdeColumnStore<F>,
        column_indices: &[usize],
        batching_coefficients: &[F],
        column_degree_bounds: &[usize],
        degree_bound: usize,
        blowup: usize,
        lde_indices: &[usize],
        expected_root: Digest32,
    ) -> Result<Vec<(F, MerklePath)>, Error> {
        if lde_indices.is_empty() {
            return Ok(Vec::new());
        }
        if column_indices.is_empty()
            || column_indices.len() != batching_coefficients.len()
            || column_indices.len() != column_degree_bounds.len()
        {
            return Err(Error::LowDegreeCheckFailed);
        }
        let lde_size = degree_bound * blowup;
        if lde_size != store.len || lde_indices.iter().any(|index| *index >= lde_size) {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        if column_degree_bounds
            .iter()
            .any(|bound| *bound == 0 || *bound > degree_bound)
        {
            return Err(Error::LowDegreeCheckFailed);
        }

        init_runtime()?;
        let coefficients = unsafe { cast_ark_bn254_slice(batching_coefficients) };
        with_context(|context| {
            let columns = store_column_refs(store, column_indices, lde_size)?;
            let combined = combine_device_lde_columns_to_device(
                &columns,
                lde_size,
                coefficients,
                column_degree_bounds,
                degree_bound,
                context,
            )?;
            let tree = build_merkle_tree_from_device_fields(
                &combined,
                lde_size,
                context.workspace.as_mut().ok_or(Error::GpuBackendFailed)?,
                &context.stream,
            )?;
            if tree.root != expected_root {
                return Err(Error::MerkleRootMismatch);
            }
            let values = copy_device_field_indices(&combined, lde_indices, &context.stream)?;
            let paths = tree.open_many(lde_indices, &context.stream)?;
            Ok(values
                .into_iter()
                .zip(paths)
                .map(|(value, path)| (unsafe { cast_bn254_value_to_f(value) }, path))
                .collect())
        })
    }

    pub fn group_paths_from_store<F: PrimeField + FftField + Send>(
        store: &PreparedLdeColumnStore<F>,
        commitment_groups: &[super::BatchCommitmentGroup],
        lde_indices: &[usize],
    ) -> Result<Vec<Vec<MerklePath>>, Error> {
        if commitment_groups.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        if lde_indices.is_empty() {
            return Ok(vec![Vec::new(); commitment_groups.len()]);
        }
        if lde_indices.iter().any(|index| *index >= store.len) {
            return Err(Error::InvalidDomainSize(store.len));
        }

        init_runtime()?;
        with_context(|context| {
            let mut paths_by_group = Vec::with_capacity(commitment_groups.len());
            for group in commitment_groups {
                let columns = store_column_refs(store, &group.columns, store.len)?;
                let tree = build_group_tree_from_device_column_refs(
                    &columns,
                    store.len,
                    context.workspace.as_mut().ok_or(Error::GpuBackendFailed)?,
                    &context.stream,
                )?;
                if tree.root != group.root {
                    return Err(Error::MerkleRootMismatch);
                }
                paths_by_group.push(tree.open_many(lde_indices, &context.stream)?);
            }
            Ok(paths_by_group)
        })
    }

    fn combine_lde_columns_to_device(
        lde_columns: &[&[ArkBn254Fr]],
        batching_coefficients: &[ArkBn254Fr],
        column_degree_bounds: &[usize],
        degree_bound: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        let count = lde_columns[0].len();
        ensure_workspace_capacity(&mut context.workspace, count)?;
        let workspace = context.workspace.as_mut().unwrap();
        let stream = &context.stream;

        let columns = lde_columns
            .iter()
            .map(|column| {
                column
                    .iter()
                    .map(ark_to_icicle)
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let coefficients = batching_coefficients
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        let weight_columns = normalized_weight_columns(
            count,
            degree_bound,
            batching_coefficients,
            column_degree_bounds,
        )?;

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        let mut combined = allocate_device_vec(count)?;
        let mut next = allocate_device_vec(count)?;

        for (column_index, column) in columns.iter().enumerate() {
            workspace.column[..count]
                .copy_from_host_async(column.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;

            let target_is_combined = column_index == 0;
            if let Some(weights) = &weight_columns[column_index] {
                workspace.weights[..count]
                    .copy_from_host_async(weights.into_slice(), stream)
                    .map_err(|_| Error::GpuBackendFailed)?;
                if target_is_combined {
                    mul_scalars(
                        &workspace.column[..count],
                        &workspace.weights[..count],
                        &mut combined[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                } else {
                    mul_scalars(
                        &workspace.column[..count],
                        &workspace.weights[..count],
                        &mut workspace.term[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                }
            } else {
                workspace.coefficient[..1]
                    .copy_from_host_async(
                        (&coefficients[column_index..column_index + 1]).into_slice(),
                        stream,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                if target_is_combined {
                    scalar_mul(
                        &workspace.coefficient[..1],
                        &workspace.column[..count],
                        &mut combined[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                } else {
                    scalar_mul(
                        &workspace.coefficient[..1],
                        &workspace.column[..count],
                        &mut workspace.term[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                }
            }

            if !target_is_combined {
                add_scalars(
                    &combined[..count],
                    &workspace.term[..count],
                    &mut next[..count],
                    &cfg,
                )
                .map_err(|_| Error::GpuBackendFailed)?;
                std::mem::swap(&mut combined, &mut next);
            }
        }

        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        Ok(combined)
    }

    fn fold_layer_on_device(
        values: &DeviceVec<IcicleBn254Fr>,
        domain_size: usize,
        alpha: ArkBn254Fr,
        context: &mut GpuResidentFriContext,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        if !domain_size.is_power_of_two() || domain_size < 2 {
            return Err(Error::InvalidDomainSize(domain_size));
        }

        let half = domain_size / 2;
        ensure_workspace_capacity(&mut context.workspace, half)?;
        let workspace = context.workspace.as_mut().unwrap();
        let stream = &context.stream;
        let two_inv = ArkBn254Fr::from(2u64)
            .inverse()
            .ok_or(Error::LowDegreeCheckFailed)?;
        let coeffs = fold_coefficients(domain_size, alpha, two_inv)?
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        let two_inv = vec![ark_to_icicle(&two_inv)?];
        let mut folded = allocate_device_vec(half)?;

        workspace.coeffs[..half]
            .copy_from_host_async(coeffs.into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;
        workspace.two_inv[..1]
            .copy_from_host_async(two_inv.into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        add_scalars(
            &values[..half],
            &values[half..domain_size],
            &mut workspace.sum[..half],
            &cfg,
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        sub_scalars(
            &values[..half],
            &values[half..domain_size],
            &mut workspace.diff[..half],
            &cfg,
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        scalar_mul(
            &workspace.two_inv[..1],
            &workspace.sum[..half],
            &mut workspace.scaled_sum[..half],
            &cfg,
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        mul_scalars(
            &workspace.diff[..half],
            &workspace.coeffs[..half],
            &mut workspace.scaled_diff[..half],
            &cfg,
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        add_scalars(
            &workspace.scaled_sum[..half],
            &workspace.scaled_diff[..half],
            &mut folded[..half],
            &cfg,
        )
        .map_err(|_| Error::GpuBackendFailed)?;

        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        Ok(folded)
    }

    fn build_merkle_tree_from_device_fields(
        fields: &DeviceVec<IcicleBn254Fr>,
        leaf_count: usize,
        workspace: &mut GpuResidentFriWorkspace,
        stream: &IcicleStream,
    ) -> Result<GpuMerkleTree, Error> {
        if leaf_count == 0 || !leaf_count.is_power_of_two() {
            return Err(Error::InvalidDomainSize(leaf_count));
        }
        ensure_workspace_has_capacity(workspace, leaf_count)?;

        let mut cfg = HashConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        gpu_fold_diag(|| "build fold merkle: create keccak hashers".to_string());
        let leaf_hasher = IcicleKeccak256::new(LEAF_PREIMAGE_BYTES as u64)
            .map_err(|_| Error::GpuBackendFailed)?;
        let node_hasher = IcicleKeccak256::new(NODE_PREIMAGE_BYTES as u64)
            .map_err(|_| Error::GpuBackendFailed)?;

        launch_leaf_preimages(
            &fields[..leaf_count],
            &mut workspace.leaf_preimages[..leaf_count * LEAF_PREIMAGE_BYTES],
            leaf_count,
            stream,
        )?;
        let mut layers = Vec::new();
        let mut counts = Vec::new();
        let mut leaf_hashes = allocate_device_bytes(leaf_count * DIGEST_BYTES)?;
        gpu_fold_diag(|| "build fold merkle: hash leaves".to_string());
        leaf_hasher
            .hash(
                &workspace.leaf_preimages[..leaf_count * LEAF_PREIMAGE_BYTES],
                &cfg,
                &mut leaf_hashes[..leaf_count * DIGEST_BYTES],
            )
            .map_err(|_| Error::GpuBackendFailed)?;
        layers.push(leaf_hashes);
        counts.push(leaf_count);

        let mut current_count = leaf_count;
        while current_count > 1 {
            let pair_count = current_count / 2;
            let mut next_hashes = allocate_device_bytes(pair_count * DIGEST_BYTES)?;
            {
                let previous = layers.last().unwrap();
                gpu_fold_diag(|| {
                    format!(
                        "build fold merkle: node preimages current_count={current_count}, pair_count={pair_count}"
                    )
                });
                launch_node_preimages(
                    &previous[..current_count * DIGEST_BYTES],
                    &mut workspace.node_preimages[..pair_count * NODE_PREIMAGE_BYTES],
                    pair_count,
                    stream,
                )?;
                gpu_fold_diag(|| {
                    format!(
                        "build fold merkle: hash nodes current_count={current_count}, pair_count={pair_count}"
                    )
                });
                node_hasher
                    .hash(
                        &workspace.node_preimages[..pair_count * NODE_PREIMAGE_BYTES],
                        &cfg,
                        &mut next_hashes[..pair_count * DIGEST_BYTES],
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
            }
            layers.push(next_hashes);
            counts.push(pair_count);
            current_count = pair_count;
        }

        let root = copy_digest_at(layers.last().unwrap(), 0, stream)?;
        Ok(GpuMerkleTree {
            layers,
            counts,
            root,
        })
    }

    fn sample_gpu_queries<F: PrimeField + FftField>(
        transcript: &mut super::Transcript,
        layers: &[GpuFriLayer],
        stream: &IcicleStream,
    ) -> Result<Vec<FriQuery<F>>, Error> {
        let query_count = super::query_count(layers[0].len);
        let top_indices = (0..query_count)
            .map(|_| transcript.challenge_index(b"fri-query", layers[0].len))
            .collect::<Vec<_>>();
        let mut queries = top_indices
            .iter()
            .copied()
            .map(|index| FriQuery {
                index,
                openings: Vec::with_capacity(layers.len().saturating_sub(1)),
            })
            .collect::<Vec<_>>();
        let mut current_indices = top_indices;

        for layer in layers.iter().take(layers.len() - 1) {
            let half = layer.len / 2;
            let sibling_indices = current_indices
                .iter()
                .copied()
                .map(|index| {
                    if index >= layer.len {
                        return Err(Error::InvalidDomainSize(index));
                    }
                    Ok(if index < half {
                        index + half
                    } else {
                        index - half
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let next_indices = current_indices
                .iter()
                .copied()
                .map(|index| index % half)
                .collect::<Vec<_>>();
            let query_count = current_indices.len();
            let mut opening_indices = Vec::with_capacity(query_count * 2);
            opening_indices.extend_from_slice(&current_indices);
            opening_indices.extend_from_slice(&sibling_indices);

            let mut values = copy_device_field_indices(&layer.values, &opening_indices, stream)?;
            let mut paths = layer.tree.open_many(&opening_indices, stream)?;
            if values.len() != query_count * 2 || paths.len() != query_count * 2 {
                return Err(Error::LowDegreeCheckFailed);
            }
            let sibling_values = values.split_off(query_count);
            let sibling_paths = paths.split_off(query_count);

            for (position, query) in queries.iter_mut().enumerate() {
                let sibling_path = super::truncate_top_sibling(sibling_paths[position].clone())?;
                query.openings.push(FriLayerOpening {
                    index: current_indices[position],
                    value: unsafe { cast_bn254_value_to_f(values[position]) },
                    sibling_value: unsafe { cast_bn254_value_to_f(sibling_values[position]) },
                    path: paths[position].clone(),
                    sibling_path,
                });
            }
            current_indices = next_indices;
        }
        Ok(queries)
    }

    fn open_initial_rows<F: FftField>(
        layer: &GpuFriLayer,
        blowup: usize,
        opened_indices: &[usize],
        stream: &IcicleStream,
    ) -> Result<Vec<(F, MerklePath)>, Error> {
        if opened_indices.is_empty() {
            return Ok(Vec::new());
        }
        let row_count = layer.len / blowup;
        let lde_indices = opened_indices
            .iter()
            .copied()
            .map(|index| {
                if index >= row_count {
                    return Err(Error::InvalidDomainSize(index));
                }
                Ok(index * blowup)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let values = copy_device_field_indices(&layer.values, &lde_indices, stream)?;
        let paths = layer.tree.open_many(&lde_indices, stream)?;
        Ok(values
            .into_iter()
            .zip(paths)
            .map(|(value, path)| (unsafe { cast_bn254_value_to_f(value) }, path))
            .collect())
    }

    fn normalized_weight_columns(
        lde_size: usize,
        degree_bound: usize,
        batching_coefficients: &[ArkBn254Fr],
        column_degree_bounds: &[usize],
    ) -> Result<Vec<Option<Vec<IcicleBn254Fr>>>, Error> {
        let domain = GeneralEvaluationDomain::<ArkBn254Fr>::new(lde_size)
            .ok_or(Error::InvalidDomainSize(lde_size))?;
        let root = domain.element(1);
        batching_coefficients
            .iter()
            .zip(column_degree_bounds)
            .map(|(coefficient, column_bound)| {
                let exponent = degree_bound - *column_bound;
                if exponent == 0 {
                    return Ok(None);
                }

                let step = root.pow(&[exponent as u64]);
                let mut power = ArkBn254Fr::from(1u64);
                let mut weights = Vec::with_capacity(lde_size);
                for _ in 0..lde_size {
                    weights.push(ark_to_icicle(&(*coefficient * power))?);
                    power *= step;
                }
                Ok(Some(weights))
            })
            .collect()
    }

    fn fold_coefficients(
        domain_size: usize,
        alpha: ArkBn254Fr,
        two_inv: ArkBn254Fr,
    ) -> Result<Vec<ArkBn254Fr>, Error> {
        let domain = GeneralEvaluationDomain::<ArkBn254Fr>::new(domain_size)
            .ok_or(Error::InvalidDomainSize(domain_size))?;
        let omega_inv = domain
            .element(1)
            .inverse()
            .ok_or(Error::LowDegreeCheckFailed)?;
        let alpha_two_inv = alpha * two_inv;
        let mut x_inv = ArkBn254Fr::from(1u64);
        let mut coeffs = Vec::with_capacity(domain_size / 2);

        for _ in 0..domain_size / 2 {
            coeffs.push(alpha_two_inv * x_inv);
            x_inv *= omega_inv;
        }

        Ok(coeffs)
    }

    fn launch_leaf_preimages(
        fields: &(impl HostOrDeviceSlice<IcicleBn254Fr> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if !fields.is_on_device() || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_leaf_preimages_cuda(
                fields.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_node_preimages(
        digests: &(impl HostOrDeviceSlice<u8> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        pair_count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if !digests.is_on_device() || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_merkle_node_preimages_cuda(
                digests.as_ptr(),
                output.as_mut_ptr(),
                pair_count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_gather_fields(
        fields: &DeviceVec<IcicleBn254Fr>,
        indices: &DeviceVec<usize>,
        output: &mut DeviceVec<IcicleBn254Fr>,
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if count == 0 || count > indices.len() || count > output.len() {
            return Err(Error::InvalidDomainSize(count));
        }
        let status = unsafe {
            fri_lookup_bn254_gather_fields_cuda(
                fields.as_ptr() as *const c_void,
                indices.as_ptr(),
                output.as_mut_ptr() as *mut c_void,
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_gather_row_fields(
        columns: &[&DeviceVec<IcicleBn254Fr>],
        indices: &DeviceVec<usize>,
        output: &mut DeviceVec<IcicleBn254Fr>,
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        let width = columns.len();
        if !(2..=5).contains(&width)
            || count == 0
            || count > indices.len()
            || count * width > output.len()
        {
            return Err(Error::InvalidDomainSize(count));
        }
        let mut column_ptrs = [std::ptr::null::<c_void>(); 5];
        for (index, column) in columns.iter().enumerate() {
            // The kernel only reads these device pointers during this launch, and
            // the backing DeviceVecs outlive the stream synchronization below.
            column_ptrs[index] = unsafe { column.as_ptr() } as *const c_void;
        }
        let status = unsafe {
            fri_lookup_bn254_gather_row_fields_cuda(
                column_ptrs[0],
                column_ptrs[1],
                column_ptrs[2],
                column_ptrs[3],
                column_ptrs[4],
                indices.as_ptr(),
                output.as_mut_ptr() as *mut c_void,
                count,
                width,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_gather_digests(
        digests: &DeviceVec<u8>,
        indices: &DeviceVec<usize>,
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if count == 0 || count > indices.len() || !output.is_on_device() {
            return Err(Error::InvalidDomainSize(count));
        }
        let status = unsafe {
            fri_lookup_gather_digests_cuda(
                digests.as_ptr(),
                indices.as_ptr(),
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn copy_digest_at(
        digests: &DeviceVec<u8>,
        index: usize,
        stream: &IcicleStream,
    ) -> Result<Digest32, Error> {
        Ok(copy_digest_indices(digests, &[index], stream)?.remove(0))
    }

    fn copy_digest_indices(
        digests: &DeviceVec<u8>,
        indices: &[usize],
        stream: &IcicleStream,
    ) -> Result<Vec<Digest32>, Error> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        let mut device_indices = allocate_device_indices(indices.len())?;
        device_indices[..indices.len()]
            .copy_from_host_async(indices.into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;
        let mut gathered = allocate_device_bytes(indices.len() * DIGEST_BYTES)?;
        launch_gather_digests(
            digests,
            &device_indices,
            &mut gathered[..indices.len() * DIGEST_BYTES],
            indices.len(),
            stream,
        )?;
        let mut bytes = vec![0u8; indices.len() * DIGEST_BYTES];
        gathered[..indices.len() * DIGEST_BYTES]
            .copy_to_host_async(bytes.into_slice_mut(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        Ok(bytes_to_digests(&bytes))
    }

    fn bytes_to_digests(bytes: &[u8]) -> Vec<Digest32> {
        bytes
            .chunks_exact(DIGEST_BYTES)
            .map(|chunk| {
                let mut digest = [0u8; DIGEST_BYTES];
                digest.copy_from_slice(chunk);
                digest
            })
            .collect()
    }

    fn copy_device_field_columns_indices(
        columns: &[&DeviceVec<IcicleBn254Fr>],
        indices: &[usize],
        stream: &IcicleStream,
    ) -> Result<Vec<Vec<ArkBn254Fr>>, Error> {
        if columns.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        if indices.is_empty() {
            return Ok(vec![Vec::new(); columns.len()]);
        }

        let mut device_indices = allocate_device_indices(indices.len())?;
        device_indices[..indices.len()]
            .copy_from_host_async(indices.into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;

        if (2..=5).contains(&columns.len()) {
            let width = columns.len();
            let mut gathered = allocate_device_vec(indices.len() * width)?;
            launch_gather_row_fields(
                columns,
                &device_indices,
                &mut gathered,
                indices.len(),
                stream,
            )?;
            let mut column_major = vec![IcicleBn254Fr::zero(); indices.len() * width];
            gathered[..indices.len() * width]
                .copy_to_host_async(column_major.into_slice_mut(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
            stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;

            return column_major
                .chunks_exact(indices.len())
                .map(|host| host.iter().map(icicle_to_ark).collect())
                .collect();
        }

        let mut gathered_columns = Vec::with_capacity(columns.len());
        let mut host_columns = Vec::with_capacity(columns.len());
        for column in columns {
            let mut gathered = allocate_device_vec(indices.len())?;
            launch_gather_fields(
                column,
                &device_indices,
                &mut gathered,
                indices.len(),
                stream,
            )?;
            let mut host_column = vec![IcicleBn254Fr::zero(); indices.len()];
            gathered[..indices.len()]
                .copy_to_host_async(host_column.into_slice_mut(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
            gathered_columns.push(gathered);
            host_columns.push(host_column);
        }
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        let _ = gathered_columns;

        host_columns
            .iter()
            .map(|host| host.iter().map(icicle_to_ark).collect())
            .collect()
    }

    fn copy_device_field_indices(
        values: &DeviceVec<IcicleBn254Fr>,
        indices: &[usize],
        stream: &IcicleStream,
    ) -> Result<Vec<ArkBn254Fr>, Error> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        let mut device_indices = allocate_device_indices(indices.len())?;
        device_indices[..indices.len()]
            .copy_from_host_async(indices.into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;
        let mut gathered = allocate_device_vec(indices.len())?;
        launch_gather_fields(
            values,
            &device_indices,
            &mut gathered,
            indices.len(),
            stream,
        )?;
        let mut host = vec![IcicleBn254Fr::zero(); indices.len()];
        gathered[..indices.len()]
            .copy_to_host_async(host.into_slice_mut(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        host.iter().map(icicle_to_ark).collect()
    }

    fn device_ntt_config(stream: &IcicleStream) -> NTTConfig<IcicleBn254Fr> {
        let mut config = NTTConfig::<IcicleBn254Fr>::default();
        config.stream_handle = **stream;
        config.are_inputs_on_device = true;
        config.are_outputs_on_device = true;
        config.is_async = false;
        config
    }

    fn device_fields_from_host(
        values: &[ArkBn254Fr],
        context: &mut GpuResidentFriContext,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        ensure_workspace_capacity(&mut context.workspace, values.len())?;
        let values = values
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        let mut device_values = allocate_device_vec(values.len())?;
        device_values[..values.len()]
            .copy_from_host_async(values.into_slice(), &context.stream)
            .map_err(|_| Error::GpuBackendFailed)?;
        Ok(device_values)
    }

    fn interpolate_device_values_bn254(
        values: &DeviceVec<IcicleBn254Fr>,
        degree_bound: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        if values.len() < degree_bound || degree_bound == 0 || !degree_bound.is_power_of_two() {
            return Err(Error::InvalidDomainSize(degree_bound));
        }
        let mut coeffs = allocate_device_vec(degree_bound)?;
        ntt(
            &values[..degree_bound],
            NTTDir::kInverse,
            &device_ntt_config(&context.stream),
            &mut coeffs[..degree_bound],
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        Ok(coeffs)
    }

    fn lde_from_device_coeffs_bn254(
        coeffs: &DeviceVec<IcicleBn254Fr>,
        degree_bound: usize,
        lde_size: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        if coeffs.len() < degree_bound
            || degree_bound == 0
            || !degree_bound.is_power_of_two()
            || lde_size < degree_bound
            || !lde_size.is_power_of_two()
        {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        ensure_workspace_capacity(&mut context.workspace, lde_size)?;
        let stream = &context.stream;
        let mut padded = allocate_device_vec(lde_size)?;
        launch_memset_zero_fields(&mut padded[..lde_size], lde_size, stream)?;
        padded[..degree_bound]
            .copy(&coeffs[..degree_bound])
            .map_err(|_| Error::GpuBackendFailed)?;
        let mut lde = allocate_device_vec(lde_size)?;
        ntt(
            &padded[..lde_size],
            NTTDir::kForward,
            &device_ntt_config(stream),
            &mut lde[..lde_size],
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        Ok(lde)
    }

    fn lde_from_coeffs_bn254(
        coeffs: &[ArkBn254Fr],
        degree_bound: usize,
        lde_size: usize,
        context: &mut GpuResidentFriContext,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        if coeffs.len() > degree_bound
            || degree_bound == 0
            || !degree_bound.is_power_of_two()
            || lde_size < degree_bound
            || !lde_size.is_power_of_two()
        {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        ensure_workspace_capacity(&mut context.workspace, lde_size)?;
        let stream = &context.stream;
        let mut padded = vec![IcicleBn254Fr::zero(); lde_size];
        for (target, coeff) in padded.iter_mut().zip(coeffs.iter()) {
            *target = ark_to_icicle(coeff)?;
        }
        let mut device_coeffs = allocate_device_vec(lde_size)?;
        device_coeffs[..lde_size]
            .copy_from_host_async(padded.into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;
        let mut lde = allocate_device_vec(lde_size)?;
        ntt(
            &device_coeffs[..lde_size],
            NTTDir::kForward,
            &device_ntt_config(stream),
            &mut lde[..lde_size],
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        Ok(lde)
    }

    fn copy_gpu_merkle_tree_to_cpu(
        tree: &GpuMerkleTree,
        stream: &IcicleStream,
    ) -> Result<MerkleTree, Error> {
        if tree.layers.len() != tree.counts.len() || tree.layers.is_empty() {
            return Err(Error::GpuBackendFailed);
        }
        let mut host_layers = Vec::with_capacity(tree.layers.len());
        for (&count, layer) in tree.counts.iter().zip(tree.layers.iter()) {
            if count == 0 || layer.len() < count * DIGEST_BYTES {
                return Err(Error::GpuBackendFailed);
            }
            let byte_len = count * DIGEST_BYTES;
            let mut host = vec![0u8; byte_len];
            layer[..byte_len]
                .copy_to_host_async(host.as_mut_slice().into_slice_mut(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
            host_layers.push(host);
        }
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;

        let digest_layers = host_layers
            .into_iter()
            .map(|layer| {
                layer
                    .chunks_exact(DIGEST_BYTES)
                    .map(|chunk| {
                        let mut digest = [0u8; DIGEST_BYTES];
                        digest.copy_from_slice(chunk);
                        digest
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Ok(MerkleTree::from_digest_layers(digest_layers))
    }

    fn copy_device_fields(
        values: &DeviceVec<IcicleBn254Fr>,
        count: usize,
        stream: &IcicleStream,
    ) -> Result<Vec<ArkBn254Fr>, Error> {
        let mut host = vec![IcicleBn254Fr::zero(); count];
        values[..count]
            .copy_to_host_async(host.into_slice_mut(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        host.iter().map(icicle_to_ark).collect()
    }

    fn launch_memset_zero_fields(
        output: &mut (impl HostOrDeviceSlice<IcicleBn254Fr> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if !output.is_on_device() || output.len() < count {
            return Err(Error::GpuBackendFailed);
        }
        let byte_count = count * std::mem::size_of::<IcicleBn254Fr>();
        let status = unsafe {
            fri_lookup_cuda_memset_zero(output.as_mut_ptr() as *mut c_void, byte_count, **stream)
        };
        cuda_status_to_result(status)
    }

    fn cuda_status_to_result(status: i32) -> Result<(), Error> {
        if status == 0 {
            Ok(())
        } else {
            gpu_fold_diag(|| format!("cuda kernel returned status={status}"));
            Err(Error::GpuBackendFailed)
        }
    }

    fn gpu_fold_diag(message: impl FnOnce() -> String) {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if *ENABLED.get_or_init(|| {
            matches!(
                env::var("FRI_LOOKUP_GPU_DIAGNOSTICS")
                    .ok()
                    .map(|value| value.to_ascii_lowercase())
                    .as_deref(),
                Some("1" | "true" | "yes" | "on")
            )
        }) {
            eprintln!("[fri-gpu-fold] {}", message());
        }
    }

    fn ensure_workspace_capacity(
        workspace: &mut Option<GpuResidentFriWorkspace>,
        required_capacity: usize,
    ) -> Result<(), Error> {
        if workspace
            .as_ref()
            .map_or(true, |workspace| workspace.capacity < required_capacity)
        {
            *workspace = Some(GpuResidentFriWorkspace::new(required_capacity)?);
        }
        Ok(())
    }

    fn ensure_workspace_has_capacity(
        workspace: &GpuResidentFriWorkspace,
        required_capacity: usize,
    ) -> Result<(), Error> {
        if workspace.capacity < required_capacity {
            Err(Error::GpuBackendFailed)
        } else {
            Ok(())
        }
    }

    fn allocate_device_vec(size: usize) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        DeviceVec::<IcicleBn254Fr>::device_malloc(size).map_err(|_| Error::GpuBackendFailed)
    }

    fn allocate_device_bytes(size: usize) -> Result<DeviceVec<u8>, Error> {
        DeviceVec::<u8>::device_malloc(size).map_err(|_| Error::GpuBackendFailed)
    }

    fn allocate_device_indices(size: usize) -> Result<DeviceVec<usize>, Error> {
        DeviceVec::<usize>::device_malloc(size).map_err(|_| Error::GpuBackendFailed)
    }

    fn init_runtime() -> Result<(), Error> {
        RUNTIME_INIT
            .get_or_init(|| {
                load_backend_from_env_or_default().map_err(|_| Error::GpuBackendFailed)?;
                let device_type =
                    env::var(ICICLE_DEVICE_ENV).unwrap_or_else(|_| "CUDA".to_string());
                let device_id = env::var(ICICLE_DEVICE_ID_ENV)
                    .ok()
                    .and_then(|value| value.parse::<i32>().ok())
                    .unwrap_or(0);
                set_device(&Device::new(&device_type, device_id))
                    .map_err(|_| Error::GpuBackendFailed)
            })
            .clone()
    }

    pub(super) fn init_ntt_domain(size: usize) -> Result<(), Error> {
        let mutex = NTT_DOMAIN_SIZE.get_or_init(|| Mutex::new(0));
        let mut current_size = mutex.lock().map_err(|_| Error::GpuBackendFailed)?;
        if *current_size >= size {
            return Ok(());
        }

        let rou =
            get_root_of_unity::<IcicleBn254Fr>(size as u64).map_err(|_| Error::GpuBackendFailed)?;
        initialize_domain(rou, &NTTInitDomainConfig::default())
            .map_err(|_| Error::GpuBackendFailed)?;
        *current_size = size;
        Ok(())
    }

    fn ark_to_icicle(value: &ArkBn254Fr) -> Result<IcicleBn254Fr, Error> {
        let mut bytes = Vec::with_capacity(32);
        value
            .into_repr()
            .write_le(&mut bytes)
            .map_err(|_| Error::GpuBackendFailed)?;
        Ok(IcicleBn254Fr::from_bytes_le(&bytes))
    }

    fn icicle_to_ark(value: &IcicleBn254Fr) -> Result<ArkBn254Fr, Error> {
        Ok(ArkBn254Fr::from_le_bytes_mod_order(&value.to_bytes_le()))
    }

    fn is_ark_bn254_fr<F: FftField>() -> bool {
        std::any::type_name::<F>() == std::any::type_name::<ArkBn254Fr>()
    }

    unsafe fn cast_ark_bn254_slice<F: FftField>(values: &[F]) -> &[ArkBn254Fr] {
        std::slice::from_raw_parts(values.as_ptr() as *const ArkBn254Fr, values.len())
    }

    unsafe fn cast_ark_bn254_value<F: FftField>(value: &F) -> ArkBn254Fr {
        *(value as *const F as *const ArkBn254Fr)
    }

    unsafe fn cast_bn254_value_to_f<F: FftField>(value: ArkBn254Fr) -> F {
        *(std::ptr::addr_of!(value) as *const F)
    }

    unsafe fn cast_bn254_vec<F: FftField>(values: Vec<ArkBn254Fr>) -> Vec<F> {
        let mut values = std::mem::ManuallyDrop::new(values);
        Vec::from_raw_parts(
            values.as_mut_ptr() as *mut F,
            values.len(),
            values.capacity(),
        )
    }
}

#[cfg(feature = "icicle-rust")]
mod gpu_batch_merkle_bn254 {
    use crate::{
        error::Error,
        merkle::{Digest32, MerklePath},
    };
    use ark_bn254::Fr as ArkBn254Fr;
    use ark_ff::{BigInteger, FftField, PrimeField, ToBytes};
    use icicle_bn254::curve::ScalarField as IcicleBn254Fr;
    use icicle_core::{
        bignum::BigNum,
        hash::HashConfig,
        ntt::{get_root_of_unity, initialize_domain, ntt, NTTConfig, NTTDir, NTTInitDomainConfig},
        vec_ops::{inv_scalars, mul_scalars, scalar_add, scalar_mul, sub_scalars, VecOpsConfig},
    };
    use icicle_hash::keccak::Keccak256 as IcicleKeccak256;
    use icicle_runtime::{
        memory::{DeviceVec, HostOrDeviceSlice, IntoIcicleSlice, IntoIcicleSliceMut},
        runtime::{load_backend_from_env_or_default, set_device},
        stream::IcicleStream,
        Device,
    };
    use std::{
        env,
        ffi::c_void,
        fmt,
        sync::{Arc, Mutex, OnceLock},
    };

    const ICICLE_DEVICE_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE";
    const ICICLE_DEVICE_ID_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE_ID";
    const DIGEST_BYTES: usize = 32;
    const LEAF_DOMAIN: &[u8] = b"leaf";
    const LEAF_PREIMAGE_BYTES: usize = 36;
    const ROW2_LEAF_PREIMAGE_BYTES: usize = 68;
    const ROW3_LEAF_PREIMAGE_BYTES: usize = 100;
    const ROW5_LEAF_PREIMAGE_BYTES: usize = 164;
    const NODE_PREIMAGE_BYTES: usize = 68;

    static RUNTIME_INIT: OnceLock<Result<(), Error>> = OnceLock::new();
    static NTT_DOMAIN_SIZE: OnceLock<Mutex<usize>> = OnceLock::new();
    static GPU_BUILD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    extern "C" {
        fn fri_lookup_bn254_leaf_preimages_cuda(
            fields: *const c_void,
            output: *mut u8,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_cuda_memset_zero(
            output: *mut c_void,
            byte_count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_bn254_row2_leaf_preimages_cuda(
            column0: *const c_void,
            column1: *const c_void,
            output: *mut u8,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_bn254_row3_leaf_preimages_cuda(
            column0: *const c_void,
            column1: *const c_void,
            column2: *const c_void,
            output: *mut u8,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_bn254_row5_leaf_preimages_cuda(
            column0: *const c_void,
            column1: *const c_void,
            column2: *const c_void,
            column3: *const c_void,
            column4: *const c_void,
            output: *mut u8,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_merkle_node_preimages_cuda(
            digests: *const u8,
            output: *mut u8,
            pair_count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
    }

    pub(super) struct GpuPreparedGroupTree {
        layers: Vec<DeviceVec<u8>>,
        counts: Vec<usize>,
        root: Digest32,
    }

    unsafe impl Send for GpuPreparedGroupTree {}
    unsafe impl Sync for GpuPreparedGroupTree {}

    #[derive(Clone)]
    pub(super) struct GpuCommittedLdeColumns {
        columns: Vec<Arc<DeviceVec<IcicleBn254Fr>>>,
        len: usize,
    }

    unsafe impl Send for GpuCommittedLdeColumns {}
    unsafe impl Sync for GpuCommittedLdeColumns {}

    impl fmt::Debug for GpuCommittedLdeColumns {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("GpuCommittedLdeColumns")
                .field("width", &self.columns.len())
                .field("len", &self.len)
                .finish()
        }
    }

    impl GpuCommittedLdeColumns {
        pub(super) fn columns(&self) -> &[Arc<DeviceVec<IcicleBn254Fr>>] {
            &self.columns
        }

        pub(super) fn len(&self) -> usize {
            self.len
        }

        pub(super) fn width(&self) -> usize {
            self.columns.len()
        }
    }

    pub(super) fn concatenate_committed_lde_columns(
        parts: Vec<GpuCommittedLdeColumns>,
    ) -> Result<GpuCommittedLdeColumns, Error> {
        if parts.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        let len = parts[0].len;
        let mut columns = Vec::new();
        for part in parts {
            if part.len != len {
                return Err(Error::InvalidDomainSize(part.len));
            }
            columns.extend(part.columns);
        }
        Ok(GpuCommittedLdeColumns { columns, len })
    }

    impl GpuPreparedGroupTree {
        pub(super) fn root(&self) -> Digest32 {
            self.root
        }

        #[allow(dead_code)]
        pub(super) fn open(&self, index: usize) -> Result<MerklePath, Error> {
            Ok(self.open_many(&[index])?.remove(0))
        }

        pub(super) fn open_many(&self, indices: &[usize]) -> Result<Vec<MerklePath>, Error> {
            if indices.is_empty() {
                return Ok(Vec::new());
            }

            crate::gpu_lock::with_gpu_lock(|| {
                init_runtime()?;
                with_shared_stream(|stream| self.open_many_with_stream(indices, stream))
            })
        }

        fn open_many_with_stream(
            &self,
            indices: &[usize],
            stream: &IcicleStream,
        ) -> Result<Vec<MerklePath>, Error> {
            for &index in indices {
                if index >= self.counts[0] {
                    return Err(Error::InvalidDomainSize(index));
                }
            }

            let mut paths = indices
                .iter()
                .copied()
                .map(|index| MerklePath {
                    index,
                    siblings: Vec::with_capacity(self.layers.len() - 1),
                })
                .collect::<Vec<_>>();
            let mut current_indices = indices.to_vec();

            for (layer, &count) in self
                .layers
                .iter()
                .zip(&self.counts)
                .take(self.layers.len() - 1)
            {
                let sibling_indices = current_indices
                    .iter()
                    .copied()
                    .map(|index| {
                        if index >= count {
                            return Err(Error::InvalidDomainSize(index));
                        }
                        Ok(if index % 2 == 0 { index + 1 } else { index - 1 })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let siblings = copy_digest_indices(layer, &sibling_indices, stream)?;
                for (path, sibling) in paths.iter_mut().zip(siblings) {
                    path.siblings.push(sibling);
                }
                for index in &mut current_indices {
                    *index /= 2;
                }
            }

            Ok(paths)
        }
    }

    pub fn supports_field<F: FftField>() -> bool {
        is_ark_bn254_fr::<F>()
    }

    fn with_shared_stream<T>(
        f: impl FnOnce(&IcicleStream) -> Result<T, Error>,
    ) -> Result<T, Error> {
        super::gpu_resident_fri_bn254::with_context(|context| {
            let result = f(&context.stream);
            let sync_result = context
                .stream
                .synchronize()
                .map_err(|_| Error::GpuBackendFailed);
            let value = result?;
            sync_result?;
            Ok(value)
        })
    }

    pub(super) struct LookupAuxiliaryGpuOutput<F: FftField> {
        pub table_reciprocal: Vec<F>,
        pub table_reciprocal_coeffs: Vec<F>,
        pub table_accumulator: Vec<F>,
        pub table_accumulator_coeffs: Vec<F>,
        pub table_tree: GpuPreparedGroupTree,
        pub table_lde_columns: GpuCommittedLdeColumns,
        pub query_reciprocal: Vec<F>,
        pub query_reciprocal_coeffs: Vec<F>,
        pub query_tree: GpuPreparedGroupTree,
        pub query_lde_columns: GpuCommittedLdeColumns,
    }

    pub(super) struct ConstraintQuotientGpuOutput<F: FftField> {
        pub coeffs: Vec<F>,
        pub base_evals: Vec<F>,
        pub tree: GpuPreparedGroupTree,
        pub lde_columns: GpuCommittedLdeColumns,
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_prepare_lookup_auxiliary<F: PrimeField + FftField>(
        table_values: &[F],
        multiplicity: &[F],
        query_values: &[F],
        beta: F,
        table_row_count: usize,
        query_row_count: usize,
        blowup: usize,
    ) -> Result<Option<LookupAuxiliaryGpuOutput<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        if table_values.len() != table_row_count
            || multiplicity.len() != table_row_count
            || query_values.len() != query_row_count
        {
            return Err(Error::InvalidDomainSize(table_row_count));
        }
        if table_row_count == 0 || query_row_count == 0 || blowup == 0 {
            return Err(Error::LowDegreeCheckFailed);
        }
        let table_lde_size = table_row_count * blowup;
        let query_lde_size = query_row_count * blowup;
        if !table_lde_size.is_power_of_two() || !query_lde_size.is_power_of_two() {
            return Err(Error::InvalidDomainSize(table_lde_size));
        }

        init_runtime()?;
        let _guard = GPU_BUILD_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| Error::GpuBackendFailed)?;
        init_ntt_domain(table_lde_size.max(query_lde_size))?;
        let prepared = with_shared_stream(|stream| {
            let table_values = unsafe { cast_ark_bn254_slice(table_values) };
            let multiplicity = unsafe { cast_ark_bn254_slice(multiplicity) };
            let query_values = unsafe { cast_ark_bn254_slice(query_values) };
            let beta = unsafe { cast_ark_bn254_value(&beta) };
            let beta_icicle = ark_to_icicle(&beta)?;
            let mut beta_device = allocate_device_vec(1)?;
            beta_device[..1]
                .copy_from_host_async((&[beta_icicle]).into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;

            let mut vec_cfg = VecOpsConfig::default();
            vec_cfg.stream_handle = **stream;
            vec_cfg.is_async = true;
            let ntt_cfg = device_ntt_config(stream);

            let table_values_device = upload_field_column_bn254(table_values, stream)?;
            let multiplicity_device = upload_field_column_bn254(multiplicity, stream)?;
            let query_values_device = upload_field_column_bn254(query_values, stream)?;

            let mut table_denominators = allocate_device_vec(table_row_count)?;
            scalar_add(
                &beta_device[..1],
                &table_values_device[..table_row_count],
                &mut table_denominators[..table_row_count],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            let mut table_reciprocal_device = allocate_device_vec(table_row_count)?;
            inv_scalars(
                &table_denominators[..table_row_count],
                &mut table_reciprocal_device[..table_row_count],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            let mut table_accumulator_device = allocate_device_vec(table_row_count)?;
            mul_scalars(
                &multiplicity_device[..table_row_count],
                &table_reciprocal_device[..table_row_count],
                &mut table_accumulator_device[..table_row_count],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let mut query_denominators = allocate_device_vec(query_row_count)?;
            scalar_add(
                &beta_device[..1],
                &query_values_device[..query_row_count],
                &mut query_denominators[..query_row_count],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            let mut query_reciprocal_device = allocate_device_vec(query_row_count)?;
            inv_scalars(
                &query_denominators[..query_row_count],
                &mut query_reciprocal_device[..query_row_count],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let table_reciprocal_coeffs_device = inverse_ntt_device_column_bn254(
                &table_reciprocal_device,
                table_row_count,
                &ntt_cfg,
            )?;
            let table_accumulator_coeffs_device = inverse_ntt_device_column_bn254(
                &table_accumulator_device,
                table_row_count,
                &ntt_cfg,
            )?;
            let query_reciprocal_coeffs_device = inverse_ntt_device_column_bn254(
                &query_reciprocal_device,
                query_row_count,
                &ntt_cfg,
            )?;

            let table_reciprocal_lde = lde_from_device_coeffs_bn254(
                &table_reciprocal_coeffs_device,
                table_row_count,
                table_lde_size,
                &ntt_cfg,
                stream,
            )?;
            let table_accumulator_lde = lde_from_device_coeffs_bn254(
                &table_accumulator_coeffs_device,
                table_row_count,
                table_lde_size,
                &ntt_cfg,
                stream,
            )?;
            let query_reciprocal_lde = lde_from_device_coeffs_bn254(
                &query_reciprocal_coeffs_device,
                query_row_count,
                query_lde_size,
                &ntt_cfg,
                stream,
            )?;

            let table_lde_columns = GpuCommittedLdeColumns {
                columns: vec![
                    Arc::new(table_reciprocal_lde),
                    Arc::new(table_accumulator_lde),
                ],
                len: table_lde_size,
            };
            let table_group_columns = table_lde_columns
                .columns
                .iter()
                .map(Arc::as_ref)
                .collect::<Vec<_>>();
            let table_tree =
                build_group_tree_from_device_columns(&table_group_columns, table_lde_size, stream)?;

            let query_lde_columns = GpuCommittedLdeColumns {
                columns: vec![Arc::new(query_reciprocal_lde)],
                len: query_lde_size,
            };
            let query_group_columns = query_lde_columns
                .columns
                .iter()
                .map(Arc::as_ref)
                .collect::<Vec<_>>();
            let query_tree =
                build_group_tree_from_device_columns(&query_group_columns, query_lde_size, stream)?;

            let mut host_columns = copy_device_columns_to_field_vecs::<F>(
                &[
                    (&table_reciprocal_device, table_row_count),
                    (&table_reciprocal_coeffs_device, table_row_count),
                    (&table_accumulator_device, table_row_count),
                    (&table_accumulator_coeffs_device, table_row_count),
                    (&query_reciprocal_device, query_row_count),
                    (&query_reciprocal_coeffs_device, query_row_count),
                ],
                stream,
            )?
            .into_iter();
            let table_reciprocal = host_columns.next().ok_or(Error::GpuBackendFailed)?;
            let table_reciprocal_coeffs = host_columns.next().ok_or(Error::GpuBackendFailed)?;
            let table_accumulator = host_columns.next().ok_or(Error::GpuBackendFailed)?;
            let table_accumulator_coeffs = host_columns.next().ok_or(Error::GpuBackendFailed)?;
            let query_reciprocal = host_columns.next().ok_or(Error::GpuBackendFailed)?;
            let query_reciprocal_coeffs = host_columns.next().ok_or(Error::GpuBackendFailed)?;

            Ok(LookupAuxiliaryGpuOutput {
                table_reciprocal,
                table_reciprocal_coeffs,
                table_accumulator,
                table_accumulator_coeffs,
                table_tree,
                table_lde_columns,
                query_reciprocal,
                query_reciprocal_coeffs,
                query_tree,
                query_lde_columns,
            })
        })?;
        Ok(Some(prepared))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_prepare_table_constraint_quotient<F: PrimeField + FftField>(
        table_coeffs: &[F],
        multiplicity_coeffs: &[F],
        reciprocal_coeffs: &[F],
        domain_size: usize,
        blowup: usize,
        eta_t1: F,
        eta_t2: F,
    ) -> Result<Option<ConstraintQuotientGpuOutput<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        if domain_size == 0
            || !domain_size.is_power_of_two()
            || blowup == 0
            || !blowup.is_power_of_two()
        {
            return Err(Error::InvalidDomainSize(domain_size));
        }
        if table_coeffs.len() > domain_size
            || multiplicity_coeffs.len() > domain_size
            || reciprocal_coeffs.len() > domain_size
        {
            return Err(Error::LowDegreeCheckFailed);
        }

        let product_fft_size = domain_size.checked_mul(2).ok_or(Error::GpuBackendFailed)?;
        let lde_size = domain_size
            .checked_mul(blowup)
            .ok_or(Error::GpuBackendFailed)?;
        if !product_fft_size.is_power_of_two() || !lde_size.is_power_of_two() {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        let quotient_len = domain_size.saturating_sub(1).max(1);

        init_runtime()?;
        let _guard = GPU_BUILD_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| Error::GpuBackendFailed)?;
        init_ntt_domain(product_fft_size.max(lde_size))?;
        let prepared = with_shared_stream(|stream| {
            let table_coeffs = unsafe { cast_ark_bn254_slice(table_coeffs) };
            let multiplicity_coeffs = unsafe { cast_ark_bn254_slice(multiplicity_coeffs) };
            let reciprocal_coeffs = unsafe { cast_ark_bn254_slice(reciprocal_coeffs) };
            let eta_t1 = ark_to_icicle(&unsafe { cast_ark_bn254_value(&eta_t1) })?;
            let eta_t2 = ark_to_icicle(&unsafe { cast_ark_bn254_value(&eta_t2) })?;
            let ntt_config = device_ntt_config(stream);
            let table_device = padded_device_coeffs_bn254(table_coeffs, product_fft_size, stream)?;
            let multiplicity_device =
                padded_device_coeffs_bn254(multiplicity_coeffs, product_fft_size, stream)?;
            let reciprocal_device =
                padded_device_coeffs_bn254(reciprocal_coeffs, product_fft_size, stream)?;

            let mut table_evals = allocate_device_vec(product_fft_size)?;
            let mut multiplicity_evals = allocate_device_vec(product_fft_size)?;
            let mut reciprocal_evals = allocate_device_vec(product_fft_size)?;
            ntt(
                &table_device[..product_fft_size],
                NTTDir::kForward,
                &ntt_config,
                &mut table_evals[..product_fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            ntt(
                &multiplicity_device[..product_fft_size],
                NTTDir::kForward,
                &ntt_config,
                &mut multiplicity_evals[..product_fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            ntt(
                &reciprocal_device[..product_fft_size],
                NTTDir::kForward,
                &ntt_config,
                &mut reciprocal_evals[..product_fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let mut vec_cfg = VecOpsConfig::default();
            vec_cfg.stream_handle = **stream;
            vec_cfg.is_async = true;

            let mut product_evals = allocate_device_vec(product_fft_size)?;
            let mut reciprocal_table_coeffs = allocate_device_vec(product_fft_size)?;
            let mut multiplicity_reciprocal_coeffs = allocate_device_vec(product_fft_size)?;
            mul_scalars(
                &reciprocal_evals[..product_fft_size],
                &table_evals[..product_fft_size],
                &mut product_evals[..product_fft_size],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            ntt(
                &product_evals[..product_fft_size],
                NTTDir::kInverse,
                &ntt_config,
                &mut reciprocal_table_coeffs[..product_fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            mul_scalars(
                &multiplicity_evals[..product_fft_size],
                &reciprocal_evals[..product_fft_size],
                &mut product_evals[..product_fft_size],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            ntt(
                &product_evals[..product_fft_size],
                NTTDir::kInverse,
                &ntt_config,
                &mut multiplicity_reciprocal_coeffs[..product_fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let mut eta_t1_device = allocate_device_vec(1)?;
            let mut eta_t2_device = allocate_device_vec(1)?;
            eta_t1_device[..1]
                .copy_from_host_async((&[eta_t1]).into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
            eta_t2_device[..1]
                .copy_from_host_async((&[eta_t2]).into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;

            let mut scaled_rt = allocate_device_vec(quotient_len)?;
            let mut scaled_mr = allocate_device_vec(quotient_len)?;
            scalar_mul(
                &eta_t1_device[..1],
                &reciprocal_table_coeffs[domain_size..domain_size + quotient_len],
                &mut scaled_rt[..quotient_len],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            scalar_mul(
                &eta_t2_device[..1],
                &multiplicity_reciprocal_coeffs[domain_size..domain_size + quotient_len],
                &mut scaled_mr[..quotient_len],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let mut quotient_coeffs = allocate_device_vec(domain_size)?;
            launch_memset_zero_fields(&mut quotient_coeffs[..domain_size], domain_size, stream)?;
            sub_scalars(
                &scaled_rt[..quotient_len],
                &scaled_mr[..quotient_len],
                &mut quotient_coeffs[..quotient_len],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            finish_constraint_quotient_from_device_coeffs::<F>(
                quotient_coeffs,
                domain_size,
                lde_size,
                &ntt_config,
                stream,
            )
        })?;
        Ok(Some(prepared))
    }

    pub fn try_prepare_query_constraint_quotient<F: PrimeField + FftField>(
        query_coeffs: &[F],
        reciprocal_coeffs: &[F],
        domain_size: usize,
        blowup: usize,
    ) -> Result<Option<ConstraintQuotientGpuOutput<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        if domain_size == 0
            || !domain_size.is_power_of_two()
            || blowup == 0
            || !blowup.is_power_of_two()
        {
            return Err(Error::InvalidDomainSize(domain_size));
        }
        if query_coeffs.len() > domain_size || reciprocal_coeffs.len() > domain_size {
            return Err(Error::LowDegreeCheckFailed);
        }

        let product_fft_size = domain_size.checked_mul(2).ok_or(Error::GpuBackendFailed)?;
        let lde_size = domain_size
            .checked_mul(blowup)
            .ok_or(Error::GpuBackendFailed)?;
        if !product_fft_size.is_power_of_two() || !lde_size.is_power_of_two() {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        let quotient_len = domain_size.saturating_sub(1).max(1);

        init_runtime()?;
        let _guard = GPU_BUILD_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| Error::GpuBackendFailed)?;
        init_ntt_domain(product_fft_size.max(lde_size))?;
        let prepared = with_shared_stream(|stream| {
            let query_coeffs = unsafe { cast_ark_bn254_slice(query_coeffs) };
            let reciprocal_coeffs = unsafe { cast_ark_bn254_slice(reciprocal_coeffs) };
            let ntt_config = device_ntt_config(stream);
            let query_device = padded_device_coeffs_bn254(query_coeffs, product_fft_size, stream)?;
            let reciprocal_device =
                padded_device_coeffs_bn254(reciprocal_coeffs, product_fft_size, stream)?;

            let mut query_evals = allocate_device_vec(product_fft_size)?;
            let mut reciprocal_evals = allocate_device_vec(product_fft_size)?;
            ntt(
                &query_device[..product_fft_size],
                NTTDir::kForward,
                &ntt_config,
                &mut query_evals[..product_fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            ntt(
                &reciprocal_device[..product_fft_size],
                NTTDir::kForward,
                &ntt_config,
                &mut reciprocal_evals[..product_fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let mut vec_cfg = VecOpsConfig::default();
            vec_cfg.stream_handle = **stream;
            vec_cfg.is_async = true;
            let mut product_evals = allocate_device_vec(product_fft_size)?;
            mul_scalars(
                &reciprocal_evals[..product_fft_size],
                &query_evals[..product_fft_size],
                &mut product_evals[..product_fft_size],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            let mut product_coeffs = allocate_device_vec(product_fft_size)?;
            ntt(
                &product_evals[..product_fft_size],
                NTTDir::kInverse,
                &ntt_config,
                &mut product_coeffs[..product_fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let mut quotient_coeffs = allocate_device_vec(domain_size)?;
            launch_memset_zero_fields(&mut quotient_coeffs[..domain_size], domain_size, stream)?;
            quotient_coeffs[..quotient_len]
                .copy(&product_coeffs[domain_size..domain_size + quotient_len])
                .map_err(|_| Error::GpuBackendFailed)?;

            finish_constraint_quotient_from_device_coeffs::<F>(
                quotient_coeffs,
                domain_size,
                lde_size,
                &ntt_config,
                stream,
            )
        })?;
        Ok(Some(prepared))
    }

    pub fn try_product_high_coefficients<F: PrimeField + FftField>(
        left: &[F],
        right: &[F],
        domain_size: usize,
    ) -> Result<Option<Vec<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        if domain_size == 0 || !domain_size.is_power_of_two() {
            return Err(Error::InvalidDomainSize(domain_size));
        }
        if left.len() > domain_size || right.len() > domain_size {
            return Err(Error::LowDegreeCheckFailed);
        }

        let quotient_len = domain_size.saturating_sub(1).max(1);
        if left.is_empty()
            || right.is_empty()
            || left.len().saturating_add(right.len()).saturating_sub(1) <= domain_size
        {
            return Ok(Some(vec![F::zero(); quotient_len]));
        }

        let result_len = left.len() + right.len() - 1;
        let fft_size = result_len.next_power_of_two();
        if fft_size < domain_size + quotient_len || !fft_size.is_power_of_two() {
            return Err(Error::InvalidDomainSize(fft_size));
        }

        init_runtime()?;
        let _guard = GPU_BUILD_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| Error::GpuBackendFailed)?;
        init_ntt_domain(fft_size)?;
        let high_coeffs = with_shared_stream(|stream| {
            let left = unsafe { cast_ark_bn254_slice(left) };
            let right = unsafe { cast_ark_bn254_slice(right) };
            let ntt_config = device_ntt_config(stream);
            let left_coeffs = padded_device_coeffs_bn254(left, fft_size, stream)?;
            let right_coeffs = padded_device_coeffs_bn254(right, fft_size, stream)?;
            let mut left_evals = allocate_device_vec(fft_size)?;
            let mut right_evals = allocate_device_vec(fft_size)?;
            ntt(
                &left_coeffs[..fft_size],
                NTTDir::kForward,
                &ntt_config,
                &mut left_evals[..fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            ntt(
                &right_coeffs[..fft_size],
                NTTDir::kForward,
                &ntt_config,
                &mut right_evals[..fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let mut vec_cfg = VecOpsConfig::default();
            vec_cfg.stream_handle = **stream;
            vec_cfg.is_async = true;
            let mut product_evals = allocate_device_vec(fft_size)?;
            mul_scalars(
                &left_evals[..fft_size],
                &right_evals[..fft_size],
                &mut product_evals[..fft_size],
                &vec_cfg,
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let mut product_coeffs = allocate_device_vec(fft_size)?;
            ntt(
                &product_evals[..fft_size],
                NTTDir::kInverse,
                &ntt_config,
                &mut product_coeffs[..fft_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;

            let mut host = vec![IcicleBn254Fr::zero(); quotient_len];
            product_coeffs[domain_size..domain_size + quotient_len]
                .copy_to_host_async(host.into_slice_mut(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
            stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
            let values = host
                .iter()
                .map(icicle_to_ark)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(unsafe { cast_bn254_vec(values) })
        })?;
        Ok(Some(high_coeffs))
    }

    pub fn try_build_group_trees_with_columns<F: PrimeField + FftField>(
        lde_columns: &[&[F]],
        commitment_groups: &[Vec<usize>],
    ) -> Result<Option<(Vec<GpuPreparedGroupTree>, GpuCommittedLdeColumns)>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        if commitment_groups
            .iter()
            .any(|group| !matches!(group.len(), 1 | 2 | 3 | 5))
        {
            return Ok(None);
        }

        init_runtime()?;
        let _guard = GPU_BUILD_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| Error::GpuBackendFailed)?;
        let prepared = with_shared_stream(|stream| {
            let lde_columns = lde_columns
                .iter()
                .map(|column| unsafe { cast_ark_bn254_slice(column) })
                .collect::<Vec<_>>();
            let committed_columns = upload_committed_lde_columns_bn254(&lde_columns, stream)?;
            let mut trees = Vec::with_capacity(commitment_groups.len());
            for group in commitment_groups {
                let group_columns = group
                    .iter()
                    .map(|&column| {
                        committed_columns
                            .columns
                            .get(column)
                            .map(Arc::as_ref)
                            .ok_or(Error::LowDegreeCheckFailed)
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                trees.push(build_group_tree_from_device_columns(
                    &group_columns,
                    committed_columns.len,
                    stream,
                )?);
            }
            stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
            Ok((trees, committed_columns))
        })?;
        Ok(Some(prepared))
    }

    pub fn try_commit_coeff_columns_as_lde<F: PrimeField + FftField>(
        coeff_columns: &[&[F]],
        degree_bound: usize,
        blowup: usize,
    ) -> Result<Option<(GpuPreparedGroupTree, GpuCommittedLdeColumns)>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }
        if !matches!(coeff_columns.len(), 1 | 2 | 3 | 5) {
            return Ok(None);
        }
        if coeff_columns.is_empty() || degree_bound == 0 || blowup == 0 {
            return Err(Error::LowDegreeCheckFailed);
        }
        if coeff_columns
            .iter()
            .any(|column| column.len() > degree_bound)
        {
            return Err(Error::LowDegreeCheckFailed);
        }
        let lde_size = degree_bound * blowup;
        if !lde_size.is_power_of_two() {
            return Err(Error::InvalidDomainSize(lde_size));
        }

        init_runtime()?;
        let _guard = GPU_BUILD_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| Error::GpuBackendFailed)?;
        init_ntt_domain(lde_size)?;
        let prepared = with_shared_stream(|stream| {
            let coeff_columns = coeff_columns
                .iter()
                .map(|column| unsafe { cast_ark_bn254_slice(column) })
                .collect::<Vec<_>>();
            let committed_columns = lde_commitment_columns_from_coeffs_bn254(
                &coeff_columns,
                degree_bound,
                lde_size,
                stream,
            )?;
            let group_columns = committed_columns
                .columns
                .iter()
                .map(Arc::as_ref)
                .collect::<Vec<_>>();
            let tree = build_group_tree_from_device_columns(&group_columns, lde_size, stream)?;
            stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
            Ok((tree, committed_columns))
        })?;
        Ok(Some(prepared))
    }

    pub fn try_build_group_trees<F: PrimeField + FftField>(
        lde_columns: &[&[F]],
        commitment_groups: &[Vec<usize>],
    ) -> Result<Option<Vec<GpuPreparedGroupTree>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }

        init_runtime()?;
        let _guard = GPU_BUILD_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| Error::GpuBackendFailed)?;
        let trees = with_shared_stream(|stream| {
            let lde_columns = lde_columns
                .iter()
                .map(|column| unsafe { cast_ark_bn254_slice(column) })
                .collect::<Vec<_>>();
            let mut trees = Vec::with_capacity(commitment_groups.len());
            for group in commitment_groups {
                let group_columns = group
                    .iter()
                    .map(|column| lde_columns[*column])
                    .collect::<Vec<_>>();
                trees.push(build_group_tree_bn254(&group_columns, stream)?);
            }
            stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
            Ok(trees)
        })?;
        Ok(Some(trees))
    }

    fn device_ntt_config(stream: &IcicleStream) -> NTTConfig<IcicleBn254Fr> {
        let mut config = NTTConfig::<IcicleBn254Fr>::default();
        config.stream_handle = **stream;
        config.are_inputs_on_device = true;
        config.are_outputs_on_device = true;
        config.is_async = false;
        config
    }

    fn upload_field_column_bn254(
        values: &[ArkBn254Fr],
        stream: &IcicleStream,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        let values = values
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        let mut device_values = allocate_device_vec(values.len())?;
        device_values[..values.len()]
            .copy_from_host_async(values.into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;

        Ok(device_values)
    }

    fn padded_device_coeffs_bn254(
        coeffs: &[ArkBn254Fr],
        len: usize,
        stream: &IcicleStream,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        if coeffs.len() > len || len == 0 || !len.is_power_of_two() {
            return Err(Error::InvalidDomainSize(len));
        }
        let mut padded = allocate_device_vec(len)?;
        launch_memset_zero_fields(&mut padded[..len], len, stream)?;
        if !coeffs.is_empty() {
            let coeffs = coeffs
                .iter()
                .map(ark_to_icicle)
                .collect::<Result<Vec<_>, _>>()?;
            padded[..coeffs.len()]
                .copy_from_host_async(coeffs.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
        }
        Ok(padded)
    }

    fn inverse_ntt_device_column_bn254(
        values: &DeviceVec<IcicleBn254Fr>,
        len: usize,
        ntt_config: &NTTConfig<IcicleBn254Fr>,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        if values.len() < len || len == 0 || !len.is_power_of_two() {
            return Err(Error::InvalidDomainSize(len));
        }
        let mut coeffs = allocate_device_vec(len)?;
        ntt(
            &values[..len],
            NTTDir::kInverse,
            ntt_config,
            &mut coeffs[..len],
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        Ok(coeffs)
    }

    fn lde_from_device_coeffs_bn254(
        coeffs: &DeviceVec<IcicleBn254Fr>,
        degree_bound: usize,
        lde_size: usize,
        ntt_config: &NTTConfig<IcicleBn254Fr>,
        stream: &IcicleStream,
    ) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        if coeffs.len() < degree_bound
            || degree_bound == 0
            || lde_size < degree_bound
            || !lde_size.is_power_of_two()
        {
            return Err(Error::InvalidDomainSize(lde_size));
        }
        let mut padded_coeffs = allocate_device_vec(lde_size)?;
        launch_memset_zero_fields(&mut padded_coeffs[..lde_size], lde_size, stream)?;
        padded_coeffs[..degree_bound]
            .copy(&coeffs[..degree_bound])
            .map_err(|_| Error::GpuBackendFailed)?;
        let mut lde = allocate_device_vec(lde_size)?;
        ntt(
            &padded_coeffs[..lde_size],
            NTTDir::kForward,
            ntt_config,
            &mut lde[..lde_size],
        )
        .map_err(|_| Error::GpuBackendFailed)?;
        Ok(lde)
    }

    fn finish_constraint_quotient_from_device_coeffs<F: FftField>(
        quotient_coeffs: DeviceVec<IcicleBn254Fr>,
        domain_size: usize,
        lde_size: usize,
        ntt_config: &NTTConfig<IcicleBn254Fr>,
        stream: &IcicleStream,
    ) -> Result<ConstraintQuotientGpuOutput<F>, Error> {
        if quotient_coeffs.len() < domain_size || domain_size == 0 || !domain_size.is_power_of_two()
        {
            return Err(Error::InvalidDomainSize(domain_size));
        }
        if lde_size < domain_size || !lde_size.is_power_of_two() {
            return Err(Error::InvalidDomainSize(lde_size));
        }

        let quotient_lde = lde_from_device_coeffs_bn254(
            &quotient_coeffs,
            domain_size,
            lde_size,
            ntt_config,
            stream,
        )?;
        let tree = build_group_tree_from_device_columns(&[&quotient_lde], lde_size, stream)?;
        let mut host_columns =
            copy_device_columns_to_field_vecs::<F>(&[(&quotient_coeffs, domain_size)], stream)?
                .into_iter();
        let coeffs = host_columns.next().ok_or(Error::GpuBackendFailed)?;
        Ok(ConstraintQuotientGpuOutput {
            coeffs,
            base_evals: Vec::new(),
            tree,
            lde_columns: GpuCommittedLdeColumns {
                columns: vec![Arc::new(quotient_lde)],
                len: lde_size,
            },
        })
    }

    fn copy_device_columns_to_field_vecs<F: FftField>(
        columns: &[(&DeviceVec<IcicleBn254Fr>, usize)],
        stream: &IcicleStream,
    ) -> Result<Vec<Vec<F>>, Error> {
        let mut hosts = Vec::with_capacity(columns.len());
        for &(column, len) in columns {
            if column.len() < len {
                return Err(Error::InvalidDomainSize(len));
            }
            hosts.push(vec![IcicleBn254Fr::zero(); len]);
        }

        for (host, (column, len)) in hosts.iter_mut().zip(columns.iter()) {
            column[..*len]
                .copy_to_host_async(host.as_mut_slice().into_slice_mut(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
        }
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;

        hosts
            .into_iter()
            .map(|host| {
                let values = host
                    .iter()
                    .map(icicle_to_ark)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(unsafe { cast_bn254_vec(values) })
            })
            .collect()
    }

    fn upload_committed_lde_columns_bn254(
        lde_columns: &[&[ArkBn254Fr]],
        stream: &IcicleStream,
    ) -> Result<GpuCommittedLdeColumns, Error> {
        if lde_columns.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        let len = lde_columns[0].len();
        if len == 0 || !len.is_power_of_two() {
            return Err(Error::InvalidDomainSize(len));
        }
        if lde_columns.iter().any(|column| column.len() != len) {
            return Err(Error::InvalidDomainSize(len));
        }

        let mut columns = Vec::with_capacity(lde_columns.len());
        for column in lde_columns {
            let column = column
                .iter()
                .map(ark_to_icicle)
                .collect::<Result<Vec<_>, _>>()?;
            let mut device_column = allocate_device_vec(len)?;
            device_column[..len]
                .copy_from_host_async(column.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;
            columns.push(Arc::new(device_column));
        }
        Ok(GpuCommittedLdeColumns { columns, len })
    }

    fn lde_commitment_columns_from_coeffs_bn254(
        coeff_columns: &[&[ArkBn254Fr]],
        degree_bound: usize,
        lde_size: usize,
        stream: &IcicleStream,
    ) -> Result<GpuCommittedLdeColumns, Error> {
        let mut ntt_config = NTTConfig::<IcicleBn254Fr>::default();
        ntt_config.stream_handle = **stream;
        ntt_config.are_inputs_on_device = true;
        ntt_config.are_outputs_on_device = true;
        ntt_config.is_async = false;

        let mut columns = Vec::with_capacity(coeff_columns.len());
        for coeffs in coeff_columns {
            if coeffs.len() > degree_bound {
                return Err(Error::LowDegreeCheckFailed);
            }
            let mut padded_coeffs = allocate_device_vec(lde_size)?;
            launch_memset_zero_fields(&mut padded_coeffs[..lde_size], lde_size, stream)?;
            stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
            if !coeffs.is_empty() {
                let coeffs = coeffs
                    .iter()
                    .map(ark_to_icicle)
                    .collect::<Result<Vec<_>, _>>()?;
                padded_coeffs[..coeffs.len()]
                    .copy_from_host(coeffs.into_slice())
                    .map_err(|_| Error::GpuBackendFailed)?;
            }

            let mut lde = allocate_device_vec(lde_size)?;
            ntt(
                &padded_coeffs[..lde_size],
                NTTDir::kForward,
                &ntt_config,
                &mut lde[..lde_size],
            )
            .map_err(|_| Error::GpuBackendFailed)?;
            columns.push(Arc::new(lde));
        }
        Ok(GpuCommittedLdeColumns {
            columns,
            len: lde_size,
        })
    }

    fn build_group_tree_from_device_columns(
        columns: &[&DeviceVec<IcicleBn254Fr>],
        leaf_count: usize,
        stream: &IcicleStream,
    ) -> Result<GpuPreparedGroupTree, Error> {
        if columns.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        if leaf_count == 0 || !leaf_count.is_power_of_two() {
            return Err(Error::InvalidDomainSize(leaf_count));
        }
        if columns.iter().any(|column| column.len() < leaf_count) {
            return Err(Error::InvalidDomainSize(leaf_count));
        }
        let leaf_preimage_bytes = match columns.len() {
            1 => LEAF_PREIMAGE_BYTES,
            2 => ROW2_LEAF_PREIMAGE_BYTES,
            3 => ROW3_LEAF_PREIMAGE_BYTES,
            5 => ROW5_LEAF_PREIMAGE_BYTES,
            _ => return Err(Error::LowDegreeCheckFailed),
        };
        let leaf_preimage_len = leaf_count * leaf_preimage_bytes;
        let mut leaf_preimages = allocate_device_bytes(leaf_preimage_len)?;
        match columns.len() {
            1 => launch_field_leaf_preimages(
                columns[0],
                &mut leaf_preimages[..leaf_preimage_len],
                leaf_count,
                stream,
            )?,
            2 => launch_row2_leaf_preimages(
                columns,
                &mut leaf_preimages[..leaf_preimage_len],
                leaf_count,
                stream,
            )?,
            3 => launch_row3_leaf_preimages(
                columns,
                &mut leaf_preimages[..leaf_preimage_len],
                leaf_count,
                stream,
            )?,
            5 => launch_row5_leaf_preimages(
                columns,
                &mut leaf_preimages[..leaf_preimage_len],
                leaf_count,
                stream,
            )?,
            _ => return Err(Error::LowDegreeCheckFailed),
        }
        let leaf_hashes = hash_device_chunks(
            &leaf_preimages,
            leaf_preimage_len,
            leaf_preimage_bytes,
            stream,
        )?;
        build_merkle_tree_from_leaf_hashes(leaf_hashes, leaf_count, stream)
    }

    fn build_group_tree_bn254(
        columns: &[&[ArkBn254Fr]],
        stream: &IcicleStream,
    ) -> Result<GpuPreparedGroupTree, Error> {
        if columns.is_empty() {
            return Err(Error::LowDegreeCheckFailed);
        }
        let leaf_count = columns[0].len();
        if leaf_count == 0 || !leaf_count.is_power_of_two() {
            return Err(Error::InvalidDomainSize(leaf_count));
        }
        if columns.iter().any(|column| column.len() != leaf_count) {
            return Err(Error::InvalidDomainSize(leaf_count));
        }

        let (leaf_preimages_host, leaf_preimage_bytes) = group_leaf_preimages(columns)?;
        let leaf_preimage_len = leaf_preimages_host.len();
        let mut leaf_preimages = allocate_device_bytes(leaf_preimage_len)?;
        leaf_preimages[..leaf_preimage_len]
            .copy_from_host_async(leaf_preimages_host.into_slice(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;

        let leaf_hashes = hash_device_chunks(
            &leaf_preimages,
            leaf_preimage_len,
            leaf_preimage_bytes,
            stream,
        )?;
        build_merkle_tree_from_leaf_hashes(leaf_hashes, leaf_count, stream)
    }

    fn build_merkle_tree_from_leaf_hashes(
        leaf_hashes: DeviceVec<u8>,
        leaf_count: usize,
        stream: &IcicleStream,
    ) -> Result<GpuPreparedGroupTree, Error> {
        let mut layers = vec![leaf_hashes];
        let mut counts = vec![leaf_count];
        let mut node_preimages =
            allocate_device_bytes((leaf_count / 2).max(1) * NODE_PREIMAGE_BYTES)?;
        let mut current_count = leaf_count;

        while current_count > 1 {
            let pair_count = current_count / 2;
            let next_hashes = {
                let previous = layers.last().unwrap();
                launch_node_preimages(
                    &previous[..current_count * DIGEST_BYTES],
                    &mut node_preimages[..pair_count * NODE_PREIMAGE_BYTES],
                    pair_count,
                    stream,
                )?;
                hash_device_chunks(
                    &node_preimages,
                    pair_count * NODE_PREIMAGE_BYTES,
                    NODE_PREIMAGE_BYTES,
                    stream,
                )?
            };
            layers.push(next_hashes);
            counts.push(pair_count);
            current_count = pair_count;
        }

        let root = copy_digest_at(layers.last().unwrap(), 0, stream)?;
        Ok(GpuPreparedGroupTree {
            layers,
            counts,
            root,
        })
    }

    fn group_leaf_preimages(columns: &[&[ArkBn254Fr]]) -> Result<(Vec<u8>, usize), Error> {
        let leaf_count = columns[0].len();
        let mut bytes = Vec::new();
        let mut leaf_preimage_bytes = None;

        for row in 0..leaf_count {
            let start = bytes.len();
            bytes.extend_from_slice(LEAF_DOMAIN);
            for column in columns {
                column[row]
                    .write(&mut bytes)
                    .map_err(|_| Error::GpuBackendFailed)?;
            }
            let current_bytes = bytes.len() - start;
            match leaf_preimage_bytes {
                Some(expected) if expected != current_bytes => return Err(Error::GpuBackendFailed),
                Some(_) => {}
                None => leaf_preimage_bytes = Some(current_bytes),
            }
        }

        Ok((
            bytes,
            leaf_preimage_bytes.ok_or(Error::InvalidDomainSize(leaf_count))?,
        ))
    }

    fn hash_device_chunks(
        input: &DeviceVec<u8>,
        input_len: usize,
        preimage_bytes: usize,
        stream: &IcicleStream,
    ) -> Result<DeviceVec<u8>, Error> {
        if preimage_bytes == 0 || input_len % preimage_bytes != 0 {
            return Err(Error::GpuBackendFailed);
        }
        let output_len = (input_len / preimage_bytes) * DIGEST_BYTES;
        let mut output = allocate_device_bytes(output_len)?;
        let mut cfg = HashConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;
        let hasher =
            IcicleKeccak256::new(preimage_bytes as u64).map_err(|_| Error::GpuBackendFailed)?;
        hasher
            .hash(&input[..input_len], &cfg, &mut output[..output_len])
            .map_err(|_| Error::GpuBackendFailed)?;
        Ok(output)
    }

    fn launch_field_leaf_preimages(
        fields: &DeviceVec<IcicleBn254Fr>,
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if !output.is_on_device() || count > fields.len() {
            return Err(Error::GpuBackendFailed);
        }
        let fields = &fields[..count];
        if !fields.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_leaf_preimages_cuda(
                fields.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_row2_leaf_preimages(
        columns: &[&DeviceVec<IcicleBn254Fr>],
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if columns.len() != 2 || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let column0 = &columns[0][..count];
        let column1 = &columns[1][..count];
        if !column0.is_on_device() || !column1.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_row2_leaf_preimages_cuda(
                column0.as_ptr() as *const c_void,
                column1.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_row3_leaf_preimages(
        columns: &[&DeviceVec<IcicleBn254Fr>],
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if columns.len() != 3 || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let column0 = &columns[0][..count];
        let column1 = &columns[1][..count];
        let column2 = &columns[2][..count];
        if !column0.is_on_device() || !column1.is_on_device() || !column2.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_row3_leaf_preimages_cuda(
                column0.as_ptr() as *const c_void,
                column1.as_ptr() as *const c_void,
                column2.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_row5_leaf_preimages(
        columns: &[&DeviceVec<IcicleBn254Fr>],
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if columns.len() != 5 || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let column0 = &columns[0][..count];
        let column1 = &columns[1][..count];
        let column2 = &columns[2][..count];
        let column3 = &columns[3][..count];
        let column4 = &columns[4][..count];
        if !column0.is_on_device()
            || !column1.is_on_device()
            || !column2.is_on_device()
            || !column3.is_on_device()
            || !column4.is_on_device()
        {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_row5_leaf_preimages_cuda(
                column0.as_ptr() as *const c_void,
                column1.as_ptr() as *const c_void,
                column2.as_ptr() as *const c_void,
                column3.as_ptr() as *const c_void,
                column4.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_node_preimages(
        digests: &(impl HostOrDeviceSlice<u8> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        pair_count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if !digests.is_on_device() || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_merkle_node_preimages_cuda(
                digests.as_ptr(),
                output.as_mut_ptr(),
                pair_count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn copy_digest_at(
        digests: &DeviceVec<u8>,
        index: usize,
        stream: &IcicleStream,
    ) -> Result<Digest32, Error> {
        let start = index * DIGEST_BYTES;
        let mut digest = [0u8; DIGEST_BYTES];
        digests[start..start + DIGEST_BYTES]
            .copy_to_host_async(digest.into_slice_mut(), stream)
            .map_err(|_| Error::GpuBackendFailed)?;
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        Ok(digest)
    }

    fn copy_digest_indices(
        digests: &DeviceVec<u8>,
        indices: &[usize],
        stream: &IcicleStream,
    ) -> Result<Vec<Digest32>, Error> {
        let mut bytes = vec![0u8; indices.len() * DIGEST_BYTES];
        for (position, &index) in indices.iter().enumerate() {
            let source_start = index * DIGEST_BYTES;
            let target_start = position * DIGEST_BYTES;
            digests[source_start..source_start + DIGEST_BYTES]
                .copy_to_host_async(
                    (&mut bytes[target_start..target_start + DIGEST_BYTES]).into_slice_mut(),
                    stream,
                )
                .map_err(|_| Error::GpuBackendFailed)?;
        }
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        Ok(bytes_to_digests(&bytes))
    }

    fn bytes_to_digests(bytes: &[u8]) -> Vec<Digest32> {
        bytes
            .chunks_exact(DIGEST_BYTES)
            .map(|chunk| {
                let mut digest = [0u8; DIGEST_BYTES];
                digest.copy_from_slice(chunk);
                digest
            })
            .collect()
    }

    fn allocate_device_vec(size: usize) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        DeviceVec::<IcicleBn254Fr>::device_malloc(size).map_err(|_| Error::GpuBackendFailed)
    }

    fn allocate_device_bytes(size: usize) -> Result<DeviceVec<u8>, Error> {
        DeviceVec::<u8>::device_malloc(size).map_err(|_| Error::GpuBackendFailed)
    }

    fn launch_memset_zero_fields(
        output: &mut (impl HostOrDeviceSlice<IcicleBn254Fr> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if !output.is_on_device() || output.len() < count {
            return Err(Error::GpuBackendFailed);
        }
        let byte_count = count * std::mem::size_of::<IcicleBn254Fr>();
        let status = unsafe {
            fri_lookup_cuda_memset_zero(output.as_mut_ptr() as *mut c_void, byte_count, **stream)
        };
        cuda_status_to_result(status)
    }

    fn cuda_status_to_result(status: i32) -> Result<(), Error> {
        if status == 0 {
            Ok(())
        } else {
            Err(Error::GpuBackendFailed)
        }
    }

    fn init_runtime() -> Result<(), Error> {
        RUNTIME_INIT
            .get_or_init(|| {
                load_backend_from_env_or_default().map_err(|_| Error::GpuBackendFailed)?;
                let device_type =
                    env::var(ICICLE_DEVICE_ENV).unwrap_or_else(|_| "CUDA".to_string());
                let device_id = env::var(ICICLE_DEVICE_ID_ENV)
                    .ok()
                    .and_then(|value| value.parse::<i32>().ok())
                    .unwrap_or(0);
                set_device(&Device::new(&device_type, device_id))
                    .map_err(|_| Error::GpuBackendFailed)
            })
            .clone()
    }

    fn init_ntt_domain(size: usize) -> Result<(), Error> {
        let mutex = NTT_DOMAIN_SIZE.get_or_init(|| Mutex::new(0));
        let mut current_size = mutex.lock().map_err(|_| Error::GpuBackendFailed)?;
        if *current_size >= size {
            return Ok(());
        }

        let rou =
            get_root_of_unity::<IcicleBn254Fr>(size as u64).map_err(|_| Error::GpuBackendFailed)?;
        initialize_domain(rou, &NTTInitDomainConfig::default())
            .map_err(|_| Error::GpuBackendFailed)?;
        *current_size = size;
        Ok(())
    }

    fn ark_to_icicle(value: &ArkBn254Fr) -> Result<IcicleBn254Fr, Error> {
        let mut bytes = Vec::with_capacity(32);
        value
            .into_repr()
            .write_le(&mut bytes)
            .map_err(|_| Error::GpuBackendFailed)?;
        Ok(IcicleBn254Fr::from_bytes_le(&bytes))
    }

    fn icicle_to_ark(value: &IcicleBn254Fr) -> Result<ArkBn254Fr, Error> {
        Ok(ArkBn254Fr::from_le_bytes_mod_order(&value.to_bytes_le()))
    }

    unsafe fn cast_ark_bn254_value<F: FftField>(value: &F) -> ArkBn254Fr {
        *(value as *const F as *const ArkBn254Fr)
    }

    unsafe fn cast_bn254_vec<F: FftField>(values: Vec<ArkBn254Fr>) -> Vec<F> {
        let mut values = std::mem::ManuallyDrop::new(values);
        Vec::from_raw_parts(
            values.as_mut_ptr() as *mut F,
            values.len(),
            values.capacity(),
        )
    }

    fn is_ark_bn254_fr<F: FftField>() -> bool {
        std::any::type_name::<F>() == std::any::type_name::<ArkBn254Fr>()
    }

    unsafe fn cast_ark_bn254_slice<F: FftField>(values: &[F]) -> &[ArkBn254Fr] {
        std::slice::from_raw_parts(values.as_ptr() as *const ArkBn254Fr, values.len())
    }
}

#[cfg(feature = "icicle-rust")]
mod gpu_batch_combine_bn254 {
    use crate::error::Error;
    use ark_bn254::Fr as ArkBn254Fr;
    use ark_ff::{BigInteger, FftField, Field, PrimeField};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use icicle_bn254::curve::ScalarField as IcicleBn254Fr;
    use icicle_core::{
        bignum::BigNum,
        vec_ops::{add_scalars, mul_scalars, scalar_mul, VecOpsConfig},
    };
    use icicle_runtime::{
        memory::{DeviceVec, IntoIcicleSlice, IntoIcicleSliceMut},
        runtime::{load_backend_from_env_or_default, set_device},
        stream::IcicleStream,
        Device,
    };
    use std::{env, sync::OnceLock};

    const ICICLE_DEVICE_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE";
    const ICICLE_DEVICE_ID_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE_ID";

    static RUNTIME_INIT: OnceLock<Result<(), Error>> = OnceLock::new();

    pub(super) struct BatchCombineWorkspace {
        capacity: usize,
        column: DeviceVec<IcicleBn254Fr>,
        weights: DeviceVec<IcicleBn254Fr>,
        coefficient: DeviceVec<IcicleBn254Fr>,
        term: DeviceVec<IcicleBn254Fr>,
        combined: DeviceVec<IcicleBn254Fr>,
        next: DeviceVec<IcicleBn254Fr>,
    }

    impl BatchCombineWorkspace {
        fn new(capacity: usize) -> Result<Self, Error> {
            Ok(Self {
                capacity,
                column: allocate_device_vec(capacity)?,
                weights: allocate_device_vec(capacity)?,
                coefficient: allocate_device_vec(1)?,
                term: allocate_device_vec(capacity)?,
                combined: allocate_device_vec(capacity)?,
                next: allocate_device_vec(capacity)?,
            })
        }
    }

    pub fn supports_field<F: FftField>() -> bool {
        is_ark_bn254_fr::<F>()
    }

    pub fn try_combine_lde_columns<F: PrimeField + FftField>(
        lde_columns: &[&[F]],
        batching_coefficients: &[F],
        column_degree_bounds: &[usize],
        degree_bound: usize,
    ) -> Result<Option<Vec<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }

        init_runtime()?;

        let columns = lde_columns
            .iter()
            .map(|column| unsafe { cast_ark_bn254_slice(column) })
            .collect::<Vec<_>>();
        let coefficients = unsafe { cast_ark_bn254_slice(batching_coefficients) };
        let result =
            combine_lde_columns_bn254(&columns, coefficients, column_degree_bounds, degree_bound)?;
        Ok(Some(unsafe { cast_bn254_vec(result) }))
    }

    fn combine_lde_columns_bn254(
        lde_columns: &[&[ArkBn254Fr]],
        batching_coefficients: &[ArkBn254Fr],
        column_degree_bounds: &[usize],
        degree_bound: usize,
    ) -> Result<Vec<ArkBn254Fr>, Error> {
        let lde_size = lde_columns[0].len();
        let coefficients = batching_coefficients
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        let weight_columns = normalized_weight_columns(
            lde_size,
            degree_bound,
            batching_coefficients,
            column_degree_bounds,
        )?;

        let columns = lde_columns
            .iter()
            .map(|column| {
                column
                    .iter()
                    .map(ark_to_icicle)
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let combined = with_batch_combine_context(|context, stream| {
            combine_lde_columns_bn254_on_device(
                &columns,
                &coefficients,
                &weight_columns,
                context,
                stream,
            )
        })?;
        combined.iter().map(icicle_to_ark).collect()
    }

    fn with_batch_combine_context<T>(
        f: impl FnOnce(&mut Option<BatchCombineWorkspace>, &IcicleStream) -> Result<T, Error>,
    ) -> Result<T, Error> {
        super::gpu_resident_fri_bn254::with_context(|context| {
            f(&mut context.batch_combine_workspace, &context.stream)
        })
    }

    fn combine_lde_columns_bn254_on_device(
        columns: &[Vec<IcicleBn254Fr>],
        coefficients: &[IcicleBn254Fr],
        weight_columns: &[Option<Vec<IcicleBn254Fr>>],
        workspace: &mut Option<BatchCombineWorkspace>,
        stream: &IcicleStream,
    ) -> Result<Vec<IcicleBn254Fr>, Error> {
        let count = columns[0].len();
        ensure_workspace_capacity(workspace, count)?;
        let workspace = workspace.as_mut().unwrap();

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        for (column_index, column) in columns.iter().enumerate() {
            workspace.column[..count]
                .copy_from_host_async(column.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed)?;

            let target_is_combined = column_index == 0;
            if let Some(weights) = &weight_columns[column_index] {
                workspace.weights[..count]
                    .copy_from_host_async(weights.into_slice(), stream)
                    .map_err(|_| Error::GpuBackendFailed)?;
                if target_is_combined {
                    mul_scalars(
                        &workspace.column[..count],
                        &workspace.weights[..count],
                        &mut workspace.combined[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                } else {
                    mul_scalars(
                        &workspace.column[..count],
                        &workspace.weights[..count],
                        &mut workspace.term[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                }
            } else {
                workspace.coefficient[..1]
                    .copy_from_host_async(
                        (&coefficients[column_index..column_index + 1]).into_slice(),
                        stream,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                if target_is_combined {
                    scalar_mul(
                        &workspace.coefficient[..1],
                        &workspace.column[..count],
                        &mut workspace.combined[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                } else {
                    scalar_mul(
                        &workspace.coefficient[..1],
                        &workspace.column[..count],
                        &mut workspace.term[..count],
                        &cfg,
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
                }
            }

            if !target_is_combined {
                add_scalars(
                    &workspace.combined[..count],
                    &workspace.term[..count],
                    &mut workspace.next[..count],
                    &cfg,
                )
                .map_err(|_| Error::GpuBackendFailed)?;
                std::mem::swap(&mut workspace.combined, &mut workspace.next);
            }
        }

        let mut combined = vec![IcicleBn254Fr::zero(); count];
        let copy_result = workspace.combined[..count]
            .copy_to_host_async(combined.into_slice_mut(), stream)
            .map_err(|_| Error::GpuBackendFailed);
        let sync_result = stream.synchronize().map_err(|_| Error::GpuBackendFailed);
        copy_result?;
        sync_result?;
        Ok(combined)
    }

    fn normalized_weight_columns(
        lde_size: usize,
        degree_bound: usize,
        batching_coefficients: &[ArkBn254Fr],
        column_degree_bounds: &[usize],
    ) -> Result<Vec<Option<Vec<IcicleBn254Fr>>>, Error> {
        let domain = GeneralEvaluationDomain::<ArkBn254Fr>::new(lde_size)
            .ok_or(Error::InvalidDomainSize(lde_size))?;
        let root = domain.element(1);
        batching_coefficients
            .iter()
            .zip(column_degree_bounds)
            .map(|(coefficient, column_bound)| {
                let exponent = degree_bound - *column_bound;
                if exponent == 0 {
                    return Ok(None);
                }

                let step = root.pow(&[exponent as u64]);
                let mut power = ArkBn254Fr::from(1u64);
                let mut weights = Vec::with_capacity(lde_size);
                for _ in 0..lde_size {
                    weights.push(ark_to_icicle(&(*coefficient * power))?);
                    power *= step;
                }
                Ok(Some(weights))
            })
            .collect()
    }

    fn ensure_workspace_capacity(
        workspace: &mut Option<BatchCombineWorkspace>,
        required_capacity: usize,
    ) -> Result<(), Error> {
        if workspace
            .as_ref()
            .map_or(true, |workspace| workspace.capacity < required_capacity)
        {
            *workspace = Some(BatchCombineWorkspace::new(required_capacity)?);
        }
        Ok(())
    }

    fn allocate_device_vec(size: usize) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        DeviceVec::<IcicleBn254Fr>::device_malloc(size).map_err(|_| Error::GpuBackendFailed)
    }

    fn init_runtime() -> Result<(), Error> {
        RUNTIME_INIT
            .get_or_init(|| {
                load_backend_from_env_or_default().map_err(|_| Error::GpuBackendFailed)?;
                let device_type =
                    env::var(ICICLE_DEVICE_ENV).unwrap_or_else(|_| "CUDA".to_string());
                let device_id = env::var(ICICLE_DEVICE_ID_ENV)
                    .ok()
                    .and_then(|value| value.parse::<i32>().ok())
                    .unwrap_or(0);
                set_device(&Device::new(&device_type, device_id))
                    .map_err(|_| Error::GpuBackendFailed)
            })
            .clone()
    }

    fn ark_to_icicle(value: &ArkBn254Fr) -> Result<IcicleBn254Fr, Error> {
        let mut bytes = Vec::with_capacity(32);
        value
            .into_repr()
            .write_le(&mut bytes)
            .map_err(|_| Error::GpuBackendFailed)?;
        Ok(IcicleBn254Fr::from_bytes_le(&bytes))
    }

    fn icicle_to_ark(value: &IcicleBn254Fr) -> Result<ArkBn254Fr, Error> {
        Ok(ArkBn254Fr::from_le_bytes_mod_order(&value.to_bytes_le()))
    }

    fn is_ark_bn254_fr<F: FftField>() -> bool {
        std::any::type_name::<F>() == std::any::type_name::<ArkBn254Fr>()
    }

    unsafe fn cast_ark_bn254_slice<F: FftField>(values: &[F]) -> &[ArkBn254Fr] {
        std::slice::from_raw_parts(values.as_ptr() as *const ArkBn254Fr, values.len())
    }

    unsafe fn cast_bn254_vec<F: FftField>(values: Vec<ArkBn254Fr>) -> Vec<F> {
        let mut values = std::mem::ManuallyDrop::new(values);
        Vec::from_raw_parts(
            values.as_mut_ptr() as *mut F,
            values.len(),
            values.capacity(),
        )
    }
}

#[cfg(feature = "icicle-rust")]
mod gpu_fold_bn254 {
    use crate::{
        error::Error,
        merkle::{Digest32, MerklePath},
    };
    use ark_bn254::Fr as ArkBn254Fr;
    use ark_ff::{BigInteger, FftField, Field, PrimeField};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use icicle_bn254::curve::ScalarField as IcicleBn254Fr;
    use icicle_core::{
        bignum::BigNum,
        hash::HashConfig,
        vec_ops::{add_scalars, mul_scalars, scalar_mul, sub_scalars, VecOpsConfig},
    };
    use icicle_hash::keccak::Keccak256 as IcicleKeccak256;
    use icicle_runtime::{
        memory::{DeviceVec, HostOrDeviceSlice, IntoIcicleSlice, IntoIcicleSliceMut},
        runtime::{load_backend_from_env_or_default, set_device},
        stream::IcicleStream,
        Device,
    };
    use std::{env, ffi::c_void, sync::OnceLock};

    const ICICLE_DEVICE_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE";
    const ICICLE_DEVICE_ID_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE_ID";
    const DIGEST_BYTES: usize = 32;
    const LEAF_PREIMAGE_BYTES: usize = 36;
    const NODE_PREIMAGE_BYTES: usize = 68;
    const GPU_FOLD_MERKLE_MIN_LEAVES_ENV: &str = "FRI_LOOKUP_GPU_FOLD_MERKLE_MIN_LEAVES";
    const DEFAULT_GPU_FOLD_MERKLE_MIN_LEAVES: usize = 1 << 16;

    static RUNTIME_INIT: OnceLock<Result<(), Error>> = OnceLock::new();

    extern "C" {
        fn fri_lookup_bn254_leaf_preimages_cuda(
            fields: *const c_void,
            output: *mut u8,
            count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
        fn fri_lookup_merkle_node_preimages_cuda(
            digests: *const u8,
            output: *mut u8,
            pair_count: usize,
            stream_handle: *mut c_void,
        ) -> i32;
    }

    pub(super) struct FoldWorkspace {
        capacity: usize,
        left: DeviceVec<IcicleBn254Fr>,
        right: DeviceVec<IcicleBn254Fr>,
        coeffs: DeviceVec<IcicleBn254Fr>,
        two_inv: DeviceVec<IcicleBn254Fr>,
        sum: DeviceVec<IcicleBn254Fr>,
        diff: DeviceVec<IcicleBn254Fr>,
        scaled_sum: DeviceVec<IcicleBn254Fr>,
        scaled_diff: DeviceVec<IcicleBn254Fr>,
        folded: DeviceVec<IcicleBn254Fr>,
        leaf_preimages: DeviceVec<u8>,
        node_preimages: DeviceVec<u8>,
    }

    pub(super) struct GpuPreparedMerkleTree {
        layers: Vec<DeviceVec<u8>>,
        counts: Vec<usize>,
        root: Digest32,
    }

    unsafe impl Send for GpuPreparedMerkleTree {}
    unsafe impl Sync for GpuPreparedMerkleTree {}

    impl GpuPreparedMerkleTree {
        pub(super) fn root(&self) -> Digest32 {
            self.root
        }

        pub(super) fn open_many(&self, indices: &[usize]) -> Result<Vec<MerklePath>, Error> {
            if indices.is_empty() {
                return Ok(Vec::new());
            }
            for &index in indices {
                if index >= self.counts[0] {
                    return Err(Error::InvalidDomainSize(index));
                }
            }

            crate::gpu_lock::with_gpu_lock(|| {
                init_runtime()?;
                with_fold_context(|_context, stream| {
                    let mut paths = indices
                        .iter()
                        .copied()
                        .map(|index| MerklePath {
                            index,
                            siblings: Vec::with_capacity(self.layers.len() - 1),
                        })
                        .collect::<Vec<_>>();
                    let mut current_indices = indices.to_vec();

                    for (layer_index, (layer, &count)) in self
                        .layers
                        .iter()
                        .zip(&self.counts)
                        .take(self.layers.len() - 1)
                        .enumerate()
                    {
                        let sibling_indices = current_indices
                            .iter()
                            .copied()
                            .map(|index| {
                                if index >= count {
                                    return Err(Error::InvalidDomainSize(index));
                                }
                                Ok(if index % 2 == 0 { index + 1 } else { index - 1 })
                            })
                            .collect::<Result<Vec<_>, Error>>()?;
                        gpu_fold_diag(|| {
                            format!(
                                "open fold merkle: layer={layer_index}, count={count}, siblings={}",
                                sibling_indices.len()
                            )
                        });
                        let siblings = copy_digest_indices(layer, &sibling_indices, stream)?;
                        for (path, sibling) in paths.iter_mut().zip(siblings) {
                            path.siblings.push(sibling);
                        }
                        for index in &mut current_indices {
                            *index /= 2;
                        }
                    }

                    Ok(paths)
                })
            })
        }
    }

    impl FoldWorkspace {
        fn new(capacity: usize) -> Result<Self, Error> {
            Ok(Self {
                capacity,
                left: allocate_device_vec(capacity)?,
                right: allocate_device_vec(capacity)?,
                coeffs: allocate_device_vec(capacity)?,
                two_inv: allocate_device_vec(1)?,
                sum: allocate_device_vec(capacity)?,
                diff: allocate_device_vec(capacity)?,
                scaled_sum: allocate_device_vec(capacity)?,
                scaled_diff: allocate_device_vec(capacity)?,
                folded: allocate_device_vec(capacity)?,
                leaf_preimages: allocate_device_bytes(capacity * LEAF_PREIMAGE_BYTES)?,
                node_preimages: allocate_device_bytes((capacity / 2).max(1) * NODE_PREIMAGE_BYTES)?,
            })
        }
    }

    pub fn supports_field<F: FftField>() -> bool {
        is_ark_bn254_fr::<F>()
    }

    pub fn try_fold_layer<F: FftField>(evals: &[F], alpha: F) -> Result<Option<Vec<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }

        init_runtime()?;

        let evals = unsafe { cast_ark_bn254_slice(evals) };
        let alpha = unsafe { cast_ark_bn254_value(&alpha) };
        let (folded, _) = fold_layer_bn254(evals, alpha, false)?;
        Ok(Some(unsafe { cast_bn254_vec(folded) }))
    }

    pub fn try_fold_layer_with_tree<F: PrimeField + FftField>(
        evals: &[F],
        alpha: F,
    ) -> Result<Option<(Vec<F>, Option<GpuPreparedMerkleTree>)>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }

        init_runtime()?;

        let evals = unsafe { cast_ark_bn254_slice(evals) };
        let alpha = unsafe { cast_ark_bn254_value(&alpha) };
        let build_gpu_tree = evals.len() / 2 >= gpu_fold_merkle_min_leaves();
        let (folded, tree) = fold_layer_bn254(evals, alpha, build_gpu_tree)?;
        Ok(Some((unsafe { cast_bn254_vec(folded) }, tree)))
    }

    fn fold_layer_bn254(
        evals: &[ArkBn254Fr],
        alpha: ArkBn254Fr,
        build_tree: bool,
    ) -> Result<(Vec<ArkBn254Fr>, Option<GpuPreparedMerkleTree>), Error> {
        if !evals.len().is_power_of_two() || evals.len() < 2 {
            return Err(Error::InvalidDomainSize(evals.len()));
        }

        let half = evals.len() / 2;
        let two_inv = ArkBn254Fr::from(2u64)
            .inverse()
            .ok_or(Error::LowDegreeCheckFailed)?;
        let coeffs = fold_coefficients(evals.len(), alpha, two_inv)?
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        let two_inv = vec![ark_to_icicle(&two_inv)?];
        let left = evals[..half]
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        let right = evals[half..]
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;

        let (folded, tree) = with_fold_context(|context, stream| {
            fold_layer_bn254_on_device(
                &left, &right, &coeffs, &two_inv, context, stream, build_tree,
            )
        })?;

        let folded = folded
            .iter()
            .map(icicle_to_ark)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((folded, tree))
    }

    fn with_fold_context<T>(
        f: impl FnOnce(&mut Option<FoldWorkspace>, &IcicleStream) -> Result<T, Error>,
    ) -> Result<T, Error> {
        super::gpu_resident_fri_bn254::with_context(|context| {
            f(&mut context.fold_workspace, &context.stream)
        })
    }

    fn fold_layer_bn254_on_device(
        left: &[IcicleBn254Fr],
        right: &[IcicleBn254Fr],
        coeffs: &[IcicleBn254Fr],
        two_inv: &[IcicleBn254Fr],
        workspace: &mut Option<FoldWorkspace>,
        stream: &IcicleStream,
        build_tree: bool,
    ) -> Result<(Vec<IcicleBn254Fr>, Option<GpuPreparedMerkleTree>), Error> {
        let half = left.len();
        ensure_workspace_capacity(workspace, half)?;
        let workspace = workspace.as_mut().unwrap();

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        let mut result = Ok(());
        if result.is_ok() {
            result = workspace.left[..half]
                .copy_from_host_async(left.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed);
        }
        if result.is_ok() {
            result = workspace.right[..half]
                .copy_from_host_async(right.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed);
        }
        if result.is_ok() {
            result = workspace.coeffs[..half]
                .copy_from_host_async(coeffs.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed);
        }
        if result.is_ok() {
            result = workspace.two_inv[..1]
                .copy_from_host_async(two_inv.into_slice(), stream)
                .map_err(|_| Error::GpuBackendFailed);
        }
        if result.is_ok() {
            result = add_scalars(
                &workspace.left[..half],
                &workspace.right[..half],
                &mut workspace.sum[..half],
                &cfg,
            )
            .map_err(|_| Error::GpuBackendFailed);
        }
        if result.is_ok() {
            result = sub_scalars(
                &workspace.left[..half],
                &workspace.right[..half],
                &mut workspace.diff[..half],
                &cfg,
            )
            .map_err(|_| Error::GpuBackendFailed);
        }
        if result.is_ok() {
            result = scalar_mul(
                &workspace.two_inv[..1],
                &workspace.sum[..half],
                &mut workspace.scaled_sum[..half],
                &cfg,
            )
            .map_err(|_| Error::GpuBackendFailed);
        }
        if result.is_ok() {
            result = mul_scalars(
                &workspace.diff[..half],
                &workspace.coeffs[..half],
                &mut workspace.scaled_diff[..half],
                &cfg,
            )
            .map_err(|_| Error::GpuBackendFailed);
        }
        if result.is_ok() {
            result = add_scalars(
                &workspace.scaled_sum[..half],
                &workspace.scaled_diff[..half],
                &mut workspace.folded[..half],
                &cfg,
            )
            .map_err(|_| Error::GpuBackendFailed);
        }

        result?;

        let tree = if build_tree {
            Some(build_merkle_tree_from_device_fields(
                workspace, half, stream,
            )?)
        } else {
            None
        };

        let mut folded = vec![IcicleBn254Fr::zero(); half];
        let copy_result = workspace.folded[..half]
            .copy_to_host_async(folded.into_slice_mut(), stream)
            .map_err(|_| Error::GpuBackendFailed);
        let sync_result = stream.synchronize().map_err(|_| Error::GpuBackendFailed);
        copy_result?;
        sync_result?;
        Ok((folded, tree))
    }

    fn gpu_fold_merkle_min_leaves() -> usize {
        env::var(GPU_FOLD_MERKLE_MIN_LEAVES_ENV)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_GPU_FOLD_MERKLE_MIN_LEAVES)
    }

    fn build_merkle_tree_from_device_fields(
        workspace: &mut FoldWorkspace,
        leaf_count: usize,
        stream: &IcicleStream,
    ) -> Result<GpuPreparedMerkleTree, Error> {
        if leaf_count == 0 || !leaf_count.is_power_of_two() {
            return Err(Error::InvalidDomainSize(leaf_count));
        }
        gpu_fold_diag(|| format!("build fold merkle: leaf_count={leaf_count}"));
        gpu_fold_diag(|| "build fold merkle: leaf preimages".to_string());
        launch_leaf_preimages(
            &workspace.folded[..leaf_count],
            &mut workspace.leaf_preimages[..leaf_count * LEAF_PREIMAGE_BYTES],
            leaf_count,
            stream,
        )?;

        let mut cfg = HashConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        gpu_fold_diag(|| "build fold merkle: create keccak hashers".to_string());
        let leaf_hasher = IcicleKeccak256::new(LEAF_PREIMAGE_BYTES as u64)
            .map_err(|_| Error::GpuBackendFailed)?;
        let node_hasher = IcicleKeccak256::new(NODE_PREIMAGE_BYTES as u64)
            .map_err(|_| Error::GpuBackendFailed)?;

        let mut layers = Vec::new();
        let mut counts = Vec::new();
        let mut leaf_hashes = allocate_device_bytes(leaf_count * DIGEST_BYTES)?;
        gpu_fold_diag(|| "build fold merkle: hash leaves".to_string());
        leaf_hasher
            .hash(
                &workspace.leaf_preimages[..leaf_count * LEAF_PREIMAGE_BYTES],
                &cfg,
                &mut leaf_hashes[..leaf_count * DIGEST_BYTES],
            )
            .map_err(|_| Error::GpuBackendFailed)?;
        layers.push(leaf_hashes);
        counts.push(leaf_count);

        let mut current_count = leaf_count;
        while current_count > 1 {
            let pair_count = current_count / 2;
            let mut next_hashes = allocate_device_bytes(pair_count * DIGEST_BYTES)?;
            {
                let previous = layers.last().unwrap();
                gpu_fold_diag(|| {
                    format!(
                        "build fold merkle: node preimages current_count={current_count}, pair_count={pair_count}"
                    )
                });
                launch_node_preimages(
                    &previous[..current_count * DIGEST_BYTES],
                    &mut workspace.node_preimages[..pair_count * NODE_PREIMAGE_BYTES],
                    pair_count,
                    stream,
                )?;
                gpu_fold_diag(|| {
                    format!(
                        "build fold merkle: hash nodes current_count={current_count}, pair_count={pair_count}"
                    )
                });
                node_hasher
                    .hash(
                        &workspace.node_preimages[..pair_count * NODE_PREIMAGE_BYTES],
                        &cfg,
                        &mut next_hashes[..pair_count * DIGEST_BYTES],
                    )
                    .map_err(|_| Error::GpuBackendFailed)?;
            }
            layers.push(next_hashes);
            counts.push(pair_count);
            current_count = pair_count;
        }

        gpu_fold_diag(|| "build fold merkle: copy root".to_string());
        let root = copy_digest_at(layers.last().unwrap(), 0, stream)?;
        gpu_fold_diag(|| "build fold merkle: done".to_string());
        Ok(GpuPreparedMerkleTree {
            layers,
            counts,
            root,
        })
    }

    fn launch_leaf_preimages(
        fields: &(impl HostOrDeviceSlice<IcicleBn254Fr> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if !fields.is_on_device() || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_bn254_leaf_preimages_cuda(
                fields.as_ptr() as *const c_void,
                output.as_mut_ptr(),
                count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn launch_node_preimages(
        digests: &(impl HostOrDeviceSlice<u8> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
        pair_count: usize,
        stream: &IcicleStream,
    ) -> Result<(), Error> {
        if !digests.is_on_device() || !output.is_on_device() {
            return Err(Error::GpuBackendFailed);
        }
        let status = unsafe {
            fri_lookup_merkle_node_preimages_cuda(
                digests.as_ptr(),
                output.as_mut_ptr(),
                pair_count,
                **stream,
            )
        };
        cuda_status_to_result(status)
    }

    fn copy_digest_at(
        digests: &DeviceVec<u8>,
        index: usize,
        stream: &IcicleStream,
    ) -> Result<Digest32, Error> {
        Ok(copy_digest_indices(digests, &[index], stream)?.remove(0))
    }

    fn copy_digest_indices(
        digests: &DeviceVec<u8>,
        indices: &[usize],
        stream: &IcicleStream,
    ) -> Result<Vec<Digest32>, Error> {
        let mut bytes = vec![0u8; indices.len() * DIGEST_BYTES];
        for (position, &index) in indices.iter().enumerate() {
            let source_start = index * DIGEST_BYTES;
            let target_start = position * DIGEST_BYTES;
            digests[source_start..source_start + DIGEST_BYTES]
                .copy_to_host_async(
                    (&mut bytes[target_start..target_start + DIGEST_BYTES]).into_slice_mut(),
                    stream,
                )
                .map_err(|_| Error::GpuBackendFailed)?;
        }
        stream.synchronize().map_err(|_| Error::GpuBackendFailed)?;
        Ok(bytes_to_digests(&bytes))
    }

    fn bytes_to_digests(bytes: &[u8]) -> Vec<Digest32> {
        bytes
            .chunks_exact(DIGEST_BYTES)
            .map(|chunk| {
                let mut digest = [0u8; DIGEST_BYTES];
                digest.copy_from_slice(chunk);
                digest
            })
            .collect()
    }

    fn cuda_status_to_result(status: i32) -> Result<(), Error> {
        if status == 0 {
            Ok(())
        } else {
            gpu_fold_diag(|| format!("cuda kernel returned status={status}"));
            Err(Error::GpuBackendFailed)
        }
    }

    fn gpu_fold_diag(message: impl FnOnce() -> String) {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if *ENABLED.get_or_init(|| {
            matches!(
                env::var("FRI_LOOKUP_GPU_DIAGNOSTICS")
                    .ok()
                    .map(|value| value.to_ascii_lowercase())
                    .as_deref(),
                Some("1" | "true" | "yes" | "on")
            )
        }) {
            eprintln!("[fri-gpu-fold] {}", message());
        }
    }

    fn ensure_workspace_capacity(
        workspace: &mut Option<FoldWorkspace>,
        required_capacity: usize,
    ) -> Result<(), Error> {
        if workspace
            .as_ref()
            .map_or(true, |workspace| workspace.capacity < required_capacity)
        {
            *workspace = Some(FoldWorkspace::new(required_capacity)?);
        }
        Ok(())
    }

    fn allocate_device_vec(size: usize) -> Result<DeviceVec<IcicleBn254Fr>, Error> {
        DeviceVec::<IcicleBn254Fr>::device_malloc(size).map_err(|_| Error::GpuBackendFailed)
    }

    fn allocate_device_bytes(size: usize) -> Result<DeviceVec<u8>, Error> {
        DeviceVec::<u8>::device_malloc(size).map_err(|_| Error::GpuBackendFailed)
    }

    fn fold_coefficients(
        domain_size: usize,
        alpha: ArkBn254Fr,
        two_inv: ArkBn254Fr,
    ) -> Result<Vec<ArkBn254Fr>, Error> {
        let domain = GeneralEvaluationDomain::<ArkBn254Fr>::new(domain_size)
            .ok_or(Error::InvalidDomainSize(domain_size))?;
        let omega_inv = domain
            .element(1)
            .inverse()
            .ok_or(Error::LowDegreeCheckFailed)?;
        let alpha_two_inv = alpha * two_inv;
        let mut x_inv = ArkBn254Fr::from(1u64);
        let mut coeffs = Vec::with_capacity(domain_size / 2);

        for _ in 0..domain_size / 2 {
            coeffs.push(alpha_two_inv * x_inv);
            x_inv *= omega_inv;
        }

        Ok(coeffs)
    }

    fn init_runtime() -> Result<(), Error> {
        RUNTIME_INIT
            .get_or_init(|| {
                load_backend_from_env_or_default().map_err(|_| Error::GpuBackendFailed)?;
                let device_type =
                    env::var(ICICLE_DEVICE_ENV).unwrap_or_else(|_| "CUDA".to_string());
                let device_id = env::var(ICICLE_DEVICE_ID_ENV)
                    .ok()
                    .and_then(|value| value.parse::<i32>().ok())
                    .unwrap_or(0);
                set_device(&Device::new(&device_type, device_id))
                    .map_err(|_| Error::GpuBackendFailed)
            })
            .clone()
    }

    fn ark_to_icicle(value: &ArkBn254Fr) -> Result<IcicleBn254Fr, Error> {
        let mut bytes = Vec::with_capacity(32);
        value
            .into_repr()
            .write_le(&mut bytes)
            .map_err(|_| Error::GpuBackendFailed)?;
        Ok(IcicleBn254Fr::from_bytes_le(&bytes))
    }

    fn icicle_to_ark(value: &IcicleBn254Fr) -> Result<ArkBn254Fr, Error> {
        Ok(ArkBn254Fr::from_le_bytes_mod_order(&value.to_bytes_le()))
    }

    fn is_ark_bn254_fr<F: FftField>() -> bool {
        std::any::type_name::<F>() == std::any::type_name::<ArkBn254Fr>()
    }

    unsafe fn cast_ark_bn254_slice<F: FftField>(values: &[F]) -> &[ArkBn254Fr] {
        std::slice::from_raw_parts(values.as_ptr() as *const ArkBn254Fr, values.len())
    }

    unsafe fn cast_ark_bn254_value<F: FftField>(value: &F) -> ArkBn254Fr {
        *(value as *const F as *const ArkBn254Fr)
    }

    unsafe fn cast_bn254_vec<F: FftField>(values: Vec<ArkBn254Fr>) -> Vec<F> {
        let mut values = std::mem::ManuallyDrop::new(values);
        Vec::from_raw_parts(
            values.as_mut_ptr() as *mut F,
            values.len(),
            values.capacity(),
        )
    }
}

fn fri_transcript<F: PrimeField>(
    root: Digest32,
    degree_bound: usize,
    blowup: usize,
    domain_size: usize,
) -> Transcript {
    let mut transcript = Transcript::new(FRI_PROTOCOL_LABEL);
    transcript.absorb_bytes(b"root", &root);
    transcript.absorb(b"degree-bound", &F::from(degree_bound as u64));
    transcript.absorb(b"blowup", &F::from(blowup as u64));
    transcript.absorb(b"domain-size", &F::from(domain_size as u64));
    transcript
}

fn query_count(domain_size: usize) -> usize {
    DEFAULT_QUERY_COUNT.min(domain_size)
}
