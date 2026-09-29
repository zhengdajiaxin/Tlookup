use ark_ff::{FftField, PrimeField};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

use crate::{
    error::Error,
    fri::{self, BatchFriProof, BatchLdeRowOpening},
    merkle::{Digest32, MerklePath, MerkleTree},
};

pub const DEFAULT_GPU_USC_MIN_SIZE: usize = 1 << 16;

#[derive(Clone, Debug)]
pub struct SpecializedUscProof<F: FftField> {
    pub batch: BatchFriProof<F>,
    pub openings: Vec<SpecializedUscOpening<F>>,
}

#[derive(Clone, Debug)]
pub struct SpecializedUscOpening<F: FftField> {
    pub lde_index: usize,
    pub existing_group_row: Vec<F>,
    pub existing_group_path: MerklePath,
    pub p_opening: BatchLdeRowOpening<F>,
}

pub fn specialized_usc_quotient_evaluations<F: PrimeField + FftField>(
    values: &[F],
    claimed_sum: F,
) -> Result<Vec<F>, Error> {
    if values.is_empty() {
        return Err(Error::InvalidDomainSize(0));
    }
    let domain = GeneralEvaluationDomain::<F>::new(values.len())
        .ok_or(Error::InvalidDomainSize(values.len()))?;
    let size = F::from(values.len() as u64);
    let size_inverse = size
        .inverse()
        .ok_or(Error::InvalidDomainSize(values.len()))?;
    let average = claimed_sum * size_inverse;

    let mut coeffs = domain.ifft(values);
    coeffs.resize(values.len(), F::zero());
    if coeffs[0] != average {
        return Err(Error::InvalidSumClaim);
    }

    let quotient_coeffs = if values.len() == 1 {
        vec![F::zero()]
    } else {
        coeffs[1..].to_vec()
    };
    let quotient = domain.fft(&quotient_coeffs);
    Ok(quotient)
}

pub fn specialized_usc_column_degree_bound(row_count: usize) -> usize {
    row_count.saturating_sub(1).max(1)
}

#[allow(clippy::too_many_arguments)]
pub fn specialized_usc_verify<F: PrimeField + FftField + Send + Sync>(
    existing_group_root: Digest32,
    existing_value_position: usize,
    claimed_sum: F,
    row_count: usize,
    blowup: usize,
    expected_batched_coefficients: &[F],
    proof: &SpecializedUscProof<F>,
) -> Result<(), Error> {
    if row_count == 0 {
        return Err(Error::InvalidDomainSize(row_count));
    }
    let size = F::from(row_count as u64);
    if size.is_zero() {
        return Err(Error::InvalidDomainSize(row_count));
    }
    let p_degree_bound = specialized_usc_column_degree_bound(row_count);
    if proof.batch.row_count != row_count
        || proof.batch.column_count != 1
        || proof.batch.column_degree_bounds != [p_degree_bound]
        || proof.batch.batching_coefficients != expected_batched_coefficients
        || proof.batch.commitment_groups.len() != 1
        || proof.batch.commitment_groups[0].columns != [0]
    {
        return Err(Error::StatementMismatch);
    }

    fri::verify_batch_low_degree(&proof.batch, row_count, blowup)?;

    let queries = &proof.batch.combined.queries;
    if proof.openings.len() != queries.len() {
        return Err(Error::InvalidSpecializedUscIdentity);
    }
    let lde_size = row_count * blowup;
    let domain =
        GeneralEvaluationDomain::<F>::new(lde_size).ok_or(Error::InvalidDomainSize(lde_size))?;

    for (query, opening) in queries.iter().zip(&proof.openings) {
        if opening.lde_index != query.index || opening.p_opening.lde_index != query.index {
            return Err(Error::InvalidSpecializedUscIdentity);
        }
        if existing_value_position >= opening.existing_group_row.len() {
            return Err(Error::InvalidSpecializedUscIdentity);
        }
        MerkleTree::verify(
            existing_group_root,
            &opening.existing_group_row,
            &opening.existing_group_path,
        )?;
        if opening.existing_group_path.index != opening.lde_index {
            return Err(Error::InvalidSpecializedUscIdentity);
        }
        fri::verify_batch_lde_row_opening(&proof.batch, &opening.p_opening, row_count, blowup)?;
        if opening.p_opening.row.len() != 1 {
            return Err(Error::InvalidSpecializedUscIdentity);
        }

        let point = domain.element(opening.lde_index);
        let value = opening.existing_group_row[existing_value_position];
        let p_value = opening.p_opening.row[0];
        if size * value - claimed_sum != size * point * p_value {
            return Err(Error::InvalidSpecializedUscIdentity);
        }
    }

    Ok(())
}

pub fn warm_up_gpu_usc<F: PrimeField + FftField>(
    row_count: usize,
    blowup: usize,
) -> Result<(), Error> {
    let _ = (row_count, blowup);
    Ok(())
}
