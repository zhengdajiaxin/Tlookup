use ark_ff::FftField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use std::{marker::PhantomData, sync::Arc};

use crate::{error::Error, utils::is_power_of_two};

#[derive(Clone, Debug)]
pub struct DomainPreprocessing<F: FftField> {
    pub table: Arc<DomainSidePreprocessing<F>>,
    pub query: DomainSidePreprocessing<F>,
}

#[derive(Clone, Debug)]
pub struct DomainSidePreprocessing<F: FftField> {
    pub base: DomainInfo<F>,
    pub lde: DomainInfo<F>,
    pub fft_plan: FftPlanPreprocessing<F>,
    pub ifft_plan: FftPlanPreprocessing<F>,
    pub lde_fft_plan: FftPlanPreprocessing<F>,
    pub coset_shift: F,
    pub vanishing_polynomial: VanishingPolynomial<F>,
    pub vanishing_on_lde: Vec<F>,
    pub fri_layers: Vec<FriLayerPreprocessing<F>>,
}

#[derive(Clone, Debug)]
pub struct DomainInfo<F: FftField> {
    pub size: usize,
    pub elements: Vec<F>,
    pub root_of_unity: F,
}

#[derive(Clone, Debug)]
pub struct FftPlanPreprocessing<F: FftField> {
    pub size: usize,
    pub size_inverse: F,
    pub root_of_unity: F,
    pub inverse_root_of_unity: F,
    pub bit_reversal: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct VanishingPolynomial<F: FftField> {
    pub degree: usize,
    pub constant: F,
    pub leading: F,
}

#[derive(Clone, Debug)]
pub struct FriLayerPreprocessing<F: FftField> {
    pub round: usize,
    pub domain: DomainInfo<F>,
    pub pair_indices: Vec<(usize, usize)>,
    pub next_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct ProtocolStructurePreprocessing<F: FftField> {
    pub merkle_leaf_layouts: Vec<MerkleLeafLayout>,
    pub table_batch_columns: Vec<ColumnLabel>,
    pub query_batch_columns: Vec<ColumnLabel>,
    pub usc_columns: Vec<ColumnLabel>,
    pub merkle_domain_separator: &'static [u8],
    pub merkle_node_separator: &'static [u8],
    pub opening_index_rule: &'static str,
    pub fri_query_to_merkle_rule: &'static str,
    pub batch_degree_normalization: DegreeNormalizationPreprocessing<F>,
    pub table_degree_normalizers_on_lde: Arc<DegreeNormalizersOnLde<F>>,
    pub query_degree_normalizers_on_lde: DegreeNormalizersOnLde<F>,
    pub transcript_order: Vec<&'static str>,
    pub verifier_challenge_order: Vec<&'static str>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DegreeNormalizersOnLde<F: FftField> {
    pub lde_size: usize,
    pub exponents: Vec<usize>,
    materialized: Option<Vec<Vec<F>>>,
    marker: PhantomData<F>,
}

impl<F: FftField> DegreeNormalizersOnLde<F> {
    pub fn compact(lde_size: usize, exponents: Vec<usize>) -> Self {
        Self {
            lde_size,
            exponents,
            materialized: None,
            marker: PhantomData,
        }
    }

    pub fn from_materialized(values: Vec<Vec<F>>) -> Self {
        let lde_size = values.first().map_or(0, Vec::len);
        Self {
            lde_size,
            exponents: Vec::new(),
            materialized: Some(values),
            marker: PhantomData,
        }
    }

    pub fn column_count(&self) -> usize {
        self.materialized
            .as_ref()
            .map_or(self.exponents.len(), Vec::len)
    }

    pub fn materialized(&self) -> Option<&[Vec<F>]> {
        self.materialized.as_deref()
    }

    pub fn materialize(&self) -> Result<Vec<Vec<F>>, Error> {
        if let Some(values) = &self.materialized {
            return Ok(values.clone());
        }
        let lde = domain_info::<F>(self.lde_size)?;
        Ok(self
            .exponents
            .iter()
            .map(|&exponent| {
                if exponent == 0 {
                    return vec![F::one(); self.lde_size];
                }
                lde.elements
                    .iter()
                    .map(|point| point.pow(&[exponent as u64]))
                    .collect()
            })
            .collect())
    }
}

#[derive(Clone, Debug)]
pub struct MerkleLeafLayout {
    pub name: &'static str,
    pub tag: &'static [u8],
    pub round: &'static [u8],
    pub domain: &'static [u8],
    pub columns: Vec<ColumnLabel>,
}

#[derive(Clone, Debug)]
pub struct ColumnLabel {
    pub label: &'static [u8],
    pub degree_bound: usize,
}

#[derive(Clone, Debug)]
pub struct DegreeNormalizationPreprocessing<F: FftField> {
    pub table_max_degree: usize,
    pub query_max_degree: usize,
    pub table_exponents: Vec<usize>,
    pub query_exponents: Vec<usize>,
    pub marker: std::marker::PhantomData<F>,
}

pub fn preprocess_domains<F: FftField>(
    table_size: usize,
    query_size: usize,
    blowup: usize,
) -> Result<DomainPreprocessing<F>, Error> {
    Ok(DomainPreprocessing {
        table: Arc::new(preprocess_domain_side(table_size, blowup)?),
        query: preprocess_domain_side(query_size, blowup)?,
    })
}

pub fn preprocess_table_domain<F: FftField>(
    table_size: usize,
    blowup: usize,
) -> Result<DomainSidePreprocessing<F>, Error> {
    preprocess_domain_side(table_size, blowup)
}

pub fn preprocess_query_domain<F: FftField>(
    query_size: usize,
    blowup: usize,
) -> Result<DomainSidePreprocessing<F>, Error> {
    preprocess_domain_side(query_size, blowup)
}

pub fn preprocess_protocol_structure<F: FftField>(
    table_size: usize,
    query_size: usize,
    blowup: usize,
) -> Result<ProtocolStructurePreprocessing<F>, Error> {
    let table_degree_normalizers_on_lde = Arc::new(degree_normalizers_on_lde::<F>(
        table_size,
        blowup,
        &table_batch_columns(table_size),
    )?);
    preprocess_protocol_structure_with_shared_table_normalizers(
        table_size,
        query_size,
        blowup,
        table_degree_normalizers_on_lde,
    )
}

pub fn preprocess_table_degree_normalizers_on_lde<F: FftField>(
    table_size: usize,
    blowup: usize,
) -> Result<DegreeNormalizersOnLde<F>, Error> {
    degree_normalizers_on_lde::<F>(table_size, blowup, &table_batch_columns(table_size))
}

pub fn preprocess_protocol_structure_with_table_normalizers<F: FftField>(
    table_size: usize,
    query_size: usize,
    blowup: usize,
    table_degree_normalizers_on_lde: Vec<Vec<F>>,
) -> Result<ProtocolStructurePreprocessing<F>, Error> {
    preprocess_protocol_structure_with_shared_table_normalizers(
        table_size,
        query_size,
        blowup,
        Arc::new(DegreeNormalizersOnLde::from_materialized(
            table_degree_normalizers_on_lde,
        )),
    )
}

pub(crate) fn preprocess_protocol_structure_with_shared_table_normalizers<F: FftField>(
    table_size: usize,
    query_size: usize,
    blowup: usize,
    table_degree_normalizers_on_lde: Arc<DegreeNormalizersOnLde<F>>,
) -> Result<ProtocolStructurePreprocessing<F>, Error> {
    let table_columns = table_batch_columns(table_size);
    let query_columns = query_batch_columns(query_size);
    let usc_columns = usc_columns(table_size, query_size);

    let query_degree_normalizers_on_lde =
        degree_normalizers_on_lde::<F>(query_size, blowup, &query_columns)?;

    Ok(protocol_structure_from_parts(
        table_columns,
        query_columns,
        usc_columns,
        table_degree_normalizers_on_lde,
        query_degree_normalizers_on_lde,
    ))
}

fn table_batch_columns(table_size: usize) -> Vec<ColumnLabel> {
    let table_columns = vec![
        ColumnLabel {
            label: b"M",
            degree_bound: table_size,
        },
        ColumnLabel {
            label: b"U",
            degree_bound: table_size,
        },
        ColumnLabel {
            label: b"A",
            degree_bound: table_size,
        },
        ColumnLabel {
            label: b"Q_T",
            degree_bound: table_size,
        },
    ];
    table_columns
}

fn query_batch_columns(query_size: usize) -> Vec<ColumnLabel> {
    vec![
        ColumnLabel {
            label: b"F",
            degree_bound: query_size,
        },
        ColumnLabel {
            label: b"V",
            degree_bound: query_size,
        },
        ColumnLabel {
            label: b"Q_F",
            degree_bound: query_size,
        },
    ]
}

fn usc_columns(table_size: usize, query_size: usize) -> Vec<ColumnLabel> {
    vec![
        ColumnLabel {
            label: b"p_A",
            degree_bound: table_size.saturating_sub(1).max(1),
        },
        ColumnLabel {
            label: b"p_V",
            degree_bound: query_size.saturating_sub(1).max(1),
        },
    ]
}

fn protocol_structure_from_parts<F: FftField>(
    table_columns: Vec<ColumnLabel>,
    query_columns: Vec<ColumnLabel>,
    usc_columns: Vec<ColumnLabel>,
    table_degree_normalizers_on_lde: Arc<DegreeNormalizersOnLde<F>>,
    query_degree_normalizers_on_lde: DegreeNormalizersOnLde<F>,
) -> ProtocolStructurePreprocessing<F> {
    ProtocolStructurePreprocessing {
        merkle_leaf_layouts: vec![
            MerkleLeafLayout {
                name: "table-main",
                tag: b"row-wise-batch",
                round: b"main-table",
                domain: b"H_T",
                columns: table_columns.clone(),
            },
            MerkleLeafLayout {
                name: "query-main",
                tag: b"row-wise-batch",
                round: b"main-query",
                domain: b"H_F",
                columns: query_columns.clone(),
            },
            MerkleLeafLayout {
                name: "usc",
                tag: b"row-wise-batch",
                round: b"usc",
                domain: b"H",
                columns: usc_columns.clone(),
            },
        ],
        table_batch_columns: table_columns.clone(),
        query_batch_columns: query_columns.clone(),
        usc_columns,
        merkle_domain_separator: b"leaf",
        merkle_node_separator: b"node",
        opening_index_rule: "Merkle path index is the base-domain row index.",
        fri_query_to_merkle_rule: "FRI layer query index opens that layer's Merkle leaf at the same index and its paired sibling.",
        batch_degree_normalization: degree_normalization(table_columns, query_columns),
        table_degree_normalizers_on_lde,
        query_degree_normalizers_on_lde,
        transcript_order: vec![
            "table-lde-root",
            "query-root",
            "multiplicity-root",
            "beta",
            "U/A/V commitments",
            "table constraint batching",
            "quotient commitments",
            "main batch FRI transcripts",
            "OOD openings",
            "common sum claim",
            "specialized USC proofs",
        ],
        verifier_challenge_order: vec![
            "beta",
            "table constraint batching",
            "main batch coefficients",
            "OOD point challenges",
            "OOD batching coefficients",
            "specialized USC coefficients",
            "FRI queries",
        ],
    }
}

fn preprocess_domain_side<F: FftField>(
    base_size: usize,
    blowup: usize,
) -> Result<DomainSidePreprocessing<F>, Error> {
    if !is_power_of_two(base_size) {
        return Err(Error::InvalidDomainSize(base_size));
    }
    if !is_power_of_two(blowup) {
        return Err(Error::InvalidBlowup(blowup));
    }
    let lde_size = base_size * blowup;
    let base = domain_info(base_size)?;
    let lde = domain_info(lde_size)?;
    let vanishing_polynomial = VanishingPolynomial {
        degree: base_size,
        constant: -F::one(),
        leading: F::one(),
    };
    let vanishing_on_lde = lde
        .elements
        .iter()
        .map(|point: &F| point.pow(&[base_size as u64]) - F::one())
        .collect();

    Ok(DomainSidePreprocessing {
        fft_plan: fft_plan(&base)?,
        ifft_plan: fft_plan(&base)?,
        lde_fft_plan: fft_plan(&lde)?,
        coset_shift: F::multiplicative_generator(),
        vanishing_polynomial,
        vanishing_on_lde,
        fri_layers: fri_layers(lde_size)?,
        base,
        lde,
    })
}

fn domain_info<F: FftField>(size: usize) -> Result<DomainInfo<F>, Error> {
    let domain = GeneralEvaluationDomain::<F>::new(size).ok_or(Error::InvalidDomainSize(size))?;
    let elements = domain.elements().collect::<Vec<_>>();
    Ok(DomainInfo {
        size,
        root_of_unity: domain.element(1),
        elements,
    })
}

fn fft_plan<F: FftField>(domain: &DomainInfo<F>) -> Result<FftPlanPreprocessing<F>, Error> {
    let size_inverse = F::from(domain.size as u64)
        .inverse()
        .ok_or(Error::InvalidDomainSize(domain.size))?;
    Ok(FftPlanPreprocessing {
        size: domain.size,
        size_inverse,
        root_of_unity: domain.root_of_unity,
        inverse_root_of_unity: domain
            .root_of_unity
            .inverse()
            .ok_or(Error::InvalidDomainSize(domain.size))?,
        bit_reversal: bit_reversal_permutation(domain.size),
    })
}

fn fri_layers<F: FftField>(lde_size: usize) -> Result<Vec<FriLayerPreprocessing<F>>, Error> {
    let mut layers = Vec::new();
    let mut size = lde_size;
    let mut round = 0;
    while size > 1 {
        let domain = domain_info(size)?;
        let half = size / 2;
        layers.push(FriLayerPreprocessing {
            round,
            domain,
            pair_indices: (0..half).map(|index| (index, index + half)).collect(),
            next_indices: (0..size).map(|index| index % half).collect(),
        });
        size = half;
        round += 1;
    }
    Ok(layers)
}

fn degree_normalization<F: FftField>(
    table_columns: Vec<ColumnLabel>,
    query_columns: Vec<ColumnLabel>,
) -> DegreeNormalizationPreprocessing<F> {
    let table_max_degree = table_columns
        .iter()
        .map(|column| column.degree_bound)
        .max()
        .unwrap_or(0);
    let query_max_degree = query_columns
        .iter()
        .map(|column| column.degree_bound)
        .max()
        .unwrap_or(0);
    DegreeNormalizationPreprocessing {
        table_max_degree,
        query_max_degree,
        table_exponents: table_columns
            .iter()
            .map(|column| table_max_degree - column.degree_bound)
            .collect(),
        query_exponents: query_columns
            .iter()
            .map(|column| query_max_degree - column.degree_bound)
            .collect(),
        marker: std::marker::PhantomData,
    }
}

fn degree_normalizers_on_lde<F: FftField>(
    base_size: usize,
    blowup: usize,
    columns: &[ColumnLabel],
) -> Result<DegreeNormalizersOnLde<F>, Error> {
    if base_size == 0 || !base_size.is_power_of_two() {
        return Err(Error::InvalidDomainSize(base_size));
    }
    if !blowup.is_power_of_two() {
        return Err(Error::InvalidBlowup(blowup));
    }
    let lde_size = base_size * blowup;
    let max_degree = columns
        .iter()
        .map(|column| column.degree_bound)
        .max()
        .unwrap_or(0);
    let exponents = columns
        .iter()
        .map(|column| max_degree - column.degree_bound)
        .collect();
    Ok(DegreeNormalizersOnLde::compact(lde_size, exponents))
}

fn bit_reversal_permutation(size: usize) -> Vec<usize> {
    let bits = size.trailing_zeros();
    (0..size)
        .map(|index| index.reverse_bits() >> (usize::BITS - bits))
        .collect()
}
