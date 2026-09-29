pub mod error;
pub mod fri;
pub(crate) mod gpu_lock;
pub mod lde;
pub mod lookup;
pub mod merkle;
pub mod preprocessing;
pub mod table;
pub mod transcript;
pub mod usc;
pub mod utils;

pub const PROTOCOL_NAME: &[u8] = b"FRI-LOOKUP-0.1";

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::{ToBytes, UniformRand};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use ark_std::{rand::Rng, test_rng};
    use sha3::{Digest, Keccak256};

    #[cfg(not(feature = "icicle-rust"))]
    use crate::lde::{self, DenseLdeBackendKind, DEFAULT_GPU_LDE_MIN_SIZE};
    use crate::{
        fri,
        lde::DenseLdePlan,
        lookup::{FriLookup, LookupStatement, LookupWitness},
        merkle::{Digest32, MerkleTree},
        table::Table,
        usc,
    };

    fn prepare(n: usize, witness_size: usize) -> (Table<Fr>, LookupWitness<Fr>, LookupStatement) {
        let mut rng = test_rng();
        let table_values: Vec<_> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let table = Table::new(&table_values).unwrap();

        let query_values: Vec<_> = (0..witness_size)
            .map(|_| {
                let index = rng.gen_range(0..n);
                table.values()[index]
            })
            .collect();

        let witness = LookupWitness::new(&query_values).unwrap();
        let statement = LookupStatement {
            query_size: witness_size,
        };
        (table, witness, statement)
    }

    #[test]
    fn roundtrip_matches_cq_shape() {
        let (table, witness, statement) = prepare(64, 8);
        let params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();
        let proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        assert_eq!(proof.table_batch.commitment_groups[0].columns, vec![0]);
        assert_eq!(proof.table_batch.commitment_groups[1].columns, vec![1, 2]);
        assert_eq!(proof.query_batch.commitment_groups[0].columns, vec![0]);
        assert_eq!(proof.table_batch.commitment_groups[2].columns, vec![3]);
        assert_eq!(proof.query_batch.commitment_groups[1].columns, vec![1]);
        assert_eq!(proof.query_batch.commitment_groups[2].columns, vec![2]);
        assert_eq!(
            proof.multiplicity_root,
            proof.table_batch.commitment_groups[0].root
        );
        assert_eq!(
            proof.query_root,
            proof.query_batch.commitment_groups[0].root
        );
        FriLookup::verify(&params, &statement, &proof).unwrap();
    }

    #[test]
    fn rejects_value_not_in_table() {
        let (table, _, statement) = prepare(32, 8);
        let mut query_values = table.values()[0..8].to_vec();
        query_values[3] = table.values()[0] + Fr::from(1u64);
        while table.index_of(&query_values[3]).is_some() {
            query_values[3] += Fr::from(1u64);
        }
        let witness = LookupWitness::new(&query_values).unwrap();
        let params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();
        assert!(FriLookup::prove(&params, &witness, &statement).is_err());
    }

    #[test]
    fn rejects_tampered_multiplicity() {
        let (table, witness, statement) = prepare(32, 8);
        let params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();
        let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        proof.table_ood_opening_proof.row_openings[0].original_row[1] += Fr::from(1u64);
        assert!(FriLookup::verify(&params, &statement, &proof).is_err());
    }

    #[test]
    fn validates_and_caches_public_table_preprocessing() {
        let (table, witness, statement) = prepare(32, 8);
        let mut params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();
        let proof = FriLookup::prove(&params, &witness, &statement).unwrap();

        FriLookup::validate_preprocessing_parameters(&params).unwrap();
        let references_after_first_validation =
            std::sync::Arc::strong_count(&params.table_low_degree);
        assert!(references_after_first_validation >= 2);
        FriLookup::verify(&params, &statement, &proof).unwrap();
        assert_eq!(
            std::sync::Arc::strong_count(&params.table_low_degree),
            references_after_first_validation
        );

        params.table_id[0] ^= 1;
        assert_eq!(
            FriLookup::verify(&params, &statement, &proof).unwrap_err(),
            crate::error::Error::TableIdentifierMismatch
        );
        params.table_id[0] ^= 1;
        FriLookup::verify(&params, &statement, &proof).unwrap();

        params.table_lde_root[0] ^= 1;
        assert_eq!(
            FriLookup::verify(&params, &statement, &proof).unwrap_err(),
            crate::error::Error::MerkleRootMismatch
        );
    }

    #[test]
    fn first_table_validation_rejects_tampered_reusable_fri() {
        let (table, witness, statement) = prepare(32, 8);
        let mut params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();
        let proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        FriLookup::verify(&params, &statement, &proof).unwrap();
        std::sync::Arc::make_mut(&mut params.table_low_degree).final_evals[0] += Fr::from(1u64);
        assert!(FriLookup::verify(&params, &statement, &proof).is_err());
    }

    #[test]
    fn preprocesses_domains_protocol_and_public_table() {
        let (table, _, statement) = prepare(32, 8);
        let params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();

        assert_eq!(params.domains.table.base.size, 32);
        assert_eq!(params.domains.query.base.size, 8);
        assert_eq!(params.domains.table.lde.size, 128);
        assert_eq!(params.domains.query.lde.size, 32);
        assert_eq!(params.domains.table.vanishing_on_lde.len(), 128);
        assert_eq!(params.domains.query.fri_layers[0].pair_indices[0], (0, 16));
        assert_eq!(params.protocol.table_batch_columns.len(), 4);
        assert_eq!(params.protocol.table_batch_columns[0].label, b"M");
        assert_eq!(params.protocol.table_batch_columns[1].label, b"U");
        assert_eq!(params.protocol.table_batch_columns[2].label, b"A");
        assert_eq!(params.protocol.table_batch_columns[3].label, b"Q_T");
        assert_eq!(params.protocol.query_batch_columns.len(), 3);
        assert_eq!(params.protocol.query_batch_columns[0].label, b"F");
        assert_eq!(params.protocol.query_batch_columns[1].label, b"V");
        assert_eq!(params.protocol.query_batch_columns[2].label, b"Q_F");
        assert!(!params
            .protocol
            .query_batch_columns
            .iter()
            .any(|column| column.label == b"B"));
        assert_eq!(params.table_preprocessing.lde_root(), params.table_lde_root);
        assert_eq!(params.table_preprocessing.lde_len(table.size(), 4), 128);
        assert_eq!(
            params.table_preprocessing.polynomial_coefficients.len(),
            table.size()
        );
    }

    #[test]
    fn reuses_table_preprocessing_across_query_sizes() {
        let (table, _, _) = prepare(64, 8);
        let reusable = FriLookup::preprocess_table(&table, 4).unwrap();
        let mut rng = test_rng();

        for query_size in [8, 16] {
            let query_values = (0..query_size)
                .map(|_| {
                    let index = rng.gen_range(0..table.size());
                    table.values()[index]
                })
                .collect::<Vec<_>>();
            let witness = LookupWitness::new(&query_values).unwrap();
            let statement = LookupStatement { query_size };
            let params = FriLookup::parameters_for_query(&reusable, query_size).unwrap();
            let proof = FriLookup::prove(&params, &witness, &statement).unwrap();

            assert_eq!(params.table_id, reusable.table_id);
            assert_eq!(params.table_lde_root, reusable.table_lde_root);
            assert_eq!(
                params.domains.table.base.size,
                reusable.table_domain.base.size
            );
            assert_eq!(
                params.domains.table.lde.size,
                reusable.table_domain.lde.size
            );
            assert_eq!(
                params.protocol.table_degree_normalizers_on_lde,
                reusable.table_degree_normalizers_on_lde
            );
            assert_eq!(
                params.table_preprocessing.lde_root(),
                reusable.table_preprocessing.lde_root()
            );
            FriLookup::verify(&params, &statement, &proof).unwrap();
        }
    }

    #[test]
    fn specialized_usc_quotient_rejects_wrong_sum() {
        let values = (0..16)
            .map(|index| Fr::from((index + 3) as u64))
            .collect::<Vec<_>>();
        let sum_claim = values.iter().copied().sum::<Fr>();
        let quotient = usc::specialized_usc_quotient_evaluations(&values, sum_claim).unwrap();
        assert_eq!(quotient.len(), values.len());

        let wrong_sum_claim = sum_claim + Fr::from(1u64);
        assert!(usc::specialized_usc_quotient_evaluations(&values, wrong_sum_claim).is_err());
    }

    #[test]
    fn rejects_tampered_table_local_constraint_columns_and_quotients() {
        let (table, witness, statement) = prepare(32, 8);
        let params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();

        let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        proof.table_ood_opening_proof.row_openings[0].original_row[2] += Fr::from(1u64);
        assert!(FriLookup::verify(&params, &statement, &proof).is_err());

        let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        proof.table_ood_opening_proof.row_openings[0].original_row[3] += Fr::from(1u64);
        assert!(FriLookup::verify(&params, &statement, &proof).is_err());

        let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        proof.c_qt[0] ^= 1;
        assert!(FriLookup::verify(&params, &statement, &proof).is_err());

        let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        proof.table_batch.commitment_groups[1].root[0] ^= 1;
        assert!(FriLookup::verify(&params, &statement, &proof).is_err());
    }

    #[test]
    fn rejects_tampered_query_local_constraint_columns_and_quotient() {
        let (table, witness, statement) = prepare(32, 8);
        let params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();

        let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        proof.query_ood_opening_proof.row_openings[0].original_row[1] += Fr::from(1u64);
        assert!(FriLookup::verify(&params, &statement, &proof).is_err());

        let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        proof.c_qf[0] ^= 1;
        assert!(FriLookup::verify(&params, &statement, &proof).is_err());
    }

    #[test]
    fn non_exact_constraint_division_returns_error() {
        let (table, witness, statement) = prepare(16, 8);
        let params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();
        let multiplicity = vec![Fr::from(0u64); table.size()];
        let reciprocal = vec![Fr::from(1u64); table.size()];
        let bad_accumulator = vec![Fr::from(3u64); table.size()];
        let err = crate::lookup::build_table_constraint_quotient(
            table.values(),
            &multiplicity,
            &reciprocal,
            &bad_accumulator,
            Fr::from(5u64),
            Fr::from(7u64),
            Fr::from(11u64),
            params.blowup,
        )
        .unwrap_err();
        assert_eq!(err, crate::error::Error::NonExactTableConstraintDivision);
        let err = crate::lookup::build_query_constraint_quotient(
            &witness.values,
            &vec![Fr::from(1u64); statement.query_size],
            Fr::from(5u64),
            params.blowup,
        )
        .unwrap_err();
        assert_eq!(err, crate::error::Error::NonExactQueryConstraintDivision);
    }

    #[test]
    fn rejects_tampered_ood_claims_and_opening_quotients() {
        let (table, witness, statement) = prepare(32, 8);
        let params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();

        for index in 0..5 {
            let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
            proof.table_ood_claimed_values[index] += Fr::from(1u64);
            assert!(FriLookup::verify(&params, &statement, &proof).is_err());
        }
        for index in 0..3 {
            let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
            proof.query_ood_claimed_values[index] += Fr::from(1u64);
            assert!(FriLookup::verify(&params, &statement, &proof).is_err());
        }

        let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        proof.table_ood_opening_proof.row_openings[0]
            .quotient_opening
            .row[0] += Fr::from(1u64);
        assert!(FriLookup::verify(&params, &statement, &proof).is_err());

        let mut proof = FriLookup::prove(&params, &witness, &statement).unwrap();
        proof.table_ood_opening_proof = proof.query_ood_opening_proof.clone();
        assert!(FriLookup::verify(&params, &statement, &proof).is_err());
    }

    #[test]
    fn specialized_usc_reuses_main_commitments_and_rejects_tampering() {
        let (table, witness, statement) = prepare(32, 8);
        let params = FriLookup::preprocess(&table, statement.query_size, 4).unwrap();
        let proof = FriLookup::prove(&params, &witness, &statement).unwrap();

        assert_eq!(proof.specialized_usc_table_proof.batch.column_count, 1);
        assert_eq!(proof.specialized_usc_query_proof.batch.column_count, 1);
        assert_eq!(
            proof
                .specialized_usc_table_proof
                .batch
                .commitment_groups
                .len(),
            1
        );
        assert_eq!(
            proof
                .specialized_usc_query_proof
                .batch
                .commitment_groups
                .len(),
            1
        );
        assert!(!params
            .protocol
            .usc_columns
            .iter()
            .any(|column| column.label == b"h"));
        assert!(!params
            .protocol
            .usc_columns
            .iter()
            .any(|column| column.label == b"phi"));

        let mut bad_sum = proof.clone();
        bad_sum.common_sum_s += Fr::from(1u64);
        assert!(FriLookup::verify(&params, &statement, &bad_sum).is_err());

        let mut bad_pa = proof.clone();
        bad_pa.specialized_usc_table_proof.openings[0].p_opening.row[0] += Fr::from(1u64);
        assert!(FriLookup::verify(&params, &statement, &bad_pa).is_err());

        let mut bad_pv = proof.clone();
        bad_pv.specialized_usc_query_proof.openings[0].p_opening.row[0] += Fr::from(1u64);
        assert!(FriLookup::verify(&params, &statement, &bad_pv).is_err());

        let mut missing_main_opening = proof;
        missing_main_opening.specialized_usc_table_proof.openings[0]
            .existing_group_row
            .clear();
        assert!(FriLookup::verify(&params, &statement, &missing_main_opening).is_err());
    }

    #[test]
    fn batch_fri_accepts_degree_normalized_columns() {
        let linear = (0..8)
            .map(|index| Fr::from(index as u64))
            .collect::<Vec<_>>();
        let constant = vec![Fr::from(7u64); 8];
        let proof = fri::prove_batch_low_degree(
            &[linear, constant],
            &[Fr::from(3u64), Fr::from(5u64)],
            &[8, 1],
            8,
            4,
            &[3],
        )
        .unwrap();

        assert_eq!(proof.column_degree_bounds, vec![8, 1]);
        fri::verify_batch_low_degree(&proof, 8, 4).unwrap();
    }

    #[test]
    fn sparse_lde_matches_dense_lde() {
        let mut sparse = vec![Fr::from(0u64); 16];
        sparse[1] = Fr::from(9u64);
        sparse[7] = Fr::from(13u64);
        sparse[12] = Fr::from(21u64);

        let dense_lde = fri::evaluate_lde(&sparse, 4).unwrap();
        let sparse_lde = fri::evaluate_sparse_lde(&sparse, 4).unwrap();

        assert_eq!(sparse_lde, dense_lde);
    }

    #[test]
    fn merkle_column_views_match_legacy_rows() {
        fn legacy_leaf(row: &[Fr]) -> Digest32 {
            let mut bytes = Vec::new();
            for value in row {
                value.write(&mut bytes).unwrap();
            }
            let mut hasher = Keccak256::new();
            hasher.update(b"leaf");
            hasher.update(&bytes);
            hasher.finalize().into()
        }

        fn legacy_pair(left: &Digest32, right: &Digest32) -> Digest32 {
            let mut bytes = Vec::with_capacity(64);
            bytes.extend_from_slice(left);
            bytes.extend_from_slice(right);
            let mut hasher = Keccak256::new();
            hasher.update(b"node");
            hasher.update(&bytes);
            hasher.finalize().into()
        }

        fn legacy_root(rows: &[Vec<Fr>]) -> Digest32 {
            let mut layer = rows.iter().map(|row| legacy_leaf(row)).collect::<Vec<_>>();
            while layer.len() > 1 {
                layer = layer
                    .chunks_exact(2)
                    .map(|pair| legacy_pair(&pair[0], &pair[1]))
                    .collect::<Vec<_>>();
            }
            layer[0]
        }

        let columns = vec![
            (0..16)
                .map(|index| Fr::from((index + 3) as u64))
                .collect::<Vec<_>>(),
            (0..16)
                .map(|index| Fr::from((index * 7 + 1) as u64))
                .collect::<Vec<_>>(),
            (0..16)
                .map(|index| Fr::from((index * index + 5) as u64))
                .collect::<Vec<_>>(),
        ];
        let column_refs = columns
            .iter()
            .map(|column| column.as_slice())
            .collect::<Vec<_>>();

        let single_rows = columns[0]
            .iter()
            .copied()
            .map(|value| vec![value])
            .collect::<Vec<_>>();
        let single_tree = MerkleTree::from_column(&columns[0]);
        assert_eq!(single_tree.root(), legacy_root(&single_rows));

        let group = vec![2, 0];
        let group_rows = (0..columns[0].len())
            .map(|row| {
                group
                    .iter()
                    .map(|&column| columns[column][row])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rows_tree = MerkleTree::from_rows(&group_rows);
        let columns_tree = MerkleTree::from_columns(&column_refs, &group);

        assert_eq!(columns_tree.root(), legacy_root(&group_rows));
        assert_eq!(columns_tree.root(), rows_tree.root());
        let rows_path = rows_tree.open(5);
        let columns_path = columns_tree.open(5);
        assert_eq!(columns_path.index, rows_path.index);
        assert_eq!(columns_path.siblings, rows_path.siblings);
        MerkleTree::verify(columns_tree.root(), &group_rows[5], &columns_path).unwrap();
    }

    #[test]
    fn coeffs_direct_lde_matches_base_eval_path() {
        let mut rng = test_rng();
        let coeffs = (0..11).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        let base_domain = GeneralEvaluationDomain::<Fr>::new(16).unwrap();
        let mut padded_coeffs = coeffs.clone();
        padded_coeffs.resize(16, Fr::from(0u64));
        let base_evals = base_domain.fft(&padded_coeffs);

        let from_base_evals = fri::evaluate_lde(&base_evals, 4).unwrap();
        let from_coeffs = fri::evaluate_lde_from_coeffs(&coeffs, 16, 4).unwrap();

        assert_eq!(from_coeffs, from_base_evals);
    }

    #[test]
    fn auto_sparse_lde_threshold_switches_to_dense_for_many_non_zero_values() {
        assert!(fri::should_use_sparse_lde(3, 1024, 4));
        assert!(!fri::should_use_sparse_lde(256, 1024, 4));

        let denseish = (0..32)
            .map(|index| Fr::from((index + 1) as u64))
            .collect::<Vec<_>>();
        let auto_lde = fri::evaluate_auto_sparse_lde(&denseish, 4).unwrap();
        let dense_lde = fri::evaluate_lde(&denseish, 4).unwrap();

        assert_eq!(auto_lde, dense_lde);
    }

    #[test]
    fn sparse_lde_reuses_multiplicity_support_for_accumulator() {
        let mut multiplicity = vec![Fr::from(0u64); 32];
        multiplicity[2] = Fr::from(3u64);
        multiplicity[11] = Fr::from(1u64);
        multiplicity[25] = Fr::from(2u64);
        let support = fri::non_zero_indices(&multiplicity);

        let reciprocal = (0..32)
            .map(|index| Fr::from((index + 5) as u64))
            .collect::<Vec<_>>();
        let accumulator = multiplicity
            .iter()
            .zip(&reciprocal)
            .map(|(mult, reciprocal)| *mult * *reciprocal)
            .collect::<Vec<_>>();

        let support_lde =
            fri::evaluate_auto_sparse_lde_with_support(&accumulator, 4, &support).unwrap();
        let dense_lde = fri::evaluate_lde(&accumulator, 4).unwrap();

        assert_eq!(support_lde, dense_lde);
    }

    #[test]
    fn dense_lde_plan_caches_fft_metadata() {
        let plan = DenseLdePlan::<Fr>::new(16, 4).unwrap();

        assert_eq!(plan.base_size, 16);
        assert_eq!(plan.blowup, 4);
        assert_eq!(plan.lde_size, 64);
        assert_eq!(plan.base_bit_reversal.len(), 16);
        assert_eq!(plan.lde_bit_reversal.len(), 64);
        assert_eq!(
            plan.base_root_of_unity * plan.base_inverse_root_of_unity,
            Fr::from(1u64)
        );
        assert_eq!(
            plan.lde_root_of_unity * plan.lde_inverse_root_of_unity,
            Fr::from(1u64)
        );
        assert_eq!(
            Fr::from(plan.base_size as u64) * plan.base_size_inverse,
            Fr::from(1u64)
        );
        assert_eq!(
            Fr::from(plan.lde_size as u64) * plan.lde_size_inverse,
            Fr::from(1u64)
        );
    }

    #[cfg(not(feature = "icicle-rust"))]
    #[test]
    fn dense_lde_auto_selects_cpu_without_gpu_feature() {
        let plan = DenseLdePlan::<Fr>::new(DEFAULT_GPU_LDE_MIN_SIZE / 4, 4).unwrap();

        assert_eq!(
            lde::select_dense_lde_backend(&plan),
            DenseLdeBackendKind::Cpu
        );
    }
}
