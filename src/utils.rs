use ark_ff::Field;

pub fn to_field<F: Field>(values: &[u64]) -> Vec<F> {
    values.iter().map(|&value| F::from(value)).collect()
}

pub fn is_power_of_two(value: usize) -> bool {
    value != 0 && (value & (value - 1)) == 0
}
