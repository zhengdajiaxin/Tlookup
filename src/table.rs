use std::collections::HashMap;

use ark_ff::{FftField, ToBytes};

use crate::{error::Error, utils::is_power_of_two};

#[derive(Clone, Debug)]
pub struct Table<F: FftField> {
    size: usize,
    values: Vec<F>,
    value_index_mapping: HashMap<Vec<u8>, usize>,
}

impl<F: FftField> ToBytes for Table<F> {
    fn write<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        self.values.write(&mut writer)
    }
}

impl<F: FftField> Table<F> {
    pub fn new(values: &[F]) -> Result<Self, Error> {
        if !is_power_of_two(values.len()) {
            return Err(Error::TableSizeNotPow2(values.len()));
        }

        let mut value_index_mapping = HashMap::with_capacity(values.len());
        for (index, value) in values.iter().copied().enumerate() {
            if value_index_mapping
                .insert(field_key(&value), index)
                .is_some()
            {
                return Err(Error::DuplicateValueInTable(format!("{}", value)));
            }
        }

        Ok(Self {
            size: values.len(),
            values: values.to_vec(),
            value_index_mapping,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn index_of(&self, value: &F) -> Option<usize> {
        self.value_index_mapping.get(&field_key(value)).copied()
    }
}

fn field_key<F: ToBytes>(value: &F) -> Vec<u8> {
    let mut bytes = Vec::new();
    value
        .write(&mut bytes)
        .expect("writing field element to Vec cannot fail");
    bytes
}
