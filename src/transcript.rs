use ark_ff::{PrimeField, ToBytes};
use sha3::{Digest, Keccak256};

pub struct Transcript {
    state: Vec<u8>,
    counter: u64,
}

impl Transcript {
    pub fn new(label: &[u8]) -> Self {
        Self {
            state: label.to_vec(),
            counter: 0,
        }
    }

    pub fn absorb_bytes(&mut self, label: &'static [u8], bytes: &[u8]) {
        self.state
            .extend_from_slice(&(label.len() as u64).to_le_bytes());
        self.state.extend_from_slice(label);
        self.state
            .extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        self.state.extend_from_slice(bytes);
    }

    pub fn absorb<T: ToBytes>(&mut self, label: &'static [u8], value: &T) {
        let mut bytes = Vec::new();
        value.write(&mut bytes).expect("writing to Vec cannot fail");
        self.absorb_bytes(label, &bytes);
    }

    pub fn challenge<F: PrimeField>(&mut self, label: &'static [u8]) -> F {
        self.challenge_with_rejection(label, |_| true)
    }

    pub fn challenge_nonzero<F: PrimeField>(&mut self, label: &'static [u8]) -> F {
        self.challenge_with_rejection(label, |value: &F| !value.is_zero())
    }

    pub fn challenge_with_rejection<F: PrimeField>(
        &mut self,
        label: &'static [u8],
        accept: impl Fn(&F) -> bool,
    ) -> F {
        self.absorb_bytes(b"challenge-label", label);
        loop {
            let mut hasher = Keccak256::new();
            hasher.update(&self.state);
            hasher.update(self.counter.to_le_bytes());
            self.counter += 1;
            let digest = hasher.finalize();
            if let Some(value) = F::from_random_bytes(&digest) {
                if accept(&value) {
                    self.absorb(label, &value);
                    return value;
                }
            }
        }
    }

    pub fn challenge_index(&mut self, label: &'static [u8], bound: usize) -> usize {
        assert!(bound > 0);
        self.absorb_bytes(b"challenge-index-label", label);

        let mut hasher = Keccak256::new();
        hasher.update(&self.state);
        hasher.update(self.counter.to_le_bytes());
        self.counter += 1;
        let digest = hasher.finalize();

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[..8]);
        let index = u64::from_le_bytes(bytes) as usize % bound;
        self.absorb_bytes(label, &index.to_le_bytes());
        index
    }
}
