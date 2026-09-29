use ark_ff::ToBytes;
use sha3::{Digest, Keccak256 as CpuKeccak256};
use std::{env, io, thread};

use crate::error::Error;

pub type Digest32 = [u8; 32];

const MERKLE_BACKEND_ENV: &str = "FRI_LOOKUP_MERKLE_BACKEND";
const GPU_MERKLE_MIN_LEAVES_ENV: &str = "FRI_LOOKUP_GPU_MERKLE_MIN_LEAVES";
const DEFAULT_GPU_MERKLE_MIN_LEAVES: usize = 1 << 16;
const LEAF_DOMAIN: &[u8] = b"leaf";
const NODE_DOMAIN: &[u8] = b"node";
const DIGEST_BYTES: usize = 32;
const NODE_PREIMAGE_BYTES: usize = NODE_DOMAIN.len() + 2 * DIGEST_BYTES;
const CPU_PARALLEL_MIN_ROWS: usize = 1 << 12;

#[derive(Clone, Debug)]
pub struct MerkleTree {
    layers: Vec<Vec<Digest32>>,
}

#[derive(Clone, Debug)]
pub struct MerklePath {
    pub index: usize,
    pub siblings: Vec<Digest32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MerkleBackendKind {
    Cpu,
    Gpu,
}

struct DigestWriter<'a> {
    hasher: &'a mut CpuKeccak256,
}

impl io::Write for DigestWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.hasher.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn hash_pair(left: &Digest32, right: &Digest32) -> Digest32 {
    let mut hasher = CpuKeccak256::new();
    hasher.update(NODE_DOMAIN);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn hash_leaf_values<'a, F, I>(values: I) -> Digest32
where
    F: ToBytes + 'a,
    I: IntoIterator<Item = &'a F>,
{
    let mut hasher = CpuKeccak256::new();
    hasher.update(LEAF_DOMAIN);
    {
        let mut writer = DigestWriter {
            hasher: &mut hasher,
        };
        for value in values {
            value
                .write(&mut writer)
                .expect("writing to digest cannot fail");
        }
    }
    hasher.finalize().into()
}

fn write_leaf_values<'a, F, I>(values: I, bytes: &mut Vec<u8>)
where
    F: ToBytes + 'a,
    I: IntoIterator<Item = &'a F>,
{
    for value in values {
        value
            .write(&mut *bytes)
            .expect("writing to Vec cannot fail");
    }
}

pub fn row_to_leaf<F: ToBytes>(row: &[F]) -> Digest32 {
    hash_leaf_values(row.iter())
}

pub fn warm_up_gpu(row_count: usize, row_byte_len: usize) {
    if select_merkle_backend(row_count) != MerkleBackendKind::Gpu {
        return;
    }
    if !try_warm_up_gpu_hashes(row_byte_len)
        && requested_merkle_backend() == Some(MerkleBackendKind::Gpu)
    {
        panic!("GPU Merkle backend selected but ICICLE GPU Keccak warmup failed");
    }
}

fn select_merkle_backend(row_count: usize) -> MerkleBackendKind {
    match requested_merkle_backend() {
        Some(kind) => kind,
        None => {
            if cfg!(feature = "icicle-rust") && row_count >= gpu_merkle_min_leaves() {
                MerkleBackendKind::Gpu
            } else {
                MerkleBackendKind::Cpu
            }
        }
    }
}

fn requested_merkle_backend() -> Option<MerkleBackendKind> {
    match env::var(MERKLE_BACKEND_ENV)
        .ok()?
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" => Some(MerkleBackendKind::Cpu),
        "gpu" => Some(MerkleBackendKind::Gpu),
        "auto" => None,
        _ => None,
    }
}

fn gpu_merkle_min_leaves() -> usize {
    env::var(GPU_MERKLE_MIN_LEAVES_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_GPU_MERKLE_MIN_LEAVES)
}

fn build_layers_from_indexed_leaves<H, W>(
    leaf_count: usize,
    hash_leaf: H,
    write_leaf_preimage: W,
) -> Vec<Vec<Digest32>>
where
    H: Fn(usize) -> Digest32 + Sync,
    W: Fn(usize, &mut Vec<u8>),
{
    if select_merkle_backend(leaf_count) == MerkleBackendKind::Gpu {
        if let Some(layers) =
            try_build_gpu_layers_from_indexed_leaves(leaf_count, write_leaf_preimage)
        {
            return layers;
        }
        if requested_merkle_backend() == Some(MerkleBackendKind::Gpu) {
            panic!(
                "GPU Merkle backend selected but ICICLE GPU Keccak failed; \
                 check ICICLE_BACKEND_INSTALL_DIR and CUDA backend availability"
            );
        }
    }

    build_cpu_layers_from_indexed_leaves(leaf_count, hash_leaf)
}

fn build_cpu_layers_from_indexed_leaves<H>(leaf_count: usize, hash_leaf: H) -> Vec<Vec<Digest32>>
where
    H: Fn(usize) -> Digest32 + Sync,
{
    let mut layers = Vec::new();
    layers.push(hash_leaf_layer_cpu_indexed(leaf_count, &hash_leaf));

    while layers.last().unwrap().len() > 1 {
        let previous = layers.last().unwrap();
        layers.push(hash_node_layer_cpu(previous));
    }

    layers
}

fn hash_leaf_layer_cpu_indexed<H>(leaf_count: usize, hash_leaf: &H) -> Vec<Digest32>
where
    H: Fn(usize) -> Digest32 + Sync,
{
    let mut hashes = vec![[0u8; DIGEST_BYTES]; leaf_count];
    let worker_count = cpu_parallel_worker_count(leaf_count);
    if worker_count <= 1 {
        for (index, digest) in hashes.iter_mut().enumerate() {
            *digest = hash_leaf(index);
        }
        return hashes;
    }

    let chunk_size = (leaf_count + worker_count - 1) / worker_count;
    thread::scope(|scope| {
        for (chunk_index, chunk) in hashes.chunks_mut(chunk_size).enumerate() {
            let start = chunk_index * chunk_size;
            scope.spawn(move || {
                for (offset, digest) in chunk.iter_mut().enumerate() {
                    *digest = hash_leaf(start + offset);
                }
            });
        }
    });
    hashes
}

fn hash_node_layer_cpu(layer: &[Digest32]) -> Vec<Digest32> {
    let pair_count = layer.len() / 2;
    let mut hashes = vec![[0u8; DIGEST_BYTES]; pair_count];
    let worker_count = cpu_parallel_worker_count(pair_count);
    if worker_count <= 1 {
        for (pair, digest) in layer.chunks_exact(2).zip(&mut hashes) {
            *digest = hash_pair(&pair[0], &pair[1]);
        }
        return hashes;
    }

    let chunk_size = (pair_count + worker_count - 1) / worker_count;
    thread::scope(|scope| {
        for (chunk_index, chunk) in hashes.chunks_mut(chunk_size).enumerate() {
            let start = chunk_index * chunk_size;
            scope.spawn(move || {
                for (offset, digest) in chunk.iter_mut().enumerate() {
                    let pair_index = start + offset;
                    *digest = hash_pair(&layer[2 * pair_index], &layer[2 * pair_index + 1]);
                }
            });
        }
    });
    hashes
}

fn cpu_parallel_worker_count(item_count: usize) -> usize {
    if item_count < CPU_PARALLEL_MIN_ROWS {
        return 1;
    }
    thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(item_count)
        .max(1)
}

fn try_build_gpu_layers_from_indexed_leaves<W>(
    leaf_count: usize,
    write_leaf_preimage: W,
) -> Option<Vec<Vec<Digest32>>>
where
    W: Fn(usize, &mut Vec<u8>),
{
    let (leaf_preimages, leaf_preimage_bytes) =
        indexed_leaf_preimages(leaf_count, write_leaf_preimage)?;
    let mut layers = Vec::new();
    layers.push(try_gpu_hash_chunks(&leaf_preimages, leaf_preimage_bytes)?);

    while layers.last().unwrap().len() > 1 {
        let previous = layers.last().unwrap();
        let node_preimages = node_preimages(previous);
        layers.push(try_gpu_hash_chunks(&node_preimages, NODE_PREIMAGE_BYTES)?);
    }

    Some(layers)
}

fn indexed_leaf_preimages<W>(leaf_count: usize, write_leaf_preimage: W) -> Option<(Vec<u8>, usize)>
where
    W: Fn(usize, &mut Vec<u8>),
{
    let mut input = Vec::new();
    let mut preimage_bytes = None;

    for index in 0..leaf_count {
        let start = input.len();
        input.extend_from_slice(LEAF_DOMAIN);
        write_leaf_preimage(index, &mut input);
        let current_bytes = input.len() - start;
        match preimage_bytes {
            Some(expected) if expected != current_bytes => return None,
            Some(_) => {}
            None => preimage_bytes = Some(current_bytes),
        }
    }

    preimage_bytes.map(|bytes| (input, bytes))
}

fn node_preimages(layer: &[Digest32]) -> Vec<u8> {
    let mut input = Vec::with_capacity((layer.len() / 2) * NODE_PREIMAGE_BYTES);
    for pair in layer.chunks_exact(2) {
        input.extend_from_slice(NODE_DOMAIN);
        input.extend_from_slice(&pair[0]);
        input.extend_from_slice(&pair[1]);
    }
    input
}

impl MerkleTree {
    #[allow(dead_code)]
    pub(crate) fn from_digest_layers(layers: Vec<Vec<Digest32>>) -> Self {
        assert!(!layers.is_empty());
        assert!(!layers[0].is_empty());
        assert!(layers[0].len().is_power_of_two());
        for pair in layers.windows(2) {
            assert_eq!(pair[0].len() / 2, pair[1].len());
        }
        assert_eq!(layers.last().unwrap().len(), 1);
        Self { layers }
    }

    pub(crate) fn from_leaf_digests(leaf_digests: Vec<Digest32>) -> Self {
        assert!(!leaf_digests.is_empty());
        assert!(leaf_digests.len().is_power_of_two());

        let mut layers = vec![leaf_digests];
        while layers.last().unwrap().len() > 1 {
            let previous = layers.last().unwrap();
            layers.push(hash_node_layer_cpu(previous));
        }
        Self { layers }
    }

    pub fn from_rows<F: ToBytes + Sync>(rows: &[Vec<F>]) -> Self {
        Self::from_indexed_leaves(
            rows.len(),
            |index| row_to_leaf(&rows[index]),
            |index, bytes| write_leaf_values(rows[index].iter(), bytes),
        )
    }

    pub fn from_column<F: ToBytes + Sync>(column: &[F]) -> Self {
        Self::from_indexed_leaves(
            column.len(),
            |index| hash_leaf_values(std::iter::once(&column[index])),
            |index, bytes| write_leaf_values(std::iter::once(&column[index]), bytes),
        )
    }

    pub fn from_columns<F: ToBytes + Sync>(columns: &[&[F]], selected_columns: &[usize]) -> Self {
        assert!(!columns.is_empty());
        assert!(!selected_columns.is_empty());
        let leaf_count = columns[0].len();
        for &column in selected_columns {
            assert!(column < columns.len());
            assert_eq!(columns[column].len(), leaf_count);
        }

        Self::from_indexed_leaves(
            leaf_count,
            |row_index| {
                hash_leaf_values(
                    selected_columns
                        .iter()
                        .map(|&column| &columns[column][row_index]),
                )
            },
            |row_index, bytes| {
                write_leaf_values(
                    selected_columns
                        .iter()
                        .map(|&column| &columns[column][row_index]),
                    bytes,
                )
            },
        )
    }

    fn from_indexed_leaves<H, W>(leaf_count: usize, hash_leaf: H, write_leaf_preimage: W) -> Self
    where
        H: Fn(usize) -> Digest32 + Sync,
        W: Fn(usize, &mut Vec<u8>),
    {
        assert!(leaf_count > 0);
        assert!(leaf_count.is_power_of_two());

        Self {
            layers: build_layers_from_indexed_leaves(leaf_count, hash_leaf, write_leaf_preimage),
        }
    }

    pub fn root(&self) -> Digest32 {
        self.layers.last().unwrap()[0]
    }

    pub fn open(&self, mut index: usize) -> MerklePath {
        let original_index = index;
        let mut siblings = Vec::with_capacity(self.layers.len() - 1);
        for layer in &self.layers[..self.layers.len() - 1] {
            let sibling = if index % 2 == 0 {
                layer[index + 1]
            } else {
                layer[index - 1]
            };
            siblings.push(sibling);
            index /= 2;
        }
        MerklePath {
            index: original_index,
            siblings,
        }
    }

    pub fn verify<F: ToBytes>(root: Digest32, row: &[F], path: &MerklePath) -> Result<(), Error> {
        Self::verify_digest(root, row_to_leaf(row), path)
    }

    pub fn verify_pair<F: ToBytes>(
        root: Digest32,
        row: &[F],
        sibling_row: &[F],
        path: &MerklePath,
        sibling_path: &MerklePath,
    ) -> Result<(), Error> {
        let Some(expected_sibling_subtree) = path.siblings.last() else {
            return Err(Error::MerkleRootMismatch);
        };
        if sibling_path.siblings.len() + 1 != path.siblings.len() {
            return Err(Error::MerkleRootMismatch);
        }
        if Self::path_root(row_to_leaf(sibling_row), sibling_path) != *expected_sibling_subtree {
            return Err(Error::MerkleRootMismatch);
        }
        Self::verify_digest(root, row_to_leaf(row), path)
    }

    pub fn verify_digest(root: Digest32, leaf: Digest32, path: &MerklePath) -> Result<(), Error> {
        let mut index = path.index;
        let mut current = leaf;
        for sibling in &path.siblings {
            current = if index % 2 == 0 {
                hash_pair(&current, sibling)
            } else {
                hash_pair(sibling, &current)
            };
            index /= 2;
        }

        if current == root {
            Ok(())
        } else {
            Err(Error::MerkleRootMismatch)
        }
    }

    fn path_root(leaf: Digest32, path: &MerklePath) -> Digest32 {
        let mut index = path.index;
        let mut current = leaf;
        for sibling in &path.siblings {
            current = if index % 2 == 0 {
                hash_pair(&current, sibling)
            } else {
                hash_pair(sibling, &current)
            };
            index /= 2;
        }
        current
    }
}

#[cfg(feature = "icicle-rust")]
fn try_gpu_hash_chunks(input: &Vec<u8>, preimage_bytes: usize) -> Option<Vec<Digest32>> {
    crate::gpu_lock::with_gpu_lock(|| gpu_merkle::hash_chunks(input, preimage_bytes))
}

#[cfg(not(feature = "icicle-rust"))]
fn try_gpu_hash_chunks(_input: &Vec<u8>, _preimage_bytes: usize) -> Option<Vec<Digest32>> {
    None
}

#[cfg(feature = "icicle-rust")]
fn try_warm_up_gpu_hashes(row_byte_len: usize) -> bool {
    crate::gpu_lock::with_gpu_lock(|| gpu_merkle::warm_up(row_byte_len))
}

#[cfg(not(feature = "icicle-rust"))]
fn try_warm_up_gpu_hashes(_row_byte_len: usize) -> bool {
    false
}

#[cfg(feature = "icicle-rust")]
mod gpu_merkle {
    use super::{Digest32, DIGEST_BYTES, LEAF_DOMAIN, NODE_DOMAIN, NODE_PREIMAGE_BYTES};
    use icicle_core::hash::HashConfig;
    use icicle_hash::keccak::Keccak256 as IcicleKeccak256;
    use icicle_runtime::{
        memory::{IntoIcicleSlice, IntoIcicleSliceMut},
        runtime::{load_backend_from_env_or_default, set_device},
        Device,
    };
    use std::{env, sync::OnceLock};

    const ICICLE_DEVICE_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE";
    const ICICLE_DEVICE_ID_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE_ID";

    static RUNTIME_INIT: OnceLock<bool> = OnceLock::new();

    pub fn hash_chunks(input: &Vec<u8>, preimage_bytes: usize) -> Option<Vec<Digest32>> {
        if preimage_bytes == 0 || input.len() % preimage_bytes != 0 {
            return None;
        }
        init_runtime()?;

        let hasher = IcicleKeccak256::new(preimage_bytes as u64).ok()?;
        let mut output = vec![0u8; (input.len() / preimage_bytes) * DIGEST_BYTES];
        hasher
            .hash(
                input.into_slice(),
                &HashConfig::default(),
                output.into_slice_mut(),
            )
            .ok()?;

        Some(bytes_to_digests(&output))
    }

    pub fn warm_up(row_byte_len: usize) -> bool {
        let mut leaf = Vec::with_capacity(LEAF_DOMAIN.len() + row_byte_len);
        leaf.extend_from_slice(LEAF_DOMAIN);
        leaf.resize(LEAF_DOMAIN.len() + row_byte_len, 0u8);

        let mut node = Vec::with_capacity(NODE_PREIMAGE_BYTES);
        node.extend_from_slice(NODE_DOMAIN);
        node.resize(NODE_PREIMAGE_BYTES, 0u8);

        hash_chunks(&leaf, leaf.len()).is_some()
            && hash_chunks(&node, NODE_PREIMAGE_BYTES).is_some()
    }

    fn init_runtime() -> Option<()> {
        RUNTIME_INIT
            .get_or_init(|| {
                if load_backend_from_env_or_default().is_err() {
                    return false;
                }
                let device_type =
                    env::var(ICICLE_DEVICE_ENV).unwrap_or_else(|_| "CUDA".to_string());
                let device_id = env::var(ICICLE_DEVICE_ID_ENV)
                    .ok()
                    .and_then(|value| value.parse::<i32>().ok())
                    .unwrap_or(0);
                set_device(&Device::new(&device_type, device_id)).is_ok()
            })
            .then_some(())
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
}
