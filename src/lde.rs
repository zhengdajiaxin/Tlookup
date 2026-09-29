use ark_ff::FftField;
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, UVPolynomial,
};
use std::{env, marker::PhantomData};

use crate::{error::Error, utils::is_power_of_two};

pub const DEFAULT_GPU_LDE_MIN_SIZE: usize = 1 << 16;

const LDE_BACKEND_ENV: &str = "FRI_LOOKUP_LDE_BACKEND";
const GPU_LDE_MIN_SIZE_ENV: &str = "FRI_LOOKUP_GPU_LDE_MIN_SIZE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseLdeBackendKind {
    Cpu,
    Gpu,
}

#[derive(Clone, Debug)]
pub struct DenseLdePlan<F: FftField> {
    pub base_size: usize,
    pub blowup: usize,
    pub lde_size: usize,
    pub base_root_of_unity: F,
    pub base_inverse_root_of_unity: F,
    pub lde_root_of_unity: F,
    pub lde_inverse_root_of_unity: F,
    pub base_size_inverse: F,
    pub lde_size_inverse: F,
    pub base_bit_reversal: Vec<usize>,
    pub lde_bit_reversal: Vec<usize>,
}

impl<F: FftField> DenseLdePlan<F> {
    pub fn new(base_size: usize, blowup: usize) -> Result<Self, Error> {
        if !is_power_of_two(base_size) {
            return Err(Error::InvalidDomainSize(base_size));
        }
        if !is_power_of_two(blowup) {
            return Err(Error::InvalidBlowup(blowup));
        }

        let lde_size = base_size * blowup;
        let base_domain = GeneralEvaluationDomain::<F>::new(base_size)
            .ok_or(Error::InvalidDomainSize(base_size))?;
        let lde_domain = GeneralEvaluationDomain::<F>::new(lde_size)
            .ok_or(Error::InvalidDomainSize(lde_size))?;
        let base_root_of_unity = base_domain.element(1);
        let lde_root_of_unity = lde_domain.element(1);

        Ok(Self {
            base_size,
            blowup,
            lde_size,
            base_root_of_unity,
            base_inverse_root_of_unity: base_root_of_unity
                .inverse()
                .ok_or(Error::InvalidDomainSize(base_size))?,
            lde_root_of_unity,
            lde_inverse_root_of_unity: lde_root_of_unity
                .inverse()
                .ok_or(Error::InvalidDomainSize(lde_size))?,
            base_size_inverse: F::from(base_size as u64)
                .inverse()
                .ok_or(Error::InvalidDomainSize(base_size))?,
            lde_size_inverse: F::from(lde_size as u64)
                .inverse()
                .ok_or(Error::InvalidDomainSize(lde_size))?,
            base_bit_reversal: bit_reversal_permutation(base_size),
            lde_bit_reversal: bit_reversal_permutation(lde_size),
        })
    }
}

pub trait DenseLdeBackend<F: FftField> {
    fn evaluate(base_evals: &[F], plan: &DenseLdePlan<F>) -> Result<Vec<F>, Error>;
}

pub struct CpuDenseLdeBackend;

impl<F: FftField> DenseLdeBackend<F> for CpuDenseLdeBackend {
    fn evaluate(base_evals: &[F], plan: &DenseLdePlan<F>) -> Result<Vec<F>, Error> {
        if base_evals.len() != plan.base_size {
            return Err(Error::InvalidDomainSize(base_evals.len()));
        }

        let poly = interpolate_cpu(base_evals)?;
        let lde_domain = GeneralEvaluationDomain::<F>::new(plan.lde_size)
            .ok_or(Error::InvalidDomainSize(plan.lde_size))?;
        Ok(lde_domain.fft(&poly.coeffs))
    }
}

pub struct GpuDenseLdeBackend<F: FftField> {
    marker: PhantomData<F>,
}

impl<F: FftField> GpuDenseLdeBackend<F> {
    pub fn is_available() -> bool {
        cfg!(feature = "icicle-rust")
    }

    pub fn supports_field() -> bool {
        #[cfg(feature = "icicle-rust")]
        {
            return icicle_bn254::supports_field::<F>();
        }

        #[cfg(not(feature = "icicle-rust"))]
        {
            false
        }
    }

    pub fn is_available_for_field() -> bool {
        Self::is_available() && Self::supports_field()
    }
}

impl<F: FftField> DenseLdeBackend<F> for GpuDenseLdeBackend<F> {
    fn evaluate(base_evals: &[F], plan: &DenseLdePlan<F>) -> Result<Vec<F>, Error> {
        evaluate_gpu_dense_lde(base_evals, plan)
    }
}

pub fn interpolate<F: FftField>(base_evals: &[F]) -> Result<DensePolynomial<F>, Error> {
    interpolate_cpu(base_evals)
}

pub fn evaluate_dense_lde<F: FftField>(base_evals: &[F], blowup: usize) -> Result<Vec<F>, Error> {
    let plan = DenseLdePlan::new(base_evals.len(), blowup)?;
    evaluate_dense_lde_with_plan(base_evals, &plan)
}

pub fn evaluate_dense_lde_from_coeffs<F: FftField>(
    coeffs: &[F],
    base_size: usize,
    blowup: usize,
) -> Result<Vec<F>, Error> {
    let plan = DenseLdePlan::new(base_size, blowup)?;
    evaluate_dense_lde_from_coeffs_with_plan(coeffs, &plan)
}

pub fn evaluate_dense_lde_from_coeffs_with_plan<F: FftField>(
    coeffs: &[F],
    plan: &DenseLdePlan<F>,
) -> Result<Vec<F>, Error> {
    if coeffs.len() > plan.base_size {
        return Err(Error::LowDegreeCheckFailed);
    }
    match select_dense_lde_backend::<F>(plan) {
        DenseLdeBackendKind::Cpu => evaluate_dense_lde_from_coeffs_cpu(coeffs, plan),
        DenseLdeBackendKind::Gpu => {
            let base_evals = evaluate_base_domain_from_coeffs(coeffs, plan)?;
            GpuDenseLdeBackend::evaluate(&base_evals, plan)
        }
    }
}

fn evaluate_dense_lde_from_coeffs_cpu<F: FftField>(
    coeffs: &[F],
    plan: &DenseLdePlan<F>,
) -> Result<Vec<F>, Error> {
    let mut padded = coeffs.to_vec();
    padded.resize(plan.lde_size, F::zero());
    let lde_domain = GeneralEvaluationDomain::<F>::new(plan.lde_size)
        .ok_or(Error::InvalidDomainSize(plan.lde_size))?;
    Ok(lde_domain.fft(&padded))
}

fn evaluate_base_domain_from_coeffs<F: FftField>(
    coeffs: &[F],
    plan: &DenseLdePlan<F>,
) -> Result<Vec<F>, Error> {
    let mut padded = coeffs.to_vec();
    padded.resize(plan.base_size, F::zero());
    let base_domain = GeneralEvaluationDomain::<F>::new(plan.base_size)
        .ok_or(Error::InvalidDomainSize(plan.base_size))?;
    Ok(base_domain.fft(&padded))
}

pub fn evaluate_dense_lde_with_plan<F: FftField>(
    base_evals: &[F],
    plan: &DenseLdePlan<F>,
) -> Result<Vec<F>, Error> {
    match select_dense_lde_backend::<F>(plan) {
        DenseLdeBackendKind::Cpu => CpuDenseLdeBackend::evaluate(base_evals, plan),
        DenseLdeBackendKind::Gpu => GpuDenseLdeBackend::evaluate(base_evals, plan),
    }
}

pub fn select_dense_lde_backend<F: FftField>(plan: &DenseLdePlan<F>) -> DenseLdeBackendKind {
    match requested_backend() {
        Some(DenseLdeBackendKind::Cpu) => DenseLdeBackendKind::Cpu,
        Some(DenseLdeBackendKind::Gpu) => {
            if GpuDenseLdeBackend::<F>::is_available_for_field() {
                DenseLdeBackendKind::Gpu
            } else {
                DenseLdeBackendKind::Cpu
            }
        }
        None => {
            if GpuDenseLdeBackend::<F>::is_available_for_field()
                && plan.lde_size >= gpu_lde_min_size()
            {
                DenseLdeBackendKind::Gpu
            } else {
                DenseLdeBackendKind::Cpu
            }
        }
    }
}

pub fn gpu_lde_min_size() -> usize {
    env::var(GPU_LDE_MIN_SIZE_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_GPU_LDE_MIN_SIZE)
}

fn requested_backend() -> Option<DenseLdeBackendKind> {
    match env::var(LDE_BACKEND_ENV)
        .ok()?
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" => Some(DenseLdeBackendKind::Cpu),
        "gpu" => Some(DenseLdeBackendKind::Gpu),
        "auto" => None,
        _ => None,
    }
}

fn interpolate_cpu<F: FftField>(base_evals: &[F]) -> Result<DensePolynomial<F>, Error> {
    if !is_power_of_two(base_evals.len()) {
        return Err(Error::InvalidDomainSize(base_evals.len()));
    }
    let domain = GeneralEvaluationDomain::<F>::new(base_evals.len())
        .ok_or(Error::InvalidDomainSize(base_evals.len()))?;
    Ok(DensePolynomial::from_coefficients_slice(
        &domain.ifft(base_evals),
    ))
}

fn evaluate_gpu_dense_lde<F: FftField>(
    base_evals: &[F],
    plan: &DenseLdePlan<F>,
) -> Result<Vec<F>, Error> {
    #[cfg(feature = "icicle-rust")]
    {
        if let Some(result) = crate::gpu_lock::with_gpu_lock(|| {
            icicle_bn254::try_evaluate_dense_lde(base_evals, plan)
        })? {
            return Ok(result);
        }
    }

    #[cfg(not(feature = "icicle-rust"))]
    {
        let _ = (base_evals, plan);
    }

    Err(Error::GpuBackendUnavailable)
}

fn bit_reversal_permutation(size: usize) -> Vec<usize> {
    let bits = size.trailing_zeros();
    (0..size)
        .map(|index| index.reverse_bits() >> (usize::BITS - bits))
        .collect()
}

#[cfg(feature = "icicle-rust")]
mod icicle_bn254 {
    use super::{CpuDenseLdeBackend, DenseLdeBackend, DenseLdePlan};
    use crate::error::Error;
    use ark_bn254::Fr as ArkBn254Fr;
    use ark_ff::{BigInteger, FftField, PrimeField};
    use icicle_bn254::curve::ScalarField as IcicleBn254Fr;
    use icicle_core::{
        bignum::BigNum,
        ntt::{get_root_of_unity, initialize_domain, ntt, NTTConfig, NTTDir, NTTInitDomainConfig},
    };
    use icicle_runtime::{
        memory::{IntoIcicleSlice, IntoIcicleSliceMut},
        runtime::{load_backend_from_env_or_default, set_device},
        Device,
    };
    use std::{
        env,
        sync::{Mutex, OnceLock},
    };

    const ICICLE_DEVICE_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE";
    const ICICLE_DEVICE_ID_ENV: &str = "FRI_LOOKUP_ICICLE_DEVICE_ID";

    static RUNTIME_INIT: OnceLock<Result<(), Error>> = OnceLock::new();
    static DOMAIN_SIZE: OnceLock<Mutex<usize>> = OnceLock::new();

    pub fn supports_field<F: FftField>() -> bool {
        is_ark_bn254_fr::<F>()
    }

    pub fn try_evaluate_dense_lde<F: FftField>(
        base_evals: &[F],
        plan: &DenseLdePlan<F>,
    ) -> Result<Option<Vec<F>>, Error> {
        if !is_ark_bn254_fr::<F>() {
            return Ok(None);
        }

        init_runtime()?;
        init_domain(plan.lde_size)?;

        let base_evals = unsafe { cast_ark_bn254_slice(base_evals) };
        let bn254_plan = DenseLdePlan::<ArkBn254Fr>::new(plan.base_size, plan.blowup)?;
        let result = evaluate_bn254_dense_lde(base_evals, &bn254_plan)?;
        Ok(Some(unsafe { cast_bn254_vec(result) }))
    }

    fn evaluate_bn254_dense_lde(
        base_evals: &[ArkBn254Fr],
        plan: &DenseLdePlan<ArkBn254Fr>,
    ) -> Result<Vec<ArkBn254Fr>, Error> {
        if base_evals.len() != plan.base_size {
            return Err(Error::InvalidDomainSize(base_evals.len()));
        }

        let input = base_evals
            .iter()
            .map(ark_to_icicle)
            .collect::<Result<Vec<_>, _>>()?;
        let mut coeffs = vec![IcicleBn254Fr::zero(); plan.base_size];
        ntt(
            input.into_slice(),
            NTTDir::kInverse,
            &NTTConfig::<IcicleBn254Fr>::default(),
            coeffs.into_slice_mut(),
        )
        .map_err(|_| Error::GpuBackendFailed)?;

        coeffs.resize(plan.lde_size, IcicleBn254Fr::zero());
        let mut lde = vec![IcicleBn254Fr::zero(); plan.lde_size];
        ntt(
            coeffs.into_slice(),
            NTTDir::kForward,
            &NTTConfig::<IcicleBn254Fr>::default(),
            lde.into_slice_mut(),
        )
        .map_err(|_| Error::GpuBackendFailed)?;

        Ok(lde
            .iter()
            .map(icicle_to_ark)
            .collect::<Result<Vec<_>, _>>()?)
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

    fn init_domain(size: usize) -> Result<(), Error> {
        let mutex = DOMAIN_SIZE.get_or_init(|| Mutex::new(0));
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

    unsafe fn cast_bn254_vec<F: FftField>(values: Vec<ArkBn254Fr>) -> Vec<F> {
        let mut values = std::mem::ManuallyDrop::new(values);
        Vec::from_raw_parts(
            values.as_mut_ptr() as *mut F,
            values.len(),
            values.capacity(),
        )
    }

    #[allow(dead_code)]
    fn cpu_reference(
        base_evals: &[ArkBn254Fr],
        plan: &DenseLdePlan<ArkBn254Fr>,
    ) -> Result<Vec<ArkBn254Fr>, Error> {
        CpuDenseLdeBackend::evaluate(base_evals, plan)
    }
}
