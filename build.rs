use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=native/fri_lookup_cuda_kernels.cu");

    if env::var_os("CARGO_FEATURE_ICICLE_RUST").is_none() {
        return;
    }

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let output = out_dir.join("libfri_lookup_cuda_kernels.a");

    let status = Command::new(&nvcc)
        .args([
            "-std=c++17",
            "-Xcompiler",
            "-fPIC",
            "-lib",
            "native/fri_lookup_cuda_kernels.cu",
            "-o",
        ])
        .arg(&output)
        .status()
        .unwrap_or_else(|error| panic!("failed to run nvcc ({nvcc}): {error}"));

    if !status.success() {
        panic!("nvcc failed while building native/fri_lookup_cuda_kernels.cu");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=fri_lookup_cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");

    if let Some(cuda_home) =
        cuda_home_from_nvcc(&nvcc).or_else(|| env::var("CUDA_HOME").ok().map(PathBuf::from))
    {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_home.join("lib64").display()
        );
    }
}

fn cuda_home_from_nvcc(nvcc: &str) -> Option<PathBuf> {
    let output = Command::new("which").arg(nvcc).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8(output.stdout).ok()?;
    let nvcc_path = PathBuf::from(path.trim());
    nvcc_path.parent()?.parent().map(PathBuf::from)
}
