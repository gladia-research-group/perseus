// Compiles the feature-selected instantiation TUs (kernels/inst/*.cu) with nvcc into one static archive.
// PC_NVCC_JOBS (default 2) nvcc processes at a time: the box's shared-CPU budget. PERSEUS_CORE_ARCH (default 120)
// selects -gencode arch=compute_<a>,code=sm_<a>, once per entry of a ';'- or ','-separated list ("89;120").
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda-13.2".into());
    let nvcc = format!("{cuda}/bin/nvcc");
    let arch_env = std::env::var("PERSEUS_CORE_ARCH").unwrap_or_else(|_| "120".into());
    let gencode: Vec<String> = arch_env
        .split([';', ','])
        .map(|a| a.trim().trim_start_matches("sm_").trim_start_matches("compute_"))
        .filter(|a| !a.is_empty())
        .flat_map(|a| ["-gencode".to_string(), format!("arch=compute_{a},code=sm_{a}")])
        .collect();
    assert!(!gencode.is_empty(), "PERSEUS_CORE_ARCH names no architecture");
    let jobs: usize = std::env::var("PC_NVCC_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(2).max(1);
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let kdir = Path::new("kernels");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=PERSEUS_CORE_ARCH");
    println!("cargo:rerun-if-env-changed=PC_NVCC_JOBS");
    let feat = |f: &str| std::env::var(format!("CARGO_FEATURE_{}", f.to_uppercase())).is_ok();
    // (tu name, source, extra defines): the (word, ModMul) tables, then the ring tables (feature `ring`: u32 for
    // log n 10..17, plus u64 with `w64`)
    let mut tus: Vec<(String, String, Vec<String>)> = Vec::new();
    let mut plain: Vec<&str> = Vec::new();
    if feat("w32") {
        plain.extend(["u32_barrett64", "u32_barrett32"]);
    }
    if feat("mont") {
        plain.push("u32_mont");
    }
    if feat("w64") {
        plain.push("u64_barrett64");
    }
    if feat("w16") {
        plain.push("u16_barrett32");
    }
    for tu in plain {
        tus.push((tu.to_string(), format!("{tu}.cu"), Vec::new()));
    }
    if feat("ring") {
        let mut words = vec![("u32", "uint32_t")];
        if feat("w64") {
            words.push(("u64", "uint64_t"));
        }
        for (w, ty) in words {
            for logn in 10..=17 {
                let name = format!("pc_ring_{w}_{logn}");
                let defs = vec![format!("-DPC_LOGN={logn}"), format!("-DPC_RING_W={ty}"), format!("-DPC_RING_TABLE={name}")];
                tus.push((name, "ring.cu".to_string(), defs));
            }
        }
    }
    // PC_TL's timestamp kernel (diagnostic)
    tus.push(("stamp".to_string(), "stamp.cu".to_string(), Vec::new()));
    let mut objs = Vec::new();
    for chunk in tus.chunks(jobs) {
        let mut kids = Vec::new();
        for (tu, file, defs) in chunk {
            let src = kdir.join("inst").join(file);
            let o = out.join(format!("{tu}.o"));
            let child = Command::new(&nvcc)
                .args(["-O3", "-std=c++17", "-lineinfo", "--use_fast_math", "-Xcompiler", "-fPIC"])
                .args(&gencode)
                .args(defs)
                .args(["-I", "kernels", "-c"])
                .arg(&src)
                .arg("-o")
                .arg(&o)
                .spawn()
                .unwrap_or_else(|e| panic!("cannot run {nvcc}: {e}"));
            kids.push((child, o, src));
        }
        for (mut c, o, s) in kids {
            let st = c.wait().expect("nvcc wait");
            assert!(st.success(), "nvcc failed on {}", s.display());
            objs.push(o);
        }
    }
    let lib = out.join("libpckern.a");
    let _ = std::fs::remove_file(&lib);
    if !objs.is_empty() {
        let st = Command::new("ar").arg("crs").arg(&lib).args(&objs).status().expect("ar");
        assert!(st.success(), "ar failed");
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=pckern");
    }
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
}
