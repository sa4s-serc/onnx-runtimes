use std::path::PathBuf;

fn main() {
    // Add rpath so the binary finds libonnxruntime.so at runtime.
    // The library search path and -lonnxruntime are handled by ort-sys via ORT_LIB_PATH.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ort_lib = manifest_dir
        .parent().unwrap() // inference
        .parent().unwrap() // scripts
        .parent().unwrap() // repo root
        .join("lib/onnxruntime/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", ort_lib.display());
}
