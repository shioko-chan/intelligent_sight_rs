use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let nested_path = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_dir = Path::new(&nested_path);

    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("cam_op/c_src/*").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("trt_op/cxx_src/*").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("xmake.lua").display()
    );

    println!(
        "cargo:rustc-link-search=native={}",
        manifest_dir.join("clibs").display()
    );

    println!("cargo:rustc-link-lib=dylib=camera_wrapper");
    println!("cargo:rustc-link-lib=dylib=tensorrt_wrapper");

    let target = env::var("TARGET").unwrap();

    if target.contains("windows") {
        println!(
            r#"cargo:rustc-link-search=native={}"#,
            manifest_dir.join(r#"linuxSDK_V2.1.0.37\lib"#).display()
        );
        println!(r#"cargo:rustc-link-search=native=D:\Program Files (x86)\TensorRT-10.0.0.6\lib\"#); // TENSOR_RT PATH
        println!(
            r#"cargo:rustc-link-search=native=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64\"# // CUDA PATH
        );
        println!("cargo:rustc-link-lib=dylib=MVCAMSDK_X64");
        println!("cargo:rustc-link-lib=static=nvinfer");
        println!("cargo:rustc-link-lib=static=cudart_static");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=dylib=MVSDK");
        println!("cargo:rustc-link-lib=dylib=nvinfer");
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else {
        panic!("unsupported platform")
    }

    let result = Command::new("xmake")
        .status()
        .expect("failed to build clibs");

    if !result.success() {
        panic!("failed to build clibs")
    }
}
