use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cam_op/c_src/*");
    println!("cargo:rerun-if-changed=trt_op/cxx_src/*");
    println!("cargo:rerun-if-changed=xmake.lua");

    println!("cargo:rustc-link-search=native=clibs");
    println!("cargo:rustc-link-lib=static=camera_wrapper");

    let target = env::var("TARGET").unwrap();
    if target.contains("windows") {
        println!("cargo:rustc-link-search=native=linuxSDK_V2.1.0.37/lib");
        println!(r#"cargo:rustc-link-search=native=D:\Program Files (x86)\TensorRT-10.0.0.6\lib"#);
        println!(
            r#"cargo:rustc-link-search=native=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include"#
        );
        println!("cargo:rustc-link-lib=static=MVCAMSDK_X64");
        println!("cargo:rustc-link-lib=dylib=nvinfer");
        // println!("cargo:rustc-link-lib=dylib=");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=static=MVSDK");
        println!("cargo:rustc-link-lib=dylib=nvinfer");
    }

    let result = Command::new("xmake")
        .status()
        .expect("failed to build clibs");

    if !result.success() {
        panic!("failed to build clibs")
    }
}
