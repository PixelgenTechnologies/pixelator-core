#[cfg(target_os = "windows")]
fn main() {
    println!("cargo:rustc-link-lib=Rstrtmgr");
}

#[cfg(target_os = "linux")]
fn main() {}

#[cfg(target_os = "macos")]
fn main() {}
