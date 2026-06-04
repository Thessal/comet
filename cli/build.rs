fn main() {
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch_cuda");
    println!("cargo:rustc-link-arg=-lc10_cuda");
    println!("cargo:rustc-link-arg=-Wl,--as-needed");
}
