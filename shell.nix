let
  pkgs = import <nixpkgs> {
    overlays = [
      (import (fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz"))
    ];
  };
  rustVersion = "1.91.1";
  myRust = pkgs.rust-bin.stable.${rustVersion}.default.override {
    extensions = [
      "rust-src" # for rust-analyzer
      "rust-analyzer"
    ];
  };

  antigravity = pkgs.callPackage ./antigravity.nix {};

in pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
  buildInputs = with pkgs; [
    cmake
    libllvm
    libffi
    libxml2
  ];
  packages = [
    #pkgs.antigravity
    pkgs.google-chrome
    antigravity
    myRust
  ] ++ (with pkgs; [
    cargo rustc gcc rustfmt clippy rust-analyzer pkg-config maturin
  ]) ;
  RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
}