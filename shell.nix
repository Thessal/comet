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
    cargo rustc gcc rustfmt clippy rust-analyzer gdb
    python313 python313.pkgs.numpy
  ]) ;
  RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
  shellHook = ''
     export CUDA_PATH=${pkgs.cudatoolkit}
     export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidiaPackages.stable}/lib:${pkgs.cudaPackages.cuda_nvrtc.lib}/lib:$LD_LIBRARY_PATH
     # LD_LIBRARY_PATH=/run/opengl-driver/lib
     export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidiaPackages.stable}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
     export EXTRA_CCFLAGS="-I/usr/include"
  '';    
}
