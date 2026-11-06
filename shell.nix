let
  pkgs = import <nixpkgs> {
    overlays = [
      (import (fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz"))
    ];
  };
  unstable = import <nixos-unstable> { config = { 
    allowUnfree = true; 
    cudaSupport = false;
    rocmSupport = false;
  }; };
  rustVersion = "1.91.1";
  myRust = pkgs.rust-bin.stable.${rustVersion}.default.override {
    extensions = [
      "rust-src" # for rust-analyzer
      "rust-analyzer"
    ];
  };

in pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
  buildInputs = with pkgs; [
    cmake
    libllvm
    libffi
    libxml2
    # rocmPackages.clr
    # rocmPackages.rocblas
  ];
  packages = [
    myRust
    unstable.antigravity-cli
  ] ++ (with pkgs; [
    cargo rustc gcc rustfmt clippy rust-analyzer gdb
    python313 libtorch-bin
  ]) ++ (with pkgs.python313Packages; [ 
    numpy torch jupyter notebook ipykernel 
  ]);
  RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";

  # libtorch & development environment
  shellHook = ''
    # Point LIBTORCH to the root of the package, not the .so file
    export LIBTORCH=${pkgs.libtorch-bin}
    export LIBTORCH_INCLUDE="${pkgs.libtorch-bin.dev}";
    export PYTHONPATH="$PWD/python:$PYTHONPATH"
  '';

  # # CUDA
  # shellHook = ''
  #    export CUDA_PATH=${pkgs.cudatoolkit}
  #    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidiaPackages.stable}/lib:${pkgs.cudaPackages.cuda_nvrtc.lib}/lib:$LD_LIBRARY_PATH
  #    # LD_LIBRARY_PATH=/run/opengl-driver/lib
  #    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidiaPackages.stable}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
  #    export EXTRA_CCFLAGS="-I/usr/include"
  # ''; 

  ## ROCM
  # shellHook = ''
  #    export ROCM_PATH=${pkgs.rocmPackages.clr}
  #    export LD_LIBRARY_PATH=${pkgs.rocmPackages.clr}/lib:${pkgs.rocmPackages.rocblas}/lib:$LD_LIBRARY_PATH
  #    export EXTRA_LDFLAGS="-L${pkgs.rocmPackages.clr}/lib -L${pkgs.rocmPackages.rocblas}/lib"
  #    export EXTRA_CCFLAGS="-I${pkgs.rocmPackages.clr}/include"
  # '';
}

