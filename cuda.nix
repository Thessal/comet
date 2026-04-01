let pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  shellHook = ''
     export CUDA_PATH=${pkgs.cudatoolkit}
     export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidiaPackages.stable}/lib:${pkgs.cudaPackages.cuda_nvrtc.lib}/lib:$LD_LIBRARY_PATH
     # LD_LIBRARY_PATH=/run/opengl-driver/lib
     export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidiaPackages.stable}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
     export EXTRA_CCFLAGS="-I/usr/include"
  ''; 
}

