{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    pandas
    holidays
    requests
    urllib3
  ]);
in
pkgs.mkShell {
  buildInputs = [
    pythonEnv
  ];
}
