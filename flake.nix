{
  description = "Python ML Flake";

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
          overlays = [(import ./overlays.nix)];
        };

        python = pkgs.python3;
        pythonPackages = pkgs.python3Packages;
        cudaPackages = pkgs.cudaPackages;

        packages = [
          pkgs.cachix
          pkgs.alejandra
          pkgs.nodejs

          cudaPackages.cudatoolkit
          cudaPackages.cudnn

          python

          # Generic
          pythonPackages.black
          pythonPackages.ipykernel
          pythonPackages.pip

          pythonPackages.jupyter
          pythonPackages.pygments
          pythonPackages.babel
          pythonPackages.python-lsp-server

          # Utils
          pythonPackages.opencv4
          pythonPackages.matplotlib
          pythonPackages.numpy
          pythonPackages.pandas
          pythonPackages.pathlib2
          pythonPackages.polars
          pythonPackages.scikit-learn
          pythonPackages.seaborn

          # ML
          pythonPackages.torch-bin
          pythonPackages.torchvision-bin
          pythonPackages.torchaudio-bin
          pythonPackages.torchmetrics
          pythonPackages.torchinfo
          pythonPackages.pytorch-lightning

          pythonPackages.torchio
          pythonPackages.monai
          pythonPackages.wandb
        ];
      in {
        formatter = pkgs.alejandra;
        devShell = pkgs.mkShell {
          inherit packages;
          name = "machine-learning";
          shellHook = ''
            ${pkgs.cachix}/bin/cachix use cuda-maintainers
            python -m venv .venv
            source .venv/bin/activate
          '';
        };
      }
    );
}
