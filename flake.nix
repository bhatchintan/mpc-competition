{
  description = "Development environment for the MPC competition repo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          packages = nixpkgs.legacyPackages.${system};
          python = packages.python311;
        in
        {
          default = packages.mkShell {
            packages = [
              python
              packages.uv
              packages.ruff
              packages.pyright
            ];

            shellHook = ''
              export UV_PYTHON=${python}/bin/python
              export PYTHONPATH=$PWD
            '';
          };
        }
      );

      formatter = forAllSystems (
        system:
        let
          packages = nixpkgs.legacyPackages.${system};
        in
        packages.nixfmt
      );
    };
}
