{
  description = "reservoir — reservoir computing written as neural circuit diagrams with PyNCD";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  # PyNCD is a git dependency in uv.lock, pinned to one commit there, and uv2nix
  # fetches it from that lock. tsncd is the git submodule tsncd/, pinned by its
  # gitlink and run by its own npm scripts. The two are a matched pair, so move the
  # pins together.
  outputs = { self, nixpkgs, uv2nix, pyproject-nix, pyproject-build-systems, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs [ "x86_64-linux" "aarch64-linux" ];

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
        members = [ "reservoir" ];
      };

      perSystem = system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python314;
          # The CUDA libraries the Linux wheel of Torch 2.15 depends on. libcuda.so.1
          # comes from the host driver rather than a Nix closure, so each gets the NixOS
          # driver runpath and keeps the wheel's own $ORIGIN runpaths. The wheels find
          # one another through $ORIGIN in a flat site-packages, and uv2nix builds each
          # wheel on its own, so autoPatchelf is taught every sibling library directory.
          nvidiaPackages = [
            "nvidia-cublas"
            "nvidia-cuda-cccl"
            "nvidia-cuda-cupti"
            "nvidia-cuda-nvrtc"
            "nvidia-cuda-runtime"
            "nvidia-cudnn-cu13"
            "nvidia-cufft"
            "nvidia-cufile"
            "nvidia-cusolver"
            "nvidia-cusparse"
            "nvidia-cusparselt-cu13"
            "nvidia-nccl-cu13"
            "nvidia-nvjitlink"
            "nvidia-nvshmem-cu13"
            "nvidia-nvtx"
          ];
          pyprojectOverrides = final: prev:
            let
              searchPathsFor = names: ''
                for dep in ${lib.concatMapStringsSep " " (name: final.${name}) names}; do
                  while IFS= read -r libdir; do
                    addAutoPatchelfSearchPath "$libdir"
                  done < <(find "$dep/${python.sitePackages}" -type d -name lib)
                done
              '';
              withDriverRunpath = name: extra:
                prev.${name}.overrideAttrs (old: extra old // {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoAddDriverRunpath ];
                  autoPatchelfFlags = (old.autoPatchelfFlags or [ ]) ++ [ "--preserve-origin" ];
                });
            in
            lib.genAttrs nvidiaPackages (name: withDriverRunpath name (_: { }))
            // {
              # A git dependency is built from source, so its build system is named here.
              pyncd = prev.pyncd.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or [ ])
                  ++ final.resolveBuildSystem { hatchling = [ ]; };
              });
              nvidia-cusolver = withDriverRunpath "nvidia-cusolver" (old: {
                preFixup = (old.preFixup or "")
                  + searchPathsFor [ "nvidia-cublas" "nvidia-cusparse" "nvidia-nvjitlink" ];
              });
              nvidia-cusparse = withDriverRunpath "nvidia-cusparse" (old: {
                preFixup = (old.preFixup or "") + searchPathsFor [ "nvidia-nvjitlink" ];
              });
              nvidia-cufile = withDriverRunpath "nvidia-cufile" (old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.rdma-core ];
              });
              nvidia-nvshmem-cu13 = withDriverRunpath "nvidia-nvshmem-cu13" (old: {
                buildInputs = (old.buildInputs or [ ]) ++ [
                  pkgs.rdma-core pkgs.libfabric pkgs.openmpi pkgs.pmix pkgs.ucx
                ];
              });
              torch = withDriverRunpath "torch" (old: {
                preFixup = (old.preFixup or "") + searchPathsFor nvidiaPackages;
                autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ])
                  ++ [ "libcuda.so.1" ];
              });
            };
          pythonSet =
            (pkgs.callPackage pyproject-nix.build.packages { inherit python; })
            .overrideScope (lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              pyprojectOverrides
            ]);
          # The dev shell runs this virtual environment: the workspace package editable,
          # every dependency built from the flake's locked inputs. A uv-synced venv would
          # install the empty packaging shim in place of PyNCD.
          editablePythonSet = pythonSet.overrideScope (lib.composeManyExtensions [
            editableOverlay
            (final: prev: {
              reservoir = prev.reservoir.overrideAttrs (old: {
                src = lib.fileset.toSource {
                  root = old.src;
                  fileset = lib.fileset.unions [
                    (old.src + "/pyproject.toml")
                    (old.src + "/src/reservoir/__init__.py")
                  ];
                };
                nativeBuildInputs = (old.nativeBuildInputs or [ ])
                  ++ final.resolveBuildSystem { editables = [ ]; };
              });
            })
          ]);
        in {
          inherit pkgs pythonSet;
          devEnv = editablePythonSet.mkVirtualEnv "reservoir-dev-env" workspace.deps.all;
        };
    in {
      devShells = forAllSystems (system:
        let s = perSystem system;
        in {
          default = s.pkgs.mkShell {
            packages = [
              s.devEnv
              s.pkgs.uv
              s.pkgs.just
              # tsncd's own `npm install`, `npm run server` and `npm run dev` in tsncd/.
              s.pkgs.nodejs_24
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PROJECT_ENVIRONMENT = "${s.devEnv}";
              UV_PYTHON = "${s.devEnv}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT="$(git rev-parse --show-toplevel)"
            '';
          };
        }
      );

      packages = forAllSystems (system:
        let s = perSystem system;
        in {
          default = s.pythonSet.mkVirtualEnv "reservoir-env" workspace.deps.default;
        }
      );
    };
}
