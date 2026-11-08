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

    # tsncd, the renderer, whose bundle the flake builds. PyNCD is a git dependency
    # in uv.lock, pinned to one commit there, and uv2nix fetches it from that lock.
    # The two are a matched pair, so move the pins together.
    tsncd = {
      url = "github:starbaser/tsncd";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, uv2nix, pyproject-nix, pyproject-build-systems, tsncd, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs [ "x86_64-linux" "aarch64-linux" ];

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
        members = [ "reservoir" ];
      };

      # tsncd's production bundle. PyNCD's page writer and headless capture read it
      # through TSNCD_DIST.
      mkTsncdBundle = pkgs:
        let manifest = builtins.fromJSON (builtins.readFile "${tsncd}/package.json");
        in pkgs.buildNpmPackage {
          pname = manifest.name;
          inherit (manifest) version;
          src = tsncd;
          nodejs = pkgs.nodejs_24;
          npmDepsHash = "sha256-bH3DY2wG6U1S4T4E2Ok+UYflahRUNQnzGw15A+7+O0E=";
          installPhase = ''
            runHook preInstall
            mkdir -p "$out"
            cp -r dist/. "$out/"
            runHook postInstall
          '';
          doInstallCheck = true;
          installCheckPhase = ''
            runHook preInstallCheck
            test -s "$out/index.html"
            test -n "$(find "$out/assets" -name 'main.*.js' -print -quit)"
            runHook postInstallCheck
          '';
        };

      perSystem = system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python314;
          # The CUDA wheels of Torch 2.15 (cu132). libcuda.so.1 comes from the host
          # driver rather than a Nix closure, so every CUDA library gets the NixOS
          # driver runpath and keeps the wheel's own $ORIGIN runpaths. The wheels find
          # one another through $ORIGIN in a flat site-packages, and uv2nix builds each
          # wheel on its own, so autoPatchelf is taught every sibling library directory.
          nvidiaPackages = [
            "nvidia-cublas"
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
          tsncdBundle = mkTsncdBundle pkgs;
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
              s.pkgs.process-compose
              s.pkgs.playwright-driver.browsers
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PROJECT_ENVIRONMENT = "${s.devEnv}";
              UV_PYTHON = "${s.devEnv}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
              # PyNCD's page writer and headless capture read the built bundle here.
              TSNCD_DIST = "${s.tsncdBundle}";
              # Playwright's own Chromium download does not run on NixOS. The nixpkgs
              # browsers match the playwright wheel pinned in pyproject.toml.
              PLAYWRIGHT_BROWSERS_PATH = "${s.pkgs.playwright-driver.browsers}";
              PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS = "1";
              PC_SOCKET_PATH = "/tmp/process-compose-reservoir.sock";
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
          tsncd = s.tsncdBundle;
        }
      );

      checks = forAllSystems (system: { tsncd = (perSystem system).tsncdBundle; });
    };
}
