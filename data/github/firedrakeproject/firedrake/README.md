# https://github.com/firedrakeproject/firedrake

```console
docs/source/_static/bibliography.bib:for GPUs from high-level specifications},
scripts/firedrake-install:    parser.add_argument("--torch", const="cpu", default=False, nargs='?', choices=["cpu", "cuda"],
scripts/firedrake-install:                        help="Install PyTorch for a CPU or CUDA backend (default: CPU).")
scripts/firedrake-install:    parser.add_argument("--jax", const="cpu", default=False, nargs='?', choices=["cpu", "cuda"],
scripts/firedrake-install:                        help="Install JAX for a CPU or CUDA backend (default: CPU).")
scripts/firedrake-install:    parser.add_argument("--torch", const="cpu", nargs='?', choices=["cpu", "cuda"], default=config["options"].get("torch", False),
scripts/firedrake-install:                        help="Install PyTorch for a CPU or CUDA backend (default: CPU).")
scripts/firedrake-install:    parser.add_argument("--jax", const="cpu", nargs='?', choices=["cpu", "cuda"], default=config["options"].get("jax", False),
scripts/firedrake-install:                        help="Install JAX for a CPU or CUDA backend (default: CPU).")
scripts/firedrake-install:            petsc_options.add("--download-hwloc-configure-arguments=--disable-opencl")
scripts/firedrake-install:            petsc_options.add("--download-mpich-configure-arguments=--disable-opencl")
scripts/firedrake-install:    """Install PyTorch for a CPU or CUDA backend."""
scripts/firedrake-install:    if osname == "Darwin" and args.torch == "cuda":
scripts/firedrake-install:        raise InstallError("CUDA installation is not available on MacOS.")
scripts/firedrake-install:    """Install JAX for a CPU or CUDA backend."""
scripts/firedrake-install:    version_name = "jax" if args.jax == "cpu" else "jax[cuda12]"
scripts/firedrake-zenodo:    ("loopy", "Transformation-Based Generation of High-Performance CPU/GPU Code"),
firedrake/ml/jax/fem_operator.py:        raise NotImplementedError("Firedrake does not support GPU/TPU tensors")
firedrake/ml/pytorch/fem_operator.py:        raise NotImplementedError("Firedrake does not support GPU/TPU tensors")

```
