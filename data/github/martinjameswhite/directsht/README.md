# https://github.com/martinjameswhite/directsht

```console
create_jax_env.sh:module load cudatoolkit
create_jax_env.sh:# Verify the versions of cudatoolkit and cudnn are compatible with JAX
create_jax_env.sh:python3 -m pip install --upgrade "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
README.md:The code is much faster when run on GPUs. When they are available and JAX is installed, the code automatically distributes computation and memory across them.
sht/shared_interp_funcs.py:        # Run loop in numpy and possibly move to GPU later
sht/legendre_jax.py:    # Now we distribute/shard it across GPUS. Note that we should only do this
sht/utils_jax.py:    Helper function to shard (i.e. distribute) an array across devices (typically GPUs).
sht/utils_jax.py:    # Move to the GPU with the desired sharding scheme
sht/sht.py:            # Mask and put in GPU memory distributing across devices (if possible)
sht/sht.py:                # Get a grid of all alm's by batching over (ell,m) -- best run on a GPU!

```
