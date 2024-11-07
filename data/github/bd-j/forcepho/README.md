# https://github.com/bd-j/forcepho

```console
forcepho/postprocess.py:            #                "/Users/bjohnson/Projects/smacs/pho/output/gpu/")
forcepho/patches/pixel_patch.py:        pixel-data in a format suitable for transfer to the GPU.  Optionally
forcepho/patches/pixel_patch.py:        # --- Pack up all the data for the gpu ---
forcepho/patches/patch.py:Data model for a patch on the sky. The corresponding GPU-side CUDA struct is
forcepho/patches/patch.py:    structures in a manner suitable for sending to the GPU. This includes
forcepho/patches/patch.py:        Whether the residual image will be returned from the GPU.
forcepho/patches/patch.py:        transfer to GPU.
forcepho/patches/patch.py:        calculate these from header/meta information and send data to the GPU
forcepho/patches/device_patch.py:"""device_patch.py - mix-in classes containing methods for communicating with GPU or CPU device
forcepho/patches/device_patch.py:from ..proposal import GPUProposer, CPUProposer
forcepho/patches/device_patch.py:    import pycuda
forcepho/patches/device_patch.py:    import pycuda.driver as cuda
forcepho/patches/device_patch.py:__all__ = ["GPUPatchMixin", "CPUPatchMixin"]
forcepho/patches/device_patch.py:        model : instance of forcepho.model.GPUPosterior
forcepho/patches/device_patch.py:        and for the GPU send the patch_struct to the GPU.  Return a pointer to
forcepho/patches/device_patch.py:        """Subtract a set of big and/or fixed sources from the data on the GPU.
forcepho/patches/device_patch.py:        Leaves the GPU-side *data* array filled with "residuals" from
forcepho/patches/device_patch.py:        or the active sources (if supplied) in the GPU-side meta data arrays.
forcepho/patches/device_patch.py:            Communicator to the GPU for proposals
forcepho/patches/device_patch.py:            Scene corresponding to the meta-data currently on the GPU.  If
forcepho/patches/device_patch.py:        # to GPU and ready to accept proposals.
forcepho/patches/device_patch.py:            print("Warning: The meta-data will be for a scene already subtracted from the GPU-side data array!")
forcepho/patches/device_patch.py:        # to communicate with the GPU
forcepho/patches/device_patch.py:class GPUPatchMixin(DevicePatchMixin):
forcepho/patches/device_patch.py:    """Mix-in class for communicating patch data with the GPU using PyCUDA.
forcepho/patches/device_patch.py:        return GPUProposer()
forcepho/patches/device_patch.py:        """Transfer all the patch data to GPU main memory.  Saves the pointers
forcepho/patches/device_patch.py:        and builds the Patch struct from patch.cc; sends that to GPU memory.
forcepho/patches/device_patch.py:        device_patch: pycuda.driver.DeviceAllocation
forcepho/patches/device_patch.py:            A host-side pointer to the Patch struct on the GPU device.
forcepho/patches/device_patch.py:        # use names of struct dtype fields to transfer all arrays to the GPU
forcepho/patches/device_patch.py:                    self.device_ptrs[arrname] = cuda.to_device(arr)
forcepho/patches/device_patch.py:            self.device_ptrs['residual'] = cuda.mem_alloc(self.xpix.nbytes)
forcepho/patches/device_patch.py:        """Replace the metadata on the GPU, as well as the Cuda pointers. This
forcepho/patches/device_patch.py:                self.device_ptrs[arrname] = cuda.to_device(arr)
forcepho/patches/device_patch.py:        """Create new patch_struct and fill with values and with CUDA pointers
forcepho/patches/device_patch.py:        to GPU arrays, and send the patch_struct to the GPU.
forcepho/patches/device_patch.py:            pass  # no gpu_patch
forcepho/patches/device_patch.py:        # Copy the new patch struct to the gpu
forcepho/patches/device_patch.py:        self.device_patch = cuda.to_device(patch_struct)
forcepho/patches/device_patch.py:            replacing the associated CUDA pointers;
forcepho/patches/device_patch.py:        2) Swap the CUDA pointers for the data and the residual;
forcepho/patches/device_patch.py:        4) Refill the patch_struct array of CUDA pointers and values,
forcepho/patches/device_patch.py:        After this call the GPU uses the former "residual" vector as the "data"
forcepho/patches/device_patch.py:        # replace gpu meta data pointers with currently packed meta
forcepho/patches/device_patch.py:        # Swap Cuda pointers for data and residual
forcepho/patches/device_patch.py:        """Retrieve a pixel array from the GPU
forcepho/patches/device_patch.py:        flatdata = cuda.from_device(self.device_ptrs[arrname],
forcepho/patches/device_patch.py:            for cuda_ptr in self.device_ptrs.values():
forcepho/patches/device_patch.py:                if cuda_ptr:
forcepho/patches/device_patch.py:                    cuda_ptr.free()
forcepho/patches/device_patch.py:            pass  # no cuda_ptrs
forcepho/patches/device_patch.py:            pass  # no gpu_patch
forcepho/patches/device_patch.py:    def test_struct_transfer(self, gpu_patch, cache_dir=False, psf_index=35):
forcepho/patches/device_patch.py:        """Run a simple PyCUDA kernel that checks that the data sent was the
forcepho/patches/device_patch.py:        from pycuda.compiler import SourceModule
forcepho/patches/device_patch.py:        retcode = kernel(gpu_patch, block=(1, 1, 1), grid=(1, 1, 1))
forcepho/patches/device_patch.py:        """Replace the metadata on the GPU, as well as the Cuda pointers. This
forcepho/patches/__init__.py:from .device_patch import GPUPatchMixin, CPUPatchMixin
forcepho/patches/__init__.py:class JadesPatch(StorePatch, GPUPatchMixin):
forcepho/patches/__init__.py:class SimplePatch(FITSPatch, GPUPatchMixin):
forcepho/patches/static_patch.py:Data model for a patch on the sky. The corresponding GPU-side CUDA struct is
forcepho/patches/static_patch.py:        exposures and sources in a manner suitable for sending to the GPU.
forcepho/patches/static_patch.py:        the GPU with PyCUDA.
forcepho/patches/static_patch.py:            Whether the residual image will be returned from the GPU.
forcepho/patches/static_patch.py:        exposures and sources in a manner suitable for sending to the GPU.
forcepho/patches/static_patch.py:        the GPU with PyCUDA.
forcepho/patches/static_patch.py:            All pixels from these stamps (padded to match GPU warp size) will
forcepho/patches/static_patch.py:            be sent to GPU. They should therefore cover similar regions of the
forcepho/patches/static_patch.py:        data to the GPU so it can apply the sky-to-pixel transformations to
forcepho/model.py:    import pycuda
forcepho/model.py:    import pycuda.autoinit
forcepho/model.py:    """A Posterior subclass that uses a GPU or C code to evaluate the likelihood
forcepho/model.py:        """Compute the log-likelihood and its gradient, using the GPU propsoer
forcepho/model.py:            set scene parameters and generate a GPU proposal.
forcepho/model.py:        # --- send to gpu and collect result ---
forcepho/model.py:class GPUPosterior(FastPosterior):
forcepho/proposal.py:This is the CPU-side interface to evaluate a likelihood on the GPU (i.e. make a
forcepho/proposal.py:proposal).  So this is where we actually launch the CUDA kernels with PyCUDA.
forcepho/proposal.py:The corresponding CUDA data model is in proposal.cu, and the CUDA kernels are
forcepho/proposal.py:    import pycuda
forcepho/proposal.py:    import pycuda.driver as cuda
forcepho/proposal.py:    from pycuda.compiler import SourceModule
forcepho/proposal.py:__all__ = ["ProposerBase", "Proposer", "GPUProposer", "CPUProposer"]
forcepho/proposal.py:class GPUProposer(ProposerBase):
forcepho/proposal.py:    This class invokes the PyCUDA kernel.
forcepho/proposal.py:        """Call the GPU kernel to evaluate the likelihood of a parameter proposal.
forcepho/proposal.py:            (and thus ready to send to the GPU).
forcepho/proposal.py:        self.evaluate_proposal_kernel(patch.device_patch, cuda.In(proposal),        # inputs
forcepho/proposal.py:                                      cuda.Out(chi_out), cuda.Out(chi_derivs_out),  # outputs
forcepho/proposal.py:class Proposer(GPUProposer):
forcepho/proposal.py:            (and thus ready to send to the GPU).
forcepho/sources.py:        """Get the parameters of all sources in the scene as a GPU-ready propsal
forcepho/sources.py:            (and thus ready to send to the GPU).
forcepho/sources.py:    can generate proposals for the GPU
forcepho/sources.py:        # For generating proposals that can be sent to the GPU
forcepho/sources.py:        """A parameter proposal in the form required for transfer to the GPU
forcepho/dispatcher.py:        # --- blocking send to parent, free GPU memory ---
forcepho/slow/likelihood.py:        time (like on a GPU)
forcepho/slow/likelihood.py:        time (like on a GPU).
forcepho/kernel_limits.py:We store some compile-time constants for the CUDA kernel in kernel_limits.h.
forcepho/kernel_limits.py:These set the sizes of the structs we use to communicate with the GPU. Thus, we
forcepho/src/patch.cc:This is the data model for the Patch class on the GPU.  A Patch contains
forcepho/src/patch.cc:PyCUDA will build this struct on the GPU side from data in the Python
forcepho/src/patch.cc:TODO: decide how to fill this struct.  CUDA memcpy, or constructor kernel?
forcepho/src/patch.cc:    // Number of bands is known from the CUDA grid size:
forcepho/src/patch.cc:    // (GPU never knows about inactive sources)
forcepho/src/common_kernel.cc:This is the shared GPU/CPU code to compute a Gaussian mixture likelihood and
forcepho/src/common_kernel.cc:CUDA_MEMBER PixFloat ComputeResidualImage(float xp, float yp,
forcepho/src/common_kernel.cc:CUDA_MEMBER void ComputeGaussianDerivative(float xp, float yp, float residual_ierr2,
forcepho/src/common_kernel.cc:CUDA_MEMBER void GetGaussianAndJacobian(PixGaussian & sersicgauss, PSFSourceGaussian & psfgauss,
forcepho/src/compute_gaussians_kernel.cu:on the GPU.  Top-level code view:
forcepho/src/proposal.cu:This is the data model for the Source class on the GPU.
forcepho/src/proposal.cu:/// The response from the GPU likelihood call kernel for for a single band for a single proposal.  A full response will consist of a list of NBANDS of these responses.
forcepho/src/kernel_limits.h:// ease fixed-sized allocations on the GPU
forcepho/src/kernel_limits.h:// The maximum number of active sources that the GPU can use
forcepho/src/kernel_limits.h:// The number of separate accumulators in each GPU block.
forcepho/src/kernel_limits.h:// Shared memory in each GPU block is limited to 48 KB, which is 12K floats.
forcepho/src/compute_gaussians_kernel.cc:    // This is inefficient, but in principle allows for more code to be shared with cuda kernel.
forcepho/src/compute_gaussians_kernel.cc:    // but that would break re-usability between cuda and cpu as currently coded in cuda
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 operator * (const matrix22& A, const float s);
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 operator * (const float s, const matrix22& A);
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 operator + (const matrix22& A, const matrix22& B);
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 operator - (const matrix22& A, const matrix22& B);
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 operator * (const matrix22& A, const matrix22& B);
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER ~matrix22() { }     // A null destructor
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER matrix22() { }    // A null constructor
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER matrix22(float _v11, float _v12, float _v21, float _v22) {
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER matrix22(float _d1, float _d2) {
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER matrix22(const float d[]) {
forcepho/src/matrix22.cc:	CUDA_CALLABLE_MEMBER inline void debug_print(){
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline void eye() { v11 = v22 = 1.0; v12 = v21 = 0.0; }
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline void zero() { v11 = v12 = v21 = v22 = 0.0; }
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline void rot(float theta) {
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline void rotation_matrix_deriv(float theta){
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline void scale(float q) {
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline void scale_matrix_deriv(float q){
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline matrix22 T() { return matrix22(v11, v21, v12, v22); }
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline float det() { return v11*v22-v12*v21; }
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline float trace() { return v11+v22; }
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline matrix22 inv() {
forcepho/src/matrix22.cc:    CUDA_CALLABLE_MEMBER inline matrix22& operator *= ( const float s) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 operator * (const matrix22& A, const float s) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 operator * (const float s, const matrix22& A) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 operator + (const matrix22& A, const matrix22& B) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 operator - (const matrix22& A, const matrix22& B) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 operator * (const matrix22& A, const matrix22& B) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 ABA(const matrix22& A, matrix22& B){ //mamma mia!
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 AAt(const matrix22& A) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 AtA(const matrix22& A) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 symABt(const matrix22& A, const matrix22& B) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 BtAB(const matrix22& A, const matrix22& B) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline matrix22 BABt(const matrix22& A, const matrix22& B) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline float vtAv(matrix22& A, float v1, float v2) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline void Av(matrix22& A, float *v) {
forcepho/src/matrix22.cc:CUDA_CALLABLE_MEMBER inline void Av(float *w, matrix22& A, float *v) {
forcepho/src/header.hh:// Here are the headers for the CUDA code
forcepho/src/header.hh:#ifdef __CUDACC__
forcepho/src/header.hh:#define CUDA_MEMBER __host__ __device__
forcepho/src/header.hh:#define CUDA_CALLABLE_MEMBER __host__ __device__
forcepho/src/header.hh:#include <cuda.h>
forcepho/src/header.hh:#define CUDA_MEMBER
forcepho/src/header.hh:#define CUDA_CALLABLE_MEMBER
docs/api/patches_api.rst:meta data for small regions on the sky, and to communicate that data to the GPU
docs/api/models_api.rst:   :members: Posterior, GPUPosterior, BoundedTransform
docs/install.md:Additional packages that may be necessary (especially for CUDA and MPI) are
docs/install.md:In addition, for GPU and multiprocessing capability the python packages will
docs/install.md:require CUDA and MPI installations (known to work with CUDA 10.1 and open-MPI).
docs/install.md:Currently Forcepho is tested to work only with V100 Nvidia GPUs
docs/install.md:module load cuda/11.4.2-fasrc01     # HeLmod latest
docs/install.md:module load cuda/11.4.2-fasrc01 hdf5/1.10.5-fasrc01
docs/install.md:module load cuda11.2 hdf5/1.10.6 gcc openmpi git slurm
docs/install.md:module load cuda11.2 hdf5/1.10.6 openmpi
docs/install.md:module load cuda11.2 hdf5/1.10.6 openmpi slurm git
docs/install.md:## GPU details
docs/install.md:# Start MPS Daemon on both GPUs on this node
docs/install.md:export CUDA_VISIBLE_DEVICES=0,1 # Select both GPUS
docs/install.md:export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Select a location that’s accessible to the given $UID
docs/install.md:export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Select a location that’s accessible to the given $UID
docs/install.md:nvidia-cuda-mps-control -d # Start the daemon.
docs/install.md:From the odyssey docs: While on GPU node, you can run `nvidia-smi` to get information about the assigned GPU
docs/install.md:-alloc_flags "gpumps"
docs/install.md:jsrun -n1 -g1 -a1  nvprof --analysis-metrics -o /gpfs/wolf/gen126/scratch/bdjohnson/large_prof_metrics%h.%p.nvvp python run_patch_gpu_test_simple.py
docs/install.md:jsrun -n1 -g1 -a1  nvprof --kernels ::EvaluateProposal:1 --metrics flop_count_sp python run_patch_gpu_test_simple.py
docs/configuration.rst:sources to fit simultaneously in a patch.  It is generally limited by GPU memory
docs/tex/forcepho.tex:some of them more amenable to GPU or MIC parallelization:
docs/concepts.md:## CPU/GPU
docs/concepts.md:runs on a GPU and is written in CUDA, requiring compute capability >= 7.0.
docs/concepts.md:Another is written fully in C++, and does not require a GPU.  These kernels do
docs/concepts.md:is handled by `pyCUDA` and `pybind11` respectively.  Each requires different
docs/glossary.md:  A set of N pixels in an exposure that share metadata and that can be completely described by their fluxes, uncertainties, and offset from a central reference pixel.  should be well matched to GPU data model, like say 32 or 128 pixels per superpixel.
tests/galsim_tests/test_sersic/test_sersic_mixture.py:from forcepho.patches import FITSPatch, CPUPatchMixin, GPUPatchMixin
tests/galsim_tests/test_sersic/test_sersic_mixture.py:    import pycuda
tests/galsim_tests/test_sersic/test_sersic_mixture.py:    import pycuda.autoinit
tests/galsim_tests/test_sersic/test_sersic_mixture.py:    HASGPU = True
tests/galsim_tests/test_sersic/test_sersic_mixture.py:    print("NO PYCUDA")
tests/galsim_tests/test_sersic/test_sersic_mixture.py:    HASGPU = False
tests/galsim_tests/test_sersic/test_sersic_mixture.py:if HASGPU:
tests/galsim_tests/test_sersic/test_sersic_mixture.py:    class Patcher(FITSPatch, GPUPatchMixin):
tests/galsim_tests/test_sersic/test_sersic_mixture.py:    print(f"HASGPU={HASGPU}")
tests/galsim_tests/test_sersic/test_sersic_cannon.sh:#SBATCH --partition=gpu_test         # queue for job submission
tests/galsim_tests/test_sersic/test_sersic_cannon.sh:#SBATCH --gres=gpu:1
tests/galsim_tests/test_sersic/test_sersic_cannon.sh:module load cuda/11.4.2-fasrc01     # HeLmod latest
tests/galsim_tests/test_sersic/readme.md:yaml file.  The code will use a GPU if one is available, otherwise the CPU will
tests/galsim_tests/test_psf/test_psf_mixture.py:from forcepho.patches import FITSPatch, CPUPatchMixin, GPUPatchMixin
tests/galsim_tests/test_psf/test_psf_mixture.py:    import pycuda
tests/galsim_tests/test_psf/test_psf_mixture.py:    import pycuda.autoinit
tests/galsim_tests/test_psf/test_psf_mixture.py:    HASGPU = True
tests/galsim_tests/test_psf/test_psf_mixture.py:    print("NO PYCUDA")
tests/galsim_tests/test_psf/test_psf_mixture.py:    HASGPU = False
tests/galsim_tests/test_psf/test_psf_mixture.py:if HASGPU:
tests/galsim_tests/test_psf/test_psf_mixture.py:    class Patcher(FITSPatch, GPUPatchMixin):
tests/galsim_tests/test_psf/test_psf_mixture.py:    print(f"HASGPU={HASGPU}")
tests/galsim_tests/test_psf/make_summary.py:    import pycuda
tests/galsim_tests/test_psf/make_summary.py:    import pycuda.autoinit
tests/galsim_tests/test_psf/make_summary.py:    HASGPU = True
tests/galsim_tests/test_psf/make_summary.py:    print("NO PYCUDA")
tests/galsim_tests/test_psf/make_summary.py:    HASGPU = False
tests/galsim_tests/test_psf/make_summary.py:    print(f"HASGPU={HASGPU}")
tests/galsim_tests/test_psf/test_psf_hst_cannon.slurm:#SBATCH --partition=gpu              # queue for job submission
tests/galsim_tests/test_psf/test_psf_hst_cannon.slurm:#SBATCH --gres=gpu:1
tests/galsim_tests/test_psf/test_psf_hst_cannon.slurm:module load cuda/11.4.2-fasrc01     # HeLmod latest
tests/galsim_tests/test_psf/test_psf_jwst_cannon.slurm:#SBATCH --partition=gpu              # queue for job submission
tests/galsim_tests/test_psf/test_psf_jwst_cannon.slurm:#SBATCH --gres=gpu:1
tests/galsim_tests/test_psf/test_psf_jwst_cannon.slurm:module load cuda/11.4.2-fasrc01     # HeLmod latest
tests/galsim_tests/test_psf/readme.md:specified by a yaml file.  The code will use a GPU if one is available,
tests/verification/verify_cannon.sh:#SBATCH --partition=gpu_test           # queue for job submission
tests/verification/verify_cannon.sh:#SBATCH --gres=gpu:1
tests/verification/verify_cannon.sh:module load cuda/11.4.2-fasrc01     # HeLmod latest
tests/verification/verify_cannon.sh:#module load cuda/10.1.243-fasrc01   # HeLmod
tests/verification/verify_reference.py:    import pycuda
tests/verification/verify_reference.py:    import pycuda.autoinit
tests/verification/verify_reference.py:    HASGPU = True
tests/verification/verify_reference.py:    print("NO PYCUDA")
tests/verification/verify_reference.py:    HASGPU = False
tests/verification/verify_reference.py:                     kernel_type="gpukernel"):
tests/verification/verify_reference.py:    if (kernel_type == "gpukernel") & HASGPU:
tests/verification/verify_reference.py:        args.kernel_type = "gpukernel"
tests/verification/verify_reference.py:    parser.add_argument("--kernel_type", type=str, choices=["cppkernel", "gpukernel"], default="gpukernel")
tests/verification/README.md:This verification will generally require a GPU, and the script can be run using
tests/verification/verify_lux.sh:module load cuda10.2 hdf5/1.10.6
README.md:GPU operation requires Nvidia GPU with compute capability >= 7.0 (developed for V100), a CUDA compiler, and the pycuda python package
scripts/client_shim.sh:# Assign one GPU to this client
scripts/client_shim.sh:# Selects from the GPUs on this node in a round-robin fashion
scripts/client_shim.sh:export CUDA_VISIBLE_DEVICES=$(( ${SLURM_LOCALID} % ${FORCEPHO_GPUS_PER_NODE} ))
scripts/client_shim.sh:echo "Client: host $(hostname), rank ${SLURM_PROCID}, gpu ${CUDA_VISIBLE_DEVICES}"
scripts/client_shim.sh:# A GPU may be assigned to more than one client, but this probably only helps with CUDA MPS,
scripts/client_shim.sh:# and indeed may crash if the GPUs are in exclusive mode.
scripts/client_shim.sh:# The client could select its own GPU instead of relying on this shim layer,
scripts/client_shim.sh:# but one might worry that top-level imports would see the wrong GPU before it could be changed.
scripts/launch_parallel_hetjob.sh:# This is useful because the server needs CPU and memory but not GPU,
scripts/launch_parallel_hetjob.sh:# while the clients need GPU but not much CPU or memory.
scripts/launch_parallel_hetjob.sh:# The risk is that some facilities don't have their GPU and CPU nodes
scripts/launch_parallel_hetjob.sh:gpus_per_node=2
scripts/launch_parallel_hetjob.sh:client_resources="-N1 --ntasks-per-node=1 --cpus-per-task=1 --mem-per-cpu=2G --gres=gpu:${gpus_per_node} -p gpu"
scripts/launch_parallel_hetjob.sh:export FORCEPHO_GPUS_PER_NODE=$gpus_per_node
scripts/launch_parallel.sh:# on the rest.  The GPUs on the server node are wasted.
scripts/launch_parallel.sh:gpus_per_node=2
scripts/launch_parallel.sh:resources="-N1 -n2 --cpus-per-task=3 --mem-per-cpu=3G -m cyclic --gres=gpu:${gpus_per_node} -p gpu"
scripts/launch_parallel.sh:export FORCEPHO_GPUS_PER_NODE=$gpus_per_node
demo/demo_mosaic/mosaic_fit.py:from forcepho.patches import FITSPatch, CPUPatchMixin, GPUPatchMixin
demo/demo_mosaic/readme.md:GPU).
demo/demo_snr/readme.md:GPU).
demo/demo_optimize/optimize.py:from forcepho.patches import FITSPatch, CPUPatchMixin, GPUPatchMixin
demo/demo_optimize/optimize.py:    import pycuda
demo/demo_optimize/optimize.py:    import pycuda.autoinit
demo/demo_optimize/optimize.py:    HASGPU = True
demo/demo_optimize/optimize.py:    print("NO PYCUDA")
demo/demo_optimize/optimize.py:    HASGPU = False
demo/demo_optimize/optimize.py:if HASGPU:
demo/demo_optimize/optimize.py:    class Patcher(FITSPatch, GPUPatchMixin):
demo/demo_optimize/optimize.py:        assert HASGPU
demo/demo_optimize/readme.md:optimization of fluxes is currently only possible when using the GPU.
demo/demo_pair/readme.md:GPU).
demo/demo_color/readme.md:kernel execution in the CPU (as opposed to the GPU).
environment_gpu.yml:    - pycuda==2020.1
optional-requirements.txt:pycuda

```
