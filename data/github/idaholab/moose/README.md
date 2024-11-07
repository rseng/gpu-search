# https://github.com/idaholab/moose

```console
python/TestHarness/TestHarness.py:        parser.add_argument('--libtorch-device', action='store', dest='libtorch_device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='Run libtorch tests with this device')
python/TestHarness/testers/RunApp.py:        params.addParam('libtorch_devices', ['CPU'], "The devices to use for this libtorch test ('CPU', 'CUDA', 'MPS'); default ('CPU')")
python/TestHarness/testers/RunApp.py:            if value.lower() not in ['cpu', 'cuda', 'mps']:
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda mps'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda mps'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda mps'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda'
test/tests/utils/libtorch_nn/ann/tests:      libtorch_devices = 'cpu cuda'
apptainer/moose-dev.def:    # Adding this to not get GPU initialization errors from MPICH
apptainer/moose-dev.def:    export MPIR_CVAR_ENABLE_GPU=0
apptainer/moose-dev.def:    # We install CUDA Toolkit if the user wants cuda-based libtorch.
apptainer/moose-dev.def:    # Right now this assumes that cuda-based distributions start with -cu-
apptainer/moose-dev.def:        CUDA_RPM=${BUILD_DIR}/cuda.rpm
apptainer/moose-dev.def:        curl -L https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-rhel8-11-4-local-11.4.0_470.42.01-1.x86_64.rpm -o ${CUDA_RPM}
apptainer/moose-dev.def:        rpm -i ${CUDA_RPM}
apptainer/moose-dev.def:        dnf -y install cuda
apptainer/moose-dev.def:        rm -rf ${CUDA_RPM}
framework/doc/content/sqa/framework_sll.md:> supports MPI, and GPUs through CUDA or OpenCL, as well as hybrid MPI-GPU parallelism. PETSc
framework/doc/content/sqa/framework_sll.md:> MPI, OpenMP and CUDA to support various forms of parallelism. It supports both real and complex datatypes,
framework/doc/content/source/userobjects/BatchMaterial.md:`BatchMaterial` implements a generic base class for a userobject that can gather MaterialProperties and Variables from all QPs in the local domain to perform a computation in a single batch (offloaded to a GPU for example).
framework/doc/content/source/userobjects/BatchMaterial.md:The resulting userobject will generate an "array of structs" for optimal cache locality in the batch computation. The input data "struct" is a tuple. The base class is templated on a tuple wrapper class. This allows use of either `std::tuple` or `cuda::std::tuple` (which has corresponding implementations on the host and the device with identical memory layouts!).
framework/doc/content/contrib/reveal/reveal.css:  background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABkklEQVRYR8WX4VHDMAxG6wnoJrABZQPYBCaBTWAD2g1gE5gg6OOsXuxIlr40d81dfrSJ9V4c2VLK7spHuTJ/5wpM07QXuXc5X0opX2tEJcadjHuV80li/FgxTIEK/5QBCICBD6xEhSMGHgQPgBgLiYVAB1dpSqKDawxTohFw4JSEA3clzgIBPCURwE2JucBR7rhPJJv5OpJwDX+SfDjgx1wACQeJG1aChP9K/IMmdZ8DtESV1WyP3Bt4MwM6sj4NMxMYiqUWHQu4KYA/SYkIjOsm3BXYWMKFDwU2khjCQ4ELJUJ4SmClRArOCmSXGuKma0fYD5CbzHxFpCSGAhfAVSSUGDUk2BWZaff2g6GE15BsBQ9nwmpIGDiyHQddwNTMKkbZaf9fajXQca1EX44puJZUsnY0ObGmITE3GVLCbEhQUjGVt146j6oasWN+49Vph2w1pZ5EansNZqKBm1txbU57iRRcZ86RWMDdWtBJUHBHwoQPi1GV+JCbntmvok7iTX4/Up9mgyTc/FJYDTcndgH/AA5A/CHsyEkVAAAAAElFTkSuQmCC); }
framework/include/userobjects/BatchMaterial.h:#ifdef CUDA_SUPPORTED
framework/include/userobjects/BatchMaterial.h:struct TupleCuda
framework/include/userobjects/BatchMaterial.h:  using type = cuda::std::tuple<Args...>;
framework/include/userobjects/BatchMaterial.h:    return cuda::std::get<N>(t);
framework/include/userobjects/BatchMaterial.h:  using size = cuda::std::tuple_size<T>;
framework/include/userobjects/BatchMaterial.h:  using element = typename cuda::std::tuple_element<I, T>::type;
framework/include/userobjects/BatchMaterial.h:  /// input data needs to use a flexible tuple type (std::tuple or cuda::std::tuple)
framework/include/base/MooseApp.h:#include <torch/cuda.h>
framework/include/base/MooseApp.h:   * @param device Enum to describe if a cpu or a gpu should be used.
framework/include/utils/Shuffle.h: *           and also with the avent of large core clusters and GPUs, there is an interest in making
framework/contrib/gtest/gtest/gtest.h:// with a TR1 tuple implementation.  NVIDIA's CUDA NVCC compiler
framework/contrib/gtest/gtest/gtest.h:# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000) \
framework/src/base/MooseApp.C:  MooseEnum libtorch_device_type("cpu cuda mps", "cpu");
framework/src/base/MooseApp.C:  if (device_enum == "cuda")
framework/src/base/MooseApp.C:    if (!torch::cuda::is_available())
framework/src/base/MooseApp.C:      mooseError("--libtorch-device=cuda: CUDA is not available");
framework/src/base/MooseApp.C:    return torch::kCUDA;
framework/src/base/MooseApp.C:    mooseError("--libtorch-device=cuda: CUDA is not supported on your platform");
modules/doc/content/getting_started/installation/install_libtorch.md:!alert! note title=GPU Support on Macs
modules/doc/content/getting_started/installation/install_libtorch.md:to the Metal Performance Shader (MPS) capabilities (GPU acceleration).
modules/doc/content/getting_started/installation/install_libtorch.md:!alert! note title=GPU Support on Linux Workstations
modules/doc/content/getting_started/installation/install_libtorch.md:using suitable system compilers. At the time these instruction are written, only `cuda`-based
modules/doc/content/getting_started/installation/install_libtorch.md:- [A sutiable Nvidia driver](https://www.nvidia.com/en-us/drivers/)
modules/doc/content/getting_started/installation/install_libtorch.md:- [Cuda toolkit](https://developer.nvidia.com/cuda-toolkit) - only strictly required if
modules/doc/content/getting_started/installation/install_libtorch.md:./scripts/setup_libtorch.sh --version=2.1 --libtorch-distribution=cuda
modules/doc/content/newsletter/2024/2024_01.md:- The [NEML2](syntax/NEML2/index.md) constitutive model library was added to the solid mechanics module as an optional dependency. NEML2 is built on top of libTorch and supports the implementation of device-independent material models (with GPU support).
modules/doc/content/newsletter/2023/2023_01.md:- Fixes and workarounds for NVidia HPC SDK compiler warnings.  These
modules/doc/content/newsletter/2022/2022_03.md:- Enable setting PETSc mat/vec types at runtime for future GPU use
modules/doc/content/newsletter/2022/2022_03.md:- PETSc has better GPU support via native solver options like GAMG, or other third-party
modules/doc/content/newsletter/2022/2022_03.md:  solver options such as HYPRE. The enhanced GPU capability can be triggered in
modules/doc/content/newsletter/2022/2022_03.md:  MOOSE via the `-mat_type aijcusparse -vec_type cuda` command line option. Note
modules/doc/content/newsletter/2022/2022_03.md:  that GPU capability is still experimental in MOOSE, and only algebra solver
modules/doc/content/newsletter/2022/2022_03.md:  operations are available on GPUs. Residual and Jacobian evaluations are not
modules/doc/content/newsletter/2022/2022_03.md:  available on GPUs yet. If you want to have meaningful performance improvement,
modules/doc/content/newsletter/2022/2022_11.md:- Fixes for errors with Nvidia HPC SDK compilers
modules/doc/content/newsletter/2022/2022_04.md:The MOOSE-LibTorch coupling only supports serial neural net training and does not support GPU-based training.
modules/doc/content/applications/thermal_hydraulics.md:| [Cardinal](https://cardinal.cels.anl.gov/) | [Open-source](https://github.com/neams-th-coe/cardinal) | [NekRS](https://github.com/Nek5000/nekRS) CFD | CPU and GPU capabilities for RANS, LES, and DNS. Additional features include Lagrangian particle transport, an ALE mesh solver, overset meshes, and more. |
modules/tensor_mechanics/include/neml2/interfaces/NEML2ModelInterface.h:      "schema: (cpu|cuda)[:<device-index>] where cpu or cuda specifies the device type, and "
modules/tensor_mechanics/include/neml2/interfaces/NEML2ModelInterface.h:      "target compute device to be CPU, and device='cuda:1' sets the target compute device to be "
modules/tensor_mechanics/include/neml2/interfaces/NEML2ModelInterface.h:      "CUDA with device ID 1.");
modules/tensor_mechanics/src/neml2/actions/NEML2Action.C:      "schema: (cpu|cuda)[:<device-index>] where cpu or cuda specifies the device type, and "
modules/richards/doc/tests/data/wli.nb:x/s+b1TFCEaWs2R9O3zyzfJxdlPDgpU3u5vD30OTY2tqO98eHD88zK+0twNu
modules/richards/doc/tests/figures/s04.eps:s+13$s+13$s+13$s+13$s3gpu!.b.;!;6B>!9X=P!9=+S!9=+P!;6B>!;6Be!7(W>
modules/richards/doc/tests/figures/ex02_3D_mesh.eps:DaIQR=?WnVb+9@d]L4$qA*dj)KO0D+!gPu?l65t4;3H1.mZCE&WbQ9"?0Fa&+`qB,
modules/richards/doc/tests/figures/bl2_initial.eps:RlYUetDm^JBFh!<&7]M<LN7Y-aMYp0j_rL9gpuBAdn<_8o6%?`Y\nCa(fCg<[-i>@
modules/richards/doc/tests/figures/th_mesh_zoom.eps:mh&TMGPU;-$d7"b".&)S!/"-15nUEk"0B),;/Ul^P,J1Hi4%ice3p0GVe0c`lj]gG
modules/richards/doc/tests/figures/th_mesh_zoom.eps:_O28Se;Z>::Un.YX=8XCo!r+/:5O@hGpubSdPKE>Pjd"HOr4QpJg5>\eAts2kqbge
modules/richards/doc/tests/figures/th_mesh_zoom.eps:&`p\\o%uT8^^[SW&%h\'WqR26'j4jN"`P+dK`(MROCmrS3]r.<Y)0LpKBA",`',lS
modules/richards/doc/tests/figures/th_mesh_zoom.eps:Ipbd&4XQmMRnAJC0qD)g;2A0.q-26i\[)Dsi]@CgPU6rSOB_kZATZ];-rF=g4&aA.
modules/richards/doc/tests/figures/bl2_seff.eps:agFr<]rrBbDpsHNWn&PG.HlqmX.%gpuCG1iinbr4]"9%9K;XKhJnc&UH#Q/9e^XTa
modules/richards/doc/tests/figures/gh_seff_50.eps:7;U>Zo?gpuIpa-fpJ+tKh^k(GT*Eu6HXD&;)mtQ3a?ik)7)8%)@qA8142Uq=BA%"X
modules/richards/doc/tests/figures/bh07.mesh.eps:UKW[#g)h^edXkD8jZ7_mk9bJBQA+jbc&X&<g0l#DG'N%gPU5C?!U5C?!U5C?!U5C?
modules/richards/doc/tests/figures/bh07.mesh.eps:SHimo.<7F$OiJgpU3crenL-)M8g>3(%JBT0<1)5g=$<8(LXE\g)H?Y<q4^kDU(\V%
modules/richards/doc/tests/figures/bh07.mesh.eps:!USml+9XqRocMG^X[5Tj3Xj6a;e?rR_elGi*X^n[N=X'*>ZIbND`r?#L[=<p<ZMso
modules/richards/doc/tests/figures/bh07.mesh.eps:OGQ&F0iQm;YIg+M@57=<GpU9Ve)d0cArgaeFqmUT[[dL_TeA-aZ]!e#;=JLmFOu3P
modules/richards/doc/tests/figures/th_mesh.eps:p8R,J7(@mB!<LW9LU%0p$^Ra[TnDF6G$M<64G_]MSHimo.<7F$OiJgpU3crenL-)M
modules/richards/doc/tests/figures/th_mesh.eps:%'#0uXd3R=BDCo]jg!G0Wpg:qR$fpe^3\?W;L$gPU[C`ZW;gOmi>Sh1>FCm/W00;`
modules/richards/doc/tests/figures/ex02_1.5.eps:*YF9H%dI7KGpU55JpF2Zc'B4,S>hh%&XjB@J_JZGI=4!4hqClb7H_V8Ljp'pgFjPU
modules/richards/doc/tests/figures/gh2_20.eps:?Eo_)cHo%UmP4MP9Z:r_$"gQRaZ?@:AMk:?+Ah@*'4u0G6rrC#rfX9$AgPu&YPn5%
modules/richards/doc/tests/figures/gh2_20.eps:YWaUP*PP8UA.`bhgpuhYW"j?[Vn-AZDZI(NiY-i`^le`%k^PL_f^!X4O06Id<bGf6
modules/richards/doc/theory/figures/bl_lumped_unlumped_zoom.eps:9f\P1:\5iY%COL=Hf*oF-h2TRYTH=cLMYp0j_rL9gpuBAdn<_8o6%?`Y\nCa(fCg<
modules/richards/doc/theory/figures/graphs.nb:I8+fsGee41AweIVzWVO6gpueEHVC+jV9Pq/QlFNyQMUigLh9U+IQ3cgrRIrU
modules/richards/doc/theory/figures/van_genuchten_mod.nb:fYJhR57qBO5LMVIyqy2hytK8u0d7AsxZV6o1aB9gpuWKMLfZBN6vvX6q4/UA
modules/richards/doc/theory/figures/bl_lumped_unlumped.eps:*V&0<cIqQOMI`(4.VrBi\?O_7`_#>qTMqa5$_nI@Ih(Pki\417Z_qRmK][4eGPUqp
modules/solid_mechanics/test/tests/capped_mohr_coulomb/figures/capped_mc_mc_with_planes.eps:$gpu]#RBPkJJ8=>/Mgt4gT7FZn4S`m/+6f4?o@Z!c2`_>\QhF\Q.k8FC=;k0l880;
modules/solid_mechanics/test/tests/capped_mohr_coulomb/figures/capped_mc_2D_base.eps:9Gpu.4'=.9$q#<nq1hSCTF>5_a`.=:=-6p4&oOhnG,5]q_RP1.[Pnf8Z-gV_mUJtO
modules/solid_mechanics/test/tests/capped_mohr_coulomb/figures/random5.eps:nl[SgPU&-"*&:?G+hoB"L0D*Qq(q`)m/BNG=FRu-1c1B'lGuo&(p;\+6rOaotOR`,
modules/solid_mechanics/test/tests/capped_mohr_coulomb/figures/capped_mc_3D_smoothed.eps:QMSp=fo\k3eJD=B/!(qS&)"1Qnqqdntkcc#,J*Lb6W+WeO$i^3QGeZGPU;1],7igj
modules/solid_mechanics/test/tests/capped_mohr_coulomb/figures/capped_mc_3D_smoothed.eps:6>Q,+Kn1KoQq^[%UnO1rtp"^k3j%?Ygo3?QAf2brccQ9lF*UdWWE^=-qi"-cWBGpU
modules/solid_mechanics/doc/content/syntax/NEML2/index.md:The field [!param](/NEML2/NEML2Action/input) specifies the relative path to the NEML2 input file. The field [!param](/NEML2/NEML2Action/model) tells MOOSE which material model to import from the NEML2 input file. The field [!param](/NEML2/NEML2Action/device) specifies where to evaluate the NEML2 model, e.g., CPU or CUDA. The parameter [!param](/NEML2/NEML2Action/mode) determines the mode of operation for NEML2, and it is important to understand the differences between the modes in order to use NEML2 most efficiently. Each mode is discussed below in detail.
modules/solid_mechanics/doc/content/syntax/NEML2/index.md:As discussed earlier, the ELEMENT mode and the ALL mode produce input vectors with different batch sizes. NEML2 handles the threading and vectorization of the batched evaluation. When the batch size is large (i.e. in the ALL mode), it allows for a potentially more aggressive use of the available computing resource, and GPUs can make the evaluation a lot faster relying on massive vectorization.
modules/solid_mechanics/doc/content/modules/solid_mechanics/NEML2.md:Unlike NEML, NEML2 vectorizes the constitutive update to efficiently run on GPUs.  NEML2 is
modules/solid_mechanics/doc/content/modules/solid_mechanics/NEML2.md:built on top of [libTorch](https://pytorch.org/cppdocs/) to provide GPU support, but this also
modules/solid_mechanics/include/neml2/interfaces/NEML2ModelInterface.h:      "schema: (cpu|cuda)[:<device-index>] where cpu or cuda specifies the device type, and "
modules/solid_mechanics/include/neml2/interfaces/NEML2ModelInterface.h:      "target compute device to be CPU, and device='cuda:1' sets the target compute device to be "
modules/solid_mechanics/include/neml2/interfaces/NEML2ModelInterface.h:      "CUDA with device ID 1.");
modules/solid_mechanics/src/neml2/actions/NEML2Action.C:      "schema: (cpu|cuda)[:<device-index>] where cpu or cuda specifies the device type, and "
conda/petsc/build.sh:#if [[ "${build_variant}" == 'cuda' ]]; then
conda/petsc/build.sh:#  ADDITIONAL_ARGS+=" --download-slate=1 --with-cuda=1 --with-cudac=${PREFIX}/bin/nvcc \
conda/petsc/build.sh:#                     --with-cuda-dir=${PREFIX}/targets/x86_64-linux \
conda/petsc/build.sh:#                     --CUDAFLAGS=-I${PREFIX}/targets/x86_64-linux/include"
conda/petsc/build.sh:## Cuda specific activation/deactivation variables (append to above created script)
conda/petsc/build.sh:#if [[ "${build_variant}" == 'cuda' ]] && [[ "$mpi" == "openmpi" ]]; then
conda/petsc/build.sh:#export OMPI_MCA_opal_cuda_support=true
conda/petsc/build.sh:#unset OMPI_MCA_opal_cuda_support
scripts/setup_libtorch.sh:  echo "--libtorch-distribution=DISTRIBUTION Specify the distribution (cpu/cuda)"

```
