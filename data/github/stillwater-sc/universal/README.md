# https://github.com/stillwater-sc/universal

```console
static/cfloat/standard/tensorfloat.cpp:// tensorfloat.cpp: test suite runner for NVIDIA's TensorFloat
static/cfloat/standard/tensorfloat.cpp:	// cast the NVIDIA TensorFloat onto the classic cfloats
static/cfloat/standard/tensorfloat.cpp:	std::cout << "Standard NVIDIA TensorFloat, which is equivalent to a cfloat<19,8> configuration tests\n";
joss/references.bib:@article{choquette2021nvidia,
joss/references.bib:  title={NVIDIA A100 tensor core GPU: Performance and innovation},
joss/references.bib:  title={Harnessing {GPU} tensor cores for fast {FP16} arithmetic to speed up mixed-precision iterative refinement solvers},
joss/references.bib:  title={TensorFloat-32 in the A100 GPU accelerates AI training HPC up to 20x},
joss/references.bib:  url={https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/},
joss/references.bib:  journal={NVIDIA Corporation, Tech. Rep},
joss/paper.md:The demand for high-performance computing (HPC), machine learning, and deep learning has grown significantly in recent years [e.g., @carmichael:2019; @cococcioni2022small;@desrentes:2022posit8], leading to increased environmental impact and financial cost due to their high energy consumption for storage and processing [@haidar:2018b]. To address these challenges, researchers are exploring ways to reduce energy consumption through redesigning algorithms and minimizing data movement and processing. The use of multi-precision arithmetic in hardware is also becoming more prevalent [@haidar:2018a]. NVIDIA has added support for low-precision formats in its GPUs to perform tensor operations [@choquette2021nvidia], including a 19-bit format with an 8-bit exponent and 10-bit mantissa (see also [@intel:2018; @kharya:2020]. Additionally, Google has developed the "Brain Floating Point Format," known as "bfloat16," which enables the training and operation of deep neural networks using Tensor Processing Units (TPUs) at higher performance and lower cost [@wang2019bfloat16]. This trend towards low-precision numerics is driving the redesign of many standard algorithms, particularly in the field of energy-efficient linear solvers, which is a rapidly growing area of research [@carson:2018; @haidar:2017; @haidar:2018a; @haidar:2018b; @higham:2019].
README.md:The library contains fast implementations of special IEEE-754 formats that do not have universal hardware implementations across x86, ARM, POWER, RISC-V, and GPUs. Special formats such as quarter precision, `quarter`, half-precision, `half`, and quad precision, `quad`, are provided, as well as vendor-specific extensions, such as NVIDIA `TensorFloat`, Google's Brain Float, `bfloat16`, or TI DSP fixed-points, `fixpnt`. In addition to these often-used specializations, *Universal* supports static and elastic integers, decimals, fixed-points, rationals, linear floats, tapered floats, logarithmic, interval and adaptive-precision integers, rationals, and floats. There are example number system skeletons to get you started quickly if you desire to add your own.
README.md:Odeint is a modern C++ library for numerically solving Ordinary Differential Equations. It is developed in a generic way using Template Metaprogramming which leads to extraordinary high flexibility at top performance. The numerical algorithms are implemented independently of the underlying arithmetics. This results in an incredible applicability of the library, especially in non-standard environments. For example, odeint supports matrix types, arbitrary precision arithmetics and even can be easily run on CUDA GPUs.
include/universal/traits/metaprogramming.hpp:#if defined(__CUDA_ARCH__)
include/universal/traits/metaprogramming.hpp:			static float (max)() { return CUDART_MAX_NORMAL_F; }
include/universal/traits/metaprogramming.hpp:			static float infinity() { return CUDART_INF_F; }
include/universal/traits/metaprogramming.hpp:			static float quiet_NaN() { return CUDART_NAN_F; }
include/universal/traits/metaprogramming.hpp:			static double infinity() { return CUDART_INF; }
include/universal/traits/metaprogramming.hpp:			static double quiet_NaN() { return CUDART_NAN; }
include/universal/traits/metaprogramming.hpp:#endif // __CUDA_ARCH__
include/universal/traits/metaprogramming.hpp:#if defined(__CUDA_ARCH__)
include/universal/native/compiler/ieee754_pgi.hpp:// specializations for IEEE-754 parameters for PGI/NVIDIA C/C++
include/universal/number/cfloat/cfloat_impl.hpp:// NVIDIA TensorFloat 
include/sqrt-algorithms-for-precise-representations.txt:That may be true for FDIV and SQRT but it is not true for other functions which can be calculated with Newton-Raphson. The Muller book indicates that one "in general" needs 2×n+3 bits to correctly round things like Ln() or exp(). But it is true that one needs only 4-bits beyond the fraction LoB for "faithful" rounding. Matula patents mention using 6-bits in the fraction (SP) transcendentals ATI (now AMD) GPUs.

```
