# https://github.com/Pranab-JD/LeXInt

```console
CUDA/Leja.hpp:#include "Kernels_CUDA_Cpp.hpp"
CUDA/Leja.hpp:    GPU_handle cublas_handle;       //? Modified handle for cublas
CUDA/Leja.hpp:    //! Allocate memory - these are device vectors if GPU support is activated
CUDA/Leja.hpp:        #ifdef __CUDACC__
CUDA/Leja.hpp:            cudaMalloc(&auxiliary_Leja, 4 * N * sizeof(double));
CUDA/Leja.hpp:            cudaMalloc(&auxiliary_expint, num_vectors * N * sizeof(double));
CUDA/Leja.hpp:        #ifdef __CUDACC__
CUDA/Leja.hpp:            cudaFree(auxiliary_Leja);
CUDA/Leja.hpp:            cudaFree(auxiliary_expint);
CUDA/Leja.hpp:                          bool GPU
CUDA/Leja.hpp:        LeXInt::Power_iterations(RHS, u_input, N, eigenvalue, auxiliary_Leja, GPU, cublas_handle);
CUDA/Leja.hpp:                          bool GPU
CUDA/Leja.hpp:                                 N, (* phi_function), Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);
CUDA/Leja.hpp:                       bool GPU
CUDA/Leja.hpp:                              N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);
CUDA/Leja.hpp:                 bool GPU
CUDA/Leja.hpp:                           N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);
CUDA/Leja.hpp:                              N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);
CUDA/Leja.hpp:                       bool GPU
CUDA/Leja.hpp:                            N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);
CUDA/Leja.hpp:                            N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);        
CUDA/Leja.hpp:                            N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);        
CUDA/Leja.hpp:                              N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);        
CUDA/Leja.hpp:                              N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);        
CUDA/Leja.hpp:                             N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);         
CUDA/Leja.hpp:                             N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);         
CUDA/Leja.hpp:                             N, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);         
CUDA/error_check.hpp:#ifdef __CUDACC__
CUDA/error_check.hpp://* https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
CUDA/error_check.hpp:#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
CUDA/error_check.hpp:inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
CUDA/error_check.hpp:   if (code != cudaSuccess) 
CUDA/error_check.hpp:      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
CUDA/Test/test_2D.cpp:    //! Set GPU spport to false
CUDA/Test/test_2D.cpp:    bool GPU_access = false;
CUDA/Test/test_2D.cpp:    // Leja<RHS_Dif_Adv_2D> leja_gpu{N, integrator};
CUDA/Test/test_2D.cpp:    Leja<RHS_Burgers_2D> leja_gpu{N, integrator};
CUDA/Test/test_2D.cpp:    leja_gpu.Power_iterations(RHS, u, eigenvalue, GPU_access);
CUDA/Test/test_2D.cpp:            leja_gpu.real_Leja_exp(RHS, u, u_sol, c, Gamma, rtol, atol, dt, iters, GPU_access);
CUDA/Test/test_2D.cpp:            LeXInt::axpby(1.0, source, 1.0, u, interp_vector, N, GPU_access);
CUDA/Test/test_2D.cpp:            LeXInt::axpby(dt, interp_vector, interp_vector, N, GPU_access);
CUDA/Test/test_2D.cpp:            leja_gpu.real_Leja_phi_nl(RHS, interp_vector, u_sol, LeXInt::phi_1, c, Gamma, rtol, atol, dt, iters, GPU_access);
CUDA/Test/test_2D.cpp:                leja_gpu.Power_iterations(RHS, u, eigenvalue, GPU_access);
CUDA/Test/test_2D.cpp:            leja_gpu.exp_int(RHS, u, u_sol, c, Gamma, rtol, atol, dt, iters, GPU_access);
CUDA/Test/test_2D.cpp:                leja_gpu.Power_iterations(RHS, u, eigenvalue, GPU_access);
CUDA/Test/test_2D.cpp:            leja_gpu.embed_exp_int(RHS, u, u_low, u_sol, error, c, Gamma, rtol, atol, dt, iters, GPU_access);
CUDA/Test/test_2D.cpp:            LeXInt::axpby(1.0, u, 1.0, u_sol, u, N, GPU_access);
CUDA/Test/test_2D.cpp:            LeXInt::copy(u_sol, u, N, GPU_access);
CUDA/Test/test_2D.cpp:    int sys_value_f = system(("mkdir -p ../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/" + integrator + "/cores_4"
CUDA/Test/test_2D.cpp:    string directory_f = "../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/" + integrator + "/cores_4"
CUDA/Test/test_2D.cu:#include <cuda.h>
CUDA/Test/test_2D.cu:    //! Set GPU support to true
CUDA/Test/test_2D.cu:    bool GPU_access = true;
CUDA/Test/test_2D.cu:    Leja<RHS_Dif_Adv_2D> leja_gpu{N, integrator};
CUDA/Test/test_2D.cu:    // Leja<RHS_Burgers_2D> leja_gpu{N, integrator};
CUDA/Test/test_2D.cu:    cudaDeviceSynchronize();
CUDA/Test/test_2D.cu:    gpuErrchk(cudaPeekAtLastError());
CUDA/Test/test_2D.cu:    //! Allocate memory on GPU
CUDA/Test/test_2D.cu:    double *device_u; cudaMalloc(&device_u, N_size);
CUDA/Test/test_2D.cu:    cudaMemcpy(device_u, &u[0], N_size, cudaMemcpyHostToDevice);                    //* Copy state variable to device
CUDA/Test/test_2D.cu:    double *device_u_sol; cudaMalloc(&device_u_sol, N_size);                        //* Solution vector
CUDA/Test/test_2D.cu:        cudaMalloc(&device_interp_vector, N_size);
CUDA/Test/test_2D.cu:        cudaMalloc(&device_source, N_size);
CUDA/Test/test_2D.cu:        cudaMemcpy(device_source, &Source[0], N_size, cudaMemcpyHostToDevice);
CUDA/Test/test_2D.cu:        cudaMalloc(&device_u_low, N_size);
CUDA/Test/test_2D.cu:    leja_gpu.Power_iterations(RHS, device_u, eigenvalue, GPU_access);
CUDA/Test/test_2D.cu:    cudaDeviceSynchronize();
CUDA/Test/test_2D.cu:    gpuErrchk(cudaPeekAtLastError());
CUDA/Test/test_2D.cu:        cudaDeviceSynchronize();
CUDA/Test/test_2D.cu:            leja_gpu.real_Leja_exp(RHS, device_u, device_u_sol, c, Gamma, rtol, atol, dt, iters, GPU_access);
CUDA/Test/test_2D.cu:            LeXInt::axpby(1.0, device_source, 1.0, device_u, device_interp_vector, N, GPU_access);
CUDA/Test/test_2D.cu:            LeXInt::axpby(dt, device_interp_vector, device_interp_vector, N, GPU_access);
CUDA/Test/test_2D.cu:            leja_gpu.real_Leja_phi_nl(RHS, device_interp_vector, device_u_sol, LeXInt::phi_1, c, Gamma, rtol, atol, dt, iters, GPU_access);
CUDA/Test/test_2D.cu:                leja_gpu.Power_iterations(RHS, device_u, eigenvalue, GPU_access);
CUDA/Test/test_2D.cu:            leja_gpu.exp_int(RHS, device_u, device_u_sol, c, Gamma, rtol, atol, dt, iters, GPU_access);
CUDA/Test/test_2D.cu:                leja_gpu.Power_iterations(RHS, device_u, eigenvalue, GPU_access);
CUDA/Test/test_2D.cu:            leja_gpu.embed_exp_int(RHS, device_u, device_u_low, device_u_sol, error, c, Gamma, rtol, atol, dt, iters, GPU_access);
CUDA/Test/test_2D.cu:            LeXInt::axpby(1.0, device_u, 1.0, device_interp_vector, device_u, N, GPU_access);
CUDA/Test/test_2D.cu:            LeXInt::copy(device_u_sol, device_u, N, GPU_access);
CUDA/Test/test_2D.cu:    cudaDeviceSynchronize(); 
CUDA/Test/test_2D.cu:    // int sys_value_f = system(("mkdir -p ../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/" + integrator
CUDA/Test/test_2D.cu:    // string directory_f = "../../LeXInt_Test/" + to_string(GPU_access) + "/Constant/" + problem + "/" + integrator
CUDA/Test/test_2D.cu:    // cudaMemcpy(&u[0], device_u, N_size, cudaMemcpyDeviceToHost);   
CUDA/Test/Dif_Adv_2D.hpp:#ifdef __CUDACC__
CUDA/Test/Dif_Adv_2D.hpp:        #ifdef __CUDACC__
CUDA/Test/Burgers_2D.hpp:#ifdef __CUDACC__
CUDA/Test/Burgers_2D.hpp:        #ifdef __CUDACC__
CUDA/Test/Problems.hpp:#ifdef __CUDACC__
CUDA/real_Leja_exp.hpp:                       bool GPU,                       //? false (0) --> CPU; true (1) --> GPU
CUDA/real_Leja_exp.hpp:                       GPU_handle& cublas_handle       //? CuBLAS handle
CUDA/real_Leja_exp.hpp:        axpby(coeffs[0], u, polynomial, N, GPU);
CUDA/real_Leja_exp.hpp:            axpby(1./Gamma, Jac_vec, (-c/Gamma - Leja_X[iters - 1]), u, u, N, GPU);
CUDA/real_Leja_exp.hpp:            axpby(coeffs[iters], u, 1.0, polynomial, polynomial, N, GPU);
CUDA/real_Leja_exp.hpp:            double poly_error = l2norm(u, N, GPU, cublas_handle)/sqrt(N);
CUDA/real_Leja_exp.hpp:            double poly_norm = l2norm(polynomial, N, GPU, cublas_handle)/sqrt(N);
CUDA/Readme.md:# CUDA
CUDA/Readme.md:![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
CUDA/Readme.md:Test examples for C++ and CUDA implementations can be found in *Test &rarr; Test_2D.cpp* and *Test &rarr; Test_2D.cu*, respectively.  To run the codes, use `bash run_cpp.sh` or `bash run_cuda.sh`. Alternatively, you could also use *sbatch* instead of *bash* if you have *slurm* installed on your computer. Problems considered include the linear diffusion-advection equation and the nonlinear Burgers' equation. To add other problems, simply define the relevant RHS function (as defined in *Burgers_2D.hpp* or *Dif_Adv_2D.hpp*) and the initial condition(s) in the test files.
CUDA/Readme.md:- NVIDIA GPU
CUDA/Readme.md:- CUDA 11.2 (or later)
CUDA/Integrators/Rosenbrock_Euler.hpp:                bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
CUDA/Integrators/Rosenbrock_Euler.hpp:                GPU_handle& cublas_handle   //? CuBLAS handle
CUDA/Integrators/Rosenbrock_Euler.hpp:        //! are device vectors if GPU support is activated.
CUDA/Integrators/Rosenbrock_Euler.hpp:        axpby(dt, f_u, f_u, N, GPU);
CUDA/Integrators/Rosenbrock_Euler.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);
CUDA/Integrators/Rosenbrock_Euler.hpp:        axpby(1.0, u, 1.0, u_exprb2, u_exprb2, N, GPU);
CUDA/Integrators/Readme.md:#  [LeXInt::CUDA::Integrators](#)
CUDA/Integrators/Readme.md:- Add ``#include "./LeXInt/CUDA/Leja.hpp"`` in the main file (main.cpp or main.cu).
CUDA/Integrators/Readme.md:- Create an object of the class as ``Leja(N, integrator_name)``, where 'N' is the total number of grid points and 'integrator_name' corresponds to the desired exponential integrator. E.g., ``Leja<RHS> leja_gpu{N, EXPRB32}``; where ``RHS``is RHS class that contains the RHS operator.
CUDA/Integrators/Readme.md:- Invoke the object of the class ``Leja`` as ``leja_gpu.embed_exp_int`` for embedded exponential integrators or ``leja_gpu.exp_int`` for non-embedded exponential integrators. For more info, see `Test -> test_2D.cu (lines 231 and 250)`.
CUDA/Integrators/EPIRK4s3A.hpp:                   bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
CUDA/Integrators/EPIRK4s3A.hpp:                   GPU_handle& cublas_handle   //? CuBLAS handle
CUDA/Integrators/EPIRK4s3A.hpp:        //! are device vectors if GPU support is activated.
CUDA/Integrators/EPIRK4s3A.hpp:        axpby(dt, f_u, f_u, N, GPU);
CUDA/Integrators/EPIRK4s3A.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3A.hpp:        axpby(1.0, u, 1./2., &u_flux[0], a, N, GPU);
CUDA/Integrators/EPIRK4s3A.hpp:        axpby(1.0, u, 2./3., &u_flux[N], b, N, GPU);
CUDA/Integrators/EPIRK4s3A.hpp:        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3A.hpp:        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3A.hpp:        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3A.hpp:        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
CUDA/Integrators/EPIRK4s3A.hpp:        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);
CUDA/Integrators/EPIRK4s3A.hpp:        axpby(32.0, R_a, -27.0/2.0, R_b, R_3, N, GPU);
CUDA/Integrators/EPIRK4s3A.hpp:        axpby(-144.0, R_a, 81.0, R_b, R_4, N, GPU);
CUDA/Integrators/EPIRK4s3A.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3A.hpp:                      phi_4, Leja_X, c, Gamma, rtol, atol, dt, iters_3, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3A.hpp:        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_3, u_epirk3, N, GPU);
CUDA/Integrators/EPIRK4s3A.hpp:        axpby(1.0, u_epirk3, 1.0, u_nl_4, u_epirk4, N, GPU);
CUDA/Integrators/EPIRK4s3A.hpp:        error = l2norm(u_nl_4, N, GPU, cublas_handle)/sqrt(N);
CUDA/Integrators/EXPRB42.hpp:                 bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
CUDA/Integrators/EXPRB42.hpp:                 GPU_handle& cublas_handle   //? CuBLAS handle
CUDA/Integrators/EXPRB42.hpp:        //! are device vectors if GPU support is activated.
CUDA/Integrators/EXPRB42.hpp:        axpby(dt, f_u, f_u, N, GPU);
CUDA/Integrators/EXPRB42.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);
CUDA/Integrators/EXPRB42.hpp:        axpby(1.0, u, 3./4., &u_flux[0], a, N, GPU);
CUDA/Integrators/EXPRB42.hpp:        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB42.hpp:        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB42.hpp:        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
CUDA/Integrators/EXPRB42.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);
CUDA/Integrators/EXPRB42.hpp:        axpby(1.0, u, 1.0, &u_flux[N], u_exprb2, N, GPU);
CUDA/Integrators/EXPRB42.hpp:        axpby(1.0, u_exprb2, 32./9., u_nl_3, u_exprb4, N, GPU);
CUDA/Integrators/EXPRB42.hpp:        axpby(32./9., u_nl_3, error_vector, N, GPU);
CUDA/Integrators/EXPRB42.hpp:        error = l2norm(error_vector, N, GPU, cublas_handle)/sqrt(N);
CUDA/Integrators/EXPRB32.hpp:                 bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
CUDA/Integrators/EXPRB32.hpp:                 GPU_handle& cublas_handle   //? CuBLAS handle
CUDA/Integrators/EXPRB32.hpp:        //! are device vectors if GPU support is activated.
CUDA/Integrators/EXPRB32.hpp:        axpby(dt, f_u, f_u, N, GPU);
CUDA/Integrators/EXPRB32.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);
CUDA/Integrators/EXPRB32.hpp:        axpby(1.0, u, 1.0, u_flux, u_exprb2, N, GPU);
CUDA/Integrators/EXPRB32.hpp:        Nonlinear_remainder(RHS, u, u,        NL_u, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB32.hpp:        Nonlinear_remainder(RHS, u, u_exprb2, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB32.hpp:        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
CUDA/Integrators/EXPRB32.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);
CUDA/Integrators/EXPRB32.hpp:        axpby(1.0, u_exprb2, 2.0, u_nl_3, u_exprb3, N, GPU);
CUDA/Integrators/EXPRB32.hpp:        axpby(2.0, u_nl_3, error_vector, N, GPU);
CUDA/Integrators/EXPRB32.hpp:        error = l2norm(error_vector, N, GPU, cublas_handle)/sqrt(N);
CUDA/Integrators/EXPRB43.hpp:                 bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
CUDA/Integrators/EXPRB43.hpp:                 GPU_handle& cublas_handle   //? CuBLAS handle
CUDA/Integrators/EXPRB43.hpp:        //! are device vectors if GPU support is activated.
CUDA/Integrators/EXPRB43.hpp:        axpby(dt, f_u, f_u, N, GPU);
CUDA/Integrators/EXPRB43.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);
CUDA/Integrators/EXPRB43.hpp:        axpby(1.0, u, 0.5, &u_flux[0], a, N, GPU);
CUDA/Integrators/EXPRB43.hpp:        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB43.hpp:        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB43.hpp:        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
CUDA/Integrators/EXPRB43.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);
CUDA/Integrators/EXPRB43.hpp:        axpby(1.0, u, 1.0, &u_flux[N], 1.0, &b_nl[0], b, N, GPU);
CUDA/Integrators/EXPRB43.hpp:        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB43.hpp:        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);
CUDA/Integrators/EXPRB43.hpp:        axpby(16.0, R_a, -2.0, R_b, R_3, N, GPU);
CUDA/Integrators/EXPRB43.hpp:        axpby(-48.0, R_a, 12.0, R_b, R_4, N, GPU);
CUDA/Integrators/EXPRB43.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_3, GPU, cublas_handle);
CUDA/Integrators/EXPRB43.hpp:                      phi_4, Leja_X, c, Gamma, rtol, atol, dt, iters_4, GPU, cublas_handle);
CUDA/Integrators/EXPRB43.hpp:        axpby(1.0, u, 1.0, &u_flux[N], 1.0, u_nl_3, &u_exprb3[0], N, GPU);
CUDA/Integrators/EXPRB43.hpp:        axpby(1.0, u_exprb3, 1.0, u_nl_4, u_exprb4, N, GPU);
CUDA/Integrators/EXPRB43.hpp:        error = l2norm(u_nl_4, N, GPU, cublas_handle)/sqrt(N);
CUDA/Integrators/EXPRB54s4.hpp:                   bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
CUDA/Integrators/EXPRB54s4.hpp:                   GPU_handle& cublas_handle   //? CuBLAS handle
CUDA/Integrators/EXPRB54s4.hpp:        //! are device vectors if GPU support is activated.
CUDA/Integrators/EXPRB54s4.hpp:        axpby(dt, f_u, f_u, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(1.0, u, 0.25, &u_flux[0], a_n, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:        Nonlinear_remainder(RHS, u, u,   NL_u, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        Nonlinear_remainder(RHS, u, a_n, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(1.0, u, 0.5, &u_flux[N], 4.0, b_nl, b_n, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:        Nonlinear_remainder(RHS, u, b_n, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_3, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(1.0, u, 9.0/10.0, &u_flux[2*N], 729.0/125.0, c_nl, c_n, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:        Nonlinear_remainder(RHS, u, c_n, NL_c, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(dt, NL_c, -dt, NL_u, R_c, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(64.0, R_a, -8.0, R_b, R_4a, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_4, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(-60.0, R_a, -285.0/8.0, R_b, 125.0/8.0, R_c, R_4b, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:                      phi_4, Leja_X, c, Gamma, rtol, atol, dt, iters_5, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(1.0, u, 1.0, &u_flux[3*N], 1.0, u_nl_4_3, 1.0, u_nl_4_4, u_exprb4, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(18.0, R_b, -250.0/81.0, R_c, R_5a, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_6, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(-60.0, R_b, 500.0/27.0, R_c, R_5b, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:                      phi_4, Leja_X, c, Gamma, rtol, atol, dt, iters_7, GPU, cublas_handle);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(1.0, u, 1.0, &u_flux[3*N], 1.0, u_nl_5_3, 1.0, u_nl_5_4, u_exprb5, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:        axpby(1.0, u_exprb5, -1.0, u_exprb4, error_vector, N, GPU);
CUDA/Integrators/EXPRB54s4.hpp:        error = l2norm(error_vector, N, GPU, cublas_handle)/sqrt(N);
CUDA/Integrators/EPIRK5P1.hpp:                  bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
CUDA/Integrators/EPIRK5P1.hpp:                  GPU_handle& cublas_handle   //? CuBLAS handle
CUDA/Integrators/EPIRK5P1.hpp:        //! are device vectors if GPU support is activated.
CUDA/Integrators/EPIRK5P1.hpp:        axpby(dt, f_u, f_u, N, GPU);
CUDA/Integrators/EPIRK5P1.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);
CUDA/Integrators/EPIRK5P1.hpp:        axpby(1.0, u, a11, &u_flux[0], a, N, GPU);
CUDA/Integrators/EPIRK5P1.hpp:        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK5P1.hpp:        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK5P1.hpp:        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
CUDA/Integrators/EPIRK5P1.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);
CUDA/Integrators/EPIRK5P1.hpp:        axpby(1.0, u, a21, &u_flux[N], a22, &u_nl_1[2*N], b, N, GPU);
CUDA/Integrators/EPIRK5P1.hpp:        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK5P1.hpp:        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);
CUDA/Integrators/EPIRK5P1.hpp:        axpby(-2.0, R_a, 1.0, R_b, R_3, N, GPU);
CUDA/Integrators/EPIRK5P1.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_3, GPU, cublas_handle);
CUDA/Integrators/EPIRK5P1.hpp:        axpby(1.0, u, b1, &u_flux[2*N], b2, &u_nl_1[0], b3, &u_nl_2[N], u_epirk4, N, GPU);
CUDA/Integrators/EPIRK5P1.hpp:        axpby(1.0, u, b1, &u_flux[2*N], b2, &u_nl_1[N], b3, &u_nl_2[0], u_epirk5, N, GPU);
CUDA/Integrators/EPIRK5P1.hpp:        axpby(1.0, u_epirk5, -1.0, u_epirk4, error_vector, N, GPU);
CUDA/Integrators/EPIRK5P1.hpp:        error = l2norm(error_vector, N, GPU, cublas_handle)/sqrt(N);
CUDA/Integrators/EPIRK4s3.hpp:                  bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
CUDA/Integrators/EPIRK4s3.hpp:                  GPU_handle& cublas_handle   //? CuBLAS handle
CUDA/Integrators/EPIRK4s3.hpp:        //! are device vectors if GPU support is activated.
CUDA/Integrators/EPIRK4s3.hpp:        axpby(dt, f_u, f_u, N, GPU);
CUDA/Integrators/EPIRK4s3.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3.hpp:        axpby(1.0, u, 1./8., &u_flux[0], a, N, GPU);
CUDA/Integrators/EPIRK4s3.hpp:        axpby(1.0, u, 1./9., &u_flux[N], b, N, GPU);
CUDA/Integrators/EPIRK4s3.hpp:        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3.hpp:        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3.hpp:        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3.hpp:        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
CUDA/Integrators/EPIRK4s3.hpp:        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);
CUDA/Integrators/EPIRK4s3.hpp:        axpby(-1024.0, R_a, 1458.0, R_b, R_3, N, GPU);
CUDA/Integrators/EPIRK4s3.hpp:        axpby(27648.0, R_a, -34992.0, R_b, R_4, N, GPU);
CUDA/Integrators/EPIRK4s3.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3.hpp:                      phi_4, Leja_X, c, Gamma, rtol, atol, dt, iters_3, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3.hpp:        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_3, u_epirk3, N, GPU);
CUDA/Integrators/EPIRK4s3.hpp:        axpby(1.0, u_epirk3, 1.0, u_nl_4, u_epirk4, N, GPU);
CUDA/Integrators/EPIRK4s3.hpp:        error = l2norm(u_nl_4, N, GPU, cublas_handle)/sqrt(N);
CUDA/Integrators/EXPRB53s3.hpp:                   bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
CUDA/Integrators/EXPRB53s3.hpp:                   GPU_handle& cublas_handle   //? CuBLAS handle
CUDA/Integrators/EXPRB53s3.hpp:        //! are device vectors if GPU support is activated.
CUDA/Integrators/EXPRB53s3.hpp:        axpby(dt, f_u, f_u, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);
CUDA/Integrators/EXPRB53s3.hpp:        axpby(1.0, u, 0.5, &u_flux[0], a, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB53s3.hpp:        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB53s3.hpp:        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);
CUDA/Integrators/EXPRB53s3.hpp:        axpby(1.0, u, 0.9, &u_flux[N], 27.0/25.0, &b_nl[0], 729.0/125.0, &b_nl[N], b, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EXPRB53s3.hpp:        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:        axpby(2.0, R_a, 150.0/81.0, R_b, R_3, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_3, GPU, cublas_handle);
CUDA/Integrators/EXPRB53s3.hpp:        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_4_3, u_exprb3, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:        axpby(18.0, R_a, -250.0/81.0, R_b, R_3, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:        axpby(-60.0, R_a, 500.0/27.0, R_b, R_4, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_4, GPU, cublas_handle);
CUDA/Integrators/EXPRB53s3.hpp:                      phi_4, Leja_X, c, Gamma, rtol, atol, dt, iters_5, GPU, cublas_handle);
CUDA/Integrators/EXPRB53s3.hpp:        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_5_3, 1.0, u_nl_5_4, u_exprb5, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:        axpby(1.0, u_exprb5, -1.0, u_exprb3, error_vector, N, GPU);
CUDA/Integrators/EXPRB53s3.hpp:        error = l2norm(error_vector, N, GPU, cublas_handle)/sqrt(N);
CUDA/Integrators/EPIRK4s3B.hpp:                   bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
CUDA/Integrators/EPIRK4s3B.hpp:                   GPU_handle& cublas_handle   //? CuBLAS handle
CUDA/Integrators/EPIRK4s3B.hpp:        //! are device vectors if GPU support is activated.
CUDA/Integrators/EPIRK4s3B.hpp:        axpby(dt, f_u, f_u, N, GPU);
CUDA/Integrators/EPIRK4s3B.hpp:                      phi_2, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3B.hpp:                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3B.hpp:        axpby(1.0, u, 2./3., &u_flux[0], a, N, GPU);
CUDA/Integrators/EPIRK4s3B.hpp:        axpby(1.0, u, 1.0, &u_flux[N], b, N, GPU);
CUDA/Integrators/EPIRK4s3B.hpp:        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3B.hpp:        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3B.hpp:        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3B.hpp:        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
CUDA/Integrators/EPIRK4s3B.hpp:        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);
CUDA/Integrators/EPIRK4s3B.hpp:        axpby(54.0, R_a, -16.0, R_b, R_3, N, GPU);
CUDA/Integrators/EPIRK4s3B.hpp:        axpby(-324.0, R_a, 144.0, R_b, R_4, N, GPU);
CUDA/Integrators/EPIRK4s3B.hpp:                        phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_3, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3B.hpp:                      phi_4, Leja_X, c, Gamma, rtol, atol, dt, iters_4, GPU, cublas_handle);
CUDA/Integrators/EPIRK4s3B.hpp:        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_3, 1.0, u_nl_4, u_epirk4, N, GPU);
CUDA/real_Leja_phi_nl.hpp:                          bool GPU,                           //? false (0) --> CPU; true (1) --> GPU
CUDA/real_Leja_phi_nl.hpp:                          GPU_handle& cublas_handle           //? CuBLAS handle
CUDA/real_Leja_phi_nl.hpp:        axpby(coeffs[0], interp_vector, polynomial, N, GPU);
CUDA/real_Leja_phi_nl.hpp:            axpby(1./Gamma, Jac_vec, (-c/Gamma - Leja_X[iters - 1]), interp_vector, interp_vector, N, GPU);
CUDA/real_Leja_phi_nl.hpp:            axpby(coeffs[iters], interp_vector, 1.0, polynomial, polynomial, N, GPU);
CUDA/real_Leja_phi_nl.hpp:            double poly_error = l2norm(interp_vector, N, GPU, cublas_handle)/sqrt(N);
CUDA/real_Leja_phi_nl.hpp:            double poly_norm = l2norm(polynomial, N, GPU, cublas_handle)/sqrt(N);
CUDA/Eigenvalues.hpp:#include "Kernels_CUDA_Cpp.hpp"
CUDA/Eigenvalues.hpp:                          bool GPU,                     //? false (0) --> CPU; true (1) --> GPU
CUDA/Eigenvalues.hpp:                          GPU_handle& cublas_handle     //? CuBLAS handle
CUDA/Eigenvalues.hpp:        eigen_ones(init_vector, N, GPU);
CUDA/Eigenvalues.hpp:            Jacobian_vector(RHS, u, init_vector, eigenvector, auxiliary_Jv, N, GPU, cublas_handle);
CUDA/Eigenvalues.hpp:            eigenvalue_ii = l2norm(eigenvector, N, GPU, cublas_handle)/sqrt(N);
CUDA/Eigenvalues.hpp:            axpby(1.0/eigenvalue_ii, eigenvector, init_vector, N, GPU);
CUDA/Eigenvalues.hpp:                #ifdef __CUDACC__
CUDA/Eigenvalues.hpp:                    cudaDeviceSynchronize();
CUDA/Eigenvalues.hpp:                    gpuErrchk(cudaPeekAtLastError());
CUDA/real_Leja_phi.hpp:                       bool GPU,                           //? false (0) --> CPU; true (1) --> GPU
CUDA/real_Leja_phi.hpp:                       GPU_handle& cublas_handle           //? CuBLAS handle
CUDA/real_Leja_phi.hpp:        copy(interp_vector, y, N, GPU);
CUDA/real_Leja_phi.hpp:            axpby(coeffs[ij][0], y, &polynomial[ij*N], N, GPU);
CUDA/real_Leja_phi.hpp:            Jacobian_vector(RHS, u, y, Jac_vec, auxiliary_Jv, N, GPU, cublas_handle);
CUDA/real_Leja_phi.hpp:            axpby(1./Gamma, Jac_vec, (-c/Gamma - Leja_X[iters - 1]), y, y, N, GPU);
CUDA/real_Leja_phi.hpp:                axpby(coeffs[ij][iters], y, 1.0, &polynomial[ij*N], &polynomial[ij*N], N, GPU);
CUDA/real_Leja_phi.hpp:            double poly_error = l2norm(y, N, GPU, cublas_handle)/sqrt(N);
CUDA/real_Leja_phi.hpp:            double poly_norm = l2norm(&polynomial[(num_interpolations - 1)*N], N, GPU, cublas_handle)/sqrt(N);
CUDA/Kernels_CUDA_Cpp.hpp://?     depending on whether GPU support is activated or not. 
CUDA/Kernels_CUDA_Cpp.hpp:    double l2norm(double *x, size_t N, bool GPU, GPU_handle& cublas_handle)
CUDA/Kernels_CUDA_Cpp.hpp:        if (GPU == true)
CUDA/Kernels_CUDA_Cpp.hpp:            #ifdef __CUDACC__
CUDA/Kernels_CUDA_Cpp.hpp:                //* CUDA
CUDA/Kernels_CUDA_Cpp.hpp:    void copy(double *x, double *y, size_t N, bool GPU)
CUDA/Kernels_CUDA_Cpp.hpp:        if (GPU == true)
CUDA/Kernels_CUDA_Cpp.hpp:            #ifdef __CUDACC__
CUDA/Kernels_CUDA_Cpp.hpp:            //* CUDA
CUDA/Kernels_CUDA_Cpp.hpp:            copy_CUDA<<<(N/128) + 1, 128>>>(x, y, N);
CUDA/Kernels_CUDA_Cpp.hpp:    void ones(double *x, size_t N, bool GPU)
CUDA/Kernels_CUDA_Cpp.hpp:        if (GPU == true)
CUDA/Kernels_CUDA_Cpp.hpp:            #ifdef __CUDACC__
CUDA/Kernels_CUDA_Cpp.hpp:            //* CUDA
CUDA/Kernels_CUDA_Cpp.hpp:            ones_CUDA<<<(N/128) + 1, 128>>>(x, N);
CUDA/Kernels_CUDA_Cpp.hpp:    void eigen_ones(double *x, size_t N, bool GPU)
CUDA/Kernels_CUDA_Cpp.hpp:        if (GPU == true)
CUDA/Kernels_CUDA_Cpp.hpp:            #ifdef __CUDACC__
CUDA/Kernels_CUDA_Cpp.hpp:            //* CUDA
CUDA/Kernels_CUDA_Cpp.hpp:            eigen_ones_CUDA<<<(N/128) + 1, 128>>>(x, N);
CUDA/Kernels_CUDA_Cpp.hpp:                         double *y, size_t N, bool GPU)
CUDA/Kernels_CUDA_Cpp.hpp:        if (GPU == true)
CUDA/Kernels_CUDA_Cpp.hpp:            #ifdef __CUDACC__
CUDA/Kernels_CUDA_Cpp.hpp:            //* CUDA
CUDA/Kernels_CUDA_Cpp.hpp:            axpby_CUDA<<<(N/128) + 1, 128>>>(a, x, y, N);
CUDA/Kernels_CUDA_Cpp.hpp:                         double *z, size_t N, bool GPU)
CUDA/Kernels_CUDA_Cpp.hpp:        if (GPU == true)
CUDA/Kernels_CUDA_Cpp.hpp:            #ifdef __CUDACC__
CUDA/Kernels_CUDA_Cpp.hpp:            //* CUDA
CUDA/Kernels_CUDA_Cpp.hpp:            axpby_CUDA<<<(N/128) + 1, 128>>>(a, x, b, y, z, N);
CUDA/Kernels_CUDA_Cpp.hpp:                         double *w, size_t N, bool GPU)
CUDA/Kernels_CUDA_Cpp.hpp:        if (GPU == true)
CUDA/Kernels_CUDA_Cpp.hpp:            #ifdef __CUDACC__
CUDA/Kernels_CUDA_Cpp.hpp:            //* CUDA
CUDA/Kernels_CUDA_Cpp.hpp:            axpby_CUDA<<<(N/128) + 1, 128>>>(a, x, b, y, c, z, w, N);
CUDA/Kernels_CUDA_Cpp.hpp:                         double *v, size_t N, bool GPU)
CUDA/Kernels_CUDA_Cpp.hpp:        if (GPU == true)
CUDA/Kernels_CUDA_Cpp.hpp:            #ifdef __CUDACC__
CUDA/Kernels_CUDA_Cpp.hpp:            //* CUDA
CUDA/Kernels_CUDA_Cpp.hpp:            axpby_CUDA<<<(N/128) + 1, 128>>>(a, x, b, y, c, z, d, w, v, N);
CUDA/Jacobian_vector.hpp:#include "Kernels_CUDA_Cpp.hpp"
CUDA/Jacobian_vector.hpp:                         bool GPU,                      //? false (0) --> CPU; true (1) --> GPU
CUDA/Jacobian_vector.hpp:                         GPU_handle& cublas_handle      //? CuBLAS handle
CUDA/Jacobian_vector.hpp:        double rhs_norm = l2norm(f_u, N, GPU, cublas_handle)/sqrt(N);
CUDA/Jacobian_vector.hpp:        axpby(1.0, u, epsilon, y, u_eps, N, GPU); 
CUDA/Jacobian_vector.hpp:        axpby(1.0, u, -epsilon, y, u_eps, N, GPU); 
CUDA/Jacobian_vector.hpp:        axpby(1.0/(2.0*epsilon), rhs_u_eps_1, -1.0/(2.0*epsilon), rhs_u_eps_2, Jac_vec, N, GPU);
CUDA/Jacobian_vector.hpp:                             bool GPU,                      //? false (0) --> CPU; true (1) --> GPU
CUDA/Jacobian_vector.hpp:                             GPU_handle& cublas_handle      //? CuBLAS handle
CUDA/Jacobian_vector.hpp:        Jacobian_vector(RHS, u, y, Linear_y, Jv, N, GPU, cublas_handle);
CUDA/Jacobian_vector.hpp:        axpby(1.0, f_y, -1.0, Linear_y, Nonlinear_y, N, GPU);
CUDA/Kernels.hpp:#ifdef __CUDACC__
CUDA/Kernels.hpp:    #include <cuda_runtime.h>
CUDA/Kernels.hpp:    #include <cuda.h>
CUDA/Kernels.hpp:struct GPU_handle
CUDA/Kernels.hpp:    #ifdef __CUDACC__
CUDA/Kernels.hpp:    GPU_handle()
CUDA/Kernels.hpp:        #ifdef __CUDACC__
CUDA/Kernels.hpp:    ~GPU_handle()
CUDA/Kernels.hpp:        #ifdef __CUDACC__
CUDA/Kernels.hpp:    #ifdef __CUDACC__
CUDA/Kernels.hpp:    __global__ void copy_CUDA(double *x, double *y, size_t N)                    
CUDA/Kernels.hpp:    __global__ void ones_CUDA(double *x, size_t N)                    
CUDA/Kernels.hpp:    __global__ void eigen_ones_CUDA(double *x, size_t N)                    
CUDA/Kernels.hpp:    __global__ void axpby_CUDA(double a, double *x, 
CUDA/Kernels.hpp:    __global__ void axpby_CUDA(double a, double *x, 
CUDA/Kernels.hpp:    __global__ void axpby_CUDA(double a, double *x, 
CUDA/Kernels.hpp:    __global__ void axpby_CUDA(double a, double *x,
README.md:![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
README.md:- For CUDA:
README.md:  - NVIDIA GPU
README.md:  - CUDA 11.2 (or later)
README.md:Deka, Moriggl, and Einkemmer (2023), *LeXInt: GPU-accelerated Exponential Integrators package* <br />
README.md:We will MPI-parallelise the CUDA/C++ code.
README.md:Alexander Moriggl contributed to the development of the CUDA version.
.gitignore:CUDA/.vscode/
.gitignore:CUDA/Test/build/

```
