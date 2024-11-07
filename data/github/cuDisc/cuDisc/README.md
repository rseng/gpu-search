# https://github.com/cuDisc/cuDisc

```console
tests/codes/test_pinte_graindist.cpp:#include "cuda_array.h"
tests/codes/test_pinte_graindist.cpp:    CudaArray<double> Re = make_CudaArray<double>(n_inner+Nouter+1) ;
tests/codes/test_pinte_graindist.cpp:    CudaArray<double> phi = make_CudaArray<double>(nPhi + 2*Nghost + 1) ;
tests/codes/test_adv_diff.cpp:#include "cuda_array.h"
tests/codes/test_coagdustpy.cpp:#include "cuda_array.h"
tests/codes/test_coagdustpy.cpp:void set_up_gas(Grid& g, Field<Prims1D>& wg, CudaArray<double>& Sig_g, Field<double>& T, Field<double>& cs) {
tests/codes/test_coagdustpy.cpp:    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost); // Gas surface density
tests/codes/test_pinte_graindist_mono.cpp:#include "cuda_array.h"
tests/codes/test_pinte_graindist_mono.cpp:    CudaArray<double> Re = make_CudaArray<double>(n_inner+Nouter+1) ;
tests/codes/test_pinte_graindist_mono.cpp:    CudaArray<double> phi = make_CudaArray<double>(nPhi + 2*Nghost + 1) ;
README.md:# cuDisc: a GPU accelerated protoplanetary disc model
README.md:You will need a NVIDIA GPU to use cuDisc.
README.md:    CUDA_HOME = /usr/local/cuda-12.0
README.md:in the makefile must be set to the correct location for your machine's CUDA installation.
makefile:CUDA_HOME = /usr/local/cuda-12.0
makefile:CUDA = nvcc 
makefile:CUDAFLAGS = -O3 -g --std=c++17 -Wno-deprecated-gpu-targets $(ARCH)
makefile:INCLUDE = -I./$(HEADER_DIR) -I$(CUDA_HOME)/include
makefile:LIB = -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcusparse
makefile:HEADERS := grid.h field.h cuda_array.h reductions.h utils.h matrix_types.h scan.h \
makefile:	$(CUDA) $(CUDAFLAGS) $(INCLUDE) -c $< -o $@
makefile:	$(CUDA) $(CUDAFLAGS) $(INCLUDE) $< -o $@ $(LIBRARY) $(LIB)
unit_tests/unit_coag_vertint.cpp:#include "cuda_array.h"
unit_tests/unit_temp.cpp:#include "cuda_array.h"
unit_tests/unit_sources.cpp:#include "cuda_array.h"
unit_tests/unit_sources.cpp:void set_up_gas(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& nu, Field<double>& T, Field<double>& cs, Field<double>& cs2, double alpha, Star& star) {
unit_tests/unit_sources.cpp:void compute_nu(const Grid &g, CudaArray<double> &nu, Field<double> &cs2, double Mstar, double alpha) {
unit_tests/unit_sources.cpp:void compute_nu(const Grid &g, CudaArray<double> &nu, double nu0, double Mstar, double alpha) {
unit_tests/unit_sources.cpp:    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost); // Gas surface density
unit_tests/unit_sources.cpp:    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost); // Kinematic viscosity
unit_tests/unit_coag.cpp:#include "cuda_array.h"
unit_tests/unit_adv_diff1D.cpp:            CudaArray<double>& nu, Field<double>& alpha2D, Field3D<double>& D, double alpha, Star& star) {
unit_tests/unit_adv_diff1D.cpp:    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost);
codes/1Ddisc.cpp:#include "cuda_array.h"
codes/1Ddisc.cpp:void setup_gas(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& u_gas, CudaArray<double>& nu, double alpha, Star& star) {
codes/1Ddisc.cpp:void setup_dust(Grid& g, CudaArray<double>& Sig_d, CudaArray<double>& Sig_g) {
codes/1Ddisc.cpp:void Sigdot_w_JO(Grid& g, CudaArray<double>& Sigdot_w, double logLx, double Mstar) {
codes/1Ddisc.cpp:    CudaArray<double> Signorm = make_CudaArray<double>(g.NR+2*g.Nghost);
codes/1Ddisc.cpp:void find_Rcav(Grid& g, CudaArray<double>& Sig_g, double& Rcav) {
codes/1Ddisc.cpp:    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost);
codes/1Ddisc.cpp:    CudaArray<double> Sigdot_w = make_CudaArray<double>(g.NR+2*g.Nghost);
codes/1Ddisc.cpp:    CudaArray<double> u_g = make_CudaArray<double>(g.NR+2*g.Nghost);
codes/1Ddisc.cpp:    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost);
codes/1Ddisc.cpp:    CudaArray<double> Sig_d = make_CudaArray<double>(g.NR+2*g.Nghost);
codes/1Ddisc.cpp:    CudaArray<double> ubar = make_CudaArray<double>(g.NR+2*g.Nghost);
codes/makefile:# Change lines 4 and 5 to the correct paths to both your cuda installation and cuDisc location.
codes/makefile:CUDA_HOME = /path/to/cuda-12.0
codes/makefile:CUDA = nvcc 
codes/makefile:CUDAFLAGS = -O3 -g --std=c++17 -Wno-deprecated-gpu-targets $(ARCH)
codes/makefile:INCLUDE = -I$(CUDISC_HOME)/headers -I$(CUDA_HOME)/include
codes/makefile:	-L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcusparse
codes/makefile:all : cuda_file cpp_file
codes/makefile:cuda_file: cuda_file.cu makefile				#replace "cuda_file" with name of the cuda file you wish to compile
codes/makefile:	$(CUDA) $(CUDAFLAGS) $(INCLUDE) $< -o $@  $(LIB)
codes/makefile:	rm -rf cuda_file cpp_file
codes/1Ddisc_multgrain.cpp:            Field<double>& cs, CudaArray<double>& nu, Field<double>& alpha2D, Field3D<double>& D, double alpha, Star& star) {
codes/1Ddisc_multgrain.cpp:    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost);
codes/steadyTD.cpp:#include "cuda_array.h"
codes/steadyTD.cpp:void Sigdot_w_PicPD(Grid& g, CudaArray<double>& Sigdot_w, double logLx) {
codes/steadyTD.cpp:void set_up_gas(Grid& g, Field<Prims>& wg, CudaArray<double>& Sig_g, CudaArray<double>& nu, Field<double>& T, Field<double>& cs, Field<double>& cs2, double alpha, Star& star) {
codes/steadyTD.cpp:void set_up_dust(Grid& g, Field3D<Prims>& qd, Field<Prims>& wg, CudaArray<double>& Sig_g, Field3D<double>& D, SizeGrid& sizes, double alpha, Field<double>& cs, double M_gas, double floor) {
codes/steadyTD.cpp:void compute_nu(const Grid &g, CudaArray<double> &nu, Field<double> &cs2, double Mstar, double alpha) {
codes/steadyTD.cpp:void compute_nu(const Grid &g, CudaArray<double> &nu, double nu0, double Mstar, double alpha) {
codes/steadyTD.cpp:void compute_D(const Grid &g, Field3D<double> &D, Field<Prims> &wg, CudaArray<double> &nu, double Sc) {
codes/steadyTD.cpp:void find_Rcav(Grid& g, CudaArray<double>& Sig_g, double& Rcav) {
codes/steadyTD.cpp:    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost); // Gas surface density
codes/steadyTD.cpp:    CudaArray<double> Sigdot_wind = make_CudaArray<double>(g.NR+2*g.Nghost); // Gas surface density
codes/steadyTD.cpp:    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost); // Kinematic viscosity
codes/isoPD.cpp:#include "cuda_array.h"
codes/isoPD.cpp:void set_up_gas(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& nu, Field<double>& T, Field<double>& cs, Field<double>& cs2, double alpha, Star& star) {
codes/isoPD.cpp:void compute_nu(const Grid &g, CudaArray<double> &nu, Field<double> &cs2, double Mstar, double alpha) {
codes/isoPD.cpp:void compute_nu(const Grid &g, CudaArray<double> &nu, double nu0, double Mstar, double alpha) {
codes/isoPD.cpp:    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost); // Gas surface density
codes/isoPD.cpp:    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost); // Kinematic viscosity
.gitignore:*.gpu
src/ILU_precond.cpp:#include <cuda_runtime.h>
src/ILU_precond.cpp:   CudaArray<char> buffer = make_CudaArray<char>(buffer_size1) ;
src/ILU_precond.cpp:            &one, LUfac.matL, rhs, tmp, CUDA_R_64F,
src/ILU_precond.cpp:    bufferL = make_CudaArray<char>(buffersize) ;
src/ILU_precond.cpp:            &one, LUfac.matL, rhs, tmp, CUDA_R_64F,
src/ILU_precond.cpp:            &one, LUfac.matU, tmp, result, CUDA_R_64F,
src/ILU_precond.cpp:    bufferU = make_CudaArray<char>(buffersize) ;
src/ILU_precond.cpp:            &one, LUfac.matU, tmp, result, CUDA_R_64F,
src/ILU_precond.cpp:            &one, LUfac.matL, rhs, tmp, CUDA_R_64F,
src/ILU_precond.cpp:            &one, LUfac.matU, tmp, result, CUDA_R_64F,
src/ILU_precond.cpp:    CudaArray<char> buffer1 ;
src/ILU_precond.cpp:    CudaArray<char> buffer2 ;
src/ILU_precond.cpp:                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
src/ILU_precond.cpp:            buffer1 = make_CudaArray<char>(bufferSize1) ;
src/ILU_precond.cpp:                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
src/ILU_precond.cpp:                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
src/ILU_precond.cpp:            buffer2 = make_CudaArray<char>(bufferSize2) ;
src/ILU_precond.cpp:                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
src/ILU_precond.cpp:                    CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
src/super_stepping.cu:#include <cuda_runtime.h>
src/super_stepping.cu:    check_CUDA_errors("_super_stepping_update_solution") ;
src/sparse_utils.cu:#include <cuda_runtime.h>
src/sparse_utils.cu:    check_CUDA_errors("_create_identity_device") ;
src/sparse_utils.cu:    cudaMemcpy(m_copy.csr_offset.get(), m.csr_offset.get(), (m.rows+1)*sizeof(int), 
src/sparse_utils.cu:               cudaMemcpyDeviceToDevice) ;
src/sparse_utils.cu:    cudaMemcpy(m_copy.col_index.get(), m.col_index.get(), m.non_zeros*sizeof(int),
src/sparse_utils.cu:               cudaMemcpyDeviceToDevice) ;
src/sparse_utils.cu:    cudaMemcpy(m_copy.data.get(), m.data.get(), m.non_zeros*sizeof(double), 
src/sparse_utils.cu:               cudaMemcpyDeviceToDevice) ;
src/bins.cu:#include <cuda_runtime.h>
src/bins.cu:CudaArray<double> WavelengthBinner::bin_data(const CudaArray<double>& input,
src/bins.cu:    CudaArray<double> result = make_CudaArray<double>(num_bands) ;
src/bins.cu:    check_CUDA_errors("_bin_field") ;
src/bins.cu:CudaArray<double> WavelengthBinner::bin_planck_data(const CudaArray<double>& input,
src/bins.cu:    CudaArray<double> result = make_CudaArray<double>(num_bands) ;
src/bins.cu:    check_CUDA_errors("_bin_planck") ;
src/bins.cu:    check_CUDA_errors("_bin_planck") ;
src/block_jacobi.cu:#include <cuda_runtime.h>
src/block_jacobi.cu:    check_CUDA_errors("_block_jacobi_solve") ;
src/grid.cu:    _Rc = make_CudaArray<double>(NR + 2*Nghost) ;
src/grid.cu:    _Re = make_CudaArray<double>(NR + 2*Nghost + 1) ;
src/grid.cu:    _Ar = make_CudaArray<double>(NR + 2*Nghost + 1) ;
src/grid.cu:    _Az = make_CudaArray<double>(NR + 2*Nghost) ;
src/grid.cu:    _V  = make_CudaArray<double>(NR + 2*Nghost) ;
src/grid.cu:    _sin_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;
src/grid.cu:    _cos_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;
src/grid.cu:    _tan_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;
src/grid.cu:    _sin_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;
src/grid.cu:    _cos_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;
src/grid.cu:    _tan_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;
src/grid.cu:           CudaArray<double> R, CudaArray<double> phi, int index_) {
src/grid.cu:    _Rc = make_CudaArray<double>(NR + 2*Nghost) ;
src/grid.cu:    _Ar = make_CudaArray<double>(NR + 2*Nghost + 1) ;
src/grid.cu:    _Az = make_CudaArray<double>(NR + 2*Nghost) ;
src/grid.cu:    _V  = make_CudaArray<double>(NR + 2*Nghost) ;
src/grid.cu:    _sin_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;
src/grid.cu:    _cos_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;
src/grid.cu:    _tan_theta_c = make_CudaArray<double>(Nphi + 2*Nghost) ;
src/grid.cu:    _sin_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;
src/grid.cu:    _cos_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;
src/grid.cu:    _tan_theta_e = make_CudaArray<double>(Nphi + 2*Nghost + 1) ;
src/grid.cu:    _Rc = make_CudaArray<double>(NR + 2*Nghost);
src/grid.cu:    _Re = make_CudaArray<double>(NR + 2*Nghost + 1);
src/grid.cu:    _Zc = make_CudaArray<double>(NZ + 2*Nghost);
src/grid.cu:    _Ze = make_CudaArray<double>(NZ + 2*Nghost + 1);
src/grid.cu:    CudaArray<double> Re_sub = make_CudaArray<double>(out_idx[sg_idx]-in_idx[sg_idx]+1);
src/grid.cu:    CudaArray<double> phie_sub = make_CudaArray<double>(g.Nphi+2*g.Nghost+1);
src/grid.cu:    CudaArray<double> Re_sub = make_CudaArray<double>(out_idx[sg_idx]-in_idx[sg_idx]+1);
src/grid.cu:    CudaArray<double> phie_sub = make_CudaArray<double>(1+2*g.Nghost+1);
src/grid.cu:    check_CUDA_errors("_copy_from_subgrid");
src/grid.cu:    check_CUDA_errors("_copy_from_subgrid");
src/grid.cu:    check_CUDA_errors("_copy_from_subgrid");
src/grid.cu:    check_CUDA_errors("_copy_from_subgrid");
src/grid.cu:void GridManager::copy_to_subgrid(Grid& g_sub, const CudaArray<T>& F_main, CudaArray<T>& F_sub) { 
src/grid.cu:void GridManager::copy_from_subgrid(Grid& g_sub, CudaArray<T>& F_main, const CudaArray<T>& F_sub) { 
src/grid.cu:template void GridManager::copy_to_subgrid<double>(Grid& g_sub, const CudaArray<double>& F_main, CudaArray<double>& F_sub);
src/grid.cu:template void GridManager::copy_from_subgrid<double>(Grid& g_sub, CudaArray<double>& F_main, const CudaArray<double>& F_sub);
src/grid.cu:template void GridManager::copy_to_subgrid<int>(Grid& g_sub, const CudaArray<int>& F_main, CudaArray<int>& F_sub);
src/grid.cu:template void GridManager::copy_from_subgrid<int>(Grid& g_sub, CudaArray<int>& F_main, const CudaArray<int>& F_sub);
src/rpsolver.cu:#include <cuda_runtime.h>
src/rpsolver.cu:// CUDA function
src/rpsolver.cu:void VL_Diff_Advect::operator() (Grid& g, Field3D<Quants>& q, const Field<Quants>& w_gas, const Field3D<double>& D, double dt, CudaArray<double>& h_w, double R_cav, CudaArray<int>& coagbool) {
src/rpsolver.cu:    check_CUDA_errors("_compute_CFL_diff") ;
src/rpsolver.cu:    check_CUDA_errors("_compute_CFL_diff") ;
src/gas1d.cu:#include "cuda_runtime.h"
src/gas1d.cu:#include "cuda_array.h"
src/gas1d.cu:void update_gas_sigma(Grid& g, CudaArray<double>& Sig_g, double dt, const CudaArray<double>& nu, int bound, double floor) {
src/gas1d.cu:    CudaArray<double> RF = make_CudaArray<double>(g.NR+2*g.Nghost);
src/gas1d.cu:void update_gas_sigma(Grid& g, CudaArray<double>& Sig_g, double dt, const Field<double>& nu, int bound, double floor) {
src/gas1d.cu:    CudaArray<double> RF = make_CudaArray<double>(g.NR+2*g.Nghost);
src/gas1d.cu:    CudaArray<double> nu1D = make_CudaArray<double>(g.NR+2*g.Nghost);
src/gas1d.cu:void update_gas_vel(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& u_gas, double alpha, Star& star) {
src/gas1d.cu:    CudaArray<double> f = make_CudaArray<double>(g.NR+2*g.Nghost+1);
src/gas1d.cu:void update_gas_sources(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& Sigdot, double dt, int bound, double floor) {
src/gas1d.cu:double calc_dt(Grid& g, const CudaArray<double>& nu) {
src/gas1d.cu:    CudaArray<double> dt_R = make_CudaArray<double>(g.NR+2*g.Nghost-1);
src/gas1d.cu:    check_CUDA_errors("_calc_dt_diff");
src/gas1d.cu:    CudaArray<double> dt_R = make_CudaArray<double>(g.NR+2*g.Nghost-1);
src/gas1d.cu:    check_CUDA_errors("_calc_dt_diff");
src/gas1d.cu:void calc_gas_velocities(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, double alpha, Star& star, int bound, double floor, double cav) {
src/gas1d.cu:void calc_gas_velocities_full(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, double alpha, Star& star, int bound, double floor, double cav) {
src/gas1d.cu:void calc_gas_velocities_wind(Grid& g, Field<Prims>& wg, CudaArray<double>& Sig_g, Field<double>& cs2, CudaArray<double>& nu, CudaArray<double>& Sig_dot_w, 
src/gas1d.cu:void _set_ubar_bounds(Grid& g, CudaArray<double>& ubar, int buff) {
src/gas1d.cu:void calculate_ubar(Grid& g, CudaArray<double>& sig, CudaArray<double>& sig_g, 
src/gas1d.cu:                    CudaArray<double>& ubar, CudaArray<double>& u_gas,
src/gas1d.cu:    CudaArray<double> P = make_CudaArray<double>(g.NR+2*g.Nghost+1);
src/gas1d.cu:    cudaDeviceSynchronize();
src/gas1d.cu:void update_dust_sigma(Grid& g, CudaArray<double>& sig, CudaArray<double>& sig_g, 
src/gas1d.cu:                    CudaArray<double>& ubar, CudaArray<double>& D, double dt, int bound) {
src/gas1d.cu:    CudaArray<double> sig_mid = make_CudaArray<double>(g.NR+2*g.Nghost);
src/gas1d.cu:    CudaArray<double> flux = make_CudaArray<double>(g.NR+2*g.Nghost);
src/gas1d.cu:double compute_CFL(Grid& g, CudaArray<double>& ubar, CudaArray<double>& D,
src/gas1d.cu:void calc_wind_surface(Grid& g, const Field<Prims>& wg, CudaArray<double>& h_w, double col) {
src/gas1d.cu:    cudaDeviceSynchronize();
src/gas1d.cu:void update_gas_sigma(Grid& g, Field<Prims1D>& W_g, double dt, const CudaArray<double>& nu, int bound, double floor) {
src/gas1d.cu:    CudaArray<double> RF = make_CudaArray<double>(g.NR+2*g.Nghost);
src/gas1d.cu:    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost);
src/gas1d.cu:    cudaDeviceSynchronize();
src/gas1d.cu:void calc_v_gas(Grid& g, Field<Prims1D>& W_g, const Field<double>& cs, CudaArray<double>& nu, double GMstar, double gasfloor) {
src/gas1d.cu:    check_CUDA_errors("_calc_v_gas");
src/gas1d.cu:    cudaDeviceSynchronize();
src/diffusion.cu:#include <cuda_runtime.h>
src/diffusion.cu:    check_CUDA_errors("compute_Drho_gas") ;           
src/diffusion.cu:    check_CUDA_errors("compute_diffusive_flux_update") ;    
src/diffusion.cu:    check_CUDA_errors("compute_Drho_gas") ;           
src/diffusion.cu:    check_CUDA_errors("compute_diffusive_flux_update") ;    
src/diffusion.cu:    check_CUDA_errors("compute_Drho_gas") ;           
src/diffusion.cu:        check_CUDA_errors("compute_diffusion_rate") ;    
src/diffusion.cu:    check_CUDA_errors("compute_Drho_gas") ;   
src/diffusion.cu:        check_CUDA_errors("compute_diffusion_rate") ;    
src/diffusion.cu:    check_CUDA_errors("_compute_CFL_limit_diffusion") ;
src/dustdynamics.cu:#include <cuda_runtime.h>
src/dustdynamics.cu:    check_CUDA_errors("_set_boundaries") ;
src/dustdynamics.cu:    check_CUDA_errors("_calc_conserved") ;
src/dustdynamics.cu:        check_CUDA_errors("_calc_donor_flux") ;
src/dustdynamics.cu:        check_CUDA_errors("_calc_donor_flux") ;
src/dustdynamics.cu:    check_CUDA_errors("_set_boundary_flux") ;
src/dustdynamics.cu:    check_CUDA_errors("_update_quants") ;
src/dustdynamics.cu:    check_CUDA_errors("_calc_prim") ; 
src/dustdynamics.cu:    check_CUDA_errors("_floor_prim") ;
src/dustdynamics.cu:    check_CUDA_errors("_set_boundaries") ;
src/dustdynamics.cu:        check_CUDA_errors("_calc_diff_flux_vl") ;
src/dustdynamics.cu:        check_CUDA_errors("_calc_diff_flux_vl") ;
src/dustdynamics.cu:    check_CUDA_errors("_set_boundary_flux") ;
src/dustdynamics.cu:    check_CUDA_errors("_update_quants") ;
src/dustdynamics.cu:    check_CUDA_errors("_calc_prim") ; 
src/dustdynamics.cu:    check_CUDA_errors("_floor_prim") ;
src/dustdynamics.cu:    check_CUDA_errors("_compute_CFL_diff") ;
src/dustdynamics.cu:    check_CUDA_errors("_compute_CFL_diff") ;
src/dustdynamics.cu:void DustDynamics::floor_above(Grid& g, Field3D<Prims>& w_dust, Field<Prims>& w_gas, CudaArray<double>& h) {
src/advection.cu:#include <cuda_runtime.h>
src/advection.cu:    check_CUDA_errors("_compute_fluxes_donor") ;
src/advection.cu:    check_CUDA_errors("_update_conserved") ;
src/advection.cu:    check_CUDA_errors("_compute_fluxes_vanleer") ;
src/advection.cu:    check_CUDA_errors("_update_conserved") ;
src/advection.cu:    check_CUDA_errors("_compute_fluxes_donor") ;
src/advection.cu:    check_CUDA_errors("_update_conserved") ;
src/advection.cu:    check_CUDA_errors("_compute_fluxes_vanleer") ;
src/advection.cu:    check_CUDA_errors("_update_conserved") ;
src/advection.cu:    check_CUDA_errors("_compute_CFL_limit_VL") ;
src/advection.cu:    check_CUDA_errors("_compute_CFL_limit_VL") ;
src/jacobi.cu:#include <cuda_runtime.h>
src/jacobi.cu:    check_CUDA_errors("get_diags") ;           
src/jacobi.cu:    check_CUDA_errors("scale_system") ;       
src/jacobi.cu:    check_CUDA_errors("scale_x") ;       
src/jacobi.cu:    check_CUDA_errors("remove_scale_from_x") ;       
src/zero_bounds.cu:#include "cuda_array.h"
src/zero_bounds.cu:    //check_CUDA_errors("zero_boundaries") ;
src/zero_bounds.cu:    check_CUDA_errors("zero_boundaries") ;
src/zero_bounds.cu:    //check_CUDA_errors("zero_boundaries") ;
src/zero_bounds.cu:    check_CUDA_errors("zero_boundaries") ;
src/zero_bounds.cu:    check_CUDA_errors("zero_boundaries") ;
src/zero_bounds.cu:    check_CUDA_errors("zero_boundaries") ;
src/zero_bounds.cu:    check_CUDA_errors("set_all_device") ;
src/zero_bounds.cu:    check_CUDA_errors("set_all_device") ;
src/coagulation_integrate.cu:    check_CUDA_errors("_compute_ytot") ;
src/coagulation_integrate.cu:        check_CUDA_errors("_compute_error_norm") ;
src/coagulation_integrate.cu:    check_CUDA_errors("_compute_ytot") ;
src/coagulation_integrate.cu:        check_CUDA_errors("_compute_error_norm") ;
src/coagulation_integrate.cu:    cudaDeviceSynchronize();
src/coagulation_integrate.cu:    cudaDeviceSynchronize();
src/coagulation_integrate.cu:    check_CUDA_errors("_Rk2_update1") ;
src/coagulation_integrate.cu:    check_CUDA_errors("_Rk2_update2") ;
src/pcg_solver.cpp:#include <cuda_runtime.h>
src/pcg_solver.cpp: * Simple Conjugate-Gradient solver from Nvidia:  
src/pcg_solver.cpp:        https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
src/pcg_solver.cpp:        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size) ;
src/pcg_solver.cpp:    CudaArray<char> buffer = make_CudaArray<char>(buffer_size) ;
src/pcg_solver.cpp:                  CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;
src/pcg_solver.cpp:                    CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;
src/pcg_solver.cpp:        https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
src/pcg_solver.cpp:        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size) ;
src/pcg_solver.cpp:    CudaArray<char> buffer = make_CudaArray<char>(buffer_size) ;
src/pcg_solver.cpp:                  CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;
src/pcg_solver.cpp:                          CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;
src/pcg_solver.cpp:                    CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;
src/pcg_solver.cpp:        cudaDeviceSynchronize();
src/pcg_solver.cpp:                    CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;
src/scan.cu:#include "cuda_array.h"
src/scan.cu:    // Setup cuda blocks / threads and shared memory size
src/scan.cu:    check_CUDA_errors("scan_R_OP") ;
src/scan.cu:    // Setup cuda blocks / threads and shared memory size
src/scan.cu:    check_CUDA_errors("scan_Z_OP") ;
src/dustdynamics1D.cu:#include "cuda_runtime.h"
src/dustdynamics1D.cu:    check_CUDA_errors("_calc_dust_vel");
src/dustdynamics1D.cu:    check_CUDA_errors("_calc_dust_vel");
src/dustdynamics1D.cu:    check_CUDA_errors("_set_bounds_d");
src/dustdynamics1D.cu:    check_CUDA_errors("_set_bounds_d");
src/dustdynamics1D.cu:    check_CUDA_errors("_set_boundary_flux");
src/dustdynamics1D.cu:    check_CUDA_errors("_update_mid_Sig");
src/dustdynamics1D.cu:    cudaDeviceSynchronize();
src/dustdynamics1D.cu:    check_CUDA_errors("_set_bounds_d");
src/dustdynamics1D.cu:    check_CUDA_errors("_calc_diff_flux_vl");
src/dustdynamics1D.cu:    check_CUDA_errors("_set_boundary_flux");
src/dustdynamics1D.cu:    check_CUDA_errors("_update_Sig");
src/dustdynamics1D.cu:    cudaDeviceSynchronize();
src/dustdynamics1D.cu:    check_CUDA_errors("_set_bounds_d");
src/dustdynamics1D.cu:    check_CUDA_errors("_set_bounds_d");
src/dustdynamics1D.cu:    check_CUDA_errors("_set_boundary_flux");
src/dustdynamics1D.cu:    check_CUDA_errors("_update_mid_Sig");
src/dustdynamics1D.cu:    cudaDeviceSynchronize();
src/dustdynamics1D.cu:    check_CUDA_errors("_set_bounds_d");
src/dustdynamics1D.cu:    check_CUDA_errors("_calc_diff_flux_vl");
src/dustdynamics1D.cu:    check_CUDA_errors("_set_boundary_flux");
src/dustdynamics1D.cu:    check_CUDA_errors("_update_Sig");
src/dustdynamics1D.cu:    cudaDeviceSynchronize();
src/check_tol.cu:#include "cuda_array.h"
src/check_tol.cu:    CudaArray<double> max_elem = make_CudaArray<double>(blocks) ;
src/check_tol.cu:    check_CUDA_errors("__max_block_wise") ;
src/check_tol.cu:    check_CUDA_errors("__compute_error") ;
src/check_tol.cu:    check_CUDA_errors("__compute_error_blocks") ;
src/coagulation.cu:    check_CUDA_errors("_compute_coagulation_rate") ;
src/radmc3d_utils.cpp:    CudaArray<double> R = make_CudaArray<double>(_radius.size()) ;
src/radmc3d_utils.cpp:    CudaArray<double> lat = make_CudaArray<double>(_colatitude.size()+2*nghost_y) ;
src/radmc3d_utils.cpp:    CudaArray<double> wle  = make_CudaArray<double>(nwav) ;
src/radmc3d_utils.cpp:    CudaArray<double> flux = make_CudaArray<double>(nwav) ;
src/radmc3d_utils.cpp:    CudaArray<double> Lband = make_CudaArray<double>(nwav) ;
src/FLD_multi.cu:#include <cuda_runtime.h>
src/FLD_multi.cu:                                 const CudaArray<double>& wle,
src/FLD_multi.cu:    check_CUDA_errors("compute_diffusion_coeff") ;           
src/FLD_multi.cu:    check_CUDA_errors("create_FLD_multi_system") ;    
src/FLD_multi.cu:    check_CUDA_errors("copy_final_values") ;     
src/star.cpp:           const CudaArray<double>& wle_, const CudaArray<double>& Lband_) 
src/star.cpp:    Lband = make_CudaArray<double>(num_wle) ;
src/star.cpp:    Lband = make_CudaArray<double>(num_wle) ;
src/stellar_irradiation.cu:#include <cuda_runtime.h>
src/stellar_irradiation.cu:    check_CUDA_errors("_volumetric_heating") ;
src/stellar_irradiation.cu:    check_CUDA_errors("_volumetric_heating_with_scattering") ;
src/stellar_irradiation.cu:    check_CUDA_errors("_volumetric_heating_with_scattering") ;
src/stellar_irradiation.cu:    // Step 0: Decomposition for gpu
src/stellar_irradiation.cu:    check_CUDA_errors("_cell_optical_depth") ;
src/stellar_irradiation.cu:    // Step 0: Decomposition for gpu
src/stellar_irradiation.cu:    check_CUDA_errors("_cell_optical_depth") ;
src/stellar_irradiation.cu:    // Step 0: Decomposition for gpu
src/stellar_irradiation.cu:    check_CUDA_errors("_cell_optical_depth") ;
src/stellar_irradiation.cu:                                             Field3D<double>& tau_inner, double t, CudaArray<double>& ts, int NZ, int Nt) {
src/stellar_irradiation.cu:    // Step 0: Decomposition for gpu
src/stellar_irradiation.cu:    check_CUDA_errors("_cell_optical_depth") ;
src/stellar_irradiation.cu:    // Step 0: Decomposition for gpu
src/stellar_irradiation.cu:    check_CUDA_errors("_cell_optical_depth") ;
src/stellar_irradiation.cu:    check_CUDA_errors("add_viscous_heating_device") ;
src/stellar_irradiation.cu:    check_CUDA_errors("add_viscous_heating_device") ;
src/stellar_irradiation.cu:                         const Field<Prims>& w_g, const CudaArray<double>& nu, 
src/stellar_irradiation.cu:    check_CUDA_errors("add_viscous_heating_device") ;
src/stellar_irradiation.cu:                         const CudaArray<double>& Sig, const CudaArray<double>& nu, 
src/stellar_irradiation.cu:    check_CUDA_errors("add_viscous_heating_device") ;
src/FLD_mono.cu:#include <cuda_runtime.h>
src/FLD_mono.cu:    check_CUDA_errors("compute_diffusion_coeff") ;           
src/FLD_mono.cu:    check_CUDA_errors("create_FLD_mono_system") ;    
src/FLD_mono.cu:    check_CUDA_errors("copy_final_values") ;           
src/copy.cu:    check_CUDA_errors("copy_field") ;
src/copy.cu:    check_CUDA_errors("copy_field") ;
src/DSHARP_opacs.cu:#include "cuda_runtime.h"
src/DSHARP_opacs.cu:    check_CUDA_errors("_calc_rho_kappa") ;
src/DSHARP_opacs.cu:    check_CUDA_errors("_calc_rho_tot") ;
src/DSHARP_opacs.cu:    check_CUDA_errors("_calc_rho_kappa") ;
src/DSHARP_opacs.cu:    check_CUDA_errors("_calc_rho_tot") ;
src/DSHARP_opacs.cu:    check_CUDA_errors("_calc_rho_kappa") ;
src/DSHARP_opacs.cu:    check_CUDA_errors("_calc_rho_kappa") ;
src/integrate_z.cu:#include "cuda_array.h"
src/integrate_z.cu:void Reduction::volume_integrate_Z_cpu(const Grid& g, const Field<double>& f, CudaArray<double>& result) {
src/integrate_z.cu:void Reduction::volume_integrate_Z(const Grid& g, const Field<double>& in, CudaArray<double>& result) {
src/integrate_z.cu:    // Setup cuda blocks / threads and shared memory size
src/integrate_z.cu:    check_CUDA_errors("volume_integrate_Z") ;
src/sources.cu:#include <cuda_runtime.h>
src/sources.cu:#include "cuda_array.h"
src/hydrostatic.cu:    check_CUDA_errors("setup_hydrostatic_maxtrix") ;
src/hydrostatic.cu:    check_CUDA_errors("convert_pressure_to_density_device") ;
src/hydrostatic.cu:                       const CudaArray<double>& Sigma, const CudaArray<double>& norm) {
src/hydrostatic.cu:    check_CUDA_errors("normalize_density") ;
src/hydrostatic.cu:                                     const Field<double>& cs2, const CudaArray<double>& Sigma) {
src/hydrostatic.cu:    CudaArray<double> norm = make_CudaArray<double>(g.NR + 2*g.Nghost) ;
src/hydrostatic.cu:                                     const Field<double>& cs2, const CudaArray<double>& Sigma, double gasfloor) {
src/hydrostatic.cu:    CudaArray<double> norm = make_CudaArray<double>(g.NR + 2*g.Nghost) ;
src/hydrostatic.cu:                                     const Field<double>& cs2, const CudaArray<double>& Sigma, Field3D<Prims>& q_d, double gasfloor) {
src/hydrostatic.cu:    CudaArray<double> norm = make_CudaArray<double>(g.NR + 2*g.Nghost) ;
src/scan3d.cu:#include "cuda_array.h"
src/scan3d.cu:    // Setup cuda blocks / threads and shared memory size
src/scan3d.cu:    check_CUDA_errors("scan3D_R_OP") ;
src/scan3d.cu:    // Setup cuda blocks / threads and shared memory size
src/scan3d.cu:    check_CUDA_errors("scan3D_Z_OP") ;
src/gmres.cpp:#include <cuda_runtime.h>
src/gmres.cpp:       https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
src/gmres.cpp:        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size) ;
src/gmres.cpp:    CudaArray<char>  buffer = make_CudaArray<char>(buffer_size) ;
src/gmres.cpp:        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer.get()) ;
src/gmres.cpp:        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, buffer) ;
headers/file_io.h:                 CudaArray<double>& Sig_g) {
headers/file_io.h:                 CudaArray<double>& Sig_g) {
headers/file_io.h:                CudaArray<double>& Sig_g,  Field<double> &T, Field3D<double> &J) {
headers/file_io.h:               CudaArray<double>& Sig_g, Field<double> &T, Field3D<double> &J) {
headers/file_io.h:void write_restart_quants(std::filesystem::path folder, Grid &g, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g, 
headers/file_io.h:void read_restart_quants(std::filesystem::path folder, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g, 
headers/file_io.h:void write_restart_prims(std::filesystem::path folder, Grid &g, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g) {
headers/file_io.h:void read_restart_prims(std::filesystem::path folder, Field3D<Prims>& wd, Field<Prims>& wg, CudaArray<double>& Sig_g) {
headers/timing.h:#include <cuda_runtime.h>
headers/timing.h:        cudaDeviceSynchronize() ;
headers/grid.h:#include "cuda_array.h"
headers/grid.h:         CudaArray<double> R, CudaArray<double> phi, int index=-1);
headers/grid.h:    CudaArray<double> _Re, _Rc, _Ar, _Az, _V ;
headers/grid.h:    CudaArray<double> _sin_theta_e, _sin_theta_c ;
headers/grid.h:    CudaArray<double> _cos_theta_e, _cos_theta_c ;
headers/grid.h:    CudaArray<double> _tan_theta_e, _tan_theta_c ;
headers/grid.h: * This class exists to enable copying Grid objects to the GPU. Since objects
headers/grid.h: * to handle this. Note that passing by value is impossible because CudaArrays
headers/grid.h:        CudaArray<double> _Re, _Rc ;
headers/grid.h:        CudaArray<double> _Ze, _Zc ;
headers/grid.h:        void copy_to_subgrid(Grid& g_sub, const CudaArray<T>& F_main, CudaArray<T>& F_sub) ;
headers/grid.h:        void copy_from_subgrid(Grid& g_sub, CudaArray<T>& F_main, const CudaArray<T>& F_sub) ;
headers/coagulation/coagulation.h:#include "cuda_array.h"
headers/coagulation/coagulation.h:        _id = make_CudaArray<id>(size * stride) ;
headers/coagulation/coagulation.h:        _Cijk = make_CudaArray<coeff>(size * stride) ;
headers/coagulation/coagulation.h:        _Cijk_frag = make_CudaArray<double>(size * stride) ;
headers/coagulation/coagulation.h:    CudaArray<id> _id ;
headers/coagulation/coagulation.h:    CudaArray<coeff> _Cijk ;
headers/coagulation/coagulation.h:    CudaArray<double> _Cijk_frag ;
headers/coagulation/size_grid.h:#include "cuda_array.h"
headers/coagulation/size_grid.h:      : _mass_e(make_CudaArray<RealType>(Nbins+1)),
headers/coagulation/size_grid.h:        _mass_c(make_CudaArray<RealType>(Nbins)),
headers/coagulation/size_grid.h:        _a_c(make_CudaArray<RealType>(Nbins)),
headers/coagulation/size_grid.h:    SizeGrid(CudaArray<RealType>& a, int Nbins, RealType rho_daux=1)
headers/coagulation/size_grid.h:      : _mass_e(make_CudaArray<RealType>(Nbins+1)),
headers/coagulation/size_grid.h:        _mass_c(make_CudaArray<RealType>(Nbins)),
headers/coagulation/size_grid.h:        _a_c(make_CudaArray<RealType>(Nbins)),
headers/coagulation/size_grid.h:    CudaArray<RealType> _mass_e, _mass_c, _a_c ;
headers/gas1d.h:#include "cuda_array.h"
headers/gas1d.h:void update_gas_sigma(Grid& g, CudaArray<double>& Sig_g, double dt, const CudaArray<double>& nu, int bound, double floor);
headers/gas1d.h:void update_gas_sigma(Grid& g, CudaArray<double>& Sig_g, double dt, const Field<double>& nu, int bound, double floor);
headers/gas1d.h:void update_gas_vel(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& u_gas, double alpha, Star& star);
headers/gas1d.h:void calc_gas_velocities(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, double alpha, Star& star, int bound, double floor, double cav=0.) ;
headers/gas1d.h:void calc_gas_velocities_full(Grid& g, CudaArray<double>& Sig_g, Field<Prims>& wg, Field<double>& cs2, CudaArray<double>& nu, double alpha, Star& star, int bound, double floor, double cav=0.) ;
headers/gas1d.h:void calc_gas_velocities_wind(Grid& g, Field<Prims>& wg, CudaArray<double>& Sig_g, Field<double>& cs2, CudaArray<double>& nu, CudaArray<double>& Sig_dot_w,
headers/gas1d.h:void update_gas_sources(Grid& g, CudaArray<double>& Sig_g, CudaArray<double>& Sigdot, double dt, int bound, double gfloor);
headers/gas1d.h:double calc_dt(Grid& g, const CudaArray<double>& nu);
headers/gas1d.h:void calc_wind_surface(Grid& g, const Field<Prims>& wg, CudaArray<double>& h_w, double col);
headers/gas1d.h:void calculate_ubar(Grid& g, CudaArray<double>& sig, CudaArray<double>& sig_g, 
headers/gas1d.h:                    CudaArray<double>& ubar, CudaArray<double>& u_gas,
headers/gas1d.h:void update_dust_sigma(Grid& g, CudaArray<double>& sig, CudaArray<double>& sig_g, 
headers/gas1d.h:                    CudaArray<double>& ubar, CudaArray<double>& D, double dt, int bound);
headers/gas1d.h:double compute_CFL(Grid& g, CudaArray<double>& ubar, CudaArray<double>& D,
headers/gas1d.h:void update_gas_sigma(Grid& g, Field<Prims1D>& W_g, double dt, const CudaArray<double>& nu, int bound, double floor);
headers/gas1d.h:void calc_v_gas(Grid& g, Field<Prims1D>& W_g, const Field<double>& cs, CudaArray<double>& nu, double GMstar, double gasfloor);
headers/reductions.h:#include "cuda_array.h"
headers/reductions.h:    void volume_integrate_Z(const Grid& g, const Field<double>& in, CudaArray<double>& result) ;
headers/reductions.h:    void volume_integrate_Z_cpu(const Grid& g, const Field<double>& in, CudaArray<double>& result) ;
headers/planck.h:#include "cuda_array.h"
headers/planck.h:        _val = make_CudaArray<double>(Nmax) ;
headers/planck.h:    CudaArray<double> _val ;
headers/planck.h: * Reference type for PlanckInegral, allowing it to be used on the GPU.
headers/pcg_solver.h:    CudaArray<char> bufferL, bufferU ;
headers/pcg_solver.h:    CudaArray<char> buffer ;
headers/FLD.h:                                 const CudaArray<double>& wle,
headers/matrix_types.h:#include "cuda_array.h"
headers/matrix_types.h: * Wrapper object for GPU Compressed Sparse Row matrix format
headers/matrix_types.h:        csr_offset(make_CudaArray<int>(rows_+1)),
headers/matrix_types.h:        col_index(make_CudaArray<int>(non_zeros_)),
headers/matrix_types.h:        data(make_CudaArray<double>(non_zeros_))
headers/matrix_types.h:                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) ;
headers/matrix_types.h:    /* Some CUDA functions will update the number of matrix
headers/matrix_types.h:        csr_offset = make_CudaArray<int>(rows+1) ;
headers/matrix_types.h:        col_index = make_CudaArray<int>(non_zeros) ;
headers/matrix_types.h:        data = make_CudaArray<double>(non_zeros) ;
headers/matrix_types.h:    CudaArray<int> csr_offset ;
headers/matrix_types.h:    CudaArray<int> col_index ;
headers/matrix_types.h:    CudaArray<double> data ;
headers/matrix_types.h:                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) ;
headers/matrix_types.h:                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) ;
headers/matrix_types.h: * Wrapper object for GPU Compressed Sparse Row matrix format using
headers/matrix_types.h: * This class exists to enable copying CSR_SpMatrix objects to the GPU. 
headers/matrix_types.h: * because CudaArrays are non-copyable.
headers/matrix_types.h: * This class exists to enable copying CSR_SpMatrix objects to the GPU. 
headers/matrix_types.h: * because CudaArrays are non-copyable.
headers/matrix_types.h:       data(make_CudaArray<double>(rows_))
headers/matrix_types.h:            cusparseCreateDnVec(&descr, rows, data.get(), CUDA_R_64F) ;
headers/matrix_types.h:            cusparseCreateDnVec(&descr, rows, data.get(), CUDA_R_64F) ;
headers/matrix_types.h:            cusparseCreateDnVec(&descr, rows, data.get(), CUDA_R_64F) ;
headers/matrix_types.h:    CudaArray<double> data ;
headers/matrix_types.h: * This class exists to enable copying DnVec objects to the GPU. 
headers/matrix_types.h: * because CudaArrays are non-copyable.
headers/matrix_types.h: * This class exists to enable copying DnVec objects to the GPU. 
headers/matrix_types.h: * because CudaArrays are non-copyable.
headers/stellar_irradiation.h:                                             Field3D<double>& tau_inner, double t, CudaArray<double>& ts, int NZ, int Nt) ; 
headers/stellar_irradiation.h:                         const Field<Prims>& w_g, const CudaArray<double>& nu, 
headers/stellar_irradiation.h:                         const CudaArray<double>& Sig, const CudaArray<double>& nu, 
headers/star.h:#include <cuda_array.h>
headers/star.h:         const CudaArray<double>& wle, const CudaArray<double>& Lband) ;
headers/star.h:        wle = make_CudaArray<double>(num_wle) ;
headers/star.h:    CudaArray<double> wle ;
headers/star.h:    CudaArray<double> Lband ;
headers/advection.h:#include "cuda_array.h"
headers/bins.h:#include "cuda_array.h"
headers/bins.h:        bands(make_CudaArray<double>(num_bands)),
headers/bins.h:        edges(make_CudaArray<double>(num_bands-1)),
headers/bins.h:        _wle_in_e(make_CudaArray<double>(nwle-1)),
headers/bins.h:        _wle_in_c(make_CudaArray<double>(nwle))
headers/bins.h:    CudaArray<double> bands, edges ;
headers/bins.h:    CudaArray<double> bin_data(const CudaArray<double>& input,
headers/bins.h:    CudaArray<double> bin_planck_data(const CudaArray<double>& input,
headers/bins.h:    CudaArray<double> _wle_in_e ;
headers/bins.h:    CudaArray<double> _wle_in_c ;
headers/dustdynamics.h:#include "cuda_array.h"
headers/dustdynamics.h:        void floor_above(Grid&g, Field3D<Prims>& w_dust, Field<Prims>& w_gas, CudaArray<double>& h);
headers/utils.h:#include <cuda_runtime.h>
headers/utils.h:/* check_CUDA_errors
headers/utils.h: * Checks that the last cuda operation succeeded. If not, it raises a 
headers/utils.h:inline void check_CUDA_errors(std::string fn_name) {
headers/utils.h:  cudaDeviceSynchronize();
headers/utils.h:  cudaError_t cudaError = cudaGetLastError();
headers/utils.h:  if (cudaError != cudaSuccess) {
headers/utils.h:    const char *error = cudaGetErrorString(cudaError);
headers/utils.h:    throw std::runtime_error("CUDA error in " + fn_name + 
headers/utils.h:inline void check_CUDA_devices() {
headers/utils.h:  cudaGetDeviceCount(&nDevices);
headers/utils.h:    cudaDeviceProp prop;
headers/utils.h:    cudaGetDeviceProperties(&prop, i);
headers/opacity.h:#include <cuda.h>
headers/hydrostatic.h:                                     const Field<double>&, const CudaArray<double>&) ;
headers/hydrostatic.h:                                     const Field<double>&, const CudaArray<double>&, double gasfloor=1e-100) ;
headers/hydrostatic.h:                                     const Field<double>&, const CudaArray<double>&, Field3D<Prims>& q_d, double gasfloor=1e-100) ;
headers/field.h:#include "cuda_array.h"
headers/field.h:        _ptr = make_CudaArray<T>(NR * stride) ;
headers/field.h:    CudaArray<T>& get() {
headers/field.h:    const CudaArray<T>& get() const {
headers/field.h:    CudaArray<T> _ptr; 
headers/field.h: * This class exists to enable copying Field objects to the GPU. Since objects
headers/field.h: * to handle this. Note that passing by value is impossible because CudaArrays
headers/field.h: * This class exists to enable copying const Field objects to the GPU. Since 
headers/field.h: * because CudaArrays are non-copyable.
headers/field.h:       _ptr = make_CudaArray<T>(NR * stride_Zd) ;
headers/field.h:    CudaArray<T>& get() {
headers/field.h:    const CudaArray<T>& get() const {
headers/field.h:    CudaArray<T> _ptr; 
headers/field.h: * This class exists to enable copying Field3D objects to the GPU. Since objects
headers/field.h: * to handle this. Note that passing by value is impossible because CudaArrays
headers/field.h: * This class exists to enable copying const Field3D objects to the GPU. Since 
headers/field.h: * because CudaArrays are non-copyable.
headers/cuda_array.h:/* cuda_array.h
headers/cuda_array.h: *  the CPU (host) and GPU (device). Since cudaMallocManaged is used there is no
headers/cuda_array.h:#ifndef _CUDISC_CUDA_ARRAY_H_
headers/cuda_array.h:#define _CUDISC_CUDA_ARRAY_H_
headers/cuda_array.h:#include <cuda_runtime.h>
headers/cuda_array.h:/* CudaArray_deleter
headers/cuda_array.h:struct CudaArray_deleter {
headers/cuda_array.h:      cudaDeviceSynchronize();
headers/cuda_array.h:      cudaFree(p);
headers/cuda_array.h:      check_CUDA_errors("CudaArray_deleter") ;
headers/cuda_array.h:/* CudaArray
headers/cuda_array.h:using CudaArray = std::unique_ptr<T[], CudaArray_deleter<T>>;
headers/cuda_array.h:/* make_CudaArray
headers/cuda_array.h: * using cudaMallocManaged. 
headers/cuda_array.h: * The CUDA runtime will automaticly handle any copies between the host and device 
headers/cuda_array.h:CudaArray<T> make_CudaArray(std::size_t n) {
headers/cuda_array.h:    cudaError_t status = cudaMallocManaged(&ptr, size);
headers/cuda_array.h:    if (status != cudaSuccess) {
headers/cuda_array.h:        throw std::runtime_error("CUDA Failed to allocate memory") ;
headers/cuda_array.h:    cudaDeviceSynchronize();
headers/cuda_array.h:    return CudaArray<T>(ptr);
headers/cuda_array.h:#endif//_CUDISC_CUDA_ARRAY_H_
headers/DSHARP_opacs.h:#include "cuda_array.h"
headers/DSHARP_opacs.h:            a_ptr = make_CudaArray<double>(n_a);
headers/DSHARP_opacs.h:            lam_ptr = make_CudaArray<double>(n_lam);
headers/DSHARP_opacs.h:            k_abs_ptr = make_CudaArray<double>(n_a*n_lam);
headers/DSHARP_opacs.h:            k_sca_ptr = make_CudaArray<double>(n_a*n_lam);
headers/DSHARP_opacs.h:            k_abs_g_ptr = make_CudaArray<double>(n_lam);
headers/DSHARP_opacs.h:            k_sca_g_ptr = make_CudaArray<double>(n_lam);
headers/DSHARP_opacs.h:            a_ptr = make_CudaArray<double>(n_a);
headers/DSHARP_opacs.h:            lam_ptr = make_CudaArray<double>(n_lam);
headers/DSHARP_opacs.h:            k_abs_ptr = make_CudaArray<double>(n_a*n_lam);
headers/DSHARP_opacs.h:            k_sca_ptr = make_CudaArray<double>(n_a*n_lam);
headers/DSHARP_opacs.h:                CudaArray<double> g = make_CudaArray<double>(n_a*n_lam);
headers/DSHARP_opacs.h:        CudaArray<double> a_ptr;
headers/DSHARP_opacs.h:        CudaArray<double> lam_ptr;
headers/DSHARP_opacs.h:        CudaArray<double> k_abs_ptr;
headers/DSHARP_opacs.h:        CudaArray<double> k_sca_ptr;
headers/DSHARP_opacs.h:        CudaArray<double> k_abs_g_ptr;
headers/DSHARP_opacs.h:        CudaArray<double> k_sca_g_ptr;

```
