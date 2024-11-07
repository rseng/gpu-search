# https://github.com/oliverphilcox/encore

```console
Makefile:# GPU Compilation
Makefile:CUFLAGS = modules/gpufuncs.o -L/usr/local/cuda/lib64 -lcudart
Makefile:#Note - will have to figure out OPENMP vs CUDA - e.g. don't want 24 threads trying to each start a CUDA kernel
Makefile:#CXXFLAGS = -O3 -DGPU -DFOURPCF -DFIVEPCF -DSIXPCF ${CUFLAGS}
Makefile:#CXXFLAGS = -O3 -DGPU -DFOURPCF -DFIVEPCF ${CUFLAGS}
Makefile:#CXXFLAGS = -O3 -DGPU -DFOURPCF -DDISCONNECTED ${CUFLAGS}
Makefile:CXXFLAGS = -O3 -DGPU -DFOURPCF -DPERIODIC -DDISCONNECTED ${CUFLAGS}
Makefile:#CXXFLAGS = -O3 -DGPU ${CUFLAGS}
Makefile:#MODES = -DGPU -DFOURPCF -DFIVEPCF -DSIXPCF
Makefile:#MODES = -DGPU -DFOURPCF -DFIVEPCF
Makefile:#MODES = -DGPU -DFOURPCF -DDISCONNECTED
Makefile:#MODES = -DGPU -DFOURPCF -DPERIODIC
Makefile:MODES = -DGPU -DFOURPCF -DPERIODIC -DDISCONNECTED
Makefile:#MODES = -DGPU -DFOURPCF
Makefile:#MODES = -DGPU
Makefile:gpu: gpufuncs encore encoreAVX
Makefile:gpufuncs: 
Makefile:	nvcc $(NVCCFLAGS) -c modules/gpufuncs.cu -o modules/gpufuncs.o
Makefile:	$(RM) encore encoreAVX CMASM.o modules/gpufuncs.o
README.md:C++ code for estimating the isotropic NPCF multipoles for an arbitrary survey geometry in O(N^2) time, with optional GPU support. This is based on code by Daniel Eisenstein, implementing the algorithm of [Philcox et al. 2021](http://arxiv.org/abs/2105.08722), and uses their conventions. This currently features support for the isotropic 2PCF, 3PCF, 4PCF, 5PCF and 6PCF, with the option to subtract the Gaussian 4PCF contributions at the estimator level. For the 4PCF, 5PCF and 6PCF algorithms, the runtime is dominated by sorting the spherical harmonics into bins, which has complexity O(N_galaxy x N_bins^3 x N_ell^5) [4PCF], O(N_galaxy x N_bins^4 x N_ell^8) [5PCF] or O(N_galaxy x N_bins^5 x N_ell^11) [6PCF]. We caution that the higher-point functions will be necessarily slow to compute unless N_bins and N_ell are small.
README.md:- *(Optional)*: CUDA (7.0 or higher) for GPU computations.
README.md:- To run the code, first compile it using ```make clean; make```. You will need to edit the Makefile depending on your particular configurations. In particular, the Makefile has the options ```-DOPENMP``` to run with OpenMP support for parallelization, ```-DFOURPCF```, ```-DFIVEPCF``` and ```-DSIXPCF``` to enable the 4PCF/5PCF/6PCF computation, ```-DPERIODIC``` to assume a periodic box geometry, ```-DALLPARITY``` to include odd-parity NPCFs, ```-DAVX``` to additionally compile the code using AVX instructions and ```-DGPU``` to run on a GPU. The 2PCF and 3PCF are always computed.
README.md:- To enable CUDA code on the GPU, add ```-DGPU``` to MODES in the Makefile and add ${CUFLAGS} to CXXFLAGS and run ```make clean; make gpu```
README.md:- ```-gpu n```: gpu mode => 0 = CPU, 1 = GPU kernel 1 (one thread per array element); 2 = GPU kernel 2 (one thread per inner loop operation).
README.md:- ```-float```: In GPU mode, use floats on the GPU for speed.
README.md:- ```-mixed```: In GPU mode, use mixed precision on the GPU with ALMs in float space and accumulations in double.
modules/ComputeMultipoles.h:#ifdef GPU
modules/ComputeMultipoles.h:#include "gpufuncs.h"
modules/ComputeMultipoles.h:#ifdef GPU
modules/ComputeMultipoles.h:    if (_gpumode > 0) {
modules/ComputeMultipoles.h:      if (_gpump == 1) {
modules/ComputeMultipoles.h:        gpu_allocate_multipoles(&msave, &csave,
modules/ComputeMultipoles.h:      } else if (_gpump == 2) {
modules/ComputeMultipoles.h:        gpu_allocate_multipoles_fast(&msave, &csave,
modules/ComputeMultipoles.h:      gpu_allocate_particle_arrays(&posx, &posy, &posz, &weights, np);
modules/ComputeMultipoles.h:      gpu_allocate_pair_arrays(&x0i, &x2i, NBIN);
modules/ComputeMultipoles.h:      if (_gpump == 2) {
modules/ComputeMultipoles.h:      gpu_allocate_alms(maxp, NBIN, NLM, !_gpufloat && !_gpumixed);
modules/ComputeMultipoles.h:    if (_gpumode > 0) {
modules/ComputeMultipoles.h:      gpu_allocate_periodic(&delta_x, &delta_y, &delta_z, nmax);
modules/ComputeMultipoles.h:	if (_gpumode > 0 && _gpump == 2) icnt += primary.np; else
modules/ComputeMultipoles.h:#ifdef GPU
modules/ComputeMultipoles.h:			if (_gpumode > 0) {
modules/ComputeMultipoles.h:			  //populate arrays only for GPU mode
modules/ComputeMultipoles.h:	    if (_gpumode == 0) {
modules/ComputeMultipoles.h:	      //Acumulate powers here - code in NPCF.h (uses GPU kernels)
modules/ComputeMultipoles.h:#ifdef GPU
modules/ComputeMultipoles.h:	if (_gpumode > 0 && (dcnt > nmax-nthresh || icnt > maxp-pthresh || ne == (grid->nf)-1)) {
modules/ComputeMultipoles.h:	  if (_gpump == 1) nthreads = (int)dcnt;
modules/ComputeMultipoles.h:	  bool usefloat = (_gpufloat || _gpumixed); 
modules/ComputeMultipoles.h:	    printf("Running add_pairs_only_periodic kernel %d, mem mode %d, with %d threads after cell %d\n", _gpump, _shared, nthreads, ne);
modules/ComputeMultipoles.h:            if (_gpump == 1) gpu_add_pairs_only_periodic(posx, posy, posz, weights, pnum, spnum, snp, sc, x0i, x2i, delta_x, delta_y, delta_z, (int)dcnt, NBIN, rmin, rmax, grid->cellsize, _shared, usefloat);
modules/ComputeMultipoles.h:            else if (_gpump == 2) gpu_add_pairs_only_periodic_fast(posx, posy, posz, weights, start_list, np_list, cellnums, x0i, x2i, nthreads, NBIN, ORDER, NMULT, rmin, rmax, grid->nside_cuboid.x, grid->nside_cuboid.y, grid->nside_cuboid.z, grid->ncells, maxsep, lastcnt, grid->cellsize,_shared, usefloat);
modules/ComputeMultipoles.h:	    printf("Running add_pairs_and_multipoles_periodic kernel %d, mem mode %d, with %d threads after cell %d\n", _gpump, _shared, nthreads, ne);
modules/ComputeMultipoles.h:            if (_gpump == 1) gpu_add_pairs_and_multipoles_periodic(msave, posx, posy, posz, weights, csave, pnum, spnum, snp, sc, x0i, x2i, delta_x, delta_y, delta_z, (int)dcnt, NBIN, ORDER, NMULT, rmin, rmax, lastcnt, grid->cellsize, _shared, usefloat);
modules/ComputeMultipoles.h:	    else if (_gpump == 2) gpu_add_pairs_and_multipoles_periodic_fast(msave, posx, posy, posz, weights, csave, start_list, np_list, cellnums, x0i, x2i, nthreads, NBIN, ORDER, NMULT, rmin, rmax, grid->nside_cuboid.x, grid->nside_cuboid.y, grid->nside_cuboid.z, grid->ncells, maxsep, lastcnt, grid->cellsize,_shared, usefloat);
modules/ComputeMultipoles.h:            printf("Running add_pairs_only kernel %d, mem mode %d, with %d threads after cell %d\n", _gpump, _shared, nthreads, ne); 
modules/ComputeMultipoles.h:            if (_gpump == 1) gpu_add_pairs_only(posx, posy, posz, weights, pnum, spnum, snp, sc, x0i, x2i, (int)dcnt, NBIN, rmin, rmax, _shared, usefloat);
modules/ComputeMultipoles.h:            else if (_gpump == 2) gpu_add_pairs_only_fast(posx, posy, posz, weights, start_list, np_list, cellnums, x0i, x2i, nthreads, NBIN, ORDER, NMULT, rmin, rmax, grid->nside_cuboid.x, grid->nside_cuboid.y, grid->nside_cuboid.z, grid->ncells, maxsep, lastcnt, _shared, usefloat);
modules/ComputeMultipoles.h:            printf("Running add_pairs_and_multipoles kernel %d, mem mode %d, with %d threads after cell %d\n", _gpump, _shared, nthreads, ne);
modules/ComputeMultipoles.h:	    if (_gpump == 1) gpu_add_pairs_and_multipoles(msave, posx, posy, posz, weights, csave, pnum, spnum, snp, sc, x0i, x2i, (int)dcnt, NBIN, ORDER, NMULT, rmin, rmax, lastcnt, _shared, usefloat); 
modules/ComputeMultipoles.h:            else if (_gpump == 2) gpu_add_pairs_and_multipoles_fast(msave, posx, posy, posz, weights, csave, start_list, np_list, cellnums, x0i, x2i, nthreads, NBIN, ORDER, NMULT, rmin, rmax, grid->nside_cuboid.x, grid->nside_cuboid.y, grid->nside_cuboid.z, grid->ncells, maxsep, lastcnt, _shared, usefloat);
modules/ComputeMultipoles.h:          //gpu_device_synchronize(); //synchronize before copying data
modules/ComputeMultipoles.h:            gpu_device_synchronize(); //synchronize before copying data
modules/ComputeMultipoles.h:	  if (_gpufloat || _gpumixed) {
modules/ComputeMultipoles.h:            gpu_compute_alms_float((int *)(npcf[0].map), msave, NBIN, NLM, maxp, ORDER, MAXORDER+1, NMULT);
modules/ComputeMultipoles.h:	    gpu_compute_alms((int *)(npcf[0].map), msave, NBIN, NLM, maxp, ORDER, MAXORDER+1, NMULT);
modules/ComputeMultipoles.h:          gpu_device_synchronize(); //synchronize before copying data
modules/ComputeMultipoles.h:          gpu_device_synchronize(); //synchronize before copying data
modules/ComputeMultipoles.h:	  npcf[thread].add_to_power3_gpu(weights, icnt);
modules/ComputeMultipoles.h:          //gpu_device_synchronize(); //synchronize before copying data
modules/ComputeMultipoles.h:	  npcf[thread].add_to_power_disconnected_gpu(weights, icnt);
modules/ComputeMultipoles.h:		//since alm calcs are already done on GPU and GPU methods
modules/ComputeMultipoles.h:	    //seems to be faster on CPU than GPU
modules/ComputeMultipoles.h:#ifdef GPU
modules/ComputeMultipoles.h:    if (_gpumode > 0) {
modules/ComputeMultipoles.h:        gpu_device_synchronize(); //synchronize before copying data
modules/ComputeMultipoles.h:        printf("\nGPU Spherical harmonics: %.3f s",sphtime.Elapsed());
modules/ComputeMultipoles.h:        npcf[0].free_gpu_memory(); //free all GPU memory
modules/ComputeMultipoles.h:	free_gpu_multipole_arrays(msave, csave, pnum, spnum, snp, sc, posx, posy, posz, weights, x0i, x2i);
modules/ComputeMultipoles.h:        free_gpu_periodic_arrays(delta_x, delta_y, delta_z);
modules/ComputeMultipoles.h:  #ifdef GPUALM
modules/ComputeMultipoles.h:    // free gpu memory
modules/ComputeMultipoles.h:    gpu_free_mult(gmult,gmult_ct);
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:#include "gpufuncs.h"
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:    //allocate both real and imaginary components of discon1 for GPU
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:  inline void add_to_power3_gpu(double *weights, int np) {
modules/NPCF.h:    //We need a separate method to call for 3PCF on the GPU
modules/NPCF.h:    //on GPU as well, this is easy to separate out.
modules/NPCF.h:    if (_gpumode == 0) return;
modules/NPCF.h:      //can only get here in _gpumode > 0
modules/NPCF.h:      if (_gpufloat) {
modules/NPCF.h:        gpu_allocate_weight3pcf(&f_weight3pcf, weight3pcf, NLM);
modules/NPCF.h:        gpu_allocate_threepcf(&f_threepcf, threepcf, NL*N3PCF);
modules/NPCF.h:        gpu_allocate_weight3pcf(&d_weight3pcf, weight3pcf, NLM);
modules/NPCF.h:        gpu_allocate_threepcf(&d_threepcf, threepcf, NL*N3PCF);
modules/NPCF.h:      gpu_allocate_luts3(&lut3_i, &lut3_j, &lut3_ct, nouter3);
modules/NPCF.h:    if (_gpufloat) {
modules/NPCF.h:      gpu_add_to_power3_orig_float(f_threepcf, f_weight3pcf,
modules/NPCF.h:    } else if (_gpumixed) {
modules/NPCF.h:      gpu_add_to_power3_orig_mixed(d_threepcf, d_weight3pcf,
modules/NPCF.h:      gpu_add_to_power3_orig(d_threepcf, d_weight3pcf,
modules/NPCF.h:    inline void add_to_power_disconnected_gpu(double *weights, int np) {
modules/NPCF.h:    //We need a separate method to call for DISCONNECTED 4PCF on the GPU
modules/NPCF.h:    //on GPU as well, this is easy to separate out.
modules/NPCF.h:    if (_gpumode == 0) return;
modules/NPCF.h:      //can only get here in _gpumode > 0
modules/NPCF.h:      if (_gpufloat) {
modules/NPCF.h:        gpu_allocate_weightdiscon(&f_weightdiscon, weightdiscon, size_w);
modules/NPCF.h:        gpu_allocate_discon1(&f_discon1_r, &f_discon1_i, discon1, ndiscon1);
modules/NPCF.h:	gpu_allocate_discon2(&f_discon2_r, &f_discon2_i, discon2, ndiscon2);
modules/NPCF.h:        gpu_allocate_weightdiscon(&d_weightdiscon, weightdiscon, size_w);
modules/NPCF.h:        gpu_allocate_discon1(&d_discon1_r, &d_discon1_i, discon1, ndiscon1); 
modules/NPCF.h:        gpu_allocate_discon2(&d_discon2_r, &d_discon2_i, discon2, ndiscon2);
modules/NPCF.h:      gpu_allocate_luts_discon1(&lut_discon_ell, &lut_discon_mm, NL*NL);
modules/NPCF.h:      gpu_allocate_luts_discon2(&lut_discon_ell1, &lut_discon_ell2,
modules/NPCF.h:      gpu_allocate_luts_discon2_inner(&lut_discon_i, &lut_discon_j, ninner);
modules/NPCF.h:    if (_gpufloat) {
modules/NPCF.h:      gpu_add_to_power_discon1_orig_float(f_discon1_r, f_discon1_i,
modules/NPCF.h:      gpu_add_to_power_discon2_final_float(f_discon2_r, f_discon2_i,
modules/NPCF.h:    } else if (_gpumixed) {
modules/NPCF.h:      gpu_add_to_power_discon1_orig_mixed(d_discon1_r, d_discon1_i,
modules/NPCF.h:      gpu_add_to_power_discon2_final_mixed(d_discon2_r, d_discon2_i,
modules/NPCF.h:      gpu_add_to_power_discon1_orig(d_discon1_r, d_discon1_i, d_weightdiscon,
modules/NPCF.h:      gpu_add_to_power_discon2_final(d_discon2_r, d_discon2_i, d_weightdiscon,
modules/NPCF.h:    if (_gpumode == 0) {
modules/NPCF.h:    //compute conjugates if not gpu mode
modules/NPCF.h:    if (_gpumode == 0) {
modules/NPCF.h:    //GPU mode uses method add_to_power3_gpu 
modules/NPCF.h:    if (_gpumode == 0) {
modules/NPCF.h:    if (_gpumode == 0) {
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:    if (_gpumode == 0) generate_luts4 = false;
modules/NPCF.h:      //can only get here in _gpumode > 0
modules/NPCF.h:      if (_gpumode == 1) {
modules/NPCF.h:        //use primary GPU kernel
modules/NPCF.h:        //generate LUTs here for primary GPU kernel
modules/NPCF.h:      } else if (_gpumode == 2) {
modules/NPCF.h:      gpu_allocate_luts4(&lut4_l1, &lut4_l2, &lut4_l3, &lut4_odd, &lut4_n,
modules/NPCF.h:      if (_gpumode == 2) {
modules/NPCF.h:        gpu_allocate_m_luts4(&lut4_m1, &lut4_m2, nouter4);
modules/NPCF.h:      if (_gpufloat) {
modules/NPCF.h:        gpu_allocate_weight4pcf(&f_weight4pcf, weight4pcf, size_w);
modules/NPCF.h:        gpu_allocate_fourpcf(&f_fourpcf, fourpcf, nell4*N4PCF);
modules/NPCF.h:        //normal mode - allocate GPU arrays and copy
modules/NPCF.h:        gpu_allocate_weight4pcf(&d_weight4pcf, weight4pcf, size_w);
modules/NPCF.h:        gpu_allocate_fourpcf(&d_fourpcf, fourpcf, nell4*N4PCF);
modules/NPCF.h:      if (_gpumode == 1) {
modules/NPCF.h:        //use primary GPU kernel
modules/NPCF.h:        //generate LUTs here for primary GPU kernel
modules/NPCF.h:              lut4_n[iouter4] = n; //this is the starting n for this GPU thread
modules/NPCF.h:              //GPU thread will then loop over ms
modules/NPCF.h:      } else if (_gpumode == 2) {
modules/NPCF.h:    if (_gpumode == 1) {
modules/NPCF.h:      //execute GPU kernel
modules/NPCF.h:      if (_gpufloat) {
modules/NPCF.h:        gpu_add_to_power4_float(f_fourpcf, f_weight4pcf,
modules/NPCF.h:      } else if (_gpumixed) {
modules/NPCF.h:        gpu_add_to_power4_mixed(d_fourpcf, d_weight4pcf,
modules/NPCF.h:        gpu_add_to_power4(d_fourpcf, d_weight4pcf,
modules/NPCF.h:    } else if (_gpumode == 2) {
modules/NPCF.h:      //execute alternate GPU kernel 
modules/NPCF.h:      if (_gpufloat) {
modules/NPCF.h:        gpu_add_to_power4_orig_float(f_fourpcf, f_weight4pcf,
modules/NPCF.h:      } else if (_gpumixed) {
modules/NPCF.h:        gpu_add_to_power4_orig_mixed(d_fourpcf, d_weight4pcf,
modules/NPCF.h:        gpu_add_to_power4_orig(d_fourpcf, d_weight4pcf,
modules/NPCF.h:    } else if (_gpumode == 0) {
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:  if (_gpumode == 0) generate_luts = false;
modules/NPCF.h:    //can only get here in _gpumode > 0
modules/NPCF.h:    if (_gpumode == 1) {
modules/NPCF.h:      //use primary GPU kernel
modules/NPCF.h:      //generate LUTs here for primary GPU kernel
modules/NPCF.h:    } else if (_gpumode == 2) {
modules/NPCF.h:    gpu_allocate_luts(&lut5_l1, &lut5_l2, &lut5_l12, &lut5_l3,
modules/NPCF.h:    if (_gpumode == 2) {
modules/NPCF.h:      gpu_allocate_m_luts(&lut5_m1, &lut5_m2, &lut5_m3, nouter5);
modules/NPCF.h:    if (_gpufloat) {
modules/NPCF.h:      gpu_allocate_weight5pcf(&f_weight5pcf, weight5pcf, size_w);
modules/NPCF.h:      gpu_allocate_fivepcf(&f_fivepcf, fivepcf, nell5*N5PCF);
modules/NPCF.h:      //normal mode - allocate GPU arrays and copy
modules/NPCF.h:      gpu_allocate_weight5pcf(&d_weight5pcf, weight5pcf, size_w); 
modules/NPCF.h:      gpu_allocate_fivepcf(&d_fivepcf, fivepcf, nell5*N5PCF);
modules/NPCF.h:    if (_gpumode == 1) {
modules/NPCF.h:      //use primary GPU kernel
modules/NPCF.h:      //generate LUTs here for primary GPU kernel
modules/NPCF.h:	        lut5_n[iouter5] = n; //this is the starting n for this GPU thread
modules/NPCF.h:	        //GPU thread will then loop over ms
modules/NPCF.h:    } else if (_gpumode == 2) {
modules/NPCF.h:  if (_gpumode == 1) {
modules/NPCF.h:    //execute GPU kernel
modules/NPCF.h:    if (_gpufloat) {
modules/NPCF.h:      gpu_add_to_power5_float(f_fivepcf, f_weight5pcf,
modules/NPCF.h:    } else if (_gpumixed) {
modules/NPCF.h:      gpu_add_to_power5_mixed(d_fivepcf, d_weight5pcf,
modules/NPCF.h:      gpu_add_to_power5(d_fivepcf, d_weight5pcf,
modules/NPCF.h:  } else if (_gpumode == 2) {
modules/NPCF.h:    //execute alternate GPU kernel 
modules/NPCF.h:    if (_gpufloat) {
modules/NPCF.h:      gpu_add_to_power5_orig_float(f_fivepcf, f_weight5pcf,
modules/NPCF.h:    } else if (_gpumixed) {
modules/NPCF.h:      gpu_add_to_power5_orig_mixed(d_fivepcf, d_weight5pcf,
modules/NPCF.h:      gpu_add_to_power5_orig(d_fivepcf, d_weight5pcf,
modules/NPCF.h:  } else if (_gpumode == 0) {
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:  if (_gpumode == 0) generate_luts6 = false;
modules/NPCF.h:    //can only get here in _gpumode > 0
modules/NPCF.h:    //ALWAYS use GPUMODE = 1 for SIXPCF
modules/NPCF.h:    if (_gpumode > 0) {
modules/NPCF.h:      //use primary GPU kernel
modules/NPCF.h:      //generate LUTs here for primary GPU kernel
modules/NPCF.h:    gpu_allocate_luts6(&lut6_l1, &lut6_l2, &lut6_l12, &lut6_l3,
modules/NPCF.h:    if (_gpufloat) {
modules/NPCF.h:      gpu_allocate_weight6pcf(&f_weight6pcf, weight6pcf, size_w);
modules/NPCF.h:      gpu_allocate_sixpcf(&f_sixpcf, sixpcf, nell6*N6PCF);
modules/NPCF.h:      //normal mode - allocate GPU arrays and copy
modules/NPCF.h:      gpu_allocate_weight6pcf(&d_weight6pcf, weight6pcf, size_w);
modules/NPCF.h:      gpu_allocate_sixpcf(&d_sixpcf, sixpcf, nell6*N6PCF);
modules/NPCF.h:    if (_gpumode > 0) {
modules/NPCF.h:      //use primary GPU kernel
modules/NPCF.h:      //generate LUTs here for primary GPU kernel
modules/NPCF.h:                    lut6_n[iouter6] = n; //this is the starting n for this GPU thread
modules/NPCF.h:                    //GPU thread will then loop over ms
modules/NPCF.h:  if (_gpumode > 0) {
modules/NPCF.h:    //execute GPU kernel
modules/NPCF.h:    if (_gpufloat) {
modules/NPCF.h:      gpu_add_to_power6_float(f_sixpcf, f_weight6pcf,
modules/NPCF.h:    } else if (_gpumixed) {
modules/NPCF.h:      gpu_add_to_power6_mixed(d_sixpcf, d_weight6pcf,
modules/NPCF.h:      gpu_add_to_power6(d_sixpcf, d_weight6pcf,
modules/NPCF.h:  } else if (_gpumode == 0) {
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:    void free_gpu_memory() {
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:      gpu_free_luts3(lut3_i, lut3_j, lut3_ct);
modules/NPCF.h:      if (_gpufloat) gpu_free_memory3(f_threepcf, f_weight3pcf); else gpu_free_memory3(d_threepcf, d_weight3pcf);
modules/NPCF.h:      gpu_free_luts_discon1(lut_discon_ell, lut_discon_mm);
modules/NPCF.h:      gpu_free_luts_discon2(lut_discon_ell1, lut_discon_ell2, lut_discon_mm1, lut_discon_mm2, lut_discon_i, lut_discon_j);
modules/NPCF.h:      if (_gpufloat) gpu_free_memory_discon1(f_discon1_r, f_discon1_i, f_weightdiscon); else gpu_free_memory_discon1(d_discon1_r, d_discon1_i, d_weightdiscon);
modules/NPCF.h:      gpu_free_luts4(lut4_l1, lut4_l2, lut4_l3, lut4_odd, lut4_n,
modules/NPCF.h:      if (_gpufloat) gpu_free_memory4(f_fourpcf, f_weight4pcf); else gpu_free_memory4(d_fourpcf, d_weight4pcf);
modules/NPCF.h:      gpu_free_memory_m4(lut4_m1, lut4_m2);
modules/NPCF.h:      gpu_free_luts(lut5_l1, lut5_l2, lut5_l12, lut5_l3, lut5_l4, lut5_odd,
modules/NPCF.h:      if (_gpufloat) gpu_free_memory(f_fivepcf, f_weight5pcf); else gpu_free_memory(d_fivepcf, d_weight5pcf);
modules/NPCF.h:      gpu_free_memory_m(lut5_m1, lut5_m2, lut5_m3);
modules/NPCF.h:      gpu_free_luts6(lut6_l1, lut6_l2, lut6_l12, lut6_l3, lut6_l123, lut6_l4,
modules/NPCF.h:      if (_gpufloat) gpu_free_memory6(f_sixpcf, f_weight6pcf); else gpu_free_memory6(d_fivepcf, d_weight6pcf);
modules/NPCF.h:      gpu_free_memory_alms(!_gpufloat && !_gpumixed);
modules/NPCF.h:#ifdef GPU
modules/NPCF.h:      if (_gpufloat) {
modules/NPCF.h:      if (_gpufloat) {
modules/NPCF.h:      if (_gpufloat) {
modules/NPCF.h:      if (_gpufloat) {
modules/NPCF.h:      if (_gpufloat) {
modules/gpufuncs.cu:#include "gpufuncs.h"
modules/gpufuncs.cu:	cudaMemcpy(mult, (*p_mult), size*sizeof(double), cudaMemcpyDeviceToHost);
modules/gpufuncs.cu:	cudaMemcpy(mult_ct, (*p_mult_ct), size_ct*sizeof(int), cudaMemcpyDeviceToHost);
modules/gpufuncs.cu:void gpu_free_mult(double *mult, int *mult_ct){
modules/gpufuncs.cu:	cudaFree(mult);
modules/gpufuncs.cu:	cudaFree(mult_ct);
modules/gpufuncs.cu:void gpu_allocate_mult(double **p_mult, double *mult, int **p_mult_ct, int *mult_ct, int size, int size_ct) {
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_mult), size*sizeof(double));
modules/gpufuncs.cu:	cudaMallocManaged(&(*p_mult_ct), size_ct*sizeof(int));
modules/gpufuncs.cu:  cudaMalloc(&dx_array, max_length*sizeof(double));
modules/gpufuncs.cu:	cudaMalloc(&dy_array, max_length*sizeof(double));
modules/gpufuncs.cu:	cudaMalloc(&dz_array, max_length*sizeof(double));
modules/gpufuncs.cu:	cudaMalloc(&dw_array, max_length*sizeof(double));
modules/gpufuncs.cu:	cudaMalloc(&dbin_array, max_length*sizeof(int));
modules/gpufuncs.cu:	cudaMemcpy(dx_array, x_array, max_length*sizeof(double), cudaMemcpyHostToDevice);
modules/gpufuncs.cu:	cudaMemcpy(dy_array, y_array, max_length*sizeof(double), cudaMemcpyHostToDevice);
modules/gpufuncs.cu:	cudaMemcpy(dz_array, z_array, max_length*sizeof(double), cudaMemcpyHostToDevice);
modules/gpufuncs.cu:	cudaMemcpy(dw_array, w_array, max_length*sizeof(double), cudaMemcpyHostToDevice);
modules/gpufuncs.cu:	cudaMemcpy(dbin_array, bin_array, max_length*sizeof(int), cudaMemcpyHostToDevice);
modules/gpufuncs.cu:	// Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  cudaDeviceSynchronize();
modules/gpufuncs.cu:  cudaFree(dx_array);
modules/gpufuncs.cu:	cudaFree(dy_array);
modules/gpufuncs.cu:	cudaFree(dz_array);
modules/gpufuncs.cu:	cudaFree(dw_array);
modules/gpufuncs.cu:  cudaFree(dbin_array);
modules/gpufuncs.cu://  GPU KERNELS                                            /
modules/gpufuncs.cu:void gpu_add_to_power3_orig(double *d_threepcf, double *d_weight3pcf,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power3_orig_float(float *d_threepcf, float *d_weight3pcf,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power3_orig_mixed(double *d_threepcf, double *d_weight3pcf,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power_discon1_orig(double *d_discon1_r, double *d_discon1_i,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called from NPCF.h 
modules/gpufuncs.cu:void gpu_add_to_power_discon1_orig_float(float *f_discon1_r,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called from NPCF.h 
modules/gpufuncs.cu:void gpu_add_to_power_discon1_orig_mixed(double *d_discon1_r,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called from NPCF.h 
modules/gpufuncs.cu:void gpu_add_to_power_discon2_orig(double *d_discon2_r, double *d_discon2_i,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called from NPCF.h 
modules/gpufuncs.cu:void gpu_add_to_power_discon2_b(double *d_discon2_r, double *d_discon2_i,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power_discon2_final(double *d_discon2_r, double *d_discon2_i,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power_discon2_final_float(float *f_discon2_r,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power_discon2_final_mixed(double *d_discon2_r, double *d_discon2_i,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power4(double *d_fourpcf, double *d_weight4pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power4_float(float *d_fourpcf, float *d_weight4pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power4_mixed(double *d_fourpcf, double *d_weight4pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power4_orig(double *d_fourpcf, double *d_weight4pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power4_orig_float(float *d_fourpcf, float *d_weight4pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power4_orig_mixed(double *d_fourpcf, double *d_weight4pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power5(double *d_fivepcf, double *d_weight5pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power5_float(float *d_fivepcf, float *d_weight5pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power5_mixed(double *d_fivepcf, double *d_weight5pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power5_orig(double *d_fivepcf, double *d_weight5pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power5_orig_float(float *d_fivepcf, float *d_weight5pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power5_orig_mixed(double *d_fivepcf, double *d_weight5pcf, 
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu://run main kernel gpu == 1
modules/gpufuncs.cu:void gpu_add_to_power6(double *d_sixpcf, double *d_weight6pcf,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  cudaDeviceSynchronize();
modules/gpufuncs.cu:  //gpu_print_cuda_error();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power6_float(float *d_sixpcf, float *d_weight6pcf,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_add_to_power6_mixed(double *d_sixpcf, double *d_weight6pcf,
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //cudaFree will be called for alms from NPCF.h via gpu_free_memory_alms 
modules/gpufuncs.cu:void gpu_allocate_luts3(int **p_lut3_i, int **p_lut3_j, int **p_lut3_ct, int nouter) {
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut3_i), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut3_j), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut3_ct), nouter*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_threepcf(double **p_threepcf, double *threepcf, int size) {
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_threepcf), size*sizeof(double));
modules/gpufuncs.cu:void gpu_allocate_weight3pcf(double **p_weight3pcf, double *weight3pcf, int size) {
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weight3pcf), size*sizeof(double));
modules/gpufuncs.cu:void gpu_allocate_threepcf(float **p_threepcf, double *threepcf, int size) {
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_threepcf), size*sizeof(float));
modules/gpufuncs.cu:void gpu_allocate_weight3pcf(float **p_weight3pcf, double *weight3pcf, int size) {
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weight3pcf), size*sizeof(float));
modules/gpufuncs.cu:void gpu_free_luts3(int *lut3_i, int *lut3_j, int *lut3_ct) {
modules/gpufuncs.cu:  cudaFree(lut3_i);
modules/gpufuncs.cu:  cudaFree(lut3_j);
modules/gpufuncs.cu:  cudaFree(lut3_ct);
modules/gpufuncs.cu:void gpu_free_memory3(double *threepcf, double *weight3pcf) {
modules/gpufuncs.cu:  cudaFree(threepcf);
modules/gpufuncs.cu:  cudaFree(weight3pcf);
modules/gpufuncs.cu:void gpu_free_memory3(float *threepcf, float *weight3pcf) {
modules/gpufuncs.cu:  cudaFree(threepcf);
modules/gpufuncs.cu:  cudaFree(weight3pcf);
modules/gpufuncs.cu:void gpu_allocate_luts_discon1(int **p_lut_discon_ell, int **p_lut_discon_mm,
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut_discon_ell), size*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut_discon_mm), size*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_discon1(double **p_discon1_r, double **p_discon1_i,
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_discon1_r), size*sizeof(double));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_discon1_i), size*sizeof(double));
modules/gpufuncs.cu:void gpu_allocate_weightdiscon(double **p_weightdiscon, double *weightdiscon,
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weightdiscon), size*sizeof(double));
modules/gpufuncs.cu:void gpu_allocate_discon1(float **p_discon1_r, float **p_discon1_i,
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_discon1_r), size*sizeof(float));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_discon1_i), size*sizeof(float));
modules/gpufuncs.cu:void gpu_allocate_weightdiscon(float **p_weightdiscon, double *weightdiscon,
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weightdiscon), size*sizeof(float));
modules/gpufuncs.cu:void gpu_allocate_luts_discon2(int **p_lut_discon_ell1,
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut_discon_ell1), size*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut_discon_ell2), size*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut_discon_mm1), size*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut_discon_mm2), size*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_luts_discon2_inner(int **p_lut_discon_i,
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut_discon_i), size*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut_discon_j), size*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_discon2(double **p_discon2_r, double **p_discon2_i,
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_discon2_r), size*sizeof(double));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_discon2_i), size*sizeof(double));
modules/gpufuncs.cu:void gpu_allocate_discon2(float **p_discon2_r, float **p_discon2_i,
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_discon2_r), size*sizeof(float));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_discon2_i), size*sizeof(float));
modules/gpufuncs.cu:void gpu_free_luts_discon1(int *lut_discon_ell, int *lut_discon_mm) {
modules/gpufuncs.cu:  cudaFree(lut_discon_ell);
modules/gpufuncs.cu:  cudaFree(lut_discon_mm);
modules/gpufuncs.cu:void gpu_free_memory_discon1(double *d_discon1_r, double *d_discon1_i,
modules/gpufuncs.cu:  cudaFree(d_discon1_r);
modules/gpufuncs.cu:  cudaFree(d_discon1_i);
modules/gpufuncs.cu:  cudaFree(weightdiscon);
modules/gpufuncs.cu:void gpu_free_memory_discon1(float *f_discon1_r, float *f_discon1_i,
modules/gpufuncs.cu:  cudaFree(f_discon1_r);
modules/gpufuncs.cu:  cudaFree(f_discon1_i);
modules/gpufuncs.cu:  cudaFree(weightdiscon);
modules/gpufuncs.cu:void gpu_free_luts_discon2(int *lut_discon_ell1, int *lut_discon_ell2,
modules/gpufuncs.cu:  cudaFree(lut_discon_ell1);
modules/gpufuncs.cu:  cudaFree(lut_discon_ell2);
modules/gpufuncs.cu:  cudaFree(lut_discon_mm1);
modules/gpufuncs.cu:  cudaFree(lut_discon_mm2);
modules/gpufuncs.cu:  cudaFree(lut_discon_i);
modules/gpufuncs.cu:  cudaFree(lut_discon_j);
modules/gpufuncs.cu:void gpu_allocate_luts4(int **p_lut4_l1, int **p_lut4_l2, int **p_lut4_l3,
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_l1), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_l2), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_l3), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_odd), nouter*sizeof(bool));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_n), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_zeta), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_i), ninner*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_j), ninner*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_k), ninner*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_m_luts4(int **p_lut4_m1, int **p_lut4_m2, int nouter) {
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_m1), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut4_m2), nouter*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_fourpcf(double **p_fourpcf, double *fourpcf, int size) {
modules/gpufuncs.cu:  //cudaMalloc(&(*p_fourpcf), size*sizeof(double));
modules/gpufuncs.cu:  //cudaMemcpy((*p_fourpcf), fourpcf, size, cudaMemcpyHostToDevice);
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_fourpcf), size*sizeof(double));
modules/gpufuncs.cu:void gpu_allocate_weight4pcf(double **p_weight4pcf, double *weight4pcf, int size) {
modules/gpufuncs.cu:  //cudaMalloc(&(*p_weight4pcf), size*sizeof(double));
modules/gpufuncs.cu:  //cudaMemcpy((*p_weight4pcf), weight4pcf, size, cudaMemcpyHostToDevice);
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weight4pcf), size*sizeof(double));
modules/gpufuncs.cu:  cudaMemcpy(fourpcf, (*p_fourpcf), size*sizeof(double), cudaMemcpyDeviceToHost);
modules/gpufuncs.cu:void gpu_allocate_fourpcf(float **p_fourpcf, double *fourpcf, int size) {
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_fourpcf), size*sizeof(float));
modules/gpufuncs.cu:void gpu_allocate_weight4pcf(float **p_weight4pcf, double *weight4pcf, int size) {
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weight4pcf), size*sizeof(float));
modules/gpufuncs.cu:void gpu_free_luts4(int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
modules/gpufuncs.cu:  cudaFree(lut4_l1);
modules/gpufuncs.cu:  cudaFree(lut4_l2);
modules/gpufuncs.cu:  cudaFree(lut4_l3);
modules/gpufuncs.cu:  cudaFree(lut4_odd);
modules/gpufuncs.cu:  cudaFree(lut4_n);
modules/gpufuncs.cu:  cudaFree(lut4_zeta);
modules/gpufuncs.cu:  cudaFree(lut4_i);
modules/gpufuncs.cu:  cudaFree(lut4_j);
modules/gpufuncs.cu:  cudaFree(lut4_k);
modules/gpufuncs.cu:void gpu_free_memory4(double *fourpcf, double *weight4pcf) {
modules/gpufuncs.cu:  cudaFree(fourpcf);
modules/gpufuncs.cu:  cudaFree(weight4pcf);
modules/gpufuncs.cu:void gpu_free_memory4(float *fourpcf, float *weight4pcf) {
modules/gpufuncs.cu:  cudaFree(fourpcf);
modules/gpufuncs.cu:  cudaFree(weight4pcf);
modules/gpufuncs.cu:void gpu_free_memory_m4(int *lut4_m1, int *lut4_m2) {
modules/gpufuncs.cu:  cudaFree(lut4_m1);
modules/gpufuncs.cu:  cudaFree(lut4_m2);
modules/gpufuncs.cu:void gpu_allocate_luts(int **p_lut5_l1, int **p_lut5_l2, int **p_lut5_l12,
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_l1), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_l2), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_l12), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_l3), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_l4), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_odd), nouter*sizeof(bool));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_n), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_zeta), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_i), ninner*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_j), ninner*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_k), ninner*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_l), ninner*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_m_luts(int **p_lut5_m1, int **p_lut5_m2, int **p_lut5_m3,
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_m1), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_m2), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut5_m3), nouter*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_fivepcf(double **p_fivepcf, double *fivepcf, int size) {
modules/gpufuncs.cu:  //cudaMalloc(&(*p_fivepcf), size*sizeof(double));
modules/gpufuncs.cu:  //cudaMemcpy((*p_fivepcf), fivepcf, size, cudaMemcpyHostToDevice);
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_fivepcf), size*sizeof(double));
modules/gpufuncs.cu:void gpu_allocate_weight5pcf(double **p_weight5pcf, double *weight5pcf, int size) {
modules/gpufuncs.cu:  //cudaMalloc(&(*p_weight5pcf), size*sizeof(double));
modules/gpufuncs.cu:  //cudaMemcpy((*p_weight5pcf), weight5pcf, size, cudaMemcpyHostToDevice);
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weight5pcf), size*sizeof(double));
modules/gpufuncs.cu:  cudaMemcpy(fivepcf, (*p_fivepcf), size*sizeof(double), cudaMemcpyDeviceToHost);
modules/gpufuncs.cu:void gpu_allocate_fivepcf(float **p_fivepcf, double *fivepcf, int size) {
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_fivepcf), size*sizeof(float));
modules/gpufuncs.cu:void gpu_allocate_weight5pcf(float **p_weight5pcf, double *weight5pcf, int size) {
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weight5pcf), size*sizeof(float));
modules/gpufuncs.cu:void gpu_free_luts(int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
modules/gpufuncs.cu:  cudaFree(lut5_l1);
modules/gpufuncs.cu:  cudaFree(lut5_l2);
modules/gpufuncs.cu:  cudaFree(lut5_l12);
modules/gpufuncs.cu:  cudaFree(lut5_l3);
modules/gpufuncs.cu:  cudaFree(lut5_l4);
modules/gpufuncs.cu:  cudaFree(lut5_odd);
modules/gpufuncs.cu:  cudaFree(lut5_n);
modules/gpufuncs.cu:  cudaFree(lut5_zeta);
modules/gpufuncs.cu:  cudaFree(lut5_i);
modules/gpufuncs.cu:  cudaFree(lut5_j);
modules/gpufuncs.cu:  cudaFree(lut5_k);
modules/gpufuncs.cu:  cudaFree(lut5_l);
modules/gpufuncs.cu:void gpu_free_memory(double *fivepcf, double *weight5pcf) {
modules/gpufuncs.cu:  cudaFree(fivepcf);
modules/gpufuncs.cu:  cudaFree(weight5pcf);
modules/gpufuncs.cu:void gpu_free_memory(float *fivepcf, float *weight5pcf) {
modules/gpufuncs.cu:  cudaFree(fivepcf);
modules/gpufuncs.cu:  cudaFree(weight5pcf);
modules/gpufuncs.cu:void gpu_free_memory_m(int *lut5_m1, int *lut5_m2, int *lut5_m3) {
modules/gpufuncs.cu:  cudaFree(lut5_m1);
modules/gpufuncs.cu:  cudaFree(lut5_m2);
modules/gpufuncs.cu:  cudaFree(lut5_m3);
modules/gpufuncs.cu:void gpu_allocate_luts6(int **p_lut6_l1, int **p_lut6_l2, int **p_lut6_l12,
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_l1), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_l2), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_l12), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_l3), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_l123), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_l4), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_l5), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_odd), nouter*sizeof(bool));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_n), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_zeta), nouter*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_i), ninner*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_j), ninner*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_k), ninner*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_l), ninner*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_lut6_m), ninner*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_sixpcf(double **p_sixpcf, double *sixpcf, int size) {
modules/gpufuncs.cu:  //cudaMalloc(&(*p_sixpcf), size*sizeof(double));
modules/gpufuncs.cu:  //cudaMemcpy((*p_sixpcf), sixpcf, size, cudaMemcpyHostToDevice);
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_sixpcf), size*sizeof(double));
modules/gpufuncs.cu:void gpu_allocate_weight6pcf(double **p_weight6pcf, double *weight6pcf, int size) {
modules/gpufuncs.cu:  //cudaMalloc(&(*p_weight6pcf), size*sizeof(double));
modules/gpufuncs.cu:  //cudaMemcpy((*p_weight6pcf), weight6pcf, size, cudaMemcpyHostToDevice);
modules/gpufuncs.cu:  //use MallocManaged because of weirdness with cudaMemcpy not seeming to work with weight4pcf
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weight6pcf), size*sizeof(double));
modules/gpufuncs.cu:  cudaMemcpy(sixpcf, (*p_sixpcf), size*sizeof(double), cudaMemcpyDeviceToHost);
modules/gpufuncs.cu:void gpu_allocate_sixpcf(float **p_sixpcf, double *sixpcf, int size) {
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_sixpcf), size*sizeof(float));
modules/gpufuncs.cu:void gpu_allocate_weight6pcf(float **p_weight6pcf, double *weight6pcf, int size) {
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weight6pcf), size*sizeof(float));
modules/gpufuncs.cu:void gpu_free_luts6(int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
modules/gpufuncs.cu:  cudaFree(lut6_l1);
modules/gpufuncs.cu:  cudaFree(lut6_l2);
modules/gpufuncs.cu:  cudaFree(lut6_l12);
modules/gpufuncs.cu:  cudaFree(lut6_l3);
modules/gpufuncs.cu:  cudaFree(lut6_l123);
modules/gpufuncs.cu:  cudaFree(lut6_l4);
modules/gpufuncs.cu:  cudaFree(lut6_l5);
modules/gpufuncs.cu:  cudaFree(lut6_odd);
modules/gpufuncs.cu:  cudaFree(lut6_n);
modules/gpufuncs.cu:  cudaFree(lut6_zeta);
modules/gpufuncs.cu:  cudaFree(lut6_i);
modules/gpufuncs.cu:  cudaFree(lut6_j);
modules/gpufuncs.cu:  cudaFree(lut6_k);
modules/gpufuncs.cu:  cudaFree(lut6_l);
modules/gpufuncs.cu:  cudaFree(lut6_m);
modules/gpufuncs.cu:void gpu_free_memory6(double *sixpcf, double *weight6pcf) {
modules/gpufuncs.cu:  cudaFree(sixpcf);
modules/gpufuncs.cu:  cudaFree(weight6pcf);
modules/gpufuncs.cu:void gpu_free_memory6(float *sixpcf, float *weight6pcf) {
modules/gpufuncs.cu:  cudaFree(sixpcf);
modules/gpufuncs.cu:  cudaFree(weight6pcf);
modules/gpufuncs.cu:void gpu_allocate_alms(int np, int nb, int nlm, bool isDouble) {
modules/gpufuncs.cu:  //d_alm and d_almconj are already declared at top of gpufuncs.cu
modules/gpufuncs.cu:    cudaMallocManaged(&d_alm, np*nb*nlm*sizeof(thrust::complex<double>));
modules/gpufuncs.cu:    cudaMallocManaged(&d_almconj, np*nb*nlm*sizeof(thrust::complex<double>));
modules/gpufuncs.cu:    cudaMallocManaged(&f_alm, np*nb*nlm*sizeof(thrust::complex<float>));
modules/gpufuncs.cu:    cudaMallocManaged(&f_almconj, np*nb*nlm*sizeof(thrust::complex<float>));
modules/gpufuncs.cu:void gpu_compute_alms(int *map, double *m, int nbin, int nlm, int maxp,
modules/gpufuncs.cu:  cudaMallocManaged(&d_map, size_map);
modules/gpufuncs.cu:  //copy map to GPU memory
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:void gpu_compute_alms_float(int *map, double *m, int nbin, int nlm, int maxp,
modules/gpufuncs.cu:  cudaMallocManaged(&d_map, size_map);
modules/gpufuncs.cu:  //copy map to GPU memory
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:void gpu_free_memory_alms(bool isDouble) {
modules/gpufuncs.cu:    cudaFree(d_alm);
modules/gpufuncs.cu:    cudaFree(d_almconj);
modules/gpufuncs.cu:    cudaFree(f_alm);
modules/gpufuncs.cu:    cudaFree(f_almconj);
modules/gpufuncs.cu:void gpu_allocate_multipoles(double **p_msave, int **p_csave,
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_msave), nmult*nbin*np*sizeof(double));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_csave), np*nbin*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_pnum), nmax*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_spnum), nmax*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_snp), nmax*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_sc), nmax*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_multipoles_fast(double **p_msave, int **p_csave,
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_msave), nmult*nbin*maxp*sizeof(double));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_csave), maxp*nbin*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_start_list), nc*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_np_list), nc*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_cellnums), np*sizeof(int));
modules/gpufuncs.cu:void gpu_allocate_particle_arrays(double **p_posx, double **p_posy, double **p_posz, double **p_weights, int np) {
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_posx), np*sizeof(double));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_posy), np*sizeof(double));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_posz), np*sizeof(double));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_weights), np*sizeof(double));
modules/gpufuncs.cu:void gpu_allocate_pair_arrays(double **p_x0i, double **p_x2i, int nbin) {
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_x0i), nbin*sizeof(double));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_x2i), nbin*sizeof(double));
modules/gpufuncs.cu:void gpu_allocate_periodic(int **p_delta_x, int **p_delta_y, int ** p_delta_z, int nmax) {
modules/gpufuncs.cu:  // Allocate Unified Memory – accessible from CPU or GPU
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_delta_x), nmax*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_delta_y), nmax*sizeof(int));
modules/gpufuncs.cu:  cudaMallocManaged(&(*p_delta_z), nmax*sizeof(int));
modules/gpufuncs.cu:void free_gpu_multipole_arrays(double *msave, int *csave,
modules/gpufuncs.cu:  cudaFree(msave);
modules/gpufuncs.cu:  cudaFree(csave);
modules/gpufuncs.cu:  cudaFree(pnum);
modules/gpufuncs.cu:  cudaFree(spnum);
modules/gpufuncs.cu:  cudaFree(snp);
modules/gpufuncs.cu:  cudaFree(sc);
modules/gpufuncs.cu:  cudaFree(posx);
modules/gpufuncs.cu:  cudaFree(posy);
modules/gpufuncs.cu:  cudaFree(posz);
modules/gpufuncs.cu:  cudaFree(weights);
modules/gpufuncs.cu:  cudaFree(x0i);
modules/gpufuncs.cu:  cudaFree(x2i);
modules/gpufuncs.cu:void free_gpu_periodic_arrays(int *delta_x, int *delta_y, int *delta_z) {
modules/gpufuncs.cu:  cudaFree(delta_x);
modules/gpufuncs.cu:  cudaFree(delta_y);
modules/gpufuncs.cu:  cudaFree(delta_z);
modules/gpufuncs.cu:// gpufloat (default FALSE) = DOUBLE vs float
modules/gpufuncs.cu:void gpu_add_pairs_only(double *posx, double *posy, double *posz, double *w,
modules/gpufuncs.cu:	int n, int nbin, float rmin, float rmax, bool shared, bool gpufloat) {
modules/gpufuncs.cu:  if (gpufloat) {
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:void gpu_add_pairs_only_periodic(double *posx, double *posy, double *posz,
modules/gpufuncs.cu:	bool gpufloat) {
modules/gpufuncs.cu:  if (gpufloat) {
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:void gpu_add_pairs_only_fast(double *posx, double *posy, double *posz,
modules/gpufuncs.cu:        int pstart, bool shared, bool gpufloat) {
modules/gpufuncs.cu:  if (gpufloat) {
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:void gpu_add_pairs_only_periodic_fast(double *posx, double *posy, double *posz,
modules/gpufuncs.cu:	double cellsize, bool shared, bool gpufloat) { 
modules/gpufuncs.cu:  if (gpufloat) {
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:void gpu_add_pairs_and_multipoles(double *m, double *posx, double *posy,
modules/gpufuncs.cu:	bool shared, bool gpufloat) {
modules/gpufuncs.cu:  if (gpufloat) {
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:void gpu_add_pairs_and_multipoles_fast(double *m, double *posx, double *posy,
modules/gpufuncs.cu:	int pstart, bool shared, bool gpufloat) {
modules/gpufuncs.cu:  if (gpufloat) {
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:void gpu_add_pairs_and_multipoles_periodic(double *m, double *posx,
modules/gpufuncs.cu:	bool shared, bool gpufloat) {
modules/gpufuncs.cu:  if (gpufloat) {
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //gpu_print_cuda_error();
modules/gpufuncs.cu:void gpu_add_pairs_and_multipoles_periodic_fast(double *m, double *posx,
modules/gpufuncs.cu:	bool shared, bool gpufloat) {
modules/gpufuncs.cu:  if (gpufloat) {
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  //cudaDeviceSynchronize();
modules/gpufuncs.cu:  //gpu_print_cuda_error();
modules/gpufuncs.cu:void gpu_device_synchronize() {
modules/gpufuncs.cu:  // Wait for GPU to finish before accessing on host
modules/gpufuncs.cu:  cudaDeviceSynchronize();
modules/gpufuncs.cu:void gpu_print_cuda_error() {
modules/gpufuncs.cu:        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
modules/gpufuncs.cu:        if ( cudaSuccess != cuda_status ){
modules/gpufuncs.cu:            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
modules/gpufuncs.cu:        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
modules/gpufuncs.cu:cudaError_t err = cudaGetLastError();
modules/gpufuncs.cu:printf("CUDA Error: %s\n", cudaGetErrorString(err));
modules/gpufuncs.h:void gpu_free_mult(double *mult, int *mult_ct);
modules/gpufuncs.h:void gpu_allocate_mult(double **p_mult, double *mult, int **p_mult_ct, int *mult_ct, int size, int size_ct);
modules/gpufuncs.h:void gpu_add_to_power3_orig(double *d_threepcf, double *d_weight3pcf,
modules/gpufuncs.h:void gpu_add_to_power3_orig_float(float *d_threepcf, float *d_weight3pcf,
modules/gpufuncs.h:void gpu_add_to_power3_orig_mixed(double *d_threepcf, double *d_weight3pcf,
modules/gpufuncs.h:void gpu_add_to_power_discon1_orig(double *d_discon1_r, double *d_discon1_i,
modules/gpufuncs.h:void gpu_add_to_power_discon1_orig_float(float *f_discon1_r,
modules/gpufuncs.h:void gpu_add_to_power_discon1_orig_mixed(double *d_discon1_r,
modules/gpufuncs.h:void gpu_add_to_power_discon2_orig(double *d_discon2_r, double *d_discon2_i,
modules/gpufuncs.h:void gpu_add_to_power_discon2_b(double *d_discon2_r, double *d_discon2_i,
modules/gpufuncs.h:void gpu_add_to_power_discon2_final(double *d_discon2_r, double *d_discon2_i,
modules/gpufuncs.h:void gpu_add_to_power_discon2_final_float(float *f_discon2_r, float *f_discon2_i,
modules/gpufuncs.h:void gpu_add_to_power_discon2_final_mixed(double *d_discon2_r, double *d_discon2_i,
modules/gpufuncs.h://run main kernel gpu == 1
modules/gpufuncs.h:void gpu_add_to_power4(double *d_fourpcf, double *d_weight4pcf, 
modules/gpufuncs.h:void gpu_add_to_power4_float(float *d_fourpcf, float *d_weight4pcf, 
modules/gpufuncs.h:void gpu_add_to_power4_mixed(double *d_fourpcf, double *d_weight4pcf, 
modules/gpufuncs.h:void gpu_add_to_power4_orig(double *d_fourpcf, double *d_weight4pcf, 
modules/gpufuncs.h:void gpu_add_to_power4_orig_float(float *d_fourpcf, float *d_weight4pcf, 
modules/gpufuncs.h:void gpu_add_to_power4_orig_mixed(double *d_fourpcf, double *d_weight4pcf, 
modules/gpufuncs.h://run main kernel gpu == 1
modules/gpufuncs.h:void gpu_add_to_power5(double *d_fivepcf, double *d_weight5pcf, 
modules/gpufuncs.h:void gpu_add_to_power5_float(float *d_fivepcf, float *d_weight5pcf, 
modules/gpufuncs.h:void gpu_add_to_power5_mixed(double *d_fivepcf, double *d_weight5pcf, 
modules/gpufuncs.h:void gpu_add_to_power5_orig(double *d_fivepcf, double *d_weight5pcf, 
modules/gpufuncs.h:void gpu_add_to_power5_orig_float(float *d_fivepcf, float *d_weight5pcf, 
modules/gpufuncs.h:void gpu_add_to_power5_orig_mixed(double *d_fivepcf, double *d_weight5pcf, 
modules/gpufuncs.h://run main kernel gpu == 1
modules/gpufuncs.h:void gpu_add_to_power6(double *d_sixpcf, double *d_weight6pcf,
modules/gpufuncs.h:void gpu_add_to_power6_float(float *d_sixpcf, float *d_weight6pcf,
modules/gpufuncs.h:void gpu_add_to_power6_mixed(double *d_sixpcf, double *d_weight6pcf,
modules/gpufuncs.h:void gpu_allocate_luts3(int **p_lut3_i, int **p_lut3_j, int **p_lut3_ct, int nouter);
modules/gpufuncs.h:void gpu_allocate_threepcf(double **p_threepcf, double *threepcf, int size);
modules/gpufuncs.h:void gpu_allocate_weight3pcf(double **p_weight3pcf, double *weight3pcf, int size);
modules/gpufuncs.h:void gpu_allocate_threepcf(float **p_threepcf, double *threepcf, int size);
modules/gpufuncs.h:void gpu_allocate_weight3pcf(float **p_weight3pcf, double *weight3pcf, int size);
modules/gpufuncs.h:void gpu_free_luts3(int *lut3_i, int *lut3_j, int *lut3_ct);
modules/gpufuncs.h:void gpu_free_memory3(double *threepcf, double *weight3pcf);
modules/gpufuncs.h:void gpu_free_memory3(float *threepcf, float *weight3pcf);
modules/gpufuncs.h:void gpu_allocate_luts_discon1(int **p_lut_discon_ell, int **p_lut_discon_mm,
modules/gpufuncs.h:void gpu_allocate_discon1(double **p_discon1_r, double **p_discon1_i,
modules/gpufuncs.h:void gpu_allocate_weightdiscon(double **p_weightdiscon, double *weightdiscon,
modules/gpufuncs.h:void gpu_allocate_discon1(float **p_discon1_r, float **p_discon1_i,
modules/gpufuncs.h:void gpu_allocate_weightdiscon(float **p_weightdiscon, double *weightdiscon,
modules/gpufuncs.h:void gpu_allocate_luts_discon2(int **p_lut_discon_ell1,
modules/gpufuncs.h:void gpu_allocate_luts_discon2_inner(int **p_lut_discon_i,
modules/gpufuncs.h:void gpu_allocate_discon2(double **p_discon2_r, double **p_discon2_i,
modules/gpufuncs.h:void gpu_allocate_discon2(float **p_discon2_r, float **p_discon2_i,
modules/gpufuncs.h:void gpu_free_luts_discon1(int *lut_discon_ell, int *lut_discon_mm);
modules/gpufuncs.h:void gpu_free_memory_discon1(double *d_discon1_r, double *d_discon1_i,
modules/gpufuncs.h:void gpu_free_memory_discon1(float *f_discon1_r, float *f_discon1_i,
modules/gpufuncs.h:void gpu_free_luts_discon2(int *lut_discon_ell1, int *lut_discon_ell2,
modules/gpufuncs.h:void gpu_allocate_luts4(int **p_lut4_l1, int **p_lut4_l2, int **p_lut4_l3,
modules/gpufuncs.h:void gpu_allocate_m_luts4(int **p_lut4_m1, int **p_lut4_m2, int nouter);
modules/gpufuncs.h:void gpu_allocate_fourpcf(double **p_fourpcf, double *fourpcf, int size);
modules/gpufuncs.h:void gpu_allocate_fourpcf(float **p_fourpcf, double *fourpcf, int size);
modules/gpufuncs.h:void gpu_allocate_weight4pcf(double **p_weight4pcf, double *weight4pcf, int size);
modules/gpufuncs.h:void gpu_allocate_weight4pcf(float **p_weight4pcf, double *weight4pcf, int size);
modules/gpufuncs.h:void gpu_free_luts4(int *lut4_l1, int *lut4_l2, int *lut4_l3, bool *lut4_odd,
modules/gpufuncs.h:void gpu_free_memory4(double *fourpcf, double *weight4pcf);
modules/gpufuncs.h:void gpu_free_memory4(float *fourpcf, float *weight4pcf);
modules/gpufuncs.h:void gpu_free_memory_m4(int *lut4_m1, int *lut4_m2);
modules/gpufuncs.h:void gpu_allocate_luts(int **p_lut5_l1, int **p_lut5_l2, int **p_lut5_l12,
modules/gpufuncs.h:void gpu_allocate_m_luts(int **p_lut5_m1, int **p_lut5_m2, int **p_lut5_m3, int nouter);
modules/gpufuncs.h:void gpu_allocate_fivepcf(double **p_fivepcf, double *fivepcf, int size);
modules/gpufuncs.h:void gpu_allocate_fivepcf(float **p_fivepcf, double *fivepcf, int size);
modules/gpufuncs.h:void gpu_allocate_weight5pcf(double **p_weight5pcf, double *weight5pcf, int size);
modules/gpufuncs.h:void gpu_allocate_weight5pcf(float **p_weight5pcf, double *weight5pcf, int size);
modules/gpufuncs.h:void gpu_free_luts(int *lut5_l1, int *lut5_l2, int *lut5_l12, int *lut5_l3,
modules/gpufuncs.h:void gpu_free_memory(double *fivepcf, double *weight5pcf);
modules/gpufuncs.h:void gpu_free_memory(float *fivepcf, float *weight5pcf);
modules/gpufuncs.h:void gpu_free_memory_m(int *lut5_m1, int *lut5_m2, int *lut5_m3);
modules/gpufuncs.h:void gpu_allocate_luts6(int **p_lut6_l1, int **p_lut6_l2, int **p_lut6_l12,
modules/gpufuncs.h:void gpu_allocate_sixpcf(double **p_sixpcf, double *sixpcf, int size);
modules/gpufuncs.h:void gpu_allocate_sixpcf(float **p_sixpcf, double *sixpcf, int size);
modules/gpufuncs.h:void gpu_allocate_weight6pcf(double **p_weight6pcf, double *weight6pcf, int size);
modules/gpufuncs.h:void gpu_allocate_weight6pcf(float **p_weight6pcf, double *weight6pcf, int size);
modules/gpufuncs.h:void gpu_free_luts6(int *lut6_l1, int *lut6_l2, int *lut6_l12, int *lut6_l3,
modules/gpufuncs.h:void gpu_free_memory6(double *sixpcf, double *weight6pcf);
modules/gpufuncs.h:void gpu_free_memory6(float *sixpcf, float *weight6pcf);
modules/gpufuncs.h:void gpu_allocate_alms(int np, int nb, int nlm, bool isDouble);
modules/gpufuncs.h:void gpu_compute_alms(int *map, double *m, int nbin, int nlm, int maxp, int order, int mapdim, int nmult);
modules/gpufuncs.h:void gpu_compute_alms_float(int *map, double *m, int nbin, int nlm, int maxp, int order, int mapdim, int nmult);
modules/gpufuncs.h:void gpu_free_memory_alms(bool isDouble);
modules/gpufuncs.h:void gpu_allocate_multipoles(double **p_msave, int **p_csave,
modules/gpufuncs.h:void gpu_allocate_multipoles_fast(double **p_msave, int **p_csave,
modules/gpufuncs.h:void gpu_allocate_particle_arrays(double **p_posx, double **p_posy, double **p_posz, double **p_weights, int np);
modules/gpufuncs.h:void gpu_allocate_pair_arrays(double **p_x0i, double **p_x2i, int nbin);
modules/gpufuncs.h:void gpu_allocate_periodic(int **p_delta_x, int **p_delta_y, int ** p_delta_z, int nmax);
modules/gpufuncs.h:void free_gpu_multipole_arrays(double *msave, int *csave,
modules/gpufuncs.h:void free_gpu_periodic_arrays(int *delta_x, int *delta_y, int *delta_z);
modules/gpufuncs.h:// gpufloat (default FALSE) = DOUBLE vs float
modules/gpufuncs.h:void gpu_add_pairs_only(double *posx, double *posy, double *posz, double *w,
modules/gpufuncs.h:        int n, int nbin, float rmin, float rmax, bool shared, bool gpufloat);
modules/gpufuncs.h:void gpu_add_pairs_only_periodic(double *posx, double *posy, double *posz,
modules/gpufuncs.h:        bool gpufloat);
modules/gpufuncs.h:void gpu_add_pairs_only_fast(double *posx, double *posy, double *posz,
modules/gpufuncs.h:        int pstart, bool shared, bool gpufloat);
modules/gpufuncs.h:void gpu_add_pairs_only_periodic_fast(double *posx, double *posy, double *posz,
modules/gpufuncs.h:	double cellsize, bool shared, bool gpufloat);
modules/gpufuncs.h:void gpu_add_pairs_and_multipoles(double *m, double *posx, double *posy,
modules/gpufuncs.h:	bool shared, bool gpufloat);
modules/gpufuncs.h:void gpu_add_pairs_and_multipoles_periodic(double *m, double *posx,
modules/gpufuncs.h:	bool gpufloat);
modules/gpufuncs.h:void gpu_add_pairs_and_multipoles_fast(double *m, double *posx, double *posy,
modules/gpufuncs.h:        int ncells, int maxsep, int pstart, bool shared, bool gpufloat);
modules/gpufuncs.h:void gpu_add_pairs_and_multipoles_periodic_fast(double *m, double *posx,
modules/gpufuncs.h:        bool shared, bool gpufloat);
modules/gpufuncs.h:// call cudaDeviceSynchronize
modules/gpufuncs.h:void gpu_device_synchronize();
modules/gpufuncs.h:void gpu_print_cuda_error();
encore.cpp:// IF NBIN is changed IT MUST ALSO BE UPDATED IN modules/gpufuncs.h!
encore.cpp://1 = GPU primary kernel
encore.cpp:short _gpumode = 0;
encore.cpp:short _gpump = 2;
encore.cpp:bool _gpufloat = false;
encore.cpp:bool _gpumixed = false;
encore.cpp:    fprintf(stderr, "    -gpu: GPU mode => 0 = CPU, 1 = GPU, 2+ = GPU alternate kernel. This requires compilation in GPU mode.\n");
encore.cpp:    fprintf(stderr, "    -float: GPU mode => use floats to speed up\n");
encore.cpp:    fprintf(stderr, "    -mixed: GPU mode => use mixed precision - alms are floats, accumulation is doubles\n");
encore.cpp:    fprintf(stderr, "    -global: GPU mode => use global memory always.  Default is to offload some calcs to shared memory.\n");
encore.cpp:    fprintf(stderr, "             Shared is faster on HPC GPUs but global is faster on some consumer grade GPUs.\n");
encore.cpp:    fprintf(stderr, "    -2pcf: GPU mode => only calculate 2PCF and exit\n");
encore.cpp:#ifdef GPU
encore.cpp:	else if (!strcmp(argv[i],"-gpu")) _gpumode = atoi(argv[++i]);
encore.cpp:        else if (!strcmp(argv[i],"-float")) _gpufloat = true;
encore.cpp:        else if (!strcmp(argv[i],"-mixed")) _gpumixed = true;
encore.cpp:	else if (!strcmp(argv[i],"-mpkernel")) _gpump = atoi(argv[++i]);
encore.cpp:    if (_gpumode > 0) nthreads = 1;

```
