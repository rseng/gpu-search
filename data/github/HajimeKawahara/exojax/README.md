# https://github.com/HajimeKawahara/exojax

```console
tests/unittests/spec/modit/modit_spectrum_test.py:    #mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/unittests/spec/modit/modit_spectrum_test.py:    #mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/unittests/spec/transmission/transmission_pure_absorption_test.py:# this test code requires gpu 
tests/endtoend/reverse/reverse_premodit_blackjax.py:mdb = MdbExomol(".database/CH4/12C-1H4/YT10to10/", nurange=nu_grid, gpu_transfer=False)
tests/endtoend/reverse/reverse_modit_hitemp.py:mdbCO = api.MdbHitemp(".database/CO", nus, crit=1.0e-30, gpu_transfer=True)
tests/endtoend/reverse/reverse_lpf.py:    ".database/CO/12C-16O/Li2015", nu_grid, crit=1.0e-46, gpu_transfer=True
tests/endtoend/reverse/reverse_premodit.py:mdb = MdbExomol(".database/CH4/12C-1H4/YT10to10/", nurange=nu_grid, gpu_transfer=False)
tests/endtoend/reverse/reverse_modit.py:    gpu_transfer=True,
tests/endtoend/reverse/reverse_precompute_grid.py:                gpu_transfer=False)
tests/endtoend/reverse/reverse_premodit_transmission.py:mdb = MdbExomol(".database/CH4/12C-1H4/YT10to10/", nurange=nu_grid, gpu_transfer=False)
tests/endtoend/jaxopt/optimize_spectrum_JAXopt_test.py:        ".database/CO/12C-16O/Li2015", nus, crit=1.0e-46, gpu_transfer=True
tests/endtoend/metals/VALD_MODIT_test.py:        ".database/H2O/1H2-16O/POKAZATEL", nus, crit=1e-50, gpu_transfer=True
tests/endtoend/metals/VALD_MODIT_test.py:        ".database/TiO/48Ti-16O/Toto", nus, crit=1e-50, gpu_transfer=True
tests/endtoend/metals/VALD_MODIT_test.py:    mdbOH = api.MdbExomol(".database/OH/16O-1H/MoLLIST", nus, gpu_transfer=True)
tests/endtoend/metals/VALD_MODIT_test.py:    mdbFeH = api.MdbExomol(".database/FeH/56Fe-1H/MoLLIST", nus, gpu_transfer=True)
tests/benchmark/fig/read_dat.py:dat=pd.read_csv("data/gpu.dat",delimiter=",")
tests/benchmark/fig/read_dat.py:dat=pd.read_csv("data/gpu2.dat",delimiter=",")
tests/benchmark/lpf_bm_wide.py:    #test1 (gpu.dat)
tests/benchmark/lpf_bm_wide.py:    #test2 (gpu2.dat)
tests/benchmark/lpf_bm_wide.py:    Sij_gpu = jnp.array(Sij)
tests/benchmark/lpf_bm_wide.py:    sigmaD_gpu = jnp.array(sigmaD)
tests/benchmark/lpf_bm_wide.py:    gammaL_gpu = jnp.array(gammaL)
tests/benchmark/lpf_bm_wide.py:        xsv = xsvector(numatrix, sigmaD_gpu, gammaL_gpu, Sij_gpu)
tests/integration/api/api_premodit_to_direct.py:mdb = api.MdbHitemp(".database/CO/05_HITEMP2019",nus,crit=1.e-30,Ttyp=1000.,gpu_transfer=True,isotope=1)
tests/integration/api/download_h2s.py:mdbH2S = MdbExomol(".database/H2S/1H2-32S/AYT2", nurange=nu_grid, gpu_transfer=False)
tests/integration/api/download_h2s.py:#mdbCO = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid, gpu_transfer=False)
tests/integration/premodit/line_strength_comparison_exomol_water.py:mdb = api.MdbExomol('.database/H2O/1H2-16O/POKAZATEL', nus, gpu_transfer=True)
tests/integration/premodit/line_strength_comparison_hitemp.py:mdb = api.MdbHitemp('CO', nus, gpu_transfer=True, isotope=1)
tests/integration/premodit_lpf/CH4_Gascellmodel_2408rev_test.py:    gpu_transfer=False,  # Trueだと計算速度低下
tests/integration/comparison/twostream/comparison_lart_fluxadd.py:    #mdb = api.MdbExomol('.database/CO/12C-1edt mru 6O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/integration/comparison/twostream/comparison_lart_fluxadd.py:    #mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/integration/comparison/premodit/comp_lsd_test.py:    mdbCH4 = api.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nu_grid, gpu_transfer=False)
tests/integration/comparison/transmission/comparison_with_kawashima_transmission.py:    # mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/integration/comparison/transmission/comparison_with_kawashima_transmission.py:                    gpu_transfer=True,
tests/integration/comparison/transmission/comparison_with_kawashima_transmission.py:    # mdb = MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/integration/unittests_long/twostream/twostream_spectrum_test.py:    # mdb = api.MdbExomol('.database/CO/12C-1edt mru 6O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/integration/unittests_long/twostream/twostream_spectrum_test.py:    # mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/integration/unittests_long/twostream/twostream_spectrum_test.py:    # mdb = api.MdbExomol('.database/CO/12C-1edt mru 6O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/integration/unittests_long/twostream/twostream_spectrum_test.py:    # mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/integration/unittests_long/twostream/twostream_reflection_test.py:    # mdb = api.MdbExomol('.database/CO/12C-1edt mru 6O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/integration/unittests_long/twostream/twostream_reflection_test.py:    # mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/integration/unittests_long/twostream/twostream_reflection_test.py:    # mdb = api.MdbExomol('.database/CO/12C-1edt mru 6O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/integration/unittests_long/twostream/twostream_reflection_test.py:    # mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/integration/unittests_long/premodit/premodit_transmission_test.py:    #mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/integration/unittests_long/premodit/premodit_transmission_test.py:    #mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/integration/unittests_long/premodit/premodit_spectrum_test.py:    #mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/integration/unittests_long/premodit/premodit_spectrum_test.py:    #mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/integration/unittests_long/premodit/premodit_spectrum_test.py:    #mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/integration/unittests_long/premodit/premodit_spectrum_test.py:    #mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/integration/unittests_long/lpf/lpf_test.py:    #mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
tests/integration/unittests_long/lpf/lpf_test.py:    #mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
tests/integration/unittests_long/transmission/transmission_grad_test.py:    mdb = MdbHitran("CO", nu_grid, gpu_transfer=True, isotope=1)
documents/userguide/api.rst:    ['dbtype', 'path', 'exact_molecule_name', 'database', 'bkgdatm', 'Tref', 'gpu_transfer', 'Ttyp', 'broadf', 'simple_molecule_name', 'molmass', 'skip_optional_data', 'activation', 'name', 'molecule', 'local_databases', 'extra_params', 'downloadable', 'format', 'engine', 'tempdir', 'ds', 'verbose', 'parallel', 'nJobs', 'batch_size', 'minimum_nfiles', 'crit', 'margin', 'nurange', 'wmin', 'wmax', 'states_file', 'pf_file', 'def_file', 'broad_file', 'isotope_fullname', 'n_Texp_def', 'alpha_ref_def', 'gQT', 'T_gQT', 'QTref', 'trans_file', 'num_tag', 'elower_max', 'QTtyp', 'df_load_mask', 'A', 'nu_lines', 'elower', 'jlower', 'jupper', 'line_strength_ref', 'gpp', 'alpha_ref', 'n_Texp', 'gamma_natural', 'dev_nu_lines', 'logsij0']
documents/userguide/api.rst:Some opacity calculator (currently only PreMODIT) does not use some arrays on a GPU device. 
documents/userguide/api.rst:Switch gpu_transfer off in this case. Then, we can save the use of the device memory.
documents/userguide/api.rst:	>>> mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=[4200.0, 4300.0], gpu_transfer=False)
documents/userguide/api.rst:This table is a short summary of the line information. "on" means gpu_transfer = True, off corresponds to False. 
documents/userguide/history.rst:- more GPU memory saved method in PreMODIT (so called diffmode) #332
documents/userguide/modit.rst:With an increase in the number of lines of :math:`N_l`, the direct LPF tends to be intractable even when using GPUs, in particular for :math:`N_l \gtrsim 10^3`. MODIT is a modified version of `Discrete Integral Transform <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ for rapid spectral synthesis, originally proposed by D.C.M van den Bekerom and E. Pannier. The modifications are as follows:
documents/userguide/benchmark.rst:Here is the benchmark for LPF (Direct) vs MODIT as a function of the number of the lines. Because the computation time depends on both the actual computation on the GPU device and data transfer from the main memory to the memory on the GPU, we show two different cases with and without data transfer to the GPU device for the direct LPF. The HMC−NUT fitting corresponds to the latter case because it reuses the values in the GPU memory many times. The computation time with data transfer was approximately ten times slower than that without transfer. For the direct LPF, the computation time is approximately proportional to the number of lines and the wavenumber. The mean computation time without transfer was ~0.1 ns per line per wavenumber bin using NVIDIA/DGX A100. The MODIT algorithm exhibits almost no dependence on the number of lines until Nline ~100,000 and converges to a linear dependence for larger Nlines. This trend is consistent with the results of van den Bekerom & Pannier (2021). See Figures 3 and 11 in their paper. Notably, MODIT does not depend significantly on the number of wavenumber bins. For a large number of lines, the calculation of the lineshape density Sjk takes so much longer than the convolution step that it dominates the computation time. For a small number of lines, this is probably because batch computation tends to be advantageous for FFT in GPU computations.
documents/userguide/memorysetting.rst:Frequent device memory overflows (memory on GPU) occur when modeling a wide wavelength range with high wavelength resolution. 
documents/userguide/memorysetting.rst:First, read the following webpage on JAX gpu memory allocation:
documents/userguide/memorysetting.rst:https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
documents/userguide/memorysetting.rst:    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
documents/userguide/memorysetting.rst:    from cuda import cudart
documents/userguide/memorysetting.rst:    cudart.cudaMemGetInfo()
documents/userguide/installation.rst:However, to take advantage of the power of JAX, you need to prepare a GPU environment (if you have). For this, jaxlib need to be linked.
documents/userguide/installation.rst:You should check the cuda version of your environment as
documents/userguide/installation.rst:	pip install -U "jax[cuda12]"
documents/tutorials/Transmission_beta.rst:        mdb_iso1 = MdbHitran("CO", nu_grid, gpu_transfer=False, isotope=1)
documents/tutorials/Transmission_beta.rst:        mdb_iso2 = MdbHitran("CO", nu_grid, gpu_transfer=False, isotope=2)
documents/tutorials/Transmission_beta.rst:        mdb_iso4 = MdbHitran("CO", nu_grid, gpu_transfer=False, isotope=4)
documents/tutorials/opacity.rst:    mdbCO = api.MdbHitran('CO', nu_grid, isotope=isotope, gpu_transfer=True)
documents/tutorials/Comparing_HITEMP_and_ExoMol.rst:        "CO", nus, isotope=1, gpu_transfer=True
documents/tutorials/Comparing_HITEMP_and_ExoMol.rst:    mdbCO_Li2015 = api.MdbExomol(emf, nus, gpu_transfer=True)
documents/tutorials/branch.rst:    2024-10-02 10:29:55.853116: W external/xla/xla/service/gpu/nvptx_compiler.cc:765] The NVIDIA driver's CUDA version is 12.2 which is older than the ptxas CUDA version (12.6.20). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
documents/tutorials/branch.rst:    mdb = MdbExomol("/home/kawahara/CO/12C-16O/Li2015",nurange=nus, crit=1.e-25, gpu_transfer=True)
documents/tutorials/branch.rst:    mdb = MdbExomol("/home/kawahara/CO/12C-16O/Li2015",nurange=nus, crit=1.e-30,gpu_transfer=True)
documents/tutorials/Reverse_modeling.rst:model and a better GPU, such as V100 or A100. Read the next section in
documents/tutorials/Forward_modeling_using_PreMODIT.rst:gpu_transfer=False can save the device memory use.
documents/tutorials/Forward_modeling_using_PreMODIT.rst:    mdbCO=api.MdbExomol('.database/CO/12C-16O/Li2015',nus,gpu_transfer=False)
documents/tutorials/Forward_modeling_using_PreMODIT.rst:    #Reload mdb beacuse we need gpu_transfer for LPF. This makes big difference in the device memory use. 
documents/tutorials/Forward_modeling_using_PreMODIT.rst:    mdbCO=api.MdbExomol('.database/CO/12C-16O/Li2015',nus, gpu_transfer=True)
documents/tutorials/Reverse_modeling_for_methane_using_PreMODIT.rst:can use gpu_transfer = False option, which siginificantly reduce divice
documents/tutorials/Reverse_modeling_for_methane_using_PreMODIT.rst:    mdbCH4=api.MdbExomol('.database/CH4/12C-1H4/YT10to10/',nus,gpu_transfer=False)
documents/tutorials/elower_setting.rst:your GPU device memory and require additional computation time . 
documents/tutorials/Forward_modeling_using_PreMODIT_Cross_Section_for_methane.rst:gpu_transfer = False. We do not need to send line information to the
documents/tutorials/Forward_modeling_using_PreMODIT_Cross_Section_for_methane.rst:    mdbCH4=api.MdbExomol('.database/CH4/12C-1H4/YT10to10/',nus,gpu_transfer=False)
documents/tutorials/reverse_premodit.rst:                    gpu_transfer=False)
documents/tutorials/reverse_precompute_grid.rst:For this example, you might need a good GPU.
documents/tutorials/reverse_precompute_grid.rst:    2024-09-29 07:22:17.164712: W external/xla/xla/service/gpu/nvptx_compiler.cc:765] The NVIDIA driver's CUDA version is 12.2 which is older than the ptxas CUDA version (12.6.20). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
documents/tutorials/reverse_precompute_grid.rst:                    gpu_transfer=False)
documents/tutorials/Cross_Section_using_Discrete_Integral_Transform.rst:        "CO", nus, isotope=1, gpu_transfer=True
documents/tutorials/opacity_exomol.rst:    mdbCO=api.MdbExomol(emf,nus,gpu_transfer=True)
documents/tutorials/opacity_exomol.rst:Although it depends on your GPU, you might need to devide the
documents/tutorials/opacity_exomol.rst:computation into multiple loops because of the limitation of the GPU
documents/tutorials/opacity_exomol.rst:memory. Here we assume 30MB for GPU memory (not exactly, memory size for
documents/tutorials/opacity_exomol.rst:    xsv=auto_xsection(nus,nu0,sigmaD,gammaL,Sij,memory_size=30) #use 30MB GPU MEMORY for numax
src/exojax/test/generate_rt.py:    # mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
src/exojax/test/generate_rt.py:    # mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
src/exojax/test/generate_rt.py:    # mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
src/exojax/test/generate_rt.py:    # mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
src/exojax/test/emulate_mdb.py:                        gpu_transfer=True)
src/exojax/test/emulate_mdb.py:                        gpu_transfer=True)
src/exojax/test/generate_xs.py:    # mdbCO = api.MdbHitemp('CO', nus, gpu_transfer=True, isotope=1)
src/exojax/test/generate.py:                          gpu_transfer=True)
src/exojax/test/generate.py:                          gpu_transfer=True)
src/exojax/test/generate_methane_trans.py:        ".database/CH4/12C-1H4/YT10to10/", nurange=nu_grid, gpu_transfer=False
src/exojax/test/generate_methane_spectrum.py:        ".database/CH4/12C-1H4/YT10to10/", nurange=nu_grid, gpu_transfer=False
src/exojax/spec/make_numatrix.py:    """Generate numatrix0 using gpu.
src/exojax/spec/multimol.py:                                          gpu_transfer=False))
src/exojax/spec/multimol.py:                                          gpu_transfer=False,
src/exojax/spec/multimol.py:                                          gpu_transfer=False,
src/exojax/spec/api.py:        gpu_transfer=True,
src/exojax/spec/api.py:            gpu_transfer: if True, some attributes will be transfered to jnp.array. False is recommended for PreMODIT.
src/exojax/spec/api.py:        self.gpu_transfer = gpu_transfer
src/exojax/spec/api.py:            and self.gpu_transfer == other.gpu_transfer
src/exojax/spec/api.py:        """Activates of moldb for Exomol,  including making attributes, computing broadening parameters, natural width, and transfering attributes to gpu arrays when self.gpu_transfer = True
src/exojax/spec/api.py:            and transfering attributes to gpu arrays when self.gpu_transfer = True
src/exojax/spec/api.py:        if self.gpu_transfer:
src/exojax/spec/api.py:        gpu_transfer=False,
src/exojax/spec/api.py:            gpu_transfer: tranfer data to jnp.array?
src/exojax/spec/api.py:        self.gpu_transfer = gpu_transfer
src/exojax/spec/api.py:            and transfering attributes to gpu arrays when self.gpu_transfer = True
src/exojax/spec/api.py:        if self.gpu_transfer:
src/exojax/spec/api.py:        gpu_transfer=False,
src/exojax/spec/api.py:            gpu_transfer: tranfer data to jnp.array?
src/exojax/spec/api.py:            gpu_transfer=gpu_transfer,
src/exojax/spec/api.py:            and self.gpu_transfer == other.gpu_transfer
src/exojax/spec/api.py:        gpu_transfer=False,
src/exojax/spec/api.py:            gpu_transfer: tranfer data to jnp.array?
src/exojax/spec/api.py:            gpu_transfer=gpu_transfer,
src/exojax/spec/api.py:            and self.gpu_transfer == other.gpu_transfer
src/exojax/spec/moldb.py:        gpu_transfer=True,
src/exojax/spec/moldb.py:            gpu_transfer: tranfer data to jnp.array?
src/exojax/spec/moldb.py:        if gpu_transfer:
src/exojax/spec/moldb.py:        gpu_transfer=True,
src/exojax/spec/moldb.py:            gpu_transfer: tranfer data to jnp.array?
src/exojax/spec/moldb.py:        if gpu_transfer:
src/exojax/spec/opacalc.py:        if not self.mdb.gpu_transfer:
src/exojax/spec/opacalc.py:            raise ValueError("For MODIT, gpu_transfer should be True in mdb.")

```
