# https://github.com/realfastvla/rfpipe

```console
setup.py:            # optional 'rfgpu', 'distributed'
setup.py:real-time and offline searches, GPU and CPU algorithms, and reproducible analysis of
docs/Preferences.rst:The most commonly used preferences define the search parameters (e.g., the range of dispersion measures), the computational limits (e.g., the maximum memory or maximum image size), and the search algorithm (e.g., CUDA or FFTW imaging).
docs/UseCases.rst:* CPU and GPU algorithms,
docs/Searching.rst:The transient search is similarly defined by the state and search functions take it as input. A range of data correction and search functions are available in the search module. Many algorithms have been implemented both for CPU (using the FFTW library) and GPU (using CUDA) environments. These are wrapped by ``rfpipe.search.prep_and_search``, which uses the ``Preferences`` to decide which function to use.
docs/Searching.rst:.. autofunction:: rfpipe.search.dedisperse_image_cuda
tests/data/realfast.yml:    fftmode: 'cuda'
tests/data/realfast.yml:    fftmode: 'cuda'
tests/data/realfast.yml:    fftmode: 'cuda'
tests/data/realfast.yml:    fftmode: 'cuda'
tests/data/realfast.yml:    fftmode: 'cuda'
tests/flagging_olympics.py:    config.addinivalue_line("datacuda", "requires cuda and data")
tests/flagging_olympics.py:    config.addinivalue_line("simcuda", "simulats data and uses fftw")
tests/flagging_olympics.py:@pytest.mark.simcuda
tests/flagging_olympics.py:def test_cuda_sim_rfi(stsim, data_prep_rfi):
tests/flagging_olympics.py:    rfgpu = pytest.importorskip('rfgpu')
tests/flagging_olympics.py:    cc = rfpipe.search.dedisperse_search_cuda(stsim, 0, data_prep_rfi)
tests/flagging_olympics.py:@pytest.mark.simcuda
tests/flagging_olympics.py:def test_cuda_sim(stsim, data_prep):
tests/flagging_olympics.py:    rfgpu = pytest.importorskip('rfgpu')
tests/flagging_olympics.py:    cc = rfpipe.search.dedisperse_search_cuda(stsim, 0, data_prep)
tests/flagging_olympics.py:    rfgpu = pytest.importorskip('rfgpu')
tests/flagging_olympics.py:@pytest.mark.datacuda
tests/flagging_olympics.py:def test_cuda_data(stdata):
tests/flagging_olympics.py:    rfgpu = pytest.importorskip('rfgpu')
tests/flagging_olympics.py:    stdata.prefs.fftmode = 'cuda'
tests/flagging_olympics.py:@pytest.mark.datafftwcuda
tests/flagging_olympics.py:    rfgpu = pytest.importorskip('rfgpu')
tests/flagging_olympics.py:    stdata.prefs.fftmode = 'cuda'
tests/flagging_olympics.py:    assert len(cc0.array) == len(cc1.array), "FFTW and CUDA search not returning the same results"
rfpipe/candidates.py:        # TODO: find good estimate that can be implemented in both CPU and GPU
rfpipe/candidates.py:        # assume first gpu, but try to infer from worker name
rfpipe/candidates.py:            logger.warning("Could not parse worker name {0}. Using default GPU devicenum {1}"
rfpipe/candidates.py:            logger.warning("No worker found. Using default GPU devicenum {0}"
rfpipe/candidates.py:            logger.warning("distributed not available. Using default GPU devicenum {0}"
rfpipe/candidates.py:    logger.info("Using gpu devicenum: {0}".format(devicenum))
rfpipe/candidates.py:    os.environ['CUDA_VISIBLE_DEVICES'] = str(devicenum)
rfpipe/candidates.py:            config.gpu_options.allow_growth = True
rfpipe/candidates.py:            config.gpu_options.per_process_gpu_memory_fraction = 0.5
rfpipe/candidates.py:        imstd = im.std()  # consistent with rfgpu
rfpipe/pipeline.py:    devicenum refers to GPU device for search.
rfpipe/pipeline.py:    if st.prefs.fftmode == "cuda":
rfpipe/pipeline.py:        candcollection = search.dedisperse_search_cuda(st, segment, data,
rfpipe/pipeline.py:        logger.warning("fftmode {0} not recognized (cuda, fftw allowed)"
rfpipe/state.py:        # supported algorithms for gpu/cpu
rfpipe/state.py:        if self.prefs.fftmode == 'cuda' and self.prefs.searchtype is not None:
rfpipe/state.py:            elif self.fftmode == "cuda":
rfpipe/state.py:                            'GB/segment when using cuda imaging.'
rfpipe/state.py:            elif self.prefs.fftmode == 'cuda':
rfpipe/state.py:            elif self.prefs.fftmode == 'cuda':
rfpipe/state.py:        """ Should the FFT be done with fftw or cuda?
rfpipe/state.py:        elif self.fftmode == "cuda":
rfpipe/state.py:        elif self.fftmode == "cuda":
rfpipe/source.py:    **TODO: figure out if this can collide with rfgpu calls in the search module**
rfpipe/source.py:        import rfgpu
rfpipe/source.py:        logger.warn("Cannot import rfgpu, so cannot calculate gridfrac")
rfpipe/source.py:    grid = rfgpu.Grid(st.nbl, st.nchan, st.readints, upix, vpix, 0)  # choose device 0
rfpipe/util.py:from numba import cuda, guvectorize
rfpipe/search.py:    import rfgpu
rfpipe/search.py:def dedisperse_search_cuda(st, segment, data, devicenum=None):
rfpipe/search.py:    Grid and image on GPU (uses rfgpu from separate repo).
rfpipe/search.py:    devicenum is int or tuple of ints that set gpu(s) to use.
rfpipe/search.py:        # assume first gpu, but try to infer from worker name
rfpipe/search.py:            devicenums = (devicenum, devicenum+1)  # TODO: smarter multi-GPU
rfpipe/search.py:            logger.debug("Using name {0} to set GPU devicenum to {1}"
rfpipe/search.py:            logger.warning("Could not parse worker name {0}. Using default GPU devicenum {1}"
rfpipe/search.py:            logger.warning("No worker found. Using default GPU devicenum {0}"
rfpipe/search.py:            logger.warning("distributed not available. Using default GPU devicenum {0}"
rfpipe/search.py:    logger.info("Using gpu devicenum(s): {0}".format(devicenums))
rfpipe/search.py:    grids = [rfgpu.Grid(st.nbl, st.nchan, st.readints, upix, vpix, dn) for dn in devicenums]
rfpipe/search.py:    images = [rfgpu.Image(st.npixx, st.npixy, dn) for dn in devicenums]
rfpipe/search.py:    # Data buffers on GPU
rfpipe/search.py:    # Vis buffers identical on all GPUs. image buffer unique.
rfpipe/search.py:    vis_raw = rfgpu.GPUArrayComplex((st.nbl, st.nchan, st.readints),
rfpipe/search.py:    vis_grids = [rfgpu.GPUArrayComplex((upix, vpix), (dn,)) for dn in devicenums]
rfpipe/search.py:    img_grids = [rfgpu.GPUArrayReal((st.npixx, st.npixy), (dn,)) for dn in devicenums]
rfpipe/search.py:    vis_raw.h2d()  # Send it to GPU memory of all
rfpipe/search.py:#                threads.append(ex.submit(rfgpu_gridimage, st, segment,
rfpipe/search.py:                threads.append(ex.submit(rfgpu_gridimage, st, segment,
rfpipe/search.py:def rfgpu_gridimage(st, segment, grid, image, vis_raw, vis_grid, img_grid,
rfpipe/search.py:    """ Dedisperse, grid, image, threshold with rfgpu
rfpipe/search.py:                        ' with image {6}x{7} (uvres {8}) on GPU {9}'
rfpipe/search.py:                        # TODO: implement phasing on GPU
rfpipe/search.py:                logger.warning("rfgpu rms is 0 in ints {0}."
rfpipe/search.py:    fftmode can be fftw or cuda.
rfpipe/search.py:    elif fftmode == 'cuda':
rfpipe/search.py:        logger.warning("Imaging with cuda not yet supported.")
rfpipe/search.py:        images = image_cuda()
rfpipe/search.py:def image_cuda():
rfpipe/search.py:    """ Run grid and image with rfgpu
rfpipe/search.py:    TODO: update to use rfgpu
rfpipe/search.py:def make_dmt(data, dmi, dmf, dmsteps, freqs, inttime, mode='GPU', devicenum=0):
rfpipe/search.py:    if mode == 'GPU':
rfpipe/search.py:        dmt = gpu_dmtime(data, dmi, dmf, dmsteps, freqs, inttime,
rfpipe/search.py:def gpu_dmtime(ft, dm_i, dm_f, dmsteps, freqs, inttime, devicenum=0):
rfpipe/search.py:    from numba import cuda
rfpipe/search.py:    os.environ['NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT'] = '1'
rfpipe/search.py:    cuda.select_device(devicenum)
rfpipe/search.py:    stream = cuda.stream()
rfpipe/search.py:    @cuda.jit(fastmath=True)
rfpipe/search.py:    def gpu_dmt(cand_data_in, all_delays, cand_data_out):
rfpipe/search.py:        ii, jj, kk = cuda.grid(3)
rfpipe/search.py:            cuda.atomic.add(cand_data_out, (kk, jj), cand_data_in[ii,
rfpipe/search.py:    with cuda.defer_cleanup():
rfpipe/search.py:        all_delays = cuda.to_device(delays.T, stream=stream)
rfpipe/search.py:        dmt_return = cuda.device_array(dm_time.shape, dtype=np.float32, stream=stream)
rfpipe/search.py:        cand_data_in = cuda.to_device(np.array(ft, dtype=np.float32), stream=stream)
rfpipe/search.py:        gpu_dmt[blockspergrid, threadsperblock, stream](cand_data_in, all_delays,  dmt_return)
rfpipe/search.py:    # cuda.close()
rfpipe/search.py:    resamplefirst is parameter that reproduces rfgpu order.
rfpipe/reproduce.py:                                                 resamplefirst=st.fftmode=='cuda')
rfpipe/reproduce.py:    logger.info("Using gpu devicenum: {0}".format(devicenum))
rfpipe/reproduce.py:    os.environ['CUDA_VISIBLE_DEVICES'] = str(devicenum)
rfpipe/reproduce.py:    imstd = im.std()  # consistent with rfgpu
rfpipe/preferences.py:    fftmode = attr.ib(default='fftw')  # either 'fftw' or 'cuda'. defines segment size and algorithm used.
README.md:- numba (for multi-core and gpu acceleration)
README.md:- rfgpu (optional; for GPU FFTs)

```
