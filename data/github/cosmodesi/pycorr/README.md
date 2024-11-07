# https://github.com/cosmodesi/pycorr

```console
README.md:- Craig Warner for GPU-izing Corrfunc 'smu' counts
pycorr/corrfunc.py:        if attrs.pop('gpu', False):
pycorr/corrfunc.py:            kwargs['gpu'] = True
pycorr/corrfunc.py:                if kwargs['gpu']:
pycorr/corrfunc.py:                    raise NotImplementedError('GPU kernel not implemented for kernel {}'.format(method)) from exc
pycorr/tests/test_twopoint_counter.py:def test_gpu(mode='smu'):
pycorr/tests/test_twopoint_counter.py:        gpu = TwoPointCounter(**kwargs, gpu=True, nthreads=4)
pycorr/tests/test_twopoint_counter.py:        dt_gpu = time.time() - t0
pycorr/tests/test_twopoint_counter.py:        print('autocorr is {}, GPU time is {:.4f} vs CPU time {:4f}'.format(autocorr, dt_gpu, dt_cpu))
pycorr/tests/test_twopoint_counter.py:        #print(gpu.wcounts - cpu.wcounts)
pycorr/tests/test_twopoint_counter.py:        assert np.allclose(gpu.wcounts, cpu.wcounts, **tol)
pycorr/tests/test_twopoint_counter.py:        assert np.allclose(gpu.seps[0], cpu.seps[0], equal_nan=True, **tol)
pycorr/tests/test_twopoint_counter.py:                TwoPointCounter(**kw, gpu=True)
pycorr/tests/test_twopoint_counter.py:                                    position_type='xyz', gpu=True, nthreads=4, verbose=False, **options)
pycorr/tests/test_twopoint_counter.py:    #test_gpu()

```
