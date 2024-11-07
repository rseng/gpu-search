# https://github.com/carronj/LensIt

```console
README.md:To use the GPU implementation of some of the routines, you will need [pyCUDA](https://mathema.tician.de/software/pycuda)
lensit/ffs_covs/ffs_cov.py:            from lensit.gpu import apply_GPU
lensit/ffs_covs/ffs_cov.py:            apply_GPU.apply_FDxiDtFt_GPU_inplace(typ, self.lib_datalm, self.lib_skyalm, ablms,
lensit/ffs_covs/ffs_cov.py:            # Try entire evaluation on GPU :
lensit/ffs_covs/ffs_cov.py:            from lensit.gpu.apply_cond3_GPU import apply_cond3_GPU_inplace as c3GPU
lensit/ffs_covs/ffs_cov.py:            c3GPU(typ, self.lib_datalm, ret, self.f, self.fi, self.cls_unl, self.cl_transf, self.cls_noise)
lensit/ffs_deflect/ffs_deflect.py:                 use_Pool(optional): set this to < 0 to perform the operation on the GPU
lensit/ffs_deflect/ffs_deflect.py:            # use of GPU :
lensit/ffs_deflect/ffs_deflect.py:                from lensit.gpu import lens_GPU
lensit/ffs_deflect/ffs_deflect.py:                assert 0, 'Import of mllens lens_GPU failed !'
lensit/ffs_deflect/ffs_deflect.py:            GPU_res = np.array(lens_GPU.GPU_HDres_max)
lensit/ffs_deflect/ffs_deflect.py:            if np.all(np.array(self.HD_res) <= GPU_res):
lensit/ffs_deflect/ffs_deflect.py:                return lens_GPU.lens_onGPU(m, self.get_dx_ingridunits(), self.get_dy_ingridunits(),
lensit/ffs_deflect/ffs_deflect.py:            LD_res, buffers = lens_GPU.get_GPUbuffers(GPU_res)
lensit/ffs_deflect/ffs_deflect.py:                      '   splitting map on GPU , chunk shape %s, buffers %s' % (dx_N.shape, buffers))
lensit/ffs_deflect/ffs_deflect.py:                lensed_map[sHDs[0]] = lens_GPU.lens_onGPU(unl_CMBN, dx_N, dy_N, do_not_prefilter=do_not_prefilter)[sLDs[0]]
lensit/ffs_deflect/ffs_deflect.py:                use_Pool(optional): calculations are performed on the GPU if negative.
lensit/ffs_deflect/ffs_deflect.py:        if use_Pool < 0:  # can we fit the full map on the GPU ?
lensit/ffs_deflect/ffs_deflect.py:            from lensit.gpu import lens_GPU
lensit/ffs_deflect/ffs_deflect.py:            GPU_res = np.array(lens_GPU.GPU_HDres_max)
lensit/ffs_deflect/ffs_deflect.py:            if np.all(np.array(self.HD_res) <= GPU_res):
lensit/ffs_deflect/ffs_deflect.py:                return lens_GPU.lens_alm_onGPU(lib_alm, lib_alm.bicubic_prefilter(alm),
lensit/ffs_deflect/ffs_deflect.py:        if use_Pool < 0:  # can we fit the full map on the GPU ? If we can't we send it the lens_map
lensit/ffs_deflect/ffs_deflect.py:            from lensit.gpu import lens_GPU
lensit/ffs_deflect/ffs_deflect.py:            GPU_res = np.array(lens_GPU.GPU_HDres_max)
lensit/ffs_deflect/ffs_deflect.py:            if np.all(np.array(self.HD_res) <= GPU_res):
lensit/ffs_deflect/ffs_deflect.py:                return lens_GPU.alm2lenmap_onGPU(lib_alm, lib_alm.bicubic_prefilter(alm),
lensit/ffs_deflect/ffs_deflect.py:                use_Pool(optional): Send the calculation to the GPU if negative
lensit/ffs_deflect/ffs_deflect.py:            # GPU calculation.
lensit/ffs_deflect/ffs_deflect.py:            from lensit.gpu_old import lens_GPU
lensit/ffs_deflect/ffs_deflect.py:            from lensit.gpu_old import inverse_GPU as inverse_GPU
lensit/ffs_deflect/ffs_deflect.py:            GPU_res = np.array(inverse_GPU.GPU_HDres_max)
lensit/ffs_deflect/ffs_deflect.py:            if np.all(np.array(self.HD_res) <= GPU_res):
lensit/ffs_deflect/ffs_deflect.py:                dx_inv, dy_inv = inverse_GPU.inverse_GPU(self.get_dx(), self.get_dy(), self.rmin, NR_iter)
lensit/ffs_deflect/ffs_deflect.py:                LD_res, buffers = lens_GPU.get_GPUbuffers(GPU_res)
lensit/ffs_deflect/ffs_deflect.py:                          '   splitting inverse on GPU , chunk shape %s, buffers %s' % (dx_N.shape, buffers))
lensit/ffs_deflect/ffs_deflect.py:                    dx_inv_N, dy_inv_N = inverse_GPU.inverse_GPU(dx_N, dy_N, self.rmin, NR_iter)

```
