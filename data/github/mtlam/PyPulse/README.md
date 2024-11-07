# https://github.com/mtlam/PyPulse

```console
pypulse/archive.py:                 wcfreq=True, thread=False, cuda=False, onlyheader=False):
pypulse/archive.py:        self.cuda = cuda
pypulse/archive.py:            cudasuccess = False
pypulse/archive.py:            if self.cuda:
pypulse/archive.py:                    gpuarray = import_module('pycuda.gpuarray')
pypulse/archive.py:                    compiler = import_module('pycuda.compiler')
pypulse/archive.py:                    driver = import_module('pycuda.driver')
pypulse/archive.py:                    autoinit = import_module('pycuda.autoinit')
pypulse/archive.py:                    cudasuccess = True
pypulse/archive.py:                    warnings.warn("PyCUDA not imported", ImportWarning)
pypulse/archive.py:                #combine(data_gpu,driver.In(DAT_SCL),data_gpu,driver.In(DAT_OFF),nbin,block=(4,4,1))
pypulse/archive.py:                #driver.memcpy_dtoh(retval, data_gpu)
pypulse/archive.py:            if self.thread and not cudasuccess:
pypulse/archive.py:            elif not cudasuccess:

```
