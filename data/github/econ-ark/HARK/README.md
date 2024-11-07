# https://github.com/econ-ark/HARK

```console
HARK/tests/OpenCLtest.py:Simple test for opencl4py, edited from the example distributed in that package.
HARK/tests/OpenCLtest.py:import opencl4py as cl
HARK/tests/OpenCLtest.py:os.environ["PYOPENCL_CTX"] = (
HARK/tests/OpenCLtest.py:    print("Will test OpenCL on " + queue.device.name + ".")
HARK/tests/OpenCLtest.py:        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
HARK/tests/OpenCLtest.py:        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
HARK/tests/OpenCLtest.py:    # Define OpenCL memory buffers, passing appropriate flags and inputs
HARK/tests/OpenCLtest.py:    print("OpenCL took " + str(t_end - t_start) + " seconds.")
HARK/tests/OpenCLtest.py:    # Make sure that OpenCL and Python actually agree on their results
HARK/tests/OpenCLtest.py:        "Maximum difference between OpenCL and Python calculations is " + str(max_diff)

```
