# https://github.com/lsds/sgx-lkl

```console
tests/ltp/ltp-batch1/ltp_disabled_tests.txt:/ltp/testcases/kernel/syscalls/rt_sigprocmask/rt_sigprocmask01
tests/ltp/ltp-batch1/ltp_disabled_tests.txt:/ltp/testcases/kernel/syscalls/rt_sigprocmask/rt_sigprocmask02
tests/ltp/ltp-batch1/ltp_disabled_tests.txt:/ltp/testcases/kernel/syscalls/sigprocmask/sigprocmask01
tests/ltp/ltp-batch2/ltp_disabled_tests.txt:#/ltp/testcases/kernel/syscalls/rt_sigprocmask/rt_sigprocmask01
tests/ltp/ltp-batch2/ltp_disabled_tests.txt:#/ltp/testcases/kernel/syscalls/rt_sigprocmask/rt_sigprocmask02
tests/ltp/ltp-batch2/ltp_disabled_tests.txt:#/ltp/testcases/kernel/syscalls/sigprocmask/sigprocmask01
samples/ml/pytorch/Dockerfile:    && DEBUG=0 USE_CUDA=0 USE_MKLDNN=0 USE_OPENMP=0 ATEN_THREADING=NATIVE BUILD_BINARY=0 \
samples/ml/tensorflow/app/benchmark/mnist_lenet_eval.py:    # Saves memory and enables this to run on smaller GPUs.
samples/ml/tensorflow/app/benchmark/mnist_lenet.py:    # Saves memory and enables this to run on smaller GPUs.
samples/ml/tensorflow/Dockerfile-TF1.15:        TF_NEED_OPENCL=0 \
samples/ml/tensorflow/Dockerfile-TF1.15:        TF_NEED_CUDA=0 \
tools/lkl_syscalls.c:    EXPORT_LKL_SYSCALL(rt_sigprocmask)
tools/lkl_bits.c:    EXPORT_LKL_SYSCALL(rt_sigprocmask)

```
