# https://github.com/SchedMD/slurm

```console
auxdir/x_ac_rsmi.m4:  # /opt/rocm is the current default location.
auxdir/x_ac_rsmi.m4:  # /opt/rocm/rocm_smi was the default location for before to 5.2.0
auxdir/x_ac_rsmi.m4:  _x_ac_rsmi_dirs="/opt/rocm /opt/rocm/rocm_smi"
auxdir/x_ac_rsmi.m4:    AC_MSG_CHECKING([whether RSMI/ROCm in installed in this system])
auxdir/x_ac_rsmi.m4:      AS_UNSET([ac_cv_header_rocm_smi_h])
auxdir/x_ac_rsmi.m4:      AS_UNSET([ac_cv_lib_rocm_smi64_rsmi_init])
auxdir/x_ac_rsmi.m4:      AS_UNSET([ac_cv_lib_rocm_smi64_dev_drm_render_minor_get])
auxdir/x_ac_rsmi.m4:      AC_CHECK_HEADER([rocm_smi/rocm_smi.h], [ac_rsmi_h=yes], [ac_rsmi_h=no])
auxdir/x_ac_rsmi.m4:      AC_CHECK_LIB([rocm_smi64], [rsmi_init], [ac_rsmi_l=yes], [ac_rsmi_l=no])
auxdir/x_ac_rsmi.m4:      AC_CHECK_LIB([rocm_smi64], [rsmi_dev_drm_render_minor_get], [ac_rsmi_version=yes], [ac_rsmi_version=no])
auxdir/x_ac_rsmi.m4:          AC_MSG_WARN([upgrade to newer version of ROCm/rsmi])
auxdir/x_ac_rsmi.m4:          AC_MSG_ERROR([upgrade to newer version of ROCm/rsmi])
auxdir/x_ac_rsmi.m4:        AC_MSG_WARN([unable to locate librocm_smi64.so and/or rocm_smi.h])
auxdir/x_ac_rsmi.m4:        AC_MSG_ERROR([unable to locate librocm_smi64.so and/or rocm_smi.h])
auxdir/libtool.m4:    nvcc*) # Cuda Compiler Driver 2.2
auxdir/libtool.m4:	nvcc*)	# Cuda Compiler Driver 2.2
auxdir/x_ac_nvml.m4:#    Determine if NVIDIA's NVML API library exists (CUDA provides stubs)
auxdir/x_ac_nvml.m4:      AS_UNSET([ac_cv_lib_nvidia_ml_nvmlInit])
auxdir/x_ac_nvml.m4:      AC_CHECK_LIB([nvidia-ml], [nvmlInit], [ac_nvml=yes], [ac_nvml=no])
auxdir/x_ac_nvml.m4:          # Check indirectly that CUDA 11.1+ was installed to see if we
auxdir/x_ac_nvml.m4:	  # gpuInstanceSliceCount in the nvmlDeviceAttributes_t struct.
auxdir/x_ac_nvml.m4:		     attributes.gpuInstanceSliceCount = 0;
auxdir/x_ac_nvml.m4:  _x_ac_nvml_dirs="/usr/local/cuda /usr/cuda"
auxdir/x_ac_nvml.m4:    AS_HELP_STRING(--with-nvml=PATH, Specify path to CUDA installation),
auxdir/x_ac_nvml.m4:          nvml_libs="-lnvidia-ml"
auxdir/x_ac_nvml.m4:          LDFLAGS="-L$d/$bit -lnvidia-ml"
auxdir/x_ac_nvml.m4:	AC_MSG_WARN([NVML was found, but can not support MIG. For MIG support both nvml.h and libnvidia-ml must be 11.1+. Please make sure they are both the same version as well.])
auxdir/x_ac_nvml.m4:        AC_MSG_WARN([unable to locate libnvidia-ml.so and/or nvml.h])
auxdir/x_ac_nvml.m4:        AC_MSG_ERROR([unable to locate libnvidia-ml.so and/or nvml.h])
configure.ac:		 src/plugins/acct_gather_energy/gpu/Makefile
configure.ac:		 src/plugins/gpu/Makefile
configure.ac:		 src/plugins/gpu/common/Makefile
configure.ac:		 src/plugins/gpu/generic/Makefile
configure.ac:		 src/plugins/gpu/nrt/Makefile
configure.ac:		 src/plugins/gpu/nvidia/Makefile
configure.ac:		 src/plugins/gpu/nvml/Makefile
configure.ac:		 src/plugins/gpu/oneapi/Makefile
configure.ac:		 src/plugins/gpu/rsmi/Makefile
configure.ac:		 src/plugins/gres/gpu/Makefile
configure.ac:		 src/plugins/switch/nvidia_imex/Makefile
testsuite/python/tests/test_105_3.py:    atf.require_config_parameter_includes("GresTypes", "gpu")
testsuite/python/tests/test_105_3.py:        "Name", {"gpu": {"File": "/dev/tty0"}, "mps": {"Count": 100}}, source="gres"
testsuite/python/tests/test_105_3.py:    atf.require_nodes(1, [("Gres", "gpu:1,mps:100")])
testsuite/python/tests/test_105_3.py:def test_mps_and_gpus():
testsuite/python/tests/test_105_3.py:    """Test with both GPUs and MPS in a single request"""
testsuite/python/tests/test_105_3.py:    results = atf.run_command('sbatch --gres=mps:1,gpu:1 -N1 -t1 --wrap "true"')
testsuite/python/tests/test_105_3.py:def test_mps_and_gpu_frequency():
testsuite/python/tests/test_105_3.py:    """Test with both GPUs and MPS in a single request"""
testsuite/python/tests/test_105_3.py:        'sbatch --gres=mps:1 --gpu-freq=high -N1 -t1 --wrap "true"'
testsuite/python/tests/test_144_7.py:    atf.require_config_parameter_includes("GresTypes", "gpu")
testsuite/python/tests/test_144_7.py:    # Require 8 tty because one test requests 8 "GPU"s (4 GPUS each for 2 nodes)
testsuite/python/tests/test_144_7.py:        "Name", {"gpu": {"File": "/dev/tty[0-7]"}}, source="gres"
testsuite/python/tests/test_144_7.py:    atf.require_nodes(2, [("Gres", f"gpu:4"), ("CPUs", 8)])
testsuite/python/tests/test_144_7.py:def test_gpus_per_node_parallel_1_delayed():
testsuite/python/tests/test_144_7.py:    """Test --gpus-per-node option by job step"""
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n1 --gpus-per-node=1 --exact -n1 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n1 --gpus-per-node=1 --exact -n1 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n1 --gpus-per-node=1 --exact -n1 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        echo 'STEP_ID:'$SLURM_STEP_ID 'CUDA_VISIBLE_DEVICES:'$CUDA_VISIBLE_DEVICES
testsuite/python/tests/test_144_7.py:        "--cpus-per-gpu=1 --gpus-per-node=2 -N1 -n3 -t1 "
testsuite/python/tests/test_144_7.py:    # Verify all steps used only 1 GPU
testsuite/python/tests/test_144_7.py:        r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+,", output
testsuite/python/tests/test_144_7.py:    ), "Not all steps used only 1 GPU"
testsuite/python/tests/test_144_7.py:    # Verify a GPU was used 3 times
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+", output)) == 3
testsuite/python/tests/test_144_7.py:    ), "A GPU was not used 3 times"
testsuite/python/tests/test_144_7.py:        # Verify all GPUs are CUDA_VISIBLE_DEVICES:0 (with ConstrainDevices)
testsuite/python/tests/test_144_7.py:            len(re.findall(r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:0", output)) == 3
testsuite/python/tests/test_144_7.py:        ), "Not all GPUs are CUDA_VISIBLE_DEVICES:0 (with ConstrainDevices)"
testsuite/python/tests/test_144_7.py:        # Verify steps split between the two GPUs (without ConstrainDevices)
testsuite/python/tests/test_144_7.py:        cuda_devices_used = re.findall(
testsuite/python/tests/test_144_7.py:            r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:(\d+)", output
testsuite/python/tests/test_144_7.py:            len(set(cuda_devices_used)) > 1
testsuite/python/tests/test_144_7.py:        ), f"The job steps weren't split among the two GPUS (CUDA devices {set(cuda_devices_used)} used instead of 0 and 1 (without ConstrainDevices))"
testsuite/python/tests/test_144_7.py:@pytest.mark.parametrize("step_args", ["-n1 --gpus-per-task=1", "-n1 --gpus=1"])
testsuite/python/tests/test_144_7.py:def test_gpus_per_node_parallel(step_args):
testsuite/python/tests/test_144_7.py:    """Test parallel step args with a job with --gpus-per-node"""
testsuite/python/tests/test_144_7.py:        srun --exact --gpus-per-node=0 --mem=0 {step_args} {step_file} &
testsuite/python/tests/test_144_7.py:        srun --exact --gpus-per-node=0 --mem=0 {step_args} {step_file} &
testsuite/python/tests/test_144_7.py:        echo 'STEP_ID:'$SLURM_STEP_ID 'CUDA_VISIBLE_DEVICES:'$CUDA_VISIBLE_DEVICES
testsuite/python/tests/test_144_7.py:        "--cpus-per-gpu=2 --gpus-per-node=2 -N1 -n2 -t1 "
testsuite/python/tests/test_144_7.py:    # Verify all steps used only 1 GPU
testsuite/python/tests/test_144_7.py:        r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+,", output
testsuite/python/tests/test_144_7.py:    ), "Not all steps used only 1 GPU"
testsuite/python/tests/test_144_7.py:    # Verify a GPU was used 2 times
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+", output)) == 2
testsuite/python/tests/test_144_7.py:    ), "A GPU was not used 2 times"
testsuite/python/tests/test_144_7.py:        # Verify all GPUs are CUDA_VISIBLE_DEVICES:0 (with ConstrainDevices)
testsuite/python/tests/test_144_7.py:            len(re.findall(r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:0", output)) == 2
testsuite/python/tests/test_144_7.py:        ), "Not all GPUs are CUDA_VISIBLE_DEVICES:0 (with ConstrainDevices)"
testsuite/python/tests/test_144_7.py:        # Verify 1 GPU is CUDA_VISIBLE_DEVICES:0 (without ConstrainDevices)
testsuite/python/tests/test_144_7.py:            len(re.findall(r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:0", output)) == 1
testsuite/python/tests/test_144_7.py:        ), "Not 1 GPU is CUDA_VISIBLE_DEVICES:0 (without ConstrainDevices)"
testsuite/python/tests/test_144_7.py:        # Verify 1 GPU is CUDA_VISIBLE_DEVICES:1 (without ConstrainDevices)
testsuite/python/tests/test_144_7.py:            len(re.findall(r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:1", output)) == 1
testsuite/python/tests/test_144_7.py:        ), "Not 1 GPU is CUDA_VISIBLE_DEVICES:1 (without ConstrainDevices)"
testsuite/python/tests/test_144_7.py:def test_gpus_per_node_different_gpus():
testsuite/python/tests/test_144_7.py:    """Test --gpus (per job or step) option by job step"""
testsuite/python/tests/test_144_7.py:        srun --exact -n2 --gpus=2 --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun --exact -n1 --gpus=1 --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        echo 'STEP_ID:'$SLURM_STEP_ID 'CUDA_VISIBLE_DEVICES:'$CUDA_VISIBLE_DEVICES
testsuite/python/tests/test_144_7.py:        "--cpus-per-gpu=1 --gpus-per-node=3 -N1 -n3 -t2 "
testsuite/python/tests/test_144_7.py:    step_2gpu = re.search(
testsuite/python/tests/test_144_7.py:        r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:(\d+),(\d+)", output
testsuite/python/tests/test_144_7.py:    step_1gpu = re.search(
testsuite/python/tests/test_144_7.py:        r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:(\d+$)", output, re.MULTILINE
testsuite/python/tests/test_144_7.py:    # Verify 1 step used 1 GPU and 2 steps used 2 GPUs
testsuite/python/tests/test_144_7.py:        len(step_1gpu) == 1 and len(step_2gpu) == 2
testsuite/python/tests/test_144_7.py:    ), f"Fail to obtain all GPUs index ({len(step_1gpu)} != 1 or {len(step_2gpu)} != 2)"
testsuite/python/tests/test_144_7.py:        # Verify if devices are constrained, CUDA_VISIBLE_DEVICES start always
testsuite/python/tests/test_144_7.py:            step_2gpu[0] == "0" and step_1gpu[0] == "0"
testsuite/python/tests/test_144_7.py:        ), "CUDA_VISIBLE_DEVICES did not always start with 0 in a step"
testsuite/python/tests/test_144_7.py:        # Verify if devices are NOT constrained, all CUDA_VISIBLE_DEVICES are
testsuite/python/tests/test_144_7.py:        assert step_1gpu[0] not in step_2gpu, "All CUDA_VISIBLE_DEVICES are not unique"
testsuite/python/tests/test_144_7.py:def test_gpus_per_node_with_gpus_per_task():
testsuite/python/tests/test_144_7.py:    """Test --gpus-per-task option by job step"""
testsuite/python/tests/test_144_7.py:    job_gpus = 3
testsuite/python/tests/test_144_7.py:    step_gpus = 2
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n1 --gpus-per-task={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n1 --gpus-per-task={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n1 --gpus-per-task={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        echo 'STEP_ID:'$SLURM_STEP_ID 'CUDA_VISIBLE_DEVICES:'$CUDA_VISIBLE_DEVICES
testsuite/python/tests/test_144_7.py:        f"--cpus-per-gpu=1 --gpus-per-node={job_gpus} -N1 -n3 -t1 "
testsuite/python/tests/test_144_7.py:    # Verify no step has more than 2 GPUs
testsuite/python/tests/test_144_7.py:        r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+,\d+,", output
testsuite/python/tests/test_144_7.py:    ), "A step has more than 2 GPUs"
testsuite/python/tests/test_144_7.py:    # Verify all steps have 2 GPUs
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+,\d+", output)) == 3
testsuite/python/tests/test_144_7.py:    ), "Not all steps have 2 GPUs"
testsuite/python/tests/test_144_7.py:def test_gpus_per_node_with_gpus():
testsuite/python/tests/test_144_7.py:    """Test --gpus option by job step"""
testsuite/python/tests/test_144_7.py:    job_gpus = 2
testsuite/python/tests/test_144_7.py:    step_gpus = 2
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n2 --gpus={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n2 --gpus={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n2 --gpus={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        echo 'HOST:'$SLURMD_NODENAME 'NODE_ID:'$SLURM_NODEID 'STEP_ID:'$SLURM_STEP_ID 'CUDA_VISIBLE_DEVICES:'$CUDA_VISIBLE_DEVICES
testsuite/python/tests/test_144_7.py:        f"--cpus-per-gpu=2 --gpus-per-node={job_gpus} -N2 -n6 -t1 "
testsuite/python/tests/test_144_7.py:    # Verify no more that 1 GPU is visible (per node)
testsuite/python/tests/test_144_7.py:        r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+,", output
testsuite/python/tests/test_144_7.py:    ), "1 GPU is not visible (per node)"
testsuite/python/tests/test_144_7.py:    # Verify step 0 had access to 2 GPUs
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:0 CUDA_VISIBLE_DEVICES:\d+", output)) == 2
testsuite/python/tests/test_144_7.py:    ), "Step 0 did not have access to 2 GPUs"
testsuite/python/tests/test_144_7.py:    # Verify step 1 had access to 2 GPUs
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:1 CUDA_VISIBLE_DEVICES:\d+", output)) == 2
testsuite/python/tests/test_144_7.py:    ), "Step 1 did not have access to 2 GPUs"
testsuite/python/tests/test_144_7.py:    # Verify step 2 had access to 2 GPUs
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:2 CUDA_VISIBLE_DEVICES:\d+", output)) == 2
testsuite/python/tests/test_144_7.py:    ), "Step 2 did not have access to 2 GPUs"
testsuite/python/tests/test_144_7.py:        # Verify all GPUs are CUDA_VISIBLE_DEVICES:0 due to ConstrainDevices
testsuite/python/tests/test_144_7.py:            len(re.findall(r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:0", output)) == 6
testsuite/python/tests/test_144_7.py:        ), "Not all GPUs are CUDA_VISIBLE_DEVICES:0 due to ConstrainDevices"
testsuite/python/tests/test_144_7.py:        cuda_val = []
testsuite/python/tests/test_144_7.py:        cuda_val.append(
testsuite/python/tests/test_144_7.py:            re.search(r"STEP_ID:0 CUDA_VISIBLE_DEVICES:(\d+)", output).group(1)
testsuite/python/tests/test_144_7.py:        cuda_val.append(
testsuite/python/tests/test_144_7.py:            re.search(r"STEP_ID:1 CUDA_VISIBLE_DEVICES:(\d+)", output).group(1)
testsuite/python/tests/test_144_7.py:        cuda_val.append(
testsuite/python/tests/test_144_7.py:            re.search(r"STEP_ID:2 CUDA_VISIBLE_DEVICES:(\d+)", output).group(1)
testsuite/python/tests/test_144_7.py:        # Verify two first steps use different GPUs (without ConstrainDevices)
testsuite/python/tests/test_144_7.py:            cuda_val[0] != cuda_val[1]
testsuite/python/tests/test_144_7.py:        ), "The two first steps did not use different GPUs (without ConstrainDevices)"
testsuite/python/tests/test_144_7.py:        # Verify last step used a previous GPU (without ConstrainDevices)
testsuite/python/tests/test_144_7.py:            cuda_val[2] == cuda_val[0] or cuda_val[2] == cuda_val[1]
testsuite/python/tests/test_144_7.py:        ), "The last step did not use one of the previous GPUs (without ConstrainDevices)"
testsuite/python/tests/test_144_7.py:def test_gpus_per_node_with_gpus_2_nodes():
testsuite/python/tests/test_144_7.py:    """Test --gpus option across 2 nodes"""
testsuite/python/tests/test_144_7.py:    job_gpus = 4
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n2 --gpus=6 --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n2 --gpus=7 --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n2 --gpus=8 --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        echo 'HOST:'$SLURMD_NODENAME 'NODE_ID:'$SLURM_NODEID 'STEP_ID:'$SLURM_STEP_ID 'CUDA_VISIBLE_DEVICES:'$CUDA_VISIBLE_DEVICES
testsuite/python/tests/test_144_7.py:        f"--cpus-per-gpu=2 --gpus-per-node={job_gpus} -N2 -n6 -t1 "
testsuite/python/tests/test_144_7.py:    # Verify all steps have less than 5 GPUs per node
testsuite/python/tests/test_144_7.py:        r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+,\d+,\d+,\d+,", output
testsuite/python/tests/test_144_7.py:    ), "Not all steps have less than 5 GPUs per node"
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:0 CUDA_VISIBLE_DEVICES:\d+", output)) == 2
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:1 CUDA_VISIBLE_DEVICES:\d+", output)) == 2
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:2 CUDA_VISIBLE_DEVICES:\d+", output)) == 2
testsuite/python/tests/test_144_7.py:def test_gpus_per_node_with_gpus_per_task_3():
testsuite/python/tests/test_144_7.py:    """Test --gpus-per-task option by job step"""
testsuite/python/tests/test_144_7.py:    job_gpus = 4
testsuite/python/tests/test_144_7.py:    step_gpus = 2
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n3 --gpus-per-task={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -n3 --gpus-per-task={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        echo 'STEP_ID:'$SLURM_STEP_ID 'CUDA_VISIBLE_DEVICES:'$CUDA_VISIBLE_DEVICES
testsuite/python/tests/test_144_7.py:        f"--cpus-per-gpu=1 --gpus-per-node={job_gpus} -N2 -n4 -t1 "
testsuite/python/tests/test_144_7.py:    # Verify no more than 4 GPUs are visible in any step
testsuite/python/tests/test_144_7.py:        r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+,\d+,\d+,\d+,", output
testsuite/python/tests/test_144_7.py:    ), "More than 4 GPUs are visible in a step"
testsuite/python/tests/test_144_7.py:    # Verify job has access to 4 GPUs
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:0 CUDA_VISIBLE_DEVICES:\d+,\d+,\d+,\d+", output)) == 4
testsuite/python/tests/test_144_7.py:    ), "Job does not have access to 4 GPUs"
testsuite/python/tests/test_144_7.py:    # Verify step 1 has 3 tasks and 2 GPUs per task
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:1 CUDA_VISIBLE_DEVICES:\d+,\d+", output)) == 3
testsuite/python/tests/test_144_7.py:    ), "Step 1 does not have 3 tasks and 2 GPUs per task"
testsuite/python/tests/test_144_7.py:    # Verify step 2 has 3 tasks and 2 GPUs per task
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:2 CUDA_VISIBLE_DEVICES:\d+,\d+", output)) == 3
testsuite/python/tests/test_144_7.py:    ), "Step 2 does not have 3 tasks and 2 GPUs per task"
testsuite/python/tests/test_144_7.py:def test_gpus_per_node_with_gpus_per_task_5():
testsuite/python/tests/test_144_7.py:    """Test --gpus-per-task option by job step"""
testsuite/python/tests/test_144_7.py:    job_gpus = 4
testsuite/python/tests/test_144_7.py:    step_gpus = 2
testsuite/python/tests/test_144_7.py:        srun -vv --exact -N1 -n1 --gpus-per-task={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -N1 -n1 --gpus-per-task={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -N1 -n1 --gpus-per-task={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -N1 -n1 --gpus-per-task={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        srun -vv --exact -N1 -n1 --gpus-per-task={step_gpus} --gpus-per-node=0 --mem=0 {step_file} &
testsuite/python/tests/test_144_7.py:        echo 'STEP_ID:'$SLURM_STEP_ID 'CUDA_VISIBLE_DEVICES:'$CUDA_VISIBLE_DEVICES
testsuite/python/tests/test_144_7.py:        f"--cpus-per-gpu=1 --gpus-per-node={job_gpus} -N2 -n5 -t1 "
testsuite/python/tests/test_144_7.py:    # Verify no more that 2 GPUs are visible in any step
testsuite/python/tests/test_144_7.py:        r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+,\d+,\d+,", output
testsuite/python/tests/test_144_7.py:    ), "More that 2 GPUs are visible in a step"
testsuite/python/tests/test_144_7.py:    # Verify all 5 steps have access to 2 GPUs
testsuite/python/tests/test_144_7.py:        len(re.findall(r"STEP_ID:\d+ CUDA_VISIBLE_DEVICES:\d+,\d+", output)) == 5
testsuite/python/tests/test_144_7.py:    ), "Not all 5 steps have access to 2 GPUs"
testsuite/python/tests/test_116_31.py:    assert re.search(r"-G, --gpus=n", output) is not None
testsuite/python/tests/test_105_1.py:    atf.require_config_parameter("Name", {"gpu": {"File": "/dev/tty0"}}, source="gres")
testsuite/python/tests/test_105_1.py:    atf.require_config_parameter_includes("GresTypes", "gpu")
testsuite/python/tests/test_105_1.py:    atf.require_nodes(1, [("Gres", "gpu:1"), ("RealMemory", 1)])
testsuite/python/tests/test_105_1.py:# Global variables set via init_gpu_vars
testsuite/python/tests/test_105_1.py:gpus_per_node = 1
testsuite/python/tests/test_105_1.py:gpu_count = 1
testsuite/python/tests/test_105_1.py:cpus_per_gpu = 1
testsuite/python/tests/test_105_1.py:gpus_per_task = 1
testsuite/python/tests/test_105_1.py:memory_per_gpu = 1
testsuite/python/tests/test_105_1.py:def init_gpu_vars():
testsuite/python/tests/test_105_1.py:    global node_count, gpus_per_node, gpu_count, task_count, gpus_per_task, mem_per_gpu
testsuite/python/tests/test_105_1.py:    nodes_with_gpus = 0
testsuite/python/tests/test_105_1.py:    min_gpus_per_node = 1024
testsuite/python/tests/test_105_1.py:            if match := re.search(r"gpu:(\d+)", node_dict["Gres"]):
testsuite/python/tests/test_105_1.py:                nodes_with_gpus += 1
testsuite/python/tests/test_105_1.py:                node_gpu_count = int(match.group(1))
testsuite/python/tests/test_105_1.py:                if node_cpu_count < node_gpu_count:
testsuite/python/tests/test_105_1.py:                    node_gpu_count = node_cpu_count
testsuite/python/tests/test_105_1.py:                if node_gpu_count < min_gpus_per_node:
testsuite/python/tests/test_105_1.py:                    min_gpus_per_node = node_gpu_count
testsuite/python/tests/test_105_1.py:    node_count = nodes_with_gpus
testsuite/python/tests/test_105_1.py:    gpus_per_node = min_gpus_per_node
testsuite/python/tests/test_105_1.py:    gpu_count = gpus_per_node * node_count
testsuite/python/tests/test_105_1.py:    if gpus_per_node % 2 == 0 and min_cpus_per_node > 1:
testsuite/python/tests/test_105_1.py:    gpus_per_task = int(gpu_count / task_count)
testsuite/python/tests/test_105_1.py:    memory_per_gpu = int(min_memory_per_node / min_gpus_per_node)
testsuite/python/tests/test_105_1.py:    if memory_per_gpu < 1:
testsuite/python/tests/test_105_1.py:            "This test requires at least one node with {min_gpus_per_node} memory"
testsuite/python/tests/test_105_1.py:def test_gpus_per_cpu(init_gpu_vars):
testsuite/python/tests/test_105_1.py:    """Test a batch job with various gpu options including ---gpus"""
testsuite/python/tests/test_105_1.py:    gpu_bind = "closest"
testsuite/python/tests/test_105_1.py:    gpu_freq = "medium"
testsuite/python/tests/test_105_1.py:        f'--cpus-per-gpu={cpus_per_gpu} --gpu-bind={gpu_bind} --gpu-freq={gpu_freq} --gpus={gpu_count} --gpus-per-node={gpus_per_node} --gpus-per-task={gpus_per_task} --mem-per-gpu={memory_per_gpu} --nodes={node_count} --ntasks={task_count} -t1 --wrap "true"',
testsuite/python/tests/test_105_1.py:    assert job_dict["CpusPerTres"] == f"gres/gpu:{cpus_per_gpu}"
testsuite/python/tests/test_105_1.py:    assert job_dict["MemPerTres"] == f"gres/gpu:{memory_per_gpu}"
testsuite/python/tests/test_105_1.py:    assert job_dict["TresBind"] == f"gres/gpu:{gpu_bind}"
testsuite/python/tests/test_105_1.py:    assert job_dict["TresFreq"] == f"gpu:{gpu_freq}"
testsuite/python/tests/test_105_1.py:    assert job_dict["TresPerJob"] == f"gres/gpu:{gpu_count}"
testsuite/python/tests/test_105_1.py:    assert job_dict["TresPerNode"] == f"gres/gpu:{gpus_per_node}"
testsuite/python/tests/test_105_1.py:    assert job_dict["TresPerTask"] == f"gres/gpu={gpus_per_task}"
testsuite/python/tests/test_105_1.py:def test_gpus_per_socket(init_gpu_vars):
testsuite/python/tests/test_105_1.py:    """Test a batch job with various gpu options including --gpus-per-socket"""
testsuite/python/tests/test_105_1.py:    gpus_per_socket = 1
testsuite/python/tests/test_105_1.py:        f'--cpus-per-gpu={cpus_per_gpu} --gpus-per-socket={gpus_per_socket} --sockets-per-node={sockets_per_node} --nodes={node_count} --ntasks={task_count} -t1 --wrap "true"',
testsuite/python/tests/test_105_1.py:    assert job_dict["TresPerSocket"] == f"gres/gpu:{gpus_per_socket}"
testsuite/python/tests/test_121_2.py:#           GresTypes=gpu,mps
testsuite/python/tests/test_121_2.py:    atf.require_config_parameter_includes("GresTypes", "gpu")
testsuite/python/tests/test_121_2.py:        "Name", {"gpu": {"File": "/dev/tty0"}, "mps": {"Count": 100}}, source="gres"
testsuite/python/tests/test_121_2.py:    atf.require_nodes(node_count, [("Gres", f"gpu:1,mps:{mps_cnt}")])
testsuite/python/tests/test_144_2.py:    atf.require_auto_config("wants to create custom gpu files and custom gres")
testsuite/python/tests/test_144_2.py:    atf.require_config_parameter("GresTypes", "gpu")
testsuite/python/tests/test_144_2.py:    atf.require_nodes(1, [("Gres", "gpu:2 Sockets=2 CoresPerSocket=1")])
testsuite/python/tests/test_144_2.py:    # GPU's need to point to existing files
testsuite/python/tests/test_144_2.py:    gpu_file = f"{str(atf.module_tmp_path)}/gpu"
testsuite/python/tests/test_144_2.py:    atf.run_command(f"touch {gpu_file + '1'}")
testsuite/python/tests/test_144_2.py:    atf.run_command(f"touch {gpu_file + '2'}")
testsuite/python/tests/test_144_2.py:        "NodeName", f"node1 Name=gpu Cores=0-1 File={gpu_file}[1-2]", source="gres"
testsuite/python/tests/test_144_2.py:def test_gpu_socket_sharing():
testsuite/python/tests/test_144_2.py:    """Test allocating multiple gpus on the same core group with enforce-binding"""
testsuite/python/tests/test_144_2.py:                    --gpus-per-task=1 scontrol show nodes node1 -d",
testsuite/python/tests/test_144_2.py:        re.search(r"GresUsed=gpu.*:2", output) is not None
testsuite/python/tests/test_144_2.py:    ), "Verify that job allocated 2 gpus"
testsuite/python/tests/test_144_2.py:def test_gpu_socket_sharing_no_alloc():
testsuite/python/tests/test_144_2.py:    """Test allocating multiple gpus on the same core group with enforce-binding without enough resources"""
testsuite/python/tests/test_144_2.py:                    --gpus-per-task=1 scontrol show nodes node1 -d",
testsuite/python/tests/test_140_1.py:    "gpu": {"acct": "gres/gpu=", "count_needed": False, "param": "gres/gpu:", "num": 1},
testsuite/python/tests/test_140_1.py:    # Requiring tty0 and tty1 to act as fake GPUs
testsuite/python/tests/test_140_1.py:            "gpu": {"File": "/dev/tty[0-1]"},
testsuite/python/tests/test_140_1.py:        "gres/gpu,gres/shard,gres/custom_gres,license/testing_license",
testsuite/python/tests/test_140_1.py:    atf.require_config_parameter_includes("GresTypes", "gpu")
testsuite/python/tests/test_140_1.py:    # Setup fake GPUs in gres.conf and add nodes to use the GPUs in slurm.conf
testsuite/python/tests/test_140_1.py:                f"gpu:2,shard:{shard_total_cnt},custom_gres:{custom_gres_total_cnt}",
testsuite/python/tests/test_121_1.py:#          Test scheduling of gres/gpu and gres/mps
testsuite/python/tests/test_121_1.py:    atf.require_config_parameter_includes("GresTypes", "gpu")
testsuite/python/tests/test_121_1.py:        {"gpu": {"File": "/dev/tty[0-1]"}, "mps": {"Count": f"{mps_cnt}"}},
testsuite/python/tests/test_121_1.py:    atf.require_nodes(2, [("Gres", f"gpu:2,mps:{mps_cnt}"), ("CPUs", 6)])
testsuite/python/tests/test_121_1.py:    echo CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES
testsuite/python/tests/test_121_1.py:    echo CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
testsuite/python/tests/test_121_1.py:    echo CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES
testsuite/python/tests/test_121_1.py:    echo CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
testsuite/python/tests/test_121_1.py:        match := re.search(r"CUDA_VISIBLE_DEVICES:(\d+)", results["stdout"])
testsuite/python/tests/test_121_1.py:    ) is not None and int(match.group(1)) == 0, "CUDA_VISIBLE_DEVICES != 0"
testsuite/python/tests/test_121_1.py:        re.search(rf"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:{job_mps}", results["stdout"])
testsuite/python/tests/test_121_1.py:    ), "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE environmental variable not correct value"
testsuite/python/tests/test_121_1.py:        re.search(r"CUDA_VISIBLE_DEVICES:\d+", file_output) is not None
testsuite/python/tests/test_121_1.py:    ), "CUDA_VISIBLE_DEVICES not found in output file"
testsuite/python/tests/test_121_1.py:    match = re.findall(r"(?s)CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:(\d+)", file_output)
testsuite/python/tests/test_121_1.py:    assert len(match) == 3, "Bad CUDA information about job (match != 3)"
testsuite/python/tests/test_121_1.py:    ), f"Bad CUDA percentage information about job (sum(map(int, match)) != {job_mps + step_mps * 2})"
testsuite/python/tests/test_121_1.py:        re.search(r"CUDA_VISIBLE_DEVICES:\d+", output) is not None
testsuite/python/tests/test_121_1.py:    ), "CUDA_VISIBLE_DEVICES not found in output"
testsuite/python/tests/test_121_1.py:    match = re.findall(r"(?s)CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:(\d+)", output)
testsuite/python/tests/test_121_1.py:    assert len(match) == 3, "Bad CUDA information about job (match != 3)"
testsuite/python/tests/test_121_1.py:    ), f"Bad CUDA percentage information about job ({sum(map(int, match))} != {job_mps + step_mps * 2})"
testsuite/python/tests/test_121_1.py:        re.search(r"CUDA_VISIBLE_DEVICES:\d+", file_output) is not None
testsuite/python/tests/test_121_1.py:    ), "CUDA_VISIBLE_DEVICES not found in output file"
testsuite/python/tests/test_121_1.py:    match = re.findall(r"(?s)CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:(\d+)", file_output)
testsuite/python/tests/test_121_1.py:    assert len(match) == 3, "Bad CUDA information about job (match != 3)"
testsuite/python/tests/test_121_1.py:    ), f"Bad CUDA percentage information about job ({sum(map(int, match))} != {step_mps * 3})"
testsuite/python/tests/test_121_1.py:        re.search(r"CUDA_VISIBLE_DEVICES:\d+", file_output) is None
testsuite/python/tests/test_121_1.py:def test_gresGPU_gresMPS_GPU_sharing(mps_nodes):
testsuite/python/tests/test_121_1.py:    """Make sure that gres/gpu and gres/mps jobs either do not share the same GPU or run at different times"""
testsuite/python/tests/test_121_1.py:    echo HOST:$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
testsuite/python/tests/test_121_1.py:    echo HOST:$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
testsuite/python/tests/test_121_1.py:        f"--gres=gpu:1 -w {mps_nodes[0]} -n1 -t1 -o {file_out1} -J 'test_job' {file_in1}"
testsuite/python/tests/test_121_1.py:        re.search(r"CUDA_VISIBLE_DEVICES:\d+", file_output) is not None
testsuite/python/tests/test_121_1.py:    ), "CUDA_VISIBLE_DEVICES not found in output file"
testsuite/python/tests/test_121_1.py:        re.search(r"gpu:\d+\(IDX:\d+\)", file_output) is not None
testsuite/python/tests/test_121_1.py:    ), "GPU device index not found in output file"
testsuite/python/tests/test_121_1.py:        re.search(r"CUDA_VISIBLE_DEVICES:\d+", file_output2) is not None
testsuite/python/tests/test_121_1.py:    ), "CUDA_VISIBLE_DEVICES not found in output2 file"
testsuite/python/tests/test_121_1.py:        re.search(rf"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:{job_mps2}", file_output2)
testsuite/python/tests/test_121_1.py:    ), f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:{job_mps2} not found in output2 file"
testsuite/python/tests/test_121_1.py:    ), "Shared mps distribution across GPU devices not found in output2 file"
testsuite/python/tests/test_144_3.py:    atf.require_config_parameter("GresTypes", "gpu")
testsuite/python/tests/test_144_3.py:    atf.require_nodes(2, [("Gres", "gpu:2"), ("CPUs", 4)])
testsuite/python/tests/test_144_3.py:    gpu_file_pattern = make_gpu_files(4)
testsuite/python/tests/test_144_3.py:        "Name", {"gpu": {"File": gpu_file_pattern}}, source="gres"
testsuite/python/tests/test_144_3.py:            srun --exact -n2 --gpus=2 --gpus-per-node=0 --mem=0 {step_path} &
testsuite/python/tests/test_144_3.py:            srun --exact -n2 --gpus=2 --gpus-per-node=0 --mem=0 {step_path} &
testsuite/python/tests/test_144_3.py:def make_gpu_files(count):
testsuite/python/tests/test_144_3.py:    """Make files in the tmp path for gpu's to point to
testsuite/python/tests/test_144_3.py:    Returns pattern TMP/gpu[1-COUNT]"""
testsuite/python/tests/test_144_3.py:        atf.run_command(f"touch {atf.module_tmp_path}/gpu{i}")
testsuite/python/tests/test_144_3.py:    return f"{atf.module_tmp_path}/gpu[1-{count}]"
testsuite/python/tests/test_144_3.py:def test_exact_gpu_full_resources():
testsuite/python/tests/test_144_3.py:        f"--cpus-per-gpu=2 --gpus-per-node=2 -N2 \
testsuite/python/tests/test_144_3.py:def test_exact_gpu_parial_resources():
testsuite/python/tests/test_144_3.py:        f"--cpus-per-gpu=1 --gpus-per-node=2 -N2 -n4 -t1 \
testsuite/python/lib/atf.py:        >>> require_config_parameter('Name', {'gpu': {'File': '/dev/tty0'}, 'mps': {'Count': 100}}, source='gres')
testsuite/python/lib/atf.py:# atf.require_nodes(2, [('Gres', 'gpu:1,mps:100')])
testsuite/python/lib/atf.py:        >>> require_nodes(2, [('CPUs', 2), ('RealMemory', 30), ('Features', 'gpu,mpi')])
testsuite/expect/test39.14:#          Increase size of job with allocated GPUs
testsuite/expect/test39.14:set gpu_cnt [get_highest_gres_count 2 "gpu"]
testsuite/expect/test39.14:if {$gpu_cnt < 1} {
testsuite/expect/test39.14:	skip "This test requires 1 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.14:# file_in1: Determine GPUs allocated, wait for dependent job to exit,
testsuite/expect/test39.14:#	expand allocation and run another GPU job
testsuite/expect/test39.14:# file_in2: Determine GPUs allocated, shrink to size 0 and exit
testsuite/expect/test39.14:# file_in3: Print the hostname and GPU IDs
testsuite/expect/test39.14:echo 'HOST:'\$SLURMD_NODENAME 'CUDA_VISIBLE_DEVICES:'\$CUDA_VISIBLE_DEVICES"
testsuite/expect/test39.14:# Submit job to expand: uses one GPU one node
testsuite/expect/test39.14:set job_id1 [submit_job -fail "-N1 --exclusive -J $test_name -t2 --gpus=1 --output=$file_out1 $file_in1"]
testsuite/expect/test39.14:# Submit job to shrink: uses one GPU one node
testsuite/expect/test39.14:set job_id2 [submit_job -fail "-N1 --exclusive -J ${test_name}_child --dependency=expand:$job_id1 -t1 --gpus=1 --output=$file_out2 $file_in2"]
testsuite/expect/test39.14:	-re "CUDA_VISIBLE_DEVICES" {
testsuite/expect/test39.14:	fail "Bad CUDA information about job 1 ($match != 3)"
testsuite/expect/test39.22:#          Test heterogeneous job GPU allocations.
testsuite/expect/test39.22:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.22:if {$gpu_cnt < 1} {
testsuite/expect/test39.22:	skip "This test requires 1 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.22:make_bash_script $file_in2 "echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES"
testsuite/expect/test39.22:spawn $salloc --gpus=$gpu_cnt --nodes=1 -t1 -J $test_name : --gpus=1 --nodes=1 $file_in1
testsuite/expect/test39.22:	-re "($number): HOST:($re_word_str) CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.22:		set gpu_alloc [cuda_count $expect_out(3,string)]
testsuite/expect/test39.22:		if {$expect_out(1,string) == 0 && $gpu_alloc == $gpu_cnt} {
testsuite/expect/test39.22:		if {$expect_out(1,string) == 1 && $gpu_alloc == 1} {
testsuite/expect/test40.1:# Request both GPUs and MPS in single request
testsuite/expect/test40.1:spawn $sbatch --gres=mps:1,gpu:1 -N1 --output=/dev/null -t1 --wrap $bin_hostname
testsuite/expect/test40.1:# Request MPS plus GPU frequency
testsuite/expect/test40.1:spawn $sbatch --gres=mps:1 --gpu-freq=high -N1 --output=/dev/null -t1 --wrap $bin_hostname
testsuite/expect/test39.12:#          Test some valid combinations of srun --gpu and non-GPU GRES options
testsuite/expect/test39.12:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.12:if {$gpu_cnt < 1} {
testsuite/expect/test39.12:	skip "This test requires 1 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.12:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.12:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.12:set sockets_with_gpus [get_gpu_socket_count $gpu_cnt $sockets_per_node]
testsuite/expect/test39.12:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.12:log_debug "Sockets with GPUs $sockets_with_gpus"
testsuite/expect/test39.12:	if {$gpu_cnt > $cores_per_node} {
testsuite/expect/test39.12:		set gpu_cnt $cores_per_node
testsuite/expect/test39.12:	if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.12:		set gpu_cnt $cpus_per_node
testsuite/expect/test39.12:set tot_gpus $gpu_cnt
testsuite/expect/test39.12:	incr tot_gpus $gpu_cnt
testsuite/expect/test39.12:set gpus_per_node $gpu_cnt
testsuite/expect/test39.12:if {$gpus_per_node > 1 && $sockets_per_node > 1} {
testsuite/expect/test39.12:	set gpus_per_socket [expr $gpus_per_node / $sockets_per_node]
testsuite/expect/test39.12:	set gpus_per_socket $gpus_per_node
testsuite/expect/test39.12:set sockets_per_node [expr $gpus_per_node / $gpus_per_socket]
testsuite/expect/test39.12:make_bash_script $file_in "echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.12:# Test --gpus options using a subset of GPUs actually available on the node
testsuite/expect/test39.12:log_info "TEST: --gpus option"
testsuite/expect/test39.12:set match_gpu 0
testsuite/expect/test39.12:if {$tot_gpus > 1} {
testsuite/expect/test39.12:	set use_gpus_per_job [expr $tot_gpus - 1]
testsuite/expect/test39.12:	set use_gpus_per_job $tot_gpus
testsuite/expect/test39.12:spawn $srun --gres=craynetwork --cpus-per-gpu=1 --gpus=$use_gpus_per_job --nodes=$nb_nodes -t1 -J $test_name -l $file_in
testsuite/expect/test39.12:	-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.12:		incr match_gpu [cuda_count $expect_out(1,string)]
testsuite/expect/test39.12:set expected_gpus $use_gpus_per_job
testsuite/expect/test39.12:subtest {$match_gpu == $expected_gpus} "Verify srun --gpus" "$match_gpu != $expected_gpus"
testsuite/expect/test39.12:if {[expr $use_gpus_per_job - 2] > $nb_nodes} {
testsuite/expect/test39.12:	log_info "TEST: --gpus option, part 2"
testsuite/expect/test39.12:	set match_gpu 0
testsuite/expect/test39.12:	incr use_gpus_per_job -2
testsuite/expect/test39.12:	spawn $srun --gres=craynetwork:1 --cpus-per-gpu=1 --gpus=$use_gpus_per_job --nodes=$nb_nodes -t1 -J $test_name -l $file_in
testsuite/expect/test39.12:		-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.12:			incr match_gpu [cuda_count $expect_out(1,string)]
testsuite/expect/test39.12:	set expected_gpus $use_gpus_per_job
testsuite/expect/test39.12:	subtest {$match_gpu == $expected_gpus} "Verify srun --gpus" "$match_gpu != $expected_gpus"
testsuite/expect/test39.12:# Test --gpus-per-node options using a subset of GPUs actually available on the node
testsuite/expect/test39.12:log_info "TEST: --gpus-per-node option"
testsuite/expect/test39.12:set match_gpu 0
testsuite/expect/test39.12:if {$gpus_per_node > 1} {
testsuite/expect/test39.12:	set use_gpus_per_node [expr $gpus_per_node - 1]
testsuite/expect/test39.12:	set use_gpus_per_node $gpus_per_node
testsuite/expect/test39.12:spawn $srun --gres=craynetwork --cpus-per-gpu=1 --gpus-per-node=$use_gpus_per_node --nodes=$nb_nodes -t1 -J $test_name -l $file_in
testsuite/expect/test39.12:	-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.12:		incr match_gpu [cuda_count $expect_out(1,string)]
testsuite/expect/test39.12:set expected_gpus [expr $use_gpus_per_node * $nb_nodes]
testsuite/expect/test39.12:subtest {$match_gpu == $expected_gpus} "Verify srun --gpus-per-node" "$match_gpu != $expected_gpus"
testsuite/expect/test39.12:# Test --gpus-per-socket options using a subset of GPUs actually available on the node
testsuite/expect/test39.12:log_info "TEST: --gpus-per-socket option"
testsuite/expect/test39.12:set match_gpu 0
testsuite/expect/test39.12:# Every node requires at least 1 GPU
testsuite/expect/test39.12:if {$use_gpus_per_job < $nb_nodes} {
testsuite/expect/test39.12:	set nb_nodes $use_gpus_per_job
testsuite/expect/test39.12:set node_list [get_nodes_by_request "--gres=craynetwork:1,gpu:1 --nodes=$nb_nodes"]
testsuite/expect/test39.12:	subskip "This test need to be able to submit jobs with at least --gres=craynetwork:1,gpu:1 --nodes=$nb_nodes"
testsuite/expect/test39.12:	set expected_gpus [expr $nb_nodes * $sockets_with_gpus]
testsuite/expect/test39.12:	if {$sockets_with_gpus > 1} {
testsuite/expect/test39.12:	spawn $srun --gres=craynetwork --gpus-per-socket=1 --sockets-per-node=$sockets_with_gpus --nodelist=[join $node_list ","] --ntasks=$nb_nodes --cpus-per-task=$cpus_per_task -t1 -J $test_name -l $file_in
testsuite/expect/test39.12:		-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.12:			incr match_gpu [cuda_count $expect_out(1,string)]
testsuite/expect/test39.12:	subtest {$match_gpu == $expected_gpus} "Verify srun --gpus-per-socket" "$match_gpu != $expected_gpus"
testsuite/expect/test39.12:# Test --gpus-per-task options using a subset of GPUs actually available on the node
testsuite/expect/test39.12:log_info "TEST: --gpus-per-task option"
testsuite/expect/test39.12:set match_gpu 0
testsuite/expect/test39.12:if {$gpu_cnt > 1} {
testsuite/expect/test39.12:	set use_gpus_per_node [expr $gpu_cnt - 1]
testsuite/expect/test39.12:	set use_gpus_per_node $gpu_cnt
testsuite/expect/test39.12:spawn $srun --gres=craynetwork --cpus-per-gpu=1 --gpus-per-task=1 -N1 --ntasks=$use_gpus_per_node -t1 -J $test_name -l $file_in
testsuite/expect/test39.12:	-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.12:		incr match_gpu [cuda_count $expect_out(1,string)]
testsuite/expect/test39.12:set expected_gpus $use_gpus_per_node
testsuite/expect/test39.12:subtest {$match_gpu == $expected_gpus} "Verify srun --gpus-per-task" "$match_gpu != $expected_gpus"
testsuite/expect/test15.27:set gpu_tot        0
testsuite/expect/test15.27:set node_name [get_nodes_by_request "--gres=gpu:2 -n1 -t1"]
testsuite/expect/test15.27:	skip "This test need to be able to submit jobs with at least --gres=gpu:2"
testsuite/expect/test15.27:if {![param_contains [get_config_param "AccountingStorageTRES"] "gres/gpu"]} {
testsuite/expect/test15.27:	skip "This test requires AccountingStorageTRES=gres/gpu"
testsuite/expect/test15.27:# Get the total number of GPUs in the test node
testsuite/expect/test15.27:set gpu_tot   [dict get [count_gres $gres_node] "gpu"]
testsuite/expect/test15.27:# Verify that all GPUs and other GRES are allocated with the --exclusive flag
testsuite/expect/test15.27:spawn $salloc -t1 -n1 -w $node_name --gres=gpu --exclusive $srun -l $bin_printenv SLURMD_NODENAME
testsuite/expect/test3.18:#          want to remove /dev/nvidia2 on a node with /dev/nvidia[0-3]).
testsuite/expect/test3.18:#          We use gres/gpu for this test since those need to be associated
testsuite/expect/test3.18:# Identify a node with gres/gpu to work with
testsuite/expect/test3.18:spawn $srun -n1 --gres=gpu:2 $file_in
testsuite/expect/test3.18:	skip "This test can't be run without a node with at least 2 gres/gpu"
testsuite/expect/test3.18:        -re "gpu:" {
testsuite/expect/test3.18:# Now try to change the count of gres/gpu to 1.  Note that a count of zero
testsuite/expect/test3.18:spawn $scontrol update NodeName=$host_name Gres=gpu:1
testsuite/expect/test1.119:# Purpose: Test --ntasks-per-gpu
testsuite/expect/test1.119:set nodes [get_nodes_by_request "--gres=gpu:2 -N2"]
testsuite/expect/test1.119:	skip "This test requires 2 or more GPUs on at least 2 nodes in the default partition"
testsuite/expect/test1.119:proc get_count_two_gpus {output} {
testsuite/expect/test1.119:		# We are only looking for two gpus so we stop here.
testsuite/expect/test1.119:# Test --ntasks-per-gpu in srun
testsuite/expect/test1.119:# Assumes 2 GPUs
testsuite/expect/test1.119:proc test_srun {expected_tot expected_gpu srun_args {srun_env ""}} {
testsuite/expect/test1.119:		set output [run_command_output "env $srun_env $srun $srun_args --quiet -t1 printenv CUDA_VISIBLE_DEVICES"]
testsuite/expect/test1.119:		set output [run_command_output "$srun $srun_args --quiet -t1 printenv CUDA_VISIBLE_DEVICES"]
testsuite/expect/test1.119:	set result [get_count_two_gpus $output]
testsuite/expect/test1.119:		subtest {$match0 == $expected_gpu} "GPU $match0_index should be bound to $expected_gpu" "$match0 != $expected_gpu"
testsuite/expect/test1.119:		subtest {$match1 == $expected_gpu} "GPU $match1_index should be bound to $expected_gpu" "$match1 != $expected_gpu"
testsuite/expect/test1.119:		# If we constrain devices then CUDA_VISIBLE_DEVICES will always equal 0 because each task will have its own cgroup.
testsuite/expect/test1.119:		subtest {$match0_index == 0} "GPU environment variable index should be 0 because it is in task cgroup"
testsuite/expect/test1.119:		subtest {$match0 == $expected_gpu*2} "GPU $match0_index should be bound to [expr {$expected_gpu * 2}]" "$match0 != $expected_gpu"
testsuite/expect/test1.119:		subtest {$match1 == 0} "There should not be any other gpu index besides 0"
testsuite/expect/test1.119:	set result [get_count_two_gpus $output]
testsuite/expect/test1.119:	subtest {$match == $expected_tasks} "Number of tasks bound to 1 GPU should be $expected_tasks" "$match != $expected_tasks"
testsuite/expect/test1.119:$srun printenv CUDA_VISIBLE_DEVICES
testsuite/expect/test1.119:# Allocate tasks to fill up the # of GPUs specified for the job
testsuite/expect/test1.119:testproc test_srun 4 2 "-N1 --gpus=2 --ntasks-per-gpu=2"
testsuite/expect/test1.119:testproc test_srun 4 2 "-N1 --gpus-per-node=2 --ntasks-per-gpu=2"
testsuite/expect/test1.119:testproc test_srun 4 2 "-N1 --gres=gpu:2 --ntasks-per-gpu=2"
testsuite/expect/test1.119:# Allocate GPUs to fill up the # of tasks specified for the job
testsuite/expect/test1.119:testproc test_srun 4 2 "-N1 -n4 --ntasks-per-gpu=2"
testsuite/expect/test1.119:testproc test_srun 4 2 "-N1 --gpus=2" "SLURM_NTASKS_PER_GPU=2"
testsuite/expect/test1.119:testproc test_srun 4 2 "-N1 -n4" "SLURM_NTASKS_PER_GPU=2"
testsuite/expect/test1.119:# Test ntasks-per-tres as well (but leave undocumented in favor of ntasks-per-gpu)
testsuite/expect/test1.119:testproc test_srun 4 2 "-N1 --gpus=2 --ntasks-per-tres=2"
testsuite/expect/test1.119:testproc test_srun 4 2 "-N1 --gpus=2" "SLURM_NTASKS_PER_TRES=2"
testsuite/expect/test1.119:# Note: sbatch does not take any input envs for --ntasks-per-[gpu|tres]
testsuite/expect/test1.119:testproc test_sbatch 4 "-N2 --ntasks-per-gpu=2 --gres=gpu:1"
testsuite/expect/test1.119:testproc test_sbatch 4 "-N1 --ntasks-per-gpu=2 --gres=gpu:2"
testsuite/expect/test1.119:testproc test_sbatch 4 "-N2 --ntasks-per-tres=2 --gres=gpu:1"
testsuite/expect/test1.119:testproc test_sbatch 4 "-N1 --ntasks-per-tres=2 --gres=gpu:2"
testsuite/expect/test1.119:testproc test_invalid "sbatch" "--ntasks-per-gpu=2 --gpus-per-task=2"
testsuite/expect/test1.119:testproc test_invalid "sbatch" "--ntasks-per-gpu=2 --gpus-per-socket=2 --sockets-per-node=1"
testsuite/expect/test1.119:testproc test_invalid "sbatch" "--ntasks-per-gpu=2 --ntasks-per-node=2"
testsuite/expect/test1.119:testproc test_invalid "sbatch" "--ntasks-per-gpu=2 --ntasks-per-tres=2"
testsuite/expect/test1.119:testproc test_invalid "srun" "--ntasks-per-gpu=2 --gpus-per-task=2"
testsuite/expect/test1.119:testproc test_invalid "srun" "--ntasks-per-gpu=2 --gpus-per-socket=2 --sockets-per-node=1"
testsuite/expect/test1.119:testproc test_invalid "srun" "--ntasks-per-gpu=2 --ntasks-per-node=2"
testsuite/expect/test1.119:testproc test_invalid "srun" "--ntasks-per-gpu=2 --ntasks-per-tres=2"
testsuite/expect/test1.119:testproc test_invalid "srun" "--ntasks-per-gpu=2" "SLURM_NTASKS_PER_TRES=2"
testsuite/expect/test1.119:testproc test_invalid "srun" "--ntasks-per-tres=2" "SLURM_NTASKS_PER_GPU=2"
testsuite/expect/test39.3:#          Test full set of srun --gpu options and scontrol show step.
testsuite/expect/test39.3:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.3:if {$gpu_cnt < 1} {
testsuite/expect/test39.3:	skip "This test requires 1 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.3:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.3:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.3:log_debug "GPUs per node count is $gpu_cnt"
testsuite/expect/test39.3:if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.3:	set gpu_cnt $cpus_per_node
testsuite/expect/test39.3:set cpus_per_gpu 1
testsuite/expect/test39.3:set gpu_bind "closest"
testsuite/expect/test39.3:set gpu_freq "medium"
testsuite/expect/test39.3:set tot_gpus $gpu_cnt
testsuite/expect/test39.3:	incr tot_gpus $gpu_cnt
testsuite/expect/test39.3:set gpus_per_node $gpu_cnt
testsuite/expect/test39.3:if {$gpus_per_node > 1 && [expr $gpus_per_node % 2] == 0 && \
testsuite/expect/test39.3:set gpus_per_task [expr $tot_gpus / $ntasks]
testsuite/expect/test39.3:set mem_per_gpu 10
testsuite/expect/test39.3:set tot_cpus [expr $tot_gpus * $cpus_per_gpu]
testsuite/expect/test39.3:# Verify if incorrect map_gpu, mask_gpu values are rejected
testsuite/expect/test39.3:set output [run_command_output -xfail "$srun --gpu-bind=verbose,mask_gpu:NaN hostname"]
testsuite/expect/test39.3:set output [run_command_output -xfail "$srun --gpu-bind=verbose,map_gpu:NaN hostname"]
testsuite/expect/test39.3:# Spawn srun job with various --gpu options
testsuite/expect/test39.3:set output [run_command_output -fail "$srun --cpus-per-gpu=$cpus_per_gpu --gpu-bind=$gpu_bind --gpu-freq=$gpu_freq --gpus=$tot_gpus --gpus-per-node=$gpus_per_node --gpus-per-task=$gpus_per_task --mem-per-gpu=$mem_per_gpu --nodes=$nb_nodes --ntasks=$ntasks -t1 -l $file_in"]
testsuite/expect/test39.3:subtest {[regexp -all "0: *CpusPerTres=gres/gpu:$cpus_per_gpu" $output] == 1} "Verify CpusPerTres"
testsuite/expect/test39.3:subtest {[regexp -all "0: *MemPerTres=gres/gpu:$mem_per_gpu" $output] == 1} "Verify MemPerTres"
testsuite/expect/test39.3:subtest {[regexp -all "0: *TresBind=gres/gpu:$gpu_bind" $output] == 1} "Verify TresBind"
testsuite/expect/test39.3:subtest {[regexp -all "0: *TresFreq=gpu:$gpu_freq" $output] == 1} "Verify TresFreq"
testsuite/expect/test39.3:subtest {[regexp -all "0: *TresPerStep=cpu:$tot_cpus,gres/gpu:$tot_gpus" $output] == 1} "Verify TresPerStep"
testsuite/expect/test39.3:subtest {[regexp -all "0: *TresPerNode=gres/gpu:$gpus_per_node" $output] == 1} "Verify TresPerNode"
testsuite/expect/test39.3:subtest {[regexp -all "0: *TresPerTask=gres/gpu=$gpus_per_task" $output] == 1} "Verify TresPerTask"
testsuite/expect/test39.3:# Spawn srun job with various --gpu options
testsuite/expect/test39.3:set gpus_per_socket 1
testsuite/expect/test39.3:set output [run_command_output -fail "$srun --cpus-per-gpu=$cpus_per_gpu --gpus-per-socket=$gpus_per_socket --sockets-per-node=$sockets_per_node --nodes=$nb_nodes --ntasks=$ntasks -t1 -l $file_in"]
testsuite/expect/test39.3:subtest {[regexp -all "0: *TresPerSocket=gres/gpu:$gpus_per_socket" $output] == 1} "Verify TresPerSocket"
testsuite/expect/test39.3:# Test srun and --gpu-bind=verbose (needs >= 2 tasks and a GPU)
testsuite/expect/test39.3:set output [run_command_output -fail "$srun -n2 --gpus=1 -l --gpu-bind=verbose,closest hostname"]
testsuite/expect/test39.3:subtest {[regexp -all "gpu-bind: usable_gres=" $output] == 2} "Verify --gpu-bind=verbose prints 2 bindings"
testsuite/expect/test40.4:	skip "This test requires 100 or more MPS per gpu on $nb_nodes nodes of the default partition"
testsuite/expect/test39.6:#          Ensure job requesting GPUs on multiple sockets gets CPUs on multiple
testsuite/expect/test39.6:		set start [string first "gpu" $gres_string $offset]
testsuite/expect/test39.6:set gpu_cnt [get_highest_gres_count 1 "gpu"]
testsuite/expect/test39.6:if {$gpu_cnt < 1} {
testsuite/expect/test39.6:	skip "This test requires 1 or more GPUs on 1 node of the default partition"
testsuite/expect/test39.6:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.6:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.6:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.6:make_bash_script $file_in "echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.6:# If $gpu_cnt > $cpus_per_node then assume there is no DefCpusPerGpu configured
testsuite/expect/test39.6:spawn $srun --cpus-per-gpu=1 --gpus-per-node=$gpu_cnt --nodes=1 --ntasks=1 -t1 -l $file_in
testsuite/expect/test39.6:	-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.6:		incr match [cuda_count $expect_out(1,string)]
testsuite/expect/test39.6:			log_warn "Allocated CPUs do not appear to span multiple sockets. This is fine if every GPU is bound to ALL sockets"
testsuite/expect/test39.6:if {$match != $gpu_cnt} {
testsuite/expect/test39.6:	fail "srun --gpus-per-node failure ($match != $gpu_cnt)"
testsuite/expect/test31.3:set test_node [get_nodes_by_request "--gres=gpu:1 -t1"]
testsuite/expect/test31.3:	subskip "This test requires being able to submit job with --gres=gpu:1"
testsuite/expect/test31.3:	log_info "Checking for GPU-related env vars"
testsuite/expect/test31.3:	set job_id [submit_job -fail "-t1 -N1 -w$test_node -o/dev/null --wrap='hostname' --gres=gpu:1"]
testsuite/expect/test31.3:	check_prolog "CUDA_VISIBLE_DEVICES="
testsuite/expect/test31.3:	check_epilog "CUDA_VISIBLE_DEVICES="
testsuite/expect/test31.3:	check_prolog "GPU_DEVICE_ORDINAL="
testsuite/expect/test31.3:	check_epilog "GPU_DEVICE_ORDINAL="
testsuite/expect/test31.3:	check_prolog "SLURM_JOB_GPUS="
testsuite/expect/test31.3:	check_epilog "SLURM_JOB_GPUS="
testsuite/expect/test31.3:	check_prolog "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="
testsuite/expect/test31.3:	check_epilog "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="
testsuite/expect/test39.4:#          Test some invalid combinations of --gpu options
testsuite/expect/test39.4:set gpu_cnt [get_highest_gres_count 1 "gpu"]
testsuite/expect/test39.4:if {$gpu_cnt < 1} {
testsuite/expect/test39.4:	skip "This test requires 1 or more GPUs on 1 node of the default partition"
testsuite/expect/test39.4:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.4:# Request more GPUs per node than exist on a single node
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus-per-node=[expr $gpu_cnt + 1] -N1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:# Specify 1 node and more GPUs than exist on a single node
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus-per-node=$gpu_cnt --gres=gpu:[expr $gpu_cnt + 1] -N1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:# Specify inconsistent --cpus-per-task and --gpus-per-task/--cpus-per-gpu
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus-per-task=1 --cpus-per-gpu=1 --cpus-per-task=2 -N1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: fatal: --cpus-per-task, --tres-per-task=cpu:#, and --cpus-per-gpu are mutually exclusive" $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus-per-task=1 --gpus-per-node=1 -n2 -N1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:# Specify gpus-per-socket, but no sockets-per-node count
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus-per-socket=1 -N1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: --gpus-per-socket option requires --sockets-per-node specification" $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus-per-task=1 --gpus-per-node=2 --ntasks-per-node=1 -N1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: Failed to validate job spec. Based on --gres=gpu/--gpus-per-node and --gpus-per-task requested number of tasks per node differ from --ntasks-per-node." $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:# Request more GPUs per node than total --gpus
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gres=gpu:2 --gpus=1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: Failed to ensure --gpus >= --gres=gpu/--gpus-per-node >= --gpus-per-socket" $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:# Request more GPUs per socket than total --gpus
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus-per-socket=2 --sockets-per-node=1 --gpus=1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: Failed to ensure --gpus >= --gres=gpu/--gpus-per-node >= --gpus-per-socket" $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:# Request --gpus-per-socket without --sockets-per-node
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus-per-socket=2 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: --gpus-per-socket option requires --sockets-per-node specification" $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:# Request more --gpus-per-task than total --gpus
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus-per-task=1 -n2 --gpus=1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: Failed to validate job spec. Based on --gpus and --gpus-per-task number of requested tasks differ from -n/--ntasks." $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:# Request --gpus not being a multiple --gres=gpu:
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gres=gpu:2 --gpus=3 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: Failed to validate job spec, --gpus is not multiple of --gres=gpu/--gpus-per-node" $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:# Request inconsistent --gres=gpu: and --gpus-per-socket
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gres=gpu:2 --gpus-per-socket=1 --sockets-per-node=1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: Failed to validate job spec. Based on --gres=gpu/--gpus-per-node and --gpus-per-socket required number of sockets differ from --sockets-per-node." $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:# Request inconsistent number of nodes (-N) with implicite --gpus and --gres=gpu:
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gres=gpu:2 --gpus=4 -N1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: Failed to validate job spec. Based on --gpu and --gres=gpu/--gpus-per-node required nodes \\(2\\) doesn't fall between min_nodes \\(1\\) and max_nodes \\(1\\) boundaries." $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:# Request --gpus not being multiple of --gpus-per-task
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus=3 --gpus-per-task=2 -N1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: Failed to validate job spec, --gpus not multiple of --gpus-per-task" $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:# Request inconsistent -n and implicite number of tasks from --gpus and --gpus-per-task
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus=4 --gpus-per-task=2 -n1 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: Failed to validate job spec. Based on --gpus and --gpus-per-task number of requested tasks differ from -n/--ntasks." $output]} "Job should be rejected at submit time"
testsuite/expect/test39.4:# Request more nodes (-N) than --gpus
testsuite/expect/test39.4:set output [run_command_output -xfail -subtest "$sbatch --gpus=1 -N2 --output=/dev/null -t1 --wrap $bin_hostname"]
testsuite/expect/test39.4:subtest {[regexp "sbatch: error: Failed to validate job spec, --gpus < -N" $output]} "Job should be rejected at submit time"
testsuite/expect/test39.13:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.13:if {$gpu_cnt < 1} {
testsuite/expect/test39.13:	skip "This test requires 1 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.13:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.13:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.13:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.13:set tot_gpus $gpu_cnt
testsuite/expect/test39.13:	incr tot_gpus $gpu_cnt
testsuite/expect/test39.13:if {$tot_gpus > 32} {
testsuite/expect/test39.13:	set tot_gpus 32
testsuite/expect/test39.13:$scontrol -dd show job \$SLURM_JOB_ID | grep gpu
testsuite/expect/test39.13:make_bash_script $file_in2 "echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES"
testsuite/expect/test39.13:# Submit job with various --gpus counters, up to 2 full nodes or 32 GPUs
testsuite/expect/test39.13:#	spawn $sbatch --cpus-per-gpu=1 --gpus=$inx -t1 -w $node_name -J $test_name --output=$ofile $file_in1
testsuite/expect/test39.13:for {set inx 1} {$inx <= $tot_gpus} {incr inx} {
testsuite/expect/test39.13:	if {$tot_gpus > $tot_cpus} {
testsuite/expect/test39.13:		# Assumes no configured DefCPUsPerGPU
testsuite/expect/test39.13:		set job_id [submit_job "--gpus=$inx -t1 -w $node_name -J $test_name --output=$ofile $file_in1"]
testsuite/expect/test39.13:		set job_id [submit_job "--cpus-per-gpu=1 --gpus=$inx -t1 -w $node_name -J $test_name --output=$ofile $file_in1"]
testsuite/expect/test39.13:for {set inx 1} {$inx <= $tot_gpus} {incr inx} {
testsuite/expect/test39.13:	set matches_list [regexp -inline -all "CUDA_VISIBLE_DEVICES:($number_commas)" $output]
testsuite/expect/test39.13:		incr match [cuda_count $devices]
testsuite/expect/test39.13:	subtest {$match == $inx} "GPU count should be $inx in output file" "$match != $inx, file: $ofile, output: $output"
testsuite/expect/test1.62:#          Test of gres/gpu plugin (if configured).
testsuite/expect/test1.62:make_bash_script $file_in {echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}
testsuite/expect/test1.62:proc run_gpu_test { gres_cnt } {
testsuite/expect/test1.62:	spawn $srun -N1 -n1 --gres=gpu:$gres_cnt -t1 $file_in
testsuite/expect/test1.62:		-re "CUDA_VISIBLE_DEVICES=($number),($number),($number)" {
testsuite/expect/test1.62:		-re "CUDA_VISIBLE_DEVICES=($number),($number)" {
testsuite/expect/test1.62:		-re "CUDA_VISIBLE_DEVICES=($number)" {
testsuite/expect/test1.62:		-re "CUDA_VISIBLE_DEVICES=" {
testsuite/expect/test1.62:			log_warn "This could indicate that gres.conf lacks device files for the GPUs"
testsuite/expect/test1.62:		log_warn "Insufficient resources to test $gres_cnt GPUs"
testsuite/expect/test1.62:		fail "Expected $gres_cnt GPUs, but was allocated $devices"
testsuite/expect/test1.62:proc run_gpu_fail { gres_spec } {
testsuite/expect/test1.62:# Test if gres/gpu is configured
testsuite/expect/test1.62:set gpu_cnt [get_highest_gres_count 1 "gpu"]
testsuite/expect/test1.62:if {$gpu_cnt < 1} {
testsuite/expect/test1.62:	skip "This test can not be run without gres/gpu configured"
testsuite/expect/test1.62:log_debug "GPUs:$gpu_cnt"
testsuite/expect/test1.62:# check count GPU devices allocated
testsuite/expect/test1.62:for {set inx 1} {$inx <= $gpu_cnt && $inx <= 3} {incr inx} {
testsuite/expect/test1.62:	run_gpu_test $inx
testsuite/expect/test1.62:run_gpu_fail "gpu:"
testsuite/expect/test1.62:run_gpu_fail "gpu::"
testsuite/expect/test1.62:run_gpu_fail "gpu:tesla:"
testsuite/expect/test1.62:run_gpu_fail "gpu:tesla:INVALID_COUNT"
testsuite/expect/test1.62:run_gpu_fail "gpu:INVALID_TYPE:"
testsuite/expect/test1.62:run_gpu_fail "gpu:INVALID_TYPE:INVALID_COUNT"
testsuite/expect/test40.5:$slurmd -N \$SLURMD_NODENAME -G 2>&1 >/dev/null | grep 'Gres Name=mps' | grep 'Index='\$CUDA_VISIBLE_DEVICES
testsuite/expect/test40.5:echo 'HOST:'\$SLURMD_NODENAME 'CUDA_VISIBLE_DEVICES:'\$CUDA_VISIBLE_DEVICES 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:'\$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
testsuite/expect/test40.5:	-re "CUDA_VISIBLE_DEVICES:($number) CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:($number)" {
testsuite/expect/test40.5:	fail "Bad CUDA information about job 1 ($match != 3)"
testsuite/expect/test40.5:# then confirm the CUDA_MPS_ACTIVE_THREAD_PERCENTAGE value above is correct
testsuite/expect/test40.5:			fail "Bad CUDA_MPS_ACTIVE_THREAD_PERCENTAGE value ($percentage != $count)"
testsuite/expect/test40.5:			log_debug "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE value is good"
testsuite/expect/test39.15:#          Test --gpus-per-tres with --overcommit option
testsuite/expect/test39.15:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.15:if {$gpu_cnt < 2} {
testsuite/expect/test39.15:	skip "This test requires 2 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.15:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.15:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.15:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.15:if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.15:	set gpu_cnt $cpus_per_node
testsuite/expect/test39.15:echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.15:spawn $srun --cpus-per-gpu=1 --gpus-per-task=1 --nodes=$nb_nodes --overcommit --ntasks=$nb_tasks -t1 -J $test_name -l $file_in
testsuite/expect/test39.15:	-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.15:		incr match [cuda_count $expect_out(1,string)]
testsuite/expect/test39.15:# NOTE: 2 tasks run on one node, so we should see 2 tasks with 2 GPUs each and 1 task with 1 GPU
testsuite/expect/test39.15:	fail "srun --gpus-per-task with --overcommit failure ($match != $nb_tasks)"
testsuite/expect/test1.12:	regexp "gres/gpu=($number)" $output - got_gres
testsuite/expect/test1.12:		subtest {$desired_gres==$got_gres} "Verify $step_id allocated TRES gres/gpu=$desired_gres"
testsuite/expect/test1.12:proc test_overlap_gpus {} {
testsuite/expect/test1.12:	  $srun --gres=gpu:1 --mem=100 $bin_sleep 60 &
testsuite/expect/test1.12:	  $srun --gres=gpu:1 --overlap --mem=100 $bin_sleep 60 &
testsuite/expect/test1.12:	  $srun --gres=gpu:1 --overlap --mem=100 $bin_sleep 60 &
testsuite/expect/test1.12:	set job_id [submit_job -fail "-N1 -n1 -w$nodes --gres=gpu:1 --mem=100 --time=1 --output=none $file_in"]
testsuite/expect/test1.12:# Only if gres/gpu is configured, and CR_*MEMORY
testsuite/expect/test1.12:if {[set_nodes_and_threads_by_request "--gres=gpu:1"] || ![param_contains [get_config_param "AccountingStorageTRES"] "*gpu"]} {
testsuite/expect/test1.12:	skip_following_testprocs "Testproc needs to be able to submit a job with --gres=gpu:1 and AccountingStorageTRES with GPUs."
testsuite/expect/test1.12:testproc test_overlap_gpus
testsuite/expect/test39.1:#          Test full set of sbatch --gpu options and scontrol show job.
testsuite/expect/test39.1:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.1:if {$gpu_cnt < 1} {
testsuite/expect/test39.1:	skip "This test requires 1 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.1:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.1:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.1:log_debug "GPUs per node count is $gpu_cnt"
testsuite/expect/test39.1:if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.1:	set gpu_cnt $cpus_per_node
testsuite/expect/test39.1:set cpus_per_gpu 1
testsuite/expect/test39.1:set gpu_bind "closest"
testsuite/expect/test39.1:set gpu_freq "medium"
testsuite/expect/test39.1:set tot_gpus $gpu_cnt
testsuite/expect/test39.1:	incr tot_gpus $gpu_cnt
testsuite/expect/test39.1:set gpus_per_node $gpu_cnt
testsuite/expect/test39.1:if {$gpus_per_node > 1 && [expr $gpus_per_node % 2] == 0 && \
testsuite/expect/test39.1:set gpus_per_task [expr $tot_gpus / $ntasks]
testsuite/expect/test39.1:set mem_per_gpu 10
testsuite/expect/test39.1:# Spawn a batch job with various --gpu options
testsuite/expect/test39.1:set job_id [submit_job -fail "--output=/dev/null --cpus-per-gpu=$cpus_per_gpu --gpu-bind=$gpu_bind --gpu-freq=$gpu_freq --gpus=$tot_gpus --gpus-per-node=$gpus_per_node --gpus-per-task=$gpus_per_task --mem-per-gpu=$mem_per_gpu --nodes=$nb_nodes --ntasks=$ntasks -t1 $file_in"]
testsuite/expect/test39.1:	-re "CpusPerTres=gres/gpu:$cpus_per_gpu" {
testsuite/expect/test39.1:	-re "MemPerTres=gres/gpu:$mem_per_gpu" {
testsuite/expect/test39.1:	-re "TresBind=gres/gpu:$gpu_bind" {
testsuite/expect/test39.1:	-re "TresFreq=gpu:$gpu_freq" {
testsuite/expect/test39.1:	-re "TresPerJob=gres/gpu:$tot_gpus" {
testsuite/expect/test39.1:	-re "TresPerNode=gres/gpu:$gpus_per_node" {
testsuite/expect/test39.1:	-re "TresPerTask=gres/gpu=$gpus_per_task" {
testsuite/expect/test39.1:	fail "sbatch gpu options not fully processed ($match != 7)"
testsuite/expect/test39.1:# Spawn a batch job with various --gpu options
testsuite/expect/test39.1:set gpus_per_socket 1
testsuite/expect/test39.1:set job_id [submit_job -fail "--output=/dev/null --cpus-per-gpu=$cpus_per_gpu --gpus-per-socket=$gpus_per_socket --sockets-per-node=$sockets_per_node --nodes=$nb_nodes --ntasks=$ntasks -t1 $file_in"]
testsuite/expect/test39.1:	-re "TresPerSocket=gres/gpu:$gpus_per_socket" {
testsuite/expect/test39.1:	fail "sbatch gpu options not fully processed ($match != 8)"
testsuite/expect/test38.18:# Purpose: Validate heterogeneous gpu job options.
testsuite/expect/test38.18:if {[get_highest_gres_count 1 "gpu"] < 2} {
testsuite/expect/test38.18:	skip "This test requires 2 or more GPUs per node in the default partition"
testsuite/expect/test38.18:proc test_gpu_bind {} {
testsuite/expect/test38.18:	log_info "Testing --gpu-bind"
testsuite/expect/test38.18:		"--gpu-bind=blah : --gpu-bind=closest" \
testsuite/expect/test38.18:		"--gpu-bind=closest : --gpu-bind=blah" \
testsuite/expect/test38.18:		"error: Invalid --gpu-bind argument: blah"
testsuite/expect/test38.18:		"error: Invalid --gpu-bind argument: blah"
testsuite/expect/test38.18:		"--gpu-bind=closest : --gpu-bind=closest" \
testsuite/expect/test38.18:		"--gpu-bind=closest : --gpu-bind=map_gpu:1" \
testsuite/expect/test38.18:		"--gpu-bind=map_gpu:1 : --gpu-bind=closest" \
testsuite/expect/test38.18:		"--gpu-bind=closest : -n1" \
testsuite/expect/test38.18:		"-n1 : --gpu-bind=closest" \
testsuite/expect/test38.18:		"^JobId=.*TresBind=gres/gpu:closest.*JobId=.*TresBind=gres/gpu:closest" \
testsuite/expect/test38.18:		"^JobId=.*TresBind=gres/gpu:closest.*JobId=.*TresBind=gres/gpu:map_gpu:1" \
testsuite/expect/test38.18:		"^JobId=.*TresBind=gres/gpu:map_gpu:1.*JobId=.*TresBind=gres/gpu:closest" \
testsuite/expect/test38.18:		"^JobId=.*TresBind=gres/gpu:closest.*JobId=" \
testsuite/expect/test38.18:		"^JobId=.*JobId=.*TresBind=gres/gpu:closest" \
testsuite/expect/test38.18:proc test_gpu_freq {} {
testsuite/expect/test38.18:	log_info "Testing --gpu-freq"
testsuite/expect/test38.18:		"--gpu-freq=blah : --gpu-freq=low" \
testsuite/expect/test38.18:		"--gpu-freq=low : --gpu-freq=blah" \
testsuite/expect/test38.18:		"error: Invalid --gpu-freq argument: gpu:blah"
testsuite/expect/test38.18:		"error: Invalid --gpu-freq argument: gpu:blah"
testsuite/expect/test38.18:		"--gpu-freq=low : --gpu-freq=low" \
testsuite/expect/test38.18:		"--gpu-freq=low : --gpu-freq=medium" \
testsuite/expect/test38.18:		"--gpu-freq=medium : --gpu-freq=low" \
testsuite/expect/test38.18:		"--gpu-freq=low : -n1" \
testsuite/expect/test38.18:		"-n1 : --gpu-freq=low" \
testsuite/expect/test38.18:		"^JobId=.*TresFreq=gpu:low.*JobId=.*TresFreq=gpu:low" \
testsuite/expect/test38.18:		"^JobId=.*TresFreq=gpu:low.*JobId=.*TresFreq=gpu:medium" \
testsuite/expect/test38.18:		"^JobId=.*TresFreq=gpu:medium.*JobId=.*TresFreq=gpu:low" \
testsuite/expect/test38.18:		"^JobId=.*TresFreq=gpu:low.*JobId=" \
testsuite/expect/test38.18:		"^JobId=.*JobId=.*TresFreq=gpu:low" \
testsuite/expect/test38.18:proc test_cpus_per_gpu {} {
testsuite/expect/test38.18:	log_info "Testing --cpus-per-gpu"
testsuite/expect/test38.18:		"--gpus=1 --cpus-per-gpu=1 : --gpus=1 --cpus-per-gpu=2" \
testsuite/expect/test38.18:		"--gpus=1 --cpus-per-gpu=2 : --gpus=1 --cpus-per-gpu=1" \
testsuite/expect/test38.18:		"--gpus=1 --cpus-per-gpu=2 : -n1" \
testsuite/expect/test38.18:		"-n1 : --gpus=1 --cpus-per-gpu=2" \
testsuite/expect/test38.18:		"^JobId=.*CpusPerTres=.*gpu:1.*JobId=.*CpusPerTres=.*gpu:2" \
testsuite/expect/test38.18:		"^JobId=.*CpusPerTres=.*gpu:2.*JobId=.*CpusPerTres=.*gpu:1" \
testsuite/expect/test38.18:		"^JobId=.*CpusPerTres=.*gpu:2.*JobId=.*" \
testsuite/expect/test38.18:		"^JobId=.*JobId=.*CpusPerTres=.*gpu:2" \
testsuite/expect/test38.18:proc test_gpus_per_job {} {
testsuite/expect/test38.18:		"-n1 --gpus=1 : -n1 --gpus=2" \
testsuite/expect/test38.18:		"-n1 --gpus=2 : -n1 --gpus=1" \
testsuite/expect/test38.18:		"-n1 --gpus=2 : -n1" \
testsuite/expect/test38.18:		"-n1 : -n1 --gpus=2" \
testsuite/expect/test38.18:		"^JobId=.*TresPerJob=.*gpu:1.*JobId=.*TresPerJob=.*gpu:2" \
testsuite/expect/test38.18:		"^JobId=.*TresPerJob=.*gpu:2.*JobId=.*TresPerJob=.*gpu:1" \
testsuite/expect/test38.18:		"^JobId=.*TresPerJob=.*gpu:2.*JobId=.*" \
testsuite/expect/test38.18:		"^JobId=.*JobId=.*TresPerJob=.*gpu:2" \
testsuite/expect/test38.18:proc test_gpus_per_node {} {
testsuite/expect/test38.18:	log_info "Testing --gpus-per-node"
testsuite/expect/test38.18:		"-n1 --gpus-per-node=1 : -n1 --gpus-per-node=2" \
testsuite/expect/test38.18:		"-n1 --gpus-per-node=2 : -n1 --gpus-per-node=1" \
testsuite/expect/test38.18:		"-n1 --gpus-per-node=2 : -n1" \
testsuite/expect/test38.18:		"-n1 : -n1 --gpus-per-node=2" \
testsuite/expect/test38.18:		"-n1 --gres=gpu:1 : -n1 --gres=gpu:2" \
testsuite/expect/test38.18:		"-n1 --gres=gpu:2 : -n1 --gres=gpu:1" \
testsuite/expect/test38.18:		"-n1 --gres=gpu:2 : -n1" \
testsuite/expect/test38.18:		"-n1 : -n1 --gres=gpu:2" \
testsuite/expect/test38.18:		"-n1 --gpus-per-node=1 --gres=gpu:2 : -n1 --gpus-per-node=2 --gres=gpu:1" \
testsuite/expect/test38.18:		"^JobId=.*TresPerNode=.*gpu:1.*JobId=.*TresPerNode=.*gpu:2" \
testsuite/expect/test38.18:		"^JobId=.*TresPerNode=.*gpu:2.*JobId=.*TresPerNode=.*gpu:1" \
testsuite/expect/test38.18:		"^JobId=.*TresPerNode=.*gpu:2.*JobId=.*" \
testsuite/expect/test38.18:		"^JobId=.*JobId=.*TresPerNode=.*gpu:2" \
testsuite/expect/test38.18:		"^JobId=.*TresPerNode=.*gpu:1.*JobId=.*TresPerNode=.*gpu:2" \
testsuite/expect/test38.18:		"^JobId=.*TresPerNode=.*gpu:2.*JobId=.*TresPerNode=.*gpu:1" \
testsuite/expect/test38.18:		"^JobId=.*TresPerNode=.*gpu:2.*JobId=.*" \
testsuite/expect/test38.18:		"^JobId=.*JobId=.*TresPerNode=.*gpu:2" \
testsuite/expect/test38.18:		"^JobId=.*TresPerNode=.*gpu:1,.*gpu:2.*JobId=.*TresPerNode=.*gpu:2,.*gpu:1" \
testsuite/expect/test38.18:proc test_gpus_per_socket {} {
testsuite/expect/test38.18:	log_info "Testing --gpus-per-socket"
testsuite/expect/test38.18:		"-n1 --gpus-per-socket=1 : -n1 " \
testsuite/expect/test38.18:		"-n1 : -n1 --gpus-per-socket=1" \
testsuite/expect/test38.18:		"--gpus-per-socket option requires --sockets-per-node specification"
testsuite/expect/test38.18:		"--gpus-per-socket option requires --sockets-per-node specification"
testsuite/expect/test38.18:		"-n1 --sockets-per-node=1 --gpus-per-socket=1 : -n1 --sockets-per-node=1 --gpus-per-socket=2" \
testsuite/expect/test38.18:		"-n1 --sockets-per-node=1 --gpus-per-socket=2 : -n1 --sockets-per-node=1 --gpus-per-socket=1" \
testsuite/expect/test38.18:		"-n1 --sockets-per-node=1 --gpus-per-socket=2 : -n1" \
testsuite/expect/test38.18:		"-n1 : -n1 --sockets-per-node=1 --gpus-per-socket=2" \
testsuite/expect/test38.18:		"^JobId=.*TresPerSocket=.*gpu:1.*JobId=.*TresPerSocket=.*gpu:2" \
testsuite/expect/test38.18:		"^JobId=.*TresPerSocket=.*gpu:2.*JobId=.*TresPerSocket=.*gpu:1" \
testsuite/expect/test38.18:		"^JobId=.*TresPerSocket=.*gpu:2.*JobId=.*" \
testsuite/expect/test38.18:		"^JobId=.*JobId=.*TresPerSocket=.*gpu:2" \
testsuite/expect/test38.18:proc test_gpus_per_task {} {
testsuite/expect/test38.18:	log_info "Testing --gpus-per-task"
testsuite/expect/test38.18:		"-n1 --gpus-per-task=1 : -n1 --gpus-per-task=2" \
testsuite/expect/test38.18:		"-n1 --gpus-per-task=2 : -n1 --gpus-per-task=1" \
testsuite/expect/test38.18:		"-n1 --gpus-per-task=2 : -n1" \
testsuite/expect/test38.18:		"-n1 : -n1 --gpus-per-task=2" \
testsuite/expect/test38.18:		"^JobId=.*TresPerTask=.*gpu=1.*JobId=.*TresPerTask=.*gpu=2" \
testsuite/expect/test38.18:		"^JobId=.*TresPerTask=.*gpu=2.*JobId=.*TresPerTask=.*gpu=1" \
testsuite/expect/test38.18:		"^JobId=.*TresPerTask=.*gpu=2.*JobId=.*" \
testsuite/expect/test38.18:		"^JobId=.*JobId=.*TresPerTask=.*gpu=2" \
testsuite/expect/test38.18:proc test_mem_per_gpu {} {
testsuite/expect/test38.18:	log_info "Testing --mem-per-gpu"
testsuite/expect/test38.18:		"-n1 --gpus=1 --mem-per-gpu=1 : -n1 --gpus=1 --mem-per-gpu=2" \
testsuite/expect/test38.18:		"-n1 --gpus=1 --mem-per-gpu=2 : -n1 --gpus=1 --mem-per-gpu=1" \
testsuite/expect/test38.18:		"-n1 --gpus=1 --mem-per-gpu=2 : -n1" \
testsuite/expect/test38.18:		"-n1 : -n1 --gpus=1 --mem-per-gpu=2" \
testsuite/expect/test38.18:		"^JobId=.*MemPerTres=.*gpu:1.*JobId=.*MemPerTres=.*gpu:2" \
testsuite/expect/test38.18:		"^JobId=.*MemPerTres=.*gpu:2.*JobId=.*MemPerTres=.*gpu:1" \
testsuite/expect/test38.18:		"^JobId=.*MemPerTres=.*gpu:2.*JobId=.*" \
testsuite/expect/test38.18:		"^JobId=.*JobId=.*MemPerTres=.*gpu:2" \
testsuite/expect/test38.18:test_gpu_bind
testsuite/expect/test38.18:test_gpu_freq
testsuite/expect/test38.18:    test_gpus_per_node
testsuite/expect/test38.18:	test_cpus_per_gpu
testsuite/expect/test38.18:	test_gpus_per_job
testsuite/expect/test38.18:	test_gpus_per_socket
testsuite/expect/test38.18:	test_gpus_per_task
testsuite/expect/test38.18:	test_mem_per_gpu
testsuite/expect/test7.17_configs/test7.17.6/slurm.conf:# GresTypes=gpu
testsuite/expect/test7.17_configs/test7.17.6/gres.conf:Name=gpu File=/dev/tty[0-3] Cores=[0-3]
testsuite/expect/test7.17_configs/test7.17.6/gres.conf:Name=gpu File=/dev/tty[4-7] Cores=[4-7]
testsuite/expect/test7.17_configs/test7.17.4/slurm.conf:GresTypes=gpu
testsuite/expect/test7.17_configs/test7.17.4/slurm.conf:Nodename=DEFAULT Sockets=2 CoresPerSocket=2 ThreadsPerCore=2 gres=gpu:8
testsuite/expect/test7.17_configs/test7.17.4/gres.conf:Name=gpu File=/dev/tty[0-3]
testsuite/expect/test7.17_configs/test7.17.4/gres.conf:Name=gpu File=/dev/tty[4-7]
testsuite/expect/test7.17_configs/test7.17.2/slurm.conf:GresTypes=gpu
testsuite/expect/test7.17_configs/test7.17.2/slurm.conf:Nodename=DEFAULT Sockets=2 CoresPerSocket=4 ThreadsPerCore=2 gres=gpu:8
testsuite/expect/test7.17_configs/test7.17.2/gres.conf:Name=gpu File=/dev/tty[0-3] Cores=0,1,2,3,4,5,6,7
testsuite/expect/test7.17_configs/test7.17.2/gres.conf:Name=gpu File=/dev/tty[4-7] Cores=8,9,10,11,12,13,14,15
testsuite/expect/test7.17_configs/test7.17.7/slurm.conf:GresTypes=gpu
testsuite/expect/test7.17_configs/test7.17.7/slurm.conf:Nodename=DEFAULT Sockets=1 CoresPerSocket=4 ThreadsPerCore=2 gres=gpu:8
testsuite/expect/test7.17_configs/test7.17.7/gres.conf:Name=gpu File=/dev/tty[0-3] Cores=[0-10000]
testsuite/expect/test7.17_configs/test7.17.7/gres.conf:Name=gpu File=/dev/tty[4-7] Cores=[4-7]
testsuite/expect/test7.17_configs/test7.17.5/slurm.conf:GresTypes=gpu
testsuite/expect/test7.17_configs/test7.17.5/slurm.conf:Nodename=DEFAULT Sockets=2 CoresPerSocket=2 ThreadsPerCore=2 gres=gpu:8
testsuite/expect/test7.17_configs/test7.17.1/slurm.conf:GresTypes=gpu
testsuite/expect/test7.17_configs/test7.17.1/slurm.conf:Nodename=DEFAULT Sockets=2 CoresPerSocket=2 ThreadsPerCore=2 gres=gpu:8
testsuite/expect/test7.17_configs/test7.17.1/gres.conf:Name=gpu File=/dev/tty[0-3] Cores=[0-3]
testsuite/expect/test7.17_configs/test7.17.1/gres.conf:Name=gpu File=/dev/tty[4-7] Cores=[4-7]
testsuite/expect/test7.17_configs/test7.17.3/slurm.conf:GresTypes=gpu
testsuite/expect/test7.17_configs/test7.17.3/slurm.conf:Nodename=DEFAULT Sockets=2 CoresPerSocket=4 ThreadsPerCore=2 gres=gpu:8
testsuite/expect/test7.17_configs/test7.17.3/gres.conf:Name=gpu File=/dev/tty0 Cores=0,1,2,3,4,5,6,7
testsuite/expect/test7.17_configs/test7.17.3/gres.conf:Name=gpu File=/dev/tty1 Cores=0,1,2,3,4,5,6,7
testsuite/expect/test7.17_configs/test7.17.3/gres.conf:Name=gpu File=/dev/tty2 Cores=0,1,2,3,4,5,6,7
testsuite/expect/test7.17_configs/test7.17.3/gres.conf:Name=gpu File=/dev/tty3 Cores=0,1,2,3,4,5,6,7
testsuite/expect/test7.17_configs/test7.17.3/gres.conf:Name=gpu File=/dev/tty4 Cores=8,9,10,11,12,13,14,15
testsuite/expect/test7.17_configs/test7.17.3/gres.conf:Name=gpu File=/dev/tty5 Cores=8,9,10,11,12,13,14,15
testsuite/expect/test7.17_configs/test7.17.3/gres.conf:Name=gpu File=/dev/tty6 Cores=8,9,10,11,12,13,14,15
testsuite/expect/test7.17_configs/test7.17.3/gres.conf:Name=gpu File=/dev/tty7 Cores=8,9,10,11,12,13,14,15
testsuite/expect/globals:set gpu_sock_list {}
testsuite/expect/globals:#		The usual coma-separated list of Gres (e.g gpu:2,craynetwork:1).
testsuite/expect/globals:#	For example: node1 has 1 GPU, node2 has 2 GPUs and node3 has 3 GPUs
testsuite/expect/globals:#	[get_highest_gres_count 1 "gpu"] returns 3 (i.e. 1 node 3 GPUs)
testsuite/expect/globals:#	[get_highest_gres_count 2 "gpu"] returns 2 (i.e. 2 nodes have at least 2 GPUs each)
testsuite/expect/globals:#	[get_highest_gres_count 3 "gpu"] returns 1 (i.e. 3 nodes have at least 1 GPU each)
testsuite/expect/globals:#	_set_gpu_socket_inx - adds a socket index to the gpu_sock_list if not already on it
testsuite/expect/globals:#	_set_gpu_socket_inx sock_inx
testsuite/expect/globals:#	Add a socket index to the array gpu_sock_list if not already
testsuite/expect/globals:#	on the list. Subroutine used by get_gpu_socket_count
testsuite/expect/globals:proc _set_gpu_socket_inx { sock_inx } {
testsuite/expect/globals:	global gpu_sock_list
testsuite/expect/globals:		set gpu_sock_list [lreplace $gpu_sock_list 0 99]
testsuite/expect/globals:	set sock_cnt [llength $gpu_sock_list]
testsuite/expect/globals:		if {[lindex $gpu_sock_list $i] == $sock_inx} {
testsuite/expect/globals:	lappend gpu_sock_list $sock_inx
testsuite/expect/globals:# Subroutine used by get_gpu_socket_count
testsuite/expect/globals:# Add a socket index to the array gpu_sock_list if not already
testsuite/expect/globals:proc _set_gpu_socket_range { sock_first_inx sock_last_inx } {
testsuite/expect/globals:	global gpu_sock_list
testsuite/expect/globals:	set sock_cnt [llength $gpu_sock_list]
testsuite/expect/globals:			if {[lindex $gpu_sock_list $i] == $s} {
testsuite/expect/globals:			lappend gpu_sock_list $s
testsuite/expect/globals:#	get_gpu_socket_count - returns the number of sockets with GPUS on a node with the given per-node GPU count
testsuite/expect/globals:#	get_gpu_socket_count gpu_cnt sockets_per_node
testsuite/expect/globals:#	Given a per-node GPU count, return the number of sockets with
testsuite/expect/globals:#	GPUs on a node with the given per-node GPU count.
testsuite/expect/globals:proc get_gpu_socket_count { gpu_cnt sockets_per_node } {
testsuite/expect/globals:	global test_dir re_word_str bin_rm number scontrol srun gpu_sock_list
testsuite/expect/globals:	set sockets_with_gpus 1
testsuite/expect/globals:	set file_in "$test_dir/test_get_gpu_socket_count"
testsuite/expect/globals:	_set_gpu_socket_inx -1
testsuite/expect/globals:	spawn $srun -N1 --gres=gpu:$gpu_cnt $file_in
testsuite/expect/globals:		-re "gpu:${number}.S:($number)-($number)" {
testsuite/expect/globals:			_set_gpu_socket_range $expect_out(1,string) $expect_out(2,string)
testsuite/expect/globals:		-re "gpu:${re_word_str}:${number}.S:($number),($number),($number),($number)" {
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(1,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(2,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(3,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(4,string)
testsuite/expect/globals:		-re "gpu:${re_word_str}:${number}.S:($number),($number),($number)" {
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(1,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(2,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(3,string)
testsuite/expect/globals:		-re "gpu:${re_word_str}:${number}.S:($number),($number)" {
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(1,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(2,string)
testsuite/expect/globals:		-re "gpu:${re_word_str}:${number}.S:($number)" {
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(1,string)
testsuite/expect/globals:		-re "gpu:${number}.S:($number),($number),($number),($number)" {
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(1,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(2,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(3,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(4,string)
testsuite/expect/globals:		-re "gpu:${number}.S:($number),($number),($number)" {
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(1,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(2,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(3,string)
testsuite/expect/globals:		-re "gpu:${number}.S:($number),($number)" {
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(1,string)
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(2,string)
testsuite/expect/globals:		-re "gpu:${number}.S:($number)" {
testsuite/expect/globals:			_set_gpu_socket_inx $expect_out(1,string)
testsuite/expect/globals:	set sock_cnt [llength $gpu_sock_list]
testsuite/expect/globals:		set sockets_with_gpus $sock_cnt
testsuite/expect/globals:	return $sockets_with_gpus
testsuite/expect/globals:#	get_highest_mps_count - get_highest_gres_count nodes mps, but for "mps per GPU"
testsuite/expect/globals:#	For a given number of nodes, returns the higest number of MPS per GPU
testsuite/expect/globals:	# We cannot use get_highest_gres_count because we need "per gpu",
testsuite/expect/globals:	# so we get all the mps per node and all gpus per node, to create
testsuite/expect/globals:	# a mps_per_gpu list to sort and get the count.
testsuite/expect/globals:	set gpu_dict [get_gres_count "gpu" $available_nodes]
testsuite/expect/globals:	set mps_per_gpu [list]
testsuite/expect/globals:			if [dict exists $gpu_dict $node] {
testsuite/expect/globals:				set gpu [dict get $gpu_dict $node]
testsuite/expect/globals:				if { $gpu > 0 } {
testsuite/expect/globals:					lappend mps_per_gpu [expr $mps / $gpu]
testsuite/expect/globals:					fail "All nodes with MPS should have a GPU"
testsuite/expect/globals:				fail "All nodes with MPS should have a GPU"
testsuite/expect/globals:	set count [lindex [lsort -decreasing -integer $mps_per_gpu] [expr $node_count - 1]]
testsuite/expect/globals:#	cuda_count - determines the count of allocated GPUs
testsuite/expect/globals:#	cuda_count cuda_string
testsuite/expect/globals:#	cuda_string
testsuite/expect/globals:#		Contents of a CUDA_VISIBLE_DEVICES environment variable
testsuite/expect/globals:#	Return the number of GPUs or -1 on error
testsuite/expect/globals:proc cuda_count { cuda_string } {
testsuite/expect/globals:	set cuda_count 0
testsuite/expect/globals:	set len [string length $cuda_string]
testsuite/expect/globals:		set cuda_char [string index $cuda_string $char_inx]
testsuite/expect/globals:		if {[string match , $cuda_char]} {
testsuite/expect/globals:				incr cuda_count
testsuite/expect/globals:				log_error "Invalid input ($cuda_string)"
testsuite/expect/globals:		} elseif {[string is digit $cuda_char]} {
testsuite/expect/globals:		incr cuda_count
testsuite/expect/globals:		log_error "Invalid input ($cuda_string)"
testsuite/expect/globals:	return $cuda_count
testsuite/expect/globals:#		e.g. "--gres=gpu:1 -n1 -t1"
testsuite/expect/globals:#		variables. For example "-env 'SLURM_NTASKS_PER_GPU=2'".
testsuite/expect/test39.19:#          Test accounting for GPU resources with various allocation options
testsuite/expect/test39.19:# Validate the job, batch step and step 0 of a job have the proper GPU counts
testsuite/expect/test39.19:# No step to test if step_gpus == -1
testsuite/expect/test39.19:proc test_acct { job_id job_gpus step_gpus req_gpus have_gpu_types batch_gpus } {
testsuite/expect/test39.19:	log_debug "Job $job_id Expecting job GPUs:$job_gpus  Step GPUs:$step_gpus"
testsuite/expect/test39.19:	wait_for_command_match -fail "$sacct -X -n -o ReqTRES --parsable2 -j $job_id" "gres/gpu"
testsuite/expect/test39.19:	if {$step_gpus != -1} {
testsuite/expect/test39.19:	# Check and count reported gpus on the step
testsuite/expect/test39.19:	if {$step_gpus != -1} {
testsuite/expect/test39.19:		set gpus_reported_count 0
testsuite/expect/test39.19:			if {$have_gpu_types} {
testsuite/expect/test39.19:				foreach {{} gpu_count} [regexp -all -inline {gres/gpu:[^=]+=(\d+)} $tres_value] {
testsuite/expect/test39.19:					subtest {$gpu_count == $step_gpus} "Step GPUs reported by sacct should be $step_gpus" "$gpu_count != $step_gpus"
testsuite/expect/test39.19:					incr gpus_reported_count
testsuite/expect/test39.19:				foreach {{} gpu_count} [regexp -all -inline {gres/gpu=(\d+)} $tres_value] {
testsuite/expect/test39.19:					subtest {$gpu_count == $step_gpus} "Step GPUs reported by sacct should be $step_gpus" "$gpu_count != $step_gpus"
testsuite/expect/test39.19:					incr gpus_reported_count
testsuite/expect/test39.19:		subtest {$gpus_reported_count == 1} "sacct should report step GPUs 1 time" "found $gpus_reported_count times"
testsuite/expect/test39.19:	# Check and count reported batch gpus on the job
testsuite/expect/test39.19:	set gpus_reported_count 0
testsuite/expect/test39.19:		if {$have_gpu_types} {
testsuite/expect/test39.19:			foreach {{} gpu_count} [regexp -all -inline {gres/gpu:[^=]+=(\d+)} $tres_value] {
testsuite/expect/test39.19:				subtest {$gpu_count == $batch_gpus} "Batch GPUs reported by sacct should be $batch_gpus" "$gpu_count != $batch_gpus"
testsuite/expect/test39.19:				incr gpus_reported_count
testsuite/expect/test39.19:			foreach {{} gpu_count} [regexp -all -inline {gres/gpu=(\d+)} $tres_value] {
testsuite/expect/test39.19:				subtest {$gpu_count == $batch_gpus} "Batch GPUs reported by sacct should be $batch_gpus" "$gpu_count != $batch_gpus"
testsuite/expect/test39.19:				incr gpus_reported_count
testsuite/expect/test39.19:	subtest {$gpus_reported_count == 1} "sacct should report batch GPUs 1 time" "found $gpus_reported_count times"
testsuite/expect/test39.19:	# Check and count reported gpus on the job
testsuite/expect/test39.19:	set gpus_reported_count 0
testsuite/expect/test39.19:		if {$have_gpu_types} {
testsuite/expect/test39.19:			foreach {{} gpu_count} [regexp -all -inline {gres/gpu:[^=]+=(\d+)} $tres_value] {
testsuite/expect/test39.19:				subtest {$gpu_count == $job_gpus} "Job GPUs reported by sacct should be $job_gpus" "$gpu_count != $job_gpus"
testsuite/expect/test39.19:				incr gpus_reported_count
testsuite/expect/test39.19:			foreach {{} gpu_count} [regexp -all -inline {gres/gpu=(\d+)} $tres_value] {
testsuite/expect/test39.19:				subtest {$gpu_count == $job_gpus} "Job GPUs reported by sacct should be $job_gpus" "$gpu_count != $job_gpus"
testsuite/expect/test39.19:				incr gpus_reported_count
testsuite/expect/test39.19:	subtest {$gpus_reported_count == 2} "sacct should report job GPUs 2 times" "found $gpus_reported_count times"
testsuite/expect/test39.19:# Validate the job, batch step and step 0 of a job have the proper GPU counts
testsuite/expect/test39.19:# No step to test if step_gpus == -1
testsuite/expect/test39.19:		-re "AllocTRES=.*,gres/gpu=($number)" {
testsuite/expect/test39.19:		-re "AllocTRES=.*,gres/gpu:($re_word_str)=($number)" {
testsuite/expect/test39.19:	subtest {$match == $target} "GPUs accounted should be $target" "$match != $target"
testsuite/expect/test39.19:# Helper function to find $batch_gpus from different outputs
testsuite/expect/test39.19:proc get_batch_gpus { file_out } {
testsuite/expect/test39.19:	set batch_gpus "unknown"
testsuite/expect/test39.19:		# >Nodes=74dc179a_n1 CPU_IDs=0-1 Mem=150 GRES=[[gpu:2]](IDX:0-1)<
testsuite/expect/test39.19:		#  Nodes=74dc179a_n2 CPU_IDs=0-1 Mem=150 GRES=gpu:1(IDX:0)
testsuite/expect/test39.19:	if {![regexp {gpu:(?:[^:( ]+:)?(\d+)} $node_line - batch_gpus]} {
testsuite/expect/test39.19:	return $batch_gpus
testsuite/expect/test39.19:set store_gpu [string first "gres/gpu" $store_tres]
testsuite/expect/test39.19:if {$store_gpu == -1} {
testsuite/expect/test39.19:	skip "This test requires accounting for GPUs"
testsuite/expect/test39.19:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.19:if {$gpu_cnt < 2} {
testsuite/expect/test39.19:	skip "This test requires 2 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.19:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.19:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.19:set sockets_with_gpus [get_gpu_socket_count $gpu_cnt $sockets_per_node]
testsuite/expect/test39.19:log_debug "GPUs per node is $gpu_cnt"
testsuite/expect/test39.19:log_debug "Sockets with GPUs $sockets_with_gpus"
testsuite/expect/test39.19:# Test --gpus-per-node option by job
testsuite/expect/test39.19:log_info "TEST 1: --gpus-per-node option by job"
testsuite/expect/test39.19:set req_gpus 2
testsuite/expect/test39.19:set target [expr $nb_nodes * $req_gpus]
testsuite/expect/test39.19:set job_id [submit_job -fail "--gres=craynetwork:0 --gpus-per-node=$req_gpus -N$nb_nodes -t1 -o $file_out -J $test_name $file_in1"]
testsuite/expect/test39.19:set batch_gpus [get_batch_gpus $file_out]
testsuite/expect/test39.19:set have_gpu_types 0
testsuite/expect/test39.19:	-re "AllocTRES=.*,gres/gpu=($number)" {
testsuite/expect/test39.19:	-re "AllocTRES=.*,gres/gpu:($re_word_str)=($number)" {
testsuite/expect/test39.19:			set have_gpu_types 1
testsuite/expect/test39.19:subtest {$match == $target} "GPUs accounted should be $target" "$match != $target"
testsuite/expect/test39.19:test_acct $job_id $target -1 $req_gpus $have_gpu_types $batch_gpus
testsuite/expect/test39.19:# Test --gpus option by job
testsuite/expect/test39.19:log_info "TEST 2: --gpus option by job"
testsuite/expect/test39.19:if {$nb_nodes >= 2 || $gpu_cnt >= 3} {
testsuite/expect/test39.19:set job_id [submit_job -fail "--gres=craynetwork:0 --gpus=$target -N$nb_nodes -t1 -o $file_out -J $test_name $file_in1"]
testsuite/expect/test39.19:set batch_gpus [get_batch_gpus $file_out]
testsuite/expect/test39.19:test_acct $job_id $target -1 $target $have_gpu_types $batch_gpus
testsuite/expect/test39.19:# Test --gpus-per-task option by job
testsuite/expect/test39.19:log_info "TEST 3: --gpus-per-task option by job"
testsuite/expect/test39.19:set req_gpus 1
testsuite/expect/test39.19:set job_id [submit_job -fail "--gres=craynetwork:0 --gpus-per-task=$req_gpus -N$nb_nodes -n$nb_tasks -t1 -o $file_out -J $test_name $file_in1"]
testsuite/expect/test39.19:set batch_gpus [get_batch_gpus $file_out]
testsuite/expect/test39.19:test_acct $job_id $nb_tasks -1 $req_gpus $have_gpu_types $batch_gpus
testsuite/expect/test39.19:# Test --gpus-per-socket option by job
testsuite/expect/test39.19:log_info "TEST 4: --gpus-per-socket option by job"
testsuite/expect/test39.19:if {$sockets_with_gpus >= 2} {
testsuite/expect/test39.19:set req_gpus 1
testsuite/expect/test39.19:set job_id [submit_job -fail "--gres=craynetwork:0 --gpus-per-socket=$req_gpus -N$nb_nodes  --ntasks=$nb_nodes --sockets-per-node=$nb_sockets --cpus-per-task=$cpus_per_task -t1 -o $file_out -J $test_name $file_in1"]
testsuite/expect/test39.19:set batch_gpus [get_batch_gpus $file_out]
testsuite/expect/test39.19:test_acct $job_id $target -1 $req_gpus $have_gpu_types $batch_gpus
testsuite/expect/test39.19:# Test --gpus-per-node option by step
testsuite/expect/test39.19:log_info "TEST 5: --gpus-per-node option by step"
testsuite/expect/test39.19:set req_gpus 2
testsuite/expect/test39.19:set target [expr $nb_nodes * $req_gpus]
testsuite/expect/test39.19:set job_id [submit_job -fail "--gres=craynetwork:0 --gpus-per-node=$req_gpus -N$nb_nodes -t1 -o $file_out -J $test_name $file_in1"]
testsuite/expect/test39.19:set batch_gpus [get_batch_gpus $file_out]
testsuite/expect/test39.19:test_acct $job_id $target $target $req_gpus $have_gpu_types $batch_gpus
testsuite/expect/test39.19:# Test --gpus option by step
testsuite/expect/test39.19:log_info "TEST 6: --gpus option by step"
testsuite/expect/test39.19:if {$nb_nodes >= 2 || $gpu_cnt >= 3} {
testsuite/expect/test39.19:set job_id [submit_job -fail "--gres=craynetwork:0 --gpus=$target -N$nb_nodes -t1 -o $file_out -J $test_name $file_in1"]
testsuite/expect/test39.19:set batch_gpus [get_batch_gpus $file_out]
testsuite/expect/test39.19:test_acct $job_id $target $target $target $have_gpu_types $batch_gpus
testsuite/expect/test39.19:# Test --gpus-per-task option by step
testsuite/expect/test39.19:log_info "TEST 7: --gpus-per-task option by step"
testsuite/expect/test39.19:set req_gpus 1
testsuite/expect/test39.19:set job_id [submit_job -fail "--gres=craynetwork:0 --gpus-per-task=$req_gpus -N$nb_nodes -n$nb_tasks -t1 -o $file_out -J $test_name $file_in1"]
testsuite/expect/test39.19:set batch_gpus [get_batch_gpus $file_out]
testsuite/expect/test39.19:test_acct $job_id $nb_tasks $nb_tasks $req_gpus $have_gpu_types $batch_gpus
testsuite/expect/test39.19:# Test --gpus-per-socket option by step
testsuite/expect/test39.19:log_info "TEST 8: --gpus-per-socket option by step"
testsuite/expect/test39.19:if {$sockets_with_gpus >= 2} {
testsuite/expect/test39.19:set req_gpus 1
testsuite/expect/test39.19:set job_id [submit_job -fail "--gres=craynetwork:0 --gpus-per-socket=$req_gpus -N$nb_nodes  --ntasks=$nb_nodes --sockets-per-node=$nb_sockets --cpus-per-task=$cpus_per_task -t1 -o $file_out -J $test_name $file_in1"]
testsuite/expect/test39.19:set batch_gpus [get_batch_gpus $file_out]
testsuite/expect/test39.19:test_acct $job_id $target $target $req_gpus $have_gpu_types $batch_gpus
testsuite/expect/test39.19:# Test --gpus-per-task option by step
testsuite/expect/test39.19:log_info "TEST 9: --gpus-per-task option by step"
testsuite/expect/test39.19:set req_gpus 1
testsuite/expect/test39.19:set job_id [submit_job -fail "--gres=craynetwork:0 --gpus-per-task=$req_gpus -N$nb_nodes -n$job_tasks -t1 -o $file_out -J $test_name $file_in1"]
testsuite/expect/test39.19:set batch_gpus [get_batch_gpus $file_out]
testsuite/expect/test39.19:test_acct $job_id $job_tasks $step_tasks $req_gpus $have_gpu_types $batch_gpus
testsuite/expect/test17.17:set gpu_tot      0
testsuite/expect/test17.17:set node_name [get_nodes_by_request "--gres=gpu:2 -n1 -t1"]
testsuite/expect/test17.17:	skip "This test need to be able to submit jobs with at least --gres=gpu:2"
testsuite/expect/test17.17:if {![param_contains [get_config_param "AccountingStorageTRES"] "gres/gpu"]} {
testsuite/expect/test17.17:	skip "This test requires AccountingStorageTRES=gres/gpu"
testsuite/expect/test17.17:# Get the total number of GPUs in the test node
testsuite/expect/test17.17:set gpu_tot   [dict get [count_gres $gres_node] "gpu"]
testsuite/expect/test17.17:# Verify that all GPUs are allocated with the --exclusive flag
testsuite/expect/test17.17:set job_id2 [submit_job -fail "-t1 -N1 -w $node_name --gres=gpu --exclusive --output=$file_out $file_in"]
testsuite/expect/test39.8:#          Test --gpu-bind options
testsuite/expect/test39.8:set gpu_cnt [get_highest_gres_count 1 "gpu"]
testsuite/expect/test39.8:if {$gpu_cnt < 2} {
testsuite/expect/test39.8:	skip "This test requires 2 or more GPUs on 1 node of the default partition"
testsuite/expect/test39.8:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.8:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.8:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.8:make_bash_script $file_in "echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.8:	spawn $srun --cpus-per-gpu=1 --gpus-per-socket=1 --sockets-per-node=2 -n2 --accel-bind=g -J $test_name -t1 $file_in
testsuite/expect/test39.8:	# Test of --gpu-bind=closest
testsuite/expect/test39.8:	spawn $srun --cpus-per-gpu=1 --gpus-per-socket=1 --sockets-per-node=2 -n2 --gpu-bind=closest -J $test_name -t1 $file_in
testsuite/expect/test39.8:# Test of --gpu-bind=map_gpu
testsuite/expect/test39.8:# Note that if the task count exceeds the provided map_gpu, the map will be cycled over for additional tasks
testsuite/expect/test39.8:if {$gpu_cnt < 4} {
testsuite/expect/test39.8:	set map_gpu "map_gpu:1,0"
testsuite/expect/test39.8:	set map_gpu "map_gpu:1,0,3,2"
testsuite/expect/test39.8:# If $gpu_cnt > $cpus_per_node then assume there is no DefCpusPerGpu configured
testsuite/expect/test39.8:if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.8:	spawn $srun --gpus-per-node=$gpu_cnt --ntasks=$tasks_per_node -N1 -l --gpu-bind=$map_gpu -J $test_name -l -t1 $file_in
testsuite/expect/test39.8:	spawn $srun --cpus-per-gpu=1 --gpus-per-node=$gpu_cnt --ntasks=$tasks_per_node -N1 -l --gpu-bind=$map_gpu -J $test_name -l -t1 $file_in
testsuite/expect/test39.8:	-re "($number): HOST:($re_word_str) CUDA_VISIBLE_DEVICES:($number)" {
testsuite/expect/test39.8:subtest {$matches == $match_goal} "Verify --gpu-bind=$map_gpu is respected" "$matches != $match_goal"
testsuite/expect/test39.8:# Test of --gpu-bind=mask_gpu
testsuite/expect/test39.8:# Note that if the task count exceeds the provided mask_gpu, the mask will be cycled over for additional tasks
testsuite/expect/test39.8:if {$gpu_cnt < 4} {
testsuite/expect/test39.8:	set mask_gpu "mask_gpu:0x3,0x2"
testsuite/expect/test39.8:	set mask_gpu "mask_gpu:0x3,0x2,0x5,0xF"
testsuite/expect/test39.8:# If $gpu_cnt > $cpus_per_node then assume there is no DefCpusPerGpu configured
testsuite/expect/test39.8:if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.8:	spawn $srun --gpus-per-node=$gpu_cnt --ntasks=$tasks_per_node -N1 -l --gpu-bind=$mask_gpu -J $test_name -l -t1 $file_in
testsuite/expect/test39.8:	spawn $srun --cpus-per-gpu=1 --gpus-per-node=$gpu_cnt --ntasks=$tasks_per_node -N1 -l --gpu-bind=$mask_gpu -J $test_name -l -t1 $file_in
testsuite/expect/test39.8:	-re "($number): HOST:($re_word_str) CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.8:subtest {$matches == $match_goal} "Verify --gpu-bind=$mask_gpu is respected" "$matches != $match_goal"
testsuite/expect/test7.17.prog.c:	orig_config = "gpu:8";
testsuite/expect/test38.19:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test38.19:if {$gpu_cnt < 2} {
testsuite/expect/test38.19:        skip "This test requires 2 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test38.19:	set job_id [submit_job -fail "-N2 --gpus=2 -o $file_out --wrap=\"$srun --gpus=1 echo ok : --gpus=1 echo ok\""]
testsuite/expect/test39.7:#          Test --cpus-per-gpu option
testsuite/expect/test39.7:proc run_gpu_per_job { cpus_per_gpu } {
testsuite/expect/test39.7:	spawn $srun --gpus=1 --cpus-per-gpu=$cpus_per_gpu -J $test_name -t1 $file_in
testsuite/expect/test39.7:	if {$cpu_count < $cpus_per_gpu} {
testsuite/expect/test39.7:		fail "srun --cpus-per-gpu failure ($cpu_count < $cpus_per_gpu)"
testsuite/expect/test39.7:proc run_gpu_per_node { cpus_per_gpu } {
testsuite/expect/test39.7:	spawn $srun --gpus-per-node=1 -N1 --cpus-per-gpu=$cpus_per_gpu -J $test_name -t1 $file_in
testsuite/expect/test39.7:	if {$cpu_count < $cpus_per_gpu} {
testsuite/expect/test39.7:		fail "srun --cpus-per-gpu failure ($cpu_count < $cpus_per_gpu)"
testsuite/expect/test39.7:proc run_gpu_per_task { cpus_per_gpu } {
testsuite/expect/test39.7:	spawn $srun --gpus-per-task=1 -n1 --cpus-per-gpu=$cpus_per_gpu -J $test_name -t1 $file_in
testsuite/expect/test39.7:	if {$cpu_count < $cpus_per_gpu} {
testsuite/expect/test39.7:		fail "srun --cpus-per-gpu failure ($cpu_count < $cpus_per_gpu)"
testsuite/expect/test39.7:set gpu_cnt [get_highest_gres_count 1 "gpu"]
testsuite/expect/test39.7:if {$gpu_cnt < 1} {
testsuite/expect/test39.7:	skip "This test requires 1 or more GPUs on 1 node of the default partition"
testsuite/expect/test39.7:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.7:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.7:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.7:make_bash_script $file_in "echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.7:# Run test job with global GPU count
testsuite/expect/test39.7:# Double cpus_per_gpu value on each iteration
testsuite/expect/test39.7:	run_gpu_per_job $inx
testsuite/expect/test39.7:# Run test job with gpus-per-node count
testsuite/expect/test39.7:# Double cpus_per_gpu value on each iteration
testsuite/expect/test39.7:	run_gpu_per_node $inx
testsuite/expect/test39.7:# Run test job with gpus-per-task count
testsuite/expect/test39.7:# Double cpus_per_gpu value on each iteration
testsuite/expect/test39.7:	run_gpu_per_task $inx
testsuite/expect/test39.7:# Run test with --gpus=2 and cpus_per_gpu value that pushed job to 2 nodes
testsuite/expect/test39.7:if {$gpu_cnt > 1 && $nb_nodes > 1} {
testsuite/expect/test39.7:	spawn $srun --gpus=2 --cpus-per-gpu=$cpus_per_node -J $test_name -t1 $file_in
testsuite/expect/test39.7:		fail "srun --cpus-per-gpu failure, bad CPU count ($cpu_count < $cpu_target)"
testsuite/expect/test39.7:		fail "srun --cpus-per-gpu failure, bad node count ($node_count < $node_target)"
testsuite/expect/test39.9:#          Test --gpu-freq options
testsuite/expect/test39.9:set freq_parse_nvml "GpuFreq=memory_freq:($number),graphics_freq:($number)"
testsuite/expect/test39.9:set freq_parse_generic "GpuFreq=control_disabled"
testsuite/expect/test39.9:set generic_msg "The gpu/generic plugin is loaded, so Slurm can't really test GPU frequency operations. Please set `Autodetect=nvml` in gres.conf to load the gpu/nvml plugin instead."
testsuite/expect/test39.9:set not_supported_msg "This test requires a GPU that supports frequency scaling."
testsuite/expect/test39.9:	skip "NVML must be installed and enabled to test GPU frequency operations"
testsuite/expect/test39.9:	skip "SlurmdUser must be root to test GPU frequency operations"
testsuite/expect/test39.9:set gpu_cnt [get_highest_gres_count 1 "gpu"]
testsuite/expect/test39.9:if {$gpu_cnt < 1} {
testsuite/expect/test39.9:	skip "This test requires 1 or more GPU on 1 node of the default partition"
testsuite/expect/test39.9:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.9:# the GPU on TEST 1.
testsuite/expect/test39.9:make_bash_script $file_in "echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.9:# Test of --gpu-freq=low,verbose
testsuite/expect/test39.9:spawn $srun --gpus-per-node=1 --gpu-freq=low,verbose -J $test_name -t1 $file_in
testsuite/expect/test39.9:subtest {$match == 1} "Verify srun --gpu-freq=low,verbose" "$match != 1"
testsuite/expect/test39.9:# Test of --gpu-freq=medium,memory=medium,verbose
testsuite/expect/test39.9:spawn $srun --gpus-per-node=1 --gpu-freq=medium,memory=medium,verbose -J $test_name -t1 $file_in
testsuite/expect/test39.9:subtest {$match == 1} "Verify srun --gpu-freq=medium,memory=medium,verbose" "$match != 1"
testsuite/expect/test39.9:# Test of --gpu-freq=highm1,verbose
testsuite/expect/test39.9:spawn $srun --gpus-per-node=1 --gpu-freq=highm1,verbose -J $test_name -t1 $file_in
testsuite/expect/test39.9:subtest {$match == 1} "Verify srun --gpu-freq=highm1,verbose" "$match != 1"
testsuite/expect/test39.9:# Test of --gpu-freq=high,memory=high,verbose
testsuite/expect/test39.9:spawn $srun --gpus-per-node=1 --gpu-freq=high,memory=high,verbose -J $test_name -t1 $file_in
testsuite/expect/test39.9:if {[subtest {$match == 2} "Verify srun --gpu-freq=x,memory=x,verbose" "$match != 2"]} {
testsuite/expect/test39.9:	spawn $srun -w $hostname --gpus-per-node=1 --gpu-freq=medium,memory=medium,verbose -J $test_name -t1 $file_in
testsuite/expect/test39.9:	spawn $srun -w $hostname --gpus-per-node=1 --gpu-freq=low,memory=low,verbose -J $test_name -t1 $file_in
testsuite/expect/test39.9:	if {[subtest {$match == 4} "Verify srun --gpu-freq=x,memory=x,verbose" "$match != 4"]} {
testsuite/expect/test39.9:# Test of --gpu-freq=verbose
testsuite/expect/test39.9:# Frequency will be system default (see "GpuFreqDef" in slurm.conf)
testsuite/expect/test39.9:spawn $srun --gpus-per-node=1 --gpu-freq=verbose -J $test_name -t1 $file_in
testsuite/expect/test39.9:subtest {$match == 1} "Verify srun --gpu-freq=verbose" "$match != 1"
testsuite/expect/test1.14:# Verify that all GPUs and other GRES are allocated with the --exclusive flag
testsuite/expect/test1.14:proc test_gpus {node_name} {
testsuite/expect/test1.14:	# Get the total number of GPUs in the test node
testsuite/expect/test1.14:	set gpu_tot   [dict get [count_gres $gres_node] "gpu"]
testsuite/expect/test1.14:	# Verify that all GPUs and other GRES are allocated with the --exclusive flag
testsuite/expect/test1.14:	set job_id [submit_job -fail "-n1 -N1 -w $node_name --gres=gpu --exclusive -e none -o none --wrap '$bin_sleep 10'"]
testsuite/expect/test1.14:set node_name [get_nodes_by_request "--gres=gpu:2 -n1 -t1"]
testsuite/expect/test1.14:	skip_following_testprocs "This test need to be able to submit jobs with at least --gres=gpu:2"
testsuite/expect/test1.14:if {![param_contains [get_config_param "AccountingStorageTRES"] "gres/gpu"]} {
testsuite/expect/test1.14:	skip_following_testprocs "This test requires AccountingStorageTRES=gres/gpu"
testsuite/expect/test1.14:testproc test_gpus $node_name
testsuite/expect/test40.6:$slurmd -N \$SLURMD_NODENAME -G 2>&1 >/dev/null | grep 'Gres Name=mps' | grep 'Index='\$CUDA_VISIBLE_DEVICES
testsuite/expect/test40.6:echo 'HOST:'\$SLURMD_NODENAME 'CUDA_VISIBLE_DEVICES:'\$CUDA_VISIBLE_DEVICES 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:'\$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
testsuite/expect/test40.6:	-re "CUDA_VISIBLE_DEVICES:($number) CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:($number)" {
testsuite/expect/test40.6:	fail "Bad CUDA information about job 1 ($match != 3)"
testsuite/expect/test40.6:# then confirm the CUDA_MPS_ACTIVE_THREAD_PERCENTAGE value above is correct
testsuite/expect/test40.6:			fail "Bad CUDA_MPS_ACTIVE_THREAD_PERCENTAGE value ($percentage != $count)"
testsuite/expect/test40.6:			log_debug "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE value is good"
testsuite/expect/test39.18.prog.c: *  Test gres.conf and system GPU normalization and merging logic.
testsuite/expect/test39.18.prog.c: * 		fake_gpus.conf.
testsuite/expect/test39.18.prog.c: * 		slurm.conf for the node. E.g., `gpu:4`.
testsuite/expect/test39.18.prog.c: *	GresTypes=gpu,mps,nic,mic,tmpdisk
testsuite/expect/test39.18.prog.c: * However, gres.conf and fake_gpus.conf do need to be re-created for each test.
testsuite/expect/test39.18.prog.c:	char *fake_gpus_conf = NULL;
testsuite/expect/test39.18.prog.c:	xstrfmtcat(fake_gpus_conf, "%s/%s", etc_dir, "fake_gpus.conf");
testsuite/expect/test39.18.prog.c:	if (stat(fake_gpus_conf, &stat_buf) != 0) {
testsuite/expect/test39.18.prog.c:		printf("FAILURE: Could not find fake_gpus_conf file at %s\n",
testsuite/expect/test39.18.prog.c:		       fake_gpus_conf);
testsuite/expect/test39.18.prog.c:	printf("fake_gpus_conf: %s\n", fake_gpus_conf);
testsuite/expect/test39.18.prog.c:	xfree(fake_gpus_conf);
testsuite/expect/test40.8.prog.cu:	// Allocate Unified Memory – accessible from CPU or GPU
testsuite/expect/test40.8.prog.cu:	if (cudaMallocManaged(&x, N * sizeof(float)) != cudaSuccess) {
testsuite/expect/test40.8.prog.cu:	if (cudaMallocManaged(&y, N * sizeof(float)) != cudaSuccess) {
testsuite/expect/test40.8.prog.cu:	// Run kernel on 256 elements at a time on the GPU
testsuite/expect/test40.8.prog.cu:	// Wait for GPU to finish before accessing on host
testsuite/expect/test40.8.prog.cu:	cudaDeviceSynchronize();
testsuite/expect/test40.8.prog.cu:	cudaFree(x);
testsuite/expect/test40.8.prog.cu:	cudaFree(y);
testsuite/expect/test39.23:#          Test --gpus-per-task with implicit task count.
testsuite/expect/test39.23:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.23:if {$gpu_cnt < 1} {
testsuite/expect/test39.23:	skip "This test requires 1 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.23:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.23:echo HOST:\$SLURMD_NODENAME NODE_CNT:\$SLURM_NNODES CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.23:make_bash_script $file_in2 "echo HOST:\$SLURMD_NODENAME NODE_CNT:\$SLURM_NNODES CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.23:# One GPU per task with node count (without range)
testsuite/expect/test39.23:log_info "TEST: One GPU per task with node count (without range)"
testsuite/expect/test39.23:spawn $srun --nodes=$nb_nodes --gpus-per-task=1 -t1 -J $test_name -l $file_in1
testsuite/expect/test39.23:	-re "NODE_CNT:($number_commas) CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.23:		incr match [cuda_count $expect_out(2,string)]
testsuite/expect/test39.23:	fail "srun --gpus-per-task failure ($match != $node_cnt)"
testsuite/expect/test39.23:# Two GPUs per task with node count (without range)
testsuite/expect/test39.23:if {$gpu_cnt > 1} {
testsuite/expect/test39.23:	log_info "TEST: Two GPUs per task with node count (without range)"
testsuite/expect/test39.23:	spawn $srun --nodes=$nb_nodes --gpus-per-task=2 -t1 -J $test_name -l $file_in1
testsuite/expect/test39.23:		-re "NODE_CNT:($number_commas) CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.23:			incr match [cuda_count $expect_out(2,string)]
testsuite/expect/test39.23:	set exp_gpu_cnt [expr $node_cnt * 2]
testsuite/expect/test39.23:	if {$match != $exp_gpu_cnt} {
testsuite/expect/test39.23:		fail "srun --gpus-per-task failure ($match != $exp_gpu_cnt)"
testsuite/expect/test39.23:# One GPU per task with node count range and task count resulting in uneven task distribution
testsuite/expect/test39.23:if {$gpu_cnt > 1 && $nb_nodes > 1} {
testsuite/expect/test39.23:	spawn $srun --nodes=2-$nb_nodes --ntasks=$task_cnt --gpus-per-task=1 -t1 -J $test_name -l $file_in1
testsuite/expect/test39.23:		-re "NODE_CNT:($number_commas) CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.23:			incr match [cuda_count $expect_out(2,string)]
testsuite/expect/test39.23:		fail "srun --gpus-per-task failure ($match != $task_cnt)"
testsuite/expect/test39.23:# One task and GPU for each GPU available at step level
testsuite/expect/test39.23:if {$gpu_cnt > 1} {
testsuite/expect/test39.23:	log_info "TEST: One task and GPU for each GPU available at step level"
testsuite/expect/test39.23:# FIXME: RANGE CHECK CPU/GPU COUNT
testsuite/expect/test39.23:	spawn $salloc -N1 --gpus=$gpu_cnt -t1 -J $test_name $srun -n $gpu_cnt -O --gpus-per-task=1 $file_in2
testsuite/expect/test39.23:		-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.23:			incr match [cuda_count $expect_out(1,string)]
testsuite/expect/test39.23:	set exp_gpu_cnt [expr $task_cnt]
testsuite/expect/test39.23:		fail "srun --gpus-per-task test failed to run"
testsuite/expect/test39.23:	} elseif {$match != $exp_gpu_cnt} {
testsuite/expect/test39.23:		fail "srun --gpus-per-task failure ($match != $exp_gpu_cnt)"
testsuite/expect/test39.23:# One task and two GPUs as resources available at step level
testsuite/expect/test39.23:if {$gpu_cnt > 1} {
testsuite/expect/test39.23:	log_info "TEST: One task and two GPUs as resources available at step level"
testsuite/expect/test39.23:	set tasks_spawned [expr $gpu_cnt / 2]
testsuite/expect/test39.23:	spawn $salloc -N1 --exclusive --gpus=$gpu_cnt -t1 -J $test_name $srun -n $tasks_spawned -O --gpus-per-task=2 $file_in2
testsuite/expect/test39.23:		-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.23:			incr match [cuda_count $expect_out(1,string)]
testsuite/expect/test39.23:	set exp_gpu_cnt [expr $task_cnt * 2]
testsuite/expect/test39.23:		fail "srun --gpus-per-task test failed to run"
testsuite/expect/test39.23:	} elseif {$match != $exp_gpu_cnt} {
testsuite/expect/test39.23:		fail "srun --gpus-per-task failure ($match != $exp_gpu_cnt)"
testsuite/expect/test39.23:# Step allocation of GPUs based upon CPUs per task
testsuite/expect/test39.23:if {$gpu_cnt > 1 && $num_cpus > 0} {
testsuite/expect/test39.23:	log_info "TEST: Step allocation of GPUs based upon CPUs per task"
testsuite/expect/test39.23:	if {$gpu_cnt > $num_cpus} {
testsuite/expect/test39.23:		set gpus_per_task [expr $gpu_cnt / $num_cpus]
testsuite/expect/test39.23:		set gpus_per_task 1
testsuite/expect/test39.23:	spawn $salloc -N1 --exclusive -w $hostname --gpus=$gpu_cnt -t1 -J $test_name $srun -c $cpus_per_task --gpus-per-task=$gpus_per_task $file_in2
testsuite/expect/test39.23:		-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.23:			incr match [cuda_count $expect_out(1,string)]
testsuite/expect/test39.23:	set exp_gpu_cnt [expr $task_cnt *$task_cnt * $gpus_per_task]
testsuite/expect/test39.23:		fail "srun --gpus-per-task test failed to run"
testsuite/expect/test39.23:	} elseif {$match != $exp_gpu_cnt} {
testsuite/expect/test39.23:		fail "srun --gpus-per-task failure ($match != $exp_gpu_cnt)"
testsuite/expect/test39.2:#          Test full set of salloc --gpu options and scontrol show job.
testsuite/expect/test39.2:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.2:if {$gpu_cnt < 1} {
testsuite/expect/test39.2:	skip "This test requires 1 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.2:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.2:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.2:log_debug "GPUs per node count is $gpu_cnt"
testsuite/expect/test39.2:if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.2:	set gpu_cnt $cpus_per_node
testsuite/expect/test39.2:set cpus_per_gpu 1
testsuite/expect/test39.2:set gpu_bind "closest"
testsuite/expect/test39.2:set gpu_freq "medium"
testsuite/expect/test39.2:set tot_gpus $gpu_cnt
testsuite/expect/test39.2:	incr tot_gpus $gpu_cnt
testsuite/expect/test39.2:set gpus_per_node $gpu_cnt
testsuite/expect/test39.2:if {$gpus_per_node > 1 && [expr $gpus_per_node % 2] == 0 && \
testsuite/expect/test39.2:set gpus_per_task [expr $tot_gpus / $ntasks]
testsuite/expect/test39.2:set mem_per_gpu 10
testsuite/expect/test39.2:# Spawn salloc job with various --gpu options
testsuite/expect/test39.2:set output [run_command_output -fail "$salloc --cpus-per-gpu=$cpus_per_gpu --gpu-bind=$gpu_bind --gpu-freq=$gpu_freq --gpus=$tot_gpus --gpus-per-node=$gpus_per_node --gpus-per-task=$gpus_per_task --mem-per-gpu=$mem_per_gpu --nodes=$nb_nodes --ntasks=$ntasks -t1 $file_in"]
testsuite/expect/test39.2:subtest {[regexp -all "CpusPerTres=gres/gpu:$cpus_per_gpu" $output] == 1} "Verify CpusPerTres"
testsuite/expect/test39.2:subtest {[regexp -all "MemPerTres=gres/gpu:$mem_per_gpu" $output] == 1} "Verify MemPerTres"
testsuite/expect/test39.2:subtest {[regexp -all "TresBind=gres/gpu:$gpu_bind" $output] == 1} "Verify TresBind"
testsuite/expect/test39.2:subtest {[regexp -all "TresFreq=gpu:$gpu_freq" $output] == 1} "Verify TresFreq"
testsuite/expect/test39.2:subtest {[regexp -all "TresPerJob=gres/gpu:$tot_gpus" $output] == 1} "Verify TresPerJob"
testsuite/expect/test39.2:subtest {[regexp -all "TresPerNode=gres/gpu:$gpus_per_node" $output] == 1} "Verify TresPerNode"
testsuite/expect/test39.2:subtest {[regexp -all "TresPerTask=gres/gpu=$gpus_per_task" $output] == 1} "Verify TresPerTask"
testsuite/expect/test39.2:# Spawn a salloc job with various --gpu options
testsuite/expect/test39.2:set gpus_per_socket 1
testsuite/expect/test39.2:set output [run_command_output -fail "$salloc --cpus-per-gpu=$cpus_per_gpu --gpus-per-socket=$gpus_per_socket --sockets-per-node=$sockets_per_node --nodes=$nb_nodes --ntasks=$ntasks -t1 $file_in"]
testsuite/expect/test39.2:subtest {[regexp -all "TresPerSocket=gres/gpu:$gpus_per_socket" $output] == 1} "Verify TresPerSocket"
testsuite/expect/test39.2:# Test salloc propagating --gpu-bind=verbose to srun (needs >= 2 tasks and a
testsuite/expect/test39.2:# GPU)
testsuite/expect/test39.2:set output [run_command_output -fail "$salloc --gpu-bind=verbose,closest -n2 --gpus=1 $srun -l hostname"]
testsuite/expect/test39.2:subtest {[regexp -all "gpu-bind: usable_gres=" $output] == 2} "Verify --gpu-bind=verbose prints 2 bindings"
testsuite/expect/test39.18:# Purpose:  Test gres.conf-specified and system-detected GPU device merging
testsuite/expect/test39.18:set dup_err		"error: gpu duplicate device file name"
testsuite/expect/test39.18:set mismatch_err	"error: This GPU specified in \\\[slurm\\\|gres\\\].conf has mismatching Cores or Links"
testsuite/expect/test39.18:set flags_no_gpu_err	"fatal: Invalid GRES record name=${re_word_str} type=${re_word_str}: Flags (${re_word_str}) contains \"no_gpu_env\", which must be mutually exclusive to all other GRES env flags of same node and name"
testsuite/expect/test39.18:set flags_default	"HAS_FILE,ENV_NVML,ENV_RSMI,ENV_ONEAPI,ENV_OPENCL,ENV_DEFAULT"
testsuite/expect/test39.18:set flags_default_type	"HAS_FILE,HAS_TYPE,ENV_NVML,ENV_RSMI,ENV_ONEAPI,ENV_OPENCL,ENV_DEFAULT"
testsuite/expect/test39.18:set flags_default_type_shared "HAS_FILE,HAS_TYPE,ENV_NVML,ENV_RSMI,ENV_ONEAPI,ENV_OPENCL,ENV_DEFAULT,SHARED,ONE_SHARING"
testsuite/expect/test39.18:	# This is all we need to trigger loading the GRES GPU plugin
testsuite/expect/test39.18:	GresTypes=gpu,mps,nic,mic,tmpdisk
testsuite/expect/test39.18:set dev "$test_dir/nvidia"
testsuite/expect/test39.18:# fake_gpus_conf -	The fake_gpus.conf to use. This file tells Slurm to
testsuite/expect/test39.18:proc test_cfg {test_minor slurm_conf_gres gres_conf fake_gpus_conf output_expected {err_msgs ""} {errs_expected 0} } {
testsuite/expect/test39.18:	generate_file $fake_gpus_conf $test_dir/fake_gpus.conf
testsuite/expect/test39.18:	# The order of GPUs is important because it directly corresponds to the
testsuite/expect/test39.18:	# bits of the GRES bitmaps used to track the GPUs, and this shouldn't
testsuite/expect/test39.18:	# Also, we will eventually want to sort GPUs by PCI bus ID if AutoDetect
testsuite/expect/test39.18:	# also want to guarantee that the GPU order in gres.conf is preserved if
testsuite/expect/test39.18:# fake_gpus.conf is of the following format, with each line representing one
testsuite/expect/test39.18:# GPU device:
testsuite/expect/test39.18:# # Test a2 - Type-less gpu specification in slurm.conf and empty gres.conf
testsuite/expect/test39.18:set slurm_conf_gres "gpu:4"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-1|(null)|${dev}1|$flags_file
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-1|(null)|${dev}2|$flags_file
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|2-3|(null)|${dev}3|$flags_file
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|2-3|(null)|${dev}4|$flags_file
testsuite/expect/test39.18:testproc_alias "a2" test_cfg "a2" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# # Test a4 - Test empty and null identifiers in fake_gpus.conf
testsuite/expect/test39.18:set slurm_conf_gres "gpu:8"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|(null)|-1,0|(null)|
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-1|-1,0|(null)|
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|(null)|(null)|(null)|
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-1|(null)|(null)|
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|(null)|-1,0|${dev}3|$flags_file
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-1|-1,0|${dev}4|$flags_file
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|(null)|(null)|${dev}1|$flags_file
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-1|(null)|${dev}2|$flags_file
testsuite/expect/test39.18:testproc_alias "a4" test_cfg "a4" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# In fake_gpus_conf, a CPU range starting with `~` will trigger the GPU plugin's
testsuite/expect/test39.18:# gpu_p_test_cpu_conv(). In order for this to exercise gpu/nvml-specific code,
testsuite/expect/test39.18:# gpu/generic will be used and CPU ranges will be set to null, failing the tests
testsuite/expect/test39.18:set slurm_conf_gres "gpu:1"
testsuite/expect/test39.18:set fake_gpus_conf "# This file was autogenerated by $test_name
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|$cpus_count|0-$cpus_count_m1|(null)|${dev}1|$flags_file
testsuite/expect/test39.18:testproc_alias "a6" test_cfg "a6" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:1"
testsuite/expect/test39.18:set fake_gpus_conf "# This file was autogenerated by $test_name
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|$cpus_count|0-$cpus_count_m1|(null)|${dev}1|$flags_file
testsuite/expect/test39.18:testproc_alias "a8" test_cfg "a8" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:1"
testsuite/expect/test39.18:set fake_gpus_conf "# This file was autogenerated by $test_name
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|$cpus_count|0-$cpus_count_m1|(null)|${dev}1|$flags_file
testsuite/expect/test39.18:testproc_alias "a10" test_cfg "a10" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:1"
testsuite/expect/test39.18:set fake_gpus_conf "# This file was autogenerated by $test_name
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|$cpus_count|0-$cpus_count_m1|(null)|${dev}1|$flags_file
testsuite/expect/test39.18:testproc_alias "a12" test_cfg "a12" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:1"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:testproc_alias "a14" test_cfg "a14" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:1"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|16|0|(null)|${dev}1|$flags_file
testsuite/expect/test39.18:testproc_alias "a16" test_cfg "a16" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:1"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-1|(null)|${dev}1|$flags_file
testsuite/expect/test39.18:testproc_alias "a18" test_cfg "a18" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:1"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|16|8-15|(null)|${dev}1|$flags_file
testsuite/expect/test39.18:testproc_alias "a20" test_cfg "a20" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:4"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-1|(null)|${dev}1|$flags_file
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-1|(null)|${dev}2|$flags_file
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|2-3|(null)|${dev}3|$flags_file
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|2-3|(null)|${dev}4|$flags_file
testsuite/expect/test39.18:testproc_alias "a22" test_cfg "a22" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:v100:4"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):v100|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):v100|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):v100|4|2-3|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):v100|4|2-3|(null)|${dev}4|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a24" test_cfg "a24" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:TESLA_V100:4"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):TESLA_V100|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):TESLA_V100|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):TESLA_V100|4|2-3|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):TESLA_V100|4|2-3|(null)|${dev}4|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a26" test_cfg "a26" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:pcie-16gb:4"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):pcie-16gb|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):pcie-16gb|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):pcie-16gb|4|2-3|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):pcie-16gb|4|2-3|(null)|${dev}4|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a28" test_cfg "a28" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla_v100-pcie-16gb:4"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_v100-pcie-16gb|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_v100-pcie-16gb|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_v100-pcie-16gb|4|2-3|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_v100-pcie-16gb|4|2-3|(null)|${dev}4|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a30" test_cfg "a30" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:k20m:1,gpu:k20m1:1,gpu:v100:1"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20m1|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20m|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):v100|4|0-1|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a32" test_cfg "a32" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:p100:1"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:nvidia-p100|4|0-1|(null)|${dev}2
testsuite/expect/test39.18:nvidia-p100|4|0-1|(null)|${dev}6
testsuite/expect/test39.18:nvidia-p100|4|0-1|(null)|${dev}3
testsuite/expect/test39.18:nvidia-p100|4|0-1|(null)|${dev}4
testsuite/expect/test39.18:nvidia-p100|4|0-1|(null)|${dev}1
testsuite/expect/test39.18:nvidia-p100|4|0-1|(null)|${dev}5
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):p100|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a34" test_cfg "a34" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:aaaa:1,gpu:a:1,gpu:aa:1,gpu:aaa:1"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):aaa|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):aa|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a|4|0-1|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):aaaa|4|0-1|(null)|${dev}4|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a36" test_cfg "a36" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:k20:2,gpu:k20m:4"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20m|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20|4|0-1|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20m|4|0-1|(null)|${dev}4|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20m|4|0-1|(null)|${dev}7|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20m|4|0-1|(null)|${dev}8|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a38" test_cfg "a38" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:k20:2,gpu:k20m:2"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20m|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20m|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20|4|0-1|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):k20|4|0-1|(null)|${dev}4|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a39" test_cfg "a39" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# 		drain, since the node reports one less aaa GPU than the
testsuite/expect/test39.18:set slurm_conf_gres "gpu:aaa:3,gpu:bbb:2,gpu:ccc:1"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):aaa|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):aaa|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):bbb|4|0-1|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):bbb|4|0-1|(null)|${dev}4|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):ccc|4|0-1|(null)|${dev}5|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a40" test_cfg "a40" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:aaa:1,gpu:aaa:1,gpu:aaa:1"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):aaa|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):aaa|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):aaa|4|0-1|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a42" test_cfg "a42" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# The node returns a list of only 3 GPUs, not 4 (since gpu:special is not
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:3,gpu:special:1"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}2|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a52" test_cfg "a52" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# gpu:special is added on via gres.conf. The final GPU list count matches what
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:3,gpu:special:1"
testsuite/expect/test39.18:Name=gpu Type=special File=${dev}5
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):special|4|(null)|(null)|${dev}5|$flags_default_type
testsuite/expect/test39.18:testproc_alias "a53" test_cfg "a53" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# The gres conf record for nvidia1 *does* match a system GPU with the same Type
testsuite/expect/test39.18:# the system device is omitted from the final GPU list. The total GPU count is 3
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:3,gpu:special:1"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=0
testsuite/expect/test39.18:Name=gpu Type=special File=${dev}5
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):special|4|(null)|(null)|${dev}5|$flags_default_type
testsuite/expect/test39.18:testproc_alias "a54" test_cfg "a54" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# nvidia[0-2] matches exactly what is found on the system, so no problem there.
testsuite/expect/test39.18:# tesla + nvidia3 does not match any type and file combo found in the system
testsuite/expect/test39.18:# GPUs, so this is assumed to be an “extra” GPU. However, it is not added, since
testsuite/expect/test39.18:# The total GPU count is 5 instead of 4, which is fine (i.e. won't set node to
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:3,gpu:special:1"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[0-2\] Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3 Cores=0
testsuite/expect/test39.18:Name=gpu Type=special File=${dev}5
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):special|4|(null)|(null)|${dev}5|$flags_default_type
testsuite/expect/test39.18:testproc_alias "a55" test_cfg "a55" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:# nvidia[0-2] are found on the system, but nvidia3 is not. However, nvidia3 is
testsuite/expect/test39.18:# assumed to be an extra GPU (like gpu:special), so it’s ok.
testsuite/expect/test39.18:# The total GPU count is 5, so there are no errors or warnings.
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:4,gpu:special:1"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[0-3\] Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=special File=${dev}5
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):special|4|(null)|(null)|${dev}5|$flags_default_type
testsuite/expect/test39.18:testproc_alias "a56" test_cfg "a56" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[0-3\] Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=special File=${dev}5
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:testproc_alias "a57" test_cfg "a57" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:# Test a58 - Test that a non-GPU GRES doesn't need an explicit entry in
testsuite/expect/test39.18:# gres.conf. Also test that GPUs are rejected unless they have a File
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:3,gpu:special:1,tmpdisk:100"
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "a58" test_cfg "a58" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:2,gpu:gtx:1"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[0-4\]
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}5
testsuite/expect/test39.18:Name=gpu Type=special File=${dev}6
testsuite/expect/test39.18:Name=gpu Type=gtx File=${dev}7
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|-1|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):gtx|4|(null)|(null)|${dev}7|$flags_default_type
testsuite/expect/test39.18:testproc_alias "a59" test_cfg "a59" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:4"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:tesla|4|0-1|(null)|${dev}1|MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/7/0
testsuite/expect/test39.18:tesla|4|0-1|(null)|${dev}2|MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/10/0
testsuite/expect/test39.18:tesla|4|2-3|(null)|${dev}3|MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/8/0
testsuite/expect/test39.18:tesla|4|2-3|(null)|${dev}4|MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/9/0
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|(null)|${dev}1|MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/7/0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|(null)|${dev}2|MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/10/0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}3|MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/8/0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}4|MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/9/0|$flags_file_type
testsuite/expect/test39.18:testproc_alias "a60" test_cfg "a60" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# # Test b2 - Test that all MPS is distributed across multiple GPU types
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:1,gpu:1080:1,gpu:gtx560:1,mps:300"
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):gtx560|4|0-1|(null)|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):1080|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:testproc_alias "b2" test_cfg "b2" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:ti:3,gpu:gtx:3"
testsuite/expect/test39.18:Name=gpu Type=ti File=${dev}0 COREs=0
testsuite/expect/test39.18:Name=gpu Type=ti File=${dev}\[1-2\] COREs=0-1
testsuite/expect/test39.18:Name=gpu Type=gtx File=${dev}\[3-5\] COREs=0-1 Links=-1,0,0,0,0,0
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):gtx|4|0-1|-1,0,0,0,0,0|${dev}5|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):ti|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):ti|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:testproc_alias "b4" test_cfg "b4" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:# # Test b5 - Test that GPUs are sorted according to links, not device file
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:8"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}0 COREs=0-1  Links=0,0,0,-1,0,0,0,0
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}1 COREs=0-1  Links=0,0,-1,0,0,0,0,0
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}2 COREs=0-1  Links=0,-1,0,0,0,0,0,0
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}3 COREs=0-1  Links=-1,0,0,0,0,0,0,0
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}4 COREs=2-3 Links=0,0,0,0,0,0,0,-1
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}5 COREs=2-3 Links=0,0,0,0,0,0,-1,0
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}6 COREs=2-3 Links=0,0,0,0,0,-1,0,0
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}7 COREs=2-3 Links=0,0,0,0,-1,0,0,0
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|-1,0,0,0,0,0,0,0|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|0,-1,0,0,0,0,0,0|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|0,0,-1,0,0,0,0,0|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|0,0,0,-1,0,0,0,0|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|2-3|0,0,0,0,-1,0,0,0|${dev}7|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|2-3|0,0,0,0,0,-1,0,0|${dev}6|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|2-3|0,0,0,0,0,0,-1,0|${dev}5|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|2-3|0,0,0,0,0,0,0,-1|${dev}4|$flags_file_type
testsuite/expect/test39.18:testproc_alias "b5" test_cfg "b5" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# # Test b6 - Test that "extra" GPUs are still used when not found on system
testsuite/expect/test39.18:set slurm_conf_gres "gpu:1080:3"
testsuite/expect/test39.18:Name=gpu Type=1080 File=${dev}\[0-2\] COREs=0-1
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):1080|4|0-1|(null)|${dev}0|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):1080|4|0-1|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):1080|4|0-1|(null)|${dev}2|$flags_default_type
testsuite/expect/test39.18:testproc_alias "b6" test_cfg "b6" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# # Test b7 - Test that GPUs are sorted according links, if specified, and
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:8"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}0 COREs=0-1  Links=0,0,0,-1
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}1 COREs=0-1  Links=0,0,-1,0
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}2 COREs=0-1  Links=0,-1,0,0
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}3 COREs=0-1  Links=-1,0,0,0
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}4 COREs=2-3
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}5 COREs=2-3
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}6 COREs=2-3
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}7 COREs=2-3
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|-1,0,0,0|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|0,-1,0,0|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|0,0,-1,0|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|0,0,0,-1|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|2-3|(null)|${dev}4|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|2-3|(null)|${dev}5|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|2-3|(null)|${dev}6|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|2-3|(null)|${dev}7|$flags_file_type
testsuite/expect/test39.18:testproc_alias "b7" test_cfg "b7" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# # Test b8 - Test that separate "extra" GPUs in gres.conf with different Cores
testsuite/expect/test39.18:set slurm_conf_gres "gpu:5"
testsuite/expect/test39.18:Name=gpu File=${dev}\[0-1\] Cores=0,1
testsuite/expect/test39.18:Name=gpu File=${dev}\[2-3\] Cores=0,1 Links=-1
testsuite/expect/test39.18:Name=gpu File=${dev}4 Cores=0
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0,1|-1|${dev}2|$flags_default
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0,1|-1|${dev}3|$flags_default
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0,1|(null)|${dev}0|$flags_default
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0,1|(null)|${dev}1|$flags_default
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0|(null)|${dev}4|$flags_default
testsuite/expect/test39.18:testproc_alias "b8" test_cfg "b8" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:1,gpu:1"
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "b10" test_cfg "b10" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:2,gpu:v100:2"
testsuite/expect/test39.18:Name=gpu            File=${dev}\[0-1\]
testsuite/expect/test39.18:Name=gpu Type=v100  File=${dev}\[2-3\]
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "b12" test_cfg "b12" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:2,gpu:v100:2"
testsuite/expect/test39.18:Name=gpu Type=tesla  File=${dev}\[0-1\]
testsuite/expect/test39.18:Name=gpu Type=v100
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "b14" test_cfg "b14" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:# # Test c2 - Test gres/gpu plus gres/mps with count
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:1,gpu:gtx560:1,mps:200"
testsuite/expect/test39.18:Name=gpu Type=gtx560 File=${dev}0 COREs=0,1
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 COREs=2,3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):gtx560|4|0,1|(null)|${dev}0|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2,3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c2" test_cfg "c2" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:1,gpu:gtx560:1,mps:210"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 COREs=2,3
testsuite/expect/test39.18:Name=gpu Type=gtx560 File=${dev}0 COREs=0,1
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):gtx560|4|0,1|(null)|${dev}0|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2,3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c4" test_cfg "c4" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:3"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[3-4\] Cores=2-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=0-1
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}4|$flags_file_type
testsuite/expect/test39.18:testproc_alias "c6" test_cfg "c6" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:6"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=2-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}2 Cores=2-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3 Cores=2-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}4 Cores=2-3
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}2|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}4|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|(null)|${dev}5|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|(null)|${dev}6|$flags_file_type
testsuite/expect/test39.18:testproc_alias "c8" test_cfg "c8" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:8"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}0 Cores=0-3 Links=-1,2,2,0,0,0,0,0
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=0-3 Links=2,-1,2,0,0,0,0,0
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}2 Cores=0-3 Links=2,2,-1,0,0,0,0,0
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3 Cores=0-3 Links=0,0,0,-1,0,1,0,0
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}4 Cores=0-3 Links=0,0,0,0,-1,1,0,0
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}5 Cores=0-3 Links=0,0,0,1,1,-1,0,0
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}6 Cores=0-3 Links=0,0,0,0,0,0,-1,2
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}7 Cores=0-3 Links=0,0,0,0,0,0,2,-1
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|-1,2,2,0,0,0,0,0|${dev}0|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|2,-1,2,0,0,0,0,0|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|2,2,-1,0,0,0,0,0|${dev}2|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|0,0,0,-1,0,1,0,0|${dev}3|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|0,0,0,0,-1,1,0,0|${dev}4|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|0,0,0,1,1,-1,0,0|${dev}5|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|0,0,0,0,0,0,-1,2|${dev}6|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|0,0,0,0,0,0,2,-1|${dev}7|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c10" test_cfg "c10" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "c12" test_cfg "c12" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:4"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[1-4\] Cores=2-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}2|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2-3|(null)|${dev}4|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c14" test_cfg "c14" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# # Test c16 - Test non-GPU GRESs with types
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:1,tmpdisk:disky:1,nic:nikki:1,mic:mickey:1"
testsuite/expect/test39.18:Name=gpu     Type=tesla File=${dev}1 Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c16" test_cfg "c16" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# # Test c17 - Test non-GPU GRESs without types
testsuite/expect/test39.18:set slurm_conf_gres "gpu:1,tmpdisk:1,nic:1,mic:1"
testsuite/expect/test39.18:Name=gpu     File=${dev}1 Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-3|(null)|${dev}1|$flags_default
testsuite/expect/test39.18:testproc_alias "c17" test_cfg "c17" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:2"
testsuite/expect/test39.18:NodeName=$nodename Name=gpu     Type=tesla File=${dev}1 Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c18" test_cfg "c18" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "c20" test_cfg "c20" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:# # Test c22 - Ensure no malloc error for large count with non-GPU GRES
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "c22" test_cfg "c22" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# # Test c23 - Ensure no errors for large count with non-GPU GRES *and* with
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "c23" test_cfg "c23" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a:2,gpu:b:2"
testsuite/expect/test39.18:Name=gpu Type=a File=${dev}1 Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=a File=${dev}2 Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=b File=${dev}5 Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=b File=${dev}6 Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a|4|0-3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a|4|0-3|(null)|${dev}2|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):b|4|0-3|(null)|${dev}5|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):b|4|0-3|(null)|${dev}6|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c24" test_cfg "c24" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla_a:4,gpu:tesla_b:4"
testsuite/expect/test39.18:Name=gpu Type=tesla_a File=${dev}\[1-2\] Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla_b File=${dev}\[3-4\] Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla_a File=${dev}5       Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla_b File=${dev}6       Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla_a File=${dev}7       Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla_b File=${dev}8       Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_a|4|0-3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_a|4|0-3|(null)|${dev}2|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_b|4|0-3|(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_b|4|0-3|(null)|${dev}4|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_a|4|0-3|(null)|${dev}5|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_b|4|0-3|(null)|${dev}6|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_a|4|0-3|(null)|${dev}7|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla_b|4|0-3|(null)|${dev}8|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c28" test_cfg "c28" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:6"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3       Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3       Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}4       Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}4       Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[3-4\] Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "c30" test_cfg "c30" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:10"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1       Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[1-2\] Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[1-3\] Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[1-4\] Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "c32" test_cfg "c32" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:10"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}4       Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[3-4\] Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[2-4\] Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[1-4\] Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "c34" test_cfg "c34" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:4"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=0
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=0-2
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "c36" test_cfg "c36" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:4"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=0
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}2 Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3 Cores=0-2
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}4 Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-1|(null)|${dev}2|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-2|(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|(null)|${dev}4|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c38" test_cfg "c38" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:4"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=0-3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}2 Cores=0-3 Links=\"\"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3 Cores=0-3 Links=null
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}4 Cores=0-3 Links=0
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|(null)|${dev}2|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|(null)|${dev}4|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c40" test_cfg "c40" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:4"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 Cores=0-3 Links=0-1
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}2 Cores=0-3 Links=0,-1
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3 Cores=0-3 Links=0-2
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}4 Cores=0-3 Links=0,-1,2
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|0,-1|${dev}2|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|0,-1,2|${dev}4|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0-3|(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c42" test_cfg "c42" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:4"
testsuite/expect/test39.18:Name=gpu File=${dev}1 Cores=0-3
testsuite/expect/test39.18:Name=gpu File=${dev}2 Cores=0-3
testsuite/expect/test39.18:Name=gpu File=${dev}3 Cores=0-3
testsuite/expect/test39.18:Name=gpu File=${dev}4 Cores=0-3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-3|(null)|${dev}1|$flags_default
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-3|(null)|${dev}2|$flags_default
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-3|(null)|${dev}3|$flags_default
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|0-3|(null)|${dev}4|$flags_default
testsuite/expect/test39.18:testproc_alias "c46" test_cfg "c46" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:5"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}2
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3 Cores=\"\"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}4 Cores=null
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}5 Cores=0
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|(null)|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|(null)|(null)|${dev}2|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4||(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|null|(null)|${dev}4|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0|(null)|${dev}5|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c48" test_cfg "c48" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:gtx560:1,gpu:tesla:1,mps:200"
testsuite/expect/test39.18:Name=gpu Type=gtx560 File=${dev}0 COREs=0,1
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 COREs=2,3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):gtx560|4|0,1|(null)|${dev}0|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2,3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c50" test_cfg "c50" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:tesla:3"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}\[0-1\] COREs=0,1
testsuite/expect/test39.18:# NOTE: nvidia2 device is out of service
testsuite/expect/test39.18:# Name=gpu Type=tesla File=${dev}\[2-3\] COREs=2,3
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3 COREs=2,3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0,1|(null)|${dev}0|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|0,1|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2,3|(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c52" test_cfg "c52" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:4"
testsuite/expect/test39.18:# NodeName=${nodename_base}\[0-15\] Name=gpu File=${dev}\[0-3\]
testsuite/expect/test39.18:NodeName=${nodename_base}\[0-2\] Name=gpu File=${dev}\[0-3\]
testsuite/expect/test39.18:NodeName=${nodename_base}3 Name=gpu File=${dev}\[0,2-3\]
testsuite/expect/test39.18:NodeName=${nodename_base}\[4-15\] Name=gpu File=${dev}\[0-3\]
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|(null)|(null)|${dev}0|$flags_default
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|(null)|(null)|${dev}1|$flags_default
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|(null)|(null)|${dev}2|$flags_default
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):(null)|4|(null)|(null)|${dev}3|$flags_default
testsuite/expect/test39.18:testproc_alias "c54" test_cfg "c54" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# # NOTE the device numbers being out of order, one GPU without a gres/mps and
testsuite/expect/test39.18:# #      a gres/mps with a device file not valid for any configured GPU
testsuite/expect/test39.18:set slurm_conf_gres "gpu:gtx560:1,gpu:tesla:2,mps:900"
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}1 COREs=2,3
testsuite/expect/test39.18:Name=gpu Type=gtx560 File=${dev}0 COREs=0,1
testsuite/expect/test39.18:Name=gpu Type=tesla File=${dev}3 COREs=2,3
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):gtx560|4|0,1|(null)|${dev}0|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2,3|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):tesla|4|2,3|(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:testproc_alias "c56" test_cfg "c56" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:4"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}0 Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}1 Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}2 Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}3 Cores=0-1
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}0|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}1|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}2|$flags_default_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}3|$flags_default_type
testsuite/expect/test39.18:testproc_alias "d1" test_cfg "d1" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:4"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}0 Cores=0-1 Flags=no_gpu_env
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}1 Cores=0-1 Flags=no_gpu_env
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}2 Cores=0-1 Flags=no_gpu_env
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}3 Cores=0-1 Flags=no_gpu_env
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:testproc_alias "d2" test_cfg "d2" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:4"
testsuite/expect/test39.18:NodeName=$nodename      Name=gpu Type=a100 File=${dev}0 Cores=0-1 Flags=amd_gpu_env
testsuite/expect/test39.18:NodeName=$nodename      Name=gpu Type=a100 File=${dev}1 Cores=0-1 Flags=amd_gpu_env
testsuite/expect/test39.18:NodeName=$nodename_diff Name=gpu Type=a100 File=${dev}2 Cores=0-1 Flags=nvidia_gpu_env
testsuite/expect/test39.18:NodeName=$nodename_diff Name=gpu Type=a100 File=${dev}3 Cores=0-1 Flags=nvidia_gpu_env
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}0|$flags_file_type,ENV_RSMI
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}1|$flags_file_type,ENV_RSMI
testsuite/expect/test39.18:testproc_alias "d3" test_cfg "d3" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:# # Test d4 - AutoDetect match with conf does NOT overwrite no_gpu_env
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:4"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}0       Cores=0-1 Flags=no_gpu_env
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}\[1-3\] Cores=0-1 Flags=no_gpu_env
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:a100|4|0-1|(null)|${dev}0|(null)|nvidia_gpu_env
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}0|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}1|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}2|$flags_file_type
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}3|$flags_file_type
testsuite/expect/test39.18:testproc_alias "d4" test_cfg "d4" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:4"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}0       Cores=0-1 Flags=nvidia_gpu_env
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}\[1-3\] Cores=0-1 Flags=nvidia_gpu_env
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:a100|4|0-1|(null)|${dev}0|(null)|nvidia_gpu_env
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}0|$flags_file_type,ENV_NVML
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}1|$flags_file_type,ENV_NVML
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}2|$flags_file_type,ENV_NVML
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}3|$flags_file_type,ENV_NVML
testsuite/expect/test39.18:testproc_alias "d5" test_cfg "d5" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:4"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}\[1-3\] Cores=0-1 Flags=nvidia_gpu_env
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:a100|4|0-1|(null)|${dev}0|(null)|nvidia_gpu_env
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}0|$flags_file_type,ENV_NVML
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}1|$flags_file_type,ENV_NVML
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}2|$flags_file_type,ENV_NVML
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}3|$flags_file_type,ENV_NVML
testsuite/expect/test39.18:testproc_alias "d6" test_cfg "d6" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:#              node, so in reality, both CUDA_* and ROCR_* would be set on all.
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:4"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}\[1-3\] Cores=0-1 Flags=amd_gpu_env
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:a100|4|0-1|(null)|${dev}0|(null)|nvidia_gpu_env
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}0|$flags_file_type,ENV_NVML
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}1|$flags_file_type,ENV_RSMI
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}2|$flags_file_type,ENV_RSMI
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}3|$flags_file_type,ENV_RSMI
testsuite/expect/test39.18:testproc_alias "d7" test_cfg "d7" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:4"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}0 Cores=0-1 Flags=amd_gpu_env,nvidia_gpu_env
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}1 Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}2 Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}3 Cores=0-1
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}0|$flags_file_type,ENV_NVML,ENV_RSMI
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}1|$flags_file_type,ENV_NVML,ENV_RSMI
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}2|$flags_file_type,ENV_NVML,ENV_RSMI
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}3|$flags_file_type,ENV_NVML,ENV_RSMI
testsuite/expect/test39.18:testproc_alias "d8" test_cfg "d8" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:4"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}\[0-1\] Cores=0-1 Flags=no_gpu_env
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}\[2-3\] Cores=0-1 Flags=nvidia_gpu_env
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:testproc_alias "d9" test_cfg "d9" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:4"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}0       Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}\[1-3\] Cores=0-1 Flags=nvidia_gpu_env
testsuite/expect/test39.18:set fake_gpus_conf "
testsuite/expect/test39.18:a100|4|0-1|(null)|${dev}0|(null)|nvidia_gpu_env
testsuite/expect/test39.18:testproc_alias "d10" test_cfg "d10" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output $err_msgs $expected_errs
testsuite/expect/test39.18:set slurm_conf_gres "gpu:a100:1,gpu:b100:1,gpu:c100:1,gpu:d100:1"
testsuite/expect/test39.18:Name=gpu Type=a100 File=${dev}0 Cores=0-1 Flags=amd_gpu_env
testsuite/expect/test39.18:Name=gpu Type=b100 File=${dev}1 Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=c100 File=${dev}2 Cores=0-1
testsuite/expect/test39.18:Name=gpu Type=d100 File=${dev}3 Cores=0-1
testsuite/expect/test39.18:set fake_gpus_conf ""
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):a100|4|0-1|(null)|${dev}0|$flags_file_type,ENV_RSMI
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):b100|4|0-1|(null)|${dev}1|$flags_file_type,ENV_RSMI
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):c100|4|0-1|(null)|${dev}2|$flags_file_type,ENV_RSMI
testsuite/expect/test39.18:GRES_PARSABLE\[gpu\](1):d100|4|0-1|(null)|${dev}3|$flags_file_type,ENV_RSMI
testsuite/expect/test39.18:testproc_alias "d11" test_cfg "d11" $slurm_conf_gres $gres_conf $fake_gpus_conf $expected_output
testsuite/expect/test39.10:#          Test --mem-per-gpu option
testsuite/expect/test39.10:proc run_gpu_per_job { mem_per_gpu } {
testsuite/expect/test39.10:	spawn $srun --gpus=1 --mem-per-gpu=$mem_per_gpu -J $test_name -t1 $file_in
testsuite/expect/test39.10:	if {$mem_size != $mem_per_gpu} {
testsuite/expect/test39.10:		fail "srun --mem-per-gpu failure ($mem_size != $mem_per_gpu)"
testsuite/expect/test39.10:proc run_gpu_per_node { mem_per_gpu } {
testsuite/expect/test39.10:	spawn $srun --gpus-per-node=1 -N1 --mem-per-gpu=$mem_per_gpu -J $test_name -t1 $file_in
testsuite/expect/test39.10:	if {$mem_size != $mem_per_gpu} {
testsuite/expect/test39.10:		fail "srun --mem-per-gpu failure ($mem_size != $mem_per_gpu)"
testsuite/expect/test39.10:proc run_gpu_per_task { mem_per_gpu gpu_cnt } {
testsuite/expect/test39.10:	spawn $srun --gpus-per-task=$gpu_cnt -n1 --mem-per-gpu=$mem_per_gpu -J $test_name -t1 $file_in
testsuite/expect/test39.10:	set mem_target [expr $mem_per_gpu * $gpu_cnt]
testsuite/expect/test39.10:		fail "srun --mem-per-gpu failure ($mem_size != $mem_target)"
testsuite/expect/test39.10:proc run_gpu_check_mem { srun_opts mem_target node_target } {
testsuite/expect/test39.10:		fail "srun --mem-per-gpu failure, bad node count ($node_count < $node_target)"
testsuite/expect/test39.10:set gpu_cnt   [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.10:if {$gpu_cnt < 2} {
testsuite/expect/test39.10:	skip "This test requires 2 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.10:set nodes [get_nodes_by_request "--gres=gpu:$gpu_cnt -t1 -N $nb_nodes"]
testsuite/expect/test39.10:	skip "This test need to be able to submit jobs with at least --gres=gpu:$gpu_cnt to $nb_nodes nodes"
testsuite/expect/test39.10:# Get the node with the maximum number of GPUs
testsuite/expect/test39.10:dict for {node gpus} [get_gres_count "gpu" [join $nodes ,]] {
testsuite/expect/test39.10:	if {$gpus >= $gpu_cnt} {
testsuite/expect/test39.10:		set gpu_cnt   $gpus
testsuite/expect/test39.10:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.10:make_bash_script $file_in "echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.10:# Run test job with global GPU count
testsuite/expect/test39.10:# Increase mem_per_gpu value 10x on each iteration
testsuite/expect/test39.10:	run_gpu_per_job $inx
testsuite/expect/test39.10:# Run test job with gpus-per-node count
testsuite/expect/test39.10:# Increase mem_per_gpu value 10x on each iteration
testsuite/expect/test39.10:	run_gpu_per_node $inx
testsuite/expect/test39.10:# Run test job with gpus-per-task count and one GPU
testsuite/expect/test39.10:# Increase mem_per_gpu value 10x on each iteration
testsuite/expect/test39.10:	run_gpu_per_task $inx 1
testsuite/expect/test39.10:# Run test job with gpus-per-task count and two GPUs
testsuite/expect/test39.10:# Increase mem_per_gpu value 10x on each iteration
testsuite/expect/test39.10:if {$gpu_cnt > 1} {
testsuite/expect/test39.10:		run_gpu_per_task $inx 2
testsuite/expect/test39.10:# Test heterogeneous GPU allocation (gpu_cnt GPUs on one node, 1 GPU on another node)
testsuite/expect/test39.10:if {$gpu_cnt > 1 && $nb_nodes > 1} {
testsuite/expect/test39.10:	set gpu_target [expr $gpu_cnt + 1]
testsuite/expect/test39.10:	set mem_target [expr $mem_spec * $gpu_target]
testsuite/expect/test39.10:	run_gpu_check_mem "--gpus=$gpu_target --mem-per-gpu=$mem_spec" $mem_target $node_target
testsuite/expect/test39.10:# Run test with --gpus=2 and mem_per_gpu value that pushed job to 2 nodes
testsuite/expect/test39.10:if {$gpu_cnt > 1 && $nb_nodes > 1} {
testsuite/expect/test39.10:	set mem_spec [expr $node_memory / $gpu_cnt + 1]
testsuite/expect/test39.10:	set mem_target [expr $mem_spec * $gpu_cnt]
testsuite/expect/test39.10:	run_gpu_check_mem "--gpus=$gpu_cnt --mem-per-gpu=$mem_spec" $mem_target $node_target
testsuite/expect/test39.10:log_info "Testing --mem-per-gpu with --exclusive and --gres=gpu:1"
testsuite/expect/test39.10:for {set inx 12} {$inx <= [expr $node_memory / $gpu_cnt]} {set inx [expr $inx * 10]} {
testsuite/expect/test39.10:	run_gpu_check_mem "--gres=gpu:1 --mem-per-gpu=$inx --exclusive -w $node_name" [expr $gpu_cnt * $inx] 1
testsuite/expect/test39.10:log_info "Testing --mem-per-gpu with --exclusie and --gpus=1"
testsuite/expect/test39.10:for {set inx 12} {$inx <= [expr $node_memory / $gpu_cnt]} {set inx [expr $inx * 10]} {
testsuite/expect/test39.10:	run_gpu_check_mem "--gpus=1 --mem-per-gpu=$inx --exclusive -w $node_name" [expr $gpu_cnt * $inx] 1
testsuite/expect/test39.10:log_info "Testing --mem-per-gpu with --exclusie and --gpus-per-task=1"
testsuite/expect/test39.10:for {set inx 12} {$inx <= [expr $node_memory / $gpu_cnt]} {set inx [expr $inx * 10]} {
testsuite/expect/test39.10:	run_gpu_check_mem "--gpus-per-task=1 --ntasks-per-node=1 --mem-per-gpu=$inx --exclusive -w $node_name" [expr $gpu_cnt * $inx] 1
testsuite/expect/test39.10:log_info "Testing --mem-per-gpu with --exclusie and --gpus-per-socket=1"
testsuite/expect/test39.10:for {set inx 12} {$inx <= [expr $node_memory / $gpu_cnt]} {set inx [expr $inx * 10]} {
testsuite/expect/test39.10:	run_gpu_check_mem "--gpus-per-socket=1 --sockets-per-node=1 --mem-per-gpu=$inx --exclusive -w $node_name" [expr $gpu_cnt * $inx] 1
testsuite/expect/test40.8:#          Simple CUDA MPS test
testsuite/expect/test40.8:env | grep CUDA_VISIBLE_DEVICES
testsuite/expect/test40.8:env | grep CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
testsuite/expect/test40.8:unset CUDA_VISIBLE_DEVICES
testsuite/expect/test40.8:# Spawn a batch job to build and run CUDA job
testsuite/expect/test40.8:		skip "This means the gpu selected doesn't support this test"
testsuite/expect/test40.8:	-re "CUDA_VISIBLE_DEVICES" {
testsuite/expect/test40.8:	-re "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" {
testsuite/expect/test40.8:	skip "Could not find program nvcc (CUDA compiler)"
testsuite/expect/test40.8:	fail "CUDA output not as expected ($matches != 6)"
testsuite/expect/test40.8:		fail "CUDA MPS jobs appear to have not run in parallel. Run time difference was $percent_time_diff percent"
testsuite/expect/test40.8:		log_debug "CUDA MPS jobs do appear to have not run in parallel"
testsuite/expect/test39.16:#          Test --gpus-per-tres with --exclusive option
testsuite/expect/test39.16:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.16:if {$gpu_cnt < 2} {
testsuite/expect/test39.16:	skip "This test requires 2 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.16:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.16:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.16:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.16:if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.16:	log_warn "This test needs CPUs >= GPUs in the node. On current configuration it will only test the expected error."
testsuite/expect/test39.16:echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.16:spawn $srun --cpus-per-gpu=1 --gpus-per-task=1 --nodes=$nb_nodes --exclusive --ntasks=$nb_tasks -t1 -J "$test_name" -l $file_in
testsuite/expect/test39.16:	-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.16:		incr match [cuda_count $expect_out(1,string)]
testsuite/expect/test39.16:if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.16:		log_info "This error is expected since GPUs ($gpu_cnt) > CPUs ($cpus_per_node)"
testsuite/expect/test39.16:		fail "srun --gpus-per-task with --exclusive failure ($match != $nb_tasks)"
testsuite/expect/test39.16:if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.16:	skip "System has more GPUs than CPUs, test is only checking an expected error"
testsuite/expect/test39.20:#           Test GPU resource limits with various allocation options
testsuite/expect/test39.20:#           AccountingStorageTRES=gres/gpu
testsuite/expect/test39.20:proc setup { gpu_limit } {
testsuite/expect/test39.20:	set acct_req(maxtres) "gres/gpu=$gpu_limit"
testsuite/expect/test39.20:set store_mps [string first "gres/gpu:" $store_tres]
testsuite/expect/test39.20:	skip "This test requires homogeneous GPU accounting (NO Type)"
testsuite/expect/test39.20:set store_gpu [string first "gres/gpu" $store_tres]
testsuite/expect/test39.20:if {$store_gpu == -1} {
testsuite/expect/test39.20:	skip "This test requires accounting for GPUs"
testsuite/expect/test39.20:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.20:if {$gpu_cnt < 2} {
testsuite/expect/test39.20:	skip "This test requires 2 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.20:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.20:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.20:log_debug "GPUs per node is $gpu_cnt"
testsuite/expect/test39.20:set gpu_limit [expr $gpu_cnt * $nb_nodes]
testsuite/expect/test39.20:if {$gpu_limit > 8} {
testsuite/expect/test39.20:	set gpu_limit 8
testsuite/expect/test39.20:	incr gpu_limit -1
testsuite/expect/test39.20:setup $gpu_limit
testsuite/expect/test39.20:	$scontrol -dd show job \${SLURM_JOBID} | grep gpu
testsuite/expect/test39.20:# Test --gpus option by job (first job over limit, second job under limit)
testsuite/expect/test39.20:log_info "TEST 1: --gpus option by job (first job over limit, second job under limit)"
testsuite/expect/test39.20:set gpu_fail_cnt [expr $gpu_limit + 1]
testsuite/expect/test39.20:set job_id1 [submit_job -fail "--account=$acct --gres=craynetwork:0 --gpus=$gpu_fail_cnt -t1 -o $file_out1 -J $test_name $file_in"]
testsuite/expect/test39.20:set job_id2 [submit_job -fail "--account=$acct --gres=craynetwork:0 --gpus=$gpu_limit -t1 -o $file_out2 -J $test_name $file_in"]
testsuite/expect/test39.20:# Test --gpus-per-node option by job (first job over limit, second job under limit)
testsuite/expect/test39.20:log_info "TEST 2: --gpus-per-node option by job (first job over limit, second job under limit)"
testsuite/expect/test39.20:set gpu_good_cnt [expr $gpu_limit / $nb_nodes]
testsuite/expect/test39.20:	set gpu_fail_cnt [expr $gpu_limit + 1]
testsuite/expect/test39.20:	set gpu_fail_cnt [expr $gpu_good_cnt + 1]
testsuite/expect/test39.20:set job_id1 [submit_job -fail "--account=$acct --gres=craynetwork:0 --gpus-per-node=$gpu_fail_cnt -N$nb_nodes -t1 -o $file_out1 -J $test_name $file_in"]
testsuite/expect/test39.20:set job_id2 [submit_job -fail "--account=$acct --gres=craynetwork:0 --gpus-per-node=$gpu_good_cnt -N$nb_nodes -t1 -o $file_out2 -J $test_name $file_in"]
testsuite/expect/test39.20:# Test --gpus-per-task option by job (first job over limit, second job under limit)
testsuite/expect/test39.20:log_info "TEST 3: --gpus-per-task option by job (first job over limit, second job under limit)"
testsuite/expect/test39.20:set gpu_good_cnt $gpu_limit
testsuite/expect/test39.20:set gpu_fail_cnt [expr $gpu_limit + 1]
testsuite/expect/test39.20:set job_id1 [submit_job -fail "--account=$acct --gres=craynetwork:0 --gpus-per-task=1 -n$gpu_fail_cnt $extra_opt -t1 -o $file_out1 -J $test_name $file_in"]
testsuite/expect/test39.20:set job_id2 [submit_job -fail "--account=$acct --gres=craynetwork:0 --gpus-per-task=1 -n$gpu_good_cnt $extra_opt -t1 -o $file_out2 -J $test_name $file_in"]
testsuite/expect/test39.5:#          Test some valid combinations of srun --gpu options
testsuite/expect/test39.5:proc _test_gpus_cnt {cmd expected_gpus_times_tasks expected_job_gpus} {
testsuite/expect/test39.5:	set job_gpus 0
testsuite/expect/test39.5:                if [dict exists $gres_dict "gpu"] {
testsuite/expect/test39.5:                        set job_gpus [dict get $gres_dict "gpu"]
testsuite/expect/test39.5:        foreach {- cuda_string} [regexp -all -inline {CUDA_VISIBLE_DEVICES:([0-9_,]+)} $output] {
testsuite/expect/test39.5:                incr match [cuda_count $cuda_string]
testsuite/expect/test39.5:	subtest {$match == $expected_gpus_times_tasks} "$cmd tasks*gpus result=$match, expected=$expected_gpus_times_tasks"
testsuite/expect/test39.5:	subtest {$job_gpus == $expected_job_gpus} "$cmd JOB_GRES result=$job_gpus, expected=$expected_job_gpus"
testsuite/expect/test39.5:proc test_gpus_cnt {srun_opts expected_gpus_times_tasks expected_job_gpus} {
testsuite/expect/test39.5:	if {$expected_gpus_times_tasks == 0 && $expected_job_gpus == 0} {
testsuite/expect/test39.5:	_test_gpus_cnt "$srun $srun_opts -t1 -J $test_name -l $file_in1" $expected_gpus_times_tasks $expected_job_gpus
testsuite/expect/test39.5:			_test_gpus_cnt "$bin_cat ./slurm-$jobid.out" $expected_gpus_times_tasks $expected_job_gpus
testsuite/expect/test39.5:			subskip -count 2 "Skipping #GPUs * #NTASKS and #GPUS subtests since job submit failed"
testsuite/expect/test39.5:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test39.5:if {$gpu_cnt < 1} {
testsuite/expect/test39.5:	skip "This test requires 1 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test39.5:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test39.5:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test39.5:log_debug "GPU count is $gpu_cnt"
testsuite/expect/test39.5:if {$gpu_cnt > $cpus_per_node} {
testsuite/expect/test39.5:	set gpu_cnt $cpus_per_node
testsuite/expect/test39.5:set tot_gpus $gpu_cnt
testsuite/expect/test39.5:	incr tot_gpus $gpu_cnt
testsuite/expect/test39.5:set gpus_per_node $gpu_cnt
testsuite/expect/test39.5:if {$gpus_per_node > 1 && $sockets_per_node > 1} {
testsuite/expect/test39.5:	set gpus_per_socket [expr $gpus_per_node / $sockets_per_node]
testsuite/expect/test39.5:	set gpus_per_socket $gpus_per_node
testsuite/expect/test39.5:make_bash_script $file_in1 "echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.5:make_bash_script $file_in2 "echo HOST:\$SLURMD_NODENAME CUDA_VISIBLE_DEVICES:\$CUDA_VISIBLE_DEVICES
testsuite/expect/test39.5:# Test --gpus options using a subset of GPUs actually available on the node
testsuite/expect/test39.5:log_info "TEST: --gpus option"
testsuite/expect/test39.5:if {$tot_gpus > 1} {
testsuite/expect/test39.5:	set use_gpus_per_job [expr $tot_gpus - 1]
testsuite/expect/test39.5:	set use_gpus_per_job $tot_gpus
testsuite/expect/test39.5:# Every node requires at least 1 GPU
testsuite/expect/test39.5:if {$use_gpus_per_job < $nb_nodes} {
testsuite/expect/test39.5:	set nb_nodes $use_gpus_per_job
testsuite/expect/test39.5:test_gpus_cnt "--cpus-per-gpu=1 --gpus=$use_gpus_per_job --nodes=$nb_nodes" $use_gpus_per_job $use_gpus_per_job
testsuite/expect/test39.5:if {$use_gpus_per_job > 2} {
testsuite/expect/test39.5:	log_info "TEST: --gpus option, part 2"
testsuite/expect/test39.5:	test_gpus_cnt "--cpus-per-gpu=1 --gpus=$use_gpus_per_job --nodes=$nb_nodes" $use_gpus_per_job $use_gpus_per_job
testsuite/expect/test39.5:# Test --gpus-per-node options using a subset of GPUs actually available on the node
testsuite/expect/test39.5:log_info "TEST: --gpus-per-node option"
testsuite/expect/test39.5:if {$gpus_per_node > 1} {
testsuite/expect/test39.5:	set use_gpus_per_node [expr $gpus_per_node - 1]
testsuite/expect/test39.5:	set use_gpus_per_node $gpus_per_node
testsuite/expect/test39.5:test_gpus_cnt "--cpus-per-gpu=1 --gpus-per-node=$use_gpus_per_node --nodes=$nb_nodes" [expr $use_gpus_per_node * $nb_nodes] [expr $use_gpus_per_node * $nb_nodes]
testsuite/expect/test39.5:# Test --gpus-per-socket options using a subset of GPUs actually available on the node
testsuite/expect/test39.5:log_info "TEST: --gpus-per-socket option"
testsuite/expect/test39.5:set sockets_with_gpus [get_gpu_socket_count $gpu_cnt $sockets_per_node]
testsuite/expect/test39.5:set node_list [get_nodes_by_request "--gpus-per-socket=1 --sockets-per-node=$sockets_with_gpus --nodes=$nb_nodes"]
testsuite/expect/test39.5:	lappend skipped "This test need to be able to submit jobs with at least --gpus-per-socket=1 --sockets-per-node=$sockets_with_gpus --nodes=$node_list"
testsuite/expect/test39.5:	spawn $srun --gpus-per-socket=1 --sockets-per-node=$sockets_with_gpus --ntasks-per-socket=1 --exclusive --nodelist=[join $node_list ","] -t1 -J "$test_name" -l $file_in1
testsuite/expect/test39.5:		-re "CUDA_VISIBLE_DEVICES:($number_commas)" {
testsuite/expect/test39.5:			incr match [cuda_count $expect_out(1,string)]
testsuite/expect/test39.5:	set gpus_per_node [get_gres_count "gpu" [join $node_list ","]]
testsuite/expect/test39.5:	set expected_gpus 0
testsuite/expect/test39.5:	dict for {node gpus} $gpus_per_node {
testsuite/expect/test39.5:		incr expected_gpus $gpus
testsuite/expect/test39.5:	if {$match < $expected_gpus} {
testsuite/expect/test39.5:		fail "srun --gpus-per-socket failure ($match < $expected_gpus)"
testsuite/expect/test39.5:# Test --gpus-per-task options using a subset of GPUs actually available on the node
testsuite/expect/test39.5:log_info "TEST: --gpus-per-task option"
testsuite/expect/test39.5:if {$gpu_cnt > 1} {
testsuite/expect/test39.5:	set use_gpus_per_node [expr $gpu_cnt - 1]
testsuite/expect/test39.5:	set use_gpus_per_node $gpu_cnt
testsuite/expect/test39.5:test_gpus_cnt "--cpus-per-gpu=1 --gpus-per-task=1 -N1 --ntasks=$use_gpus_per_node $extra_opt" $use_gpus_per_node $use_gpus_per_node
testsuite/expect/test39.5:# Test --gpus-per-task option without task count
testsuite/expect/test39.5:log_info "TEST: --gpus-per-task option, part 2 (implicit task count)"
testsuite/expect/test39.5:if {$gpu_cnt > 1} {
testsuite/expect/test39.5:	set use_gpus_per_node [expr ($gpu_cnt - 1) / $use_tasks_per_node]
testsuite/expect/test39.5:	set use_gpus_per_node [expr $gpu_cnt / $use_tasks_per_node]
testsuite/expect/test39.5:if {$use_gpus_per_node == 0} {
testsuite/expect/test39.5:	set use_gpus_per_node $gpu_cnt
testsuite/expect/test39.5:test_gpus_cnt "--gpus-per-task=1 --ntasks-per-node=$use_tasks_per_node -N $nb_nodes" [expr $use_tasks_per_node * $nb_nodes] [expr $use_tasks_per_node * $nb_nodes]
testsuite/expect/test39.5:# Test --gpus-per-task option without task count
testsuite/expect/test39.5:log_info "TEST: --gpus-per-task option, part 3 (implicit task count)"
testsuite/expect/test39.5:if {$gpu_cnt > 1} {
testsuite/expect/test39.5:	set use_gpus_per_node [expr $gpu_cnt - 1]
testsuite/expect/test39.5:	set use_gpus_per_node $gpu_cnt
testsuite/expect/test39.5:test_gpus_cnt "--gpus-per-task=$use_gpus_per_node -N $nb_nodes" [expr $nb_nodes * $use_gpus_per_node] [expr $nb_nodes * $use_gpus_per_node]
testsuite/expect/test39.5:if {$sockets_per_node <= $gpu_cnt} {
testsuite/expect/test39.5:	set target_gpus $sockets_per_node
testsuite/expect/test39.5:	set target_gpus $gpu_cnt
testsuite/expect/test39.5:# Test --cpus-per-task with different gpu options (Bug 9937)
testsuite/expect/test39.5:log_info "TEST: --cpus-per-task option with different gpu options(Bug 9937)"
testsuite/expect/test39.5:test_gpus_cnt "--gres=gpu:1 --ntasks-per-core=1 -c $cpus_per_node" 1 1
testsuite/expect/test39.5:test_gpus_cnt "--gpus=1 --ntasks-per-core=1 -c $cpus_per_node" 1 1
testsuite/expect/test39.5:test_gpus_cnt "-n1 --gpus-per-task=1 --ntasks-per-core=1 -c $cpus_per_node" 1 1
testsuite/expect/test39.5:test_gpus_cnt "--sockets-per-node=1 --gpus-per-socket=1 --ntasks-per-core=1 -c $cpus_per_node" $target_gpus $target_gpus
testsuite/expect/test39.5:# Because we're doing one sockets worth of cpus and if there are two sockets we'll get two gpus.
testsuite/expect/test39.5:test_gpus_cnt "--gpus-per-socket=1 --sockets-per-node=1 --threads-per-core=1 -c [expr $cpus_per_node / $threads_per_core]" $sockets_per_node $target_gpus
testsuite/expect/test39.5:	test_gpus_cnt "--gpus-per-socket=1 --sockets-per-node=1 --threads-per-core=2 -c [expr $cpus_per_node / 2]" $sockets_per_node $sockets_per_node
testsuite/expect/test39.5:test_gpus_cnt "--sockets-per-node=1 --gpus-per-socket=1 --ntasks-per-core=1 -c [expr $cpus_per_node * 2]" 0 0
testsuite/expect/test39.5:test_gpus_cnt "--gres=gpu:1 -c [expr $cpus_per_node * 2]" 0 0
testsuite/expect/test39.5:test_gpus_cnt "--gpus=1 -c [expr $cpus_per_node * 2]" 0 0
testsuite/expect/test39.5:	test_gpus_cnt "--gpus=1 --threads-per-core=1 -c $cpus_per_node" 0 0
testsuite/expect/test39.5:	test_gpus_cnt "--gpus=1 --threads-per-core=1 -c [expr $cpus_per_node]" 1 1
testsuite/expect/test39.5:test_gpus_cnt "--gres=gpu:1 --ntasks-per-node=1 -c $cpus_per_node" 1 1
testsuite/expect/test39.5:test_gpus_cnt "--gres=gpu:1 --ntasks-per-node=1 -c $cpus_per_socket" 1 1
testsuite/expect/test39.5:test_gpus_cnt "--gres=gpu:1 --ntasks-per-node=1 -c 1" 1 1
testsuite/expect/test39.5:test_gpus_cnt "--gres=gpu:1 --ntasks-per-socket=1 -c $cpus_per_node" 1 1
testsuite/expect/test39.5:test_gpus_cnt "--gres=gpu:1 --ntasks-per-socket=1 -c $cpus_per_node" 1 1
testsuite/expect/test39.5:test_gpus_cnt "--gpus=1 --ntasks-per-node=1 -c $cpus_per_node" 1 1
testsuite/expect/test39.5:test_gpus_cnt "--gpus=1 --ntasks-per-socket=1 -c $cpus_per_node" 1 1
testsuite/expect/test39.5:test_gpus_cnt "-n1 --gpus-per-task=1 --ntasks-per-node=1 -c $cpus_per_node" 1 1
testsuite/expect/test39.5:test_gpus_cnt "-n1 --gpus-per-task=1 --ntasks-per-socket=1 -c $cpus_per_node" 1 1
testsuite/expect/test39.5:test_gpus_cnt "--sockets-per-node=$sockets_per_node --gpus-per-socket=1 --ntasks-per-core=1 -c 1 -n$sockets_per_node" [expr $sockets_per_node * $sockets_per_node] $sockets_per_node
testsuite/expect/test39.5:test_gpus_cnt "--sockets-per-node=1 --gpus-per-socket=1 --ntasks-per-core=1 -c $cpus_per_socket" 1 1
testsuite/expect/test39.5:set target_gpus [expr min($sockets_per_node, $gpu_cnt)]
testsuite/expect/test39.5:test_gpus_cnt "--sockets-per-node=$sockets_per_node --gpus-per-socket=1 --ntasks-per-core=1 -c $cpus_per_node" $target_gpus $target_gpus
testsuite/expect/test20.14:#          Test PBS/qsub -l gpu options
testsuite/expect/test20.14:set gpu_req_cnt 1
testsuite/expect/test20.14:set gpu_cnt [get_highest_gres_count $nb_nodes "gpu"]
testsuite/expect/test20.14:if {$gpu_cnt < 1} {
testsuite/expect/test20.14:	skip "This test requires 1 or more GPUs on $nb_nodes nodes of the default partition"
testsuite/expect/test20.14:set node_name [get_nodes_by_request "--gres=gpu:1 -n1 -t1"]
testsuite/expect/test20.14:	skip "This test need to be able to submit jobs with at least --gres=gpu:1"
testsuite/expect/test20.14:	global number job_id time_str ppn_cnt gpu_req_cnt scontrol
testsuite/expect/test20.14:		-re "TresPerNode=.*gpu:$gpu_req_cnt" {
testsuite/expect/test20.14:	"-l nodes=$node_cnt:ppn=$ppn_cnt,naccelerators=$gpu_req_cnt,walltime=$time_str" \
testsuite/expect/test20.14:	"-l nodes=$node_cnt:ppn=$ppn_cnt:gpus=$gpu_req_cnt,walltime=$time_str" \
testsuite/expect/test7.17:spawn ./$test_prog "gres/gpu:2" "$cfgdir" "/test7.17.1" 8 4 2
testsuite/expect/test7.17:spawn ./$test_prog "gres/gpu:2" "$cfgdir" "/test7.17.2" 16 8 2
testsuite/expect/test7.17:spawn ./$test_prog "gres/gpu:2" "$cfgdir" "/test7.17.3" 16 8 2
testsuite/expect/test7.17:spawn ./$test_prog "gres/gpu:2" "$cfgdir" "/test7.17.4" 16 8 2
testsuite/expect/test7.17:spawn ./$test_prog "gres/gpu:2" "$cfgdir" "/test7.17.6" 0 0 0
testsuite/expect/test7.17:spawn ./$test_prog "gres/gpu:2" "$cfgdir" "/test7.17.7" 8 4 2
testsuite/expect/test7.17:subtest {$fail_match == 1} "A job run with invalid job allocation (gpu:2) should fail" "$fail_match"
testsuite/README:test1.62   Test of gres/gpu plugin (if configured).
testsuite/README:test1.119  Test of srun --ntasks-per-gpu option.
testsuite/README:test20.14  Test PBS/qsub -l gpu options
testsuite/README:test38.18  Validate heterogeneous gpu job options.
testsuite/README:test39.#   Test of job select/cons_tres and --gpu options.
testsuite/README:test39.1   Test full set of sbatch --gpu options and scontrol show job
testsuite/README:test39.2   Test full set of salloc --gpu options and scontrol show job
testsuite/README:test39.3   Test full set of srun --gpu options and scontrol show step
testsuite/README:test39.4   Test some invalid combinations of --gpu options
testsuite/README:test39.5   Test some combinations of srun and sbatch --gpu options
testsuite/README:test39.6   Ensure job requesting GPUs on multiple sockets gets CPUs on them
testsuite/README:test39.7   Test --cpus-per-gpu option
testsuite/README:test39.8   Test --gpu-bind options
testsuite/README:test39.9   Test --gpu-freq options
testsuite/README:test39.10  Test --mem-per-gpu option
testsuite/README:test39.12  Test some valid combinations of srun --gpu and non-GPU GRES options
testsuite/README:test39.14  Increase size of job with allocated GPUs
testsuite/README:test39.15  Test --gpus-per-tres with --overcommit option
testsuite/README:test39.16  Test --gpus-per-tres with --exclusive option
testsuite/README:test39.18  Test gres.conf and system GPU normalization and merging logic
testsuite/README:test39.19  Test accounting for GPU resources with various allocation options
testsuite/README:test39.20  Test GPU resource limits with various allocation options
testsuite/README:test39.22  Test heterogeneous job GPU allocations.
testsuite/README:test39.23  Test --gpus-per-task with implicit task count
testsuite/README:test40.8   Simple CUDA MPS test
testsuite/README:test_105_1   /commands/sbatch/test_gpu_options.py
testsuite/README:test_144_2   Test enforce-binding with GPUs that share cores
testsuite/README:test_144_3   Test --exact with GPUs and multiple steps
RELEASE_NOTES: -- Add autodetected gpus to the output of slurmd -C
doc/html/select_design.shtml:resources, such as GPUs.
doc/html/select_design.shtml:memory allocations. If you need to track other resources, such as GPUs,
doc/html/configurator.html.in:<b>cons_tres</b>: Allocate individual processors, memory, GPUs, and other
doc/html/slurm.shtml:plus an <a href="http://www.nvidia.com">NVIDIA</a> Tesla K20X GPUs
doc/html/slurm.shtml:128 <a href="http://www.nvidia.com">NVIDIA</a> GPUs
doc/html/slurm.shtml:with 52,168 Intel Xeon processing cores and 8,840 NVIDIA GPUs.</li>
doc/html/slurm.shtml:a combined CPU-GPU Linux cluster at
doc/html/slurm.shtml:performance) plus 778 ATI Radeon 5870 GPUs (2.1 Petaflops peak
doc/html/heterogeneous_jobs.shtml:<p>An example scenario would be if you have a task that needs to use 1 GPU
doc/html/heterogeneous_jobs.shtml:per processor while another task needs all the available GPUs on a node with
doc/html/heterogeneous_jobs.shtml:$ salloc -N2 --exclusive --gpus=10
doc/html/heterogeneous_jobs.shtml:$ srun -N1 -n4 --gpus=4 printenv SLURMD_NODENAME : -N1 -n1 --gpus=6 printenv SLURMD_NODENAME
doc/html/overview.shtml:  control generic resources, including Graphical Processing Units (GPUs).
doc/html/publications.shtml:<li>Technical: <a href="SLUG19/GPU_Scheduling_and_Cons_Tres.pdf">GPU Scheduling and the cons_tres plugin</a>,
doc/html/publications.shtml:<li>Technical: <a href="SLUG19/NVIDIA_Containers.pdf">Slurm: Seamless Integration With Unprivileged Containers</a>,
doc/html/publications.shtml:Luke Yeager et al., NVIDIA</li>
doc/html/publications.shtml:<li>Technical: <a href="SLUG15/rcuda.pdf">Increasing cluster throughput with Slurm and rCUDA</a>,
doc/html/publications.shtml:<li><a href="SUG14/remote_gpu.pdf">Extending Slurm with Support for Remote GPU Virtualization</a>
doc/html/publications.shtml:<li><a href="slurm_ug_2012/SUG2012-Soner.pdf">Integer Programming Based Herogeneous CPU-GPU Clusters</a>,
doc/html/platforms.shtml:<li><b>gres/gpu</b> &mdash; several autodetection plugins are available for
doc/html/platforms.shtml:<li><b>AutoDetect=nvml</b> enables autodetection of NVIDIA GPUs through their
doc/html/platforms.shtml:<li><b>AutoDetect=nvidia</b> also enables autodetection of NVIDIA GPUs, but
doc/html/platforms.shtml:<li><b>AutoDetect=rsmi</b> enables autodetection of AMD GPUs through their
doc/html/platforms.shtml:<li><b>AutoDetect=oneapi</b> enables autodetection of Intel GPUs through their
doc/html/platforms.shtml:<li><b>gres/mps</b> &mdash; NVIDIA CUDA Multi-Process Service provides ways to
doc/html/platforms.shtml:share GPUs between multiple compute processes</li>
doc/html/platforms.shtml:GPUs between multiple compute processes</li>
doc/html/gres.shtml:<li><a href="#GPU_Management">GPU Management</a></li>
doc/html/gres.shtml:including Graphics Processing Units (GPUs), CUDA Multi-Process Service (MPS)
doc/html/gres.shtml:# Configure four GPUs (with MPS), plus bandwidth
doc/html/gres.shtml:GresTypes=gpu,mps,bandwidth
doc/html/gres.shtml:NodeName=tux[0-7] Gres=gpu:tesla:2,gpu:kepler:2,mps:400,bandwidth:lustre:no_consume:4G
doc/html/gres.shtml:<DT><I>--gpus</I></DT>
doc/html/gres.shtml:<DD>GPUs required per job</DD>
doc/html/gres.shtml:<DT><I>--gpus-per-node</I></DT>
doc/html/gres.shtml:<DD>GPUs required per node. Equivalent to the <I>--gres</I> option for GPUs.</DD>
doc/html/gres.shtml:<DT><I>--gpus-per-socket</I></DT>
doc/html/gres.shtml:<DD>GPUs required per socket. Requires the job to specify a task socket.</DD>
doc/html/gres.shtml:<DT><I>--gpus-per-task</I></DT>
doc/html/gres.shtml:<DD>GPUs required per task. Requires the job to specify a task count.</DD>
doc/html/gres.shtml:Note that all of the <I>--gpu*</I> options are only supported by Slurm's
doc/html/gres.shtml:while all of the <I>--gpu*</I> options require an argument of the form
doc/html/gres.shtml:specific model of GPU).
doc/html/gres.shtml:<I>sbatch --gres=gpu:kepler:2 ...</I>.</P>
doc/html/gres.shtml:within a job. For example, if you request <i>--gres=gpu:2</i> with
doc/html/gres.shtml:<b>sbatch</b>, you would not be able to request <i>--gres=gpu:tesla:2</i>
doc/html/gres.shtml:if you request a typed GPU to create a job allocation, you should request
doc/html/gres.shtml:a GPU of the same type to create a job step.</p>
doc/html/gres.shtml:specifically for GPUs and detailed descriptions about these options are
doc/html/gres.shtml:As for the <I>--gpu*</I> option, these options are only supported by Slurm's
doc/html/gres.shtml:<DT><I>--cpus-per-gpu</I></DT>
doc/html/gres.shtml:<DD>Count of CPUs allocated per GPU.</DD>
doc/html/gres.shtml:<DT><I>--gpu-bind</I></DT>
doc/html/gres.shtml:<DD>Define how tasks are bound to GPUs.</DD>
doc/html/gres.shtml:<DT><I>--gpu-freq</I></DT>
doc/html/gres.shtml:<DD>Specify GPU frequency and/or GPU memory frequency.</DD>
doc/html/gres.shtml:<DT><I>--mem-per-gpu</I></DT>
doc/html/gres.shtml:<DD>Memory allocated per GPU.</DD>
doc/html/gres.shtml:# sbatch --gres=gpu:4 -n4 -N1-1 gres_test.bash
doc/html/gres.shtml:srun --gres=gpu:2 -n2 --exclusive show_device.sh &
doc/html/gres.shtml:srun --gres=gpu:1 -n1 --exclusive show_device.sh &
doc/html/gres.shtml:srun --gres=gpu:1 -n1 --exclusive show_device.sh &
doc/html/gres.shtml:<p>If <i>AutoDetect=nvml</i>, <i>AutoDetect=nvidia</i>, <i>AutoDetect=rsmi</i>,
doc/html/gres.shtml:GPU. This removes the need to explicitly configure GPUs in gres.conf, though the
doc/html/gres.shtml:and <i>AutoDetect=oneapi</i> need their corresponding GPU management libraries
doc/html/gres.shtml:Both <i>AutoDetect=nvml</i> and <i>AutoDetect=nvidia</i> detect NVIDIA GPUs.
doc/html/gres.shtml:<i>AutoDetect=nvidia</i> (added in Slurm 24.11) doesn't require the
doc/html/gres.shtml:However, if <i>Type</i> and <i>File</i> in gres.conf match a GPU on
doc/html/gres.shtml:If the system-detected GPU differs from its matching GPU configuration, then the
doc/html/gres.shtml:GPU is omitted from the node with an error.
doc/html/gres.shtml:administrators of any unexpected changes in GPU properties.
doc/html/gres.shtml:# Configure four GPUs (with MPS), plus bandwidth
doc/html/gres.shtml:Name=gpu Type=gp100  File=/dev/nvidia0 Cores=0,1
doc/html/gres.shtml:Name=gpu Type=gp100  File=/dev/nvidia1 Cores=0,1
doc/html/gres.shtml:Name=gpu Type=p6000  File=/dev/nvidia2 Cores=2,3
doc/html/gres.shtml:Name=gpu Type=p6000  File=/dev/nvidia3 Cores=2,3
doc/html/gres.shtml:Name=mps Count=200  File=/dev/nvidia0
doc/html/gres.shtml:Name=mps Count=200  File=/dev/nvidia1
doc/html/gres.shtml:Name=mps Count=100  File=/dev/nvidia2
doc/html/gres.shtml:Name=mps Count=100  File=/dev/nvidia3
doc/html/gres.shtml:for each GPU will be checked against a corresponding GPU found on the system
doc/html/gres.shtml:If a matching system GPU is not found, no validation takes place and the GPU is
doc/html/gres.shtml:match or be a substring of the GPU name reported by slurmd via the AutoDetect
doc/html/gres.shtml:mechanism. This GPU name will have all spaces replaced with underscores. To see
doc/html/gres.shtml:the detected GPUs and their names, run: <code class="commandline">slurmd -C
doc/html/gres.shtml:NodeName=node0 ... Gres=gpu:geforce_rtx_2060:1 ...
doc/html/gres.shtml:Found gpu:geforce_rtx_2060:1 with Autodetect=nvml (Substring of gpu name may be used instead)
doc/html/gres.shtml:<P>In this example, the GPU's name is reported as
doc/html/gres.shtml:gres.conf, the GPU <i>Type</i> can be set to <code class="commandline">
doc/html/gres.shtml:<h2 id="GPU_Management">GPU Management
doc/html/gres.shtml:<a class="slurm_link" href="#GPU_Management"></a>
doc/html/gres.shtml:<P>In the case of Slurm's GRES plugin for GPUs, the environment variable
doc/html/gres.shtml:<code class="commandline">CUDA_VISIBLE_DEVICES</code>
doc/html/gres.shtml:is set for each job step to determine which GPUs are
doc/html/gres.shtml:for the <i>sbatch</i> command only reflects the GPUs allocated to that job
doc/html/gres.shtml:CUDA version 3.1 (or higher) uses this environment
doc/html/gres.shtml:variable in order to run multiple jobs or job steps on a node with GPUs
doc/html/gres.shtml:case, <code class="commandline">CUDA_VISIBLE_DEVICES</code>
doc/html/gres.shtml:JobStep=1234.0 CUDA_VISIBLE_DEVICES=0,1
doc/html/gres.shtml:JobStep=1234.1 CUDA_VISIBLE_DEVICES=2
doc/html/gres.shtml:JobStep=1234.2 CUDA_VISIBLE_DEVICES=3
doc/html/gres.shtml:<p>The <code class="commandline">CUDA_VISIBLE_DEVICES</code>
doc/html/gres.shtml:For example, if a job is allocated the device "/dev/nvidia1", then
doc/html/gres.shtml:<code class="commandline">CUDA_VISIBLE_DEVICES</code> will be set to a value of
doc/html/gres.shtml:<code class="commandline">CUDA_VISIBLE_DEVICES</code> will be set to a
doc/html/gres.shtml:value of "0" (i.e. the first GPU device visible to the job).
doc/html/gres.shtml:<p>When possible, Slurm automatically determines the GPUs on the system using
doc/html/gres.shtml:<code class="commandline">nvidia-smi</code> tool) numbers GPUs in order by their
doc/html/gres.shtml:PCI bus IDs. For this numbering to match the numbering reported by CUDA, the
doc/html/gres.shtml:<code class="commandline">CUDA_DEVICE_ORDER</code> environmental variable must
doc/html/gres.shtml:be set to <code class="commandline">CUDA_DEVICE_ORDER=PCI_BUS_ID</code>.</p>
doc/html/gres.shtml:<p>GPU device files (e.g. <i>/dev/nvidia1</i>) are
doc/html/gres.shtml:However, an after-bootup check is required to guarantee that a GPU device is
doc/html/gres.shtml:<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars">
doc/html/gres.shtml:NVIDIA CUDA documentation</a> for more information about the
doc/html/gres.shtml:<code class="commandline">CUDA_VISIBLE_DEVICES</code> and
doc/html/gres.shtml:<code class="commandline">CUDA_DEVICE_ORDER</code> environmental variables.</p>
doc/html/gres.shtml:<p><a href="https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf">
doc/html/gres.shtml:CUDA Multi-Process Service (MPS)</a> provides a mechanism where GPUs can be
doc/html/gres.shtml:GPU's resources.
doc/html/gres.shtml:the <I>slurm.conf</I> file (e.g. "NodeName=tux[1-16] Gres=gpu:2,mps:200").
doc/html/gres.shtml:<I>slurm.conf</I> will be evenly distributed across all GPUs configured on the
doc/html/gres.shtml:node. For example, "NodeName=tux[1-16] Gres=gpu:2,mps:200" will configure
doc/html/gres.shtml:a count of 100 gres/mps resources on each of the two GPUs.</li>
doc/html/gres.shtml:The count of gres/mps elements will be evenly distributed across all GPUs
doc/html/gres.shtml:GPU and the <I>Count</I> should identify the number of gres/mps resources
doc/html/gres.shtml:available for that specific GPU device.
doc/html/gres.shtml:For example, some GPUs on a node may be more powerful than others and thus be
doc/html/gres.shtml:Another use case would be to prevent some GPUs from being used for MPS (i.e.
doc/html/gres.shtml:That information is copied from the gres/gpu configuration.</p>
doc/html/gres.shtml:<p>Note that if NVIDIA's NVML library is installed, the GPU configuration
doc/html/gres.shtml:<p>By default, job requests for MPS are required to fit on a single gpu on
doc/html/gres.shtml:<p>Note the same GPU can be allocated either as a GPU type of GRES or as
doc/html/gres.shtml:In other words, once a GPU has been allocated as a gres/gpu resource it will
doc/html/gres.shtml:Likewise, once a GPU has been allocated as a gres/mps resource it will
doc/html/gres.shtml:not be available as a gres/gpu.
doc/html/gres.shtml:However the same GPU can be allocated as MPS generic resources to multiple jobs
doc/html/gres.shtml:GRES (GPU) this option only allocates all sharing GRES and no underlying shared
doc/html/gres.shtml:# Configure four GPUs (with MPS)
doc/html/gres.shtml:Name=gpu Type=gp100 File=/dev/nvidia0 Cores=0,1
doc/html/gres.shtml:Name=gpu Type=gp100 File=/dev/nvidia1 Cores=0,1
doc/html/gres.shtml:Name=gpu Type=p6000 File=/dev/nvidia2 Cores=2,3
doc/html/gres.shtml:Name=gpu Type=p6000 File=/dev/nvidia3 Cores=2,3
doc/html/gres.shtml:# Set gres/mps Count value to 100 on each of the 4 available GPUs
doc/html/gres.shtml:# Configure four different GPU types (with MPS)
doc/html/gres.shtml:Name=gpu Type=gtx1080 File=/dev/nvidia0 Cores=0,1
doc/html/gres.shtml:Name=gpu Type=gtx1070 File=/dev/nvidia1 Cores=0,1
doc/html/gres.shtml:Name=gpu Type=gtx1060 File=/dev/nvidia2 Cores=2,3
doc/html/gres.shtml:Name=gpu Type=gtx1050 File=/dev/nvidia3 Cores=2,3
doc/html/gres.shtml:Name=mps Count=1300   File=/dev/nvidia0
doc/html/gres.shtml:Name=mps Count=1200   File=/dev/nvidia1
doc/html/gres.shtml:Name=mps Count=1100   File=/dev/nvidia2
doc/html/gres.shtml:Name=mps Count=1000   File=/dev/nvidia3
doc/html/gres.shtml:that the request must be satisfied using only one GPU per node and only one
doc/html/gres.shtml:GPU per node may be configured for use with MPS.
doc/html/gres.shtml:20 percent of one GPU and 30 percent of a second GPU on a single node.
doc/html/gres.shtml:Note that GRES types of GPU <u>and</u> MPS can not be requested within
doc/html/gres.shtml:Also jobs requesting MPS resources can not specify a GPU frequency.</p>
doc/html/gres.shtml:Prolog will have the <code class="commandline">CUDA_VISIBLE_DEVICES</code>,
doc/html/gres.shtml:<code class="commandline">CUDA_MPS_ACTIVE_THREAD_PERCENTAGE</code>, and
doc/html/gres.shtml:The Prolog should then make sure that an MPS server is started for that GPU
doc/html/gres.shtml:It also records the GPU device ID in a local file.
doc/html/gres.shtml:If a job is allocated gres/gpu resources then the Prolog will have the
doc/html/gres.shtml:<code class="commandline">CUDA_VISIBLE_DEVICES</code> and
doc/html/gres.shtml:(no <code class="commandline">CUDA_MPS_ACTIVE_THREAD_PERCENTAGE</code>).
doc/html/gres.shtml:The Prolog should then terminate any MPS server associated with that GPU.
doc/html/gres.shtml:<code class="commandline">CUDA_VISIBLE_DEVICES</code>
doc/html/gres.shtml:and <code class="commandline">CUDA_DEVICE_ORDER</code> environment variables set.
doc/html/gres.shtml:always have a value of zero in the current implementation (only one GPU will be
doc/html/gres.shtml:<code class="commandline">CUDA_MPS_ACTIVE_THREAD_PERCENTAGE</code>
doc/html/gres.shtml:the assigned GPU.
doc/html/gres.shtml:15% of the gtx1080 (File=/dev/nvidia0, 200 x 100 / 1300 = 15), or<br>
doc/html/gres.shtml:16% of the gtx1070 (File=/dev/nvidia0, 200 x 100 / 1200 = 16), or<br>
doc/html/gres.shtml:18% of the gtx1060 (File=/dev/nvidia0, 200 x 100 / 1100 = 18), or<br>
doc/html/gres.shtml:20% of the gtx1050 (File=/dev/nvidia0, 200 x 100 / 1000 = 20).</p>
doc/html/gres.shtml:GPUs then trigger the starting of an MPS server based upon comments in the job.
doc/html/gres.shtml:For example, if a job is allocated whole GPUs then search for a comment of
doc/html/gres.shtml:"mps-per-gpu" or "mps-per-node" in the job (using the "scontrol show job"
doc/html/gres.shtml:command) and use that as a basis for starting one MPS daemon per GPU or across
doc/html/gres.shtml:all GPUs respectively.</p>
doc/html/gres.shtml:<a href="https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf">
doc/html/gres.shtml:NVIDIA Multi-Process Service documentation</a> for more information about MPS.</p>
doc/html/gres.shtml:Note that a vulnerability exists in previous versions of the NVIDIA driver that
doc/html/gres.shtml:may affect users when sharing GPUs. More information can be found in
doc/html/gres.shtml:<a href="https://nvidia.custhelp.com/app/answers/detail/a_id/4772">
doc/html/gres.shtml:Security Bulletin: NVIDIA GPU Display Driver - February 2019</a>.</p>
doc/html/gres.shtml:<p>NVIDIA MPS has a built-in limitation regarding GPU sharing among different
doc/html/gres.shtml:leading to serialized exclusive access of the GPU between users (see
doc/html/gres.shtml:<a href="https://docs.nvidia.com/deploy/mps/index.html#topic_3_3_1_1">
doc/html/gres.shtml:truly run concurrently on the GPU with MPS; rather, the GPU will be time-sliced
doc/html/gres.shtml:<a href="https://docs.nvidia.com/deploy/mps/index.html#topic_4_3">
doc/html/gres.shtml:<p>Beginning in version 21.08, Slurm now supports NVIDIA
doc/html/gres.shtml:<i>Multi-Instance GPU</i> (MIG) devices. This feature allows some newer NVIDIA
doc/html/gres.shtml:GPUs (like the A100) to split up a GPU into up to seven separate, isolated GPU
doc/html/gres.shtml:instances. Slurm can treat these MIG instances as individual GPUs, complete with
doc/html/gres.shtml:in <i>slurm.conf</i> as if the MIGs were regular GPUs, like this:
doc/html/gres.shtml:<code class="commandline">NodeName=tux[1-16] gres=gpu:2</code>. An optional
doc/html/gres.shtml:other, as well as from other GPUs in the cluster. This type must be a substring
doc/html/gres.shtml:for a system with 2 gpus, one of which is partitioned into 2 MIGs where the
doc/html/gres.shtml:"MIG Profile" is <code class="commandline">nvidia_a100_3g.20gb</code>:</p>
doc/html/gres.shtml:AccountingStorageTRES=gres/gpu,gres/gpu:a100,gres/gpu:a100_3g.20gb
doc/html/gres.shtml:GresTypes=gpu
doc/html/gres.shtml:NodeName=tux[1-16] gres=gpu:a100:1,gpu:a100_3g.20gb:2
doc/html/gres.shtml:allows you to specify multiple device files for the GPU card.</p>
doc/html/gres.shtml:<p>For more information on NVIDIA MIGs (including how to partition them), see
doc/html/gres.shtml:<a href="https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html">
doc/html/gres.shtml:Sharding provides a generic mechanism where GPUs can be
doc/html/gres.shtml:GPU it does not fence the processes running on the GPU, it only allows the GPU
doc/html/gres.shtml:the <I>slurm.conf</I> file (e.g. "NodeName=tux[1-16] Gres=gpu:2,shard:64").
doc/html/gres.shtml:<I>slurm.conf</I> will be evenly distributed across all GPUs configured on the
doc/html/gres.shtml:node. For example, "NodeName=tux[1-16] Gres=gpu:2,shard:64" will configure
doc/html/gres.shtml:a count of 32 gres/shard resources on each of the two GPUs.</li>
doc/html/gres.shtml:The count of gres/shard elements will be evenly distributed across all GPUs
doc/html/gres.shtml:GPU and the <I>Count</I> should identify the number of gres/shard resources
doc/html/gres.shtml:available for that specific GPU device.
doc/html/gres.shtml:For example, some GPUs on a node may be more powerful than others and thus be
doc/html/gres.shtml:Another use case would be to prevent some GPUs from being used for sharding (i.e.
doc/html/gres.shtml:That information is copied from the gres/gpu configuration.</p>
doc/html/gres.shtml:<p>Note that if NVIDIA's NVML library is installed, the GPU configuration
doc/html/gres.shtml:<p>Note the same GPU can be allocated either as a GPU type of GRES or as
doc/html/gres.shtml:In other words, once a GPU has been allocated as a gres/gpu resource it will
doc/html/gres.shtml:Likewise, once a GPU has been allocated as a gres/shard resource it will
doc/html/gres.shtml:not be available as a gres/gpu.
doc/html/gres.shtml:However the same GPU can be allocated as shard generic resources to multiple jobs
doc/html/gres.shtml:<p>By default, job requests for shards are required to fit on a single gpu on
doc/html/gres.shtml:AccountingStorageTRES=gres/gpu,gres/shard
doc/html/gres.shtml:GresTypes=gpu,shard
doc/html/gres.shtml:NodeName=tux[1-16] Gres=gpu:2,shard:64
doc/html/gres.shtml:# Configure four GPUs (with Sharding)
doc/html/gres.shtml:Name=gpu Type=gp100 File=/dev/nvidia0 Cores=0,1
doc/html/gres.shtml:Name=gpu Type=gp100 File=/dev/nvidia1 Cores=0,1
doc/html/gres.shtml:Name=gpu Type=p6000 File=/dev/nvidia2 Cores=2,3
doc/html/gres.shtml:Name=gpu Type=p6000 File=/dev/nvidia3 Cores=2,3
doc/html/gres.shtml:# Set gres/shard Count value to 8 on each of the 4 available GPUs
doc/html/gres.shtml:# Configure four different GPU types (with Sharding)
doc/html/gres.shtml:Name=gpu Type=gtx1080 File=/dev/nvidia0 Cores=0,1
doc/html/gres.shtml:Name=gpu Type=gtx1070 File=/dev/nvidia1 Cores=0,1
doc/html/gres.shtml:Name=gpu Type=gtx1060 File=/dev/nvidia2 Cores=2,3
doc/html/gres.shtml:Name=gpu Type=gtx1050 File=/dev/nvidia3 Cores=2,3
doc/html/gres.shtml:Name=shard Count=8    File=/dev/nvidia0
doc/html/gres.shtml:Name=shard Count=8    File=/dev/nvidia1
doc/html/gres.shtml:Name=shard Count=8    File=/dev/nvidia2
doc/html/gres.shtml:Name=shard Count=8    File=/dev/nvidia3
doc/html/gres.shtml:<p>Job requests for shards can not specify a GPU frequency.</p>
doc/html/gres.shtml:<code class="commandline">CUDA_VISIBLE_DEVICES</code>, <code class="commandline">ROCR_VISIBLE_DEVICES</code>,
doc/html/gres.shtml:or <code class="commandline">GPU_DEVICE_ORDINAL</code> environment variable set
doc/html/gres.shtml:which would be the same as if it were a GPU.
doc/html/resource_limits.shtml:sacctmgr modify user bob set GrpTRES=cpu=1500,mem=200,gres/gpu=50
doc/html/resource_limits.shtml:sacctmgr modify user bob set GrpTRES=cpu=-1,mem=-1,gres/gpu=-1
doc/html/resource_limits.shtml:  over this specific type (e.g. <i>MaxTRESPerUser=gres/gpu:tesla=1</i>) if a
doc/html/resource_limits.shtml:  become useful. For example, if one requests <i>--gres=gpu:2</i> having a
doc/html/resource_limits.shtml:  limit set of <i>MaxTRESPerUser=gres/gpu:tesla=1</i>, the limit won't be
doc/html/resource_limits.shtml:	 bad = string.match(g,'^gres/gpu[:=]*[0-9]*$')
doc/html/resource_limits.shtml:	    slurm.log_info("User specified gpu GRES without type: %s", bad)
doc/html/resource_limits.shtml:	    slurm.user_msg("You must always specify a type when requesting gpu GRES")
doc/html/resource_limits.shtml:  specify a gpu with its type, thus enforcing the limits for each specific
doc/html/resource_limits.shtml:non-typed resources should be included. For example, if you have 'tesla' GPUs
doc/html/resource_limits.shtml:GPU resource, then those weights will not be applied to the generic GPUs.</p>
doc/html/resource_limits.shtml:  of a gres won't be accounted for. For example, to track generic GPUs and
doc/html/resource_limits.shtml:  Tesla GPUs, you would set this in your slurm.conf:
doc/html/resource_limits.shtml:  AccountingStorageTRES=gres/gpu,gres/gpu:tesla
doc/html/configurator.easy.html.in:<b>cons_tres</b>: Allocate individual processors, memory, GPUs, and other
doc/html/rest_api.shtml:    "gpu_spec" : "gpu_spec",
doc/html/rest_api.shtml:    "res_cores_per_gpu" : 6,
doc/html/rest_api.shtml:    "gpu_spec" : "gpu_spec",
doc/html/rest_api.shtml:    "res_cores_per_gpu" : 6,
doc/html/rest_api.shtml:    "gpu_spec" : "gpu_spec",
doc/html/rest_api.shtml:    "res_cores_per_gpu" : 6,
doc/html/rest_api.shtml:    "gpu_spec" : "gpu_spec",
doc/html/rest_api.shtml:    "res_cores_per_gpu" : 6,
doc/html/rest_api.shtml:<div class="param">gpu_spec (optional)</div><div class="param-desc"><span class="param-type"><a href="#string">String</a></span>  </div>
doc/html/rest_api.shtml:<div class="param">res_cores_per_gpu (optional)</div><div class="param-desc"><span class="param-type"><a href="#integer">Integer</a></span>  format: int32</div>
doc/html/cgroups.shtml:<li>Confine jobs, steps and tasks to their allocated gres, including gpus.</li>
doc/html/cons_tres.shtml:related to GPUs.</p>
doc/html/cons_tres.shtml:  <li><b>DefCpuPerGPU</b>: Default number of CPUs allocated per GPU.</li>
doc/html/cons_tres.shtml:  <li><b>DefMemPerGPU</b>: Default amount of memory allocated per GPU.</li>
doc/html/cons_tres.shtml:  <li><b>--cpus-per-gpu=</b>: Number of CPUs for every GPU.</li>
doc/html/cons_tres.shtml:  <li><b>--gpus=</b>: Count of GPUs for entire job allocation.</li>
doc/html/cons_tres.shtml:  <li><b>--gpu-bind=</b>: Bind task to specific GPU(s).</li>
doc/html/cons_tres.shtml:  <li><b>--gpu-freq=</b>: Request specific GPU/memory frequencies.</li>
doc/html/cons_tres.shtml:  <li><b>--gpus-per-node=</b>: Number of GPUs per node.</li>
doc/html/cons_tres.shtml:  <li><b>--gpus-per-socket=</b>: Number of GPUs per socket.</li>
doc/html/cons_tres.shtml:  <li><b>--gpus-per-task=</b>: Number of GPUs per task.</li>
doc/html/cons_tres.shtml:  <li><b>--mem-per-gpu=</b>: Amount of memory for each GPU.</li>
doc/html/cons_tres.shtml:co-scheduled on nodes when resources permit it. Generic resources (such as GPUs)
doc/html/dynamic_nodes.shtml:slurmd -Z --conf "RealMemory=80000 Gres=gpu:2 Feature=f1"
doc/html/dynamic_nodes.shtml:scontrol create NodeName=d[1-100] CPUs=16 Boards=1 SocketsPerBoard=1 CoresPerSocket=8 ThreadsPerCore=2 RealMemory=31848 Gres=gpu:2 Feature=f1 State=cloud
doc/html/slurm_ug_agenda.shtml:<p>CoreWeave is a premium cloud provider specializing in high performance GPU-powered workloads for AI/ML, batch processing, and scientific discovery. CoreWeave deploys massive scales of compute and some of the largest dedicated training clusters on the planet, all on top of Kubernetes. As the top choice for scheduling and managing HPC workloads, Slurm is a must-have solution for utilizing compute at this scale for batch workloads. In this talk, we will present the soon to be open-sourced Slurm on Kubernetes (SUNK) solution, a project in collaboration with SchedMD, that brings Kubernetes containerized deployments and Slurm together to provide the ultimate computing platform. We will discuss how SUNK was developed, its range of capabilities, and the role it played in the record-breaking MLPerf submission we completed with NVIDIA.</p>
doc/html/slurm_ug_agenda.shtml:<p>This presentation will discuss how the Research Data Services (RDS) team at the San Diego Supercomputer Center (SDSC) uses Slurm to support genomics researchers developing machine learning techniques for conducting genome-wide association studies and computational network biology at the University of California San Diego (UCSD). Genomics machine learning requires high throughput computing across heterogenous hardware to meet the workflow demands of novel model development and training. The presentation will go through the configuration of a specially built National Resource for Network Biology (NRNB) compute cluster. The NRNB cluster consists of a heterogeneous node configuration including standard compute nodes, high memory nodes, and different GPU nodes to support about 50 genomics researchers. Slurm is used to manage resources on the cluster to reduce time to discovery for the researchers by tuning the environment for their specific needs. The presentation will discuss Slurm job throughput tuning for thousands of sub-node sized jobs, heterogeneous resource allocation and fair use, storage allocation, and deploying development Jupyter environments through Slurm. Furthermore, the presentation will demonstrate how Slurm is being used to automate sequence data ingestion and processing as part of the Institute for Genomics Medicine to support computational genomics efforts.</p>
doc/html/faq.shtml:<li><a href="#opencl_pmix">Multi-Instance GPU not working with Slurm and
doc/html/faq.shtml:  PMIx; GPUs are &quot;In use by another client&quot;</a></li>
doc/html/faq.shtml:<p><a id="opencl_pmix"><b>Multi-Instance GPU not working with Slurm and PMIx;
doc/html/faq.shtml:  GPUs are &quot;In use by another client&quot;</b></a><br/>
doc/html/faq.shtml:querying the OpenCL devices, which creates handles on <i>/dev/nvidia*</i> files.
doc/html/faq.shtml:$ nvidia-smi mig --id 1 --create-gpu-instance FOO,FOO --default-compute-instance
doc/html/faq.shtml:Unable to create a GPU instance on GPU 1 using profile FOO: In use by another client
doc/html/faq.shtml:In order to use Multi-Instance GPUs with Slurm and PMIx you can instruct hwloc
doc/html/faq.shtml:to not query OpenCL devices by setting the
doc/html/faq.shtml:<span class="commandline">HWLOC_COMPONENTS=-opencl</span> environment
doc/html/tres.shtml:  <pre>AccountingStorageTRES=gres/gpu,license/iop1</pre>
doc/html/tres.shtml:  with a GRES called gpu, as well as a license called iop1. Whenever these
doc/html/tres.shtml:be accounted for. For example, if we want to account for gres/gpu:tesla,
doc/html/tres.shtml:we would also include gres/gpu for accounting gpus in requests like
doc/html/tres.shtml:<i>srun --gres=gpu:1</i>.
doc/html/tres.shtml:<pre>AccountingStorageTRES=gres/gpu,gres/gpu:tesla</pre>
doc/html/tres.shtml:<p><b>NOTE</b>: Setting gres/gpu will also set gres/gpumem and gres/gpuutil.
doc/html/tres.shtml:gres/gpumem and gres/gpuutil can be set individually when gres/gpu is not set.
doc/html/tres.shtml:<pre>PriorityWeightTRES=CPU=1000,Mem=2000,GRES/gpu=3000</pre>
doc/html/tres.shtml:<pre>TRESBillingWeights="CPU=1.0,Mem=0.25G,GRES/gpu=2.0,license/licA=1.5"</pre>
doc/html/containers.shtml:<h3 id="nvidia_create_start">
doc/html/containers.shtml:oci.conf example for nvidia-container-runtime using create/start:
doc/html/containers.shtml:<a class="slurm_link" href="#nvidia_create_start"></a></h3>
doc/html/containers.shtml:RunTimeQuery="nvidia-container-runtime --rootless=true --root=/run/user/%U/ state %n.%u.%j.%s.%t"
doc/html/containers.shtml:RunTimeCreate="nvidia-container-runtime --rootless=true --root=/run/user/%U/ create %n.%u.%j.%s.%t -b %b"
doc/html/containers.shtml:RunTimeStart="nvidia-container-runtime --rootless=true --root=/run/user/%U/ start %n.%u.%j.%s.%t"
doc/html/containers.shtml:RunTimeKill="nvidia-container-runtime --rootless=true --root=/run/user/%U/ kill -a %n.%u.%j.%s.%t"
doc/html/containers.shtml:RunTimeDelete="nvidia-container-runtime --rootless=true --root=/run/user/%U/ delete --force %n.%u.%j.%s.%t"
doc/html/containers.shtml:<h3 id="nvidia_run">
doc/html/containers.shtml:oci.conf example for nvidia-container-runtime using run (recommended over using
doc/html/containers.shtml:create/start):<a class="slurm_link" href="#nvidia_run"></a></h3>
doc/html/containers.shtml:RunTimeQuery="nvidia-container-runtime --rootless=true --root=/run/user/%U/ state %n.%u.%j.%s.%t"
doc/html/containers.shtml:RunTimeKill="nvidia-container-runtime --rootless=true --root=/run/user/%U/ kill -a %n.%u.%j.%s.%t"
doc/html/containers.shtml:RunTimeDelete="nvidia-container-runtime --rootless=true --root=/run/user/%U/ delete --force %n.%u.%j.%s.%t"
doc/html/containers.shtml:RunTimeRun="nvidia-container-runtime --rootless=true --root=/run/user/%U/ run %n.%u.%j.%s.%t -b %b"
doc/html/containers.shtml:<a href="https://github.com/NVIDIA/enroot">Enroot</a> (3.3.0)
doc/html/containers.shtml:<p><a href="https://github.com/NVIDIA/enroot">Enroot</a> is a user namespace
doc/html/containers.shtml:container system sponsored by <a href="https://www.nvidia.com">NVIDIA</a>
doc/html/containers.shtml:		<a href="https://github.com/NVIDIA/pyxis">pyxis</a>
doc/html/containers.shtml:	<li>Native support for Nvidia GPUs</li>
doc/html/containers.shtml:	<li>NVIDIA GPU Support</li>
doc/html/reservations.shtml:In the following example, a reservation is created in the 'gpu' partition
doc/html/reservations.shtml:   user=user1 partition=gpu tres=cpu=24,node=4
doc/html/reservations.shtml:   Features=(null) PartitionName=gpu
doc/html/prolog_epilog.shtml:<li><b>CUDA_MPS_ACTIVE_THREAD_PERCENTAGE</b>
doc/html/prolog_epilog.shtml:Specifies the percentage of a GPU that should be allocated to the job.
doc/html/prolog_epilog.shtml:<li><b>CUDA_VISIBLE_DEVICES</b>
doc/html/prolog_epilog.shtml:Specifies the GPU devices for the job allocation.
doc/html/prolog_epilog.shtml:The value is set only if the gres/gpu or gres/mps plugin is configured and the
doc/html/prolog_epilog.shtml:For example, if a job is allocated the device "/dev/nvidia1", then
doc/html/prolog_epilog.shtml:<span class="commandline">CUDA_VISIBLE_DEVICES</span> will be set to a value of
doc/html/prolog_epilog.shtml:<span class="commandline">CUDA_VISIBLE_DEVICES</span> will be set to a
doc/html/prolog_epilog.shtml:value of "0" (i.e. the first GPU device visible to the job).
doc/html/prolog_epilog.shtml:<span class="commandline">CUDA_VISIBLE_DEVICES</span> will be set unless
doc/html/prolog_epilog.shtml:<i>gres.conf</i>. See also <span class="commandline">SLURM_JOB_GPUS</span>.
doc/html/prolog_epilog.shtml:<li><b>GPU_DEVICE_ORDINAL</b>
doc/html/prolog_epilog.shtml:Specifies the GPU devices for the job allocation. The considerations for
doc/html/prolog_epilog.shtml:<span class="commandline">CUDA_VISIBLE_DEVICES</span> also apply to
doc/html/prolog_epilog.shtml:<span class="commandline">GPU_DEVICE_ORDINAL</span>.
doc/html/prolog_epilog.shtml:Specifies the GPU devices for the job allocation. The considerations for
doc/html/prolog_epilog.shtml:<span class="commandline">CUDA_VISIBLE_DEVICES</span> also apply to
doc/html/prolog_epilog.shtml:<li><b>SLURM_GPUS</b>
doc/html/prolog_epilog.shtml:Count of the GPUs available to the job. Available in SrunProlog, TaskProlog,
doc/html/prolog_epilog.shtml:<li><b>SLURM_JOB_GPUS</b>
doc/html/prolog_epilog.shtml:The GPU IDs of GPUs in the job allocation (if any).
doc/html/gres_design.shtml:GRES use would be GPUs. GRES are identified by a specific name and use an
doc/html/gres_design.shtml:(e.g. "gpu:2,nic:1"). This string is also visible to various Slurm commands
doc/html/gres_design.shtml:(e.g. one data structure for "gpu" and a second structure with information
doc/html/gres_design.shtml:| gres = "gpu:2,nic:1" |
doc/html/gres_design.shtml:   | id = 123 (gpu) |               | id = 124 (nic) |
doc/html/gres_design.shtml:node (a job can not have 2 GPUs on one node and 1 GPU on a second node),
doc/html/gres_design.shtml:program directing it to GRES which have been allocated for its use (the CUDA
doc/html/gres_design.shtml:libraries base their GPU selection upon environment variables, so this logic
doc/html/gres_design.shtml:should work for CUDA today if users do not attempt to manipulate the
doc/html/gres_design.shtml:environment variables reserved for CUDA use).</p>
doc/html/quickstart_admin.shtml:<li> <b>AMD GPU Support</b> Autodetection of AMD GPUs will be available
doc/html/quickstart_admin.shtml:		if the <i>ROCm</i> development library is installed.
doc/html/quickstart_admin.shtml:<li> <b>Intel GPU Support</b> Autodetection of Intel GPUs will be available
doc/html/quickstart_admin.shtml:<li> <b>NVIDIA GPU Support</b> Autodetection of NVIDIA GPUs with MIGs and
doc/html/quickstart_admin.shtml:		NVlinks will be available if the <i>libnvidia-ml</i> development
doc/man/man5/helpers.conf.5:Copyright (C) 2021 NVIDIA CORPORATION. All rights reserved.
doc/man/man5/job_container.conf.5:The second will only be on gpu[1\-10], will be expected to exist and will run
doc/man/man5/job_container.conf.5:NodeName=gpu[1\-10] BasePath=/var/nvme/storage_b InitScript=/etc/slurm/init.sh
doc/man/man5/gres.conf.5:\fBnvidia\fR
doc/man/man5/gres.conf.5:Automatically detect NVIDIA GPUs. No library required, but doesn't detect MIGs
doc/man/man5/gres.conf.5:Automatically detect NVIDIA GPUs. Requires the NVIDIA Management Library (NVML).
doc/man/man5/gres.conf.5:Do not automatically detect any GPUs. Used to override other options.
doc/man/man5/gres.conf.5:Automatically detect Intel GPUs. Requires the Intel Graphics Compute Runtime for
doc/man/man5/gres.conf.5:oneAPI Level Zero and OpenCL Driver (oneapi).
doc/man/man5/gres.conf.5:Automatically detect AMD GPUs. Requires the ROCm System Management Interface
doc/man/man5/gres.conf.5:(ROCm SMI) Library.
doc/man/man5/gres.conf.5:E.g.: \fINodeName=tux3 AutoDetect=off Name=gpu File=/dev/nvidia[0\-3]\fR.
doc/man/man5/gres.conf.5:\-\-exact and task binding through \-\-gpu\-\bind
doc/man/man5/gres.conf.5:more CPUs than are bound to a GRES (e.g. if a GPU is bound to the CPUs on one
doc/man/man5/gres.conf.5:(e.g. \fIFile=/dev/nvidia[0\-3]\fR).
doc/man/man5/gres.conf.5:The exception to this is MPS/Sharding. For either of these GRES, each GPU would be identified by device
doc/man/man5/gres.conf.5:entries that would correspond to that GPU. For MPS, typically 100 or some
doc/man/man5/gres.conf.5:simultaneously share that GPU.
doc/man/man5/gres.conf.5:If using a card with Multi-Instance GPU functionality, use \fBMultipleFiles\fR
doc/man/man5/gres.conf.5:\fBNOTE\fR: \fBFile\fR is required for all \fIgpu\fR typed GRES.
doc/man/man5/gres.conf.5:parameters (e.g. if you want to add or remove GPUs from a node's configuration).
doc/man/man5/gres.conf.5:gres (gpu) only allow one of the sharing gres to be used by the shared gres.
doc/man/man5/gres.conf.5:used to allow all sharing gres (gpu) on a node to be used for shared gres (mps).
doc/man/man5/gres.conf.5:\fBnvidia_gpu_env\fR
doc/man/man5/gres.conf.5:Set environment variable \fICUDA_VISIBLE_DEVICES\fR for all GPUs on the
doc/man/man5/gres.conf.5:\fBamd_gpu_env\fR
doc/man/man5/gres.conf.5:Set environment variable \fIROCR_VISIBLE_DEVICES\fR for all GPUs on the
doc/man/man5/gres.conf.5:\fBintel_gpu_env\fR
doc/man/man5/gres.conf.5:Set environment variable \fIZE_AFFINITY_MASK\fR for all GPUs on the
doc/man/man5/gres.conf.5:\fBopencl_env\fR
doc/man/man5/gres.conf.5:Set environment variable \fIGPU_DEVICE_ORDINAL\fR for all GPUs on the
doc/man/man5/gres.conf.5:\fBno_gpu_env\fR
doc/man/man5/gres.conf.5:Set no GPU\-specific environment variables. This is mutually exclusive to all
doc/man/man5/gres.conf.5:If no environment\-related flags are specified, then \fInvidia_gpu_env\fR,
doc/man/man5/gres.conf.5:\fIamd_gpu_env\fR, \fIintel_gpu_env\fR, and \fIopencl_env\fR will be
doc/man/man5/gres.conf.5:then \fIAutoDetect=nvml\fR or \fIAutoDetect=nvidia\fR will set
doc/man/man5/gres.conf.5:\fInvidia_gpu_env\fR, \fIAutoDetect=rsmi\fR will set \fIamd_gpu_env\fR,
doc/man/man5/gres.conf.5:and \fIAutoDetect=oneapi\fR will set \fIintel_gpu_env\fR.
doc/man/man5/gres.conf.5:Note that there is a known issue with the AMD ROCm runtime where
doc/man/man5/gres.conf.5:\fICUDA_VISIBLE_DEVICES\fR is processed. To avoid the issues caused by this, set
doc/man/man5/gres.conf.5:\fIFlags=amd_gpu_env\fR for AMD GPUs so only \fIROCR_VISIBLE_DEVICES\fR is set.
doc/man/man5/gres.conf.5:A typical use case would be to identify GPUs having NVLink connectivity.
doc/man/man5/gres.conf.5:Note that for GPUs, the minor number assigned by the OS and used in the device
doc/man/man5/gres.conf.5:file (i.e. the X in \fI/dev/nvidiaX\fR) is not necessarily the same as the
doc/man/man5/gres.conf.5:device number/index. The device number is created by sorting the GPUs by PCI bus
doc/man/man5/gres.conf.5:See \fIhttps://slurm.schedmd.com/gres.html#GPU_Management\fR
doc/man/man5/gres.conf.5:Graphics cards using Multi-Instance GPU (MIG) technology will present multiple
doc/man/man5/gres.conf.5:(e.g. MultipleFiles=/dev/nvidia[0-3]).
doc/man/man5/gres.conf.5:parameter, such as when adding or removing GPUs from a node's configuration.
doc/man/man5/gres.conf.5:When not using GPUs with MIG functionality, use \fBFile\fR instead.
doc/man/man5/gres.conf.5:\fBgpu\fR
doc/man/man5/gres.conf.5:CUDA Multi\-Process Service (MPS)
doc/man/man5/gres.conf.5:Shards of a gpu
doc/man/man5/gres.conf.5:For example, this might be used to identify a specific model of GPU, which users
doc/man/man5/gres.conf.5:# Define GPU devices with MPS support, with AutoDetect sanity checking
doc/man/man5/gres.conf.5:Name=gpu Type=gtx560 File=/dev/nvidia0 COREs=0,1
doc/man/man5/gres.conf.5:Name=gpu Type=tesla  File=/dev/nvidia1 COREs=2,3
doc/man/man5/gres.conf.5:Name=mps Count=100 File=/dev/nvidia0 COREs=0,1
doc/man/man5/gres.conf.5:Name=mps Count=100  File=/dev/nvidia1 COREs=2,3
doc/man/man5/gres.conf.5:# Overwrite system defaults and explicitly configure three GPUs
doc/man/man5/gres.conf.5:Name=gpu Type=tesla File=/dev/nvidia[0\-1] COREs=0,1
doc/man/man5/gres.conf.5:# Name=gpu Type=tesla  File=/dev/nvidia[2\-3] COREs=2,3
doc/man/man5/gres.conf.5:# NOTE: nvidia2 device is out of service
doc/man/man5/gres.conf.5:Name=gpu Type=tesla  File=/dev/nvidia3 COREs=2,3
doc/man/man5/gres.conf.5:# NodeName=tux[0\-15]  Name=gpu File=/dev/nvidia[0\-3]
doc/man/man5/gres.conf.5:# NOTE: tux3 nvidia1 device is out of service
doc/man/man5/gres.conf.5:NodeName=tux[0\-2]  Name=gpu File=/dev/nvidia[0\-3]
doc/man/man5/gres.conf.5:NodeName=tux3  Name=gpu File=/dev/nvidia[0,2\-3]
doc/man/man5/gres.conf.5:NodeName=tux[4\-15]  Name=gpu File=/dev/nvidia[0\-3]
doc/man/man5/gres.conf.5:# Use NVML to gather GPU configuration information
doc/man/man5/gres.conf.5:NodeName=tux3 AutoDetect=off Name=gpu File=/dev/nvidia[0\-3]
doc/man/man5/gres.conf.5:NodeName=tux[12\-15] Name=gpu File=/dev/nvidia[0\-3]
doc/man/man5/slurm.conf.5:If multiple GRES of different types are tracked (e.g. GPUs of different types),
doc/man/man5/slurm.conf.5:"AccountingStorageTRES=gres/gpu,gres/gpu:tesla,gres/gpu:volta"
doc/man/man5/slurm.conf.5:Then "gres/gpu:tesla" and "gres/gpu:volta" will track only jobs that explicitly
doc/man/man5/slurm.conf.5:request those two GPU types, while "gres/gpu" will track allocated GPUs of any
doc/man/man5/slurm.conf.5:type ("tesla", "volta" or any other GPU type).
doc/man/man5/slurm.conf.5:"AccountingStorageTRES=gres/gpu:tesla,gres/gpu:volta"
doc/man/man5/slurm.conf.5:Then "gres/gpu:tesla" and "gres/gpu:volta" will track jobs that explicitly
doc/man/man5/slurm.conf.5:request those GPU types.
doc/man/man5/slurm.conf.5:If a job requests GPUs, but does not explicitly specify the GPU type, then
doc/man/man5/slurm.conf.5:its resource allocation will be accounted for as either "gres/gpu:tesla" or
doc/man/man5/slurm.conf.5:"gres/gpu:volta", although the accounting may not match the actual GPU type
doc/man/man5/slurm.conf.5:allocated to the job and the GPUs allocated to the job could be heterogeneous.
doc/man/man5/slurm.conf.5:In an environment containing various GPU types, use of a job_submit plugin
doc/man/man5/slurm.conf.5:may be desired in order to force jobs to explicitly specify some GPU type.
doc/man/man5/slurm.conf.5:\fBNOTE\fR: Setting gres/gpu will also set gres/gpumem and gres/gpuutil.
doc/man/man5/slurm.conf.5:gres/gpumem and gres/gpuutil can be set individually when gres/gpu is not set.
doc/man/man5/slurm.conf.5:\fBacct_gather_energy/gpu\fR
doc/man/man5/slurm.conf.5:Energy consumption data is collected from the GPU management library (e.g. rsmi)
doc/man/man5/slurm.conf.5:for the corresponding type of GPU. Only available for rsmi at present.
doc/man/man5/slurm.conf.5:\fBDefCpuPerGPU\fR
doc/man/man5/slurm.conf.5:Default count of CPUs allocated per allocated GPU. This value is used only if
doc/man/man5/slurm.conf.5:the job didn't specify \-\-cpus\-per\-task and \-\-cpus\-per\-gpu.
doc/man/man5/slurm.conf.5:Also see \fBDefMemPerGPU\fR, \fBDefMemPerNode\fR and \fBMaxMemPerCPU\fR.
doc/man/man5/slurm.conf.5:\fBDefMemPerCPU\fR, \fBDefMemPerGPU\fR and \fBDefMemPerNode\fR are
doc/man/man5/slurm.conf.5:\fBDefMemPerGPU\fR
doc/man/man5/slurm.conf.5:Default real memory size available per allocated GPU in megabytes.
doc/man/man5/slurm.conf.5:\fBDefMemPerCPU\fR, \fBDefMemPerGPU\fR and \fBDefMemPerNode\fR are
doc/man/man5/slurm.conf.5:Also see \fBDefMemPerCPU\fR, \fBDefMemPerGPU\fR and \fBMaxMemPerCPU\fR.
doc/man/man5/slurm.conf.5:\fBDefMemPerCPU\fR, \fBDefMemPerGPU\fR and \fBDefMemPerNode\fR are
doc/man/man5/slurm.conf.5:\fIGresTypes=gpu,mps\fR).
doc/man/man5/slurm.conf.5:\fBGpuFreqDef\fR=[<\fItype\fR]=\fIvalue\fR>[,<\fItype\fR=\fIvalue\fR>]
doc/man/man5/slurm.conf.5:Default GPU frequency to use when running a job step if it
doc/man/man5/slurm.conf.5:has not been explicitly set using the \-\-gpu\-freq option.
doc/man/man5/slurm.conf.5:This option can be used to independently configure the GPU and its memory
doc/man/man5/slurm.conf.5:There is no default value. If unset, no attempt to change the GPU frequency
doc/man/man5/slurm.conf.5:is made if the \-\-gpu\-freq option has not been set.
doc/man/man5/slurm.conf.5:After the job is completed, the frequencies of all affected GPUs will be reset
doc/man/man5/slurm.conf.5:If \fItype\fR is not specified, the GPU frequency is implied.
doc/man/man5/slurm.conf.5:Examples of use include "GpuFreqDef=medium,memory=high and "GpuFreqDef=450".
doc/man/man5/slurm.conf.5:The JobAcctGather plugin collects memory, cpu, io, interconnect, energy and gpu
doc/man/man5/slurm.conf.5:\fBDisableGPUAcct\fR
doc/man/man5/slurm.conf.5:Do not do accounting of GPU usage and skip any gpu driver library call. This
doc/man/man5/slurm.conf.5:parameter can help to improve performance if the GPU driver response is slow.
doc/man/man5/slurm.conf.5:Also see \fBDefMemPerCPU\fR, \fBDefMemPerGPU\fR and \fBMaxMemPerNode\fR.
doc/man/man5/slurm.conf.5:PriorityWeightTRES=CPU=1000,Mem=2000,GRES/gpu=3000
doc/man/man5/slurm.conf.5:The resources (cores, memory, GPUs and all other trackable resources) within
doc/man/man5/slurm.conf.5:node. In cases where generic resources (such as GPUs) need to be tracked,
doc/man/man5/slurm.conf.5:Example: If there are 10 shards to a gpu and 12 shards are requested, instead of
doc/man/man5/slurm.conf.5:being denied the job will be allocated with 2 gpus. 1 using 10 shards and the
doc/man/man5/slurm.conf.5:On systems configured with \fBSwitchType=switch/nvidia_imex\fR, the following
doc/man/man5/slurm.conf.5:\fBswitch/nvidia_imex\fR
doc/man/man5/slurm.conf.5:For allocating unique channels within an NVIDIA IMEX domain.
doc/man/man5/slurm.conf.5:for a way to overcommit GPUs to multiple processes at the time you may be
doc/man/man5/slurm.conf.5:(e.g."Gres=gpu:tesla:1,gpu:kepler:1,bandwidth:lustre:no_consume:4G").
doc/man/man5/slurm.conf.5:\fBRestrictedCoresPerGPU\fR
doc/man/man5/slurm.conf.5:Number of cores per GPU restricted for only GPU use. If a job does not request a
doc/man/man5/slurm.conf.5:GPU it will not have access to these cores.
doc/man/man5/slurm.conf.5:\fBNOTE\fR: Configuring multiple GPU types on overlapping sockets can result in
doc/man/man5/slurm.conf.5:erroneous GPU type and restricted core pairings in allocations requesting gpus
doc/man/man5/slurm.conf.5:\fBDefCpuPerGPU\fR
doc/man/man5/slurm.conf.5:Default count of CPUs allocated per allocated GPU. This value is used only if
doc/man/man5/slurm.conf.5:the job didn't specify \-\-cpus\-per\-task and \-\-cpus\-per\-gpu.
doc/man/man5/slurm.conf.5:Also see \fBDefMemPerGPU\fR, \fBDefMemPerNode\fR and \fBMaxMemPerCPU\fR.
doc/man/man5/slurm.conf.5:\fBDefMemPerCPU\fR, \fBDefMemPerGPU\fR and \fBDefMemPerNode\fR are mutually
doc/man/man5/slurm.conf.5:\fBDefMemPerGPU\fR
doc/man/man5/slurm.conf.5:Default real memory size available per allocated GPU in megabytes.
doc/man/man5/slurm.conf.5:\fBDefMemPerCPU\fR, \fBDefMemPerGPU\fR and \fBDefMemPerNode\fR are mutually
doc/man/man5/slurm.conf.5:Also see \fBDefMemPerCPU\fR, \fBDefMemPerGPU\fR and \fBMaxMemPerCPU\fR.
doc/man/man5/slurm.conf.5:\fBDefMemPerCPU\fR, \fBDefMemPerGPU\fR and \fBDefMemPerNode\fR are mutually
doc/man/man5/slurm.conf.5:This can be especially useful to schedule GPUs. For example a node can be
doc/man/man5/slurm.conf.5:associated with two Slurm partitions (e.g. "cpu" and "gpu") and the
doc/man/man5/slurm.conf.5:ensuring that one or more CPUs would be available to jobs in the "gpu"
doc/man/man5/slurm.conf.5:partition. This can be especially useful to schedule GPUs.
doc/man/man5/slurm.conf.5:TRESBillingWeights="CPU=1.0,Mem=0.25G,GRES/gpu=2.0,license/licA=1.5", the
doc/man/man5/slurm.conf.5:\fBSLURM_JOB_GPUS\fR
doc/man/man5/slurm.conf.5:The GPU IDs of GPUs in the job allocation (if any).
doc/man/man5/acct_gather.conf.5:.SH acct_gather_energy/gpu
doc/man/man5/acct_gather.conf.5:AcctGatherEnergyType=acct_gather_energy/gpu
doc/man/man8/slurmrestd.8:      "gpu_spec": "",
doc/man/man8/slurmrestd.8:      "gres": "gpu:fake1:1(S:0),gpu:fake2:1(S:0)",
doc/man/man8/slurmrestd.8:      "gres_used": "gpu:fake1:0(IDX:N\/A),gpu:fake2:0(IDX:N\/A)",
doc/man/man8/slurmrestd.8:      "res_cores_per_gpu": 0,
doc/man/man8/slurmrestd.8:      "tres": "cpu=32,mem=127927M,billing=32,gres\/gpu=2",
doc/man/man8/slurmd.8:\-\-conf "Gres=gpu:2"
doc/man/man8/slurmd.8:NodeName=node1 CPUs=16 Boards=1 SocketsPerBoard=1 CoresPerSocket=8 ThreadsPerCore=2 RealMemory=31848 Gres=gpu:2
doc/man/man8/slurmd.8:\-\-conf "CPUs=16 RealMemory=30000 Gres=gpu:2"
doc/man/man8/slurmd.8:NodeName=node1 CPUs=16 RealMemory=30000 Gres=gpu:2"
doc/man/man1/scrun.1:\fBSCRUN_CPUS_PER_GPU\fR
doc/man/man1/scrun.1:See \fBSLURM_CPUS_PER_GPU\fR from \fBsrun\fR(1).
doc/man/man1/scrun.1:\fBSCRUN_GPU_BIND\fR
doc/man/man1/scrun.1:See \fBSLURM_GPU_BIND\fR from \fBsrun\fR(1).
doc/man/man1/scrun.1:\fBSCRUN_GPU_FREQ\fR
doc/man/man1/scrun.1:See \fBSLURM_GPU_FREQ\fR from \fBsrun\fR(1).
doc/man/man1/scrun.1:\fBSCRUN_GPUS\fR
doc/man/man1/scrun.1:See \fBSLURM_GPUS\fR from \fBsrun\fR(1).
doc/man/man1/scrun.1:\fBSCRUN_GPUS_PER_NODE\fR
doc/man/man1/scrun.1:See \fBSLURM_GPUS_PER_NODE\fR from \fBsrun\fR(1).
doc/man/man1/scrun.1:\fBSCRUN_GPUS_PER_SOCKET\fR
doc/man/man1/scrun.1:See \fBSLURM_GPUS_PER_SOCKET\fR from \fBsalloc\fR(1).
doc/man/man1/scrun.1:\fBSCRUN_GPUS_PER_TASK\fR
doc/man/man1/scrun.1:See \fBSLURM_GPUS_PER_TASK\fR from \fBsrun\fR(1).
doc/man/man1/scrun.1:\fBSCRUN_MEM_PER_GPU\fR
doc/man/man1/scrun.1:See \fBSLURM_MEM_PER_GPU\fR from \fBsrun\fR(1).
doc/man/man1/scrun.1:\fBSCRUN_NTASKS_PER_GPU\fR
doc/man/man1/scrun.1:See \fBSLURM_NTASKS_PER_GPU\fR from \fBsrun\fR(1).
doc/man/man1/scrun.1:\fBSLURM_CPUS_PER_GPU\fR
doc/man/man1/scrun.1:Number of CPUs requested per allocated GPU.
doc/man/man1/scrun.1:\fBSLURM_GPU_BIND\fR
doc/man/man1/scrun.1:Requested binding of tasks to GPU.
doc/man/man1/scrun.1:\fBSLURM_GPU_FREQ\fR
doc/man/man1/scrun.1:Requested GPU frequency.
doc/man/man1/scrun.1:\fBSLURM_GPUS\fR
doc/man/man1/scrun.1:Number of GPUs requested.
doc/man/man1/scrun.1:\fBSLURM_GPUS_PER_NODE\fR
doc/man/man1/scrun.1:Requested GPU count per allocated node.
doc/man/man1/scrun.1:\fBSLURM_GPUS_PER_SOCKET\fR
doc/man/man1/scrun.1:Requested GPU count per allocated socket.
doc/man/man1/scrun.1:\fBSLURM_GPUS_PER_TASK\fR
doc/man/man1/scrun.1:Requested GPU count per allocated task.
doc/man/man1/scrun.1:\fBSLURM_JOB_GPUS\fR
doc/man/man1/scrun.1:The global GPU IDs of the GPUs allocated to this job. The GPU IDs are not
doc/man/man1/scrun.1:\fBSLURM_MEM_PER_GPU\fR
doc/man/man1/scrun.1:Requested memory per allocated GPU.
doc/man/man1/scrun.1:\fBSLURM_NTASKS_PER_GPU\fR
doc/man/man1/scrun.1:Request that there are \fIntasks\fR tasks invoked for every GPU.
doc/man/man1/scrun.1:Number of GPU Shards available to the step on this node.
doc/man/man1/scrun.1:\fB\-\-gpus\-per\-task\fR is specified, it is also set in
doc/man/man1/sbatch.1:For example, \fB\-\-constraint="intel&gpu"\fR
doc/man/man1/sbatch.1:\fB\-\-cpus\-per\-gpu\fR=<\fIncpus\fR>
doc/man/man1/sbatch.1:Request that \fIncpus\fR processors be allocated per allocated GPU.
doc/man/man1/sbatch.1:sharing GRES (GPU) this option only allocates all sharing GRES and no underlying
doc/man/man1/sbatch.1:\fB\-\-gpu\-bind\fR=[verbose,]<\fItype\fR>
doc/man/man1/sbatch.1:Equivalent to \-\-tres\-bind=gres/gpu:[verbose,]<\fItype\fR>
doc/man/man1/sbatch.1:\fB\-\-gpu\-freq\fR=[<\fItype\fR]=\fIvalue\fR>[,<\fItype\fR=\fIvalue\fR>][,verbose]
doc/man/man1/sbatch.1:Request that GPUs allocated to the job are configured with specific frequency
doc/man/man1/sbatch.1:This option can be used to independently configure the GPU and its memory
doc/man/man1/sbatch.1:After the job is completed, the frequencies of all affected GPUs will be reset
doc/man/man1/sbatch.1:If \fItype\fR is not specified, the GPU frequency is implied.
doc/man/man1/sbatch.1:The \fIverbose\fR option causes current GPU frequency information to be logged.
doc/man/man1/sbatch.1:Examples of use include "\-\-gpu\-freq=medium,memory=high" and
doc/man/man1/sbatch.1:"\-\-gpu\-freq=450".
doc/man/man1/sbatch.1:\fB\-G\fR, \fB\-\-gpus\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/sbatch.1:Specify the total number of GPUs required for the job.
doc/man/man1/sbatch.1:An optional GPU type specification can be supplied.
doc/man/man1/sbatch.1:For example "\-\-gpus=volta:3".
doc/man/man1/sbatch.1:See also the \fB\-\-gpus\-per\-node\fR, \fB\-\-gpus\-per\-socket\fR and
doc/man/man1/sbatch.1:\fB\-\-gpus\-per\-task\fR options.
doc/man/man1/sbatch.1:\fBNOTE\fR: The allocation has to contain at least one GPU per node, or one of
doc/man/man1/sbatch.1:each GPU type per node if types are used. Use heterogeneous jobs if different
doc/man/man1/sbatch.1:nodes need different GPU types.
doc/man/man1/sbatch.1:\fB\-\-gpus\-per\-node\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/sbatch.1:Specify the number of GPUs required for the job on each node included in
doc/man/man1/sbatch.1:An optional GPU type specification can be supplied.
doc/man/man1/sbatch.1:For example "\-\-gpus\-per\-node=volta:3".
doc/man/man1/sbatch.1:"\-\-gpus\-per\-node=volta:3,kepler:1".
doc/man/man1/sbatch.1:See also the \fB\-\-gpus\fR, \fB\-\-gpus\-per\-socket\fR and
doc/man/man1/sbatch.1:\fB\-\-gpus\-per\-task\fR options.
doc/man/man1/sbatch.1:\fB\-\-gpus\-per\-socket\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/sbatch.1:Specify the number of GPUs required for the job on each socket included in
doc/man/man1/sbatch.1:An optional GPU type specification can be supplied.
doc/man/man1/sbatch.1:For example "\-\-gpus\-per\-socket=volta:3".
doc/man/man1/sbatch.1:"\-\-gpus\-per\-socket=volta:3,kepler:1".
doc/man/man1/sbatch.1:See also the \fB\-\-gpus\fR, \fB\-\-gpus\-per\-node\fR and
doc/man/man1/sbatch.1:\fB\-\-gpus\-per\-task\fR options.
doc/man/man1/sbatch.1:\fB\-\-gpus\-per\-task\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/sbatch.1:Specify the number of GPUs required for the job on each task to be spawned
doc/man/man1/sbatch.1:An optional GPU type specification can be supplied.
doc/man/man1/sbatch.1:For example "\-\-gpus\-per\-task=volta:1". Multiple options can be
doc/man/man1/sbatch.1:"\-\-gpus\-per\-task=volta:3,kepler:1". See also the \fB\-\-gpus\fR,
doc/man/man1/sbatch.1:\fB\-\-gpus\-per\-socket\fR and \fB\-\-gpus\-per\-node\fR options.
doc/man/man1/sbatch.1:This option requires an explicit task count, e.g. \-n, \-\-ntasks or "\-\-gpus=X
doc/man/man1/sbatch.1:\-\-gpus\-per\-task=Y" rather than an ambiguous range of nodes with \-N, \-\-nodes.
doc/man/man1/sbatch.1:This option will implicitly set \-\-tres\-bind=gres/gpu:per_task:<gpus_per_task>,
doc/man/man1/sbatch.1:but that can be overridden with an explicit \-\-tres\-bind=gres/gpu
doc/man/man1/sbatch.1:The \fIname\fR is the type of consumable resource (e.g. gpu).
doc/man/man1/sbatch.1:Examples of use include "\-\-gres=gpu:2", "\-\-gres=gpu:kepler:2", and
doc/man/man1/sbatch.1:For example a job requiring two GPUs and one CPU will be delayed until both
doc/man/man1/sbatch.1:GPUs on a single socket are available rather than using GPUs bound to separate
doc/man/man1/sbatch.1:Also see \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR.
doc/man/man1/sbatch.1:The \fB\-\-mem\fR, \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR
doc/man/man1/sbatch.1:\fB\-\-mem\-per\-gpu\fR are specified as command line arguments, then they will
doc/man/man1/sbatch.1:Also see \fB\-\-mem\fR and \fB\-\-mem\-per\-gpu\fR.
doc/man/man1/sbatch.1:The \fB\-\-mem\fR, \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR
doc/man/man1/sbatch.1:\fB\-\-mem\-per\-gpu\fR=<\fIsize\fR>[\fIunits\fR]
doc/man/man1/sbatch.1:Minimum memory required per allocated GPU.
doc/man/man1/sbatch.1:Default value is \fBDefMemPerGPU\fR and is available on both a global and
doc/man/man1/sbatch.1:The \fB\-\-mem\fR, \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR
doc/man/man1/sbatch.1:\fB--gpus\fR.
doc/man/man1/sbatch.1:\fB\-\-ntasks\-per\-gpu\fR=<\fIntasks\fR>
doc/man/man1/sbatch.1:Request that there are \fIntasks\fR tasks invoked for every GPU.
doc/man/man1/sbatch.1:addition, in which case a type\-less GPU specification will be automatically
doc/man/man1/sbatch.1:determined to satisfy \fB\-\-ntasks\-per\-gpu\fR, or 2) specify the GPUs wanted
doc/man/man1/sbatch.1:(e.g. via \fB\-\-gpus\fR or \fB\-\-gres\fR) without specifying \fB\-\-ntasks\fR,
doc/man/man1/sbatch.1:This option will implicitly set \fB\-\-tres\-bind=gres/gpu:single:<ntasks>\fR,
doc/man/man1/sbatch.1:but that can be overridden with an explicit \fB\-\-tres\-bind=gres/gpu\fR
doc/man/man1/sbatch.1:This option is not compatible with \fB\-\-gpus\-per\-task\fR,
doc/man/man1/sbatch.1:\fB\-\-gpus\-per\-socket\fR, or \fB\-\-ntasks\-per\-node\fR.
doc/man/man1/sbatch.1:(e.g. gres/gpu)
doc/man/man1/sbatch.1:Example: \-\-tres\-bind=gres/gpu:verbose,map:0,1,2,3+gres/nic:closest
doc/man/man1/sbatch.1:\-\-tres\-per\-task and \-\-gpus\-per\-task).
doc/man/man1/sbatch.1:as gres, license, etc. (e.g. gpu, gpu:a100).
doc/man/man1/sbatch.1:\-\-tres\-per\-task=gres/gpu=1
doc/man/man1/sbatch.1:\-\-tres\-per\-task=gres/gpu:a100=2
doc/man/man1/sbatch.1:\fBNOTE\fR: This option with gres/gpu or gres/shard will implicitly set
doc/man/man1/sbatch.1:\-\-tres\-bind=per_task:(gpu or shard)<tres_per_task>; this can be overridden
doc/man/man1/sbatch.1:\fBSBATCH_CPUS_PER_GPU\fR
doc/man/man1/sbatch.1:Same as \fB\-\-cpus\-per\-gpu\fR
doc/man/man1/sbatch.1:\fBSBATCH_GPU_BIND\fR
doc/man/man1/sbatch.1:Same as \fB\-\-gpu\-bind\fR
doc/man/man1/sbatch.1:\fBSBATCH_GPU_FREQ\fR
doc/man/man1/sbatch.1:Same as \fB\-\-gpu\-freq\fR
doc/man/man1/sbatch.1:\fBSBATCH_GPUS\fR
doc/man/man1/sbatch.1:Same as \fB\-G, \-\-gpus\fR
doc/man/man1/sbatch.1:\fBSBATCH_GPUS_PER_NODE\fR
doc/man/man1/sbatch.1:Same as \fB\-\-gpus\-per\-node\fR
doc/man/man1/sbatch.1:\fBSBATCH_GPUS_PER_TASK\fR
doc/man/man1/sbatch.1:Same as \fB\-\-gpus\-per\-task\fR
doc/man/man1/sbatch.1:\fBSBATCH_MEM_PER_GPU\fR
doc/man/man1/sbatch.1:Same as \fB\-\-mem\-per\-gpu\fR
doc/man/man1/sbatch.1:\fBSLURM_CPUS_PER_GPU\fR
doc/man/man1/sbatch.1:Number of CPUs requested per allocated GPU.
doc/man/man1/sbatch.1:Only set if the \fB\-\-cpus\-per\-gpu\fR option is specified.
doc/man/man1/sbatch.1:\fBSLURM_GPU_BIND\fR
doc/man/man1/sbatch.1:Requested binding of tasks to GPU.
doc/man/man1/sbatch.1:Only set if the \fB\-\-gpu\-bind\fR option is specified.
doc/man/man1/sbatch.1:\fBSLURM_GPU_FREQ\fR
doc/man/man1/sbatch.1:Requested GPU frequency.
doc/man/man1/sbatch.1:Only set if the \fB\-\-gpu\-freq\fR option is specified.
doc/man/man1/sbatch.1:\fBSLURM_GPUS\fR
doc/man/man1/sbatch.1:Number of GPUs requested.
doc/man/man1/sbatch.1:Only set if the \fB\-G, \-\-gpus\fR option is specified.
doc/man/man1/sbatch.1:\fBSLURM_GPUS_ON_NODE\fR
doc/man/man1/sbatch.1:Number of GPUs allocated to the batch step.
doc/man/man1/sbatch.1:\fBSLURM_GPUS_PER_NODE\fR
doc/man/man1/sbatch.1:Requested GPU count per allocated node.
doc/man/man1/sbatch.1:Only set if the \fB\-\-gpus\-per\-node\fR option is specified.
doc/man/man1/sbatch.1:\fBSLURM_GPUS_PER_SOCKET\fR
doc/man/man1/sbatch.1:Requested GPU count per allocated socket.
doc/man/man1/sbatch.1:Only set if the \fB\-\-gpus\-per\-socket\fR option is specified.
doc/man/man1/sbatch.1:\fBSLURM_JOB_GPUS\fR
doc/man/man1/sbatch.1:The global GPU IDs of the GPUs allocated to this job. The GPU IDs are not
doc/man/man1/sbatch.1:\fBSLURM_MEM_PER_GPU\fR
doc/man/man1/sbatch.1:Requested memory per allocated GPU.
doc/man/man1/sbatch.1:Only set if the \fB\-\-mem\-per\-gpu\fR option is specified.
doc/man/man1/sbatch.1:the \fB\-\-ntasks\-per\-node\fR or \fB\-\-ntasks\-per\-gpu\fR options are
doc/man/man1/sbatch.1:the \fB\-\-ntasks\-per\-node\fR or \fB\-\-ntasks\-per\-gpu\fR options are
doc/man/man1/sbatch.1:\fBSLURM_NTASKS_PER_GPU\fR
doc/man/man1/sbatch.1:Number of tasks requested per GPU.
doc/man/man1/sbatch.1:Only set if the \fB\-\-ntasks\-per\-gpu\fR option is specified.
doc/man/man1/sbatch.1:Number of GPU Shards available to the step on this node.
doc/man/man1/sbatch.1:\fB\-\-gpus\-per\-task\fR is specified, it is also set in
doc/man/man1/srun.1:For example, \fB\-\-constraint="intel&gpu"\fR
doc/man/man1/srun.1:\fB\-\-cpus\-per\-gpu\fR=<\fIncpus\fR>
doc/man/man1/srun.1:Request that \fIncpus\fR processors be allocated per allocated GPU.
doc/man/man1/srun.1:sharing GRES (GPU) this option only allocates all sharing GRES and no underlying
doc/man/man1/srun.1:\fB\-\-gpu\-bind\fR=[verbose,]<\fItype\fR>
doc/man/man1/srun.1:Equivalent to \-\-tres\-bind=gres/gpu:[verbose,]<\fItype\fR>
doc/man/man1/srun.1:\fB\-\-gpu\-freq\fR=[<\fItype\fR]=\fIvalue\fR>[,<\fItype\fR=\fIvalue\fR>][,verbose]
doc/man/man1/srun.1:Request that GPUs allocated to the job are configured with specific frequency
doc/man/man1/srun.1:This option can be used to independently configure the GPU and its memory
doc/man/man1/srun.1:After the job is completed, the frequencies of all affected GPUs will be reset
doc/man/man1/srun.1:If \fItype\fR is not specified, the GPU frequency is implied.
doc/man/man1/srun.1:The \fIverbose\fR option causes current GPU frequency information to be logged.
doc/man/man1/srun.1:Examples of use include "\-\-gpu\-freq=medium,memory=high" and
doc/man/man1/srun.1:"\-\-gpu\-freq=450".
doc/man/man1/srun.1:\fB\-G\fR, \fB\-\-gpus\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/srun.1:Specify the total number of GPUs required for the job.
doc/man/man1/srun.1:An optional GPU type specification can be supplied.
doc/man/man1/srun.1:See also the \fB\-\-gpus\-per\-node\fR, \fB\-\-gpus\-per\-socket\fR and
doc/man/man1/srun.1:\fB\-\-gpus\-per\-task\fR options.
doc/man/man1/srun.1:\fBNOTE\fR: The allocation has to contain at least one GPU per node, or one of
doc/man/man1/srun.1:each GPU type per node if types are used. Use heterogeneous jobs if different
doc/man/man1/srun.1:nodes need different GPU types.
doc/man/man1/srun.1:\fB\-\-gpus\-per\-node\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/srun.1:Specify the number of GPUs required for the job on each node included in
doc/man/man1/srun.1:An optional GPU type specification can be supplied.
doc/man/man1/srun.1:For example "\-\-gpus\-per\-node=volta:3".
doc/man/man1/srun.1:"\-\-gpus\-per\-node=volta:3,kepler:1".
doc/man/man1/srun.1:See also the \fB\-\-gpus\fR, \fB\-\-gpus\-per\-socket\fR and
doc/man/man1/srun.1:\fB\-\-gpus\-per\-task\fR options.
doc/man/man1/srun.1:\fB\-\-gpus\-per\-socket\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/srun.1:Specify the number of GPUs required for the job on each socket included in
doc/man/man1/srun.1:An optional GPU type specification can be supplied.
doc/man/man1/srun.1:For example "\-\-gpus\-per\-socket=volta:3".
doc/man/man1/srun.1:"\-\-gpus\-per\-socket=volta:3,kepler:1".
doc/man/man1/srun.1:See also the \fB\-\-gpus\fR, \fB\-\-gpus\-per\-node\fR and
doc/man/man1/srun.1:\fB\-\-gpus\-per\-task\fR options.
doc/man/man1/srun.1:\fB\-\-gpus\-per\-task\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/srun.1:Specify the number of GPUs required for the job on each task to be spawned
doc/man/man1/srun.1:An optional GPU type specification can be supplied.
doc/man/man1/srun.1:For example "\-\-gpus\-per\-task=volta:1". Multiple options can be
doc/man/man1/srun.1:"\-\-gpus\-per\-task=volta:3,kepler:1". See also the \fB\-\-gpus\fR,
doc/man/man1/srun.1:\fB\-\-gpus\-per\-socket\fR and \fB\-\-gpus\-per\-node\fR options.
doc/man/man1/srun.1:This option requires an explicit task count, e.g. \-n, \-\-ntasks or "\-\-gpus=X
doc/man/man1/srun.1:\-\-gpus\-per\-task=Y" rather than an ambiguous range of nodes with \-N, \-\-nodes.
doc/man/man1/srun.1:This option will implicitly set \-\-tres\-bind=gres/gpu:per_task:<gpus_per_task>,
doc/man/man1/srun.1:but that can be overridden with an explicit \-\-tres\-bind=gres/gpu
doc/man/man1/srun.1:The \fIname\fR is the type of consumable resource (e.g. gpu).
doc/man/man1/srun.1:Examples of use include "\-\-gres=gpu:2", "\-\-gres=gpu:kepler:2", and
doc/man/man1/srun.1:Allow tasks access to each GPU within the job's allocation that is on the same
doc/man/man1/srun.1:node as the task. This is useful when using \-\-gpu\-bind or
doc/man/man1/srun.1:\-\-tres\-bind=gres/gpu to bind GPUs to specific tasks, but GPU communication
doc/man/man1/srun.1:For example a job requiring two GPUs and one CPU will be delayed until both
doc/man/man1/srun.1:GPUs on a single socket are available rather than using GPUs bound to separate
doc/man/man1/srun.1:Also see \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR.
doc/man/man1/srun.1:The \fB\-\-mem\fR, \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR
doc/man/man1/srun.1:\fB\-\-mem\-per\-gpu\fR are specified as command line arguments, then they will
doc/man/man1/srun.1:Also see \fB\-\-mem\fR and \fB\-\-mem\-per\-gpu\fR.
doc/man/man1/srun.1:The \fB\-\-mem\fR, \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR
doc/man/man1/srun.1:\fB\-\-mem\-per\-gpu\fR=<\fIsize\fR>[\fIunits\fR]
doc/man/man1/srun.1:Minimum memory required per allocated GPU.
doc/man/man1/srun.1:Default value is \fBDefMemPerGPU\fR and is available on both a global and
doc/man/man1/srun.1:The \fB\-\-mem\fR, \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR
doc/man/man1/srun.1:\fB--gpus\fR.
doc/man/man1/srun.1:\fB\-\-ntasks\-per\-gpu\fR=<\fIntasks\fR>
doc/man/man1/srun.1:Request that there are \fIntasks\fR tasks invoked for every GPU.
doc/man/man1/srun.1:addition, in which case a type\-less GPU specification will be automatically
doc/man/man1/srun.1:determined to satisfy \fB\-\-ntasks\-per\-gpu\fR, or 2) specify the GPUs wanted
doc/man/man1/srun.1:(e.g. via \fB\-\-gpus\fR or \fB\-\-gres\fR) without specifying \fB\-\-ntasks\fR,
doc/man/man1/srun.1:This option will implicitly set \fB\-\-tres\-bind=gres/gpu:single:<ntasks>\fR,
doc/man/man1/srun.1:but that can be overridden with an explicit \fB\-\-tres\-bind=gres/gpu\fR
doc/man/man1/srun.1:This option is not compatible with \fB\-\-gpus\-per\-task\fR,
doc/man/man1/srun.1:\fB\-\-gpus\-per\-socket\fR, or \fB\-\-ntasks\-per\-node\fR.
doc/man/man1/srun.1:(e.g. gres/gpu)
doc/man/man1/srun.1:Example: \-\-tres\-bind=gres/gpu:verbose,map:0,1,2,3+gres/nic:closest
doc/man/man1/srun.1:\-\-tres\-per\-task and \-\-gpus\-per\-task).
doc/man/man1/srun.1:as gres, license, etc. (e.g. gpu, gpu:a100).
doc/man/man1/srun.1:\-\-tres\-per\-task=gres/gpu=1
doc/man/man1/srun.1:\-\-tres\-per\-task=gres/gpu:a100=2
doc/man/man1/srun.1:\fBNOTE\fR: This option with gres/gpu or gres/shard will implicitly set
doc/man/man1/srun.1:\-\-tres\-bind=per_task:(gpu or shard)<tres_per_task>; this can be overridden
doc/man/man1/srun.1:\fBSLURM_CPUS_PER_GPU\fR
doc/man/man1/srun.1:Same as \fB\-\-cpus\-per\-gpu\fR
doc/man/man1/srun.1:\fBSLURM_GPU_BIND\fR
doc/man/man1/srun.1:Same as \fB\-\-gpu\-bind\fR
doc/man/man1/srun.1:\fBSLURM_GPU_FREQ\fR
doc/man/man1/srun.1:Same as \fB\-\-gpu\-freq\fR
doc/man/man1/srun.1:\fBSLURM_GPUS\fR
doc/man/man1/srun.1:Same as \fB\-G, \-\-gpus\fR
doc/man/man1/srun.1:\fBSLURM_GPUS_PER_NODE\fR
doc/man/man1/srun.1:Same as \fB\-\-gpus\-per\-node\fR except within an existing allocation, in which
doc/man/man1/srun.1:case it will be ignored if \fB\-\-gpus\fR is specified.
doc/man/man1/srun.1:\fBSLURM_GPUS_PER_TASK\fR
doc/man/man1/srun.1:Same as \fB\-\-gpus\-per\-task\fR
doc/man/man1/srun.1:\fBSLURM_MEM_PER_GPU\fR
doc/man/man1/srun.1:Same as \fB\-\-mem\-per\-gpu\fR
doc/man/man1/srun.1:\fBSLURM_NTASKS_PER_GPU\fR
doc/man/man1/srun.1:Same as \fB\-\-ntasks\-per\-gpu\fR
doc/man/man1/srun.1:Same as \fB\-\-tres\-bind\fR If \fB\-\-gpu\-bind\fR is specified, it is also set
doc/man/man1/srun.1:\fB\-\-gpus\-per\-task\fR is specified, it is also set in
doc/man/man1/srun.1:\fBSLURM_GPUS_ON_NODE\fR
doc/man/man1/srun.1:Number of GPUs available to the step on this node.
doc/man/man1/srun.1:\fBSLURM_JOB_GPUS\fR
doc/man/man1/srun.1:The global GPU IDs of the GPUs allocated to this job. The GPU IDs are not
doc/man/man1/srun.1:Number of GPU Shards available to the step on this node.
doc/man/man1/srun.1:\fBSLURM_STEP_GPUS\fR
doc/man/man1/srun.1:The global GPU IDs of the GPUs allocated to this step (excluding batch and
doc/man/man1/srun.1:interactive steps). The GPU IDs are not relative to any device cgroup, even
doc/man/man1/scontrol.1:Examples of use include "Gres=gpus:2*cpu,disk=40G" and "Gres=help".
doc/man/man1/scontrol.1:Modification of GRES count associated with specific files (e.g. GPUs) is not
doc/man/man1/scontrol.1:For example if all 4 GPUs on a node are all associated with socket zero, then
doc/man/man1/scontrol.1:"Gres=gpu:4(S:0)". If associated with sockets 0 and 1 then "Gres=gpu:4(S:0\-1)".
doc/man/man1/scontrol.1:The information of which specific GPUs are associated with specific GPUs is not
doc/man/man1/scontrol.1:\fBDefCpuPerGPU\fR
doc/man/man1/scontrol.1:Default number of CPUs per allocated GPU.
doc/man/man1/scontrol.1:\fBDefMemPerGPU\fR
doc/man/man1/scontrol.1:Default memory limit (in megabytes) per allocated GPU.
doc/man/man1/scontrol.1:TRES=gres/gpu:a100=2
doc/man/man1/scontrol.1:TRESPerNode=gres/gpu:a100=2
doc/man/man1/scontrol.1:Above will allocate 2 gpu:a100 per node as specified in nodecnt.
doc/man/man1/salloc.1:For example, \fB\-\-constraint="intel&gpu"\fR
doc/man/man1/salloc.1:\fB\-\-cpus\-per\-gpu\fR=<\fIncpus\fR>
doc/man/man1/salloc.1:Request that \fIncpus\fR processors be allocated per allocated GPU.
doc/man/man1/salloc.1:sharing GRES (GPU) this option only allocates all sharing GRES and no underlying
doc/man/man1/salloc.1:\fB\-\-gpu\-bind\fR=[verbose,]<\fItype\fR>
doc/man/man1/salloc.1:Equivalent to \-\-tres\-bind=gres/gpu:[verbose,]<\fItype\fR>
doc/man/man1/salloc.1:\fB\-\-gpu\-freq\fR=[<\fItype\fR]=\fIvalue\fR>[,<\fItype\fR=\fIvalue\fR>][,verbose]
doc/man/man1/salloc.1:Request that GPUs allocated to the job are configured with specific frequency
doc/man/man1/salloc.1:This option can be used to independently configure the GPU and its memory
doc/man/man1/salloc.1:After the job is completed, the frequencies of all affected GPUs will be reset
doc/man/man1/salloc.1:If \fItype\fR is not specified, the GPU frequency is implied.
doc/man/man1/salloc.1:The \fIverbose\fR option causes current GPU frequency information to be logged.
doc/man/man1/salloc.1:Examples of use include "\-\-gpu\-freq=medium,memory=high" and
doc/man/man1/salloc.1:"\-\-gpu\-freq=450".
doc/man/man1/salloc.1:\fB\-G\fR, \fB\-\-gpus\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/salloc.1:Specify the total number of GPUs required for the job.
doc/man/man1/salloc.1:An optional GPU type specification can be supplied.
doc/man/man1/salloc.1:For example "\-\-gpus=volta:3".
doc/man/man1/salloc.1:See also the \fB\-\-gpus\-per\-node\fR, \fB\-\-gpus\-per\-socket\fR and
doc/man/man1/salloc.1:\fB\-\-gpus\-per\-task\fR options.
doc/man/man1/salloc.1:\fBNOTE\fR: The allocation has to contain at least one GPU per node, or one of
doc/man/man1/salloc.1:each GPU type per node if types are used. Use heterogeneous jobs if different
doc/man/man1/salloc.1:nodes need different GPU types.
doc/man/man1/salloc.1:\fB\-\-gpus\-per\-node\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/salloc.1:Specify the number of GPUs required for the job on each node included in
doc/man/man1/salloc.1:An optional GPU type specification can be supplied.
doc/man/man1/salloc.1:For example "\-\-gpus\-per\-node=volta:3".
doc/man/man1/salloc.1:"\-\-gpus\-per\-node=volta:3,kepler:1".
doc/man/man1/salloc.1:See also the \fB\-\-gpus\fR, \fB\-\-gpus\-per\-socket\fR and
doc/man/man1/salloc.1:\fB\-\-gpus\-per\-task\fR options.
doc/man/man1/salloc.1:\fB\-\-gpus\-per\-socket\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/salloc.1:Specify the number of GPUs required for the job on each socket included in
doc/man/man1/salloc.1:An optional GPU type specification can be supplied.
doc/man/man1/salloc.1:For example "\-\-gpus\-per\-socket=volta:3".
doc/man/man1/salloc.1:"\-\-gpus\-per\-socket=volta:3,kepler:1".
doc/man/man1/salloc.1:See also the \fB\-\-gpus\fR, \fB\-\-gpus\-per\-node\fR and
doc/man/man1/salloc.1:\fB\-\-gpus\-per\-task\fR options.
doc/man/man1/salloc.1:\fB\-\-gpus\-per\-task\fR=[\fItype\fR:]<\fInumber\fR>
doc/man/man1/salloc.1:Specify the number of GPUs required for the job on each task to be spawned
doc/man/man1/salloc.1:An optional GPU type specification can be supplied.
doc/man/man1/salloc.1:For example "\-\-gpus\-per\-task=volta:1". Multiple options can be
doc/man/man1/salloc.1:"\-\-gpus\-per\-task=volta:3,kepler:1". See also the \fB\-\-gpus\fR,
doc/man/man1/salloc.1:\fB\-\-gpus\-per\-socket\fR and \fB\-\-gpus\-per\-node\fR options.
doc/man/man1/salloc.1:This option requires an explicit task count, e.g. \-n, \-\-ntasks or "\-\-gpus=X
doc/man/man1/salloc.1:\-\-gpus\-per\-task=Y" rather than an ambiguous range of nodes with \-N, \-\-nodes.
doc/man/man1/salloc.1:This option will implicitly set \-\-tres\-bind=gres/gpu:per_task:<gpus_per_task>,
doc/man/man1/salloc.1:but that can be overridden with an explicit \-\-tres\-bind=gres/gpu
doc/man/man1/salloc.1:The \fIname\fR is the type of consumable resource (e.g. gpu).
doc/man/man1/salloc.1:Examples of use include "\-\-gres=gpu:2", "\-\-gres=gpu:kepler:2", and
doc/man/man1/salloc.1:For example a job requiring two GPUs and one CPU will be delayed until both
doc/man/man1/salloc.1:GPUs on a single socket are available rather than using GPUs bound to separate
doc/man/man1/salloc.1:Also see \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR.
doc/man/man1/salloc.1:The \fB\-\-mem\fR, \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR
doc/man/man1/salloc.1:\fB\-\-mem\-per\-gpu\fR are specified as command line arguments, then they will
doc/man/man1/salloc.1:Also see \fB\-\-mem\fR and \fB\-\-mem\-per\-gpu\fR.
doc/man/man1/salloc.1:The \fB\-\-mem\fR, \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR
doc/man/man1/salloc.1:\fB\-\-mem\-per\-gpu\fR=<\fIsize\fR>[\fIunits\fR]
doc/man/man1/salloc.1:Minimum memory required per allocated GPU.
doc/man/man1/salloc.1:Default value is \fBDefMemPerGPU\fR and is available on both a global and
doc/man/man1/salloc.1:The \fB\-\-mem\fR, \fB\-\-mem\-per\-cpu\fR and \fB\-\-mem\-per\-gpu\fR
doc/man/man1/salloc.1:\fB--gpus\fR.
doc/man/man1/salloc.1:\fB\-\-ntasks\-per\-gpu\fR=<\fIntasks\fR>
doc/man/man1/salloc.1:Request that there are \fIntasks\fR tasks invoked for every GPU.
doc/man/man1/salloc.1:addition, in which case a type\-less GPU specification will be automatically
doc/man/man1/salloc.1:determined to satisfy \fB\-\-ntasks\-per\-gpu\fR, or 2) specify the GPUs wanted
doc/man/man1/salloc.1:(e.g. via \fB\-\-gpus\fR or \fB\-\-gres\fR) without specifying \fB\-\-ntasks\fR,
doc/man/man1/salloc.1:This option will implicitly set \fB\-\-tres\-bind=gres/gpu:single:<ntasks>\fR,
doc/man/man1/salloc.1:but that can be overridden with an explicit \fB\-\-tres\-bind=gres/gpu\fR
doc/man/man1/salloc.1:This option is not compatible with \fB\-\-gpus\-per\-task\fR,
doc/man/man1/salloc.1:\fB\-\-gpus\-per\-socket\fR, or \fB\-\-ntasks\-per\-node\fR.
doc/man/man1/salloc.1:(e.g. gres/gpu)
doc/man/man1/salloc.1:Example: \-\-tres\-bind=gres/gpu:verbose,map:0,1,2,3+gres/nic:closest
doc/man/man1/salloc.1:\-\-tres\-per\-task and \-\-gpus\-per\-task).
doc/man/man1/salloc.1:as gres, license, etc. (e.g. gpu, gpu:a100).
doc/man/man1/salloc.1:\-\-tres\-per\-task=gres/gpu=1
doc/man/man1/salloc.1:\-\-tres\-per\-task=gres/gpu:a100=2
doc/man/man1/salloc.1:\fBNOTE\fR: This option with gres/gpu or gres/shard will implicitly set
doc/man/man1/salloc.1:\-\-tres\-bind=per_task:(gpu or shard)<tres_per_task>; this can be overridden
doc/man/man1/salloc.1:\fBSALLOC_CPUS_PER_GPU\fR
doc/man/man1/salloc.1:Same as \fB\-\-cpus\-per\-gpu\fR
doc/man/man1/salloc.1:\fBSALLOC_GPU_BIND\fR
doc/man/man1/salloc.1:Same as \fB\-\-gpu\-bind\fR
doc/man/man1/salloc.1:\fBSALLOC_GPU_FREQ\fR
doc/man/man1/salloc.1:Same as \fB\-\-gpu\-freq\fR
doc/man/man1/salloc.1:\fBSALLOC_GPUS\fR
doc/man/man1/salloc.1:Same as \fB\-G, \-\-gpus\fR
doc/man/man1/salloc.1:\fBSALLOC_GPUS_PER_NODE\fR
doc/man/man1/salloc.1:Same as \fB\-\-gpus\-per\-node\fR
doc/man/man1/salloc.1:\fBSALLOC_GPUS_PER_TASK\fR
doc/man/man1/salloc.1:Same as \fB\-\-gpus\-per\-task\fR
doc/man/man1/salloc.1:\fBSALLOC_MEM_PER_GPU\fR
doc/man/man1/salloc.1:Same as \fB\-\-mem\-per\-gpu\fR
doc/man/man1/salloc.1:\fBSLURM_CPUS_PER_GPU\fR
doc/man/man1/salloc.1:Number of CPUs requested per allocated GPU.
doc/man/man1/salloc.1:Only set if the \fB\-\-cpus\-per\-gpu\fR option is specified.
doc/man/man1/salloc.1:\fBSLURM_GPU_BIND\fR
doc/man/man1/salloc.1:Requested binding of tasks to GPU.
doc/man/man1/salloc.1:Only set if the \fB\-\-gpu\-bind\fR option is specified.
doc/man/man1/salloc.1:\fBSLURM_GPU_FREQ\fR
doc/man/man1/salloc.1:Requested GPU frequency.
doc/man/man1/salloc.1:Only set if the \fB\-\-gpu\-freq\fR option is specified.
doc/man/man1/salloc.1:\fBSLURM_GPUS\fR
doc/man/man1/salloc.1:Number of GPUs requested.
doc/man/man1/salloc.1:Only set if the \fB\-G, \-\-gpus\fR option is specified.
doc/man/man1/salloc.1:\fBSLURM_GPUS_PER_NODE\fR
doc/man/man1/salloc.1:Requested GPU count per allocated node.
doc/man/man1/salloc.1:Only set if the \fB\-\-gpus\-per\-node\fR option is specified.
doc/man/man1/salloc.1:\fBSLURM_GPUS_PER_SOCKET\fR
doc/man/man1/salloc.1:Requested GPU count per allocated socket.
doc/man/man1/salloc.1:Only set if the \fB\-\-gpus\-per\-socket\fR option is specified.
doc/man/man1/salloc.1:\fBSLURM_JOB_GPUS\fR
doc/man/man1/salloc.1:The global GPU IDs of the GPUs allocated to this job. The GPU IDs are not
doc/man/man1/salloc.1:\fBSLURM_MEM_PER_GPU\fR
doc/man/man1/salloc.1:Requested memory per allocated GPU.
doc/man/man1/salloc.1:Only set if the \fB\-\-mem\-per\-gpu\fR option is specified.
doc/man/man1/salloc.1:the \fB\-\-ntasks\-per\-node\fR or \fB\-\-ntasks\-per\-gpu\fR options are
doc/man/man1/salloc.1:the \fB\-\-ntasks\-per\-node\fR or \fB\-\-ntasks\-per\-gpu\fR options are
doc/man/man1/salloc.1:\fBSLURM_NTASKS_PER_GPU\fR
doc/man/man1/salloc.1:Set to value of the \fB\-\-ntasks\-per\-gpu\fR option, if specified.
doc/man/man1/salloc.1:Number of GPU Shards available to the step on this node.
doc/man/man1/salloc.1:\fB\-\-gpus\-per\-task\fR is specified, it is also set in
doc/man/man1/sacctmgr.1:type then name is the denomination of the GRES itself e.g. GPU.
doc/man/man1/sacctmgr.1:      gres         gpu:tesla     1001
etc/job_submit.lua.example:	-- Change partition to GPU if job requested any GPU
etc/job_submit.lua.example:	-- --gres=gpu: -> tres_per_node
etc/job_submit.lua.example:	-- --gpus-per-task -> tres_per_task
etc/job_submit.lua.example:	-- --gpus-per-socket -> tres_per_socket
etc/job_submit.lua.example:	-- --gpus -> tres_per_job
etc/job_submit.lua.example:	if _find_in_str(job_desc['tres_per_node'], "gpu") or
etc/job_submit.lua.example:	   _find_in_str(job_desc['tres_per_task'], "gpu") or
etc/job_submit.lua.example:	   _find_in_str(job_desc['tres_per_socket'], "gpu") or
etc/job_submit.lua.example:	   _find_in_str(job_desc['tres_per_job'], "gpu") then
etc/job_submit.lua.example:		job_desc.partition = 'gpu'
etc/prolog.example:# Determine which GPU the MPS server is running on
etc/prolog.example:# If job requires MPS, determine if it is running now on wrong (old) GPU assignment
etc/prolog.example:if [ -n "${CUDA_VISIBLE_DEVICES}" ] &&
etc/prolog.example:   [ -n "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}" ] &&
etc/prolog.example:   [[ ${CUDA_VISIBLE_DEVICES} != ${MPS_DEV_ID} ]]; then
etc/prolog.example:# If job requires full GPU(s) then kill the MPS server if it is still running
etc/prolog.example:# on any of the GPUs allocated to this job.
etc/prolog.example:# This string compare assumes there are not more than 10 GPUs per node.
etc/prolog.example:elif [ -n "${CUDA_VISIBLE_DEVICES}" ] &&
etc/prolog.example:     [ -z "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}" ] &&
etc/prolog.example:     [[ ${CUDA_VISIBLE_DEVICES} == *${MPS_DEV_ID}* ]]; then
etc/prolog.example:	ps aux | grep nvidia-cuda-mps-control | grep -v grep > /dev/null
etc/prolog.example:		# Reset GPU mode to default
etc/prolog.example:		${MPS_CMD_DIR}nvidia-smi -c ${CUDA_VISIBLE_DEVICES}
etc/prolog.example:		echo quit | ${MPS_CMD_DIR}nvidia-cuda-mps-control
etc/prolog.example:		ps aux | grep nvidia-cuda-mps | grep -v grep > /dev/null
etc/prolog.example:		# Check GPU sanity, simple check
etc/prolog.example:		${MPS_CMD_DIR}nvidia-smi > /dev/null
etc/prolog.example:			logger "`hostname` Slurm Prolog: GPU not operational! Downing node"
etc/prolog.example:			${SLURM_CMD_DIR}scontrol update nodename=${SLURMD_NODENAME} State=DOWN Reason="GPU not operational"
etc/prolog.example:if [ -n "${CUDA_VISIBLE_DEVICES}" ] &&
etc/prolog.example:   [ -n "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}" ]; then
etc/prolog.example:	echo ${CUDA_VISIBLE_DEVICES} >${MPS_DEV_ID_FILE}
etc/prolog.example:	unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
etc/prolog.example:	export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps_${CUDA_VISIBLE_DEVICES}
etc/prolog.example:	export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log_${CUDA_VISIBLE_DEVICES}
etc/prolog.example:	${MPS_CMD_DIR}nvidia-cuda-mps-control -d && echo "MPS control daemon started"
etc/prolog.example:	${MPS_CMD_DIR}nvidia-cuda-mps-control start_server -uid $SLURM_JOB_UID && echo "MPS server started for $SLURM_JOB_UID"
configure:  --with-nvml=PATH        Specify path to CUDA installation
configure:    nvcc*) # Cuda Compiler Driver 2.2
configure:	nvcc*)	# Cuda Compiler Driver 2.2
configure:      { ac_cv_lib_nvidia_ml_nvmlInit=; unset ac_cv_lib_nvidia_ml_nvmlInit;}
configure:      { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking for nvmlInit in -lnvidia-ml" >&5
configure:printf %s "checking for nvmlInit in -lnvidia-ml... " >&6; }
configure:if test ${ac_cv_lib_nvidia_ml_nvmlInit+y}
configure:LIBS="-lnvidia-ml  $LIBS"
configure:  ac_cv_lib_nvidia_ml_nvmlInit=yes
configure:  ac_cv_lib_nvidia_ml_nvmlInit=no
configure:{ printf "%s\n" "$as_me:${as_lineno-$LINENO}: result: $ac_cv_lib_nvidia_ml_nvmlInit" >&5
configure:printf "%s\n" "$ac_cv_lib_nvidia_ml_nvmlInit" >&6; }
configure:if test "x$ac_cv_lib_nvidia_ml_nvmlInit" = xyes
configure:          # Check indirectly that CUDA 11.1+ was installed to see if we
configure:	  # gpuInstanceSliceCount in the nvmlDeviceAttributes_t struct.
configure:		     attributes.gpuInstanceSliceCount = 0;
configure:  _x_ac_nvml_dirs="/usr/local/cuda /usr/cuda"
configure:          nvml_libs="-lnvidia-ml"
configure:          LDFLAGS="-L$d/$bit -lnvidia-ml"
configure:	{ printf "%s\n" "$as_me:${as_lineno-$LINENO}: WARNING: NVML was found, but can not support MIG. For MIG support both nvml.h and libnvidia-ml must be 11.1+. Please make sure they are both the same version as well." >&5
configure:printf "%s\n" "$as_me: WARNING: NVML was found, but can not support MIG. For MIG support both nvml.h and libnvidia-ml must be 11.1+. Please make sure they are both the same version as well." >&2;}
configure:        { printf "%s\n" "$as_me:${as_lineno-$LINENO}: WARNING: unable to locate libnvidia-ml.so and/or nvml.h" >&5
configure:printf "%s\n" "$as_me: WARNING: unable to locate libnvidia-ml.so and/or nvml.h" >&2;}
configure:        as_fn_error $? "unable to locate libnvidia-ml.so and/or nvml.h" "$LINENO" 5
configure:  # /opt/rocm is the current default location.
configure:  # /opt/rocm/rocm_smi was the default location for before to 5.2.0
configure:  _x_ac_rsmi_dirs="/opt/rocm /opt/rocm/rocm_smi"
configure:    { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking whether RSMI/ROCm in installed in this system" >&5
configure:printf %s "checking whether RSMI/ROCm in installed in this system... " >&6; }
configure:      { ac_cv_header_rocm_smi_h=; unset ac_cv_header_rocm_smi_h;}
configure:      { ac_cv_lib_rocm_smi64_rsmi_init=; unset ac_cv_lib_rocm_smi64_rsmi_init;}
configure:      { ac_cv_lib_rocm_smi64_dev_drm_render_minor_get=; unset ac_cv_lib_rocm_smi64_dev_drm_render_minor_get;}
configure:      ac_fn_c_check_header_compile "$LINENO" "rocm_smi/rocm_smi.h" "ac_cv_header_rocm_smi_rocm_smi_h" "$ac_includes_default"
configure:if test "x$ac_cv_header_rocm_smi_rocm_smi_h" = xyes
configure:      { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking for rsmi_init in -lrocm_smi64" >&5
configure:printf %s "checking for rsmi_init in -lrocm_smi64... " >&6; }
configure:if test ${ac_cv_lib_rocm_smi64_rsmi_init+y}
configure:LIBS="-lrocm_smi64  $LIBS"
configure:  ac_cv_lib_rocm_smi64_rsmi_init=yes
configure:  ac_cv_lib_rocm_smi64_rsmi_init=no
configure:{ printf "%s\n" "$as_me:${as_lineno-$LINENO}: result: $ac_cv_lib_rocm_smi64_rsmi_init" >&5
configure:printf "%s\n" "$ac_cv_lib_rocm_smi64_rsmi_init" >&6; }
configure:if test "x$ac_cv_lib_rocm_smi64_rsmi_init" = xyes
configure:      { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking for rsmi_dev_drm_render_minor_get in -lrocm_smi64" >&5
configure:printf %s "checking for rsmi_dev_drm_render_minor_get in -lrocm_smi64... " >&6; }
configure:if test ${ac_cv_lib_rocm_smi64_rsmi_dev_drm_render_minor_get+y}
configure:LIBS="-lrocm_smi64  $LIBS"
configure:  ac_cv_lib_rocm_smi64_rsmi_dev_drm_render_minor_get=yes
configure:  ac_cv_lib_rocm_smi64_rsmi_dev_drm_render_minor_get=no
configure:{ printf "%s\n" "$as_me:${as_lineno-$LINENO}: result: $ac_cv_lib_rocm_smi64_rsmi_dev_drm_render_minor_get" >&5
configure:printf "%s\n" "$ac_cv_lib_rocm_smi64_rsmi_dev_drm_render_minor_get" >&6; }
configure:if test "x$ac_cv_lib_rocm_smi64_rsmi_dev_drm_render_minor_get" = xyes
configure:          { printf "%s\n" "$as_me:${as_lineno-$LINENO}: WARNING: upgrade to newer version of ROCm/rsmi" >&5
configure:printf "%s\n" "$as_me: WARNING: upgrade to newer version of ROCm/rsmi" >&2;}
configure:          as_fn_error $? "upgrade to newer version of ROCm/rsmi" "$LINENO" 5
configure:        { printf "%s\n" "$as_me:${as_lineno-$LINENO}: WARNING: unable to locate librocm_smi64.so and/or rocm_smi.h" >&5
configure:printf "%s\n" "$as_me: WARNING: unable to locate librocm_smi64.so and/or rocm_smi.h" >&2;}
configure:        as_fn_error $? "unable to locate librocm_smi64.so and/or rocm_smi.h" "$LINENO" 5
configure:ac_config_files="$ac_config_files Makefile auxdir/Makefile contribs/Makefile contribs/lua/Makefile contribs/nss_slurm/Makefile contribs/openlava/Makefile contribs/pam/Makefile contribs/pam_slurm_adopt/Makefile contribs/perlapi/Makefile contribs/perlapi/libslurm/Makefile contribs/perlapi/libslurm/perl/Makefile.PL contribs/perlapi/libslurmdb/Makefile contribs/perlapi/libslurmdb/perl/Makefile.PL contribs/pmi/Makefile contribs/pmi2/Makefile contribs/seff/Makefile contribs/sgather/Makefile contribs/sjobexit/Makefile contribs/slurm_completion_help/Makefile contribs/torque/Makefile doc/Makefile doc/html/Makefile doc/html/configurator.easy.html doc/html/configurator.html doc/man/Makefile doc/man/man1/Makefile doc/man/man5/Makefile doc/man/man8/Makefile etc/Makefile src/Makefile src/api/Makefile src/bcast/Makefile src/common/Makefile src/conmgr/Makefile src/curl/Makefile src/database/Makefile src/interfaces/Makefile src/lua/Makefile src/plugins/Makefile src/plugins/accounting_storage/Makefile src/plugins/accounting_storage/common/Makefile src/plugins/accounting_storage/ctld_relay/Makefile src/plugins/accounting_storage/mysql/Makefile src/plugins/accounting_storage/slurmdbd/Makefile src/plugins/acct_gather_energy/Makefile src/plugins/acct_gather_energy/gpu/Makefile src/plugins/acct_gather_energy/ibmaem/Makefile src/plugins/acct_gather_energy/ipmi/Makefile src/plugins/acct_gather_energy/pm_counters/Makefile src/plugins/acct_gather_energy/rapl/Makefile src/plugins/acct_gather_energy/xcc/Makefile src/plugins/acct_gather_filesystem/Makefile src/plugins/acct_gather_filesystem/lustre/Makefile src/plugins/acct_gather_interconnect/Makefile src/plugins/acct_gather_interconnect/ofed/Makefile src/plugins/acct_gather_interconnect/sysfs/Makefile src/plugins/acct_gather_profile/Makefile src/plugins/acct_gather_profile/hdf5/Makefile src/plugins/acct_gather_profile/hdf5/sh5util/Makefile src/plugins/acct_gather_profile/influxdb/Makefile src/plugins/auth/Makefile src/plugins/auth/jwt/Makefile src/plugins/auth/munge/Makefile src/plugins/auth/none/Makefile src/plugins/auth/slurm/Makefile src/plugins/burst_buffer/Makefile src/plugins/burst_buffer/common/Makefile src/plugins/burst_buffer/datawarp/Makefile src/plugins/burst_buffer/lua/Makefile src/plugins/certmgr/Makefile src/plugins/certmgr/script/Makefile src/plugins/cgroup/Makefile src/plugins/cgroup/common/Makefile src/plugins/cgroup/v1/Makefile src/plugins/cgroup/v2/Makefile src/plugins/cli_filter/Makefile src/plugins/cli_filter/common/Makefile src/plugins/cli_filter/lua/Makefile src/plugins/cli_filter/syslog/Makefile src/plugins/cli_filter/user_defaults/Makefile src/plugins/cred/Makefile src/plugins/cred/common/Makefile src/plugins/cred/munge/Makefile src/plugins/cred/none/Makefile src/plugins/data_parser/Makefile src/plugins/data_parser/v0.0.40/Makefile src/plugins/data_parser/v0.0.41/Makefile src/plugins/data_parser/v0.0.42/Makefile src/plugins/gpu/Makefile src/plugins/gpu/common/Makefile src/plugins/gpu/generic/Makefile src/plugins/gpu/nrt/Makefile src/plugins/gpu/nvidia/Makefile src/plugins/gpu/nvml/Makefile src/plugins/gpu/oneapi/Makefile src/plugins/gpu/rsmi/Makefile src/plugins/gres/Makefile src/plugins/gres/common/Makefile src/plugins/gres/gpu/Makefile src/plugins/gres/mps/Makefile src/plugins/gres/nic/Makefile src/plugins/gres/shard/Makefile src/plugins/hash/Makefile src/plugins/hash/common_xkcp/Makefile src/plugins/hash/k12/Makefile src/plugins/hash/sha3/Makefile src/plugins/job_container/Makefile src/plugins/job_container/tmpfs/Makefile src/plugins/job_submit/Makefile src/plugins/job_submit/all_partitions/Makefile src/plugins/job_submit/defaults/Makefile src/plugins/job_submit/logging/Makefile src/plugins/job_submit/lua/Makefile src/plugins/job_submit/partition/Makefile src/plugins/job_submit/pbs/Makefile src/plugins/job_submit/require_timelimit/Makefile src/plugins/job_submit/throttle/Makefile src/plugins/jobacct_gather/Makefile src/plugins/jobacct_gather/cgroup/Makefile src/plugins/jobacct_gather/common/Makefile src/plugins/jobacct_gather/linux/Makefile src/plugins/jobcomp/Makefile src/plugins/jobcomp/common/Makefile src/plugins/jobcomp/elasticsearch/Makefile src/plugins/jobcomp/filetxt/Makefile src/plugins/jobcomp/kafka/Makefile src/plugins/jobcomp/lua/Makefile src/plugins/jobcomp/mysql/Makefile src/plugins/jobcomp/script/Makefile src/plugins/mcs/Makefile src/plugins/mcs/account/Makefile src/plugins/mcs/group/Makefile src/plugins/mcs/label/Makefile src/plugins/mcs/user/Makefile src/plugins/mpi/Makefile src/plugins/mpi/cray_shasta/Makefile src/plugins/mpi/pmi2/Makefile src/plugins/mpi/pmix/Makefile src/plugins/node_features/Makefile src/plugins/node_features/helpers/Makefile src/plugins/node_features/knl_generic/Makefile src/plugins/preempt/Makefile src/plugins/preempt/partition_prio/Makefile src/plugins/preempt/qos/Makefile src/plugins/prep/Makefile src/plugins/prep/script/Makefile src/plugins/priority/Makefile src/plugins/priority/basic/Makefile src/plugins/priority/multifactor/Makefile src/plugins/proctrack/Makefile src/plugins/proctrack/cgroup/Makefile src/plugins/proctrack/linuxproc/Makefile src/plugins/proctrack/pgid/Makefile src/plugins/sched/Makefile src/plugins/sched/backfill/Makefile src/plugins/sched/builtin/Makefile src/plugins/select/Makefile src/plugins/select/cons_tres/Makefile src/plugins/select/linear/Makefile src/plugins/serializer/Makefile src/plugins/serializer/json/Makefile src/plugins/serializer/url-encoded/Makefile src/plugins/serializer/yaml/Makefile src/plugins/site_factor/Makefile src/plugins/site_factor/example/Makefile src/plugins/switch/Makefile src/plugins/switch/hpe_slingshot/Makefile src/plugins/switch/nvidia_imex/Makefile src/plugins/task/Makefile src/plugins/task/affinity/Makefile src/plugins/task/cgroup/Makefile src/plugins/tls/Makefile src/plugins/tls/none/Makefile src/plugins/tls/s2n/Makefile src/plugins/topology/Makefile src/plugins/topology/3d_torus/Makefile src/plugins/topology/block/Makefile src/plugins/topology/common/Makefile src/plugins/topology/default/Makefile src/plugins/topology/tree/Makefile src/sacct/Makefile src/sackd/Makefile src/sacctmgr/Makefile src/salloc/Makefile src/sattach/Makefile src/scrun/Makefile src/sbatch/Makefile src/sbcast/Makefile src/scancel/Makefile src/scontrol/Makefile src/scrontab/Makefile src/sdiag/Makefile src/sinfo/Makefile src/slurmctld/Makefile src/slurmd/Makefile src/slurmd/common/Makefile src/slurmd/slurmd/Makefile src/slurmd/slurmstepd/Makefile src/slurmdbd/Makefile src/slurmrestd/Makefile src/slurmrestd/plugins/Makefile src/slurmrestd/plugins/auth/Makefile src/slurmrestd/plugins/auth/jwt/Makefile src/slurmrestd/plugins/auth/local/Makefile src/slurmrestd/plugins/openapi/Makefile src/slurmrestd/plugins/openapi/slurmctld/Makefile src/slurmrestd/plugins/openapi/slurmdbd/Makefile src/sprio/Makefile src/squeue/Makefile src/sreport/Makefile src/srun/Makefile src/sshare/Makefile src/sstat/Makefile src/stepmgr/Makefile src/strigger/Makefile src/sview/Makefile testsuite/Makefile testsuite/testsuite.conf.sample testsuite/expect/Makefile testsuite/slurm_unit/Makefile testsuite/slurm_unit/common/Makefile testsuite/slurm_unit/common/bitstring/Makefile testsuite/slurm_unit/common/hostlist/Makefile testsuite/slurm_unit/common/slurm_protocol_defs/Makefile testsuite/slurm_unit/common/slurm_protocol_pack/Makefile testsuite/slurm_unit/common/slurmdb_defs/Makefile testsuite/slurm_unit/common/slurmdb_pack/Makefile"
configure:    "src/plugins/acct_gather_energy/gpu/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/acct_gather_energy/gpu/Makefile" ;;
configure:    "src/plugins/gpu/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/gpu/Makefile" ;;
configure:    "src/plugins/gpu/common/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/gpu/common/Makefile" ;;
configure:    "src/plugins/gpu/generic/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/gpu/generic/Makefile" ;;
configure:    "src/plugins/gpu/nrt/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/gpu/nrt/Makefile" ;;
configure:    "src/plugins/gpu/nvidia/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/gpu/nvidia/Makefile" ;;
configure:    "src/plugins/gpu/nvml/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/gpu/nvml/Makefile" ;;
configure:    "src/plugins/gpu/oneapi/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/gpu/oneapi/Makefile" ;;
configure:    "src/plugins/gpu/rsmi/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/gpu/rsmi/Makefile" ;;
configure:    "src/plugins/gres/gpu/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/gres/gpu/Makefile" ;;
configure:    "src/plugins/switch/nvidia_imex/Makefile") CONFIG_FILES="$CONFIG_FILES src/plugins/switch/nvidia_imex/Makefile" ;;
NEWS: -- In existing allocations SLURM_GPUS_PER_NODE environment variable will be
NEWS:    ignored by srun if --gpus is specified.
NEWS: -- Add autodetected gpus to the output of slurmd -C
NEWS: -- Fix memory leak when requesting typed gres and --[cpus|mem]-per-gpu.
NEWS: -- slurmctld - Fix memory leak when using RestrictedCoresPerGPU.
NEWS:    subset of all GPUs on a node and the --gres-flags=allow-task-sharing option
NEWS: -- switch/nvidia_imex - Fix security issue managing IMEX channel access.
NEWS: -- switch/nvidia_imex - Allow for compatibility with job_container/tmpfs.
NEWS: -- Prevent slurmd from crashing if acct_gather_energy/gpu is configured but
NEWS: -- gpu/nvml - Fix gpuutil/gpumem only tracking last GPU in step. Now,
NEWS:    gpuutil/gpumem will record sums of all GPUS in the step.
NEWS: -- switch/nvidia_imex - Move setup call after spank_init() to allow namespace
NEWS: -- switch/nvidia_imex - Skip plugin operation if nvidia-caps-imex-channels
NEWS: -- switch/nvidia_imex - Skip plugin operation if job_container/tmpfs
NEWS: -- switch/nvidia_imex - Remove any pre-existing channels when slurmd starts.
NEWS: -- Calculate a job's min_cpus with consideration to --cpus-per-gpu.
NEWS: -- sview - Fix nodes tab if a node has RestrictedCoresPerGPU configured.
NEWS:    actually pending on Resources when GPUs are requested per job.
NEWS: -- Add RestrictedCoresPerGPU configuration option.
NEWS: -- Prevent slurmd from crashing if acct_gather_energy/gpu is configured but
NEWS: -- sacct - Fix "gpuutil" TRES usage output being incorrect when using --units.
NEWS: -- Fix scheduling jobs that request --gpus and nodes have different node
NEWS:    weights and different numbers of gpus.
NEWS:    --gres=gpu:1,tmpfs:foo:2,tmpfs:bar:7). Otherwise, the job could bypass
NEWS: -- Fix issue where you could have a gpu allocated as well as a shard on that
NEWS:    gpu allocated at the same time.
NEWS: -- Add --gres-flags=allow-task-sharing to allow GPUs to still be accessible
NEWS:    among all tasks when binding GPUs to specific tasks.
NEWS: -- Fix issue with CUDA_VISIBLE_DEVICES showing the same MIG device for all
NEWS:    tasks when using MIGs with --tres-per-task or --gpus-per-task.
NEWS: -- Prevent a slurmctld deadlock if the gpu plugin fails to load when
NEWS: -- Fix jobs getting rejected when submitting with --gpus option from older
NEWS: -- gpu/rsmi - Disable gpu usage statistics when not using ROCM 6.0.0+
NEWS: -- Prevent jobs using none typed gpus from being killed by the controller after
NEWS:    GRES (gpu) per node refuse it unless SelectTypeparameters has
NEWS: -- For jobs that request --cpus-per-gpu, ensure that the --cpus-per-gpu request
NEWS: -- Fix --cpus-per-gpu for step allocations, which was previously ignored for
NEWS:    job steps. --cpus-per-gpu implies --exact.
NEWS: -- Fix mutual exclusivity of --cpus-per-gpu and --cpus-per-task: fatal if both
NEWS: -- gpu/nvml - Reduce chances of NVML_ERROR_INSUFFICIENT_SIZE error when getting
NEWS:    gpu memory information.
NEWS:    tres_per_task=cpu:# and cpus_per_gpu.
NEWS: -- Add gpu/nrt plugin for nodes using Trainium/Inferentia devices.
NEWS: -- gpu/nvml - Fix issue that resulted in the wrong MIG devices being
NEWS: -- gpu/nvml - Fix linking issue with MIGs that prevented multiple MIGs being
NEWS: -- Add JobAcctGatherParams=DisableGPUAcct to disable gpu accounting.
NEWS: -- gpu/oneapi - Add support for new env vars ZE_FLAT_DEVICE_HIERARCHY and
NEWS: -- Fix loading of the gpu account gather energy plugin.
NEWS: -- Fix some mig profile names in slurm not matching nvidia mig profiles.
NEWS: -- Fix jobacctgather/cgroup collection of disk/io, gpumem, gpuutil TRES values.
NEWS: -- Fix intel oneapi autodetect: detect the /dev/dri/renderD[0-9]+ gpus, and do
NEWS: -- Fix node selection for jobs that request --gpus and a number of tasks fewer
NEWS:    than gpus, which resulted in incorrectly rejecting these jobs.
NEWS: -- gpu/oneapi - Store cores correctly so CPU affinity is tracked.
NEWS: -- Fix collected GPUUtilization values for acct_gather_profile plugins.
NEWS:    "error: Attempt to change gres/gpu Count".
NEWS: -- Fix --gpu-bind=single binding tasks to wrong gpus, leading to some gpus
NEWS:    having more tasks than they should and other gpus being unused.
NEWS: -- Fix regression in 23.02.2 that ignored the partition DefCpuPerGPU setting
NEWS:    on the first pass of scheduling a job requesting --gpus --ntasks.
NEWS: -- Fix TresUsageIn[Tot|Ave] calculation for gres/gpumem and gres/gpuutil.
NEWS: -- Avoid unnecessary gres/gpumem and gres/gpuutil TRES position lookups.
NEWS: -- Fix issue in the gpu plugins where gpu frequencies would only be set if both
NEWS:    gpu memory and gpu frequencies were set, while one or the other suffices.
NEWS:    request --cpus-per-gpu and gpus with types.
NEWS: -- Fix job layout calculations with --ntasks-per-gpu, especially when --nodes
NEWS: -- For the select/cons_tres plugin, improve the best effort GPU to core
NEWS:    binding, for requests with per job task count (-n) and GPU (--gpus)
NEWS: -- gpu/oneapi - Fix CPU affinity handling.
NEWS: -- gpu/nvml - Remove E-cores from NVML's cpu affinity bitmap when
NEWS:    --gpu-bind=single request.
NEWS: -- gpu/nvml - Fix gpu usage when graphics processes are running on the gpu.
NEWS:    longer enforce optimal core-gpu job placement.
NEWS: -- Improve error message for using --cpus-per-gpus without any GPUs.
NEWS: -- Fix GPU setup on CRAY systems when using the CRAY_CUDA_MPS environment
NEWS:    variable. GPUs are now correctly detected in such scenarios.
NEWS: -- Removed the default setting for GpuFreqDef. If unset, no attempt to change
NEWS:    the GPU frequency will be made if --gpu-freq is not set for the step.
NEWS: -- Fixed GpuFreqDef option. When set in slurm.conf, it will be used if
NEWS:    --gpu-freq was not explicitly set by the job step.
NEWS: -- NVML - Add usage gathering for Nvidia gpus.
NEWS: -- Fix tasks binding to GPUs when using --ntasks-per-gpu and GPUs have
NEWS:    restart for gpu and xcc implementations.
NEWS: -- RSMI - Add usage gathering for AMD gpus (requires ROCM 5.5+).
NEWS: -- Validate --gpu-bind options more strictly.
NEWS: -- Fix GPU setup on CRAY systems when using the CRAY_CUDA_MPS environment
NEWS:    variable. GPUs are now correctly detected in such scenarios.
NEWS: -- Fix number of CPUs allocated if --cpus-per-gpu used.
NEWS: -- Fix issue where '*' wasn't accepted in gpu/cpu bind.
NEWS: -- Fix SLURM_GPUS_ON_NODE for shared GPU gres (MPS, shards).
NEWS: -- gpu/nvml - Fix MIG minor number generation when GPU minor number
NEWS:    (/dev/nvidia[minor_number]) and index (as seen in nvidia-smi) do not match.
NEWS: -- Fix issue where shards were selected from multiple gpus and failed to
NEWS: -- Fix step cpu count calculation when using --ntasks-per-gpu=.
NEWS: -- Fix regression in task count calculation for --ntasks-per-gpu with multiple
NEWS:    Nvidia drivers.
NEWS: -- Fix task distribution calculations with --ntasks-per-gpu specified without
NEWS: -- Fix regression which prevented a cons_tres gpu job to be submitted to a
NEWS: -- GPU - Fix checking frequencies to check them all and not skip the last one.
NEWS: -- GPU - Fix logic to set frequencies properly when handling multiple GPUs.
NEWS: -- Fix a segfault that may happen on gpu configured as no_consume.
NEWS: -- Fix "--gpu-bind=single:" having wrong env variables.
NEWS: -- gres/gpu - Avoid stripping GRES type field during normalization if any
NEWS: -- acct_gather_energy_rsmi has been renamed acct_gather_energy_gpu.
NEWS: -- gres/gpu - Fix configured/system-detected GRES match for some combinations
NEWS: -- Fix gpus spanning more resources than needed when usng --cpus-per-gpu.
NEWS: -- Skip --mem-per-gpu impact on GRES availability when CR_MEMORY not set.
NEWS: -- Add new shard plugin for sharing gpus but not with mps.
NEWS: -- Add validation of numbers provided to --gpu-bind=map_gpu and
NEWS:    --gpu-bind=mask_gpu=.
NEWS: -- Fix for --gpus-per-task parsing when using --cpus-per-gpu and multiple gpu
NEWS: -- Fix "--gpu-bind=single:" having wrong env variables.
NEWS: -- Fix a segfault that may happen on gpu configured as no_consume.
NEWS: -- Fix regression which prevented a cons_tres gpu job to be submitted to a
NEWS: -- scrontab - fix handling of --gpus and --ntasks-per-gpu options.
NEWS: -- Correctly determine task count when giving --cpus-per-gpu, --gpus and
NEWS:    SLURM_NTASKS_PER_GPU are set.
NEWS: -- Fix error in GPU frequency validation logic.
NEWS: -- Fix --gpu-bind=verbose to work correctly.
NEWS: -- Fix off by one error in --gpu-bind=mask_gpu.
NEWS: -- Add --gpu-bind=none to disable gpu binding when using --gpus-per-task.
NEWS: -- Require requesting a GPU if --mem-per-gpu is requested.
NEWS: -- Return error early if a job is requesting --ntasks-per-gpu and no gpus or
NEWS: -- Restored --gpu-bind=single:<ntasks> to check core affinity like
NEWS:    --gpu-bind=closest does. This removal of this behavior only was in rc2.
NEWS: -- Set step memory when using MemPerGPU or DefMemPerGPU. Previously a step's
NEWS:    memory was not set even when it requested --mem-per-gpu and at least one
NEWS:    GPU.
NEWS: -- Make job's SLURM_JOB_GPUS print global GPU IDs instead of MIG unique_ids.
NEWS: -- Fix miscounting of GPU envs in prolog/epilog if MultipleFiles was used.
NEWS: -- Support MIGs in prolog/epilog's CUDA_VISIBLE_DEVICES & co.
NEWS: -- Add SLURM_JOB_GPUS back into Prolog; add it to Epilog.
NEWS: -- Fix memory in requested TRES when --mem-per-gpu is used.
NEWS: -- Changed --gpu-bind=single:<ntasks> to no longer check core affinity like
NEWS:    --gpu-bind=closest does. This consequently affects --ntasks-per-gpu.
NEWS: -- GPUs: Use index instead of dev_num for CUDA_VISIBLE_DEVICES
NEWS: -- Add --gpu-bind=per_task:<gpus_per_task> option, --gpus-per-task will now
NEWS: -- GRES - Fix loading state of jobs using --gpus to request gpus.
NEWS: -- Fix error in GPU frequency validation logic.
NEWS: -- Add GRES environment variables (e.g., CUDA_VISIBLE_DEVICES) into the
NEWS: -- Fail step creation if -n is not multiple of --ntasks-per-gpu.
NEWS: -- Ignore DefCpuPerGpu when --cpus-per-task given.
NEWS: -- cons_tres - Fix DefCpuPerGPU, increase cpus-per-task to match with
NEWS:    gpus-per-task * cpus-per-gpu.
NEWS: -- Fix scheduling issue with --gpus.
NEWS: -- Fix gpu allocations that request --cpus-per-task.
NEWS: -- Allow countless gpu:<type> node GRES specifications in slurm.conf.
NEWS: -- Don't green-light any GPU validation when core conversion fails.
NEWS: -- Fix job rejection when --gres is less than --gpus.
NEWS: -- Fix situation when --gpus is given but not max nodes (-N1-1) in a job
NEWS:    --ntasks-per-gpu.
NEWS: -- cons_tres - Fix DefCpuPerGPU
NEWS: -- Enforce invalid argument combinations with --ntasks-per-gpu
NEWS: -- cons_tres - fix regression regarding gpus with --cpus-per-task.
NEWS: -- Add --gpu-bind=mask_gpu reusability functionality if tasks > elements.
NEWS: -- Enable -lnodes=#:gpus=# in #PBS/qsub -l nodes syntax.
NEWS: -- Add --ntasks-per-gpu option.
NEWS: -- Add --gpu-bind=single option.
NEWS: -- cons_tres - Fix DefCpuPerGPU
NEWS: -- cons_tres - fix regression regarding gpus with --cpus-per-task.
NEWS: -- cons_tres - fix job not getting access to socket without GPU or with less
NEWS:    than --gpus-per-socket when not enough cpus available on required socket
NEWS: -- Fix AMD GPU ROCM 3.5 support.
NEWS: -- Avoid partial GRES allocation when --gpus-per-job is not satisfied.
NEWS: -- Fix node estimation for jobs that use GPUs or --cpus-per-task.
NEWS: -- Fix propagation of gpu options through hetjob components.
NEWS: -- Fix gpu bind issue when CPUs=Cores and ThreadsPerCore > 1 on a node.
NEWS: -- Fix --mem-per-gpu for heterogenous --gres requests.
NEWS: -- Fix regresion validating that --gpus-per-socket requires --sockets-per-node
NEWS: -- Fix MPS without File with 1 GPU, and without GPUs.
NEWS: -- Add client error when using --gpus-per-socket without --sockets-per-node.
NEWS: -- Fix _verify_node_state memory requested as --mem-per-gpu DefMemPerGPU.
NEWS: -- Fix --gpu-bind=map_gpu reusability if tasks > elements.
NEWS: -- Prolog/Epilog - Fix missing GPU information.
NEWS: -- Improve handling of --gpus-per-task to make sure appropriate number of GPUs
NEWS: -- Add gpu/rsmi plugin to support AMD GPUs
NEWS: -- Add energy accounting plugin for AMD GPU
NEWS: -- Fix _verify_node_state memory requested as --mem-per-gpu DefMemPerGPU.
NEWS: -- Fix gpu bind issue when CPUs=Cores and ThreadsPerCore > 1 on a node.
NEWS: -- Fix --mem-per-gpu for heterogenous --gres requests.
NEWS: -- Fix for --gpu-bind when no gpus requested.
NEWS: -- Improve handling of --gpus-per-task to make sure appropriate number of GPUs
NEWS: -- Reset --mem and --mem-per-cpu options correctly when using --mem-per-gpu.
NEWS: -- Fix issues with --gpu-bind while using cgroups.
NEWS: -- Fix issue where GPU devices are denied access when MPS is enabled.
NEWS: -- Fix regression with SLURM_STEP_GPUS env var being renamed SLURM_STEP_GRES.
NEWS: -- Change GRES type set by gpu/gpu_nvml plugin to be more specific - based
NEWS: -- Remove premature call to get system gpus before querying fake gpus that
NEWS: -- Fix issue when --gpus plus --cpus-per-gres was forcing socket binding
NEWS: -- Calculate task count for job with --gpus-per-task option, but no explicit
NEWS: -- Set CUDA_VISIBLE_DEVICES environment variable in Prolog and Epilog for jobs
NEWS:    requesting gres/gpu.
NEWS: -- Do not set CUDA_VISIBLE_DEVICES=NoDevFiles when no gres requested.
NEWS: -- Reset GPU-related arguments to salloc/sbatch/srun for each separate
NEWS: -- Support GRES types that include numbers (e.g. "--gres=gpu:123g:2").
NEWS: -- Set CUDA_VISIBLE_DEVICES and CUDA_MPS_ACTIVE_THREAD_PERCENTAGE environment
NEWS:    output of "scontrol show node". For example if all 4 GPUs on a node are
NEWS:    all associated with socket zero, then "Gres=gpu:4(S:0)". If associated
NEWS:    with sockets 0 and 1 then "Gres=gpu:4(S:0-1)". The information of which
NEWS:    specific GPUs are associated with specific GPUs is not reported, but only
NEWS: -- Add configuration parameter "GpuFreqDef" to control a job's default GPU
NEWS: -- Add DefCpuPerGpu and DefMemPerGpu to global and per-partition configuration
NEWS:    currently required to use more CPUs than are bound to a GRES (i.e. if a GPU
NEWS:    (i.e. gres=gpu/tesla) it would get a count of 0.
NEWS: -- Properly set SLURM_JOB_GPUS environment variable for Prolog.
NEWS:    index numbers (e.g. "GresUsed=gpu:alpha:2(IDX:0,2),gpu:beta:0(IDX:N/A)").
NEWS: -- Make it so jobs/steps track ':' named gres/tres, before hand gres/gpu:tesla
NEWS:    would only track gres/gpu, now it will track both gres/gpu and
NEWS:    gres/gpu:tesla as separate gres if configured like
NEWS:    AccountingStorageTRES=gres/gpu,gres/gpu:tesla
NEWS:    memory or GPUs. This would result in underflow/overflow errors in select
NEWS: -- Add srun --accel-bind option to control how tasks are bound to GPUs and NIC
NEWS:    GPUs allocated to the job.
NEWS: -- Add SLURM_JOB_GPUS environment variable to those available in the Prolog.
NEWS: -- Enable CUDA v7.0+ use with a Slurm configuration of TaskPlugin=task/cgroup
NEWS:    CUDA_VISIBLE_DEVICES will start at 0 rather than the device number.
NEWS:    (e.g. request a Kepler GPU, a Tesla GPU, or a GPU of any type).
NEWS: -- Added support for selecting AMD GPU by setting GPU_DEVICE_ORDINAL env var.
NEWS:    -l accelerator=true|false	(GPU use)
NEWS:    -l naccelerators=#	(GPU count)
NEWS: -- gres/gpu and gres/mic - Do not treat the existence of an empty gres.conf
NEWS:    especially useful to schedule systems with GPUs.
NEWS: -- gres/gpu - Fix for gres.conf file with multiple files on a single line
NEWS:    using a slurm expression (e.g. "File=/dev/nvidia[0-1]").
NEWS: -- Gres/gpu plugin - If no GPUs requested, set CUDA_VISIBLE_DEVICES=NoDevFiles.
NEWS:    This bug was introduced in 2.5.2 for the case where a GPU count was
NEWS:    the gres option (e.g. "--licenses=foo:2 --gres=gpu:2"). The "*" will still
NEWS: -- Add logic to cache GPU file information (bitmap index mapping to device
NEWS:    appropriate CUDA_VISIBLE_DEVICES environment variable value when the
NEWS:    devices are not in strict numeric order (e.g. some GPUs are skipped).
NEWS: -- CRAY - Add support for GPU memory allocation using SLURM GRES (Generic
NEWS: -- Cray - Remove the "family" specification from the GPU reservation request.
NEWS:    a constraint field with a count (e.g. "srun --constraint=gpu*2 -N4 a.out").
NEWS: -- Gres/gpu now sets the CUDA_VISIBLE_DEVICES environment to control which
NEWS:    GPU devices should be used for each job or job step and CUDA version 3.1+
NEWS:    -Integrate with HWLOC library to identify GPUs and NICs configured on each
slurm/slurm.h:	SWITCH_PLUGIN_NVIDIA_IMEX = 105,
slurm/slurm.h:	ACCEL_BIND_CLOSEST_GPU     = 0x02, /* 'g' Use closest GPU to the CPU */
slurm/slurm.h:	uint16_t ntasks_per_tres;/* number of tasks that can access each gpu */
slurm/slurm.h:	uint16_t ntasks_per_tres;/* number of tasks that can access each gpu */
slurm/slurm.h:	uint16_t ntasks_per_tres;/* number of tasks that can access each gpu */
slurm/slurm.h:	uint16_t res_cores_per_gpu; /* number of cores per GPU to allow
slurm/slurm.h:				     * to only GPU jobs */
slurm/slurm.h:	char *gpu_spec;         /* node's cores reserved for GPU jobs */
slurm/slurm.h:#define JOB_DEF_CPU_PER_GPU	0x0001
slurm/slurm.h:#define JOB_DEF_MEM_PER_GPU	0x0002
slurm/slurm.h:	uint16_t ntasks_per_tres;/* number of tasks that can access each gpu */
slurm/slurm.h:	char *gpu_freq_def;	/* default GPU frequency / voltage */
slurm/slurmdb.h:			 * (e.g. "gpu" or "gpu:tesla") */
slurm/slurm_errno.h:	ESLURM_RES_CORES_PER_GPU_UNIQUE,
slurm/slurm_errno.h:	ESLURM_RES_CORES_PER_GPU_TOPO,
slurm/slurm_errno.h:	ESLURM_RES_CORES_PER_GPU_NO,
slurm.spec:exec %{__find_requires} "$@" | grep -E -v '^libpmix.so|libevent|libnvidia-ml'
contribs/torque/qsub.pl:$command .= " --gpus-per-node=$node_opts{gpu_cnt}" if $node_opts{gpu_cnt};
contribs/torque/qsub.pl:$command .= " --gres=gpu:$res_opts{naccelerators}"  if $res_opts{naccelerators};
contribs/torque/qsub.pl:		   'gpu_cnt'  => 0,
contribs/torque/qsub.pl:	while($node_string =~ /gpus=(\d+)/g) {
contribs/torque/qsub.pl:	        $opt{gpu_cnt} += $1;
contribs/torque/qsub.pl:			   ($sub_part =~ /gpus=(\d+)/)) {
contribs/torque/pbsnodes.pl:	    my $gpus = 0;
contribs/torque/pbsnodes.pl:			  if ( $#elt>0 && $elt[0] eq "gpu" ) {
contribs/torque/pbsnodes.pl:					$gpus += int($elt[1]);
contribs/torque/pbsnodes.pl:					$gpus += int($elt[2]);
contribs/torque/pbsnodes.pl:		    printf "    gpus = %d\n", $gpus if $gpus>0;
contribs/slurm_completion_help/slurm_completion.sh:	local gpubind_types=(
contribs/slurm_completion_help/slurm_completion.sh:		"map_gpu:"
contribs/slurm_completion_help/slurm_completion.sh:		"mask_gpu:"
contribs/slurm_completion_help/slurm_completion.sh:	local gpufreq_types=(
contribs/slurm_completion_help/slurm_completion.sh:	--gpu-bind) __slurm_compreply "${gpubind_types[*]}" ;;
contribs/slurm_completion_help/slurm_completion.sh:	--gpu-freq) __slurm_compreply "${gpufreq_types[*]}" ;;
contribs/slurm_completion_help/slurm_completion.sh:		"defcpupergpu="
contribs/slurm_completion_help/slurm_completion.sh:		"defmempergpu="
src/plugins/gres/shard/gres_shard.c: *  Sharding is a mechanism to share GPUs generically.
src/plugins/gres/shard/gres_shard.c:#include "src/interfaces/gpu.h"
src/plugins/gres/shard/gres_shard.c:		gres_conf_list, &gres_devices, config, "gpu");
src/plugins/gres/shard/gres_shard.c:		char *gpus_on_node = xstrdup_printf("%"PRIu64,
src/plugins/gres/shard/gres_shard.c:				    gpus_on_node);
src/plugins/gres/shard/gres_shard.c:		xfree(gpus_on_node);
src/plugins/gres/shard/gres_shard.c:	gres_common_gpu_set_env(&gres_env);
src/plugins/gres/shard/gres_shard.c:	gres_common_gpu_set_env(&gres_env);
src/plugins/gres/shard/gres_shard.c:	gres_common_gpu_set_env(&gres_env);
src/plugins/gres/shard/gres_shard.c:	gpu_g_step_hardware_init(usable_gres, settings);
src/plugins/gres/shard/gres_shard.c:	gpu_g_step_hardware_fini();
src/plugins/gres/mps/gres_mps.c: *  MPS or CUDA Multi-Process Services is a mechanism to share GPUs.
src/plugins/gres/mps/gres_mps.c:		gres_conf_list, &gres_devices, config, "gpu");
src/plugins/gres/mps/gres_mps.c:	gres_common_gpu_set_env(gres_env);
src/plugins/gres/mps/gres_mps.c:				    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
src/plugins/gres/mps/gres_mps.c:				    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
src/plugins/gres/mps/gres_mps.c:			  "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
src/plugins/gres/mps/gres_mps.c:					"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
src/plugins/gres/common/gres_c_s.c:		/* The default for MPS is to have only one gpu sharing */
src/plugins/gres/common/gres_c_s.c:static bool _test_gpu_list_fake(void)
src/plugins/gres/common/gres_c_s.c:	char *fake_gpus_file = NULL;
src/plugins/gres/common/gres_c_s.c:	bool have_fake_gpus = false;
src/plugins/gres/common/gres_c_s.c:	fake_gpus_file = get_extra_conf_path("fake_gpus.conf");
src/plugins/gres/common/gres_c_s.c:	if (stat(fake_gpus_file, &config_stat) >= 0) {
src/plugins/gres/common/gres_c_s.c:		have_fake_gpus = true;
src/plugins/gres/common/gres_c_s.c:	xfree(fake_gpus_file);
src/plugins/gres/common/gres_c_s.c:	return have_fake_gpus;
src/plugins/gres/common/gres_c_s.c:/* Translate device file name to numeric index "/dev/nvidia2" -> 2 */
src/plugins/gres/common/gres_c_s.c: * unique device (i.e. convert a record with "File=nvidia[0-3]" into 4 separate
src/plugins/gres/common/gres_c_s.c:		 * device, as opposed to File=nvidia[0-3] which corresponds to
src/plugins/gres/common/gres_c_s.c: * a unique device (i.e. convert a record with "File=nvidia[0-3]" into 4
src/plugins/gres/common/gres_c_s.c: * separate records). Similar to _build_gpu_list(), but we copy more fields,
src/plugins/gres/common/gres_c_s.c:	 * "File=nvidia[0-3]" into 4 separate records).
src/plugins/gres/common/gres_c_s.c:	if (_test_gpu_list_fake()) {
src/plugins/gres/common/gres_common.c:		 * in order to set the CUDA_VISIBLE_DEVICES env var
src/plugins/gres/common/gres_common.c:		 * file name; GPUs, however, want it to match the order
src/plugins/gres/common/gres_common.c:		fprintf(stderr, "gpu-bind: usable_gres=%s; bit_alloc=%s; local_inx=%d; global_list=%s; local_list=%s\n",
src/plugins/gres/common/gres_common.c:extern void gres_common_gpu_set_env(common_gres_env_t *gres_env)
src/plugins/gres/common/gres_common.c:		slurm_env_var = "SLURM_JOB_GPUS";
src/plugins/gres/common/gres_common.c:		slurm_env_var = "SLURM_STEP_GPUS";
src/plugins/gres/common/gres_common.c:	 * sharing GRES (GPU).
src/plugins/gres/common/gres_common.c:	 * NOTE: Use gres_env->bit_alloc to ensure SLURM_GPUS_ON_NODE is
src/plugins/gres/common/gres_common.c:		char *gpus_on_node = xstrdup_printf("%"PRIu64,
src/plugins/gres/common/gres_common.c:		env_array_overwrite(gres_env->env_ptr, "SLURM_GPUS_ON_NODE",
src/plugins/gres/common/gres_common.c:				    gpus_on_node);
src/plugins/gres/common/gres_common.c:		xfree(gpus_on_node);
src/plugins/gres/common/gres_common.c:		unsetenvp(*gres_env->env_ptr, "SLURM_GPUS_ON_NODE");
src/plugins/gres/common/gres_common.c:					    "CUDA_VISIBLE_DEVICES",
src/plugins/gres/common/gres_common.c:		if (gres_env->gres_conf_flags & GRES_CONF_ENV_OPENCL)
src/plugins/gres/common/gres_common.c:					    "GPU_DEVICE_ORDINAL",
src/plugins/gres/common/gres_common.c:			unsetenvp(*gres_env->env_ptr, "CUDA_VISIBLE_DEVICES");
src/plugins/gres/common/gres_common.c:		if (gres_env->gres_conf_flags & GRES_CONF_ENV_OPENCL)
src/plugins/gres/common/gres_common.c:			unsetenvp(*gres_env->env_ptr, "GPU_DEVICE_ORDINAL");
src/plugins/gres/common/gres_common.c:	char *vendor_gpu_str = NULL;
src/plugins/gres/common/gres_common.c:	char *slurm_gpu_str = NULL;
src/plugins/gres/common/gres_common.c:				xstrfmtcat(vendor_gpu_str, "%s%s", sep,
src/plugins/gres/common/gres_common.c:				xstrfmtcat(vendor_gpu_str, "%s%d", sep,
src/plugins/gres/common/gres_common.c:			xstrfmtcat(slurm_gpu_str, "%s%d", sep,
src/plugins/gres/common/gres_common.c:	if (vendor_gpu_str) {
src/plugins/gres/common/gres_common.c:					    "CUDA_VISIBLE_DEVICES",
src/plugins/gres/common/gres_common.c:					    vendor_gpu_str);
src/plugins/gres/common/gres_common.c:					    vendor_gpu_str);
src/plugins/gres/common/gres_common.c:					    vendor_gpu_str);
src/plugins/gres/common/gres_common.c:		if (gres_conf_flags & GRES_CONF_ENV_OPENCL)
src/plugins/gres/common/gres_common.c:					    "GPU_DEVICE_ORDINAL",
src/plugins/gres/common/gres_common.c:					    vendor_gpu_str);
src/plugins/gres/common/gres_common.c:		xfree(vendor_gpu_str);
src/plugins/gres/common/gres_common.c:	if (slurm_gpu_str) {
src/plugins/gres/common/gres_common.c:		env_array_overwrite(prep_env_ptr, "SLURM_JOB_GPUS",
src/plugins/gres/common/gres_common.c:				    slurm_gpu_str);
src/plugins/gres/common/gres_common.c:		xfree(slurm_gpu_str);
src/plugins/gres/common/gres_common.c:	if (gres_slurmd_conf->config_flags & GRES_CONF_ENV_OPENCL)
src/plugins/gres/common/gres_common.c:		*node_flags |= GRES_CONF_ENV_OPENCL;
src/plugins/gres/common/gres_common.h: * Set the appropriate env variables for all gpu like gres.
src/plugins/gres/common/gres_common.h:extern void gres_common_gpu_set_env(common_gres_env_t *gres_env);
src/plugins/gres/Makefile.in:SUBDIRS = common gpu nic mps shard
src/plugins/gres/Makefile.am:SUBDIRS = common gpu nic mps shard
src/plugins/gres/gpu/Makefile.in:# Makefile for gres/gpu plugin
src/plugins/gres/gpu/Makefile.in:subdir = src/plugins/gres/gpu
src/plugins/gres/gpu/Makefile.in:gres_gpu_la_DEPENDENCIES = ../common/libgres_common.la
src/plugins/gres/gpu/Makefile.in:am_gres_gpu_la_OBJECTS = gres_gpu.lo
src/plugins/gres/gpu/Makefile.in:gres_gpu_la_OBJECTS = $(am_gres_gpu_la_OBJECTS)
src/plugins/gres/gpu/Makefile.in:gres_gpu_la_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
src/plugins/gres/gpu/Makefile.in:	$(gres_gpu_la_LDFLAGS) $(LDFLAGS) -o $@
src/plugins/gres/gpu/Makefile.in:am__depfiles_remade = ./$(DEPDIR)/gres_gpu.Plo
src/plugins/gres/gpu/Makefile.in:SOURCES = $(gres_gpu_la_SOURCES)
src/plugins/gres/gpu/Makefile.in:pkglib_LTLIBRARIES = gres_gpu.la
src/plugins/gres/gpu/Makefile.in:# Gres GPU plugin.
src/plugins/gres/gpu/Makefile.in:gres_gpu_la_SOURCES = gres_gpu.c
src/plugins/gres/gpu/Makefile.in:gres_gpu_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gres/gpu/Makefile.in:gres_gpu_la_LIBADD = ../common/libgres_common.la
src/plugins/gres/gpu/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign src/plugins/gres/gpu/Makefile'; \
src/plugins/gres/gpu/Makefile.in:	  $(AUTOMAKE) --foreign src/plugins/gres/gpu/Makefile
src/plugins/gres/gpu/Makefile.in:gres_gpu.la: $(gres_gpu_la_OBJECTS) $(gres_gpu_la_DEPENDENCIES) $(EXTRA_gres_gpu_la_DEPENDENCIES) 
src/plugins/gres/gpu/Makefile.in:	$(AM_V_CCLD)$(gres_gpu_la_LINK) -rpath $(pkglibdir) $(gres_gpu_la_OBJECTS) $(gres_gpu_la_LIBADD) $(LIBS)
src/plugins/gres/gpu/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/gres_gpu.Plo@am__quote@ # am--include-marker
src/plugins/gres/gpu/Makefile.in:		-rm -f ./$(DEPDIR)/gres_gpu.Plo
src/plugins/gres/gpu/Makefile.in:		-rm -f ./$(DEPDIR)/gres_gpu.Plo
src/plugins/gres/gpu/Makefile.in:$(gres_gpu_la_LIBADD) : force
src/plugins/gres/gpu/Makefile.am:# Makefile for gres/gpu plugin
src/plugins/gres/gpu/Makefile.am:pkglib_LTLIBRARIES = gres_gpu.la
src/plugins/gres/gpu/Makefile.am:# Gres GPU plugin.
src/plugins/gres/gpu/Makefile.am:gres_gpu_la_SOURCES = gres_gpu.c
src/plugins/gres/gpu/Makefile.am:gres_gpu_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gres/gpu/Makefile.am:gres_gpu_la_LIBADD = ../common/libgres_common.la
src/plugins/gres/gpu/Makefile.am:$(gres_gpu_la_LIBADD) : force
src/plugins/gres/gpu/gres_gpu.c: *  gres_gpu.c - Support GPUs as a generic resources.
src/plugins/gres/gpu/gres_gpu.c:#include "src/interfaces/gpu.h"
src/plugins/gres/gpu/gres_gpu.c:const char plugin_name[] = "Gres GPU plugin";
src/plugins/gres/gpu/gres_gpu.c:const char	plugin_type[]		= "gres/gpu";
src/plugins/gres/gpu/gres_gpu.c:extern void gres_p_step_hardware_init(bitstr_t *usable_gpus, char *tres_freq)
src/plugins/gres/gpu/gres_gpu.c:	gpu_g_step_hardware_init(usable_gpus, tres_freq);
src/plugins/gres/gpu/gres_gpu.c:	gpu_g_step_hardware_fini();
src/plugins/gres/gpu/gres_gpu.c: * Sort gres/gpu records by descending length of type_name. If length is equal,
src/plugins/gres/gpu/gres_gpu.c:static int _sort_gpu_by_type_name(void *x, void *y)
src/plugins/gres/gpu/gres_gpu.c:	list_sort(gres_list_conf_single, _sort_gpu_by_type_name);
src/plugins/gres/gpu/gres_gpu.c:	list_sort(gres_list_system, _sort_gpu_by_type_name);
src/plugins/gres/gpu/gres_gpu.c:/* Sort gres/gpu records by "File" value in ascending order, with nulls last */
src/plugins/gres/gpu/gres_gpu.c:static int _sort_gpu_by_file(void *x, void *y)
src/plugins/gres/gpu/gres_gpu.c: * Sort GPUs by the order they are specified in links.
src/plugins/gres/gpu/gres_gpu.c: * current GPU at the position it was enumerated in. The GPUs will be sorted so
src/plugins/gres/gpu/gres_gpu.c:static int _sort_gpu_by_links_order(void *x, void *y)
src/plugins/gres/gpu/gres_gpu.c: * gres_list_non_gpu and gres_list_conf_single. All GPU records are split into
src/plugins/gres/gpu/gres_gpu.c: * matches, those records are added to gres_list_gpu. Finally, the old
src/plugins/gres/gpu/gres_gpu.c: * gres_list_conf is cleared, gres_list_gpu and gres_list_non_gpu are combined,
src/plugins/gres/gpu/gres_gpu.c: * If a conf GPU corresponds to a system GPU, CPUs and Links are checked to see
src/plugins/gres/gpu/gres_gpu.c: * gres_list_system - (in) The gpu devices detected by the system. Each record
src/plugins/gres/gpu/gres_gpu.c: * A conf GPU and system GPU will be matched if the following fields are equal:
src/plugins/gres/gpu/gres_gpu.c:	list_t *gres_list_conf_single, *gres_list_gpu = NULL, *gres_list_non_gpu;
src/plugins/gres/gpu/gres_gpu.c:	gres_list_non_gpu = list_create(destroy_gres_slurmd_conf);
src/plugins/gres/gpu/gres_gpu.c:	gres_list_gpu = list_create(destroy_gres_slurmd_conf);
src/plugins/gres/gpu/gres_gpu.c:		if (xstrcasecmp(gres_slurmd_conf->name, "gpu")) {
src/plugins/gres/gpu/gres_gpu.c:			/* Move record into non-GPU GRES list */
src/plugins/gres/gpu/gres_gpu.c:			list_append(gres_list_non_gpu, gres_slurmd_conf);
src/plugins/gres/gpu/gres_gpu.c:			/* Already count of 1; move into single-GPU GRES list */
src/plugins/gres/gpu/gres_gpu.c:			 * Split this record into multiple single-GPU records
src/plugins/gres/gpu/gres_gpu.c:			 * and add them to the single-GPU GRES list
src/plugins/gres/gpu/gres_gpu.c:			 * Split this record into multiple single-GPU,
src/plugins/gres/gpu/gres_gpu.c:			 * single-file records and add to single-GPU GRES list
src/plugins/gres/gpu/gres_gpu.c:	list_sort(gres_list_conf_single, _sort_gpu_by_file);
src/plugins/gres/gpu/gres_gpu.c:	list_sort(gres_list_system, _sort_gpu_by_file);
src/plugins/gres/gpu/gres_gpu.c:				error("This GPU specified in [slurm|gres].conf has mismatching Cores or Links from the device found on the system. Ignoring it.");
src/plugins/gres/gpu/gres_gpu.c:			 * Links do not match the corresponding system GPU
src/plugins/gres/gpu/gres_gpu.c:			 * Since the system GPU matches up completely with a
src/plugins/gres/gpu/gres_gpu.c:			 * configured GPU, add the system GPU to the final list
src/plugins/gres/gpu/gres_gpu.c:			debug("Including the following GPU matched between system and configuration:");
src/plugins/gres/gpu/gres_gpu.c:			list_append(gres_list_gpu, gres_slurmd_conf_sys);
src/plugins/gres/gpu/gres_gpu.c:		/* Else, config-only GPU */
src/plugins/gres/gpu/gres_gpu.c:			 * Add the "extra" configured GPU to the final list, but
src/plugins/gres/gpu/gres_gpu.c:			debug("Including the following config-only GPU:");
src/plugins/gres/gpu/gres_gpu.c:			list_append(gres_list_gpu, gres_slurmd_conf);
src/plugins/gres/gpu/gres_gpu.c:			 * Either the conf GPU was specified in slurm.conf only,
src/plugins/gres/gpu/gres_gpu.c:			 * or File (a required parameter for GPUs) was not
src/plugins/gres/gpu/gres_gpu.c:			error("Discarding the following config-only GPU due to lack of File specification:");
src/plugins/gres/gpu/gres_gpu.c:	/* Reset the system GPU counts, in case system list is used after */
src/plugins/gres/gpu/gres_gpu.c:	/* Print out all the leftover system GPUs that are not being used */
src/plugins/gres/gpu/gres_gpu.c:		warning("The following autodetected GPUs are being ignored:");
src/plugins/gres/gpu/gres_gpu.c:	/* Add GPUs + non-GPUs to gres_list_conf */
src/plugins/gres/gpu/gres_gpu.c:	if (gres_list_gpu && list_count(gres_list_gpu)) {
src/plugins/gres/gpu/gres_gpu.c:		list_sort(gres_list_gpu, _sort_gpu_by_file);
src/plugins/gres/gpu/gres_gpu.c:		list_sort(gres_list_gpu, _sort_gpu_by_links_order);
src/plugins/gres/gpu/gres_gpu.c:		debug2("gres_list_gpu");
src/plugins/gres/gpu/gres_gpu.c:		print_gres_list(gres_list_gpu, LOG_LEVEL_DEBUG2);
src/plugins/gres/gpu/gres_gpu.c:		list_transfer(gres_list_conf, gres_list_gpu);
src/plugins/gres/gpu/gres_gpu.c:	if (gres_list_non_gpu && list_count(gres_list_non_gpu))
src/plugins/gres/gpu/gres_gpu.c:		list_transfer(gres_list_conf, gres_list_non_gpu);
src/plugins/gres/gpu/gres_gpu.c:	FREE_NULL_LIST(gres_list_gpu);
src/plugins/gres/gpu/gres_gpu.c:	FREE_NULL_LIST(gres_list_non_gpu);
src/plugins/gres/gpu/gres_gpu.c: * Parses fake_gpus_file for fake GPU devices and adds them to gres_list_system
src/plugins/gres/gpu/gres_gpu.c: * Each line represents a single GPU device. Therefore, <device_file> can't
src/plugins/gres/gpu/gres_gpu.c:static void _add_fake_gpus_from_file(list_t *gres_list_system,
src/plugins/gres/gpu/gres_gpu.c:				     char *fake_gpus_file)
src/plugins/gres/gpu/gres_gpu.c:	FILE *f = fopen(fake_gpus_file, "r");
src/plugins/gres/gpu/gres_gpu.c:		error("Unable to read \"%s\": %m", fake_gpus_file);
src/plugins/gres/gpu/gres_gpu.c:			.name = "gpu",
src/plugins/gres/gpu/gres_gpu.c:						gpu_g_test_cpu_conv(tok);
src/plugins/gres/gpu/gres_gpu.c:			error("Line #%d in fake_gpus.conf failed to parse! Make sure that the line has no empty tokens and that the format is <type>|<sys_cpu_count>|<cpu_range>|<links>|<device_file>[|<unique_id>[|<flags>]]",
src/plugins/gres/gpu/gres_gpu.c:		// Add the GPU specified by the parsed line
src/plugins/gres/gpu/gres_gpu.c: * Creates and returns a list of system GPUs if fake_gpus.conf exists
src/plugins/gres/gpu/gres_gpu.c: * GPU system info will be artificially set to whatever fake_gpus.conf specifies
src/plugins/gres/gpu/gres_gpu.c: * If fake_gpus.conf does not exist, or an error occurs, returns NULL
src/plugins/gres/gpu/gres_gpu.c:static list_t *_get_system_gpu_list_fake(void)
src/plugins/gres/gpu/gres_gpu.c:	char *fake_gpus_file = NULL;
src/plugins/gres/gpu/gres_gpu.c:	 * Only add "fake" data if fake_gpus.conf exists
src/plugins/gres/gpu/gres_gpu.c:	fake_gpus_file = get_extra_conf_path("fake_gpus.conf");
src/plugins/gres/gpu/gres_gpu.c:	if (stat(fake_gpus_file, &config_stat) >= 0) {
src/plugins/gres/gpu/gres_gpu.c:		info("Adding fake system GPU data from %s", fake_gpus_file);
src/plugins/gres/gpu/gres_gpu.c:		_add_fake_gpus_from_file(gres_list_system, fake_gpus_file);
src/plugins/gres/gpu/gres_gpu.c:	xfree(fake_gpus_file);
src/plugins/gres/gpu/gres_gpu.c:	gpu_plugin_fini();
src/plugins/gres/gpu/gres_gpu.c:	gres_list_system = _get_system_gpu_list_fake();
src/plugins/gres/gpu/gres_gpu.c:		gres_list_system = gpu_g_get_system_gpu_list(node_config);
src/plugins/gres/gpu/gres_gpu.c:				"There were 0 GPUs detected on the system");
src/plugins/gres/gpu/gres_gpu.c:			"%s: Merging configured GRES with system GPUs",
src/plugins/gres/gpu/gres_gpu.c:	gres_common_gpu_set_env(&gres_env);
src/plugins/gres/gpu/gres_gpu.c:	gres_common_gpu_set_env(&gres_env);
src/plugins/gres/gpu/gres_gpu.c:	gres_common_gpu_set_env(&gres_env);
src/plugins/gres/gpu/gres_gpu.c:/* Send GPU-specific GRES information to slurmstepd via a buffer */
src/plugins/gres/gpu/gres_gpu.c:/* Receive GPU-specific GRES information from slurmd via a buffer */
src/plugins/select/cons_tres/cons_helpers.c: * Get configured DefCpuPerGPU information from a list
src/plugins/select/cons_tres/cons_helpers.c:extern uint64_t cons_helpers_get_def_cpu_per_gpu(list_t *job_defaults_list)
src/plugins/select/cons_tres/cons_helpers.c:	uint64_t cpu_per_gpu = NO_VAL64;
src/plugins/select/cons_tres/cons_helpers.c:		return cpu_per_gpu;
src/plugins/select/cons_tres/cons_helpers.c:		if (job_defaults->type == JOB_DEF_CPU_PER_GPU) {
src/plugins/select/cons_tres/cons_helpers.c:			cpu_per_gpu = job_defaults->value;
src/plugins/select/cons_tres/cons_helpers.c:	return cpu_per_gpu;
src/plugins/select/cons_tres/cons_helpers.c: * Get configured DefMemPerGPU information from a list
src/plugins/select/cons_tres/cons_helpers.c:extern uint64_t cons_helpers_get_def_mem_per_gpu(list_t *job_defaults_list)
src/plugins/select/cons_tres/cons_helpers.c:	uint64_t mem_per_gpu = NO_VAL64;
src/plugins/select/cons_tres/cons_helpers.c:		return mem_per_gpu;
src/plugins/select/cons_tres/cons_helpers.c:		if (job_defaults->type == JOB_DEF_MEM_PER_GPU) {
src/plugins/select/cons_tres/cons_helpers.c:			mem_per_gpu = job_defaults->value;
src/plugins/select/cons_tres/cons_helpers.c:	return mem_per_gpu;
src/plugins/select/cons_tres/cons_helpers.c:	bool req_gpu = false;
src/plugins/select/cons_tres/cons_helpers.c:	uint32_t gpu_plugin_id = gres_get_gpu_plugin_id();
src/plugins/select/cons_tres/cons_helpers.c:			     gres_find_id, &gpu_plugin_id)))
src/plugins/select/cons_tres/cons_helpers.c:		req_gpu = true;
src/plugins/select/cons_tres/cons_helpers.c:		 * If the job isn't requesting a GPU we will remove those cores
src/plugins/select/cons_tres/cons_helpers.c:		 * that are reserved for gpu jobs.
src/plugins/select/cons_tres/cons_helpers.c:		if (node_ptr->gpu_spec_bitmap && !req_gpu) {
src/plugins/select/cons_tres/cons_helpers.c:				if (!bit_test(node_ptr->gpu_spec_bitmap, i)) {
src/plugins/select/cons_tres/gres_select_util.h: *		  gres_name="gpu" would apply to "gpu:tesla", "gpu:volta", etc.)
src/plugins/select/cons_tres/gres_select_util.h: * IN cpu_per_gpu - value to set as default
src/plugins/select/cons_tres/gres_select_util.h: * IN mem_per_gpu - value to set as default
src/plugins/select/cons_tres/gres_select_util.h: * IN/OUT *cpus_per_task - Increased if cpu_per_gpu * gres_per_task is more than
src/plugins/select/cons_tres/gres_select_util.h:					  uint64_t cpu_per_gpu,
src/plugins/select/cons_tres/gres_select_util.h:					  uint64_t mem_per_gpu,
src/plugins/select/cons_tres/gres_select_util.h: * ntasks_per_tres IN - # of tasks per GPU
src/plugins/select/cons_tres/gres_select_filter.c: * OUT avail_gpus - Count of available GPUs on this node
src/plugins/select/cons_tres/gres_select_filter.c: * OUT near_gpus - Count of GPUs available on sockets with available CPUs
src/plugins/select/cons_tres/gres_select_filter.c:					      uint16_t *avail_gpus,
src/plugins/select/cons_tres/gres_select_filter.c:					      uint16_t *near_gpus)
src/plugins/select/cons_tres/gres_select_filter.c:	*avail_gpus = 0;
src/plugins/select/cons_tres/gres_select_filter.c:	*near_gpus = 0;
src/plugins/select/cons_tres/gres_select_filter.c:			*avail_gpus += sock_gres->total_cnt;
src/plugins/select/cons_tres/gres_select_filter.c:			if (*near_gpus + near_gres_cnt < 0xff)
src/plugins/select/cons_tres/gres_select_filter.c:				*near_gpus += near_gres_cnt;
src/plugins/select/cons_tres/gres_select_filter.c:				*near_gpus = 0xff;
src/plugins/select/cons_tres/gres_select_filter.c:		error("Not enough shared gres on required sockets to satisfy allocated restricted gpu cores for job %u on node %d",
src/plugins/select/cons_tres/gres_select_filter.c:					error("Not enough shared gres on required sockets to satisfy allocated restricted gpu cores for job %u on node %d",
src/plugins/select/cons_tres/gres_select_filter.c:		 * least one gpu on the node anyway.
src/plugins/select/cons_tres/gres_select_filter.c:		 * For example --gpus=typeA:2,typeB:1 where there is only one
src/plugins/select/cons_tres/gres_select_filter.c:		 * is not a heterogenous job a typeB gpu does have to be
src/plugins/select/cons_tres/gres_select_filter.c:	uint32_t res_cores_per_gpu =
src/plugins/select/cons_tres/gres_select_filter.c:		node_record_table_ptr[args->node_inx]->res_cores_per_gpu;
src/plugins/select/cons_tres/gres_select_filter.c:	if (!res_cores_per_gpu)
src/plugins/select/cons_tres/gres_select_filter.c:			    bit_test(gres_js->res_gpu_cores[args->node_inx], j))
src/plugins/select/cons_tres/gres_select_filter.c:				res_cores_per_gpu);
src/plugins/select/cons_tres/gres_select_filter.c:			error("Restricted gpu cores on multiple sockets which requires MULTIPLE_SHARING_GRES_PJ to be set");
src/plugins/select/cons_tres/gres_select_filter.c:		 * Having multiple gpu types on the same socket could result it
src/plugins/select/cons_tres/gres_select_filter.c:		 * picking the wrong gpu type here if the job request non-typed
src/plugins/select/cons_tres/gres_select_filter.c:		 * gpus.
src/plugins/select/cons_tres/gres_select_filter.c:			error("%s: More restricted gpu cores allocated then should be possible for job %u on node %d",
src/plugins/select/cons_tres/gres_select_filter.c:	if (gres_js->res_gpu_cores && gres_js->res_gpu_cores[node_inx]) {
src/plugins/select/cons_tres/gres_select_filter.h: * OUT avail_gpus - Count of available GPUs on this node
src/plugins/select/cons_tres/gres_select_filter.h: * OUT near_gpus - Count of GPUs available on sockets with available CPUs
src/plugins/select/cons_tres/gres_select_filter.h:					      uint16_t *avail_gpus,
src/plugins/select/cons_tres/gres_select_filter.h:					      uint16_t *near_gpus);
src/plugins/select/cons_tres/job_test.h:extern uint64_t def_cpu_per_gpu;
src/plugins/select/cons_tres/job_test.h:extern uint64_t def_mem_per_gpu;
src/plugins/select/cons_tres/select_cons_tres.c:	def_cpu_per_gpu = 0;
src/plugins/select/cons_tres/select_cons_tres.c:	def_mem_per_gpu = 0;
src/plugins/select/cons_tres/select_cons_tres.c:		def_cpu_per_gpu = cons_helpers_get_def_cpu_per_gpu(
src/plugins/select/cons_tres/select_cons_tres.c:		def_mem_per_gpu = cons_helpers_get_def_mem_per_gpu(
src/plugins/select/cons_tres/gres_sock_list.h: * IN gpu_spec_bitmap - bitmap of reserved gpu cores
src/plugins/select/cons_tres/gres_sock_list.h: * IN res_cores_per_gpu - number of cores reserved for each gpu
src/plugins/select/cons_tres/gres_sock_list.h:	const uint32_t node_inx, bitstr_t *gpu_spec_bitmap,
src/plugins/select/cons_tres/gres_sock_list.h:	uint32_t res_cores_per_gpu, uint16_t cr_type);
src/plugins/select/cons_tres/cons_helpers.h: * Get configured DefCpuPerGPU information from a list
src/plugins/select/cons_tres/cons_helpers.h:extern uint64_t cons_helpers_get_def_cpu_per_gpu(list_t *job_defaults_list);
src/plugins/select/cons_tres/cons_helpers.h: * Get configured DefMemPerGPU information from a list
src/plugins/select/cons_tres/cons_helpers.h:extern uint64_t cons_helpers_get_def_mem_per_gpu(list_t *job_defaults_list);
src/plugins/select/cons_tres/gres_select_util.c: *		  gres_name="gpu" would apply to "gpu:tesla", "gpu:volta", etc.)
src/plugins/select/cons_tres/gres_select_util.c: * IN cpu_per_gpu - value to set as default
src/plugins/select/cons_tres/gres_select_util.c: * IN mem_per_gpu - value to set as default
src/plugins/select/cons_tres/gres_select_util.c: * IN/OUT *cpus_per_task - Increased if cpu_per_gpu * gres_per_task is more than
src/plugins/select/cons_tres/gres_select_util.c:					  uint64_t cpu_per_gpu,
src/plugins/select/cons_tres/gres_select_util.c:					  uint64_t mem_per_gpu,
src/plugins/select/cons_tres/gres_select_util.c:	 * Currently only GPU supported, check how cpus_per_tres/mem_per_tres
src/plugins/select/cons_tres/gres_select_util.c:	xassert(!xstrcmp(gres_name, "gpu"));
src/plugins/select/cons_tres/gres_select_util.c:		gres_js->def_cpus_per_gres = cpu_per_gpu;
src/plugins/select/cons_tres/gres_select_util.c:		gres_js->def_mem_per_gres = mem_per_gpu;
src/plugins/select/cons_tres/gres_select_util.c:			if (cpu_per_gpu)
src/plugins/select/cons_tres/gres_select_util.c:				xstrfmtcat(*cpus_per_tres, "gpu:%"PRIu64,
src/plugins/select/cons_tres/gres_select_util.c:					   cpu_per_gpu);
src/plugins/select/cons_tres/gres_select_util.c:			if (mem_per_gpu)
src/plugins/select/cons_tres/gres_select_util.c:				xstrfmtcat(*mem_per_tres, "gpu:%"PRIu64,
src/plugins/select/cons_tres/gres_select_util.c:					   mem_per_gpu);
src/plugins/select/cons_tres/gres_select_util.c:		if (cpu_per_gpu && gres_js->gres_per_task) {
src/plugins/select/cons_tres/gres_select_util.c:					      cpu_per_gpu));
src/plugins/select/cons_tres/gres_select_util.c: * ntasks_per_tres IN - # of tasks per GPU
src/plugins/select/cons_tres/gres_select_util.c:		 * is --mem-per-gpu adding another option will require change
src/plugins/select/cons_tres/job_test.c:uint64_t def_cpu_per_gpu = 0;
src/plugins/select/cons_tres/job_test.c:uint64_t def_mem_per_gpu = 0;
src/plugins/select/cons_tres/job_test.c:static void _set_gpu_defaults(job_record_t *job_ptr)
src/plugins/select/cons_tres/job_test.c:	static uint64_t last_cpu_per_gpu = NO_VAL64;
src/plugins/select/cons_tres/job_test.c:	static uint64_t last_mem_per_gpu = NO_VAL64;
src/plugins/select/cons_tres/job_test.c:	uint64_t cpu_per_gpu, mem_per_gpu;
src/plugins/select/cons_tres/job_test.c:		last_cpu_per_gpu = cons_helpers_get_def_cpu_per_gpu(
src/plugins/select/cons_tres/job_test.c:		last_mem_per_gpu = cons_helpers_get_def_mem_per_gpu(
src/plugins/select/cons_tres/job_test.c:	if ((last_cpu_per_gpu != NO_VAL64) &&
src/plugins/select/cons_tres/job_test.c:		cpu_per_gpu = last_cpu_per_gpu;
src/plugins/select/cons_tres/job_test.c:	else if ((def_cpu_per_gpu != NO_VAL64) &&
src/plugins/select/cons_tres/job_test.c:		cpu_per_gpu = def_cpu_per_gpu;
src/plugins/select/cons_tres/job_test.c:		cpu_per_gpu = 0;
src/plugins/select/cons_tres/job_test.c:	if (last_mem_per_gpu != NO_VAL64)
src/plugins/select/cons_tres/job_test.c:		mem_per_gpu = last_mem_per_gpu;
src/plugins/select/cons_tres/job_test.c:	else if (def_mem_per_gpu != NO_VAL64)
src/plugins/select/cons_tres/job_test.c:		mem_per_gpu = def_mem_per_gpu;
src/plugins/select/cons_tres/job_test.c:		mem_per_gpu = 0;
src/plugins/select/cons_tres/job_test.c:	gres_select_util_job_set_defs(job_ptr->gres_list_req, "gpu",
src/plugins/select/cons_tres/job_test.c:				      cpu_per_gpu, mem_per_gpu,
src/plugins/select/cons_tres/job_test.c:					node_ptr->gpu_spec_bitmap,
src/plugins/select/cons_tres/job_test.c:					node_ptr->res_cores_per_gpu,
src/plugins/select/cons_tres/job_test.c:		uint16_t near_gpu_cnt = 0;
src/plugins/select/cons_tres/job_test.c:			&avail_res->avail_gpus, &near_gpu_cnt);
src/plugins/select/cons_tres/job_test.c:		/* Favor nodes with more co-located GPUs */
src/plugins/select/cons_tres/job_test.c:			(0xff - near_gpu_cnt);
src/plugins/select/cons_tres/job_test.c:	avail_res->avail_res_cnt = cpus + avail_res->avail_gpus;
src/plugins/select/cons_tres/job_test.c:	_set_gpu_defaults(job_ptr);
src/plugins/select/cons_tres/job_test.c:					   details_ptr->ntasks_per_tres, "gpu",
src/plugins/select/cons_tres/job_test.c: * 0x00000000000## - Reserved for cons_tres, favor nodes with co-located CPU/GPU
src/plugins/select/cons_tres/gres_sock_list.c:	bitstr_t *gpu_spec_bitmap;
src/plugins/select/cons_tres/gres_sock_list.c:	uint32_t res_cores_per_gpu;
src/plugins/select/cons_tres/gres_sock_list.c:} foreach_res_gpu_t;
src/plugins/select/cons_tres/gres_sock_list.c:	uint16_t sockets, uint16_t cores_per_sock, uint32_t res_cores_per_gpu,
src/plugins/select/cons_tres/gres_sock_list.c:		    !res_cores_per_gpu) {
src/plugins/select/cons_tres/gres_sock_list.c:static bool _pick_core_group(bitstr_t *gpu_res_core_bitmap,
src/plugins/select/cons_tres/gres_sock_list.c:		if (!bit_test(gpu_res_core_bitmap, cur_inx) ||
src/plugins/select/cons_tres/gres_sock_list.c: * Reduce the number of restricted cores to just that of the gpu type requested
src/plugins/select/cons_tres/gres_sock_list.c:				   bitstr_t *gpu_spec_cpy,
src/plugins/select/cons_tres/gres_sock_list.c:				   uint32_t res_cores_per_gpu,
src/plugins/select/cons_tres/gres_sock_list.c:	int *picked_cores = xcalloc(res_cores_per_gpu, sizeof(int));
src/plugins/select/cons_tres/gres_sock_list.c:	if (!gres_js->res_gpu_cores) {
src/plugins/select/cons_tres/gres_sock_list.c:		gres_js->res_gpu_cores = xcalloc(gres_js->res_array_size,
src/plugins/select/cons_tres/gres_sock_list.c:	gres_js->res_gpu_cores[node_i] = bit_alloc(bit_size(core_bitmap));
src/plugins/select/cons_tres/gres_sock_list.c:				 * Need to pick in groups of res_cores_per_gpu
src/plugins/select/cons_tres/gres_sock_list.c:				 * since not every gpu job will use all the
src/plugins/select/cons_tres/gres_sock_list.c:					    res_cores_per_gpu,
src/plugins/select/cons_tres/gres_sock_list.c:				c = picked_cores[res_cores_per_gpu - 1] -
src/plugins/select/cons_tres/gres_sock_list.c:				for (int j = 0; j < res_cores_per_gpu; j++) {
src/plugins/select/cons_tres/gres_sock_list.c:					bit_set(gpu_spec_cpy, picked_cores[j]);
src/plugins/select/cons_tres/gres_sock_list.c:					bit_set(gres_js->res_gpu_cores[node_i],
src/plugins/select/cons_tres/gres_sock_list.c:static int _foreach_restricted_gpu(void *x, void *arg)
src/plugins/select/cons_tres/gres_sock_list.c:	foreach_res_gpu_t *args = arg;
src/plugins/select/cons_tres/gres_sock_list.c:	if ((gres_state_job->plugin_id != gres_get_gpu_plugin_id()) ||
src/plugins/select/cons_tres/gres_sock_list.c:	    !args->res_cores_per_gpu)
src/plugins/select/cons_tres/gres_sock_list.c:	_pick_restricted_cores(args->core_bitmap, args->gpu_spec_bitmap,
src/plugins/select/cons_tres/gres_sock_list.c:			       args->res_cores_per_gpu, args->sockets,
src/plugins/select/cons_tres/gres_sock_list.c:	const uint32_t node_inx, bitstr_t *gpu_spec_bitmap,
src/plugins/select/cons_tres/gres_sock_list.c:	uint32_t res_cores_per_gpu)
src/plugins/select/cons_tres/gres_sock_list.c:	bitstr_t *gpu_spec_cpy;
src/plugins/select/cons_tres/gres_sock_list.c:	uint32_t gpu_plugin_id = gres_get_gpu_plugin_id();
src/plugins/select/cons_tres/gres_sock_list.c:	foreach_res_gpu_t args = {
src/plugins/select/cons_tres/gres_sock_list.c:		.res_cores_per_gpu = res_cores_per_gpu,
src/plugins/select/cons_tres/gres_sock_list.c:	if (!gpu_spec_bitmap || !core_bitmap ||
src/plugins/select/cons_tres/gres_sock_list.c:					  &gpu_plugin_id);
src/plugins/select/cons_tres/gres_sock_list.c:	gpu_spec_cpy = bit_copy(gpu_spec_bitmap);
src/plugins/select/cons_tres/gres_sock_list.c:	args.gpu_spec_bitmap = gpu_spec_cpy;
src/plugins/select/cons_tres/gres_sock_list.c:	list_for_each(job_gres_list, _foreach_restricted_gpu, &args);
src/plugins/select/cons_tres/gres_sock_list.c:	bit_and(core_bitmap, gpu_spec_cpy);
src/plugins/select/cons_tres/gres_sock_list.c:	bit_free(gpu_spec_cpy);
src/plugins/select/cons_tres/gres_sock_list.c:	const uint32_t node_inx, bitstr_t *gpu_spec_bitmap,
src/plugins/select/cons_tres/gres_sock_list.c:	uint32_t res_cores_per_gpu, uint16_t cr_type)
src/plugins/select/cons_tres/gres_sock_list.c:		if (gpu_spec_bitmap && core_bitmap)
src/plugins/select/cons_tres/gres_sock_list.c:			bit_and(core_bitmap, gpu_spec_bitmap);
src/plugins/select/cons_tres/gres_sock_list.c:					   node_inx, gpu_spec_bitmap,
src/plugins/select/cons_tres/gres_sock_list.c:					   res_cores_per_gpu);
src/plugins/select/cons_tres/gres_sock_list.c:				res_cores_per_gpu, job_id, node_name,
src/plugins/node_features/helpers/node_features_helpers.c: *  Copyright (C) 2021 NVIDIA CORPORATION. All rights reserved.
src/plugins/node_features/helpers/node_features_helpers.c: *  Written by NVIDIA CORPORATION.
src/plugins/acct_gather_energy/Makefile.in:SUBDIRS = gpu ibmaem ipmi pm_counters rapl xcc
src/plugins/acct_gather_energy/Makefile.am:SUBDIRS = gpu ibmaem ipmi pm_counters rapl xcc
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c: *  acct_gather_energy_gpu.c - slurm energy accounting plugin for GPUs.
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:#include "src/interfaces/gpu.h"
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:#define DEFAULT_GPU_TIMEOUT 10
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:#define DEFAULT_GPU_FREQ 30
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:const char plugin_name[] = "AcctGatherEnergy gpu plugin";
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:const char plugin_type[] = "acct_gather_energy/gpu";
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:// copy of usable gpus and is only used by stepd for a job
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:static bitstr_t	*saved_usable_gpus = NULL;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:static gpu_status_t *gpus = NULL;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:static uint16_t gpus_len = 0;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:static pthread_mutex_t gpu_mutex = PTHREAD_MUTEX_INITIALIZER;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:static pthread_cond_t gpu_cond = PTHREAD_COND_INITIALIZER;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:pthread_t thread_gpu_id_launcher = 0;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:pthread_t thread_gpu_id_run = 0;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	uint64_t data[gpus_len];
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	time_t last_time = gpus[gpus_len - 1].last_update_time;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		acct_gather_profile_dataset_t dataset[gpus_len + 1];
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		for (i = 0; i < gpus_len; i++) {
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			dataset[i].name = xstrdup_printf("GPU%dPower", i);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		for (i = 0; i < gpus_len; i++)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	/* pack an array of uint64_t with current power of gpus */
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	for (i = 0; i < gpus_len; i++) {
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		data[i] = gpus[i].energy.current_watts;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		last_time = gpus[i].energy.poll_time;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		for (i = 0; i < gpus_len; i++) {
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			info("PROFILE-Energy: GPU%dPower=%"PRIu64"",
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:/* updates the given energy according to the last watts reading of the gpu
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c: * gpu		(IN/OUT) A pointer to gpu_status_t structure
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:static void _update_energy(gpu_status_t *gpu, uint32_t readings)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	acct_gather_energy_t *e = &gpu->energy;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		e->current_watts = gpu->last_update_watt;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		if (gpu->previous_update_time == 0)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:					gpu->previous_update_time,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:					gpu->last_update_time,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		e->current_watts = gpu->last_update_watt;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c: * _thread_update_node_energy calls _read_gpu_values and updates all values
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	for (i = 0; i < gpus_len; i++) {
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		rc = gpu_g_energy_read(i, &gpus[i]);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			_update_energy(&gpus[i], readings);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		for (i = 0; i < gpus_len; i++)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			info("gpu-thread: gpu %u current_watts: %u, consumed %"PRIu64" Joules %"PRIu64" new, ave watts %u",
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			     gpus[i].energy.current_watts,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			     gpus[i].energy.consumed_energy,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			     gpus[i].energy.base_consumed_energy,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			     gpus[i].energy.ave_watts);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c: * _thread_init initializes values and conf for the gpu thread
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	if (gpus_len && gpus) {
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		error("%s thread init failed, no GPU available", plugin_name);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c: * _thread_gpu_run is the thread calling gpu periodically
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c: * and read the energy values from the AMD GPUs
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:static void *_thread_gpu_run(void *no_data)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	log_flag(ENERGY, "gpu-thread: launched");
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		log_flag(ENERGY, "gpu-thread: aborted");
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		abs.tv_sec += DEFAULT_GPU_FREQ;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_cond_timedwait(&gpu_cond, &gpu_mutex, &abs);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	log_flag(ENERGY, "gpu-thread: ended");
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c: * _thread_launcher is the thread that launches gpu thread
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	slurm_thread_create(&thread_gpu_id_run, _thread_gpu_run, NULL);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	abs.tv_sec = tvnow.tv_sec + DEFAULT_GPU_TIMEOUT;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		 * It is a known thing we can hang up on GPU calls cancel if
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		pthread_cancel(thread_gpu_id_run);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			int gpu_num)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	 * the gpus
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	log_flag(ENERGY, "gpu: %d, current_watts: %u, consumed %"PRIu64" Joules %"PRIu64" new, ave watts %u",
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		 gpu_num, energy_new->current_watts,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	 * If saved_usable_gpus doesn't exist it means we don't have any gpus to
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	if (!saved_usable_gpus)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	// Check if GPUs are constrained by cgroups
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	// If both of these are true, then GPUs will be constrained
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	// sum the energy of all gpus for this job
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	for (i = 0; i < gpus_len; i++) {
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		if (cgroups_active && !bit_test(saved_usable_gpus, i)) {
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			log_flag(ENERGY, "Passing over gpu %u", i);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		_add_energy(energy, &gpus[i].energy, i);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	// sum the energy of all gpus for this node
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	for (i = 0; i < gpus_len; i++)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		_add_energy(energy, &gpus[i].energy, i);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	/* gpus list */
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	uint16_t gpu_cnt = 0;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	if (slurm_get_node_energy(conf->node_name, context_id, delta, &gpu_cnt,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	/* If there are no gpus then there is no energy to get, just return. */
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	if (!gpu_cnt)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		gpus_len = gpu_cnt;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		gpus = xcalloc(sizeof(gpu_status_t), gpus_len);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		start_current_energies = xcalloc(sizeof(uint64_t), gpus_len);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	if (gpu_cnt != gpus_len) {
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		      __func__, gpu_cnt, gpus_len);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	for (i = 0; i < gpu_cnt; i++) {
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		old = &gpus[i].energy;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	slurm_thread_join(thread_gpu_id_launcher);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	slurm_cond_signal(&gpu_cond);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	slurm_thread_join(thread_gpu_id_run);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	 * xfree(gpus);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	 * saved_usable_gpus = NULL;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:	uint16_t *gpu_cnt = (uint16_t *)data;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			slurm_cond_signal(&gpu_cond);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		if (gpus)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			*last_poll = gpus[gpus_len-1].last_update_time;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		*gpu_cnt = gpus_len;
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		for (i = 0; i < gpus_len; i++)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			memcpy(&energy[i], &gpus[i].energy,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			slurm_cond_signal(&gpu_cond);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		for (i = 0; i < gpus_len; ++i)
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			memcpy(&energy[i], &gpus[i].energy,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_lock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		slurm_mutex_unlock(&gpu_mutex);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		 * Get the GPUs used in the step so we only poll those when
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		rc = gres_get_step_info(step->step_gres_list, "gpu", 0,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:					&saved_usable_gpus);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:		 * If a step isn't using gpus it will return ESLURM_INVALID_GRES
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			log_flag(ENERGY, "usable_gpus = %d of %"PRId64,
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:				 bit_set_count(saved_usable_gpus),
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:				 bit_size(saved_usable_gpus));
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			debug2("Step most likely doesn't have any gpus, no power gathering");
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:				gpu_g_get_device_count(
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:					(unsigned int *) &gpus_len);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:			if (gpus_len) {
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:				gpus = xcalloc(sizeof(gpu_status_t), gpus_len);
src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c:				slurm_thread_create(&thread_gpu_id_launcher,
src/plugins/acct_gather_energy/gpu/Makefile.in:# Makefile for acct_gather_energy/gpu plugin
src/plugins/acct_gather_energy/gpu/Makefile.in:subdir = src/plugins/acct_gather_energy/gpu
src/plugins/acct_gather_energy/gpu/Makefile.in:acct_gather_energy_gpu_la_LIBADD =
src/plugins/acct_gather_energy/gpu/Makefile.in:am_acct_gather_energy_gpu_la_OBJECTS = acct_gather_energy_gpu.lo
src/plugins/acct_gather_energy/gpu/Makefile.in:acct_gather_energy_gpu_la_OBJECTS =  \
src/plugins/acct_gather_energy/gpu/Makefile.in:	$(am_acct_gather_energy_gpu_la_OBJECTS)
src/plugins/acct_gather_energy/gpu/Makefile.in:acct_gather_energy_gpu_la_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC \
src/plugins/acct_gather_energy/gpu/Makefile.in:	$(AM_CFLAGS) $(CFLAGS) $(acct_gather_energy_gpu_la_LDFLAGS) \
src/plugins/acct_gather_energy/gpu/Makefile.in:am__depfiles_remade = ./$(DEPDIR)/acct_gather_energy_gpu.Plo
src/plugins/acct_gather_energy/gpu/Makefile.in:SOURCES = $(acct_gather_energy_gpu_la_SOURCES)
src/plugins/acct_gather_energy/gpu/Makefile.in:pkglib_LTLIBRARIES = acct_gather_energy_gpu.la
src/plugins/acct_gather_energy/gpu/Makefile.in:# AMD gpu energy accounting plugin.
src/plugins/acct_gather_energy/gpu/Makefile.in:acct_gather_energy_gpu_la_SOURCES = acct_gather_energy_gpu.c
src/plugins/acct_gather_energy/gpu/Makefile.in:acct_gather_energy_gpu_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/acct_gather_energy/gpu/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign src/plugins/acct_gather_energy/gpu/Makefile'; \
src/plugins/acct_gather_energy/gpu/Makefile.in:	  $(AUTOMAKE) --foreign src/plugins/acct_gather_energy/gpu/Makefile
src/plugins/acct_gather_energy/gpu/Makefile.in:acct_gather_energy_gpu.la: $(acct_gather_energy_gpu_la_OBJECTS) $(acct_gather_energy_gpu_la_DEPENDENCIES) $(EXTRA_acct_gather_energy_gpu_la_DEPENDENCIES) 
src/plugins/acct_gather_energy/gpu/Makefile.in:	$(AM_V_CCLD)$(acct_gather_energy_gpu_la_LINK) -rpath $(pkglibdir) $(acct_gather_energy_gpu_la_OBJECTS) $(acct_gather_energy_gpu_la_LIBADD) $(LIBS)
src/plugins/acct_gather_energy/gpu/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/acct_gather_energy_gpu.Plo@am__quote@ # am--include-marker
src/plugins/acct_gather_energy/gpu/Makefile.in:		-rm -f ./$(DEPDIR)/acct_gather_energy_gpu.Plo
src/plugins/acct_gather_energy/gpu/Makefile.in:		-rm -f ./$(DEPDIR)/acct_gather_energy_gpu.Plo
src/plugins/acct_gather_energy/gpu/Makefile.am:# Makefile for acct_gather_energy/gpu plugin
src/plugins/acct_gather_energy/gpu/Makefile.am:pkglib_LTLIBRARIES = acct_gather_energy_gpu.la
src/plugins/acct_gather_energy/gpu/Makefile.am:# AMD gpu energy accounting plugin.
src/plugins/acct_gather_energy/gpu/Makefile.am:acct_gather_energy_gpu_la_SOURCES = acct_gather_energy_gpu.c
src/plugins/acct_gather_energy/gpu/Makefile.am:acct_gather_energy_gpu_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/task/cgroup/task_cgroup_devices.c:	 * salloc --gres=gpu must have access to the allocated GPUs. If we do
src/plugins/accounting_storage/mysql/as_mysql_tres.c:				      "(i.e. Gres/GPU).  You gave none",
src/plugins/job_container/tmpfs/job_container_tmpfs.c:		 * switch/nvidia_imex needs to create an ephemeral device
src/plugins/Makefile.in:	cred data_parser gpu gres hash jobacct_gather jobcomp \
src/plugins/Makefile.in:	burst_buffer certmgr cli_filter cred data_parser gpu gres hash \
src/plugins/switch/nvidia_imex/imex_device.c:#define IMEX_DEV_DIR "/dev/nvidia-caps-imex-channels"
src/plugins/switch/nvidia_imex/imex_device.c:#define TARGET_DEV_LINE "nvidia-caps-imex-channels"
src/plugins/switch/nvidia_imex/imex_device.c:		warning("%s: nvidia-caps-imex-channels major device not found, plugin disabled",
src/plugins/switch/nvidia_imex/imex_device.c:		info("nvidia-caps-imex-channels major: %d", device_major);
src/plugins/switch/nvidia_imex/switch_nvidia_imex.c: *  switch_nvidia_imex.c
src/plugins/switch/nvidia_imex/switch_nvidia_imex.c:#include "src/plugins/switch/nvidia_imex/imex_device.h"
src/plugins/switch/nvidia_imex/switch_nvidia_imex.c:const char plugin_name[] = "switch NVIDIA IMEX plugin";
src/plugins/switch/nvidia_imex/switch_nvidia_imex.c:const char plugin_type[] = "switch/nvidia_imex";
src/plugins/switch/nvidia_imex/switch_nvidia_imex.c:const uint32_t plugin_id = SWITCH_PLUGIN_NVIDIA_IMEX;
src/plugins/switch/nvidia_imex/Makefile.in:subdir = src/plugins/switch/nvidia_imex
src/plugins/switch/nvidia_imex/Makefile.in:switch_nvidia_imex_la_LIBADD =
src/plugins/switch/nvidia_imex/Makefile.in:am_switch_nvidia_imex_la_OBJECTS = imex_device.lo \
src/plugins/switch/nvidia_imex/Makefile.in:	switch_nvidia_imex.lo
src/plugins/switch/nvidia_imex/Makefile.in:switch_nvidia_imex_la_OBJECTS = $(am_switch_nvidia_imex_la_OBJECTS)
src/plugins/switch/nvidia_imex/Makefile.in:switch_nvidia_imex_la_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC \
src/plugins/switch/nvidia_imex/Makefile.in:	$(AM_CFLAGS) $(CFLAGS) $(switch_nvidia_imex_la_LDFLAGS) \
src/plugins/switch/nvidia_imex/Makefile.in:	./$(DEPDIR)/switch_nvidia_imex.Plo
src/plugins/switch/nvidia_imex/Makefile.in:SOURCES = $(switch_nvidia_imex_la_SOURCES)
src/plugins/switch/nvidia_imex/Makefile.in:pkglib_LTLIBRARIES = switch_nvidia_imex.la
src/plugins/switch/nvidia_imex/Makefile.in:switch_nvidia_imex_la_SOURCES = \
src/plugins/switch/nvidia_imex/Makefile.in:	switch_nvidia_imex.c
src/plugins/switch/nvidia_imex/Makefile.in:switch_nvidia_imex_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/switch/nvidia_imex/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign src/plugins/switch/nvidia_imex/Makefile'; \
src/plugins/switch/nvidia_imex/Makefile.in:	  $(AUTOMAKE) --foreign src/plugins/switch/nvidia_imex/Makefile
src/plugins/switch/nvidia_imex/Makefile.in:switch_nvidia_imex.la: $(switch_nvidia_imex_la_OBJECTS) $(switch_nvidia_imex_la_DEPENDENCIES) $(EXTRA_switch_nvidia_imex_la_DEPENDENCIES) 
src/plugins/switch/nvidia_imex/Makefile.in:	$(AM_V_CCLD)$(switch_nvidia_imex_la_LINK) -rpath $(pkglibdir) $(switch_nvidia_imex_la_OBJECTS) $(switch_nvidia_imex_la_LIBADD) $(LIBS)
src/plugins/switch/nvidia_imex/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/switch_nvidia_imex.Plo@am__quote@ # am--include-marker
src/plugins/switch/nvidia_imex/Makefile.in:	-rm -f ./$(DEPDIR)/switch_nvidia_imex.Plo
src/plugins/switch/nvidia_imex/Makefile.in:	-rm -f ./$(DEPDIR)/switch_nvidia_imex.Plo
src/plugins/switch/nvidia_imex/Makefile.am:pkglib_LTLIBRARIES = switch_nvidia_imex.la
src/plugins/switch/nvidia_imex/Makefile.am:switch_nvidia_imex_la_SOURCES = \
src/plugins/switch/nvidia_imex/Makefile.am:	switch_nvidia_imex.c
src/plugins/switch/nvidia_imex/Makefile.am:switch_nvidia_imex_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/switch/Makefile.in:@LINUX_BUILD_TRUE@am__append_1 = nvidia_imex
src/plugins/switch/Makefile.in:DIST_SUBDIRS = nvidia_imex hpe_slingshot
src/plugins/switch/Makefile.am:SUBDIRS += nvidia_imex
src/plugins/Makefile.am:	gpu			\
src/plugins/gpu/rsmi/gpu_rsmi.c: *  gpu_rsmi.c - Support rsmi interface to an AMD GPU.
src/plugins/gpu/rsmi/gpu_rsmi.c: *  who borrowed heavily from SLURM gpu and nvml plugin.
src/plugins/gpu/rsmi/gpu_rsmi.c:#include <rocm_smi/rocm_smi.h>
src/plugins/gpu/rsmi/gpu_rsmi.c:#include "../common/gpu_common.h"
src/plugins/gpu/rsmi/gpu_rsmi.c:static bitstr_t	*saved_gpus;
src/plugins/gpu/rsmi/gpu_rsmi.c:/* ROCM release version >= 6.0.0 required for gathering usage */
src/plugins/gpu/rsmi/gpu_rsmi.c: * PCI information about a GPU device.
src/plugins/gpu/rsmi/gpu_rsmi.c:const char plugin_name[] = "GPU RSMI plugin";
src/plugins/gpu/rsmi/gpu_rsmi.c:const char	plugin_type[]		= "gpu/rsmi";
src/plugins/gpu/rsmi/gpu_rsmi.c:static int gpumem_pos = -1;
src/plugins/gpu/rsmi/gpu_rsmi.c:static int gpuutil_pos = -1;
src/plugins/gpu/rsmi/gpu_rsmi.c:		 * false, so we won't set gpumem_pos and gpuutil_pos which
src/plugins/gpu/rsmi/gpu_rsmi.c:		 * effectively disables gpu accounting.
src/plugins/gpu/rsmi/gpu_rsmi.c:			gpu_get_tres_pos(&gpumem_pos, &gpuutil_pos);
src/plugins/gpu/rsmi/gpu_rsmi.c:	rsmi_rc = rsmi_dev_gpu_clk_freq_get(
src/plugins/gpu/rsmi/gpu_rsmi.c:	debug3("rsmi_dev_gpu_clk_freq_get() took %ld microseconds",
src/plugins/gpu/rsmi/gpu_rsmi.c:	rsmi_rc = rsmi_dev_gpu_clk_freq_get(
src/plugins/gpu/rsmi/gpu_rsmi.c:	debug3("rsmi_dev_gpu_clk_freq_get() took %ld microseconds",
src/plugins/gpu/rsmi/gpu_rsmi.c:	gpu_common_print_freqs(mem_freqs, size, l, "GPU Memory", 0);
src/plugins/gpu/rsmi/gpu_rsmi.c:	gpu_common_print_freqs(gfx_freqs, size, l, "GPU Graphics", 0);
src/plugins/gpu/rsmi/gpu_rsmi.c:	gpu_common_get_nearest_freq(mem_freq, mem_freqs_size, mem_freqs_sort);
src/plugins/gpu/rsmi/gpu_rsmi.c:	gpu_common_get_nearest_freq(gfx_freq, gfx_freqs_size, gfx_freqs_sort);
src/plugins/gpu/rsmi/gpu_rsmi.c: * Set the memory and graphics clock frequencies for the GPU
src/plugins/gpu/rsmi/gpu_rsmi.c:	rsmi_rc = rsmi_dev_gpu_clk_freq_set(
src/plugins/gpu/rsmi/gpu_rsmi.c:	debug3("rsmi_dev_gpu_clk_freq_set(0x%lx) for memory took %ld microseconds",
src/plugins/gpu/rsmi/gpu_rsmi.c:		error("RSMI: Failed to set memory frequency GPU %u error: %s",
src/plugins/gpu/rsmi/gpu_rsmi.c:	rsmi_rc = rsmi_dev_gpu_clk_freq_set(dv_ind,
src/plugins/gpu/rsmi/gpu_rsmi.c:	debug3("rsmi_dev_gpu_clk_freq_set(0x%lx) for graphics took %ld microseconds",
src/plugins/gpu/rsmi/gpu_rsmi.c:		error("RSMI: Failed to set graphic frequency GPU %u error: %s",
src/plugins/gpu/rsmi/gpu_rsmi.c: * Reset the memory and graphics clock frequencies for the GPU to the same
src/plugins/gpu/rsmi/gpu_rsmi.c: * Get the memory or graphics clock frequency that the GPU is currently running
src/plugins/gpu/rsmi/gpu_rsmi.c:	rsmi_rc = rsmi_dev_gpu_clk_freq_get(dv_ind, type, &rsmi_freqs);
src/plugins/gpu/rsmi/gpu_rsmi.c:	debug3("rsmi_dev_gpu_clk_freq_get(%s) took %ld microseconds",
src/plugins/gpu/rsmi/gpu_rsmi.c:		error("RSMI: Failed to get the GPU frequency type %s, error: %s",
src/plugins/gpu/rsmi/gpu_rsmi.c: * Reset the frequencies of each GPU in the step to the hardware default
src/plugins/gpu/rsmi/gpu_rsmi.c: * gpus		(IN) A bitmap specifying the GPUs on which to operate.
src/plugins/gpu/rsmi/gpu_rsmi.c:static void _reset_freq(bitstr_t *gpus)
src/plugins/gpu/rsmi/gpu_rsmi.c:	int gpu_len = bit_size(gpus);
src/plugins/gpu/rsmi/gpu_rsmi.c:	for (i = 0; i < gpu_len; i++) {
src/plugins/gpu/rsmi/gpu_rsmi.c:		if (!bit_test(gpus, i))
src/plugins/gpu/rsmi/gpu_rsmi.c:			log_flag(GRES, "Successfully reset GPU[%d]", i);
src/plugins/gpu/rsmi/gpu_rsmi.c:			log_flag(GRES, "Failed to reset GPU[%d]", i);
src/plugins/gpu/rsmi/gpu_rsmi.c:		log_flag(GRES, "%s: Could not reset frequencies for all GPUs %d/%d total GPUs",
src/plugins/gpu/rsmi/gpu_rsmi.c:		fprintf(stderr, "Could not reset frequencies for all GPUs %d/%d total GPUs\n",
src/plugins/gpu/rsmi/gpu_rsmi.c: * Set the frequencies of each GPU specified for the step
src/plugins/gpu/rsmi/gpu_rsmi.c: * gpus		(IN) A bitmap specifying the GPUs on which to operate.
src/plugins/gpu/rsmi/gpu_rsmi.c: * gpu_freq	(IN) The frequencies to set each of the GPUs to. If a NULL or
src/plugins/gpu/rsmi/gpu_rsmi.c: *		empty memory or graphics frequency is specified, then GpuFreqDef
src/plugins/gpu/rsmi/gpu_rsmi.c:static void _set_freq(bitstr_t *gpus, char *gpu_freq)
src/plugins/gpu/rsmi/gpu_rsmi.c:	int gpu_len = 0;
src/plugins/gpu/rsmi/gpu_rsmi.c:	unsigned int gpu_freq_num = 0, mem_freq_num = 0;
src/plugins/gpu/rsmi/gpu_rsmi.c:	debug2("_parse_gpu_freq(%s)", gpu_freq);
src/plugins/gpu/rsmi/gpu_rsmi.c:	gpu_common_parse_gpu_freq(gpu_freq, &gpu_freq_num, &mem_freq_num,
src/plugins/gpu/rsmi/gpu_rsmi.c:	tmp = gpu_common_freq_value_to_string(mem_freq_num);
src/plugins/gpu/rsmi/gpu_rsmi.c:	debug2("Requested GPU memory frequency: %s", tmp);
src/plugins/gpu/rsmi/gpu_rsmi.c:	tmp = gpu_common_freq_value_to_string(gpu_freq_num);
src/plugins/gpu/rsmi/gpu_rsmi.c:	debug2("Requested GPU graphics frequency: %s", tmp);
src/plugins/gpu/rsmi/gpu_rsmi.c:	if (!mem_freq_num && !gpu_freq_num) {
src/plugins/gpu/rsmi/gpu_rsmi.c:	// Check if GPUs are constrained by cgroups
src/plugins/gpu/rsmi/gpu_rsmi.c:	// If both of these are true, then GPUs will be constrained
src/plugins/gpu/rsmi/gpu_rsmi.c:		gpu_len = bit_set_count(gpus);
src/plugins/gpu/rsmi/gpu_rsmi.c:		debug2("%s: cgroups are configured. Using LOCAL GPU IDs",
src/plugins/gpu/rsmi/gpu_rsmi.c:		gpu_len = bit_size(gpus);
src/plugins/gpu/rsmi/gpu_rsmi.c:		debug2("%s: cgroups are NOT configured. Assuming GLOBAL GPU IDs",
src/plugins/gpu/rsmi/gpu_rsmi.c:	for (i = 0; i < gpu_len; i++) {
src/plugins/gpu/rsmi/gpu_rsmi.c:		uint64_t mem_bitmask = 0, gpu_bitmask = 0;
src/plugins/gpu/rsmi/gpu_rsmi.c:		unsigned int gpu_freq = gpu_freq_num, mem_freq = mem_freq_num;
src/plugins/gpu/rsmi/gpu_rsmi.c:		// Only check the global GPU bitstring if not using cgroups
src/plugins/gpu/rsmi/gpu_rsmi.c:		if (!cgroups_active && !bit_test(gpus, i)) {
src/plugins/gpu/rsmi/gpu_rsmi.c:					&gpu_freq, &gpu_bitmask);
src/plugins/gpu/rsmi/gpu_rsmi.c:		freq_set = _rsmi_set_freqs(i, mem_bitmask, gpu_bitmask);
src/plugins/gpu/rsmi/gpu_rsmi.c:		if (gpu_freq) {
src/plugins/gpu/rsmi/gpu_rsmi.c:			xstrfmtcat(tmp, "%sgraphics_freq:%u", sep, gpu_freq);
src/plugins/gpu/rsmi/gpu_rsmi.c:			log_flag(GRES, "Successfully set GPU[%d] %s", i, tmp);
src/plugins/gpu/rsmi/gpu_rsmi.c:			log_flag(GRES, "Failed to set GPU[%d] %s", i, tmp);
src/plugins/gpu/rsmi/gpu_rsmi.c:			fprintf(stderr, "GpuFreq=%s\n", tmp);
src/plugins/gpu/rsmi/gpu_rsmi.c:			freq_logged = true;	/* Just log for first GPU */
src/plugins/gpu/rsmi/gpu_rsmi.c:		log_flag(GRES, "%s: Could not set frequencies for all GPUs %d/%d total GPUs",
src/plugins/gpu/rsmi/gpu_rsmi.c:		fprintf(stderr, "Could not set frequencies for all GPUs %d/%d total GPUs\n",
src/plugins/gpu/rsmi/gpu_rsmi.c: * driver	(OUT) A string to return version of AMD GPU driver
src/plugins/gpu/rsmi/gpu_rsmi.c: * len		(OUT) Length for version of AMD GPU driver
src/plugins/gpu/rsmi/gpu_rsmi.c: * Get the version of the ROCM-SMI library
src/plugins/gpu/rsmi/gpu_rsmi.c:			error("%s: GPU usage accounting disabled. RSMI version >= 6.0.0 required.",
src/plugins/gpu/rsmi/gpu_rsmi.c: * Get the total # of GPUs in the system
src/plugins/gpu/rsmi/gpu_rsmi.c: * device_count	(OUT) Number of available GPU devices
src/plugins/gpu/rsmi/gpu_rsmi.c:extern void gpu_p_get_device_count(uint32_t *device_count)
src/plugins/gpu/rsmi/gpu_rsmi.c: * Get the name of the GPU
src/plugins/gpu/rsmi/gpu_rsmi.c: * device_name	(OUT) Name of GPU devices
src/plugins/gpu/rsmi/gpu_rsmi.c:		error("RSMI: Failed to get name of the GPU: %s", status_string);
src/plugins/gpu/rsmi/gpu_rsmi.c:	gpu_common_underscorify_tolower(device_name);
src/plugins/gpu/rsmi/gpu_rsmi.c: * Get the brand of the GPU
src/plugins/gpu/rsmi/gpu_rsmi.c: * device_brand	(OUT) Brand of GPU devices
src/plugins/gpu/rsmi/gpu_rsmi.c:		error("RSMI: Failed to get brand of the GPU: %s",
src/plugins/gpu/rsmi/gpu_rsmi.c: * Retrieves minor number of the render device. Each AMD GPU will have a device node file
src/plugins/gpu/rsmi/gpu_rsmi.c:		error("RSMI: Failed to get minor number of GPU: %s",
src/plugins/gpu/rsmi/gpu_rsmi.c: * Get the PCI Info of the GPU
src/plugins/gpu/rsmi/gpu_rsmi.c: * pci			(OUT) PCI Info of GPU devices
src/plugins/gpu/rsmi/gpu_rsmi.c:		error("RSMI: Failed to get PCI Info of the GPU: %s",
src/plugins/gpu/rsmi/gpu_rsmi.c: * Get the Unique ID of the GPU
src/plugins/gpu/rsmi/gpu_rsmi.c: * id			(OUT) Unique ID of GPU devices
src/plugins/gpu/rsmi/gpu_rsmi.c:		error("RSMI: Failed to get Unique ID of the GPU: %s",
src/plugins/gpu/rsmi/gpu_rsmi.c:		error("RSMI: Failed to get numa affinity of the GPU: %s",
src/plugins/gpu/rsmi/gpu_rsmi.c: * Creates and returns a gres conf list of detected AMD gpus on the node.
src/plugins/gpu/rsmi/gpu_rsmi.c: * If the AMD ROCM-SMI API exists, then query GPU info,
src/plugins/gpu/rsmi/gpu_rsmi.c:static list_t *_get_system_gpu_list_rsmi(node_config_load_t *node_config)
src/plugins/gpu/rsmi/gpu_rsmi.c:	gpu_p_get_device_count(&device_count);
src/plugins/gpu/rsmi/gpu_rsmi.c:	// Loop through all the GPUs on the system and add to gres_list_system
src/plugins/gpu/rsmi/gpu_rsmi.c:			.name = "gpu",
src/plugins/gpu/rsmi/gpu_rsmi.c:		debug2("GPU index %u:", i);
src/plugins/gpu/rsmi/gpu_rsmi.c:			debug("Note: GPU index %u is different from minor # %u",
src/plugins/gpu/rsmi/gpu_rsmi.c:	info("%u GPU system device(s) detected", device_count);
src/plugins/gpu/rsmi/gpu_rsmi.c:extern list_t *gpu_p_get_system_gpu_list(node_config_load_t *node_config)
src/plugins/gpu/rsmi/gpu_rsmi.c:	list_t *gres_list_system = _get_system_gpu_list_rsmi(node_config);
src/plugins/gpu/rsmi/gpu_rsmi.c:		error("System GPU detection failed");
src/plugins/gpu/rsmi/gpu_rsmi.c:extern void gpu_p_step_hardware_init(bitstr_t *usable_gpus, char *tres_freq)
src/plugins/gpu/rsmi/gpu_rsmi.c:	xassert(usable_gpus);
src/plugins/gpu/rsmi/gpu_rsmi.c:	if (!usable_gpus)
src/plugins/gpu/rsmi/gpu_rsmi.c:		return;		/* Job allocated no GPUs */
src/plugins/gpu/rsmi/gpu_rsmi.c:	tmp = strstr(tres_freq, "gpu:");
src/plugins/gpu/rsmi/gpu_rsmi.c:		return;		/* No GPU frequency spec */
src/plugins/gpu/rsmi/gpu_rsmi.c:	// Save a copy of the GPUs affected, so we can reset things afterwards
src/plugins/gpu/rsmi/gpu_rsmi.c:	FREE_NULL_BITMAP(saved_gpus);
src/plugins/gpu/rsmi/gpu_rsmi.c:	saved_gpus = bit_copy(usable_gpus);
src/plugins/gpu/rsmi/gpu_rsmi.c:	// Set the frequency of each GPU index specified in the bitstr
src/plugins/gpu/rsmi/gpu_rsmi.c:	_set_freq(usable_gpus, freq);
src/plugins/gpu/rsmi/gpu_rsmi.c:extern void gpu_p_step_hardware_fini(void)
src/plugins/gpu/rsmi/gpu_rsmi.c:	if (!saved_gpus)
src/plugins/gpu/rsmi/gpu_rsmi.c:	_reset_freq(saved_gpus);
src/plugins/gpu/rsmi/gpu_rsmi.c:	FREE_NULL_BITMAP(saved_gpus);
src/plugins/gpu/rsmi/gpu_rsmi.c:extern char *gpu_p_test_cpu_conv(char *cpu_range)
src/plugins/gpu/rsmi/gpu_rsmi.c: * gpu_p_energy_read read current average watts and update last_update_watt
src/plugins/gpu/rsmi/gpu_rsmi.c: * energy         (IN) A pointer to gpu_status_t structure
src/plugins/gpu/rsmi/gpu_rsmi.c:extern int gpu_p_energy_read(uint32_t dv_ind, gpu_status_t *gpu)
src/plugins/gpu/rsmi/gpu_rsmi.c:		gpu->energy.current_watts = NO_VAL;
src/plugins/gpu/rsmi/gpu_rsmi.c:	gpu->last_update_watt = curr_milli_watts/1000000;
src/plugins/gpu/rsmi/gpu_rsmi.c:	gpu->previous_update_time = gpu->last_update_time;
src/plugins/gpu/rsmi/gpu_rsmi.c:	gpu->last_update_time = time(NULL);
src/plugins/gpu/rsmi/gpu_rsmi.c:extern int gpu_p_usage_read(pid_t pid, acct_gather_data_t *data)
src/plugins/gpu/rsmi/gpu_rsmi.c:	bool track_gpumem, track_gpuutil;
src/plugins/gpu/rsmi/gpu_rsmi.c:	track_gpumem = (gpumem_pos != -1);
src/plugins/gpu/rsmi/gpu_rsmi.c:	track_gpuutil = (gpuutil_pos != -1);
src/plugins/gpu/rsmi/gpu_rsmi.c:	if (!track_gpuutil && !track_gpumem) {
src/plugins/gpu/rsmi/gpu_rsmi.c:		debug2("%s: We are not tracking TRES gpuutil/gpumem", __func__);
src/plugins/gpu/rsmi/gpu_rsmi.c:	if (track_gpuutil)
src/plugins/gpu/rsmi/gpu_rsmi.c:		data[gpuutil_pos].size_read = proc.cu_occupancy;
src/plugins/gpu/rsmi/gpu_rsmi.c:	if (track_gpumem)
src/plugins/gpu/rsmi/gpu_rsmi.c:		data[gpumem_pos].size_read = proc.vram_usage;
src/plugins/gpu/rsmi/gpu_rsmi.c:	log_flag(JAG, "pid %d has GPUUtil=%lu and MemMB=%lu",
src/plugins/gpu/rsmi/gpu_rsmi.c:		 data[gpuutil_pos].size_read,
src/plugins/gpu/rsmi/gpu_rsmi.c:		 data[gpumem_pos].size_read / 1048576);
src/plugins/gpu/rsmi/Makefile.in:# Makefile for gpu/rsmi plugin
src/plugins/gpu/rsmi/Makefile.in:subdir = src/plugins/gpu/rsmi
src/plugins/gpu/rsmi/Makefile.in:gpu_rsmi_la_DEPENDENCIES = ../common/libgpu_common.la
src/plugins/gpu/rsmi/Makefile.in:am__objects_1 = gpu_rsmi.lo
src/plugins/gpu/rsmi/Makefile.in:am_gpu_rsmi_la_OBJECTS = $(am__objects_1)
src/plugins/gpu/rsmi/Makefile.in:gpu_rsmi_la_OBJECTS = $(am_gpu_rsmi_la_OBJECTS)
src/plugins/gpu/rsmi/Makefile.in:gpu_rsmi_la_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
src/plugins/gpu/rsmi/Makefile.in:	$(gpu_rsmi_la_LDFLAGS) $(LDFLAGS) -o $@
src/plugins/gpu/rsmi/Makefile.in:am__depfiles_remade = ./$(DEPDIR)/gpu_rsmi.Plo
src/plugins/gpu/rsmi/Makefile.in:SOURCES = $(gpu_rsmi_la_SOURCES)
src/plugins/gpu/rsmi/Makefile.in:RSMI_SOURCES = gpu_rsmi.c
src/plugins/gpu/rsmi/Makefile.in:pkglib_LTLIBRARIES = gpu_rsmi.la
src/plugins/gpu/rsmi/Makefile.in:gpu_rsmi_la_SOURCES = $(RSMI_SOURCES)
src/plugins/gpu/rsmi/Makefile.in:gpu_rsmi_la_LDFLAGS = $(NUMA_LIBS) $(PLUGIN_FLAGS)
src/plugins/gpu/rsmi/Makefile.in:gpu_rsmi_la_LIBADD = ../common/libgpu_common.la
src/plugins/gpu/rsmi/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign src/plugins/gpu/rsmi/Makefile'; \
src/plugins/gpu/rsmi/Makefile.in:	  $(AUTOMAKE) --foreign src/plugins/gpu/rsmi/Makefile
src/plugins/gpu/rsmi/Makefile.in:gpu_rsmi.la: $(gpu_rsmi_la_OBJECTS) $(gpu_rsmi_la_DEPENDENCIES) $(EXTRA_gpu_rsmi_la_DEPENDENCIES) 
src/plugins/gpu/rsmi/Makefile.in:	$(AM_V_CCLD)$(gpu_rsmi_la_LINK) -rpath $(pkglibdir) $(gpu_rsmi_la_OBJECTS) $(gpu_rsmi_la_LIBADD) $(LIBS)
src/plugins/gpu/rsmi/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/gpu_rsmi.Plo@am__quote@ # am--include-marker
src/plugins/gpu/rsmi/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_rsmi.Plo
src/plugins/gpu/rsmi/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_rsmi.Plo
src/plugins/gpu/rsmi/Makefile.am:# Makefile for gpu/rsmi plugin
src/plugins/gpu/rsmi/Makefile.am:RSMI_SOURCES = gpu_rsmi.c
src/plugins/gpu/rsmi/Makefile.am:pkglib_LTLIBRARIES = gpu_rsmi.la
src/plugins/gpu/rsmi/Makefile.am:gpu_rsmi_la_SOURCES = $(RSMI_SOURCES)
src/plugins/gpu/rsmi/Makefile.am:gpu_rsmi_la_LDFLAGS = $(NUMA_LIBS) $(PLUGIN_FLAGS)
src/plugins/gpu/rsmi/Makefile.am:gpu_rsmi_la_LIBADD = ../common/libgpu_common.la
src/plugins/gpu/nvml/Makefile.in:# Makefile for gpu/nvml plugin
src/plugins/gpu/nvml/Makefile.in:subdir = src/plugins/gpu/nvml
src/plugins/gpu/nvml/Makefile.in:gpu_nvml_la_DEPENDENCIES = ../common/libgpu_common.la
src/plugins/gpu/nvml/Makefile.in:am__objects_1 = gpu_nvml.lo
src/plugins/gpu/nvml/Makefile.in:am_gpu_nvml_la_OBJECTS = $(am__objects_1)
src/plugins/gpu/nvml/Makefile.in:gpu_nvml_la_OBJECTS = $(am_gpu_nvml_la_OBJECTS)
src/plugins/gpu/nvml/Makefile.in:gpu_nvml_la_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
src/plugins/gpu/nvml/Makefile.in:	$(gpu_nvml_la_LDFLAGS) $(LDFLAGS) -o $@
src/plugins/gpu/nvml/Makefile.in:am__depfiles_remade = ./$(DEPDIR)/gpu_nvml.Plo
src/plugins/gpu/nvml/Makefile.in:SOURCES = $(gpu_nvml_la_SOURCES)
src/plugins/gpu/nvml/Makefile.in:NVML_SOURCES = gpu_nvml.c
src/plugins/gpu/nvml/Makefile.in:pkglib_LTLIBRARIES = gpu_nvml.la
src/plugins/gpu/nvml/Makefile.in:gpu_nvml_la_SOURCES = $(NVML_SOURCES)
src/plugins/gpu/nvml/Makefile.in:gpu_nvml_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gpu/nvml/Makefile.in:gpu_nvml_la_LIBADD = ../common/libgpu_common.la
src/plugins/gpu/nvml/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign src/plugins/gpu/nvml/Makefile'; \
src/plugins/gpu/nvml/Makefile.in:	  $(AUTOMAKE) --foreign src/plugins/gpu/nvml/Makefile
src/plugins/gpu/nvml/Makefile.in:gpu_nvml.la: $(gpu_nvml_la_OBJECTS) $(gpu_nvml_la_DEPENDENCIES) $(EXTRA_gpu_nvml_la_DEPENDENCIES) 
src/plugins/gpu/nvml/Makefile.in:	$(AM_V_CCLD)$(gpu_nvml_la_LINK) -rpath $(pkglibdir) $(gpu_nvml_la_OBJECTS) $(gpu_nvml_la_LIBADD) $(LIBS)
src/plugins/gpu/nvml/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/gpu_nvml.Plo@am__quote@ # am--include-marker
src/plugins/gpu/nvml/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_nvml.Plo
src/plugins/gpu/nvml/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_nvml.Plo
src/plugins/gpu/nvml/Makefile.am:# Makefile for gpu/nvml plugin
src/plugins/gpu/nvml/Makefile.am:NVML_SOURCES = gpu_nvml.c
src/plugins/gpu/nvml/Makefile.am:pkglib_LTLIBRARIES = gpu_nvml.la
src/plugins/gpu/nvml/Makefile.am:gpu_nvml_la_SOURCES = $(NVML_SOURCES)
src/plugins/gpu/nvml/Makefile.am:gpu_nvml_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gpu/nvml/Makefile.am:gpu_nvml_la_LIBADD = ../common/libgpu_common.la
src/plugins/gpu/nvml/gpu_nvml.c: *  gpu_nvml.c - Support nvml interface to an Nvidia GPU.
src/plugins/gpu/nvml/gpu_nvml.c:#include "../common/gpu_common.h"
src/plugins/gpu/nvml/gpu_nvml.c:	char *files; /* Includes MIG cap files and parent GPU device file */
src/plugins/gpu/nvml/gpu_nvml.c:	char *profile_name; /* <GPU_type>_<slice_cnt>g.<mem>gb */
src/plugins/gpu/nvml/gpu_nvml.c:	/* `MIG-<GPU-UUID>/<GPU instance ID>/<compute instance ID>` */
src/plugins/gpu/nvml/gpu_nvml.c:static bitstr_t	*saved_gpus = NULL;
src/plugins/gpu/nvml/gpu_nvml.c:const char plugin_name[] = "GPU NVML plugin";
src/plugins/gpu/nvml/gpu_nvml.c:const char	plugin_type[]		= "gpu/nvml";
src/plugins/gpu/nvml/gpu_nvml.c:static int gpumem_pos = -1;
src/plugins/gpu/nvml/gpu_nvml.c:static int gpuutil_pos = -1;
src/plugins/gpu/nvml/gpu_nvml.c: * Get the handle to the GPU for the passed index
src/plugins/gpu/nvml/gpu_nvml.c: * index 	(IN) The GPU index (corresponds to PCI Bus ID order)
src/plugins/gpu/nvml/gpu_nvml.c:		error("Failed to get device handle for GPU %d: %s", index,
src/plugins/gpu/nvml/gpu_nvml.c:		      "GPU : %s", __func__, nvmlErrorString(nvml_rc));
src/plugins/gpu/nvml/gpu_nvml.c:		      " GPU at mem frequency %u: %s", __func__, mem_freq,
src/plugins/gpu/nvml/gpu_nvml.c:	gpu_common_print_freqs(gfx_freqs, size, l, "GPU Graphics", 8);
src/plugins/gpu/nvml/gpu_nvml.c:	log_var(l, "Possible GPU Memory Frequencies (%u):", mem_size);
src/plugins/gpu/nvml/gpu_nvml.c: * device		(IN) The NVML GPU device handle
src/plugins/gpu/nvml/gpu_nvml.c:	gpu_common_get_nearest_freq(mem_freq, mem_freqs_size, mem_freqs);
src/plugins/gpu/nvml/gpu_nvml.c:	gpu_common_get_nearest_freq(gfx_freq, gfx_freqs_size, gfx_freqs);
src/plugins/gpu/nvml/gpu_nvml.c: * Set the memory and graphics clock frequencies for the GPU
src/plugins/gpu/nvml/gpu_nvml.c: * device	(IN) The NVML GPU device handle
src/plugins/gpu/nvml/gpu_nvml.c:		      "pair (%u, %u) for the GPU: %s", __func__, mem_freq,
src/plugins/gpu/nvml/gpu_nvml.c: * Reset the memory and graphics clock frequencies for the GPU to the same
src/plugins/gpu/nvml/gpu_nvml.c: * device	(IN) The NVML GPU device handle
src/plugins/gpu/nvml/gpu_nvml.c:		error("%s: Failed to reset GPU frequencies to the hardware default: %s",
src/plugins/gpu/nvml/gpu_nvml.c: * Get the memory or graphics clock frequency that the GPU is currently running
src/plugins/gpu/nvml/gpu_nvml.c: * device	(IN) The NVML GPU device handle
src/plugins/gpu/nvml/gpu_nvml.c:		error("%s: Failed to get the GPU %s frequency: %s", __func__,
src/plugins/gpu/nvml/gpu_nvml.c: * Reset the frequencies of each GPU in the step to the hardware default
src/plugins/gpu/nvml/gpu_nvml.c: * gpus		(IN) A bitmap specifying the GPUs on which to operate.
src/plugins/gpu/nvml/gpu_nvml.c:static void _reset_freq(bitstr_t *gpus)
src/plugins/gpu/nvml/gpu_nvml.c:	int gpu_len = bit_size(gpus);
src/plugins/gpu/nvml/gpu_nvml.c:	for (i = 0; i < gpu_len; i++) {
src/plugins/gpu/nvml/gpu_nvml.c:		if (!bit_test(gpus, i))
src/plugins/gpu/nvml/gpu_nvml.c:			log_flag(GRES, "Successfully reset GPU[%d]", i);
src/plugins/gpu/nvml/gpu_nvml.c:			log_flag(GRES, "Failed to reset GPU[%d]", i);
src/plugins/gpu/nvml/gpu_nvml.c:		log_flag(GRES, "%s: Could not reset frequencies for all GPUs. Set %d/%d total GPUs",
src/plugins/gpu/nvml/gpu_nvml.c:		fprintf(stderr, "Could not reset frequencies for all GPUs. "
src/plugins/gpu/nvml/gpu_nvml.c:			"Set %d/%d total GPUs\n", count_set, count);
src/plugins/gpu/nvml/gpu_nvml.c: * Set the frequencies of each GPU specified for the step
src/plugins/gpu/nvml/gpu_nvml.c: * gpus		(IN) A bitmap specifying the GPUs on which to operate.
src/plugins/gpu/nvml/gpu_nvml.c: * gpu_freq	(IN) The frequencies to set each of the GPUs to. If a NULL or
src/plugins/gpu/nvml/gpu_nvml.c: * 		empty memory or graphics frequency is specified, then GpuFreqDef
src/plugins/gpu/nvml/gpu_nvml.c:static void _set_freq(bitstr_t *gpus, char *gpu_freq)
src/plugins/gpu/nvml/gpu_nvml.c:	int gpu_len = 0;
src/plugins/gpu/nvml/gpu_nvml.c:	unsigned int gpu_freq_num = 0, mem_freq_num = 0;
src/plugins/gpu/nvml/gpu_nvml.c:	debug2("_parse_gpu_freq(%s)", gpu_freq);
src/plugins/gpu/nvml/gpu_nvml.c:	gpu_common_parse_gpu_freq(gpu_freq, &gpu_freq_num, &mem_freq_num,
src/plugins/gpu/nvml/gpu_nvml.c:	tmp = gpu_common_freq_value_to_string(mem_freq_num);
src/plugins/gpu/nvml/gpu_nvml.c:	debug2("Requested GPU memory frequency: %s", tmp);
src/plugins/gpu/nvml/gpu_nvml.c:	tmp = gpu_common_freq_value_to_string(gpu_freq_num);
src/plugins/gpu/nvml/gpu_nvml.c:	debug2("Requested GPU graphics frequency: %s", tmp);
src/plugins/gpu/nvml/gpu_nvml.c:	if (!mem_freq_num && !gpu_freq_num) {
src/plugins/gpu/nvml/gpu_nvml.c:	// Check if GPUs are constrained by cgroups
src/plugins/gpu/nvml/gpu_nvml.c:	// If both of these are true, then GPUs will be constrained
src/plugins/gpu/nvml/gpu_nvml.c:		gpu_len = bit_set_count(gpus);
src/plugins/gpu/nvml/gpu_nvml.c:		debug2("%s: cgroups are configured. Using LOCAL GPU IDs",
src/plugins/gpu/nvml/gpu_nvml.c:	 	gpu_len = bit_size(gpus);
src/plugins/gpu/nvml/gpu_nvml.c:		debug2("%s: cgroups are NOT configured. Assuming GLOBAL GPU IDs",
src/plugins/gpu/nvml/gpu_nvml.c:	for (i = 0; i < gpu_len; i++) {
src/plugins/gpu/nvml/gpu_nvml.c:		unsigned int gpu_freq = gpu_freq_num, mem_freq = mem_freq_num;
src/plugins/gpu/nvml/gpu_nvml.c:		// Only check the global GPU bitstring if not using cgroups
src/plugins/gpu/nvml/gpu_nvml.c:		if (!cgroups_active && !bit_test(gpus, i)) {
src/plugins/gpu/nvml/gpu_nvml.c:		_nvml_get_nearest_freqs(&device, &mem_freq, &gpu_freq);
src/plugins/gpu/nvml/gpu_nvml.c:		freq_set = _nvml_set_freqs(&device, mem_freq, gpu_freq);
src/plugins/gpu/nvml/gpu_nvml.c:		if (gpu_freq) {
src/plugins/gpu/nvml/gpu_nvml.c:			xstrfmtcat(tmp, "%sgraphics_freq:%u", sep, gpu_freq);
src/plugins/gpu/nvml/gpu_nvml.c:			log_flag(GRES, "Successfully set GPU[%d] %s", i, tmp);
src/plugins/gpu/nvml/gpu_nvml.c:			log_flag(GRES, "Failed to set GPU[%d] %s", i, tmp);
src/plugins/gpu/nvml/gpu_nvml.c:			fprintf(stderr, "GpuFreq=%s\n", tmp);
src/plugins/gpu/nvml/gpu_nvml.c:			freq_logged = true;	/* Just log for first GPU */
src/plugins/gpu/nvml/gpu_nvml.c:		log_flag(GRES, "%s: Could not set frequencies for all GPUs. Set %d/%d total GPUs",
src/plugins/gpu/nvml/gpu_nvml.c:		fprintf(stderr, "Could not set frequencies for all GPUs. "
src/plugins/gpu/nvml/gpu_nvml.c:			"Set %d/%d total GPUs\n", count_set, count);
src/plugins/gpu/nvml/gpu_nvml.c:		error("Failed to get the NVIDIA graphics driver version: %s",
src/plugins/gpu/nvml/gpu_nvml.c: * Get the total # of GPUs in the system
src/plugins/gpu/nvml/gpu_nvml.c:extern void gpu_p_get_device_count(uint32_t *device_count)
src/plugins/gpu/nvml/gpu_nvml.c: * Get the name of the GPU
src/plugins/gpu/nvml/gpu_nvml.c:		error("Failed to get name of the GPU: %s",
src/plugins/gpu/nvml/gpu_nvml.c:	gpu_common_underscorify_tolower(device_name);
src/plugins/gpu/nvml/gpu_nvml.c:		error("Failed to get UUID of GPU: %s",
src/plugins/gpu/nvml/gpu_nvml.c:		error("Failed to get PCI info of GPU: %s",
src/plugins/gpu/nvml/gpu_nvml.c: * such that the Nvidia device node file for each GPU will have the form
src/plugins/gpu/nvml/gpu_nvml.c: * /dev/nvidia[minor_number].
src/plugins/gpu/nvml/gpu_nvml.c:		error("Failed to get minor number of GPU: %s",
src/plugins/gpu/nvml/gpu_nvml.c: * the ideal CPU affinity for the GPU.
src/plugins/gpu/nvml/gpu_nvml.c:		error("Failed to get cpu affinity of GPU: %s",
src/plugins/gpu/nvml/gpu_nvml.c: * device - the GPU device
src/plugins/gpu/nvml/gpu_nvml.c: * device - the current GPU to get the nvlink info for
src/plugins/gpu/nvml/gpu_nvml.c: * index - the index of the current GPU as returned by NVML. Based on PCI bus id
src/plugins/gpu/nvml/gpu_nvml.c: * device_lut - an array of PCI busid's for each GPU. The index is the GPU index
src/plugins/gpu/nvml/gpu_nvml.c:			error("Failed to get nvlink info from GPU: %s",
src/plugins/gpu/nvml/gpu_nvml.c:/* MIG requires CUDA 11.1 and NVIDIA driver 450.80.02 or later */
src/plugins/gpu/nvml/gpu_nvml.c: * Get the handle to the MIG device for the passed GPU device and MIG index
src/plugins/gpu/nvml/gpu_nvml.c: * device	(IN) The GPU device handle
src/plugins/gpu/nvml/gpu_nvml.c: * Get the GPU instance ID of a MIG device handle
src/plugins/gpu/nvml/gpu_nvml.c:static void _nvml_get_gpu_instance_id(nvmlDevice_t *mig, unsigned int *gi_id)
src/plugins/gpu/nvml/gpu_nvml.c:	nvmlReturn_t nvml_rc = nvmlDeviceGetGpuInstanceId(*mig, gi_id);
src/plugins/gpu/nvml/gpu_nvml.c:		error("Failed to get MIG GPU instance ID: %s",
src/plugins/gpu/nvml/gpu_nvml.c:		error("Failed to get MIG GPU instance ID: %s",
src/plugins/gpu/nvml/gpu_nvml.c:		error("Failed to get MIG mode of the GPU: %s",
src/plugins/gpu/nvml/gpu_nvml.c: * Get the minor numbers for the GPU instance and compute instance for a MIG
src/plugins/gpu/nvml/gpu_nvml.c: * gpu_minor	(IN) The minor number of the parent GPU of the MIG device.
src/plugins/gpu/nvml/gpu_nvml.c: * gi_id	(IN) The GPU instance ID of the MIG device.
src/plugins/gpu/nvml/gpu_nvml.c: * gi_minor	(OUT) The minor number of the GPU instance.
src/plugins/gpu/nvml/gpu_nvml.c:static int _nvml_get_mig_minor_numbers(unsigned int gpu_minor,
src/plugins/gpu/nvml/gpu_nvml.c:	char *path = "/proc/driver/nvidia-caps/mig-minors";
src/plugins/gpu/nvml/gpu_nvml.c:	/* You can't have more than 7 compute instances per GPU instance */
src/plugins/gpu/nvml/gpu_nvml.c:	snprintf(gi_fmt, MIG_LINE_SIZE, "gpu%u/gi%u/access", gpu_minor,
src/plugins/gpu/nvml/gpu_nvml.c:	snprintf(ci_fmt, MIG_LINE_SIZE, "gpu%u/gi%u/ci%u/access", gpu_minor,
src/plugins/gpu/nvml/gpu_nvml.c:			error("mig-minors: %d: Reached end of file. Could not find GPU=%u|GI=%u|CI=%u",
src/plugins/gpu/nvml/gpu_nvml.c:			      i, gpu_minor, gi_id, ci_id);
src/plugins/gpu/nvml/gpu_nvml.c:			debug3("GPU:%u|GI:%u,GI_minor=%u|CI:%u,CI_minor=%u",
src/plugins/gpu/nvml/gpu_nvml.c:			      gpu_minor, gi_id, *gi_minor, ci_id, *ci_minor);
src/plugins/gpu/nvml/gpu_nvml.c:		info("MIG is disabled, but set to be enabled on next GPU reset");
src/plugins/gpu/nvml/gpu_nvml.c:		info("MIG is enabled, but set to be disabled on next GPU reset");
src/plugins/gpu/nvml/gpu_nvml.c: * According to NVIDIA documentation:
src/plugins/gpu/nvml/gpu_nvml.c: * "With drivers >= R470 (470.42.01+), each MIG device is assigned a GPU UUID
src/plugins/gpu/nvml/gpu_nvml.c: * https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#:~:text=CUDA_VISIBLE_DEVICES%20has%20been,instance%20ID%3E
src/plugins/gpu/nvml/gpu_nvml.c: * gpu_minor	(IN) The GPU minor number
src/plugins/gpu/nvml/gpu_nvml.c: * gpu_uuid	(IN) The UUID string of the parent GPU
src/plugins/gpu/nvml/gpu_nvml.c: * 		populated with the parent GPU type string, and files should
src/plugins/gpu/nvml/gpu_nvml.c: * 		already be populated with the parent GPU device file.
src/plugins/gpu/nvml/gpu_nvml.c: * files includes a comma-separated string of NVIDIA capability device files
src/plugins/gpu/nvml/gpu_nvml.c:static int _handle_mig(nvmlDevice_t *device, unsigned int gpu_minor,
src/plugins/gpu/nvml/gpu_nvml.c:		       unsigned int mig_index, char *gpu_uuid,
src/plugins/gpu/nvml/gpu_nvml.c:	_nvml_get_gpu_instance_id(&mig, &gi_id);
src/plugins/gpu/nvml/gpu_nvml.c:	if (_nvml_get_mig_minor_numbers(gpu_minor, gi_id, ci_id, &gi_minor,
src/plugins/gpu/nvml/gpu_nvml.c:		    attributes.gpuInstanceSliceCount)
src/plugins/gpu/nvml/gpu_nvml.c:			   attributes.gpuInstanceSliceCount,
src/plugins/gpu/nvml/gpu_nvml.c:		xstrfmtcat(nvml_mig->unique_id, "MIG-%s/%u/%u", gpu_uuid, gi_id, ci_id);
src/plugins/gpu/nvml/gpu_nvml.c:	/* Allow access to both the GPU instance and the compute instance */
src/plugins/gpu/nvml/gpu_nvml.c:	xstrfmtcat(nvml_mig->files, ",/dev/nvidia-caps/nvidia-cap%u,/dev/nvidia-caps/nvidia-cap%u",
src/plugins/gpu/nvml/gpu_nvml.c:	debug2("GPU minor %u, MIG index %u:", gpu_minor, mig_index);
src/plugins/gpu/nvml/gpu_nvml.c:	debug2("    GPU Instance (GI) ID: %u", gi_id);
src/plugins/gpu/nvml/gpu_nvml.c: * Creates and returns a gres conf list of detected nvidia gpus on the node.
src/plugins/gpu/nvml/gpu_nvml.c: * If the NVIDIA NVML API exists (comes with CUDA), then query GPU info,
src/plugins/gpu/nvml/gpu_nvml.c:static list_t *_get_system_gpu_list_nvml(node_config_load_t *node_config)
src/plugins/gpu/nvml/gpu_nvml.c:	gpu_p_get_device_count(&device_count);
src/plugins/gpu/nvml/gpu_nvml.c:	 * Loop through all the GPUs on the system and add to gres_list_system
src/plugins/gpu/nvml/gpu_nvml.c:			.name = "gpu",
src/plugins/gpu/nvml/gpu_nvml.c:			error("Creating null GRES GPU record");
src/plugins/gpu/nvml/gpu_nvml.c:		xstrfmtcat(device_file, "/dev/nvidia%u", minor_number);
src/plugins/gpu/nvml/gpu_nvml.c:		debug2("GPU index %u:", i);
src/plugins/gpu/nvml/gpu_nvml.c:			debug("Note: GPU index %u is different from minor "
src/plugins/gpu/nvml/gpu_nvml.c:				 * after the real type of gpu since we are going
src/plugins/gpu/nvml/gpu_nvml.c:				error("MIG mode is enabled, but no MIG devices were found. Please either create MIG instances, disable MIG mode, remove AutoDetect=nvml, or remove GPUs from the configuration completely.");
src/plugins/gpu/nvml/gpu_nvml.c:				 * device name will be the same as non-MIG GPU.
src/plugins/gpu/nvml/gpu_nvml.c:	info("%u GPU system device(s) detected", device_count);
src/plugins/gpu/nvml/gpu_nvml.c:			/* Store MB usedGpuMemory is in bytes */
src/plugins/gpu/nvml/gpu_nvml.c:			data[gpumem_pos].size_read += proc_info[i].usedGpuMemory;
src/plugins/gpu/nvml/gpu_nvml.c:		log_flag(JAG, "pid %d has GPUUtil=%lu and MemMB=%lu",
src/plugins/gpu/nvml/gpu_nvml.c:			 pid, data[gpuutil_pos].size_read,
src/plugins/gpu/nvml/gpu_nvml.c:			 data[gpumem_pos].size_read / 1048576);
src/plugins/gpu/nvml/gpu_nvml.c:static int _get_gpumem(nvmlDevice_t device, pid_t pid, acct_gather_data_t *data)
src/plugins/gpu/nvml/gpu_nvml.c:static int _get_gpuutil(nvmlDevice_t device, pid_t pid,
src/plugins/gpu/nvml/gpu_nvml.c:					     data[gpuutil_pos].last_time);
src/plugins/gpu/nvml/gpu_nvml.c:		error("NVML: Failed to get process count for gpu utilization(%d): %s",
src/plugins/gpu/nvml/gpu_nvml.c:					     data[gpuutil_pos].last_time);
src/plugins/gpu/nvml/gpu_nvml.c:		debug2("On MIG-enabled GPUs, querying process utilization is not currently supported.");
src/plugins/gpu/nvml/gpu_nvml.c:		data[gpuutil_pos].last_time = proc_util[i].timeStamp;
src/plugins/gpu/nvml/gpu_nvml.c:		data[gpuutil_pos].size_read += proc_util[i].smUtil;
src/plugins/gpu/nvml/gpu_nvml.c:		gpu_get_tres_pos(&gpumem_pos, &gpuutil_pos);
src/plugins/gpu/nvml/gpu_nvml.c:extern list_t *gpu_p_get_system_gpu_list(node_config_load_t *node_config)
src/plugins/gpu/nvml/gpu_nvml.c:	if (!(gres_list_system = _get_system_gpu_list_nvml(node_config)))
src/plugins/gpu/nvml/gpu_nvml.c:		error("System GPU detection failed");
src/plugins/gpu/nvml/gpu_nvml.c:extern void gpu_p_step_hardware_init(bitstr_t *usable_gpus, char *tres_freq)
src/plugins/gpu/nvml/gpu_nvml.c:	xassert(usable_gpus);
src/plugins/gpu/nvml/gpu_nvml.c:	if (!usable_gpus)
src/plugins/gpu/nvml/gpu_nvml.c:		return;		/* Job allocated no GPUs */
src/plugins/gpu/nvml/gpu_nvml.c:	if (!(tmp = strstr(tres_freq, "gpu:")))
src/plugins/gpu/nvml/gpu_nvml.c:		return;		/* No GPU frequency spec */
src/plugins/gpu/nvml/gpu_nvml.c:	// Save a copy of the GPUs affected, so we can reset things afterwards
src/plugins/gpu/nvml/gpu_nvml.c:	FREE_NULL_BITMAP(saved_gpus);
src/plugins/gpu/nvml/gpu_nvml.c:	saved_gpus = bit_copy(usable_gpus);
src/plugins/gpu/nvml/gpu_nvml.c:	// Set the frequency of each GPU index specified in the bitstr
src/plugins/gpu/nvml/gpu_nvml.c:	_set_freq(usable_gpus, freq);
src/plugins/gpu/nvml/gpu_nvml.c:extern void gpu_p_step_hardware_fini(void)
src/plugins/gpu/nvml/gpu_nvml.c:	if (!saved_gpus)
src/plugins/gpu/nvml/gpu_nvml.c:	_reset_freq(saved_gpus);
src/plugins/gpu/nvml/gpu_nvml.c:	FREE_NULL_BITMAP(saved_gpus);
src/plugins/gpu/nvml/gpu_nvml.c:extern char *gpu_p_test_cpu_conv(char *cpu_range)
src/plugins/gpu/nvml/gpu_nvml.c:extern int gpu_p_energy_read(uint32_t dv_ind, gpu_status_t *gpu)
src/plugins/gpu/nvml/gpu_nvml.c:extern int gpu_p_usage_read(pid_t pid, acct_gather_data_t *data)
src/plugins/gpu/nvml/gpu_nvml.c:	bool track_gpumem, track_gpuutil;
src/plugins/gpu/nvml/gpu_nvml.c:	track_gpumem = (gpumem_pos != -1);
src/plugins/gpu/nvml/gpu_nvml.c:	track_gpuutil = (gpuutil_pos != -1);
src/plugins/gpu/nvml/gpu_nvml.c:	if (!track_gpuutil && !track_gpumem) {
src/plugins/gpu/nvml/gpu_nvml.c:		debug2("%s: We are not tracking TRES gpuutil/gpumem", __func__);
src/plugins/gpu/nvml/gpu_nvml.c:	gpu_p_get_device_count(&device_count);
src/plugins/gpu/nvml/gpu_nvml.c:	data[gpumem_pos].size_read = 0;
src/plugins/gpu/nvml/gpu_nvml.c:	data[gpuutil_pos].size_read = 0;
src/plugins/gpu/nvml/gpu_nvml.c:		if (track_gpumem)
src/plugins/gpu/nvml/gpu_nvml.c:			_get_gpumem(device, pid, data);
src/plugins/gpu/nvml/gpu_nvml.c:		if (track_gpuutil)
src/plugins/gpu/nvml/gpu_nvml.c:			_get_gpuutil(device, pid, data);
src/plugins/gpu/nvml/gpu_nvml.c:		log_flag(JAG, "pid %d has GPUUtil=%lu and MemMB=%lu",
src/plugins/gpu/nvml/gpu_nvml.c:			 data[gpuutil_pos].size_read,
src/plugins/gpu/nvml/gpu_nvml.c:			 data[gpumem_pos].size_read / 1048576);
src/plugins/gpu/common/gpu_common.c: *  gpu_common.c - GPU plugin common functions
src/plugins/gpu/common/gpu_common.c:#include "gpu_common.h"
src/plugins/gpu/common/gpu_common.c:static unsigned int _xlate_freq_code(char *gpu_freq)
src/plugins/gpu/common/gpu_common.c:	if (!gpu_freq || !gpu_freq[0])
src/plugins/gpu/common/gpu_common.c:	if ((gpu_freq[0] >= '0') && (gpu_freq[0] <= '9'))
src/plugins/gpu/common/gpu_common.c:	if (!xstrcasecmp(gpu_freq, "low"))
src/plugins/gpu/common/gpu_common.c:		return GPU_LOW;
src/plugins/gpu/common/gpu_common.c:	else if (!xstrcasecmp(gpu_freq, "medium"))
src/plugins/gpu/common/gpu_common.c:		return GPU_MEDIUM;
src/plugins/gpu/common/gpu_common.c:	else if (!xstrcasecmp(gpu_freq, "highm1"))
src/plugins/gpu/common/gpu_common.c:		return GPU_HIGH_M1;
src/plugins/gpu/common/gpu_common.c:	else if (!xstrcasecmp(gpu_freq, "high"))
src/plugins/gpu/common/gpu_common.c:		return GPU_HIGH;
src/plugins/gpu/common/gpu_common.c:	debug("%s: %s: Invalid job GPU frequency (%s)",
src/plugins/gpu/common/gpu_common.c:	      plugin_type, __func__, gpu_freq);
src/plugins/gpu/common/gpu_common.c:static unsigned int _xlate_freq_value(char *gpu_freq)
src/plugins/gpu/common/gpu_common.c:	if (!gpu_freq || ((gpu_freq[0] < '0') && (gpu_freq[0] > '9')))
src/plugins/gpu/common/gpu_common.c:	value = strtoul(gpu_freq, NULL, 10);
src/plugins/gpu/common/gpu_common.c:static void _parse_gpu_freq2(char *gpu_freq, unsigned int *gpu_freq_code,
src/plugins/gpu/common/gpu_common.c:			     unsigned int *gpu_freq_value,
src/plugins/gpu/common/gpu_common.c:	if (!gpu_freq || !gpu_freq[0])
src/plugins/gpu/common/gpu_common.c:	tmp = xstrdup(gpu_freq);
src/plugins/gpu/common/gpu_common.c:					debug("Invalid job GPU memory frequency: %s",
src/plugins/gpu/common/gpu_common.c:			if (!(*gpu_freq_code = _xlate_freq_code(tok)) &&
src/plugins/gpu/common/gpu_common.c:			    !(*gpu_freq_value = _xlate_freq_value(tok))) {
src/plugins/gpu/common/gpu_common.c:				debug("Invalid job GPU frequency: %s", tok);
src/plugins/gpu/common/gpu_common.c:extern char *gpu_common_freq_value_to_string(unsigned int freq)
src/plugins/gpu/common/gpu_common.c:	case GPU_LOW:
src/plugins/gpu/common/gpu_common.c:	case GPU_MEDIUM:
src/plugins/gpu/common/gpu_common.c:	case GPU_HIGH:
src/plugins/gpu/common/gpu_common.c:	case GPU_HIGH_M1:
src/plugins/gpu/common/gpu_common.c:extern void gpu_common_get_nearest_freq(unsigned int *freq,
src/plugins/gpu/common/gpu_common.c:	case GPU_LOW:
src/plugins/gpu/common/gpu_common.c:		debug2("Frequency GPU_LOW: %u MHz", *freq);
src/plugins/gpu/common/gpu_common.c:	case GPU_MEDIUM:
src/plugins/gpu/common/gpu_common.c:		debug2("Frequency GPU_MEDIUM: %u MHz", *freq);
src/plugins/gpu/common/gpu_common.c:	case GPU_HIGH_M1:
src/plugins/gpu/common/gpu_common.c:		debug2("Frequency GPU_HIGH_M1: %u MHz", *freq);
src/plugins/gpu/common/gpu_common.c:	case GPU_HIGH:
src/plugins/gpu/common/gpu_common.c:		debug2("Frequency GPU_HIGH: %u MHz", *freq);
src/plugins/gpu/common/gpu_common.c: * 		E.g., a value of "GPU Graphics" would print a header of
src/plugins/gpu/common/gpu_common.c: * 		"Possible GPU Graphics Frequencies". Set to "" or NULL to just
src/plugins/gpu/common/gpu_common.c:extern void gpu_common_print_freqs(unsigned int freqs[], unsigned int size,
src/plugins/gpu/common/gpu_common.c:extern void gpu_common_underscorify_tolower(char *str)
src/plugins/gpu/common/gpu_common.c:extern void gpu_common_parse_gpu_freq(char *gpu_freq,
src/plugins/gpu/common/gpu_common.c:				      unsigned int *gpu_freq_num,
src/plugins/gpu/common/gpu_common.c:	unsigned int def_gpu_freq_code = 0, def_gpu_freq_value = 0;
src/plugins/gpu/common/gpu_common.c:	unsigned int job_gpu_freq_code = 0, job_gpu_freq_value = 0;
src/plugins/gpu/common/gpu_common.c:	_parse_gpu_freq2(gpu_freq, &job_gpu_freq_code, &job_gpu_freq_value,
src/plugins/gpu/common/gpu_common.c:	def_freq = slurm_get_gpu_freq_def();
src/plugins/gpu/common/gpu_common.c:	_parse_gpu_freq2(def_freq, &def_gpu_freq_code, &def_gpu_freq_value,
src/plugins/gpu/common/gpu_common.c:	if (job_gpu_freq_code)
src/plugins/gpu/common/gpu_common.c:		*gpu_freq_num = job_gpu_freq_code;
src/plugins/gpu/common/gpu_common.c:	else if (job_gpu_freq_value)
src/plugins/gpu/common/gpu_common.c:		*gpu_freq_num = job_gpu_freq_value;
src/plugins/gpu/common/gpu_common.c:	else if (def_gpu_freq_code)
src/plugins/gpu/common/gpu_common.c:		*gpu_freq_num = def_gpu_freq_code;
src/plugins/gpu/common/gpu_common.c:	else if (def_gpu_freq_value)
src/plugins/gpu/common/gpu_common.c:		*gpu_freq_num = def_gpu_freq_value;
src/plugins/gpu/common/gpu_common.h: *  gpu_common.h - GPU plugin common header file
src/plugins/gpu/common/gpu_common.h:#ifndef _GPU_COMMON_H
src/plugins/gpu/common/gpu_common.h:#define _GPU_COMMON_H
src/plugins/gpu/common/gpu_common.h:#include "src/interfaces/gpu.h"
src/plugins/gpu/common/gpu_common.h:#define GPU_LOW         ((unsigned int) -1)
src/plugins/gpu/common/gpu_common.h:#define GPU_MEDIUM      ((unsigned int) -2)
src/plugins/gpu/common/gpu_common.h:#define GPU_HIGH_M1     ((unsigned int) -3)
src/plugins/gpu/common/gpu_common.h:#define GPU_HIGH        ((unsigned int) -4)
src/plugins/gpu/common/gpu_common.h:extern char *gpu_common_freq_value_to_string(unsigned int freq);
src/plugins/gpu/common/gpu_common.h:extern void gpu_common_get_nearest_freq(unsigned int *freq,
src/plugins/gpu/common/gpu_common.h:extern void gpu_common_parse_gpu_freq(char *gpu_freq,
src/plugins/gpu/common/gpu_common.h:				      unsigned int *gpu_freq_num,
src/plugins/gpu/common/gpu_common.h: * 		E.g., a value of "GPU Graphics" would print a header of
src/plugins/gpu/common/gpu_common.h: * 		"Possible GPU Graphics Frequencies". Set to "" or NULL to just
src/plugins/gpu/common/gpu_common.h:extern void gpu_common_print_freqs(unsigned int freqs[], unsigned int size,
src/plugins/gpu/common/gpu_common.h:extern void gpu_common_underscorify_tolower(char *str);
src/plugins/gpu/common/gpu_common.h:#endif /* !_GPU_COMMON_H */
src/plugins/gpu/common/Makefile.in:# Makefile for gpu/common
src/plugins/gpu/common/Makefile.in:subdir = src/plugins/gpu/common
src/plugins/gpu/common/Makefile.in:libgpu_common_la_LIBADD =
src/plugins/gpu/common/Makefile.in:am_libgpu_common_la_OBJECTS = gpu_common.lo
src/plugins/gpu/common/Makefile.in:libgpu_common_la_OBJECTS = $(am_libgpu_common_la_OBJECTS)
src/plugins/gpu/common/Makefile.in:am__depfiles_remade = ./$(DEPDIR)/gpu_common.Plo
src/plugins/gpu/common/Makefile.in:SOURCES = $(libgpu_common_la_SOURCES)
src/plugins/gpu/common/Makefile.in:noinst_LTLIBRARIES = libgpu_common.la
src/plugins/gpu/common/Makefile.in:libgpu_common_la_SOURCES = \
src/plugins/gpu/common/Makefile.in:	gpu_common.c		\
src/plugins/gpu/common/Makefile.in:	gpu_common.h
src/plugins/gpu/common/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign src/plugins/gpu/common/Makefile'; \
src/plugins/gpu/common/Makefile.in:	  $(AUTOMAKE) --foreign src/plugins/gpu/common/Makefile
src/plugins/gpu/common/Makefile.in:libgpu_common.la: $(libgpu_common_la_OBJECTS) $(libgpu_common_la_DEPENDENCIES) $(EXTRA_libgpu_common_la_DEPENDENCIES) 
src/plugins/gpu/common/Makefile.in:	$(AM_V_CCLD)$(LINK)  $(libgpu_common_la_OBJECTS) $(libgpu_common_la_LIBADD) $(LIBS)
src/plugins/gpu/common/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/gpu_common.Plo@am__quote@ # am--include-marker
src/plugins/gpu/common/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_common.Plo
src/plugins/gpu/common/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_common.Plo
src/plugins/gpu/common/Makefile.am:# Makefile for gpu/common
src/plugins/gpu/common/Makefile.am:noinst_LTLIBRARIES = libgpu_common.la
src/plugins/gpu/common/Makefile.am:libgpu_common_la_SOURCES =	\
src/plugins/gpu/common/Makefile.am:	gpu_common.c		\
src/plugins/gpu/common/Makefile.am:	gpu_common.h
src/plugins/gpu/oneapi/Makefile.in:# Makefile for gpu/oneapi plugin
src/plugins/gpu/oneapi/Makefile.in:subdir = src/plugins/gpu/oneapi
src/plugins/gpu/oneapi/Makefile.in:gpu_oneapi_la_DEPENDENCIES = ../common/libgpu_common.la
src/plugins/gpu/oneapi/Makefile.in:am__objects_1 = gpu_oneapi.lo
src/plugins/gpu/oneapi/Makefile.in:am_gpu_oneapi_la_OBJECTS = $(am__objects_1)
src/plugins/gpu/oneapi/Makefile.in:gpu_oneapi_la_OBJECTS = $(am_gpu_oneapi_la_OBJECTS)
src/plugins/gpu/oneapi/Makefile.in:gpu_oneapi_la_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
src/plugins/gpu/oneapi/Makefile.in:	$(gpu_oneapi_la_LDFLAGS) $(LDFLAGS) -o $@
src/plugins/gpu/oneapi/Makefile.in:am__depfiles_remade = ./$(DEPDIR)/gpu_oneapi.Plo
src/plugins/gpu/oneapi/Makefile.in:SOURCES = $(gpu_oneapi_la_SOURCES)
src/plugins/gpu/oneapi/Makefile.in:ONEAPI_SOURCES = gpu_oneapi.c
src/plugins/gpu/oneapi/Makefile.in:pkglib_LTLIBRARIES = gpu_oneapi.la
src/plugins/gpu/oneapi/Makefile.in:gpu_oneapi_la_SOURCES = $(ONEAPI_SOURCES)
src/plugins/gpu/oneapi/Makefile.in:gpu_oneapi_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gpu/oneapi/Makefile.in:gpu_oneapi_la_LIBADD = ../common/libgpu_common.la
src/plugins/gpu/oneapi/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign src/plugins/gpu/oneapi/Makefile'; \
src/plugins/gpu/oneapi/Makefile.in:	  $(AUTOMAKE) --foreign src/plugins/gpu/oneapi/Makefile
src/plugins/gpu/oneapi/Makefile.in:gpu_oneapi.la: $(gpu_oneapi_la_OBJECTS) $(gpu_oneapi_la_DEPENDENCIES) $(EXTRA_gpu_oneapi_la_DEPENDENCIES) 
src/plugins/gpu/oneapi/Makefile.in:	$(AM_V_CCLD)$(gpu_oneapi_la_LINK) -rpath $(pkglibdir) $(gpu_oneapi_la_OBJECTS) $(gpu_oneapi_la_LIBADD) $(LIBS)
src/plugins/gpu/oneapi/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/gpu_oneapi.Plo@am__quote@ # am--include-marker
src/plugins/gpu/oneapi/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_oneapi.Plo
src/plugins/gpu/oneapi/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_oneapi.Plo
src/plugins/gpu/oneapi/Makefile.am:# Makefile for gpu/oneapi plugin
src/plugins/gpu/oneapi/Makefile.am:ONEAPI_SOURCES = gpu_oneapi.c
src/plugins/gpu/oneapi/Makefile.am:pkglib_LTLIBRARIES = gpu_oneapi.la
src/plugins/gpu/oneapi/Makefile.am:gpu_oneapi_la_SOURCES = $(ONEAPI_SOURCES)
src/plugins/gpu/oneapi/Makefile.am:gpu_oneapi_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gpu/oneapi/Makefile.am:gpu_oneapi_la_LIBADD = ../common/libgpu_common.la
src/plugins/gpu/oneapi/gpu_oneapi.c: *  gpu_oneapi.c - Support oneAPI interface to an Intel GPU.
src/plugins/gpu/oneapi/gpu_oneapi.c: *  Based on gpu_nvml.c, written by Danny Auble <da@schedmd.com>
src/plugins/gpu/oneapi/gpu_oneapi.c:#include "src/plugins/gpu/common/gpu_common.h"
src/plugins/gpu/oneapi/gpu_oneapi.c:#define MAX_GPU_NUM 256
src/plugins/gpu/oneapi/gpu_oneapi.c:static bitstr_t	*saved_gpus;
src/plugins/gpu/oneapi/gpu_oneapi.c:const char plugin_name[] = "GPU oneAPI plugin";
src/plugins/gpu/oneapi/gpu_oneapi.c:const char plugin_type[] = "gpu/oneapi";
src/plugins/gpu/oneapi/gpu_oneapi.c: * Print GPU driver version and API version
src/plugins/gpu/oneapi/gpu_oneapi.c: * Get all of GPU device handles
src/plugins/gpu/oneapi/gpu_oneapi.c: * gpu_handles		(IN/OUT) The device handles
src/plugins/gpu/oneapi/gpu_oneapi.c: * gpu_size 		(IN/OUT) The size of the gpu_handles array. This will
src/plugins/gpu/oneapi/gpu_oneapi.c:static void _oneapi_get_device_handles(ze_device_handle_t *gpu_handles,
src/plugins/gpu/oneapi/gpu_oneapi.c:				       uint32_t *gpu_size,
src/plugins/gpu/oneapi/gpu_oneapi.c:	int gpu_count = 0;
src/plugins/gpu/oneapi/gpu_oneapi.c:	bool gpu_driver = false;
src/plugins/gpu/oneapi/gpu_oneapi.c:		gpu_driver = false;
src/plugins/gpu/oneapi/gpu_oneapi.c:			/* Filter non-GPU devices */
src/plugins/gpu/oneapi/gpu_oneapi.c:			if (ZE_DEVICE_TYPE_GPU != device_properties.type)
src/plugins/gpu/oneapi/gpu_oneapi.c:			gpu_driver = true;
src/plugins/gpu/oneapi/gpu_oneapi.c:			 * If the number of GPU exceeds the buffer length,
src/plugins/gpu/oneapi/gpu_oneapi.c:			if (gpu_count + 1 > *gpu_size)
src/plugins/gpu/oneapi/gpu_oneapi.c:			gpu_handles[gpu_count++] = all_devices[j];
src/plugins/gpu/oneapi/gpu_oneapi.c:		if (print_version && gpu_driver)
src/plugins/gpu/oneapi/gpu_oneapi.c:		debug2("Device count: %d", gpu_count);
src/plugins/gpu/oneapi/gpu_oneapi.c:	*gpu_size = gpu_count;
src/plugins/gpu/oneapi/gpu_oneapi.c:	gpu_common_get_nearest_freq(freq, freqs_size, freqs_sort);
src/plugins/gpu/oneapi/gpu_oneapi.c:	if ((freq_prop->type != ZES_FREQ_DOMAIN_GPU) &&
src/plugins/gpu/oneapi/gpu_oneapi.c:		freq_prop->type == ZES_FREQ_DOMAIN_GPU ? "Graphics" : "Memory",
src/plugins/gpu/oneapi/gpu_oneapi.c: * NOTE: Intel GPU supports tiles. One GPU may have two tiles, so the
src/plugins/gpu/oneapi/gpu_oneapi.c:		if (freq_prop.type == ZES_FREQ_DOMAIN_GPU)
src/plugins/gpu/oneapi/gpu_oneapi.c:			gpu_common_print_freqs(freqs, freqs_size, l,
src/plugins/gpu/oneapi/gpu_oneapi.c:					       "GPU Graphics", 8);
src/plugins/gpu/oneapi/gpu_oneapi.c:			gpu_common_print_freqs(freqs, freqs_size, l,
src/plugins/gpu/oneapi/gpu_oneapi.c:					       "GPU Memory", 8);
src/plugins/gpu/oneapi/gpu_oneapi.c:	if (freq_type != ZES_FREQ_DOMAIN_GPU &&
src/plugins/gpu/oneapi/gpu_oneapi.c:		freq_type == ZES_FREQ_DOMAIN_GPU ? "Graphics" :
src/plugins/gpu/oneapi/gpu_oneapi.c: * Set frequency for the GPU
src/plugins/gpu/oneapi/gpu_oneapi.c: * gpu_freq_num (IN) The gpu frequency code. It will be ingorned
src/plugins/gpu/oneapi/gpu_oneapi.c: * NOTE: Intel GPU supports tiles. One GPU may have two tiles, so all of tiles
src/plugins/gpu/oneapi/gpu_oneapi.c:			      unsigned int gpu_freq_num,
src/plugins/gpu/oneapi/gpu_oneapi.c:		 * If the frequency is not GPU or memory fequency or it cannot
src/plugins/gpu/oneapi/gpu_oneapi.c:		if (((freq_prop.type != ZES_FREQ_DOMAIN_GPU) &&
src/plugins/gpu/oneapi/gpu_oneapi.c:			freq = (freq_prop.type == ZES_FREQ_DOMAIN_GPU) ?
src/plugins/gpu/oneapi/gpu_oneapi.c:				gpu_freq_num : mem_freq_num;
src/plugins/gpu/oneapi/gpu_oneapi.c:			if (freq_prop.type == ZES_FREQ_DOMAIN_GPU)
src/plugins/gpu/oneapi/gpu_oneapi.c: * Reset the frequencies for the GPU to the same default frequencies
src/plugins/gpu/oneapi/gpu_oneapi.c: * Reset the frequencies of each GPU in the step to the hardware default
src/plugins/gpu/oneapi/gpu_oneapi.c: * gpus		(IN) A bitmap specifying the GPUs on which to operate
src/plugins/gpu/oneapi/gpu_oneapi.c:static void _reset_freq(bitstr_t *gpus)
src/plugins/gpu/oneapi/gpu_oneapi.c:	int gpu_len = bit_size(gpus);
src/plugins/gpu/oneapi/gpu_oneapi.c:	ze_device_handle_t all_devices[MAX_GPU_NUM];
src/plugins/gpu/oneapi/gpu_oneapi.c:	uint32_t gpu_num = MAX_GPU_NUM;
src/plugins/gpu/oneapi/gpu_oneapi.c:	_oneapi_get_device_handles(all_devices, &gpu_num, false);
src/plugins/gpu/oneapi/gpu_oneapi.c:	if (gpu_num == 0) {
src/plugins/gpu/oneapi/gpu_oneapi.c:	 * If the gpu length is greater than the total GPU number,
src/plugins/gpu/oneapi/gpu_oneapi.c:	 * use the total GPU number
src/plugins/gpu/oneapi/gpu_oneapi.c:	if (gpu_len > gpu_num)
src/plugins/gpu/oneapi/gpu_oneapi.c:		gpu_len = gpu_num;
src/plugins/gpu/oneapi/gpu_oneapi.c:	for (int i = 0; i < gpu_len; i++) {
src/plugins/gpu/oneapi/gpu_oneapi.c:		if (!bit_test(gpus, i))
src/plugins/gpu/oneapi/gpu_oneapi.c:			log_flag(GRES, "Successfully reset GPU[%d]", i);
src/plugins/gpu/oneapi/gpu_oneapi.c:			log_flag(GRES, "Failed to reset GPU[%d]", i);
src/plugins/gpu/oneapi/gpu_oneapi.c:		log_flag(GRES, "%s: Could not reset frequencies for all GPUs %d/%d total GPUs",
src/plugins/gpu/oneapi/gpu_oneapi.c:		fprintf(stderr, "Could not reset frequencies for all GPUs %d/%d total GPUs\n",
src/plugins/gpu/oneapi/gpu_oneapi.c: * Set the frequencies of each GPU specified for the step
src/plugins/gpu/oneapi/gpu_oneapi.c: * gpus		(IN) A bitmap specifying the GPUs on which to operate.
src/plugins/gpu/oneapi/gpu_oneapi.c: * gpu_freq	(IN) The frequencies to set each of the GPUs to. If a NULL or
src/plugins/gpu/oneapi/gpu_oneapi.c:		GpuFreqDef will be consulted, which defaults to
src/plugins/gpu/oneapi/gpu_oneapi.c:static void _set_freq(bitstr_t *gpus, char *gpu_freq)
src/plugins/gpu/oneapi/gpu_oneapi.c:	int gpu_len = 0;
src/plugins/gpu/oneapi/gpu_oneapi.c:	unsigned int gpu_freq_num = 0, mem_freq_num = 0;
src/plugins/gpu/oneapi/gpu_oneapi.c:	ze_device_handle_t all_devices[MAX_GPU_NUM];
src/plugins/gpu/oneapi/gpu_oneapi.c:	uint32_t gpu_num = MAX_GPU_NUM;
src/plugins/gpu/oneapi/gpu_oneapi.c:	debug2("_parse_gpu_freq(%s)", gpu_freq);
src/plugins/gpu/oneapi/gpu_oneapi.c:	gpu_common_parse_gpu_freq(gpu_freq, &gpu_freq_num, &mem_freq_num,
src/plugins/gpu/oneapi/gpu_oneapi.c:	tmp = gpu_common_freq_value_to_string(mem_freq_num);
src/plugins/gpu/oneapi/gpu_oneapi.c:	debug2("Requested GPU memory frequency: %s", tmp);
src/plugins/gpu/oneapi/gpu_oneapi.c:	tmp = gpu_common_freq_value_to_string(gpu_freq_num);
src/plugins/gpu/oneapi/gpu_oneapi.c:	debug2("Requested GPU graphics frequency: %s", tmp);
src/plugins/gpu/oneapi/gpu_oneapi.c:	if (!mem_freq_num && !gpu_freq_num) {
src/plugins/gpu/oneapi/gpu_oneapi.c:	/* Check if GPUs are constrained by cgroups */
src/plugins/gpu/oneapi/gpu_oneapi.c:	/* If both of these are true, then GPUs will be constrained */
src/plugins/gpu/oneapi/gpu_oneapi.c:		gpu_len = bit_set_count(gpus);
src/plugins/gpu/oneapi/gpu_oneapi.c:		debug2("%s: cgroups are configured. Using LOCAL GPU IDs",
src/plugins/gpu/oneapi/gpu_oneapi.c:		gpu_len = bit_size(gpus);
src/plugins/gpu/oneapi/gpu_oneapi.c:		debug2("%s: cgroups are NOT configured. Assuming GLOBAL GPU IDs",
src/plugins/gpu/oneapi/gpu_oneapi.c:	_oneapi_get_device_handles(all_devices, &gpu_num, false);
src/plugins/gpu/oneapi/gpu_oneapi.c:	if (gpu_num == 0) {
src/plugins/gpu/oneapi/gpu_oneapi.c:	if (gpu_len > gpu_num)
src/plugins/gpu/oneapi/gpu_oneapi.c:		gpu_len = gpu_num;
src/plugins/gpu/oneapi/gpu_oneapi.c:	for (int i = 0; i < gpu_len; i++) {
src/plugins/gpu/oneapi/gpu_oneapi.c:		/* Only check the global GPU bitstring if not using cgroups */
src/plugins/gpu/oneapi/gpu_oneapi.c:		if (!cgroups_active && !bit_test(gpus, i)) {
src/plugins/gpu/oneapi/gpu_oneapi.c:					     gpu_freq_num, mem_freq_num,
src/plugins/gpu/oneapi/gpu_oneapi.c:			log_flag(GRES, "Successfully set GPU[%d] %s", i, tmp);
src/plugins/gpu/oneapi/gpu_oneapi.c:			log_flag(GRES, "Failed to set GPU[%d] %s", i, tmp);
src/plugins/gpu/oneapi/gpu_oneapi.c:			fprintf(stderr, "GpuFreq=%s\n", tmp);
src/plugins/gpu/oneapi/gpu_oneapi.c:			freq_logged = true;	/* Just log for first GPU */
src/plugins/gpu/oneapi/gpu_oneapi.c:		log_flag(GRES, "%s: Could not set frequencies for all GPUs %d/%d total GPUs",
src/plugins/gpu/oneapi/gpu_oneapi.c:		fprintf(stderr, "Could not set frequencies for all GPUs %d/%d total GPUs\n",
src/plugins/gpu/oneapi/gpu_oneapi.c: * There are no APIs to get minor number of Intel GPU at the moment, so we
src/plugins/gpu/oneapi/gpu_oneapi.c: * Creates and returns a gres conf list of detected Intel gpus on the node.
src/plugins/gpu/oneapi/gpu_oneapi.c: * If the Intel oneAPI exists, then query GPU info,
src/plugins/gpu/oneapi/gpu_oneapi.c:static list_t *_get_system_gpu_list_oneapi(node_config_load_t *node_config)
src/plugins/gpu/oneapi/gpu_oneapi.c:	ze_device_handle_t all_devices[MAX_GPU_NUM];
src/plugins/gpu/oneapi/gpu_oneapi.c:	uint32_t gpu_num = MAX_GPU_NUM;
src/plugins/gpu/oneapi/gpu_oneapi.c:	_oneapi_get_device_handles(all_devices, &gpu_num, true);
src/plugins/gpu/oneapi/gpu_oneapi.c:	if (gpu_num == 0) {
src/plugins/gpu/oneapi/gpu_oneapi.c:	/* Loop all of GPU device handles */
src/plugins/gpu/oneapi/gpu_oneapi.c:	for (i = 0; i < gpu_num; i++) {
src/plugins/gpu/oneapi/gpu_oneapi.c:			.name = "gpu",
src/plugins/gpu/oneapi/gpu_oneapi.c:			error("Failed to get device card name for GPU: %u", i);
src/plugins/gpu/oneapi/gpu_oneapi.c:			error("Failed to get device affinity for GPU: %u", i);
src/plugins/gpu/oneapi/gpu_oneapi.c:		gres_slurmd_conf.links = gres_links_create_empty(i, gpu_num);
src/plugins/gpu/oneapi/gpu_oneapi.c:		debug2("GPU index %u:", i);
src/plugins/gpu/oneapi/gpu_oneapi.c:		/* Add the GPU to list */
src/plugins/gpu/oneapi/gpu_oneapi.c:extern list_t *gpu_p_get_system_gpu_list(node_config_load_t *node_config)
src/plugins/gpu/oneapi/gpu_oneapi.c:	list_t *gres_list_system = _get_system_gpu_list_oneapi(node_config);
src/plugins/gpu/oneapi/gpu_oneapi.c:		error("System GPU detection failed");
src/plugins/gpu/oneapi/gpu_oneapi.c:extern void gpu_p_step_hardware_init(bitstr_t *usable_gpus, char *tres_freq)
src/plugins/gpu/oneapi/gpu_oneapi.c:	debug2("enter gpu_p_step_hardware_init()");
src/plugins/gpu/oneapi/gpu_oneapi.c:	xassert(usable_gpus);
src/plugins/gpu/oneapi/gpu_oneapi.c:	if (!usable_gpus)
src/plugins/gpu/oneapi/gpu_oneapi.c:		return;		/* Job allocated no GPUs */
src/plugins/gpu/oneapi/gpu_oneapi.c:	tmp = strstr(tres_freq, "gpu:");
src/plugins/gpu/oneapi/gpu_oneapi.c:		return;		/* No GPU frequency spec */
src/plugins/gpu/oneapi/gpu_oneapi.c:	 * Save a copy of the GPUs affected, so we can reset things afterwards
src/plugins/gpu/oneapi/gpu_oneapi.c:	FREE_NULL_BITMAP(saved_gpus);
src/plugins/gpu/oneapi/gpu_oneapi.c:	saved_gpus = bit_copy(usable_gpus);
src/plugins/gpu/oneapi/gpu_oneapi.c:	/* Set the frequency of each GPU index specified in the bitstr */
src/plugins/gpu/oneapi/gpu_oneapi.c:	_set_freq(usable_gpus, freq);
src/plugins/gpu/oneapi/gpu_oneapi.c:	debug2("exit gpu_p_step_hardware_init() normally");
src/plugins/gpu/oneapi/gpu_oneapi.c:extern void gpu_p_step_hardware_fini(void)
src/plugins/gpu/oneapi/gpu_oneapi.c:	debug2("enter gpu_p_step_hardware_fini()");
src/plugins/gpu/oneapi/gpu_oneapi.c:	if (!saved_gpus)
src/plugins/gpu/oneapi/gpu_oneapi.c:	_reset_freq(saved_gpus);
src/plugins/gpu/oneapi/gpu_oneapi.c:	FREE_NULL_BITMAP(saved_gpus);
src/plugins/gpu/oneapi/gpu_oneapi.c:	debug2("exit gpu_p_step_hardware_fini() normally");
src/plugins/gpu/oneapi/gpu_oneapi.c:extern char *gpu_p_test_cpu_conv(char *cpu_range)
src/plugins/gpu/oneapi/gpu_oneapi.c:extern void gpu_p_get_device_count(uint32_t *device_count)
src/plugins/gpu/oneapi/gpu_oneapi.c:	ze_device_handle_t all_devices[MAX_GPU_NUM];
src/plugins/gpu/oneapi/gpu_oneapi.c:	uint32_t gpu_num = MAX_GPU_NUM;
src/plugins/gpu/oneapi/gpu_oneapi.c:	_oneapi_get_device_handles(all_devices, &gpu_num, false);
src/plugins/gpu/oneapi/gpu_oneapi.c:	if (gpu_num == 0) {
src/plugins/gpu/oneapi/gpu_oneapi.c:		*device_count = gpu_num;
src/plugins/gpu/oneapi/gpu_oneapi.c:extern int gpu_p_energy_read(uint32_t dv_ind, gpu_status_t *gpu)
src/plugins/gpu/oneapi/gpu_oneapi.c:extern int gpu_p_usage_read(pid_t pid, acct_gather_data_t *data)
src/plugins/gpu/Makefile.in:# Makefile for gpu plugins
src/plugins/gpu/Makefile.in:subdir = src/plugins/gpu
src/plugins/gpu/Makefile.in:DIST_SUBDIRS = common generic nrt nvidia rsmi nvml oneapi
src/plugins/gpu/Makefile.in:SUBDIRS = common generic nrt nvidia $(am__append_1) $(am__append_2) \
src/plugins/gpu/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --gnu src/plugins/gpu/Makefile'; \
src/plugins/gpu/Makefile.in:	  $(AUTOMAKE) --gnu src/plugins/gpu/Makefile
src/plugins/gpu/nrt/gpu_nrt.c: *  gpu_nrt.c
src/plugins/gpu/nrt/gpu_nrt.c:#include "../common/gpu_common.h"
src/plugins/gpu/nrt/gpu_nrt.c:const char plugin_name[] = "GPU NRT plugin";
src/plugins/gpu/nrt/gpu_nrt.c:const char plugin_type[] = "gpu/nrt";
src/plugins/gpu/nrt/gpu_nrt.c:static list_t *_get_system_gpu_list_neuron(node_config_load_t *node_conf)
src/plugins/gpu/nrt/gpu_nrt.c:				.name = "gpu",
src/plugins/gpu/nrt/gpu_nrt.c:			debug2("GPU index %u:", dev_inx);
src/plugins/gpu/nrt/gpu_nrt.c:			/* Add the GPU to list */
src/plugins/gpu/nrt/gpu_nrt.c:extern void gpu_p_get_device_count(uint32_t *device_count)
src/plugins/gpu/nrt/gpu_nrt.c:extern list_t *gpu_p_get_system_gpu_list(node_config_load_t *node_conf)
src/plugins/gpu/nrt/gpu_nrt.c:	list_t *gres_list_system = _get_system_gpu_list_neuron(node_conf);
src/plugins/gpu/nrt/gpu_nrt.c:		error("System GPU detection failed");
src/plugins/gpu/nrt/gpu_nrt.c:extern void gpu_p_step_hardware_init(bitstr_t *usable_gpus, char *tres_freq)
src/plugins/gpu/nrt/gpu_nrt.c:extern void gpu_p_step_hardware_fini(void)
src/plugins/gpu/nrt/gpu_nrt.c:extern char *gpu_p_test_cpu_conv(char *cpu_range)
src/plugins/gpu/nrt/gpu_nrt.c:extern int gpu_p_energy_read(uint32_t dv_ind, gpu_status_t *gpu)
src/plugins/gpu/nrt/gpu_nrt.c:extern int gpu_p_usage_read(pid_t pid, acct_gather_data_t *data)
src/plugins/gpu/nrt/gpu_nrt.c: *static int gpumem_pos = -1; // Init this in init()
src/plugins/gpu/nrt/gpu_nrt.c: *		data[gpumem_pos].size_read += dev_mem;
src/plugins/gpu/nrt/gpu_nrt.c: *extern int gpu_p_usage_read(pid_t pid, acct_gather_data_t *data)
src/plugins/gpu/nrt/gpu_nrt.c: *	bool track_gpumem;
src/plugins/gpu/nrt/gpu_nrt.c: *	track_gpumem = (gpumem_pos != -1);
src/plugins/gpu/nrt/gpu_nrt.c: *	if (!track_gpumem) {
src/plugins/gpu/nrt/gpu_nrt.c: *		debug2("%s: We are not tracking TRES gpumem", __func__);
src/plugins/gpu/nrt/gpu_nrt.c: *	data[gpumem_pos].size_read = 0;
src/plugins/gpu/nrt/gpu_nrt.c: *		if (track_gpumem) {
src/plugins/gpu/nrt/gpu_nrt.c: *			 data[gpumem_pos].size_read / 1048576);
src/plugins/gpu/nrt/Makefile.in:# Makefile for gpu/nrt plugin
src/plugins/gpu/nrt/Makefile.in:subdir = src/plugins/gpu/nrt
src/plugins/gpu/nrt/Makefile.in:gpu_nrt_la_LIBADD =
src/plugins/gpu/nrt/Makefile.in:am__objects_1 = gpu_nrt.lo
src/plugins/gpu/nrt/Makefile.in:am_gpu_nrt_la_OBJECTS = $(am__objects_1)
src/plugins/gpu/nrt/Makefile.in:gpu_nrt_la_OBJECTS = $(am_gpu_nrt_la_OBJECTS)
src/plugins/gpu/nrt/Makefile.in:gpu_nrt_la_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
src/plugins/gpu/nrt/Makefile.in:	$(gpu_nrt_la_LDFLAGS) $(LDFLAGS) -o $@
src/plugins/gpu/nrt/Makefile.in:am__depfiles_remade = ./$(DEPDIR)/gpu_nrt.Plo
src/plugins/gpu/nrt/Makefile.in:SOURCES = $(gpu_nrt_la_SOURCES)
src/plugins/gpu/nrt/Makefile.in:NRT_SOURCES = gpu_nrt.c
src/plugins/gpu/nrt/Makefile.in:pkglib_LTLIBRARIES = gpu_nrt.la
src/plugins/gpu/nrt/Makefile.in:gpu_nrt_la_SOURCES = $(NRT_SOURCES)
src/plugins/gpu/nrt/Makefile.in:gpu_nrt_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gpu/nrt/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign src/plugins/gpu/nrt/Makefile'; \
src/plugins/gpu/nrt/Makefile.in:	  $(AUTOMAKE) --foreign src/plugins/gpu/nrt/Makefile
src/plugins/gpu/nrt/Makefile.in:gpu_nrt.la: $(gpu_nrt_la_OBJECTS) $(gpu_nrt_la_DEPENDENCIES) $(EXTRA_gpu_nrt_la_DEPENDENCIES) 
src/plugins/gpu/nrt/Makefile.in:	$(AM_V_CCLD)$(gpu_nrt_la_LINK) -rpath $(pkglibdir) $(gpu_nrt_la_OBJECTS) $(gpu_nrt_la_LIBADD) $(LIBS)
src/plugins/gpu/nrt/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/gpu_nrt.Plo@am__quote@ # am--include-marker
src/plugins/gpu/nrt/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_nrt.Plo
src/plugins/gpu/nrt/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_nrt.Plo
src/plugins/gpu/nrt/Makefile.am:# Makefile for gpu/nrt plugin
src/plugins/gpu/nrt/Makefile.am:NRT_SOURCES = gpu_nrt.c
src/plugins/gpu/nrt/Makefile.am:pkglib_LTLIBRARIES = gpu_nrt.la
src/plugins/gpu/nrt/Makefile.am:gpu_nrt_la_SOURCES = $(NRT_SOURCES)
src/plugins/gpu/nrt/Makefile.am:gpu_nrt_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gpu/Makefile.am:# Makefile for gpu plugins
src/plugins/gpu/Makefile.am:SUBDIRS = common generic nrt nvidia
src/plugins/gpu/nvidia/gpu_nvidia.c: *  gpu_nvidia.c
src/plugins/gpu/nvidia/gpu_nvidia.c:#include "../common/gpu_common.h"
src/plugins/gpu/nvidia/gpu_nvidia.c:#define NVIDIA_PROC_DRIVER_PREFIX "/proc/driver/nvidia/gpus/"
src/plugins/gpu/nvidia/gpu_nvidia.c:#define NVIDIA_INFORMATION_PREFIX "/proc/driver/nvidia/gpus/%s/information"
src/plugins/gpu/nvidia/gpu_nvidia.c:#define NVIDIA_CPULIST_PREFIX "/sys/bus/pci/drivers/nvidia/%s/local_cpulist"
src/plugins/gpu/nvidia/gpu_nvidia.c:const char plugin_name[] = "GPU Nvidia plugin";
src/plugins/gpu/nvidia/gpu_nvidia.c:const char plugin_type[] = "gpu/nvidia";
src/plugins/gpu/nvidia/gpu_nvidia.c:	DIR *dr = opendir(NVIDIA_PROC_DRIVER_PREFIX);
src/plugins/gpu/nvidia/gpu_nvidia.c:	path = xstrdup_printf(NVIDIA_CPULIST_PREFIX, bus_id);
src/plugins/gpu/nvidia/gpu_nvidia.c:	path = xstrdup_printf(NVIDIA_INFORMATION_PREFIX, bus_id);
src/plugins/gpu/nvidia/gpu_nvidia.c:			xstrfmtcat(*device_file, "/dev/nvidia%u", minor_number);
src/plugins/gpu/nvidia/gpu_nvidia.c:			gpu_common_underscorify_tolower(*device_name);
src/plugins/gpu/nvidia/gpu_nvidia.c:static list_t *_get_system_gpu_list_nvidia(node_config_load_t *node_conf)
src/plugins/gpu/nvidia/gpu_nvidia.c:	DIR *dr = opendir(NVIDIA_PROC_DRIVER_PREFIX);
src/plugins/gpu/nvidia/gpu_nvidia.c:			.name = "gpu",
src/plugins/gpu/nvidia/gpu_nvidia.c:		/* Add the GPU to list */
src/plugins/gpu/nvidia/gpu_nvidia.c:extern void gpu_p_get_device_count(unsigned int *device_count)
src/plugins/gpu/nvidia/gpu_nvidia.c:extern void gpu_p_reconfig(void)
src/plugins/gpu/nvidia/gpu_nvidia.c:extern list_t *gpu_p_get_system_gpu_list(node_config_load_t *node_conf)
src/plugins/gpu/nvidia/gpu_nvidia.c:	list_t *gres_list_system = _get_system_gpu_list_nvidia(node_conf);
src/plugins/gpu/nvidia/gpu_nvidia.c:		error("System GPU detection failed");
src/plugins/gpu/nvidia/gpu_nvidia.c:extern void gpu_p_step_hardware_init(bitstr_t *usable_gpus, char *tres_freq)
src/plugins/gpu/nvidia/gpu_nvidia.c:extern void gpu_p_step_hardware_fini(void)
src/plugins/gpu/nvidia/gpu_nvidia.c:extern char *gpu_p_test_cpu_conv(char *cpu_range)
src/plugins/gpu/nvidia/gpu_nvidia.c:extern int gpu_p_energy_read(uint32_t dv_ind, gpu_status_t *gpu)
src/plugins/gpu/nvidia/gpu_nvidia.c:extern int gpu_p_usage_read(pid_t pid, acct_gather_data_t *data)
src/plugins/gpu/nvidia/Makefile.in:# Makefile for gpu/nvidia plugin
src/plugins/gpu/nvidia/Makefile.in:subdir = src/plugins/gpu/nvidia
src/plugins/gpu/nvidia/Makefile.in:gpu_nvidia_la_DEPENDENCIES = ../common/libgpu_common.la
src/plugins/gpu/nvidia/Makefile.in:am__objects_1 = gpu_nvidia.lo
src/plugins/gpu/nvidia/Makefile.in:am_gpu_nvidia_la_OBJECTS = $(am__objects_1)
src/plugins/gpu/nvidia/Makefile.in:gpu_nvidia_la_OBJECTS = $(am_gpu_nvidia_la_OBJECTS)
src/plugins/gpu/nvidia/Makefile.in:gpu_nvidia_la_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
src/plugins/gpu/nvidia/Makefile.in:	$(gpu_nvidia_la_LDFLAGS) $(LDFLAGS) -o $@
src/plugins/gpu/nvidia/Makefile.in:am__depfiles_remade = ./$(DEPDIR)/gpu_nvidia.Plo
src/plugins/gpu/nvidia/Makefile.in:SOURCES = $(gpu_nvidia_la_SOURCES)
src/plugins/gpu/nvidia/Makefile.in:NVIDIA_SOURCES = gpu_nvidia.c
src/plugins/gpu/nvidia/Makefile.in:pkglib_LTLIBRARIES = gpu_nvidia.la
src/plugins/gpu/nvidia/Makefile.in:gpu_nvidia_la_SOURCES = $(NVIDIA_SOURCES)
src/plugins/gpu/nvidia/Makefile.in:gpu_nvidia_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gpu/nvidia/Makefile.in:gpu_nvidia_la_LIBADD = ../common/libgpu_common.la
src/plugins/gpu/nvidia/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign src/plugins/gpu/nvidia/Makefile'; \
src/plugins/gpu/nvidia/Makefile.in:	  $(AUTOMAKE) --foreign src/plugins/gpu/nvidia/Makefile
src/plugins/gpu/nvidia/Makefile.in:gpu_nvidia.la: $(gpu_nvidia_la_OBJECTS) $(gpu_nvidia_la_DEPENDENCIES) $(EXTRA_gpu_nvidia_la_DEPENDENCIES) 
src/plugins/gpu/nvidia/Makefile.in:	$(AM_V_CCLD)$(gpu_nvidia_la_LINK) -rpath $(pkglibdir) $(gpu_nvidia_la_OBJECTS) $(gpu_nvidia_la_LIBADD) $(LIBS)
src/plugins/gpu/nvidia/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/gpu_nvidia.Plo@am__quote@ # am--include-marker
src/plugins/gpu/nvidia/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_nvidia.Plo
src/plugins/gpu/nvidia/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_nvidia.Plo
src/plugins/gpu/nvidia/Makefile.am:# Makefile for gpu/nvidia plugin
src/plugins/gpu/nvidia/Makefile.am:NVIDIA_SOURCES = gpu_nvidia.c
src/plugins/gpu/nvidia/Makefile.am:pkglib_LTLIBRARIES = gpu_nvidia.la
src/plugins/gpu/nvidia/Makefile.am:gpu_nvidia_la_SOURCES = $(NVIDIA_SOURCES)
src/plugins/gpu/nvidia/Makefile.am:gpu_nvidia_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gpu/nvidia/Makefile.am:gpu_nvidia_la_LIBADD = ../common/libgpu_common.la
src/plugins/gpu/generic/gpu_generic.c: *  gpu_generic.c - Support generic interface to a GPU.
src/plugins/gpu/generic/gpu_generic.c:#include "src/interfaces/gpu.h"
src/plugins/gpu/generic/gpu_generic.c:const char plugin_name[] = "GPU Generic plugin";
src/plugins/gpu/generic/gpu_generic.c:const char	plugin_type[]		= "gpu/generic";
src/plugins/gpu/generic/gpu_generic.c:extern list_t *gpu_p_get_system_gpu_list(node_config_load_t *node_config)
src/plugins/gpu/generic/gpu_generic.c:extern void gpu_p_step_hardware_init(bitstr_t *usable_gpus, char *tres_freq)
src/plugins/gpu/generic/gpu_generic.c:	xassert(usable_gpus);
src/plugins/gpu/generic/gpu_generic.c:	if (!usable_gpus)
src/plugins/gpu/generic/gpu_generic.c:		return;		/* Job allocated no GPUs */
src/plugins/gpu/generic/gpu_generic.c:	if (!strstr(tres_freq, "gpu:"))
src/plugins/gpu/generic/gpu_generic.c:		return;		/* No GPU frequency spec */
src/plugins/gpu/generic/gpu_generic.c:	fprintf(stderr, "GpuFreq=control_disabled\n");
src/plugins/gpu/generic/gpu_generic.c:extern void gpu_p_step_hardware_fini(void)
src/plugins/gpu/generic/gpu_generic.c:extern char *gpu_p_test_cpu_conv(char *cpu_range)
src/plugins/gpu/generic/gpu_generic.c:extern int gpu_p_energy_read(uint32_t dv_ind, gpu_status_t *gpu)
src/plugins/gpu/generic/gpu_generic.c:extern void gpu_p_get_device_count(uint32_t *device_count)
src/plugins/gpu/generic/gpu_generic.c:extern int gpu_p_usage_read(pid_t pid, acct_gather_data_t *data)
src/plugins/gpu/generic/Makefile.in:# Makefile for gpu/generic plugin
src/plugins/gpu/generic/Makefile.in:subdir = src/plugins/gpu/generic
src/plugins/gpu/generic/Makefile.in:gpu_generic_la_LIBADD =
src/plugins/gpu/generic/Makefile.in:am_gpu_generic_la_OBJECTS = gpu_generic.lo
src/plugins/gpu/generic/Makefile.in:gpu_generic_la_OBJECTS = $(am_gpu_generic_la_OBJECTS)
src/plugins/gpu/generic/Makefile.in:gpu_generic_la_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC \
src/plugins/gpu/generic/Makefile.in:	$(AM_CFLAGS) $(CFLAGS) $(gpu_generic_la_LDFLAGS) $(LDFLAGS) -o \
src/plugins/gpu/generic/Makefile.in:am__depfiles_remade = ./$(DEPDIR)/gpu_generic.Plo
src/plugins/gpu/generic/Makefile.in:SOURCES = $(gpu_generic_la_SOURCES)
src/plugins/gpu/generic/Makefile.in:pkglib_LTLIBRARIES = gpu_generic.la
src/plugins/gpu/generic/Makefile.in:# GPU GENERIC plugin.
src/plugins/gpu/generic/Makefile.in:gpu_generic_la_SOURCES = gpu_generic.c
src/plugins/gpu/generic/Makefile.in:gpu_generic_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gpu/generic/Makefile.in:	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign src/plugins/gpu/generic/Makefile'; \
src/plugins/gpu/generic/Makefile.in:	  $(AUTOMAKE) --foreign src/plugins/gpu/generic/Makefile
src/plugins/gpu/generic/Makefile.in:gpu_generic.la: $(gpu_generic_la_OBJECTS) $(gpu_generic_la_DEPENDENCIES) $(EXTRA_gpu_generic_la_DEPENDENCIES) 
src/plugins/gpu/generic/Makefile.in:	$(AM_V_CCLD)$(gpu_generic_la_LINK) -rpath $(pkglibdir) $(gpu_generic_la_OBJECTS) $(gpu_generic_la_LIBADD) $(LIBS)
src/plugins/gpu/generic/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/gpu_generic.Plo@am__quote@ # am--include-marker
src/plugins/gpu/generic/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_generic.Plo
src/plugins/gpu/generic/Makefile.in:		-rm -f ./$(DEPDIR)/gpu_generic.Plo
src/plugins/gpu/generic/Makefile.in:$(gpu_generic_la_LIBADD) : force
src/plugins/gpu/generic/Makefile.am:# Makefile for gpu/generic plugin
src/plugins/gpu/generic/Makefile.am:pkglib_LTLIBRARIES = gpu_generic.la
src/plugins/gpu/generic/Makefile.am:# GPU GENERIC plugin.
src/plugins/gpu/generic/Makefile.am:gpu_generic_la_SOURCES = gpu_generic.c
src/plugins/gpu/generic/Makefile.am:gpu_generic_la_LDFLAGS = $(PLUGIN_FLAGS)
src/plugins/gpu/generic/Makefile.am:$(gpu_generic_la_LIBADD) : force
src/plugins/data_parser/v0.0.41/parsers.c:	add_parse(STRING, gpu_spec, "gpu_spec", "CPU cores reserved for jobs that also use a GPU"),
src/plugins/data_parser/v0.0.41/parsers.c:	add_parse(UINT16, res_cores_per_gpu, "res_cores_per_gpu", "Number of CPU cores per GPU restricted to GPU jobs"),
src/plugins/data_parser/v0.0.41/parsers.c:	add_parse(STRING, cpus_per_tres, "cpus_per_tres", "Semicolon delimited list of TRES=# values indicating how many CPUs should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.41/parsers.c:	add_parse(STRING, mem_per_tres, "memory_per_tres", "Semicolon delimited list of TRES=# values indicating how much memory in megabytes should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.41/parsers.c:	add_parse(UINT16_NO_VAL, ntasks_per_tres, "tasks_per_tres", "Number of tasks that can assess each GPU"),
src/plugins/data_parser/v0.0.41/parsers.c:	add_parse(STRING, cpus_per_tres, "tres/per/cpu", "Semicolon delimited list of TRES=# values indicating how many CPUs should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.41/parsers.c:	add_parse(STRING, mem_per_tres, "tres/per/memory", "Semicolon delimited list of TRES=# values indicating how much memory should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.41/parsers.c:	add_parse(STRING, cpus_per_tres, "cpus_per_tres", "Semicolon delimited list of TRES=# values values indicating how many CPUs should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.41/parsers.c:	add_parse(STRING, mem_per_tres, "memory_per_tres", "Semicolon delimited list of TRES=# values indicating how much memory in megabytes should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.41/parsers.c:	add_parse(UINT16, ntasks_per_tres, "ntasks_per_tres", "Number of tasks that can access each GPU"),
src/plugins/data_parser/v0.0.42/parsers.c:	add_parse(STRING, gpu_spec, "gpu_spec", "CPU cores reserved for jobs that also use a GPU"),
src/plugins/data_parser/v0.0.42/parsers.c:	add_parse(UINT16, res_cores_per_gpu, "res_cores_per_gpu", "Number of CPU cores per GPU restricted to GPU jobs"),
src/plugins/data_parser/v0.0.42/parsers.c:	add_parse(STRING, cpus_per_tres, "cpus_per_tres", "Semicolon delimited list of TRES=# values indicating how many CPUs should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.42/parsers.c:	add_parse(STRING, mem_per_tres, "memory_per_tres", "Semicolon delimited list of TRES=# values indicating how much memory in megabytes should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.42/parsers.c:	add_parse(UINT16_NO_VAL, ntasks_per_tres, "tasks_per_tres", "Number of tasks that can assess each GPU"),
src/plugins/data_parser/v0.0.42/parsers.c:	add_parse(STRING, cpus_per_tres, "tres/per/cpu", "Semicolon delimited list of TRES=# values indicating how many CPUs should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.42/parsers.c:	add_parse(STRING, mem_per_tres, "tres/per/memory", "Semicolon delimited list of TRES=# values indicating how much memory should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.42/parsers.c:	add_parse(STRING, cpus_per_tres, "cpus_per_tres", "Semicolon delimited list of TRES=# values values indicating how many CPUs should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.42/parsers.c:	add_parse(STRING, mem_per_tres, "memory_per_tres", "Semicolon delimited list of TRES=# values indicating how much memory in megabytes should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.42/parsers.c:	add_parse(UINT16, ntasks_per_tres, "ntasks_per_tres", "Number of tasks that can access each GPU"),
src/plugins/data_parser/v0.0.40/parsers.c:	add_parse(STRING, cpus_per_tres, "cpus_per_tres", "Semicolon delimited list of TRES=# values indicating how many CPUs should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.40/parsers.c:	add_parse(STRING, mem_per_tres, "memory_per_tres", "Semicolon delimited list of TRES=# values indicating how much memory in megabytes should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.40/parsers.c:	add_parse(UINT16_NO_VAL, ntasks_per_tres, "tasks_per_tres", "Number of tasks that can assess each GPU"),
src/plugins/data_parser/v0.0.40/parsers.c:	add_parse(STRING, cpus_per_tres, "tres/per/cpu", "Semicolon delimited list of TRES=# values indicating how many CPUs should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.40/parsers.c:	add_parse(STRING, mem_per_tres, "tres/per/memory", "Semicolon delimited list of TRES=# values indicating how much memory should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.40/parsers.c:	add_parse(STRING, cpus_per_tres, "cpus_per_tres", "Semicolon delimited list of TRES=# values values indicating how many CPUs should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.40/parsers.c:	add_parse(STRING, mem_per_tres, "memory_per_tres", "Semicolon delimited list of TRES=# values indicating how much memory in megabytes should be allocated for each specified TRES (currently only used for gres/gpu)"),
src/plugins/data_parser/v0.0.40/parsers.c:	add_parse(UINT16, ntasks_per_tres, "ntasks_per_tres", "Number of tasks that can access each GPU"),
src/plugins/job_submit/lua/job_submit_lua.c:	} else if (!xstrcmp(name, "ntasks_per_gpu")) {
src/plugins/job_submit/lua/job_submit_lua.c:	} else if (!xstrcmp(name, "ntasks_per_gpu")) {
src/plugins/jobacct_gather/common/common_jag.c:#include "src/interfaces/gpu.h"
src/plugins/jobacct_gather/common/common_jag.c:	static int disable_gpu_acct = -1;
src/plugins/jobacct_gather/common/common_jag.c:	if (disable_gpu_acct == -1) {
src/plugins/jobacct_gather/common/common_jag.c:				"DisableGPUAcct")) {
src/plugins/jobacct_gather/common/common_jag.c:			disable_gpu_acct = 1;
src/plugins/jobacct_gather/common/common_jag.c:			log_flag(JAG, "GPU accounting disabled as JobAcctGatherParams=DisableGpuAcct is set.");
src/plugins/jobacct_gather/common/common_jag.c:			disable_gpu_acct = 0;
src/plugins/jobacct_gather/common/common_jag.c:	if (!disable_gpu_acct)
src/plugins/jobacct_gather/common/common_jag.c:		gpu_g_usage_read(pid, prec->tres_data);
src/plugins/jobacct_gather/common/common_jag.c:		FIELD_GPUMEM,
src/plugins/jobacct_gather/common/common_jag.c:		FIELD_GPUUTIL,
src/plugins/jobacct_gather/common/common_jag.c:		{ "GPUMemMB", PROFILE_FIELD_UINT64 },
src/plugins/jobacct_gather/common/common_jag.c:		{ "GPUUtilization", PROFILE_FIELD_DOUBLE },
src/plugins/jobacct_gather/common/common_jag.c:	static int gpumem_pos = -1;
src/plugins/jobacct_gather/common/common_jag.c:	static int gpuutil_pos = -1;
src/plugins/jobacct_gather/common/common_jag.c:		gpu_get_tres_pos(&gpumem_pos, &gpuutil_pos);
src/plugins/jobacct_gather/common/common_jag.c:		data[FIELD_GPUUTIL].d = 0.0;
src/plugins/jobacct_gather/common/common_jag.c:		if (gpumem_pos != -1) {
src/plugins/jobacct_gather/common/common_jag.c:			/* Profile gpumem as MB */
src/plugins/jobacct_gather/common/common_jag.c:			data[FIELD_GPUMEM].u64 =
src/plugins/jobacct_gather/common/common_jag.c:				jobacct->tres_usage_in_tot[gpumem_pos] /
src/plugins/jobacct_gather/common/common_jag.c:			data[FIELD_GPUUTIL].d =
src/plugins/jobacct_gather/common/common_jag.c:				jobacct->tres_usage_in_tot[gpuutil_pos];
src/plugins/topology/block/eval_nodes_block.c:				 * To many restricted gpu cores were removed
src/plugins/topology/tree/topology_tree.c: *  Copyright (C) 2023 NVIDIA CORPORATION.
src/plugins/topology/tree/eval_nodes_tree.c:				 * Too many restricted gpu cores removed due to
src/plugins/topology/common/gres_sched.c:					   uint16_t res_cores_per_gpu,
src/plugins/topology/common/gres_sched.c:	if (!gres_js->res_gpu_cores ||
src/plugins/topology/common/gres_sched.c:	    !gres_js->res_gpu_cores[node_i])
src/plugins/topology/common/gres_sched.c:	max_res_cores = *gres_limit * res_cores_per_gpu;
src/plugins/topology/common/gres_sched.c:	res_cores = bit_copy(gres_js->res_gpu_cores[node_i]);
src/plugins/topology/common/gres_sched.c:				max_res_cores = *gres_limit * res_cores_per_gpu;
src/plugins/topology/common/gres_sched.c: * IN res_cores_per_gpu - Number of restricted cores per gpu
src/plugins/topology/common/gres_sched.c:			   uint16_t res_cores_per_gpu,
src/plugins/topology/common/gres_sched.c:		if ((gres_state_job->plugin_id == gres_get_gpu_plugin_id()) &&
src/plugins/topology/common/gres_sched.c:		    res_cores_per_gpu) {
src/plugins/topology/common/gres_sched.c:				&gres_limit, gres_js, res_cores_per_gpu,
src/plugins/topology/common/eval_nodes.h: *  IN enforce_binding - Enforce GPU Binding or not
src/plugins/topology/common/eval_nodes.h: * This also check as the gres gets layed out that if restricted gpu cores are
src/plugins/topology/common/eval_nodes.h: * This is to allow for restricted cores per gpu.
src/plugins/topology/common/eval_nodes.h: * If the GRES available gets reduced and RestrictedCoresPerGPU
src/plugins/topology/common/common_topo.c:	 * GPU count if using GPUs, otherwise the CPU count) and retry
src/plugins/topology/common/eval_nodes.c:			      uint16_t res_cores_per_gpu,
src/plugins/topology/common/eval_nodes.c:		if ((gres_job_state->plugin_id != gres_get_gpu_plugin_id()) ||
src/plugins/topology/common/eval_nodes.c:		    !gres_js->res_gpu_cores ||
src/plugins/topology/common/eval_nodes.c:		    !gres_js->res_gpu_cores[node_i])
src/plugins/topology/common/eval_nodes.c:		max_res_cores = max_gres * res_cores_per_gpu;
src/plugins/topology/common/eval_nodes.c:		res_cores = bit_copy(gres_js->res_gpu_cores[node_i]);
src/plugins/topology/common/eval_nodes.c:				max_res_cores = max_gres * res_cores_per_gpu;
src/plugins/topology/common/eval_nodes.c:	uint16_t res_cores_per_gpu =  node_ptr->res_cores_per_gpu;
src/plugins/topology/common/eval_nodes.c:		_reduce_res_cores(topo_eval, &maxtasks, res_cores_per_gpu,
src/plugins/topology/common/eval_nodes.c: * If the GRES available gets reduced and RestrictedCoresPerGPU
src/plugins/topology/common/eval_nodes.c:			node_ptr->res_cores_per_gpu,
src/plugins/topology/common/eval_nodes.c:					avail_res_array[i]->avail_gpus;
src/plugins/topology/common/eval_nodes.c:				   avail_res->avail_gpus;
src/plugins/topology/common/eval_nodes.c:		/* Node ranges not allowed with --ntasks-per-gpu */
src/plugins/topology/common/eval_nodes.c:			node_ptr->res_cores_per_gpu,
src/plugins/topology/common/gres_filter.h:				  uint16_t res_cores_per_gpu,
src/plugins/topology/common/gres_sched.h: * IN res_cores_per_gpu - Number of restricted cores per gpu
src/plugins/topology/common/gres_sched.h:			   uint16_t res_cores_per_gpu,
src/plugins/topology/common/gres_filter.c:		 * sockets as the GPU
src/plugins/topology/common/gres_filter.c:			 * attempt giving each GPU same number of CPUs
src/plugins/topology/common/gres_filter.c:			 * For instance --gpus=8 -n2 -c8 will result in
src/plugins/topology/common/gres_filter.c:			 * in case of ---gpus=8 -n2 -c3 we don't attempt that
src/plugins/topology/common/gres_filter.c:	 * min_core_cnt partial res_cores_per_gpu might be left.
src/plugins/topology/common/gres_filter.c:				  uint16_t res_cores_per_gpu,
src/plugins/topology/common/gres_filter.c:		bool is_res_gpu = false;
src/plugins/topology/common/gres_filter.c:		     gres_get_gpu_plugin_id()) &&
src/plugins/topology/common/gres_filter.c:		    res_cores_per_gpu && gres_js->res_gpu_cores &&
src/plugins/topology/common/gres_filter.c:		    gres_js->res_gpu_cores[node_i])
src/plugins/topology/common/gres_filter.c:			is_res_gpu = true;
src/plugins/topology/common/gres_filter.c:		if (is_res_gpu) {
src/plugins/topology/common/gres_filter.c:				bit_copy(gres_js->res_gpu_cores[node_i]);
src/plugins/topology/common/gres_filter.c:			 * DefCPUPerGPU to be cores instead of cpus.
src/plugins/topology/common/gres_filter.c:						if (is_res_gpu &&
src/plugins/topology/common/gres_filter.c:							res_gpu_cores[node_i],
src/plugins/topology/common/gres_filter.c:						if (is_res_gpu) {
src/plugins/topology/common/gres_filter.c:						     res_gpu_cores[node_i],
src/plugins/topology/common/gres_filter.c:		 * If the gres is a gpu and RestrictedCoresPerGPU is configured
src/plugins/topology/common/gres_filter.c:		if (is_res_gpu) {
src/plugins/topology/common/gres_filter.c:			max_res_cores = cnt_avail_total * res_cores_per_gpu;
src/plugins/topology/common/gres_filter.c:						 gres_js->res_gpu_cores[node_i],
src/plugins/topology/common/gres_filter.c:				if (is_res_gpu) {
src/plugins/topology/common/gres_filter.c:						res_cores_per_gpu;
src/plugins/topology/common/gres_filter.c:						gres_js->res_gpu_cores[node_i],
src/plugins/topology/common/gres_filter.c:					if (is_res_gpu) {
src/plugins/topology/common/gres_filter.c:				if (is_res_gpu) {
src/api/config_info.c:	add_key_pair(ret_list, "GpuFreqDef", "%s", conf->gpu_freq_def);
src/api/node_info.c:	/* cores per gpu (optional) */
src/api/node_info.c:	if (node_ptr->res_cores_per_gpu) {
src/api/node_info.c:		xstrfmtcat(out, "RestrictedCoresPerGPU=%u(%s) ",
src/api/node_info.c:			   node_ptr->res_cores_per_gpu, node_ptr->gpu_spec);
src/salloc/opt.c:  { "SALLOC_CPUS_PER_GPU", LONG_OPT_CPUS_PER_GPU },
src/salloc/opt.c:  { "SALLOC_GPUS", 'G' },
src/salloc/opt.c:  { "SALLOC_GPU_BIND", LONG_OPT_GPU_BIND },
src/salloc/opt.c:  { "SALLOC_GPU_FREQ", LONG_OPT_GPU_FREQ },
src/salloc/opt.c:  { "SALLOC_GPUS_PER_NODE", LONG_OPT_GPUS_PER_NODE },
src/salloc/opt.c:  { "SALLOC_GPUS_PER_SOCKET", LONG_OPT_GPUS_PER_SOCKET },
src/salloc/opt.c:  { "SALLOC_GPUS_PER_TASK", LONG_OPT_GPUS_PER_TASK },
src/salloc/opt.c:  { "SALLOC_MEM_PER_GPU", LONG_OPT_MEM_PER_GPU },
src/salloc/opt.c:	if ((opt.ntasks_per_gpu != NO_VAL) &&
src/salloc/opt.c:	    (getenv("SLURM_NTASKS_PER_GPU") == NULL)) {
src/salloc/opt.c:		setenvf(NULL, "SLURM_NTASKS_PER_GPU", "%d",
src/salloc/opt.c:			opt.ntasks_per_gpu);
src/salloc/opt.c:"              [--cpus-per-gpu=n] [--gpus=n] [--gpu-bind=...] [--gpu-freq=...]\n"
src/salloc/opt.c:"              [--gpus-per-node=n] [--gpus-per-socket=n] [--gpus-per-task=n]\n"
src/salloc/opt.c:"              [--mem-per-gpu=MB] [--tres-bind=...] [--tres-per-task=list]\n"
src/salloc/opt.c:"GPU scheduling options:\n"
src/salloc/opt.c:"      --cpus-per-gpu=n        number of CPUs required per allocated GPU\n"
src/salloc/opt.c:"  -G, --gpus=n                count of GPUs required for the job\n"
src/salloc/opt.c:"      --gpu-bind=...          task to gpu binding options\n"
src/salloc/opt.c:"      --gpu-freq=...          frequency and voltage of GPUs\n"
src/salloc/opt.c:"      --gpus-per-node=n       number of GPUs required per allocated node\n"
src/salloc/opt.c:"      --gpus-per-socket=n     number of GPUs required per allocated socket\n"
src/salloc/opt.c:"      --gpus-per-task=n       number of GPUs required per spawned task\n"
src/salloc/opt.c:"      --mem-per-gpu=n         real memory required per allocated GPU\n"
src/slurmctld/job_mgr.c:	 * This won't work if two different gres names (for example, "gpu" and
src/slurmctld/job_mgr.c:	 * mem_per_gres for GPU so this works.
src/slurmctld/job_mgr.c:	 * for all types (e.g., gpu:k80 vs gpu:tesla) of that same gres (gpu).
src/slurmctld/controller.c:				      "(i.e. Gres/GPU).  You gave %s",
src/slurmctld/node_mgr.c:					node_ptr->res_cores_per_gpu =
src/slurmctld/node_mgr.c:						res_cores_per_gpu;
src/slurmctld/node_mgr.c:			node_ptr->gpu_spec_bitmap =
src/slurmctld/node_mgr.c:				node_state_rec->gpu_spec_bitmap;
src/slurmctld/node_mgr.c:			node_state_rec->gpu_spec_bitmap = NULL;
src/slurmctld/node_mgr.c:			node_ptr->res_cores_per_gpu =
src/slurmctld/node_mgr.c:				node_state_rec->res_cores_per_gpu;
src/slurmctld/node_mgr.c:			node_ptr->gpu_spec_bitmap =
src/slurmctld/node_mgr.c:				node_state_rec->gpu_spec_bitmap;
src/slurmctld/node_mgr.c:			node_state_rec->gpu_spec_bitmap = NULL;
src/slurmctld/node_mgr.c:	    (c1->res_cores_per_gpu == c2->res_cores_per_gpu) &&
src/slurmctld/node_mgr.c:		packstr(dump_node_ptr->gpu_spec, buffer);
src/slurmctld/node_mgr.c:		pack16(dump_node_ptr->res_cores_per_gpu, buffer);
src/slurmctld/node_mgr.c:	new_config_ptr->res_cores_per_gpu = config_ptr->res_cores_per_gpu;
src/slurmctld/node_mgr.c:static int _set_gpu_spec(node_record_t *node_ptr, char **reason_down)
src/slurmctld/node_mgr.c:	static uint32_t gpu_plugin_id = NO_VAL;
src/slurmctld/node_mgr.c:	uint32_t res_cnt = node_ptr->res_cores_per_gpu;
src/slurmctld/node_mgr.c:	xfree(node_ptr->gpu_spec);
src/slurmctld/node_mgr.c:	FREE_NULL_BITMAP(node_ptr->gpu_spec_bitmap);
src/slurmctld/node_mgr.c:	if (gpu_plugin_id == NO_VAL)
src/slurmctld/node_mgr.c:		gpu_plugin_id = gres_build_id("gpu");
src/slurmctld/node_mgr.c:						&gpu_plugin_id))) {
src/slurmctld/node_mgr.c:		/* No GPUs but we throught there were */
src/slurmctld/node_mgr.c:		xstrfmtcat(*reason_down, "%sRestrictedCoresPerGPU=%u but no gpus on node %s",
src/slurmctld/node_mgr.c:		return ESLURM_RES_CORES_PER_GPU_NO;
src/slurmctld/node_mgr.c:		xstrfmtcat(*reason_down, "%sRestrictedCoresPerGPU=%u but the gpus given don't have any topology on node %s.",
src/slurmctld/node_mgr.c:		return ESLURM_RES_CORES_PER_GPU_TOPO;
src/slurmctld/node_mgr.c:	node_ptr->gpu_spec_bitmap = bit_alloc(node_ptr->tot_cores);
src/slurmctld/node_mgr.c:		uint32_t this_gpu_res_cnt;
src/slurmctld/node_mgr.c:		this_gpu_res_cnt = res_cnt * gres_ns->topo_gres_cnt_avail[i];
src/slurmctld/node_mgr.c:			if (bit_test(node_ptr->gpu_spec_bitmap, j))
src/slurmctld/node_mgr.c:			bit_set(node_ptr->gpu_spec_bitmap, j);
src/slurmctld/node_mgr.c:			if (++cnt >= this_gpu_res_cnt)
src/slurmctld/node_mgr.c:		if (cnt != this_gpu_res_cnt) {
src/slurmctld/node_mgr.c:			FREE_NULL_BITMAP(node_ptr->gpu_spec_bitmap);
src/slurmctld/node_mgr.c:			xstrfmtcat(*reason_down, "%sRestrictedCoresPerGPU: We can't restrict %u core(s) per gpu. GPU %s(%d) doesn't have access to that many unique cores (%d).",
src/slurmctld/node_mgr.c:			return ESLURM_RES_CORES_PER_GPU_UNIQUE;
src/slurmctld/node_mgr.c:	/* info("set %s", bit_fmt_full(node_ptr->gpu_spec_bitmap)); */
src/slurmctld/node_mgr.c:	node_ptr->gpu_spec = bit_fmt_full(node_ptr->gpu_spec_bitmap);
src/slurmctld/node_mgr.c:	bit_not(node_ptr->gpu_spec_bitmap);
src/slurmctld/node_mgr.c:	/* info("sending back %s", bit_fmt_full(node_ptr->gpu_spec_bitmap)); */
src/slurmctld/node_mgr.c:	if (node_ptr->res_cores_per_gpu) {
src/slurmctld/node_mgr.c:		 * We need to make gpu_spec_bitmap now that we know the cores
src/slurmctld/node_mgr.c:		error_code = _set_gpu_spec(node_ptr, &reason_down);
src/slurmctld/node_mgr.c:		FREE_NULL_BITMAP(node_ptr->gpu_spec_bitmap);
src/slurmctld/proc_req.c:	conf_ptr->gpu_freq_def        = xstrdup(conf->gpu_freq_def);
src/slurmctld/node_scheduler.c:	uint64_t gpu_cnt;
src/slurmctld/node_scheduler.c:} foreach_node_gpu_args_t;
src/slurmctld/node_scheduler.c:static int _get_node_gpu_sum(void *x, void *arg)
src/slurmctld/node_scheduler.c:	foreach_node_gpu_args_t *args = arg;
src/slurmctld/node_scheduler.c:	if (gres_job_state->plugin_id != gres_get_gpu_plugin_id())
src/slurmctld/node_scheduler.c:	args->gpu_cnt += gres_js->gres_cnt_node_select[args->node_inx];
src/slurmctld/node_scheduler.c:static uint64_t _get_max_node_gpu_cnt(bitstr_t *node_bitmap, list_t* gres_list)
src/slurmctld/node_scheduler.c:	foreach_node_gpu_args_t args;
src/slurmctld/node_scheduler.c:	uint64_t max_node_gpu_cnt = 0;
src/slurmctld/node_scheduler.c:		args.gpu_cnt = 0;
src/slurmctld/node_scheduler.c:		/* Get the sum of all gpu types on the node */
src/slurmctld/node_scheduler.c:		list_for_each(gres_list, _get_node_gpu_sum, &args);
src/slurmctld/node_scheduler.c:		max_node_gpu_cnt = MAX(max_node_gpu_cnt, args.gpu_cnt);
src/slurmctld/node_scheduler.c:	return max_node_gpu_cnt;
src/slurmctld/node_scheduler.c:			uint64_t max_gpu_per_node =
src/slurmctld/node_scheduler.c:				_get_max_node_gpu_cnt(
src/slurmctld/node_scheduler.c:			if (max_gpu_per_node > slurm_conf.max_tasks_per_node)
src/slurmctld/node_scheduler.c:				max_gpu_per_node =
src/slurmctld/node_scheduler.c:				(uint16_t) max_gpu_per_node *
src/slurmctld/node_scheduler.c: * 0x000000000## - Reserved for cons_tres, favor nodes with co-located CPU/GPU
src/slurmd/slurmd/slurmd.c:#include "src/interfaces/gpu.h"
src/slurmd/slurmd/slurmd.c:		.gres_name = "gpu",
src/slurmd/slurmd/slurmd.c:	gres_get_autodetected_gpus(node_conf, &gres_str, &autodetect_str);
src/slurmd/slurmstepd/slurmstepd.c:#include "src/interfaces/gpu.h"
src/slurmd/slurmstepd/mgr.c:#include "src/interfaces/gpu.h"
src/slurmd/slurmstepd/mgr.c:	int gpumem_pos = -1, gpuutil_pos = -1;
src/slurmd/slurmstepd/mgr.c:	gpu_get_tres_pos(&gpumem_pos, &gpuutil_pos);
src/slurmd/slurmstepd/mgr.c:	 * Max to the total (i.e. Mem, VMem, gpumem, gpuutil) since the total
src/slurmd/slurmstepd/mgr.c:	if (gpumem_pos != -1)
src/slurmd/slurmstepd/mgr.c:		from->tres_usage_in_tot[gpumem_pos] =
src/slurmd/slurmstepd/mgr.c:			from->tres_usage_in_max[gpumem_pos];
src/slurmd/slurmstepd/mgr.c:	if (gpuutil_pos != -1)
src/slurmd/slurmstepd/mgr.c:		from->tres_usage_in_tot[gpuutil_pos] =
src/slurmd/slurmstepd/mgr.c:			from->tres_usage_in_max[gpuutil_pos];
src/slurmd/slurmstepd/mgr.c:		uint64_t gpu_cnt, nic_cnt;
src/slurmd/slurmstepd/mgr.c:		gpu_cnt = gres_step_count(step->step_gres_list, "gpu");
src/slurmd/slurmstepd/mgr.c:		if ((gpu_cnt <= 1) || (gpu_cnt == NO_VAL64))
src/slurmd/slurmstepd/mgr.c:			step->accel_bind_type &= (~ACCEL_BIND_CLOSEST_GPU);
src/slurmd/slurmstepd/mgr.c:	 * Reset GRES hardware, if needed. This is where GPU frequency is reset.
src/slurmd/slurmstepd/mgr.c:	 * the GPUs allocated to the step (and eventually other GRES hardware
src/slurmd/slurmstepd/mgr.c:		/* Handle GpuFreqDef option */
src/slurmd/slurmstepd/mgr.c:		if (!step->tres_freq && slurm_conf.gpu_freq_def) {
src/slurmd/slurmstepd/mgr.c:			debug("Setting GPU to GpuFreqDef=%s",
src/slurmd/slurmstepd/mgr.c:			      slurm_conf.gpu_freq_def);
src/slurmd/slurmstepd/mgr.c:			xstrfmtcat(step->tres_freq, "gpu:%s",
src/slurmd/slurmstepd/mgr.c:				   slurm_conf.gpu_freq_def);
src/slurmd/common/slurmstepd_init.c:	packstr(slurm_conf.gpu_freq_def, buffer);
src/slurmd/common/slurmstepd_init.c:	safe_unpackstr(&slurm_conf.gpu_freq_def, buffer);
src/interfaces/Makefile.in:	gpu.lo gres.lo hash.lo jobacct_gather.lo jobcomp.lo mcs.lo \
src/interfaces/Makefile.in:	./$(DEPDIR)/gpu.Plo ./$(DEPDIR)/gres.Plo ./$(DEPDIR)/hash.Plo \
src/interfaces/Makefile.in:	gpu.c					\
src/interfaces/Makefile.in:	gpu.h					\
src/interfaces/Makefile.in:@AMDEP_TRUE@@am__include@ @am__quote@./$(DEPDIR)/gpu.Plo@am__quote@ # am--include-marker
src/interfaces/Makefile.in:	-rm -f ./$(DEPDIR)/gpu.Plo
src/interfaces/Makefile.in:	-rm -f ./$(DEPDIR)/gpu.Plo
src/interfaces/select.h:	uint16_t avail_gpus;	/* Count of available GPUs */
src/interfaces/select.h:	uint16_t avail_res_cnt;	/* Count of available CPUs + GPUs */
src/interfaces/gpu.c: *  gpu.c - driver for gpu plugin
src/interfaces/gpu.c:#include "src/interfaces/gpu.h"
src/interfaces/gpu.c:	list_t *(*get_system_gpu_list) 	(node_config_load_t *node_conf);
src/interfaces/gpu.c:	void	(*step_hardware_init)	(bitstr_t *usable_gpus,
src/interfaces/gpu.c:	int     (*energy_read)          (uint32_t dv_ind, gpu_status_t *gpu);
src/interfaces/gpu.c:	"gpu_p_get_system_gpu_list",
src/interfaces/gpu.c:	"gpu_p_step_hardware_init",
src/interfaces/gpu.c:	"gpu_p_step_hardware_fini",
src/interfaces/gpu.c:	"gpu_p_test_cpu_conv",
src/interfaces/gpu.c:	"gpu_p_energy_read",
src/interfaces/gpu.c:	"gpu_p_get_device_count",
src/interfaces/gpu.c:	"gpu_p_usage_read",
src/interfaces/gpu.c: *  Common function to dlopen() the appropriate gpu libraries, and
src/interfaces/gpu.c:static char *_get_gpu_type(void)
src/interfaces/gpu.c:	 *  Here we are dlopening the gpu .so to verify it exists on this node.
src/interfaces/gpu.c:	if (autodetect_flags & GRES_AUTODETECT_GPU_NVML) {
src/interfaces/gpu.c:		if (!dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL))
src/interfaces/gpu.c:			return "gpu/nvml";
src/interfaces/gpu.c:	} else if (autodetect_flags & GRES_AUTODETECT_GPU_RSMI) {
src/interfaces/gpu.c:		if (!dlopen("librocm_smi64.so", RTLD_NOW | RTLD_GLOBAL))
src/interfaces/gpu.c:			return "gpu/rsmi";
src/interfaces/gpu.c:	} else if (autodetect_flags & GRES_AUTODETECT_GPU_ONEAPI) {
src/interfaces/gpu.c:			return "gpu/oneapi";
src/interfaces/gpu.c:	} else if (autodetect_flags & GRES_AUTODETECT_GPU_NRT) {
src/interfaces/gpu.c:		return "gpu/nrt";
src/interfaces/gpu.c:	} else if (autodetect_flags & GRES_AUTODETECT_GPU_NVIDIA) {
src/interfaces/gpu.c:		return "gpu/nvidia";
src/interfaces/gpu.c:	return "gpu/generic";
src/interfaces/gpu.c:extern int gpu_plugin_init(void)
src/interfaces/gpu.c:	char *plugin_type = "gpu";
src/interfaces/gpu.c:	type = _get_gpu_type();
src/interfaces/gpu.c:extern int gpu_plugin_fini(void)
src/interfaces/gpu.c:extern void gpu_get_tres_pos(int *gpumem_pos, int *gpuutil_pos)
src/interfaces/gpu.c:	static int loc_gpumem_pos = -1;
src/interfaces/gpu.c:	static int loc_gpuutil_pos = -1;
src/interfaces/gpu.c:		tres_rec.name = "gpuutil";
src/interfaces/gpu.c:		loc_gpuutil_pos = assoc_mgr_find_tres_pos(&tres_rec, false);
src/interfaces/gpu.c:		tres_rec.name = "gpumem";
src/interfaces/gpu.c:		loc_gpumem_pos = assoc_mgr_find_tres_pos(&tres_rec, false);
src/interfaces/gpu.c:	if (gpumem_pos)
src/interfaces/gpu.c:		*gpumem_pos = loc_gpumem_pos;
src/interfaces/gpu.c:	if (gpuutil_pos)
src/interfaces/gpu.c:		*gpuutil_pos = loc_gpuutil_pos;
src/interfaces/gpu.c:extern list_t *gpu_g_get_system_gpu_list(node_config_load_t *node_conf)
src/interfaces/gpu.c:	return (*(ops.get_system_gpu_list))(node_conf);
src/interfaces/gpu.c:extern void gpu_g_step_hardware_init(bitstr_t *usable_gpus, char *tres_freq)
src/interfaces/gpu.c:	(*(ops.step_hardware_init))(usable_gpus, tres_freq);
src/interfaces/gpu.c:extern void gpu_g_step_hardware_fini(void)
src/interfaces/gpu.c:extern char *gpu_g_test_cpu_conv(char *cpu_range)
src/interfaces/gpu.c:extern int gpu_g_energy_read(uint32_t dv_ind, gpu_status_t *gpu)
src/interfaces/gpu.c:	return (*(ops.energy_read))(dv_ind, gpu);
src/interfaces/gpu.c:extern void gpu_g_get_device_count(uint32_t *device_count)
src/interfaces/gpu.c:extern int gpu_g_usage_read(pid_t pid, acct_gather_data_t *data)
src/interfaces/Makefile.am:	gpu.c					\
src/interfaces/Makefile.am:	gpu.h					\
src/interfaces/gres.c:#include "src/interfaces/gpu.h"
src/interfaces/gres.c:	char *		gres_name;		/* name (e.g. "gpu") */
src/interfaces/gres.c:	char *		gres_name_colon;	/* name + colon (e.g. "gpu:") */
src/interfaces/gres.c:	char *		gres_type;		/* plugin name (e.g. "gres/gpu") */
src/interfaces/gres.c:	bool no_gpu_env;
src/interfaces/gres.c:static uint32_t gpu_plugin_id = NO_VAL;
src/interfaces/gres.c:			       bool *updated_gpu_cnt);
src/interfaces/gres.c:	bool have_gpu = false, have_shared = false;
src/interfaces/gres.c:	/* Ensure that "gres/'shared'" follows "gres/gpu" */
src/interfaces/gres.c:	have_gpu = false;
src/interfaces/gres.c:			if (!have_gpu) {
src/interfaces/gres.c:				/* "shared" must follow "gpu" */
src/interfaces/gres.c:		} else if (!xstrcmp(one_name, "gpu")) {
src/interfaces/gres.c:			have_gpu = true;
src/interfaces/gres.c:			gpu_plugin_id = gres_build_id("gpu");
src/interfaces/gres.c:		if (!have_gpu)
src/interfaces/gres.c:			fatal("GresTypes: gres/'shared' requires that gres/gpu also be configured");
src/interfaces/gres.c:		warning("Ignoring file-less GPU %s:%s from final GRES list",
src/interfaces/gres.c: * given GPU position indicated by index. Caller must xfree() the returned
src/interfaces/gres.c: * Used to record the enumeration order (PCI bus ID order) of GPUs for sorting,
src/interfaces/gres.c: * even when the GPU does not support nvlinks. E.g. for three total GPUs, their
src/interfaces/gres.c: * GPU at index 0: -1,0,0
src/interfaces/gres.c: * GPU at index 1: 0,-1,0
src/interfaces/gres.c: * GPU at index 2: -0,0,-1
src/interfaces/gres.c: * the GPU (-1) in the links string.
src/interfaces/gres.c: * Returns a non-zero-based index of the GPU in the links string, if found.
src/interfaces/gres.c: * 0+: GPU index
src/interfaces/gres.c: *     * the 'self' GPU identifier isn't found (i.e. no -1)
src/interfaces/gres.c: *     * there is more than one 'self' GPU identifier found
src/interfaces/gres.c:	/* If the current GPU (-1) wasn't found, that's an error */
src/interfaces/gres.c:	if (!(autodetect_flags & GRES_AUTODETECT_GPU_FLAGS))
src/interfaces/gres.c:		if (autodetect_flags & GRES_AUTODETECT_GPU_NVML)
src/interfaces/gres.c:		else if (autodetect_flags & GRES_AUTODETECT_GPU_RSMI)
src/interfaces/gres.c:		else if (autodetect_flags & GRES_AUTODETECT_GPU_ONEAPI)
src/interfaces/gres.c:		else if (autodetect_flags & GRES_AUTODETECT_GPU_NRT)
src/interfaces/gres.c:		else if (autodetect_flags & GRES_AUTODETECT_GPU_NVIDIA)
src/interfaces/gres.c:			xstrfmtcat(flags, "%snvidia", flags ? "," : "");
src/interfaces/gres.c:		else if (autodetect_flags & GRES_AUTODETECT_GPU_OFF)
src/interfaces/gres.c:	/* Set the node-local gpus value of autodetect_flags */
src/interfaces/gres.c:		flags |= GRES_AUTODETECT_GPU_NVML;
src/interfaces/gres.c:		flags |= GRES_AUTODETECT_GPU_RSMI;
src/interfaces/gres.c:		flags |= GRES_AUTODETECT_GPU_ONEAPI;
src/interfaces/gres.c:		flags |= GRES_AUTODETECT_GPU_NRT;
src/interfaces/gres.c:	else if (xstrcasestr(str, "nvidia"))
src/interfaces/gres.c:		flags |= GRES_AUTODETECT_GPU_NVIDIA;
src/interfaces/gres.c:		flags |= GRES_AUTODETECT_GPU_OFF;
src/interfaces/gres.c:	/* Set the node-local gpus value of autodetect_flags */
src/interfaces/gres.c:	/* If GPU flags exist, node-local value was already specified */
src/interfaces/gres.c:	if (autodetect_flags & GRES_AUTODETECT_GPU_FLAGS)
src/interfaces/gres.c:		debug2("gres.conf: AutoDetect GPU flags were locally set, so ignoring global flags");
src/interfaces/gres.c:	/* We only need to check type name because they should all be gpus */
src/interfaces/gres.c:extern void gres_get_autodetected_gpus(node_config_load_t node_conf,
src/interfaces/gres.c:		GRES_AUTODETECT_GPU_NVML,
src/interfaces/gres.c:		GRES_AUTODETECT_GPU_NVIDIA,
src/interfaces/gres.c:		GRES_AUTODETECT_GPU_RSMI,
src/interfaces/gres.c:		GRES_AUTODETECT_GPU_ONEAPI,
src/interfaces/gres.c:		GRES_AUTODETECT_GPU_NRT,
src/interfaces/gres.c:		if (gpu_plugin_init() != SLURM_SUCCESS)
src/interfaces/gres.c:		gres_list_system = gpu_g_get_system_gpu_list(&node_conf);
src/interfaces/gres.c:		gpu_plugin_fini();
src/interfaces/gres.c:		if (autodetect_flags == GRES_AUTODETECT_GPU_NVML)
src/interfaces/gres.c:			i++; /* Skip NVIDIA if NVML finds gpus */
src/interfaces/gres.c:		xstrfmtcat(*autodetect_str, "Found %s with Autodetect=%s (Substring of gpu name may be used instead)",
src/interfaces/gres.c: * Save off env flags, GRES name, and no_gpu_env (for the next gres.conf line to
src/interfaces/gres.c:				 bool no_gpu_env)
src/interfaces/gres.c:	prev_gres->no_gpu_env = no_gpu_env;
src/interfaces/gres.c:extern uint32_t gres_flags_parse(char *input, bool *no_gpu_env,
src/interfaces/gres.c:	if (xstrcasestr(input, "nvidia_gpu_env"))
src/interfaces/gres.c:	if (xstrcasestr(input, "amd_gpu_env"))
src/interfaces/gres.c:	if (xstrcasestr(input, "intel_gpu_env"))
src/interfaces/gres.c:	if (xstrcasestr(input, "opencl_env"))
src/interfaces/gres.c:		flags |= GRES_CONF_ENV_OPENCL;
src/interfaces/gres.c:	/* String 'no_gpu_env' will clear all GPU env vars */
src/interfaces/gres.c:	if (no_gpu_env)
src/interfaces/gres.c:		*no_gpu_env = xstrcasestr(input, "no_gpu_env");
src/interfaces/gres.c:		bool no_gpu_env = false;
src/interfaces/gres.c:		uint32_t flags = gres_flags_parse(tmp_str, &no_gpu_env,
src/interfaces/gres.c:		/* The default for MPS is to have only one gpu sharing */
src/interfaces/gres.c:		if (env_flags && no_gpu_env)
src/interfaces/gres.c:			fatal("Invalid GRES record name=%s type=%s: Flags (%s) contains \"no_gpu_env\", which must be mutually exclusive to all other GRES env flags of same node and name",
src/interfaces/gres.c:		     (prev_gres.no_gpu_env != no_gpu_env)))
src/interfaces/gres.c:				     no_gpu_env);
src/interfaces/gres.c:	} else if ((prev_gres.flags || prev_gres.no_gpu_env) &&
src/interfaces/gres.c:	/* Flags not set. By default, all env vars are set for GPUs */
src/interfaces/gres.c:	if (set_default_envs && !xstrcasecmp(p->name, "gpu")) {
src/interfaces/gres.c:		 * each GPU can have arbitrary count of MPS elements.
src/interfaces/gres.c:	/* E.g. slurm_conf_list will be NULL in the case of --gpu-bind */
src/interfaces/gres.c:	if (!xstrcasecmp(gres_ctx->gres_name, "gpu"))
src/interfaces/gres.c:	if ((rc = gpu_plugin_init()) != SLURM_SUCCESS)
src/interfaces/gres.c:		    !((autodetect_flags & GRES_AUTODETECT_GPU_FLAGS) &
src/interfaces/gres.c:		      GRES_AUTODETECT_GPU_OFF)) {
src/interfaces/gres.c:	/* Remove every GPU with an empty File */
src/interfaces/gres.c:			       &gpu_plugin_id);
src/interfaces/gres.c: * string for multiple types (e.g. "gres=gpu:kepler:1,gpu:tesla:2").
src/interfaces/gres.c: * IN gres_name - name of the gres type (e.g. "gpu")
src/interfaces/gres.c:				 * Type name, no count (e.g. "gpu:tesla").
src/interfaces/gres.c: *		  this name (e.g. GPU model)
src/interfaces/gres.c:		error("%s: %s If using AutoDetect the amount of GPUs configured in slurm.conf does not match what was detected. If this is intentional, please turn off AutoDetect and manually specify them in gres.conf.",
src/interfaces/gres.c:		 * slurmctld restart with running jobs. The number of gpus,
src/interfaces/gres.c:/* The GPU count on a node changed. Update SHARED data structures to match */
src/interfaces/gres.c:	gres_state_t *gres_state_node, *gres_gpu_ptr = NULL;
src/interfaces/gres.c:			gres_gpu_ptr = gres_state_node;
src/interfaces/gres.c:	_sync_node_shared_to_sharing(gres_gpu_ptr);
src/interfaces/gres.c:			  bool *updated_gpu_cnt)
src/interfaces/gres.c:	xassert(updated_gpu_cnt);
src/interfaces/gres.c:	*updated_gpu_cnt = false;
src/interfaces/gres.c:				*updated_gpu_cnt = true;
src/interfaces/gres.c:	gres_state_t *gpu_gres_state_node = NULL;
src/interfaces/gres.c:		bool updated_gpu_cnt = false;
src/interfaces/gres.c:				    &gres_context[i], &updated_gpu_cnt);
src/interfaces/gres.c:		if (updated_gpu_cnt)
src/interfaces/gres.c:			gpu_gres_state_node = gres_state_node;
src/interfaces/gres.c:	/* Now synchronize gres/gpu and gres/'shared' state */
src/interfaces/gres.c:	if (gpu_gres_state_node) {
src/interfaces/gres.c:		/* Update gres/'shared' counts and bitmaps to match gres/gpu */
src/interfaces/gres.c:		_sync_node_shared_to_sharing(gpu_gres_state_node);
src/interfaces/gres.c:	if (gres_js->res_gpu_cores) {
src/interfaces/gres.c:			FREE_NULL_BITMAP(gres_js->res_gpu_cores[i]);
src/interfaces/gres.c:		xfree(gres_js->res_gpu_cores);
src/interfaces/gres.c:		} else if (!xstrcmp(gres_state_job->gres_name, "gpu")) {
src/interfaces/gres.c:	bool requested_gpu = false;
src/interfaces/gres.c:	uint32_t cpus_per_gres = 0; /* At the moment its only for gpus */
src/interfaces/gres.c:			 * (e.g., gpu:k80 vs gpu:tesla) of that same gres (gpu)
src/interfaces/gres.c:			if (!requested_gpu &&
src/interfaces/gres.c:			    (!xstrcmp(gres_state_job->gres_name, "gpu")))
src/interfaces/gres.c:				requested_gpu = true;
src/interfaces/gres.c:			if (!requested_gpu &&
src/interfaces/gres.c:			    (!xstrcmp(gres_state_job->gres_name, "gpu")))
src/interfaces/gres.c:				requested_gpu = true;
src/interfaces/gres.c:			if (!requested_gpu &&
src/interfaces/gres.c:			    (!xstrcmp(gres_state_job->gres_name, "gpu")))
src/interfaces/gres.c:				requested_gpu = true;
src/interfaces/gres.c:			if (!requested_gpu &&
src/interfaces/gres.c:			    (!xstrcmp(gres_state_job->gres_name, "gpu")))
src/interfaces/gres.c:				requested_gpu = true;
src/interfaces/gres.c:	} else if (requested_gpu && list_count(*gres_js_val->gres_list)) {
src/interfaces/gres.c:		/* Set num_tasks = gpus * ntasks/gpu */
src/interfaces/gres.c:		uint64_t gpus = _get_job_gres_list_cnt(
src/interfaces/gres.c:			*gres_js_val->gres_list, "gpu", NULL);
src/interfaces/gres.c:		if (gpus != NO_VAL64)
src/interfaces/gres.c:				gpus * *gres_js_val->ntasks_per_tres;
src/interfaces/gres.c:			error("%s: Can't set num_tasks = gpus * *ntasks_per_tres because there are no allocated GPUs",
src/interfaces/gres.c:		 * then derive GPUs according to how many tasks there are.
src/interfaces/gres.c:		 * GPU GRES = [ntasks / (ntasks_per_tres)]
src/interfaces/gres.c:		 * For now, only generate type-less GPUs.
src/interfaces/gres.c:		uint32_t gpus = *gres_js_val->num_tasks /
src/interfaces/gres.c:		xstrfmtcat(gres, "gres/gpu:%u", gpus);
src/interfaces/gres.c:			requested_gpu = true;
src/interfaces/gres.c:		error("%s: --ntasks-per-tres needs either a GRES GPU specification or a node/ntask specification",
src/interfaces/gres.c:	 * If someone requested [mem|cpus]_per_tres but didn't request any GPUs
src/interfaces/gres.c:	 * GPUs since --[mem|cpus]-per-gpu are the only allowed
src/interfaces/gres.c:	 * GPUs are explicitly requested when --[mem|cpus]-per-gpu is used.
src/interfaces/gres.c:	if (mem_per_tres && (!requested_gpu)) {
src/interfaces/gres.c:		error("Requested mem_per_tres=%s but did not request any GPU.",
src/interfaces/gres.c:	if (cpus_per_tres && (!requested_gpu)) {
src/interfaces/gres.c:		error("Requested cpus_per_tres=%s but did not request any GPU.",
src/interfaces/gres.c:	 * Check for record overlap (e.g. "gpu:2,gpu:tesla:1")
src/interfaces/gres.c:		    (gres_state_job->plugin_id == gres_get_gpu_plugin_id()))
src/interfaces/gres.c:	    strstr(tres_freq, "gpu")) {
src/interfaces/gres.c: * NOTE: For gres/'shared' return count of gres/gpu
src/interfaces/gres.c:		plugin_id = gpu_plugin_id;
src/interfaces/gres.c:			plugin_id = gpu_plugin_id;
src/interfaces/gres.c:	if (gres_js->res_gpu_cores) {
src/interfaces/gres.c:		new_gres_js->res_gpu_cores = xcalloc(gres_js->res_array_size,
src/interfaces/gres.c:			if (gres_js->res_gpu_cores[i] == NULL)
src/interfaces/gres.c:			new_gres_js->res_gpu_cores[i] =
src/interfaces/gres.c:				bit_copy(gres_js->res_gpu_cores[i]);
src/interfaces/gres.c:				/* Does job have a sharing GRES (GPU)? */
src/interfaces/gres.c:		 * a GPU (sharing GRES) when a GPU is allocated but an
src/interfaces/gres.c:		 * shared GRES, so we don't need to protect MPS/Shard from GPU.
src/interfaces/gres.c:			 * case of MPS and GPU)
src/interfaces/gres.c:				 * like with GPU and MPS)
src/interfaces/gres.c:	uint64_t tmp = _get_step_gres_list_cnt(new_step_list, "gpu", NULL);
src/interfaces/gres.c:		 * Generate GPUs from ntasks_per_tres when not specified
src/interfaces/gres.c:		uint32_t gpus = *num_tasks / ntasks_per_tres;
src/interfaces/gres.c:		/* For now, do type-less GPUs */
src/interfaces/gres.c:		xstrfmtcat(gres, "gres/gpu:%u", gpus);
src/interfaces/gres.c:		if (*num_tasks != ntasks_per_tres * gpus) {
src/interfaces/gres.c:			log_flag(GRES, "%s: -n/--ntasks %u is not a multiple of --ntasks-per-gpu=%u",
src/interfaces/gres.c:		error("%s: ntasks_per_tres was specified, but there was either no task count or no GPU specification to go along with it, or both were already specified.",
src/interfaces/gres.c:		uint64_t gpu_cnt = _get_step_gres_list_cnt(new_step_list,
src/interfaces/gres.c:		if (gpu_cnt == NO_VAL64) {
src/interfaces/gres.c:			*cpu_count = gpu_cnt * cpus_per_gres;
src/interfaces/gres.c:	if (accel_bind_type & ACCEL_BIND_CLOSEST_GPU) {
src/interfaces/gres.c:		xstrfmtcat(tres_bind_str, "%sgres/gpu:closest",
src/interfaces/gres.c:		if (!xstrncasecmp(sep, "map_gpu:", 8)) { // Old Syntax
src/interfaces/gres.c:		} else if (!xstrncasecmp(sep, "mask_gpu:", 9)) { // Old Syntax
src/interfaces/gres.c:				step, gpu_plugin_id, proc_id);
src/interfaces/gres.c:			/* Does step have a sharing GRES (GPU)? */
src/interfaces/gres.c:		 * a GPU (sharing GRES) when a GPU is allocated but an
src/interfaces/gres.c:		 * shared GRES, so we don't need to protect MPS/Shard from GPU.
src/interfaces/gres.c: * gres_g_step_set_env()). Use this to implement GPU task binding.
src/interfaces/gres.c:			/* Does task have a sharing GRES (GPU)? */
src/interfaces/gres.c:		 * a GPU (sharing GRES) when a GPU is allocated but an
src/interfaces/gres.c:		 * shared GRES, so we don't need to protect MPS/Shard from GPU.
src/interfaces/gres.c: * consumes subsets of its resources (e.g. GPU)
src/interfaces/gres.c:	if (plugin_id == gpu_plugin_id)
src/interfaces/gres.c:	if (config_flags & GRES_CONF_ENV_OPENCL) {
src/interfaces/gres.c:		strcat(flag_str, "ENV_OPENCL");
src/interfaces/gres.c:/* Return the plugin id made from gres_build_id("gpu") */
src/interfaces/gres.c:extern uint32_t gres_get_gpu_plugin_id(void)
src/interfaces/gres.c:	return gpu_plugin_id;
src/interfaces/topology.h:	bool enforce_binding; /* Enforce GPU Binding or not */
src/interfaces/gres.h:	 * If a job/step/task has sharing GRES (GPU), don't let shared GRES
src/interfaces/gres.h:	char *unique_id; /* Used for GPU binding with MIGs */
src/interfaces/gres.h:#define GRES_CONF_ENV_NVML   SLURM_BIT(5) /* Set CUDA_VISIBLE_DEVICES */
src/interfaces/gres.h:#define GRES_CONF_ENV_OPENCL SLURM_BIT(7) /* Set GPU_DEVICE_ORDINAL */
src/interfaces/gres.h:#define GRES_AUTODETECT_GPU_NVML  0x00000001
src/interfaces/gres.h:#define GRES_AUTODETECT_GPU_RSMI  0x00000002
src/interfaces/gres.h:#define GRES_AUTODETECT_GPU_OFF   0x00000004 /* Do NOT use global */
src/interfaces/gres.h:#define GRES_AUTODETECT_GPU_ONEAPI 0x00000008
src/interfaces/gres.h:#define GRES_AUTODETECT_GPU_NRT 0x00000010
src/interfaces/gres.h:#define GRES_AUTODETECT_GPU_NVIDIA 0x00000020
src/interfaces/gres.h:#define GRES_AUTODETECT_GPU_FLAGS 0x000000ff /* reserve first 8 bits for gpu
src/interfaces/gres.h:	/* Used for GPU binding with MIGs */
src/interfaces/gres.h:	 * one topo record per file (GPU) and the size of the GRES bitmaps (i.e.
src/interfaces/gres.h:	 * GPUs on the node while the count is a site-configurable value.
src/interfaces/gres.h:	 * Only initialized for gpus. One entry per node on the cluster.
src/interfaces/gres.h:	 * gpu type has access to.
src/interfaces/gres.h:	bitstr_t **res_gpu_cores;
src/interfaces/gres.h:	char *gres_name;		/* GRES name (e.g. "gpu") */
src/interfaces/gres.h:	uint16_t ntasks_per_tres; /* number of tasks that can access each gpu */
src/interfaces/gres.h: * gres_g_step_set_env()). Use this to implement GPU task binding.
src/interfaces/gres.h:extern void gres_get_autodetected_gpus(node_config_load_t node_conf,
src/interfaces/gres.h:extern uint32_t gres_flags_parse(char *input, bool *no_gpu_env,
src/interfaces/gres.h: * consumes subsets of its resources (e.g. GPU)
src/interfaces/gres.h: * given GPU position indicated by index. Caller must xfree() the returned
src/interfaces/gres.h: * Used to record the enumeration order (PCI bus ID order) of GPUs for sorting,
src/interfaces/gres.h: * even when the GPU does not support nvlinks. E.g. for three total GPUs, their
src/interfaces/gres.h: * GPU at index 0: -1,0,0
src/interfaces/gres.h: * GPU at index 1: 0,-1,0
src/interfaces/gres.h: * GPU at index 2: 0,0,-1
src/interfaces/gres.h: * the GPU (-1) in the links string.
src/interfaces/gres.h: * Returns a non-zero-based index of the GPU in the links string, if found.
src/interfaces/gres.h: * 0+: GPU index
src/interfaces/gres.h: *     * the 'self' GPU identifier isn't found (i.e. no -1)
src/interfaces/gres.h: *     * there is more than one 'self' GPU identifier found
src/interfaces/gres.h:/* Return the plugin id made from gres_build_id("gpu") */
src/interfaces/gres.h:extern uint32_t gres_get_gpu_plugin_id(void);
src/interfaces/gpu.h: *  gpu.h - driver for gpu plugin
src/interfaces/gpu.h:#ifndef _INTERFACES_GPU_H
src/interfaces/gpu.h:#define _INTERFACES_GPU_H
src/interfaces/gpu.h:// array of struct to track the status of a GPU
src/interfaces/gpu.h:} gpu_status_t;
src/interfaces/gpu.h:extern int gpu_plugin_init(void);
src/interfaces/gpu.h:extern int gpu_plugin_fini(void);
src/interfaces/gpu.h:extern void gpu_get_tres_pos(int *gpumem_pos, int *gpuutil_pos);
src/interfaces/gpu.h:extern list_t *gpu_g_get_system_gpu_list(node_config_load_t *node_conf);
src/interfaces/gpu.h:extern void gpu_g_step_hardware_init(bitstr_t *usable_gpus, char *tres_freq);
src/interfaces/gpu.h:extern void gpu_g_step_hardware_fini(void);
src/interfaces/gpu.h:extern char *gpu_g_test_cpu_conv(char *cpu_range);
src/interfaces/gpu.h:extern int gpu_g_energy_read(uint32_t dv_ind, gpu_status_t *gpu);
src/interfaces/gpu.h:extern void gpu_g_get_device_count(uint32_t *device_count);
src/interfaces/gpu.h:extern int gpu_g_usage_read(pid_t pid, acct_gather_data_t *data);
src/sbatch/xlate.c:		if (!xstrncmp(node_opts+i, "gpus=", 5)) {
src/sbatch/xlate.c:						     LONG_OPT_GPUS_PER_NODE,
src/sbatch/xlate.c:	int gpus = 0;
src/sbatch/xlate.c:			if (!xstrncasecmp(rl+i, "true", 4) && (gpus < 1))
src/sbatch/xlate.c:				gpus = 1;
src/sbatch/xlate.c:				gpus = parse_int("naccelerators", temp, true);
src/sbatch/xlate.c:	if (gpus > 0) {
src/sbatch/xlate.c:			temp = xstrdup_printf("%s,gpu:%d", opt.gres, gpus);
src/sbatch/xlate.c:			temp = xstrdup_printf("gpu:%d", gpus);
src/sbatch/opt.c:  { "SBATCH_CPUS_PER_GPU", LONG_OPT_CPUS_PER_GPU },
src/sbatch/opt.c:  { "SBATCH_GPUS", 'G' },
src/sbatch/opt.c:  { "SBATCH_GPU_BIND", LONG_OPT_GPU_BIND },
src/sbatch/opt.c:  { "SBATCH_GPU_FREQ", LONG_OPT_GPU_FREQ },
src/sbatch/opt.c:  { "SBATCH_GPUS_PER_NODE", LONG_OPT_GPUS_PER_NODE },
src/sbatch/opt.c:  { "SBATCH_GPUS_PER_SOCKET", LONG_OPT_GPUS_PER_SOCKET },
src/sbatch/opt.c:  { "SBATCH_GPUS_PER_TASK", LONG_OPT_GPUS_PER_TASK },
src/sbatch/opt.c:  { "SBATCH_MEM_PER_GPU", LONG_OPT_MEM_PER_GPU },
src/sbatch/opt.c:	else if (opt.ntasks_per_gpu != NO_VAL)
src/sbatch/opt.c:		het_job_env.ntasks_per_gpu = opt.ntasks_per_gpu;
src/sbatch/opt.c:"              [--cpus-per-gpu=n] [--gpus=n] [--gpu-bind=...] [--gpu-freq=...]\n"
src/sbatch/opt.c:"              [--gpus-per-node=n] [--gpus-per-socket=n] [--gpus-per-task=n]\n"
src/sbatch/opt.c:"              [--mem-per-gpu=MB] [--tres-bind=...] [--tres-per-task=list]\n"
src/sbatch/opt.c:"GPU scheduling options:\n"
src/sbatch/opt.c:"      --cpus-per-gpu=n        number of CPUs required per allocated GPU\n"
src/sbatch/opt.c:"  -G, --gpus=n                count of GPUs required for the job\n"
src/sbatch/opt.c:"      --gpu-bind=...          task to gpu binding options\n"
src/sbatch/opt.c:"      --gpu-freq=...          frequency and voltage of GPUs\n"
src/sbatch/opt.c:"      --gpus-per-node=n       number of GPUs required per allocated node\n"
src/sbatch/opt.c:"      --gpus-per-socket=n     number of GPUs required per allocated socket\n"
src/sbatch/opt.c:"      --gpus-per-task=n       number of GPUs required per spawned task\n"
src/sbatch/opt.c:"      --mem-per-gpu=n         real memory required per allocated GPU\n"
src/sbatch/opt.c:	local_env->ntasks_per_gpu	= NO_VAL;
src/sbatch/opt.c:	if ((local_env->ntasks_per_gpu  != NO_VAL) &&
src/sbatch/opt.c:	    !env_array_overwrite_het_fmt(array_ptr, "SLURM_NTASKS_PER_GPU",
src/sbatch/opt.c:					 local_env->ntasks_per_gpu)) {
src/sbatch/opt.c:		error("Can't set SLURM_NTASKS_PER_GPU env variable");
src/sbatch/opt.h:	uint32_t ntasks_per_gpu;
src/common/slurmdb_defs.c:							"gpuutil")) {
src/common/slurmdb_defs.c:				   !xstrcasecmp(tres_rec->name, "gpumem") ||
src/common/env.c:	if (env->ntasks_per_gpu &&
src/common/env.c:	    setenvf(&env->env, "SLURM_NTASKS_PER_GPU", "%d",
src/common/env.c:		    env->ntasks_per_gpu)) {
src/common/env.c:		error("Unable to set SLURM_NTASKS_PER_GPU");
src/common/env.c:	if (opt->cpus_per_gpu) {
src/common/env.c:		env_array_overwrite_het_fmt(dest, "SLURM_CPUS_PER_GPU",
src/common/env.c:					    opt->cpus_per_gpu);
src/common/env.c:	if (opt->gpus) {
src/common/env.c:		env_array_overwrite_het_fmt(dest, "SLURM_GPUS",
src/common/env.c:					    opt->gpus);
src/common/env.c:	if (opt->gpu_freq) {
src/common/env.c:		env_array_overwrite_het_fmt(dest, "SLURM_GPU_FREQ",
src/common/env.c:					    opt->gpu_freq);
src/common/env.c:	if (opt->gpus_per_node) {
src/common/env.c:		env_array_overwrite_het_fmt(dest, "SLURM_GPUS_PER_NODE",
src/common/env.c:					    opt->gpus_per_node);
src/common/env.c:	if (opt->gpus_per_socket) {
src/common/env.c:		env_array_overwrite_het_fmt(dest, "SLURM_GPUS_PER_SOCKET",
src/common/env.c:					    opt->gpus_per_socket);
src/common/env.c:	if (opt->mem_per_gpu != NO_VAL64) {
src/common/env.c:		env_array_overwrite_het_fmt(dest, "SLURM_MEM_PER_GPU",
src/common/env.c:					    opt->mem_per_gpu);
src/common/job_record.h:	uint16_t ntasks_per_tres;	/* number of tasks on each GPU */
src/common/assoc_mgr.c: * So tres_rec->name of "gpu" can match accounting TRES name of "gpu:tesla".
src/common/slurm_protocol_defs.h:	uint16_t ntasks_per_tres;/* number of tasks that can access each gpu */
src/common/slurm_protocol_defs.h:	uint16_t  ntasks_per_tres; /* number of tasks that can access each gpu */
src/common/slurm_protocol_defs.h:	char     *tres_bind;	/* task binding to TRES (e.g. GPUs) */
src/common/slurm_protocol_defs.h:	char     *tres_freq;	/* frequency/power for TRES (e.g. GPUs) */
src/common/slurm_protocol_defs.h:	char *tres_bind;	/* task binding to TRES (e.g. GPUs),
src/common/slurm_protocol_defs.h:	char *tres_freq;	/* frequency/power for TRES (e.g. GPUs) */
src/common/slurm_opt.h:	LONG_OPT_CPUS_PER_GPU,
src/common/slurm_opt.h:	LONG_OPT_GPU_BIND,
src/common/slurm_opt.h:	LONG_OPT_GPU_FREQ,
src/common/slurm_opt.h:	LONG_OPT_GPUS,
src/common/slurm_opt.h:	LONG_OPT_GPUS_PER_NODE,
src/common/slurm_opt.h:	LONG_OPT_GPUS_PER_SOCKET,
src/common/slurm_opt.h:	LONG_OPT_GPUS_PER_TASK,
src/common/slurm_opt.h:	LONG_OPT_MEM_PER_GPU,
src/common/slurm_opt.h:	LONG_OPT_NTASKSPERGPU,
src/common/slurm_opt.h:	int ntasks_per_gpu;		/* --ntasks-per-gpu=n		*/
src/common/slurm_opt.h:	int ntasks_per_tres;		/* --ntasks-per-gpu=n	*/
src/common/slurm_opt.h:	int cpus_per_gpu;		/* --cpus-per-gpu		*/
src/common/slurm_opt.h:	char *gpus;			/* --gpus			*/
src/common/slurm_opt.h:	char *gpu_bind;			/* --gpu_bind			*/
src/common/slurm_opt.h:	char *gpu_freq;			/* --gpu_freq			*/
src/common/slurm_opt.h:	char *gpus_per_node;		/* --gpus_per_node		*/
src/common/slurm_opt.h:	char *gpus_per_socket;		/* --gpus_per_socket		*/
src/common/slurm_opt.h:	char *gpus_per_task;		/* --gpus_per_task		*/
src/common/slurm_opt.h:	uint64_t mem_per_gpu;		/* --mem-per-gpu		*/
src/common/slurm_opt.h:	char *tres_bind;		/* derived from gpu_bind	*/
src/common/slurm_opt.h:	char *tres_freq;		/* derived from gpu_freq	*/
src/common/tres_bind.h: * Example: gpu:closest
src/common/tres_bind.h: *          gpu:map:0,1
src/common/tres_bind.h: *          gpu:mask:0x3,0x3
src/common/tres_bind.h: *          gpu:map:0,1;nic:closest
src/common/tres_frequency.c: * Test for valid GPU frequency specification
src/common/tres_frequency.c:static int _valid_gpu_freq(const char *arg)
src/common/tres_frequency.c: * Example: gpu:medium,memory=high
src/common/tres_frequency.c: *          gpu:450
src/common/tres_frequency.c:		if (!strcmp(tok, "gpu")) {	/* Only support GPUs today */
src/common/tres_frequency.c:			if (_valid_gpu_freq(sep) != 0) {
src/common/slurm_errno.c:	{ ERRTAB_ENTRY(ESLURM_RES_CORES_PER_GPU_UNIQUE),
src/common/slurm_errno.c:	  "RestrictedCoresPerGPU: Not enough unique cores per GPU" },
src/common/slurm_errno.c:	{ ERRTAB_ENTRY(ESLURM_RES_CORES_PER_GPU_TOPO),
src/common/slurm_errno.c:	  "RestrictedCoresPerGPU: Missing core topology for GPUs" },
src/common/slurm_errno.c:	{ ERRTAB_ENTRY(ESLURM_RES_CORES_PER_GPU_NO),
src/common/slurm_errno.c:	  "RestrictedCoresPerGPU: No GPUs configured on node" },
src/common/read_config.h:	uint16_t res_cores_per_gpu; /* number of cores per GPU to allow
src/common/read_config.h:				     * to only GPU jobs */
src/common/list.c: *  Copyright (C) 2021 NVIDIA Corporation.
src/common/node_conf.h:	uint16_t res_cores_per_gpu; /* number of cores per GPU to allow
src/common/node_conf.h:				     * to only GPU jobs */
src/common/node_conf.h:	char *gpu_spec;                 /* node's cores reserved for GPU jobs */
src/common/node_conf.h:	bitstr_t *gpu_spec_bitmap;	/* node gpu core specialization
src/common/node_conf.h:	uint16_t res_cores_per_gpu;	/* number of cores per GPU to allow to
src/common/node_conf.h:					 * only GPU jobs */
src/common/tres_bind.c:	if (!xstrncasecmp(arg, "map_gpu:", 8) || //Old syntax
src/common/tres_bind.c:	if (!xstrncasecmp(arg, "mask_gpu:", 9) || //Old syntax
src/common/tres_bind.c: * Example: gres/gpu:closest
src/common/tres_bind.c: *          gres/gpu:single:2
src/common/tres_bind.c: *          gres/gpu:map:0,1
src/common/tres_bind.c: *          gres/gpu:mask:0x3,0x3s
src/common/tres_bind.c: *          gres/gpu:map:0,1+nic:closest
src/common/slurm_protocol_api.h: * returns the configured GpuFreqDef value
src/common/slurm_protocol_api.h: * RET char *    - GpuFreqDef value,  MUST be xfreed by caller
src/common/slurm_protocol_api.h:char *slurm_get_gpu_freq_def(void);
src/common/slurm_opt.c:		opt->srun_opt->accel_bind_type |= ACCEL_BIND_CLOSEST_GPU;
src/common/slurm_opt.c:	if (opt->srun_opt->accel_bind_type & ACCEL_BIND_CLOSEST_GPU)
src/common/slurm_opt.c:COMMON_INT_OPTION(cpus_per_gpu, "--cpus-per-gpu");
src/common/slurm_opt.c:static slurm_cli_opt_t slurm_opt_cpus_per_gpu = {
src/common/slurm_opt.c:	.name = "cpus-per-gpu",
src/common/slurm_opt.c:	.val = LONG_OPT_CPUS_PER_GPU,
src/common/slurm_opt.c:	.set_func = arg_set_cpus_per_gpu,
src/common/slurm_opt.c:	.get_func = arg_get_cpus_per_gpu,
src/common/slurm_opt.c:	.reset_func = arg_reset_cpus_per_gpu,
src/common/slurm_opt.c:static int arg_set_gpu_bind(slurm_opt_t *opt, const char *arg)
src/common/slurm_opt.c:	xfree(opt->gpu_bind);
src/common/slurm_opt.c:	opt->gpu_bind = xstrdup(arg);
src/common/slurm_opt.c:	xstrfmtcat(opt->tres_bind, "gres/gpu:%s", opt->gpu_bind);
src/common/slurm_opt.c:		error("Invalid --gpu-bind argument: %s", opt->gpu_bind);
src/common/slurm_opt.c:static void arg_reset_gpu_bind(slurm_opt_t *opt)
src/common/slurm_opt.c:	xfree(opt->gpu_bind);
src/common/slurm_opt.c:COMMON_STRING_OPTION_GET(gpu_bind);
src/common/slurm_opt.c:static slurm_cli_opt_t slurm_opt_gpu_bind = {
src/common/slurm_opt.c:	.name = "gpu-bind",
src/common/slurm_opt.c:	.val = LONG_OPT_GPU_BIND,
src/common/slurm_opt.c:	.set_func = arg_set_gpu_bind,
src/common/slurm_opt.c:	.get_func = arg_get_gpu_bind,
src/common/slurm_opt.c:	.reset_func = arg_reset_gpu_bind,
src/common/slurm_opt.c:static int arg_set_gpu_freq(slurm_opt_t *opt, const char *arg)
src/common/slurm_opt.c:	xfree(opt->gpu_freq);
src/common/slurm_opt.c:	opt->gpu_freq = xstrdup(arg);
src/common/slurm_opt.c:	xstrfmtcat(opt->tres_freq, "gpu:%s", opt->gpu_freq);
src/common/slurm_opt.c:		error("Invalid --gpu-freq argument: %s", opt->tres_freq);
src/common/slurm_opt.c:static void arg_reset_gpu_freq(slurm_opt_t *opt)
src/common/slurm_opt.c:	xfree(opt->gpu_freq);
src/common/slurm_opt.c:COMMON_STRING_OPTION_GET(gpu_freq);
src/common/slurm_opt.c:static slurm_cli_opt_t slurm_opt_gpu_freq = {
src/common/slurm_opt.c:	.name = "gpu-freq",
src/common/slurm_opt.c:	.val = LONG_OPT_GPU_FREQ,
src/common/slurm_opt.c:	.set_func = arg_set_gpu_freq,
src/common/slurm_opt.c:	.get_func = arg_get_gpu_freq,
src/common/slurm_opt.c:	.reset_func = arg_reset_gpu_freq,
src/common/slurm_opt.c:COMMON_STRING_OPTION(gpus);
src/common/slurm_opt.c:static slurm_cli_opt_t slurm_opt_gpus = {
src/common/slurm_opt.c:	.name = "gpus",
src/common/slurm_opt.c:	.set_func = arg_set_gpus,
src/common/slurm_opt.c:	.get_func = arg_get_gpus,
src/common/slurm_opt.c:	.reset_func = arg_reset_gpus,
src/common/slurm_opt.c:COMMON_STRING_OPTION(gpus_per_node);
src/common/slurm_opt.c:static slurm_cli_opt_t slurm_opt_gpus_per_node = {
src/common/slurm_opt.c:	.name = "gpus-per-node",
src/common/slurm_opt.c:	.val = LONG_OPT_GPUS_PER_NODE,
src/common/slurm_opt.c:	.set_func = arg_set_gpus_per_node,
src/common/slurm_opt.c:	.get_func = arg_get_gpus_per_node,
src/common/slurm_opt.c:	.reset_func = arg_reset_gpus_per_node,
src/common/slurm_opt.c:COMMON_STRING_OPTION(gpus_per_socket);
src/common/slurm_opt.c:static slurm_cli_opt_t slurm_opt_gpus_per_socket = {
src/common/slurm_opt.c:	.name = "gpus-per-socket",
src/common/slurm_opt.c:	.val = LONG_OPT_GPUS_PER_SOCKET,
src/common/slurm_opt.c:	.set_func = arg_set_gpus_per_socket,
src/common/slurm_opt.c:	.get_func = arg_get_gpus_per_socket,
src/common/slurm_opt.c:	.reset_func = arg_reset_gpus_per_socket,
src/common/slurm_opt.c:COMMON_STRING_OPTION(gpus_per_task);
src/common/slurm_opt.c:static slurm_cli_opt_t slurm_opt_gpus_per_task = {
src/common/slurm_opt.c:	.name = "gpus-per-task",
src/common/slurm_opt.c:	.val = LONG_OPT_GPUS_PER_TASK,
src/common/slurm_opt.c:	.set_func = arg_set_gpus_per_task,
src/common/slurm_opt.c:	.get_func = arg_get_gpus_per_task,
src/common/slurm_opt.c:	.reset_func = arg_reset_gpus_per_task,
src/common/slurm_opt.c:COMMON_MBYTES_OPTION(mem_per_gpu, --mem-per-gpu);
src/common/slurm_opt.c:static slurm_cli_opt_t slurm_opt_mem_per_gpu = {
src/common/slurm_opt.c:	.name = "mem-per-gpu",
src/common/slurm_opt.c:	.val = LONG_OPT_MEM_PER_GPU,
src/common/slurm_opt.c:	.set_func = arg_set_mem_per_gpu,
src/common/slurm_opt.c:	.get_func = arg_get_mem_per_gpu,
src/common/slurm_opt.c:	.reset_func = arg_reset_mem_per_gpu,
src/common/slurm_opt.c:COMMON_INT_OPTION_SET(ntasks_per_gpu, "--ntasks-per-gpu");
src/common/slurm_opt.c:COMMON_INT_OPTION_GET(ntasks_per_gpu);
src/common/slurm_opt.c:COMMON_OPTION_RESET(ntasks_per_gpu, NO_VAL);
src/common/slurm_opt.c:static slurm_cli_opt_t slurm_opt_ntasks_per_gpu = {
src/common/slurm_opt.c:	.name = "ntasks-per-gpu",
src/common/slurm_opt.c:	.val = LONG_OPT_NTASKSPERGPU,
src/common/slurm_opt.c:	.set_func = arg_set_ntasks_per_gpu,
src/common/slurm_opt.c:	.get_func = arg_get_ntasks_per_gpu,
src/common/slurm_opt.c:	.reset_func = arg_reset_ntasks_per_gpu,
src/common/slurm_opt.c:	&slurm_opt_cpus_per_gpu,
src/common/slurm_opt.c:	&slurm_opt_gpu_bind,
src/common/slurm_opt.c:	&slurm_opt_gpu_freq,
src/common/slurm_opt.c:	&slurm_opt_gpus,
src/common/slurm_opt.c:	&slurm_opt_gpus_per_node,
src/common/slurm_opt.c:	&slurm_opt_gpus_per_socket,
src/common/slurm_opt.c:	&slurm_opt_gpus_per_task,
src/common/slurm_opt.c:	&slurm_opt_mem_per_gpu,
src/common/slurm_opt.c:	&slurm_opt_ntasks_per_gpu,
src/common/slurm_opt.c: * Validate that the three memory options (--mem, --mem-per-cpu, --mem-per-gpu)
src/common/slurm_opt.c:	     slurm_option_set_by_cli(opt, LONG_OPT_MEM_PER_GPU)) > 1) {
src/common/slurm_opt.c:		fatal("--mem, --mem-per-cpu, and --mem-per-gpu are mutually exclusive.");
src/common/slurm_opt.c:		slurm_option_reset(opt, "mem-per-gpu");
src/common/slurm_opt.c:		slurm_option_reset(opt, "mem-per-gpu");
src/common/slurm_opt.c:	} else if (slurm_option_set_by_cli(opt, LONG_OPT_MEM_PER_GPU)) {
src/common/slurm_opt.c:		    slurm_option_set_by_env(opt, LONG_OPT_MEM_PER_GPU)) > 1) {
src/common/slurm_opt.c:		fatal("SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive.");
src/common/slurm_opt.c:		else if (slurm_option_isset(opt, "mem-per-gpu"))
src/common/slurm_opt.c:			info("Configured SelectTypeParameters doesn't treat memory as a consumable resource. In this case value of --mem-per-gpu is ignored.");
src/common/slurm_opt.c:static void _validate_ntasks_per_gpu(slurm_opt_t *opt)
src/common/slurm_opt.c:	bool gpu = slurm_option_set_by_cli(opt, LONG_OPT_NTASKSPERGPU);
src/common/slurm_opt.c:	bool gpu_env = slurm_option_set_by_env(opt, LONG_OPT_NTASKSPERGPU);
src/common/slurm_opt.c:	bool any = (tres || gpu || tres_env || gpu_env);
src/common/slurm_opt.c:	/* Validate --ntasks-per-gpu and --ntasks-per-tres */
src/common/slurm_opt.c:	if (gpu && tres) {
src/common/slurm_opt.c:		if (opt->ntasks_per_gpu != opt->ntasks_per_tres)
src/common/slurm_opt.c:			fatal("Inconsistent values set to --ntasks-per-gpu=%d and --ntasks-per-tres=%d ",
src/common/slurm_opt.c:			      opt->ntasks_per_gpu,
src/common/slurm_opt.c:	} else if (gpu && tres_env) {
src/common/slurm_opt.c:			info("Ignoring SLURM_NTASKS_PER_TRES since --ntasks-per-gpu given as command line option");
src/common/slurm_opt.c:	} else if (tres && gpu_env) {
src/common/slurm_opt.c:			info("Ignoring SLURM_NTASKS_PER_GPU since --ntasks-per-tres given as command line option");
src/common/slurm_opt.c:		slurm_option_reset(opt, "ntasks-per-gpu");
src/common/slurm_opt.c:	} else if (gpu_env && tres_env) {
src/common/slurm_opt.c:		if (opt->ntasks_per_gpu != opt->ntasks_per_tres)
src/common/slurm_opt.c:			fatal("Inconsistent values set by environment variables SLURM_NTASKS_PER_GPU=%d and SLURM_NTASKS_PER_TRES=%d ",
src/common/slurm_opt.c:			      opt->ntasks_per_gpu,
src/common/slurm_opt.c:		fatal("--tres-per-task is mutually exclusive with --ntasks-per-gpu and SLURM_NTASKS_PER_GPU");
src/common/slurm_opt.c:		fatal("SLURM_TRES_PER_TASK is mutually exclusive with --ntasks-per-gpu and SLURM_NTASKS_PER_GPU");
src/common/slurm_opt.c:	if (slurm_option_set_by_cli(opt, LONG_OPT_GPUS_PER_TASK))
src/common/slurm_opt.c:		fatal("--gpus-per-task is mutually exclusive with --ntasks-per-gpu and SLURM_NTASKS_PER_GPU");
src/common/slurm_opt.c:	if (slurm_option_set_by_env(opt, LONG_OPT_GPUS_PER_TASK))
src/common/slurm_opt.c:		fatal("SLURM_GPUS_PER_TASK is mutually exclusive with --ntasks-per-gpu and SLURM_NTASKS_PER_GPU");
src/common/slurm_opt.c:	if (slurm_option_set_by_cli(opt, LONG_OPT_GPUS_PER_SOCKET))
src/common/slurm_opt.c:		fatal("--gpus-per-socket is mutually exclusive with --ntasks-per-gpu and SLURM_NTASKS_PER_GPU");
src/common/slurm_opt.c:	if (slurm_option_set_by_env(opt, LONG_OPT_GPUS_PER_SOCKET))
src/common/slurm_opt.c:		fatal("SLURM_GPUS_PER_SOCKET is mutually exclusive with --ntasks-per-gpu and SLURM_NTASKS_PER_GPU");
src/common/slurm_opt.c:		fatal("--ntasks-per-node is mutually exclusive with --ntasks-per-gpu and SLURM_NTASKS_PER_GPU");
src/common/slurm_opt.c:		fatal("SLURM_NTASKS_PER_NODE is mutually exclusive with --ntasks-per-gpu and SLURM_NTASKS_PER_GPU");
src/common/slurm_opt.c: * tres_per_task takes a form similar to "cpu=10,gres/gpu:gtx=1,license/iop1=1".
src/common/slurm_opt.c:static bool _get_gpu_cnt_and_str(slurm_opt_t *opt, int *gpu_cnt, char **gpu_str)
src/common/slurm_opt.c:	if (!opt->gpus_per_task)
src/common/slurm_opt.c:	xstrcat(*gpu_str, "gres/gpu");
src/common/slurm_opt.c:	if ((num_str = xstrstr(opt->gpus_per_task, ":")))
src/common/slurm_opt.c:	else if ((num_str = xstrstr(opt->gpus_per_task, "=")))
src/common/slurm_opt.c:		/* Add type string to gpu_str */
src/common/slurm_opt.c:		xstrfmtcat(*gpu_str, ":%s", opt->gpus_per_task);
src/common/slurm_opt.c:		num_str = opt->gpus_per_task;
src/common/slurm_opt.c:	if (gpu_cnt)
src/common/slurm_opt.c:		(*gpu_cnt) = strtol(num_str, NULL, 10);
src/common/slurm_opt.c:	if (optval == LONG_OPT_GPUS_PER_TASK) {
src/common/slurm_opt.c:		set = _get_gpu_cnt_and_str(opt, &cnt, &str);
src/common/slurm_opt.c:		env_variable = "SLURM_GPUS_PER_TASK";
src/common/slurm_opt.c:		/* This function only supports [gpus|cpus]_per_task */
src/common/slurm_opt.c:	if (optval == LONG_OPT_GPUS_PER_TASK) {
src/common/slurm_opt.c:		opt->gpus_per_task = opt_in_tpt_ptr;
src/common/slurm_opt.c:	_set_tres_per_task_from_sibling_opt(opt, LONG_OPT_GPUS_PER_TASK);
src/common/slurm_opt.c:	     slurm_option_set_by_cli(opt, LONG_OPT_CPUS_PER_GPU)) ||
src/common/slurm_opt.c:	     slurm_option_set_by_env(opt, LONG_OPT_CPUS_PER_GPU))) {
src/common/slurm_opt.c:		fatal("--cpus-per-task, --tres-per-task=cpu:#, and --cpus-per-gpu are mutually exclusive");
src/common/slurm_opt.c:	    slurm_option_set_by_env(opt, LONG_OPT_CPUS_PER_GPU)) {
src/common/slurm_opt.c:				env_str = "SALLOC_CPUS_PER_GPU";
src/common/slurm_opt.c:				env_str = "SBATCH_CPUS_PER_GPU";
src/common/slurm_opt.c:				env_str = "SLURM_CPUS_PER_GPU";
src/common/slurm_opt.c:		slurm_option_reset(opt, "cpus-per-gpu");
src/common/slurm_opt.c:	} else if (slurm_option_set_by_cli(opt, LONG_OPT_CPUS_PER_GPU) &&
src/common/slurm_opt.c:			info("Ignoring cpus_per_task from the environment since --cpus-per-gpu was given as a command line option");
src/common/slurm_opt.c:	_validate_ntasks_per_gpu(opt);
src/common/slurm_opt.c:	if (opt_local->cpus_per_gpu)
src/common/slurm_opt.c:		xstrfmtcat(job_desc->cpus_per_tres, "gres/gpu:%d",
src/common/slurm_opt.c:			   opt_local->cpus_per_gpu);
src/common/slurm_opt.c:	if (opt_local->mem_per_gpu != NO_VAL64)
src/common/slurm_opt.c:		xstrfmtcat(job_desc->mem_per_tres, "gres/gpu:%"PRIu64,
src/common/slurm_opt.c:			   opt_local->mem_per_gpu);
src/common/slurm_opt.c:	xfmt_tres(&job_desc->tres_per_job, "gres/gpu", opt_local->gpus);
src/common/slurm_opt.c:	xfmt_tres(&job_desc->tres_per_node, "gres/gpu",
src/common/slurm_opt.c:		  opt_local->gpus_per_node);
src/common/slurm_opt.c:	xfmt_tres(&job_desc->tres_per_socket, "gres/gpu",
src/common/slurm_opt.c:		  opt_local->gpus_per_socket);
src/common/slurm_opt.c:	else if (opt_local->ntasks_per_gpu != NO_VAL)
src/common/slurm_opt.c:		job_desc->ntasks_per_tres = opt_local->ntasks_per_gpu;
src/common/proc_args.h: * prefix IN - TRES type (e.g. "gpu")
src/common/proc_args.h: * prefix IN - TRES type (e.g. "gpu")
src/common/read_config.c:	{"DefCPUPerGPU" , S_P_UINT64},
src/common/read_config.c:	{"DefMemPerGPU" , S_P_UINT64},
src/common/read_config.c:	{"GpuFreqDef", S_P_STRING},
src/common/read_config.c:		{"RestrictedCoresPerGPU", S_P_UINT16},
src/common/read_config.c:		if (!s_p_get_uint16(&n->res_cores_per_gpu,
src/common/read_config.c:				    "RestrictedCoresPerGPU", tbl))
src/common/read_config.c:			s_p_get_uint16(&n->res_cores_per_gpu,
src/common/read_config.c:				       "RestrictedCoresPerGPU", dflt);
src/common/read_config.c:	case JOB_DEF_CPU_PER_GPU:
src/common/read_config.c:		return "DefCpuPerGPU";
src/common/read_config.c:	case JOB_DEF_MEM_PER_GPU:
src/common/read_config.c:		return "DefMemPerGPU";
src/common/read_config.c:	if (!xstrcasecmp(type, "DefCpuPerGPU"))
src/common/read_config.c:		return JOB_DEF_CPU_PER_GPU;
src/common/read_config.c:	if (!xstrcasecmp(type, "DefMemPerGPU"))
src/common/read_config.c:		return JOB_DEF_MEM_PER_GPU;
src/common/read_config.c:	uint64_t def_cpu_per_gpu = 0, def_mem_per_gpu = 0;
src/common/read_config.c:		{"DefCPUPerGPU" , S_P_UINT64},
src/common/read_config.c:		{"DefMemPerGPU" , S_P_UINT64},
src/common/read_config.c:		if (s_p_get_uint64(&def_cpu_per_gpu, "DefCPUPerGPU", tbl) ||
src/common/read_config.c:		    s_p_get_uint64(&def_cpu_per_gpu, "DefCPUPerGPU", dflt)) {
src/common/read_config.c:			job_defaults->type  = JOB_DEF_CPU_PER_GPU;
src/common/read_config.c:			job_defaults->value = def_cpu_per_gpu;
src/common/read_config.c:		if (s_p_get_uint64(&def_mem_per_gpu, "DefMemPerGPU", tbl) ||
src/common/read_config.c:		    s_p_get_uint64(&def_mem_per_gpu, "DefMemPerGPU", dflt)) {
src/common/read_config.c:			job_defaults->type  = JOB_DEF_MEM_PER_GPU;
src/common/read_config.c:			job_defaults->value = def_mem_per_gpu;
src/common/read_config.c:	xfree (ctl_conf_ptr->gpu_freq_def);
src/common/read_config.c:	uint64_t def_cpu_per_gpu = 0, def_mem_per_gpu = 0, tot_prio_weight;
src/common/read_config.c:				  "rsmi", "gpu");
src/common/read_config.c:	if (s_p_get_uint64(&def_cpu_per_gpu, "DefCPUPerGPU", hashtbl)) {
src/common/read_config.c:		job_defaults->type  = JOB_DEF_CPU_PER_GPU;
src/common/read_config.c:		job_defaults->value = def_cpu_per_gpu;
src/common/read_config.c:	if (s_p_get_uint64(&def_mem_per_gpu, "DefMemPerGPU", hashtbl)) {
src/common/read_config.c:		job_defaults->type  = JOB_DEF_MEM_PER_GPU;
src/common/read_config.c:		job_defaults->value = def_mem_per_gpu;
src/common/read_config.c:	(void) s_p_get_string(&conf->gpu_freq_def, "GpuFreqDef", hashtbl);
src/common/read_config.c:		 * If we are tracking gres/gpu, also add the usage tres to the
src/common/read_config.c:				    "gres/gpu")) {
src/common/read_config.c:					      "gres/gpumem,gres/gpuutil");
src/common/assoc_mgr.h: * So tres_rec->name of "gpu" can match accounting TRES name of "gpu:tesla".
src/common/assoc_mgr.h: * For example: "cpu:2,gres/gpu:kepler:2,gres/craynetwork:1"
src/common/slurm_protocol_api.c: * returns the configured GpuFreqDef value
src/common/slurm_protocol_api.c: * RET char *    - GpuFreqDef value,  MUST be xfreed by caller
src/common/slurm_protocol_api.c:char *slurm_get_gpu_freq_def(void)
src/common/slurm_protocol_api.c:	char *gpu_freq_def = NULL;
src/common/slurm_protocol_api.c:		gpu_freq_def = xstrdup(conf->gpu_freq_def);
src/common/slurm_protocol_api.c:	return gpu_freq_def;
src/common/env.h:	int ntasks_per_gpu;	/* --ntasks-per-gpu		*/
src/common/slurm_protocol_defs.c:		xfree(node->gpu_spec);
src/common/slurm_protocol_defs.c:			/* Bad format (e.g. "gpu:") */
src/common/slurm_protocol_defs.c:	 * and do not return it. For example in the case of "gres/gpu:tesla:0",
src/common/slurm_protocol_defs.c:	 * we would have: tres_type=gres, name = gpu, type = tesla, value = 0
src/common/list.h: *  Copyright (C) 2021 NVIDIA Corporation.
src/common/node_conf.c:	config_ptr->res_cores_per_gpu = conf_node->res_cores_per_gpu;
src/common/node_conf.c:	node_ptr->res_cores_per_gpu = config_ptr->res_cores_per_gpu;
src/common/node_conf.c:	xfree(node_ptr->gpu_spec);
src/common/node_conf.c:	FREE_NULL_BITMAP(node_ptr->gpu_spec_bitmap);
src/common/node_conf.c:		pack16(object->res_cores_per_gpu, buffer);
src/common/node_conf.c:		pack_bit_str_hex(object->gpu_spec_bitmap, buffer);
src/common/node_conf.c:		pack16(object->res_cores_per_gpu, buffer);
src/common/node_conf.c:		pack_bit_str_hex(object->gpu_spec_bitmap, buffer);
src/common/node_conf.c:		safe_unpack16(&object->res_cores_per_gpu, buffer);
src/common/node_conf.c:		unpack_bit_str_hex(&object->gpu_spec_bitmap, buffer);
src/common/node_conf.c:		safe_unpack16(&object->res_cores_per_gpu, buffer);
src/common/node_conf.c:		unpack_bit_str_hex(&object->gpu_spec_bitmap, buffer);
src/common/node_conf.c:	config_ptr->res_cores_per_gpu = node_ptr->res_cores_per_gpu;
src/common/proc_args.c: * prefix IN - TRES type (e.g. "gres/gpu")
src/common/proc_args.c: * prefix IN - TRES type (e.g. "gres/gpu")
src/common/slurm_protocol_pack.c:		safe_unpackstr(&node->gpu_spec, buffer);
src/common/slurm_protocol_pack.c:		safe_unpack16(&node->res_cores_per_gpu, buffer);
src/common/slurm_protocol_pack.c:		packstr(build_ptr->gpu_freq_def, buffer);
src/common/slurm_protocol_pack.c:		packstr(build_ptr->gpu_freq_def, buffer);
src/common/slurm_protocol_pack.c:		packstr(build_ptr->gpu_freq_def, buffer);
src/common/slurm_protocol_pack.c:		packstr(build_ptr->gpu_freq_def, buffer);
src/common/slurm_protocol_pack.c:		safe_unpackstr(&build_ptr->gpu_freq_def, buffer);
src/common/slurm_protocol_pack.c:		safe_unpackstr(&build_ptr->gpu_freq_def, buffer);
src/common/slurm_protocol_pack.c:		safe_unpackstr(&build_ptr->gpu_freq_def, buffer);
src/common/slurm_protocol_pack.c:		safe_unpackstr(&build_ptr->gpu_freq_def, buffer);
src/sview/node_info.c:	SORTID_RES_CORES_PER_GPU,
src/sview/node_info.c:	{G_TYPE_INT, SORTID_RES_CORES_PER_GPU,
src/sview/node_info.c:	 "RestrictedCoresPerGPU", false,
src/sview/node_info.c:	convert_num_unit((float)node_ptr->res_cores_per_gpu,
src/sview/node_info.c:						 SORTID_RES_CORES_PER_GPU),
src/sview/node_info.c:			   SORTID_RES_CORES_PER_GPU,
src/sview/node_info.c:			   node_ptr->res_cores_per_gpu,
src/srun/srun_job.c:	uint16_t ntasks_per_tres; /* number of tasks that can access each gpu */
src/srun/srun_job.c:static void _check_gpus_per_socket(slurm_opt_t *opt_local)
src/srun/srun_job.c:	if (!opt_local->gpus_per_socket || checked)
src/srun/srun_job.c:	if (opt_local->gpus_per_socket &&
src/srun/srun_job.c:	    !slurm_option_set_by_env(opt_local, LONG_OPT_GPUS_PER_SOCKET)) {
src/srun/srun_job.c:		 * gpus_per_socket does not work for steps.
src/srun/srun_job.c:		warning("Ignoring --gpus-per-socket because it can only be specified at job allocation time, not during step allocation.");
src/srun/srun_job.c:				_check_gpus_per_socket(opt_local);
src/srun/srun_job.h:	uint16_t ntasks_per_tres; /* number of tasks that can access each gpu */
src/srun/launch.c:static int _parse_gpu_request(char *in_str)
src/srun/launch.c:	int gpus_val = 0;
src/srun/launch.c:			gpus_val += tmp;
src/srun/launch.c:	return gpus_val;
src/srun/launch.c:	if (opt_local->cpus_per_gpu) {
src/srun/launch.c:		xstrfmtcat(step_req->cpus_per_tres, "gres/gpu:%d",
src/srun/launch.c:			   opt_local->cpus_per_gpu);
src/srun/launch.c:			info("Ignoring --whole since --cpus-per-gpu used");
src/srun/launch.c:			verbose("Implicitly setting --exact, because --cpus-per-gpu given.");
src/srun/launch.c:	if (opt_local->mem_per_gpu != NO_VAL64)
src/srun/launch.c:		xstrfmtcat(step_req->mem_per_tres, "gres/gpu:%"PRIu64,
src/srun/launch.c:			   opt.mem_per_gpu);
src/srun/launch.c:	} else if (opt_local->cpus_per_gpu) {
src/srun/launch.c:		if (opt_local->gpus) {
src/srun/launch.c:			int gpus_per_step;
src/srun/launch.c:			gpus_per_step = _parse_gpu_request(opt_local->gpus);
src/srun/launch.c:			step_req->cpu_count = gpus_per_step *
src/srun/launch.c:				opt_local->cpus_per_gpu;
src/srun/launch.c:		} else if (opt_local->gpus_per_node) {
src/srun/launch.c:			int gpus_per_node;
src/srun/launch.c:			gpus_per_node =
src/srun/launch.c:				_parse_gpu_request(opt_local->gpus_per_node);
src/srun/launch.c:				gpus_per_node * opt_local->cpus_per_gpu;
src/srun/launch.c:					  "gres/gpu:"))) {
src/srun/launch.c:			pos += 9; /* Don't include "gres/gpu:" */
src/srun/launch.c:					      _parse_gpu_request(pos) *
src/srun/launch.c:					      opt_local->cpus_per_gpu;
src/srun/launch.c:			uint64_t gpus_per_node = 0;
src/srun/launch.c:				       opt_local->gres, "gpu",
src/srun/launch.c:				       &gpus_per_node, &save_ptr,
src/srun/launch.c:			 * Same math as gpus_per_node
src/srun/launch.c:			 * If gpus_per_node == 0, then the step did not request
src/srun/launch.c:			 * gpus, but the step may still inherit gpus from the
src/srun/launch.c:			 * case gpus_per_node == 0.
src/srun/launch.c:				gpus_per_node * opt_local->cpus_per_gpu;
src/srun/launch.c:		} else if (opt_local->gpus_per_socket) {
src/srun/launch.c:			 * gpus_per_socket is not fully supported for steps and
src/srun/launch.c:			 * does not affect the gpus allocated to the step:
src/srun/launch.c:		   (opt_local->ntasks_per_gpu != NO_VAL)) {
src/srun/launch.c:	else if (opt_local->ntasks_per_gpu != NO_VAL)
src/srun/launch.c:		step_req->ntasks_per_tres = opt_local->ntasks_per_gpu;
src/srun/launch.c:	     (opt_local->ntasks_per_gpu != NO_VAL))) {
src/srun/launch.c:		/* Implicit single GPU binding with ntasks-per-tres/gpu */
src/srun/launch.c:			xstrfmtcat(opt_local->tres_bind, "gres/gpu:single:%d",
src/srun/launch.c:			xstrfmtcat(opt_local->tres_bind, "gres/gpu:single:%d",
src/srun/launch.c:				   opt_local->ntasks_per_gpu);
src/srun/launch.c:	xfmt_tres(&step_req->tres_per_step, "gres/gpu", opt_local->gpus);
src/srun/launch.c:	 * Do not also send --gpus-per-node which could have been set from the
src/srun/launch.c:		xfmt_tres(&step_req->tres_per_node, "gres/gpu",
src/srun/launch.c:			  opt_local->gpus_per_node);
src/srun/launch.c:	xfmt_tres(&step_req->tres_per_socket, "gres/gpu",
src/srun/launch.c:		  opt_local->gpus_per_socket);
src/srun/opt.c:	opt.gpu_bind = NULL;		/* Moved by memcpy */
src/srun/opt.c:	opt.gpu_freq = NULL;		/* Moved by memcpy */
src/srun/opt.c:	opt.gpus = NULL;		/* Moved by memcpy */
src/srun/opt.c:	opt.gpus_per_node = NULL;	/* Moved by memcpy */
src/srun/opt.c:	opt.gpus_per_socket = NULL;	/* Moved by memcpy */
src/srun/opt.c:	opt.gpus_per_task = NULL;	/* Moved by memcpy */
src/srun/opt.c:  { "SLURM_CPUS_PER_GPU", LONG_OPT_CPUS_PER_GPU },
src/srun/opt.c:  { "SLURM_GPUS", 'G' },
src/srun/opt.c:  { "SLURM_GPU_BIND", LONG_OPT_GPU_BIND },
src/srun/opt.c:  { "SLURM_GPU_FREQ", LONG_OPT_GPU_FREQ },
src/srun/opt.c:  { "SLURM_GPUS_PER_NODE", LONG_OPT_GPUS_PER_NODE },
src/srun/opt.c:  { "SLURM_GPUS_PER_SOCKET", LONG_OPT_GPUS_PER_SOCKET },
src/srun/opt.c:  { "SLURM_GPUS_PER_TASK", LONG_OPT_GPUS_PER_TASK },
src/srun/opt.c:  { "SLURM_MEM_PER_GPU", LONG_OPT_MEM_PER_GPU },
src/srun/opt.c:  { "SLURM_NTASKS_PER_GPU", LONG_OPT_NTASKSPERGPU },
src/srun/opt.c:	 * Specifying --gpus should override SLURM_GPUS_PER_NODE env if present
src/srun/opt.c:	if (slurm_option_set_by_env(&opt, LONG_OPT_GPUS_PER_NODE) &&
src/srun/opt.c:		slurm_option_reset(&opt, "gpus-per-node");
src/srun/opt.c:"            [--cpus-per-gpu=n] [--gpus=n] [--gpu-bind=...] [--gpu-freq=...]\n"
src/srun/opt.c:"            [--gpus-per-node=n] [--gpus-per-socket=n] [--gpus-per-task=n]\n"
src/srun/opt.c:"            [--mem-per-gpu=MB] [--tres-bind=...] [--tres-per-task=list]\n"
src/srun/opt.c:"GPU scheduling options:\n"
src/srun/opt.c:"      --cpus-per-gpu=n        number of CPUs required per allocated GPU\n"
src/srun/opt.c:"  -G, --gpus=n                count of GPUs required for the job\n"
src/srun/opt.c:"      --gpu-bind=...          task to gpu binding options\n"
src/srun/opt.c:"      --gpu-freq=...          frequency and voltage of GPUs\n"
src/srun/opt.c:"      --gpus-per-node=n       number of GPUs required per allocated node\n"
src/srun/opt.c:"      --gpus-per-socket=n     number of GPUs required per allocated socket\n"
src/srun/opt.c:"      --gpus-per-task=n       number of GPUs required per spawned task\n"
src/srun/opt.c:"      --mem-per-gpu=n         real memory required per allocated GPU\n"
src/srun/srun.c:	else if (opt_local->ntasks_per_gpu != NO_VAL)
src/srun/srun.c:		env->ntasks_per_tres = opt_local->ntasks_per_gpu;
src/scrontab/opt.c:	if (opt.cpus_per_gpu)
src/scrontab/opt.c:		xstrfmtcat(desc->cpus_per_tres, "gres/gpu:%d", opt.cpus_per_gpu);
src/scrontab/opt.c:	xfmt_tres(&desc->tres_per_job, "gres/gpu", opt.gpus);
src/scrontab/opt.c:	xfmt_tres(&desc->tres_per_node, "gres/gpu", opt.gpus_per_node);
src/scrontab/opt.c:	xfmt_tres(&desc->tres_per_socket, "gres/gpu", opt.gpus_per_socket);
src/scrontab/opt.c:	if (opt.mem_per_gpu != NO_VAL64)
src/scrontab/opt.c:		xstrfmtcat(desc->mem_per_tres, "gres/gpu:%"PRIu64, opt.mem_per_gpu);
src/scrun/alloc.c:	{ "SCRUN_CPUS_PER_GPU", LONG_OPT_CPUS_PER_GPU },
src/scrun/alloc.c:	{ "SCRUN_GPU_BIND", LONG_OPT_GPU_BIND },
src/scrun/alloc.c:	{ "SCRUN_GPU_FREQ", LONG_OPT_GPU_FREQ },
src/scrun/alloc.c:	{ "SCRUN_GPUS", 'G' },
src/scrun/alloc.c:	{ "SCRUN_GPUS_PER_NODE", LONG_OPT_GPUS_PER_NODE },
src/scrun/alloc.c:	{ "SCRUN_GPUS_PER_SOCKET", LONG_OPT_GPUS_PER_SOCKET },
src/scrun/alloc.c:	{ "SCRUN_GPUS_PER_TASK", LONG_OPT_GPUS_PER_TASK },
src/scrun/alloc.c:	{ "SCRUN_MEM_PER_GPU", LONG_OPT_MEM_PER_GPU },
src/scrun/alloc.c:	{ "SCRUN_NTASKS_PER_GPU", LONG_OPT_NTASKSPERGPU },
src/stepmgr/gres_stepmgr.c:			 * "--gres=gpu:1,tmpfs:foo:2,tmpfs:bar:7" where typeless
src/stepmgr/gres_stepmgr.c:			 * is found for GRES name "gpu" but then for "tmpfs"
src/stepmgr/gres_stepmgr.c:		 * allocated 1 GPU of type "tesla" and 1 GPU of type "volta",
src/stepmgr/gres_stepmgr.c:		 * 2 GPUs.
src/stepmgr/gres_stepmgr.c:		 * GRES like "gpu:tesla", where you might want to track both as
src/stepmgr/gres_stepmgr.c:		 * --mem-per-gpu. Adding another option will require a change
src/stepmgr/gres_stepmgr.c:		 * for GRES like "gpu:tesla", where you might

```
