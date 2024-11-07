# https://github.com/DataBiosphere/toil

```console
setup.cfg:    docker_cuda
setup.cfg:    local_cuda
docs/running/cliOptions.rst:                        specification can have a type (gpu [default], nvidia,
docs/running/cliOptions.rst:                        amd, cuda, rocm, opencl, or a specific model like
docs/running/cliOptions.rst:                        nvidia-tesla-k80), and a count [default: 1]. If both a
docs/running/cloud/amazon.rst:Now that our cluster is launched, we use the :ref:`rsyncCluster` utility to copy
docs/running/cloud/clusterUtils.rst:.. _rsyncCluster:
docs/gettingStarted/quickStart.rst:#. Copy ``helloWorld.py`` to the ``/tmp`` directory on the leader node using the :ref:`rsyncCluster` command::
docs/gettingStarted/quickStart.rst:   the :ref:`rsyncCluster` command::
src/toil/job.py:    """Requirement for one or more computational accelerators, like a GPU or FPGA."""
src/toil/job.py:    What kind of accelerator is required. Can be "gpu". Other kinds defined in
src/toil/job.py:    "nvidia-tesla-k80" might be expected to work. If a specific model of
src/toil/job.py:    its accleerators; strings like "nvidia" or "amd" might be expected to work.
src/toil/job.py:    "cuda". Other APIs supported in the future might be "rocm", "opencl",
src/toil/job.py:    # TODO: support requesting any GPU with X amount of vram
src/toil/job.py:    {'count': 8, 'kind': 'gpu'}
src/toil/job.py:    {'count': 1, 'kind': 'gpu'}
src/toil/job.py:    >>> parse_accelerator("nvidia-tesla-k80")
src/toil/job.py:    {'count': 1, 'kind': 'gpu', 'brand': 'nvidia', 'model': 'nvidia-tesla-k80'}
src/toil/job.py:    >>> parse_accelerator("nvidia-tesla-k80:2")
src/toil/job.py:    {'count': 2, 'kind': 'gpu', 'brand': 'nvidia', 'model': 'nvidia-tesla-k80'}
src/toil/job.py:    >>> parse_accelerator("gpu")
src/toil/job.py:    {'count': 1, 'kind': 'gpu'}
src/toil/job.py:    >>> parse_accelerator("cuda:1")
src/toil/job.py:    {'count': 1, 'kind': 'gpu', 'brand': 'nvidia', 'api': 'cuda'}
src/toil/job.py:    >>> parse_accelerator({"kind": "gpu"})
src/toil/job.py:    {'count': 1, 'kind': 'gpu'}
src/toil/job.py:    >>> parse_accelerator({"brand": "nvidia", "count": 5})
src/toil/job.py:    {'count': 5, 'kind': 'gpu', 'brand': 'nvidia'}
src/toil/job.py:    Assumes that if not specified, we are talking about GPUs, and about one
src/toil/job.py:    of them. Knows that "gpu" is a kind, and "cuda" is an API, and "nvidia"
src/toil/job.py:    KINDS = {"gpu"}
src/toil/job.py:    BRANDS = {"nvidia", "amd"}
src/toil/job.py:    APIS = {"cuda", "rocm", "opencl"}
src/toil/job.py:    parsed: AcceleratorRequirement = {"count": 1, "kind": "gpu"}
src/toil/job.py:    if parsed["kind"] == "gpu":
src/toil/job.py:        # Use some smarts about what current GPUs are like to elaborate the
src/toil/job.py:            if parsed["api"] == "cuda":
src/toil/job.py:                # Only nvidia makes cuda cards
src/toil/job.py:                parsed["brand"] = "nvidia"
src/toil/job.py:            elif parsed["api"] == "rocm":
src/toil/job.py:                # Only amd makes rocm cards
src/toil/job.py:                ]  # accelerators={'kind': 'gpu', 'brand': 'nvidia', 'count': 2}
src/toil/job.py:        """Any accelerators, such as GPUs, that are needed."""
src/toil/job.py:        """Any accelerators, such as GPUs, that are needed."""
src/toil/cwl/cwltoil.py:        if req.get("cudaDeviceCount", 0) > 0:
src/toil/cwl/cwltoil.py:            # There's a CUDARequirement, which cwltool processed for us
src/toil/cwl/cwltoil.py:                    "kind": "gpu",
src/toil/cwl/cwltoil.py:                    "api": "cuda",
src/toil/cwl/cwltoil.py:                    "count": cast(int, req["cudaDeviceCount"]),
src/toil/cwl/cwltoil.py:        # expressions in the value for us like it does for CUDARequirement
src/toil/cwl/cwltoil.py:            "http://commonwl.org/cwltool#CUDARequirement",
src/toil/test/cwl/nvidia_smi.cwl:# Example CUDARequirement test tool from https://github.com/common-workflow-language/cwltool/pull/1581#issue-1087165113
src/toil/test/cwl/nvidia_smi.cwl:baseCommand: nvidia-smi
src/toil/test/cwl/nvidia_smi.cwl:  cwltool:CUDARequirement:
src/toil/test/cwl/nvidia_smi.cwl:    cudaVersionMin: "11.4"
src/toil/test/cwl/nvidia_smi.cwl:    cudaComputeCapabilityMin: "3.0"
src/toil/test/cwl/nvidia_smi.cwl:    dockerPull: nvidia/cuda:11.4.0-base-ubuntu20.04
src/toil/test/cwl/cwlTest.py:    needs_docker_cuda,
src/toil/test/cwl/cwlTest.py:    needs_local_cuda,
src/toil/test/cwl/cwlTest.py:    @needs_docker_cuda
src/toil/test/cwl/cwlTest.py:    @needs_local_cuda
src/toil/test/cwl/cwlTest.py:    def test_cuda(self) -> None:
src/toil/test/cwl/cwlTest.py:            "src/toil/test/cwl/nvidia_smi.cwl",
src/toil/test/wdl/wdltoil_test.py:    needs_docker_cuda,
src/toil/test/wdl/wdltoil_test.py:    58,  # test_gpu, needs gpu to run, else warning
src/toil/test/wdl/wdltoil_test.py:    @needs_docker_cuda
src/toil/test/wdl/wdltoil_test.py:        """Test if Giraffe and GPU DeepVariant run. This could take 25 minutes."""
src/toil/test/wdl/wdltoil_test.py:        # TODO: enable test if nvidia-container-runtime and Singularity are installed but Docker isn't.
src/toil/test/wdl/wdltoil_test.py:            # Write some inputs. We need to override the example inputs to use a GPU container, but that means we need absolute input URLs.
src/toil/test/wdl/wdltoil_test.py:                    "GiraffeDeepVariant.runDeepVariantCallVariants.in_dv_gpu_container": "google/deepvariant:1.3.0-gpu",
src/toil/test/wdl/wdltoil_test.py:        # TODO: enable test if nvidia-container-runtime and Singularity are installed but Docker isn't.
src/toil/test/__init__.py:    have_working_nvidia_docker_runtime,
src/toil/test/__init__.py:    have_working_nvidia_smi,
src/toil/test/__init__.py:def needs_local_cuda(test_item: MT) -> MT:
src/toil/test/__init__.py:    a CUDA setup legible to cwltool (i.e. providing userspace nvidia-smi) is present.
src/toil/test/__init__.py:    test_item = _mark_test("local_cuda", test_item)
src/toil/test/__init__.py:    if have_working_nvidia_smi():
src/toil/test/__init__.py:            "Install nvidia-smi, an nvidia proprietary driver, and a CUDA-capable nvidia GPU to include this test."
src/toil/test/__init__.py:def needs_docker_cuda(test_item: MT) -> MT:
src/toil/test/__init__.py:    a CUDA setup is available through Docker.
src/toil/test/__init__.py:    test_item = _mark_test("docker_cuda", needs_online(test_item))
src/toil/test/__init__.py:    if have_working_nvidia_docker_runtime():
src/toil/test/__init__.py:            "Install nvidia-container-runtime on your Docker server and configure an 'nvidia' runtime to include this test."
src/toil/batchSystems/lsf.py:            gpus: Optional[int] = None,
src/toil/batchSystems/gridengine.py:            gpus: Optional[int] = None,
src/toil/batchSystems/slurm.py:        default_gpu_partition: SlurmBatchSystem.PartitionInfo | None
src/toil/batchSystems/slurm.py:        gpu_partitions: set[str]
src/toil/batchSystems/slurm.py:            self._get_gpu_partitions()
src/toil/batchSystems/slurm.py:        def _get_gpu_partitions(self) -> None:
src/toil/batchSystems/slurm.py:            Get all available GPU partitions. Also get the default GPU partition.
src/toil/batchSystems/slurm.py:            gpu_partitions = [
src/toil/batchSystems/slurm.py:            self.gpu_partitions = {p.partition_name for p in gpu_partitions}
src/toil/batchSystems/slurm.py:            # Grab the lowest priority GPU partition
src/toil/batchSystems/slurm.py:            # If no GPU partitions are available, then set the default to None
src/toil/batchSystems/slurm.py:            self.default_gpu_partition = None
src/toil/batchSystems/slurm.py:            if len(gpu_partitions) > 0:
src/toil/batchSystems/slurm.py:                self.default_gpu_partition = sorted(
src/toil/batchSystems/slurm.py:                    gpu_partitions, key=lambda x: x.priority
src/toil/batchSystems/slurm.py:            gpus: int | None = None,
src/toil/batchSystems/slurm.py:                cpu, memory, jobID, jobName, job_environment, gpus
src/toil/batchSystems/slurm.py:            gpus: int | None,
src/toil/batchSystems/slurm.py:            if gpus:
src/toil/batchSystems/slurm.py:                # This block will add a gpu supported partition only if no partition is supplied by the user
src/toil/batchSystems/slurm.py:                sbatch_line = sbatch_line[:1] + [f"--gres=gpu:{gpus}"] + sbatch_line[1:]
src/toil/batchSystems/slurm.py:                    # try to get the name of the lowest priority gpu supported partition
src/toil/batchSystems/slurm.py:                    lowest_gpu_partition = self.boss.partitions.default_gpu_partition
src/toil/batchSystems/slurm.py:                    if lowest_gpu_partition is None:
src/toil/batchSystems/slurm.py:                        # no gpu partitions are available, raise an error
src/toil/batchSystems/slurm.py:                            f"The job {jobName} is requesting GPUs, but the Slurm cluster does not appear to have an accessible partition with GPUs"
src/toil/batchSystems/slurm.py:                        and lowest_gpu_partition.time_limit < time_limit
src/toil/batchSystems/slurm.py:                        # TODO: find the lowest-priority GPU partition that has at least each job's time limit!
src/toil/batchSystems/slurm.py:                            lowest_gpu_partition.partition_name,
src/toil/batchSystems/slurm.py:                            lowest_gpu_partition.time_limit,
src/toil/batchSystems/slurm.py:                        f"--partition={lowest_gpu_partition.partition_name}"
src/toil/batchSystems/slurm.py:                    # there is a partition specified already, check if the partition has GPUs
src/toil/batchSystems/slurm.py:                            available_gpu_partitions = (
src/toil/batchSystems/slurm.py:                                self.boss.partitions.gpu_partitions
src/toil/batchSystems/slurm.py:                            if partition_name not in available_gpu_partitions:
src/toil/batchSystems/slurm.py:                                    f"Job {jobName} needs {gpus} GPUs, but specified partition {partition_name} is incompatible. This job may not work."
src/toil/batchSystems/slurm.py:                                    f"Try specifying one of these partitions instead: {', '.join(available_gpu_partitions)}."
src/toil/batchSystems/slurm.py:            gpus = self.count_needed_gpus(job_desc)
src/toil/batchSystems/slurm.py:                    gpus,
src/toil/batchSystems/slurm.py:            if accelerator["kind"] != "gpu":
src/toil/batchSystems/slurm.py:                        "The Toil Slurm batch system only supports gpu accelerators at the moment."
src/toil/batchSystems/torque.py:            gpus: Optional[int] = None,
src/toil/batchSystems/awsBatch.py:                accelerator["kind"] != "gpu"
src/toil/batchSystems/awsBatch.py:                or accelerator.get("brand", "nvidia") != "nvidia"
src/toil/batchSystems/awsBatch.py:                # We can only provide GPUs, and of those only nvidia ones.
src/toil/batchSystems/awsBatch.py:                        "AWS Batch can only provide nvidia gpu accelerators.",
src/toil/batchSystems/awsBatch.py:            gpus_needed = 0
src/toil/batchSystems/awsBatch.py:                if accelerator["kind"] == "gpu":
src/toil/batchSystems/awsBatch.py:                    # We just assume that all GPUs are equivalent when running
src/toil/batchSystems/awsBatch.py:                    gpus_needed += accelerator["count"]
src/toil/batchSystems/awsBatch.py:            if gpus_needed > 0:
src/toil/batchSystems/awsBatch.py:                # We need some GPUs so ask for them.
src/toil/batchSystems/awsBatch.py:                    {"type": "GPU", "value": gpus_needed}
src/toil/batchSystems/abstractGridEngineBatchSystem.py:                jobID, cpu, memory, command, jobName, environment, gpus = (
src/toil/batchSystems/abstractGridEngineBatchSystem.py:                    cpu, memory, jobID, command, jobName, environment, gpus
src/toil/batchSystems/abstractGridEngineBatchSystem.py:            gpus: Optional[int] = None,
src/toil/batchSystems/abstractGridEngineBatchSystem.py:    def count_needed_gpus(self, job_desc: JobDescription):
src/toil/batchSystems/abstractGridEngineBatchSystem.py:        Count the number of cluster-allocateable GPUs we want to allocate for the given job.
src/toil/batchSystems/abstractGridEngineBatchSystem.py:        gpus = 0
src/toil/batchSystems/abstractGridEngineBatchSystem.py:                if accelerator["kind"] == "gpu":
src/toil/batchSystems/abstractGridEngineBatchSystem.py:                    gpus += accelerator["count"]
src/toil/batchSystems/abstractGridEngineBatchSystem.py:            gpus = job_desc.accelerators
src/toil/batchSystems/abstractGridEngineBatchSystem.py:        return gpus
src/toil/batchSystems/abstractGridEngineBatchSystem.py:            gpus = self.count_needed_gpus(job_desc)
src/toil/batchSystems/abstractGridEngineBatchSystem.py:                    gpus,
src/toil/batchSystems/kubernetes.py:            if accelerator["kind"] != "gpu" and "model" not in accelerator:
src/toil/batchSystems/kubernetes.py:                # We can only provide GPUs or things with a model right now
src/toil/batchSystems/kubernetes.py:                        "The Toil Kubernetes batch system only knows how to request gpu accelerators or accelerators with a defined model.",
src/toil/batchSystems/kubernetes.py:            # Add in requirements for accelerators (GPUs).
src/toil/batchSystems/kubernetes.py:            # See https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
src/toil/batchSystems/kubernetes.py:            if accelerator["kind"] == "gpu":
src/toil/batchSystems/kubernetes.py:                # We can't schedule GPUs without a brand, because the
src/toil/batchSystems/kubernetes.py:                # Kubernetes resources are <brand>.com/gpu. If no brand is
src/toil/batchSystems/kubernetes.py:                # specified, default to nvidia, which is very popular.
src/toil/batchSystems/kubernetes.py:                vendor = accelerator.get("brand", "nvidia")
src/toil/batchSystems/kubernetes.py:        # the UCSC Kubernetes admins want it that way. For GPUs, Kubernetes
src/toil/options/common.py:        "Each accelerator specification can have a type (gpu [default], nvidia, amd, cuda, rocm, opencl, "
src/toil/options/common.py:        "or a specific model like nvidia-tesla-k80), and a count [default: 1]. If both a type and a count "
src/toil/wdl/wdltoil.py:            "gpu"
src/toil/wdl/wdltoil.py:            # For old WDL versions, guess whether the task wants GPUs if not specified.
src/toil/wdl/wdltoil.py:            use_gpus = (
src/toil/wdl/wdltoil.py:                runtime_bindings.has_binding("gpuCount")
src/toil/wdl/wdltoil.py:                or runtime_bindings.has_binding("gpuType")
src/toil/wdl/wdltoil.py:                or runtime_bindings.has_binding("nvidiaDriverVersion")
src/toil/wdl/wdltoil.py:            # The gpu field is the WDL 1.1 standard with a default value of false,
src/toil/wdl/wdltoil.py:            # truth on whether to use GPUs or not.
src/toil/wdl/wdltoil.py:            # Fields such as gpuType and gpuCount will control what GPUs are provided.
src/toil/wdl/wdltoil.py:            use_gpus = cast(
src/toil/wdl/wdltoil.py:                WDL.Value.Boolean, runtime_bindings.get("gpu", WDL.Value.Boolean(False))
src/toil/wdl/wdltoil.py:        if use_gpus:
src/toil/wdl/wdltoil.py:            # We want to have GPUs
src/toil/wdl/wdltoil.py:            # Get the GPU count if set, or 1 if not,
src/toil/wdl/wdltoil.py:            gpu_count: int = cast(
src/toil/wdl/wdltoil.py:                WDL.Value.Int, runtime_bindings.get("gpuCount", WDL.Value.Int(1))
src/toil/wdl/wdltoil.py:            # Get the GPU model constraint if set, or None if not
src/toil/wdl/wdltoil.py:            gpu_model: str | None = cast(
src/toil/wdl/wdltoil.py:                runtime_bindings.get("gpuType", WDL.Value.Null()),
src/toil/wdl/wdltoil.py:            # We can't enforce a driver version, but if an nvidia driver
src/toil/wdl/wdltoil.py:            # version is set, manually set nvidia brand
src/toil/wdl/wdltoil.py:            gpu_brand: str | None = (
src/toil/wdl/wdltoil.py:                "nvidia"
src/toil/wdl/wdltoil.py:                if runtime_bindings.has_binding("nvidiaDriverVersion")
src/toil/wdl/wdltoil.py:            accelerator_spec: dict[str, str | int] = {"kind": "gpu", "count": gpu_count}
src/toil/wdl/wdltoil.py:            if gpu_model is not None:
src/toil/wdl/wdltoil.py:                accelerator_spec["model"] = gpu_model
src/toil/wdl/wdltoil.py:            if gpu_brand is not None:
src/toil/wdl/wdltoil.py:                accelerator_spec["brand"] = gpu_brand
src/toil/wdl/wdltoil.py:                runtime_bindings.has_binding("gpuType")
src/toil/wdl/wdltoil.py:                or runtime_bindings.has_binding("gpuCount")
src/toil/wdl/wdltoil.py:                or runtime_bindings.has_binding("nvidiaDriverVersion")
src/toil/wdl/wdltoil.py:                    "Accelerator and GPU support "
src/toil/wdl/wdltoil.py:                # We might need to send GPUs and the current miniwdl doesn't do
src/toil/wdl/wdltoil.py:                    command line, and then adjust the result to pass GPUs and not
src/toil/wdl/wdltoil.py:                            # This logic will not work if a workflow needs to specify multiple GPUs of different types
src/toil/wdl/wdltoil.py:                            # Right now this assumes all GPUs on the node are the same; we only look at the first available GPU
src/toil/wdl/wdltoil.py:                            if accelerator["kind"] == "gpu":
src/toil/wdl/wdltoil.py:                                # Grab detected GPUs
src/toil/wdl/wdltoil.py:                                local_gpus: list[str | None] = [
src/toil/wdl/wdltoil.py:                                    if accel["kind"] == "gpu"
src/toil/wdl/wdltoil.py:                                # Tell singularity the GPU type
src/toil/wdl/wdltoil.py:                                gpu_brand = accelerator.get("brand") or local_gpus[0]
src/toil/wdl/wdltoil.py:                                if gpu_brand == "nvidia":
src/toil/wdl/wdltoil.py:                                    # Tell Singularity to expose nvidia GPUs
src/toil/wdl/wdltoil.py:                                elif gpu_brand == "amd":
src/toil/wdl/wdltoil.py:                                    # Tell Singularity to expose ROCm GPUs
src/toil/wdl/wdltoil.py:                                    extra_flags.add("--rocm")
src/toil/lib/accelerators.py:"""Accelerator (i.e. GPU) utilities for Toil"""
src/toil/lib/accelerators.py:def have_working_nvidia_smi() -> bool:
src/toil/lib/accelerators.py:    Return True if the nvidia-smi binary, from nvidia's CUDA userspace
src/toil/lib/accelerators.py:    it can fulfill a CUDARequirement.
src/toil/lib/accelerators.py:        subprocess.check_call(["nvidia-smi"])
src/toil/lib/accelerators.py:    Can be used with Docker's --gpus='"device=#,#,#"' option to forward the
src/toil/lib/accelerators.py:    right GPUs as seen from a Docker daemon.
src/toil/lib/accelerators.py:        "SLURM_STEP_GPUS",
src/toil/lib/accelerators.py:        "SLURM_JOB_GPUS",
src/toil/lib/accelerators.py:        "CUDA_VISIBLE_DEVICES",
src/toil/lib/accelerators.py:        "NVIDIA_VISIBLE_DEVICES",
src/toil/lib/accelerators.py:        # Any of these can have a list of GPU numbers, but the CUDA/NVIDIA ones
src/toil/lib/accelerators.py:        # also support a system of GPU GUIDs that we don't support.
src/toil/lib/accelerators.py:    # If we don't see a set of limits we understand, say we have all nvidia GPUs
src/toil/lib/accelerators.py:    return list(range(count_nvidia_gpus()))
src/toil/lib/accelerators.py:def have_working_nvidia_docker_runtime() -> bool:
src/toil/lib/accelerators.py:    Return True if Docker exists and can handle an "nvidia" runtime and the "--gpus" option.
src/toil/lib/accelerators.py:        # The runtime injects nvidia-smi; it doesn't seem to have to be in the image we use here
src/toil/lib/accelerators.py:                "nvidia",
src/toil/lib/accelerators.py:                "--gpus",
src/toil/lib/accelerators.py:                "nvidia-smi",
src/toil/lib/accelerators.py:def count_nvidia_gpus() -> int:
src/toil/lib/accelerators.py:    Return the number of nvidia GPUs seen by nvidia-smi, or 0 if it is not working.
src/toil/lib/accelerators.py:    # I don't have nvidia-smi, but cwltool knows how to do this, so we do what
src/toil/lib/accelerators.py:    # <https://github.com/common-workflow-language/cwltool/blob/6f29c59fb1b5426ef6f2891605e8fa2d08f1a8da/cwltool/cuda.py>
src/toil/lib/accelerators.py:                minidom.parseString(subprocess.check_output(["nvidia-smi", "-q", "-x"]))
src/toil/lib/accelerators.py:                .getElementsByTagName("attached_gpus")[0]
src/toil/lib/accelerators.py:    # TODO: Parse each gpu > product_name > text content and convert to some
src/toil/lib/accelerators.py:def count_amd_gpus() -> int:
src/toil/lib/accelerators.py:    Return the number of amd GPUs seen by rocm-smi, or 0 if it is not working.
src/toil/lib/accelerators.py:        # we believe this is the expected output for amd-smi, but we don't actually have and amd gpu to test against
src/toil/lib/accelerators.py:        # https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/using-AMD-SMI-CLI-tool.html
src/toil/lib/accelerators.py:        gpu_count = len(
src/toil/lib/accelerators.py:            [line for line in out.decode("utf-8").split("\n") if line.startswith("gpu")]
src/toil/lib/accelerators.py:        return gpu_count
src/toil/lib/accelerators.py:        # if the amd-smi command fails, try rocm-smi
src/toil/lib/accelerators.py:        # similarly, since we don't have an AMD gpu to test against, assume the output from the rocm-smi documentation:
src/toil/lib/accelerators.py:        # https://rocm.blogs.amd.com/software-tools-optimization/affinity/part-2/README.html#gpu-numa-configuration-rocm-smi-showtoponuma
src/toil/lib/accelerators.py:        out = subprocess.check_output(["rocm-smi"])
src/toil/lib/accelerators.py:        gpu_count = len(
src/toil/lib/accelerators.py:        return gpu_count
src/toil/lib/accelerators.py:    gpus: list[AcceleratorRequirement] = [
src/toil/lib/accelerators.py:        {"kind": "gpu", "brand": "nvidia", "api": "cuda", "count": 1}
src/toil/lib/accelerators.py:        for _ in range(count_nvidia_gpus())
src/toil/lib/accelerators.py:    gpus.extend(
src/toil/lib/accelerators.py:            {"kind": "gpu", "brand": "amd", "api": "rocm", "count": 1}
src/toil/lib/accelerators.py:            for _ in range(count_amd_gpus())
src/toil/lib/accelerators.py:    return gpus
src/toil/lib/accelerators.py:    # Since we only know about nvidia GPUs right now, we can just say our
src/toil/lib/accelerators.py:    # accelerator numbering space is the same as nvidia's GPU numbering space.
src/toil/lib/accelerators.py:    gpu_list = ",".join(str(i) for i in accelerator_numbers)
src/toil/lib/accelerators.py:    # Put this in several places: CUDA_VISIBLE_DEVICES for controlling
src/toil/lib/accelerators.py:    # processes right here, and SINGULARITYENV_CUDA_VISIBLE_DEVICES for
src/toil/lib/accelerators.py:        "CUDA_VISIBLE_DEVICES": gpu_list,
src/toil/lib/accelerators.py:        "SINGULARITYENV_CUDA_VISIBLE_DEVICES": gpu_list,
src/toil/lib/docker.py:    :param accelerators: Toil accelerator numbers (usually GPUs) to forward to
src/toil/lib/docker.py:        # TODO: Here we assume that the host accelerators are all GPUs
src/toil/lib/docker.py:                device_ids=[",".join(host_accelerators)], capabilities=[["gpu"]]
src/toil/utils/toilMain.py:    from toil.utils import toilRsyncCluster  # noqa

```
