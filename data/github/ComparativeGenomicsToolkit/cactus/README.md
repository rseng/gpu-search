# https://github.com/ComparativeGenomicsToolkit/cactus

```console
ReleaseNotes.md:- GPU lastz implementation changed from `SegAlign` to `KegAlign`, and should be more robust and better supported as a result. 
ReleaseNotes.md:- Updated local alignment selection criteria. At each internal node of the guide tree Cactus picks a set of pairwise local alignments between the genomes being aligned to construct an initial sequence graph representing the whole genome alignment. This sequence graph is then refined and an ancestral sequence inferred to complete the alignment process for the internal node. The pairwise local alignments are generated with LASTZ (or SegAlign if using the GPU mode). To create a reliable subset of local alignments Cactus employs a chaining process that organizes the pairwise local alignments into pairwise chains of syntenic alignments, using a process akin to the chains and nets procedure used by the UCSC Browser. Previously, each genome being aligned, including both ingroup and outgroup genomes, was used to select a set of primary chains. That is, for each genome sequence non-overlapping chains of pairwise alignments were chosen, each of which could be to any of the other genomes in the set. Only these primary chains were then fed into the Cactus process to construct the sequence graph. This heuristic works reasonably well, in effect it allows each subsequence to choose a sequence in another genome with which it shares a most recent common ancestor. In the new, updated version we tweak this process slightly to avoid rare edge cases. Now each sequence in each ingroup genome picks primary chains only to other ingroup genomes. Thus the set of primary chains for ingroup genomes does not include any outgroup alignments. The outgroup genomes then get to pick primary chains to the ingroups, effectively voting on which parts of the ingroups they are syntenic too. The result of this change is that the outgroups are effectively only used to determine ancestral orderings and do not ever prevent the syntenic portions of two ingroups from aligning together.
ReleaseNotes.md:- Fix error where gpu support on singularity is completely broken.
ReleaseNotes.md:- CPU docker release now made locally as done for GPU
ReleaseNotes.md:- `--binariesMode docker` will automatically point to release image (using GPU one as appropriate)
ReleaseNotes.md:This release patches a Toil bug that broke GPU support on single-machine.
ReleaseNotes.md:- Update to Toil v5.12, which fixes issue where trying to use GPUs on single machine batch systems would lead to a crash
ReleaseNotes.md:- Upgrade to Toil 5.8.0, which allows GPU counts to be assigned to jobs (which can be passed from cactus via the `--gpu` option). Toil only currently supports this functionality in single machine mode.
ReleaseNotes.md:- Fix bug introduced in v2.3.1 that broke GPU-enabled lastz preprocessing
ReleaseNotes.md:- Fix `cactus-blast` crash when using GPU on mammalian-sized genomes 
ReleaseNotes.md:- Update Segalign to fix crash while lastz-repeatmasking certain (fragmented?) assemblies using GPUs.
ReleaseNotes.md:- Upgrade release GPU Docker image from Ubuntu 18.04 / Cuda 10.2 to Ubuntu 20.04 / Cuda 11.4.3 (the most recent Cuda currently supported by Terra)
ReleaseNotes.md:The `--gpu` option still doesn't always work.  When using the GPU outside the gpu Docker Release, it is still advised to set gpuLastz="true" in src/cactus/cactus_progressive_config.xml (and rerun `pip install -U`).  
ReleaseNotes.md:- Fix bug where `cactus_fasta_softmask_intervals.py` was expecting 1-based intervals from GPU lastz repeatmasker.
ReleaseNotes.md:GPU Lastz version used in GPU-enabled Docker image: [8b63a0fe1c06b3511dfc4660bd0f9fb7ad7176e7](https://github.com/ComparativeGenomicsToolkit/SegAlign/commit/8b63a0fe1c06b3511dfc4660bd0f9fb7ad7176e7)
ReleaseNotes.md:- GPU lastz updated for more disk-efficient repeat masking and better error handling
ReleaseNotes.md:GPU Lastz version used in GPU-enabled Docker image: [8b63a0fe1c06b3511dfc4660bd0f9fb7ad7176e7](https://github.com/ComparativeGenomicsToolkit/SegAlign/commit/8b63a0fe1c06b3511dfc4660bd0f9fb7ad7176e7)
ReleaseNotes.md:This release fixes bugs related to GPU lastz
ReleaseNotes.md:- Cactus fixed to correctly handle GPU lastz repeatmasking output, as well to keep temporary files inside Toil's working directory.
ReleaseNotes.md:- GPU lastz updated to fix crash
ReleaseNotes.md:GPU Lastz version used in GPU-enabled Docker image: [12af3c295da7e1ca87e01186ddf5b0088cb29685](https://github.com/ComparativeGenomicsToolkit/SegAlign/commit/12af3c295da7e1ca87e01186ddf5b0088cb29685)
ReleaseNotes.md:This release adds GPU lastz repeatmasking support in the preprocessor, and includes hal2vg
ReleaseNotes.md: - GPU lastz repeat masking is about an order of magnitude faster than CPU masking, and should provide better results.  It's toggled on in the config file or by using the GPU-enabled Docker image.
ReleaseNotes.md:GPU Lastz version used in GPU-enabled Docker image: [f84a94663bbd6c42543f63b50c5843b0b5025dda](https://github.com/ComparativeGenomicsToolkit/SegAlign/commit/f84a94663bbd6c42543f63b50c5843b0b5025dda)
ReleaseNotes.md:GPU Lastz version used in GPU-enabled Docker image: [3e14c3b8ceeb139b3b929b5993d96d8e5d3ef9fa](https://github.com/ComparativeGenomicsToolkit/SegAlign/commit/3e14c3b8ceeb139b3b929b5993d96d8e5d3ef9fa)
ReleaseNotes.md:    - GPU lastz support
ReleaseNotes.md:- Provide GPU-enabled Docker image for release
ReleaseNotes.md:GPU Lastz version used in GPU-enabled Docker image: [3e14c3b8ceeb139b3b929b5993d96d8e5d3ef9fa](https://github.com/ComparativeGenomicsToolkit/SegAlign/commit/3e14c3b8ceeb139b3b929b5993d96d8e5d3ef9fa)
Dockerfile.kegalign:# Reminder: if updating this image, also update it in build-tools/makeGpuDockerRelease
Dockerfile.kegalign:FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as builder
Dockerfile.kegalign:FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04
doc/progressive.md:* [GPU Acceleration](#gpu-acceleration)
doc/progressive.md:In order to use `--gpu`, you *must* use `--binariesMode docker` or `--binariesMode singularity` since `kegalign` is not included in the binary release.  
doc/progressive.md:* **preprocess/blast**: `n1-standard-32 + 8 v100 GPUs`
doc/progressive.md:               --blastDisk 375Gi --blastCores 32 --gpu 8 --blastMemory 120Gi \
doc/progressive.md:## GPU Acceleration
doc/progressive.md:[KegAlign](https://github.com/galaxyproject/KegAlign), a GPU-accelerated version of lastz, can be used in the "blast" phase to speed up the runtime considerably, provided the right hardware is available. Unlike lastz, the input sequences do not need to be chunked before running KegAlign, so it also reduces the number of Toil jobs substantially.  The [GPU-enabled Docker releases](https://github.com/ComparativeGenomicsToolkit/cactus/releases) have KegAlign turned on by default and require no extra options from the user.  Otherwise, it is possible to [manually install it](https://github.com/galaxyproject/KegAlign#-installation) and then enable it in `cactus` using the `--gpu` command line option. One effective way of ensuring that only GPU-enabled parts of the workflow are run on GPU nodes is on Terra with `cactus-prepare --gpu --wdl` (see above example).
doc/progressive.md:By default `--gpu` will give all available GPUs to each KegAlign job. This can be tuned by passing in a numeric value, ex `--gpu 8` to assign 8 GPUs to each KegAlign job.  In non-single-machine batch systems, it is mandatory to set an exact value with `--gpu`.  
doc/progressive.md:GPUs must
doc/progressive.md:* support CUDA
doc/progressive.md:* have at least 8GB GPU memory (for mammal-sized input)
doc/progressive.md:We've tested KegAlign on Nvidia V100 and A10G GPUs. See the Terra example above for suggested node type on GCP.   
doc/progressive.md:### Using GPU Acceleration on a Cluster
doc/progressive.md:Since `KegAlign` is only released in the GPU-enabled docker image, that's the easiest way to run it. When running on a cluster, this usually means the best way to use it is with `--binariesMode docker --gpu <N>`.  This way cactus is installed locally on your virtual environment and can run slurm commands like `sbatch` (that aren't available in the Cactus container), but KegAlign itself will be run from inside Docker.
doc/progressive.md:**Important**: Consider using `--lastzMemory` when using GPU acceleration on a cluster. Like `--consMemory`, it lets you override the amount of memory Toil requests which can help with errors if Cactus's automatic estimate is either too low (cluster evicts the job) or too high (cluster cannot schedule the job).  
doc/progressive.md:**Q**: The `--gpu` option isn't working for me.
doc/progressive.md:**A**: Unless you've set up KegAlign yourself, the GPU option will only work using the gpu-enabled Docker image (name ending in `-gpu`).  If you are running directly from the container make sure you use `docker run`'s `--gpus` option to enable GPUs in your container. If you are using `singularity`, the option is `--nv`.
doc/progressive.md:**Q**: But what if I want to use `--gpu` on my cluster? When I try from inside the GPU-enabled container, none of my cluster commands (ex `qsub`) are available.
doc/mc-paper/hprc/terra-files/bwa_deepvariant.wdl:        # DeepVariant container to use for GPU steps
doc/mc-paper/hprc/terra-files/bwa_deepvariant.wdl:        String DV_GPU_CONTAINER = "google/deepvariant:1.3.0-gpu"
doc/mc-paper/hprc/terra-files/bwa_deepvariant.wdl:                in_dv_gpu_container=DV_GPU_CONTAINER,
doc/mc-paper/hprc/terra-files/bwa_deepvariant.wdl:        String in_dv_gpu_container
doc/mc-paper/hprc/terra-files/bwa_deepvariant.wdl:        gpuType: "nvidia-tesla-t4"
doc/mc-paper/hprc/terra-files/bwa_deepvariant.wdl:        gpuCount: 1
doc/mc-paper/hprc/terra-files/bwa_deepvariant.wdl:        nvidiaDriverVersion: "418.87.00"
doc/mc-paper/hprc/terra-files/bwa_deepvariant.wdl:        docker: in_dv_gpu_container
README.md:- B Gulhan, R Burhans, R Harris, M Kandemir, M Haeussler, A Nekrutenko for [KegAlign](https://github.com/galaxyproject/KegAlign), the GPU-accelerated version of LastZ.
README.md:By default, cactus will use the image corresponding to the latest release when running docker binaries. This is usually okay, but can be overridden with the `CACTUS_DOCKER_ORG` and `CACTUS_DOCKER_TAG` environment variables.  For example, to use GPU release 2.4.4, run `export CACTUS_DOCKER_TAG=v2.4.4-gpu` before running cactus.
build-tools/makeGpuDockerRelease:# Make a gpu-enabled docker image and push it to quay
build-tools/makeGpuDockerRelease:binBuildDir="${buildDir}/gpu-docker-tmp"
build-tools/makeGpuDockerRelease:sed '0,/FROM/! s/FROM.*/FROM kegalign:local/' Dockerfile  | sed -e '0,/FROM/s/FROM.*/FROM nvidia\/cuda:11.7.1-devel-ubuntu22.04 as builder/g' > Dockerfile.gpu
build-tools/makeGpuDockerRelease:# enable gpu by default
build-tools/makeGpuDockerRelease:sed -i src/cactus/cactus_progressive_config.xml -e 's/gpu="0"/gpu="all"/g' -e 's/realign="1"/realign="0"/'
build-tools/makeGpuDockerRelease:docker build . -f Dockerfile.gpu -t ${dockname}:${REL_TAG}-gpu
build-tools/makeGpuDockerRelease:sed -i src/cactus/cactus_progressive_config.xml -e 's/gpu="all"/gpu="0"/g' -e 's/realign="0"/realign="1"/'
build-tools/makeGpuDockerRelease:read -p "Are you sure you want to push ${dockname}:${REL_TAG}-gpu to quay?" yn
build-tools/makeGpuDockerRelease:    [Yy]* ) docker push ${dockname}:${REL_TAG}-gpu; break;;
src/cactus/cactus_progressive_config.xml:	<!-- The preprocessor for cactus_lastzRepeatMask masks every seed that is part of more than XX other alignments, this stops a combinatorial explosion in pairwise alignments. gpu sets the number of gpus (if >0, use kegalign in stead of lastz. can be set to 'all' for all available GPUs). Note: Setting unmask to 1 will cause an assertion failure if gpu is not 0. -->
src/cactus/cactus_progressive_config.xml:	<preprocessor unmask="0" chunkSize="10000000" proportionToSample="0.2" memory="littleMemory" preprocessJob="lastzRepeatMask" minPeriod="50" lastzOpts='--step=3 --ambiguous=iupac,100,100 --ungapped --queryhsplimit=keep,nowarn:1500' gpu="0" active="0"/>
src/cactus/cactus_progressive_config.xml:	<!-- gpu Toggle on kegalign instead of lastz and set the number of gpus for each kegalign job. 0: disable kegalign, 'all': use all availabled GPUs-->
src/cactus/cactus_progressive_config.xml:		   gpu="0"
src/cactus/refmap/cactus_pangenome.py:    options.gpu = False
src/cactus/blast/cactus_blast.py:    parser.add_argument("--gpu", nargs='?', const='all', default=None, help="toggle on GPU-enabled lastz, and specify number of GPUs (all available if no value provided)")
src/cactus/blast/cactus_blast.py:    parser.add_argument("--lastzCores", type=int, default=None, help="Number of cores for each lastz/segalign job, only relevant when running with --gpu")
src/cactus/blast/cactus_blast.py:            # apply gpu override
src/cactus/blast/cactus_blast.py:            config_wrapper.initGPU(options)
src/cactus/paf/local_alignment.py:    gpu = getOptionalAttrib(lastz_params_node, 'gpu', typeFn=int, default=0)
src/cactus/paf/local_alignment.py:    if gpu:
src/cactus/paf/local_alignment.py:        assert gpu > 0
src/cactus/paf/local_alignment.py:        lastz_params += ' --num_gpu {} --num_threads {}'.format(gpu, job.cores)
src/cactus/paf/local_alignment.py:    kegalign_messages = cactus_call(parameters=lastz_cmd, outfile=alignment_file, work_dir=work_dir, returnStdErr=gpu>0, gpus=gpu,
src/cactus/paf/local_alignment.py:    if gpu:
src/cactus/paf/local_alignment.py:    gpu = getOptionalAttrib(lastz_params_node, 'gpu', typeFn=int, default=0)
src/cactus/paf/local_alignment.py:    if gpu:
src/cactus/paf/local_alignment.py:        # wga-gpu has a 6G limit, so we always override
src/cactus/paf/local_alignment.py:    accelerators = 'cuda:{}'.format(gpu) if gpu else None
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:            gpu=0,
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:            gpuLastzInterval=3000000,
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:        self.gpu = gpu
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:        self.gpuLastzInterval = gpuLastzInterval
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:        elif repeatMaskOptions.gpu:
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:        accelerators = ['cuda:{}'.format(repeatMaskOptions.gpu)] if repeatMaskOptions.gpu else None            
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:        if self.repeatMaskOptions.gpu:
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:        tool = 'run_kegalign' if self.repeatMaskOptions.gpu else 'lastz'
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:        if self.repeatMaskOptions.gpu:
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:                                        returnStdErr=self.repeatMaskOptions.gpu,
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:                                        gpus=self.repeatMaskOptions.gpu)
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:        if self.repeatMaskOptions.gpu:
src/cactus/preprocessor/lastzRepeatMasking/cactus_lastzRepeatMask.py:        # covered_intervals is part of segalign, so only run if not in gpu mode
src/cactus/preprocessor/cactus_preprocessor.py:                 gpu=0, lastz_memory=None, dnabrnnOpts=None,
src/cactus/preprocessor/cactus_preprocessor.py:        self.gpu = gpu
src/cactus/preprocessor/cactus_preprocessor.py:        self.gpuLastzInterval = self.chunkSize
src/cactus/preprocessor/cactus_preprocessor.py:        if self.gpu:
src/cactus/preprocessor/cactus_preprocessor.py:            # wga-gpu has a 6G limit, so we always override
src/cactus/preprocessor/cactus_preprocessor.py:                                                  gpu=self.prepOptions.gpu,
src/cactus/preprocessor/cactus_preprocessor.py:                                                  gpuLastzInterval=self.prepOptions.gpuLastzInterval,
src/cactus/preprocessor/cactus_preprocessor.py:            if self.prepOptions.gpu:
src/cactus/preprocessor/cactus_preprocessor.py:                # when using gpu lastz, we pass through the proportion directly to segalign
src/cactus/preprocessor/cactus_preprocessor.py:                                              gpu = getOptionalAttrib(prepNode, "gpu", typeFn=int, default=0),
src/cactus/preprocessor/cactus_preprocessor.py:                  gpu_override=False, options=None):
src/cactus/preprocessor/cactus_preprocessor.py:    parser.add_argument("--gpu", nargs='?', const='all', default=None, help="toggle on GPU-enabled lastz, and specify number of GPUs (all available if no value provided)")
src/cactus/preprocessor/cactus_preprocessor.py:    parser.add_argument("--lastzCores", type=int, default=None, help="Number of cores for each lastz/segalign job, only relevant when running with --gpu")
src/cactus/preprocessor/cactus_preprocessor.py:    # toggle on the gpu
src/cactus/preprocessor/cactus_preprocessor.py:    config_wrapper.initGPU(options)
src/cactus/setup/cactus_align.py:    parser.add_argument("--gpu", action="store_true",
src/cactus/setup/cactus_align.py:                        help="Enable GPU acceleration by using Segaling instead of lastz")
src/cactus/setup/cactus_align.py:        config_wrapper.initGPU(options)
src/cactus/shared/configWrapper.py:from toil.lib.accelerators import count_nvidia_gpus
src/cactus/shared/configWrapper.py:    def initGPU(self, options):
src/cactus/shared/configWrapper.py:        """ Turn on GPU and / or check options make sense """
src/cactus/shared/configWrapper.py:        # first, we override the config with --gpu from the options
src/cactus/shared/configWrapper.py:        # (we'll never use the options.gpu after -- only the config)
src/cactus/shared/configWrapper.py:        if options.gpu:
src/cactus/shared/configWrapper.py:            findRequiredNode(self.xmlRoot, "blast").attrib["gpu"] = str(options.gpu)
src/cactus/shared/configWrapper.py:                    node.attrib["gpu"] = str(options.gpu)
src/cactus/shared/configWrapper.py:                raise RuntimeError('--latest cannot be used with --gpu')
src/cactus/shared/configWrapper.py:        # we need to make sure that gpu is set to an integer (replacing 'all' with a count)
src/cactus/shared/configWrapper.py:        def get_gpu_count():
src/cactus/shared/configWrapper.py:                gpu_count = count_nvidia_gpus()
src/cactus/shared/configWrapper.py:                if not gpu_count:
src/cactus/shared/configWrapper.py:                    raise RuntimeError('Unable to automatically determine number of GPUs: Please set with --gpu N')
src/cactus/shared/configWrapper.py:                raise RuntimeError('--gpu N required to set number of GPUs on non single_machine batch systems')
src/cactus/shared/configWrapper.py:            return gpu_count
src/cactus/shared/configWrapper.py:        # ensure integer gpu for blast phase
src/cactus/shared/configWrapper.py:        lastz_gpu = findRequiredNode(self.xmlRoot, "blast").attrib["gpu"]
src/cactus/shared/configWrapper.py:        if lastz_gpu.lower() == 'false':
src/cactus/shared/configWrapper.py:            lastz_gpu = "0"
src/cactus/shared/configWrapper.py:        elif lastz_gpu.lower() == 'true':
src/cactus/shared/configWrapper.py:            lastz_gpu = 'all'            
src/cactus/shared/configWrapper.py:        if lastz_gpu == 'all':
src/cactus/shared/configWrapper.py:            lastz_gpu = get_gpu_count()
src/cactus/shared/configWrapper.py:        elif not lastz_gpu.isdigit() or int(lastz_gpu) < 0:
src/cactus/shared/configWrapper.py:            raise RuntimeError('Invalid value for blast gpu count, {}. Please specify a numeric value with --gpu'.format(lastz_gpu))
src/cactus/shared/configWrapper.py:        findRequiredNode(self.xmlRoot, "blast").attrib["gpu"] = str(lastz_gpu)
src/cactus/shared/configWrapper.py:        # ensure integer gpu for lastz repeatmasking in preprocessor phase
src/cactus/shared/configWrapper.py:                pp_gpu = str(node.attrib["gpu"])
src/cactus/shared/configWrapper.py:                if pp_gpu.lower() == 'false':
src/cactus/shared/configWrapper.py:                    pp_gpu = "0"
src/cactus/shared/configWrapper.py:                elif pp_gpu.lower() == 'true':
src/cactus/shared/configWrapper.py:                    pp_gpu = 'all'
src/cactus/shared/configWrapper.py:                if pp_gpu == 'all':
src/cactus/shared/configWrapper.py:                    pp_gpu = get_gpu_count()
src/cactus/shared/configWrapper.py:                elif not pp_gpu.isdigit() or int(pp_gpu) < 0:
src/cactus/shared/configWrapper.py:                    raise RuntimeError('Invalid value for repeatmask gpu count, {}. Please specify a numeric value with --gpu'.format(lastz_gpu))
src/cactus/shared/configWrapper.py:                node.attrib["gpu"] = str(pp_gpu)
src/cactus/shared/configWrapper.py:        if getOptionalAttrib(findRequiredNode(self.xmlRoot, "blast"), 'gpu', typeFn=int, default=0):
src/cactus/shared/configWrapper.py:                    raise RuntimeError('--lastzCores must be used with --gpu on non-singlemachine batch systems')
src/cactus/shared/configWrapper.py:        # make absolutely sure realign is never turned on with the gpu.  they don't work together because
src/cactus/shared/configWrapper.py:        # realign is cpu based, which is wasteful on a gpu node
src/cactus/shared/configWrapper.py:        # realign is too slow, and largely negates gpu speed boost
src/cactus/shared/configWrapper.py:        if getOptionalAttrib(findRequiredNode(self.xmlRoot, "blast"), "gpu", typeFn=int, default=0) and \
src/cactus/shared/configWrapper.py:            logger.warning("Switching off blast realignment as it is incompatible with GPU mode")
src/cactus/shared/common.py:def getDockerTag(gpu=False):
src/cactus/shared/common.py:        return 'v2.9.2' + ('-gpu' if gpu else '')
src/cactus/shared/common.py:def getDockerImage(gpu=False):
src/cactus/shared/common.py:            gpu = bool(options.gpu) if 'gpu' in options else False
src/cactus/shared/common.py:                                       "docker://" + getDockerImage(gpu=gpu)], stderr=subprocess.PIPE)
src/cactus/shared/common.py:                                       "docker://" + getDockerImage(gpu=gpu)])
src/cactus/shared/common.py:                       gpus=None,
src/cactus/shared/common.py:    if gpus:
src/cactus/shared/common.py:            tool = getDockerImage(gpu = bool(gpus))
src/cactus/shared/common.py:                  gpus=None,
src/cactus/shared/common.py:    if gpus:
src/cactus/shared/common.py:        if 'SLURM_JOB_GPUS' in os.environ:
src/cactus/shared/common.py:            # this allows slurm to identify which gpus are free
src/cactus/shared/common.py:            base_docker_call += ['--gpus', '"device={}"'.format(os.environ['SLURM_JOB_GPUS'])]            
src/cactus/shared/common.py:            base_docker_call += ['--gpus', str(gpus)]
src/cactus/shared/common.py:    docker_tag = getDockerTag(gpu=bool(gpus))
src/cactus/shared/common.py:                gpus=None,
src/cactus/shared/common.py:                                            gpus=gpus, cpus=cpus)
src/cactus/shared/common.py:                                  gpus=gpus, cpus=cpus)
src/cactus/progressive/cactus_progressive.py:    parser.add_argument("--gpu", nargs='?', const='all', default=None, help="toggle on GPU-enabled lastz, and specify number of GPUs (all available if no value provided)")
src/cactus/progressive/cactus_progressive.py:    parser.add_argument("--lastzCores", type=int, default=None, help="Number of cores for each lastz/segalign job, only relevant when running with --gpu")
src/cactus/progressive/cactus_progressive.py:            # apply gpu override
src/cactus/progressive/cactus_progressive.py:            config_wrapper.initGPU(options)
src/cactus/progressive/cactus_prepare.py:from toil.lib.accelerators import count_nvidia_gpus
src/cactus/progressive/cactus_prepare.py:    parser.add_argument("--gpu", nargs='?', const='all', default=None, help="toggle on GPU-enabled lastz, and specify number of GPUs (all available if no value provided)")
src/cactus/progressive/cactus_prepare.py:    parser.add_argument("--gpuType", default="nvidia-tesla-v100", help="GPU type (to set in WDL runtime parameters, use only with --wdl)")
src/cactus/progressive/cactus_prepare.py:    parser.add_argument("--nvidiaDriver", default="470.82.01", help="Nvidia driver version")
src/cactus/progressive/cactus_prepare.py:    parser.add_argument("--gpuZone", default="us-central1-a", help="zone used for gpu task")
src/cactus/progressive/cactus_prepare.py:    parser.add_argument("--zone", default="us-west2-a", help="zone used for all but gpu tasks")
src/cactus/progressive/cactus_prepare.py:        if options.gpu == 'all':
src/cactus/progressive/cactus_prepare.py:                options.gpu = count_nvidia_gpus()
src/cactus/progressive/cactus_prepare.py:                if not options.gpu:
src/cactus/progressive/cactus_prepare.py:                    raise RuntimeError('Unable to automatically determine number of GPUs: Please set with --gpu N')
src/cactus/progressive/cactus_prepare.py:                raise RuntimeError('--gpu N required in order to use GPUs on non single_machine batch systems')
src/cactus/progressive/cactus_prepare.py:    if not options.wdl and options.gpuType != "nvidia-tesla-v100":
src/cactus/progressive/cactus_prepare.py:        raise RuntimeError("--gpuType can only be used with --wdl ")
src/cactus/progressive/cactus_prepare.py:    if options.wdl and options.gpu == 'all':
src/cactus/progressive/cactus_prepare.py:        raise RuntimeError("Number of gpus N must be specified with --gpu N when using --wdl")
src/cactus/progressive/cactus_prepare.py:    # https://cromwell.readthedocs.io/en/stable/RuntimeAttributes/#gpucount-gputype-and-nvidiadriverversion
src/cactus/progressive/cactus_prepare.py:    # note: k80 not included as WGA_GPU doesn't run on it.  
src/cactus/progressive/cactus_prepare.py:    acceptable_gpus = ['nvidia-tesla-v100', 'nvidia-tesla-p100', 'nvidia-tesla-p4', 'nvidia-tesla-t4']
src/cactus/progressive/cactus_prepare.py:    if options.gpuType not in acceptable_gpus:
src/cactus/progressive/cactus_prepare.py:        raise RuntimeError('--gpuType {} not supported by Terra.  Acceptable types are {}'.format(
src/cactus/progressive/cactus_prepare.py:            options.gpuType, acceptable_gpus))
src/cactus/progressive/cactus_prepare.py:    options.gpu_preprocessor = options.gpu and config.getPreprocessorActive('lastzRepeatMask')
src/cactus/progressive/cactus_prepare.py:                ' --gpu {}'.format(options.gpu) if options.gpu_preprocessor else '',
src/cactus/progressive/cactus_prepare.py:                    ' --gpu {}'.format(options.gpu) if options.gpu else '',
src/cactus/progressive/cactus_prepare.py:                                                                ' --gpu {}'.format(options.gpu) if options.gpu else '')
src/cactus/progressive/cactus_prepare.py:    if options.gpu_preprocessor:
src/cactus/progressive/cactus_prepare.py:        s += '        gpuType: \"{}\"\n'.format(options.gpuType)
src/cactus/progressive/cactus_prepare.py:        s += '        gpuCount: {}\n'.format(options.gpu)
src/cactus/progressive/cactus_prepare.py:        s += '        nvidiaDriverVersion: \"{}\"\n'.format(options.nvidiaDriver)
src/cactus/progressive/cactus_prepare.py:        s += '        docker: \"{}\"\n'.format(getDockerImage(gpu=True))
src/cactus/progressive/cactus_prepare.py:        s += '        zones: \"{}\"\n'.format(options.gpuZone)
src/cactus/progressive/cactus_prepare.py:    if options.gpu_preprocessor:
src/cactus/progressive/cactus_prepare.py:        cmd += ['--gpu', options.gpu]
src/cactus/progressive/cactus_prepare.py:                                                                                    ' --gpu {}'.format(options.gpu) if options.gpu else '')
src/cactus/progressive/cactus_prepare.py:    if options.gpu:
src/cactus/progressive/cactus_prepare.py:        s += '        gpuType: \"{}\"\n'.format(options.gpuType)
src/cactus/progressive/cactus_prepare.py:        s += '        gpuCount: {}\n'.format(options.gpu)
src/cactus/progressive/cactus_prepare.py:        s += '        nvidiaDriverVersion: \"{}\"\n'.format(options.nvidiaDriver)
src/cactus/progressive/cactus_prepare.py:        s += '        docker: \"{}\"\n'.format(getDockerImage(gpu=True))
src/cactus/progressive/cactus_prepare.py:        s += '        zones: \"{}\"\n'.format(options.gpuZone)
src/cactus/progressive/cactus_prepare.py:    if options.gpu:
src/cactus/progressive/cactus_prepare.py:        blast_cmd += ['--gpu', options.gpu]

```
