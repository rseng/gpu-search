# https://github.com/BayraktarLab/cell2location

```console
Dockerfile:# base image maintained by the NVIDIA CUDA Installer Team - https://hub.docker.com/r/nvidia/cuda/
Dockerfile:FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
docs/commonerrors.md:#### 1. Training cell2location on GPU takes forever (>50 hours)
docs/commonerrors.md:1. Training cell2location using `cell2location.run_cell2location()` on GPU takes forever (>50 hours). Please check that cell2location is actually using the GPU. It is crucial to add this line in your script / notebook:
docs/commonerrors.md:os.environ["THEANO_FLAGS"] = 'device=cuda,floatX=float32,force_device=True'
docs/commonerrors.md:which tells theano (cell2location dependency) to use the GPU before importing cell2location (or it's dependencies - theano & pymc3).
docs/commonerrors.md:For data with 4039 locations and 10241 genes the analysis should take about 17-40 minutes depending on GPU hardware.
docs/commonerrors.md:**A.** Numerical accuracy issues with older CUDA versions. **Solution**: use our singularity and docker images with CUDA 10.2.
docs/commonerrors.md:#### 3. Theano fails to use the GPU at all (or cuDNN in particular)
docs/commonerrors.md:1. Use dockers/singularity images that are fully set up to work with the GPU (recommended).
docs/commonerrors.md:2. Add path to system CUDA installation to the following environmental variables by adding these lines to your `.bashrc` (modify accordingly for your system):
docs/commonerrors.md:# cuda v
docs/commonerrors.md:cuda_v=-10.2
docs/commonerrors.md:export CUDA_HOME=/usr/local/cuda$cuda_v
docs/commonerrors.md:export CUDA_PATH=$CUDA_HOME
docs/commonerrors.md:export LD_LIBRARY_PATH=/usr/local/cuda$cuda_v/lib64:$LD_LIBRARY_PATH
docs/commonerrors.md:export PATH=/usr/local/cuda$cuda_v/bin:$PATH
docs/notebooks/colab/setup_collab.sh:conda install -q -y --prefix /usr/local python=3.6 numpy pandas jupyter leidenalg python-igraph scanpy louvain hyperopt loompy cmake nose tornado dill ipython bbknn seaborn matplotlib request mkl-service pygpu theano --channel bioconda --channel conda-forge
docs/dockersingularity.md:   1. (recommended) If you plan to utilize GPU install [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
docs/dockersingularity.md:3. Run docker container with GPU support
docs/dockersingularity.md:       docker run -i --rm -p 8848:8888 --gpus all quay.io/vitkl/cell2location:latest
docs/dockersingularity.md:       docker run -v /host/directory:/container/directory -i --rm -p 8848:8888 --gpus all quay.io/vitkl/cell2location:latest
docs/dockersingularity.md:   1. For running without GPU support use
docs/dockersingularity.md:Singularity environments are used in the compute cluster environments (check with your local IT if Singularity is provided on your cluster). Follow the steps here to use it on your system, assuming that you need to use the GPU:
docs/dockersingularity.md:2. Submit a cluster job (LSF system) with GPU requested and start jupyter a notebook within a container (`--nv` option needed to use GPU):
docs/dockersingularity.md:bsub -q gpu_queue_name -M60000 \
docs/dockersingularity.md:  -R"select[mem>60000] rusage[mem=60000, ngpus_physical=1.00] span[hosts=1]"  \
docs/dockersingularity.md:  -gpu "mode=shared:j_exclusive=yes" -Is \
docs/installing_pymc3.md:## Installation of pymc3 version with GPU support
docs/installing_pymc3.md:Prior to installing cell2location package you need to install miniconda and create a conda environment containing pymc3 and theano ready for use on GPU. Follow the steps below:
docs/installing_pymc3.md:mkl-service pygpu --channel bioconda --channel conda-forge
docs/installing_pymc3.md:Do not install pymc3 and theano with conda because it will not use the system cuda (GPU drivers) and we had problems with cuda installed in the local environment, install them with pip:
tests/test_cell2location.py:    if torch.cuda.is_available():
tests/test_cell2location.py:        use_gpu = int(torch.cuda.is_available())
tests/test_cell2location.py:        accelerator = "gpu"
tests/test_cell2location.py:        use_gpu = None
tests/test_cell2location.py:    if use_gpu:
tests/test_cell2location.py:        device = f"cuda:{use_gpu}"
tests/test_cell2location.py:    #    #use_gpu=use_gpu,
tests/test_cell2location.py:    #    #use_gpu=use_gpu,
tests/test_cell2location.py:    if use_gpu:
tests/test_cell2location.py:        device = f"cuda:{use_gpu}"
README.md:### Conda environment for A100 GPUs
README.md:conda create -y -n cell2location_cuda118_torch22 python=3.10
README.md:conda activate cell2location_cuda118_torch22
README.md:python -m ipykernel install --user --name=cell2location_cuda118_torch22 --display-name='Environment (cell2location_cuda118_torch22)'
cell2location/models/_cell2location_WTA_model.py:    use_gpu
cell2location/models/_cell2location_WTA_model.py:        Use the GPU?
cell2location/models/_cell2location_WTA_model.py:            data is copied to device (e.g., GPU).
cell2location/models/_cell2location_WTA_model.py:        use_gpu: Optional[Union[str, int, bool]] = None,
cell2location/models/_cell2location_WTA_model.py:        use_gpu
cell2location/models/_cell2location_WTA_model.py:            Use default GPU if available (if None or True), or index of GPU to use (if int),
cell2location/models/_cell2location_WTA_model.py:            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
cell2location/models/_cell2location_WTA_model.py:            data is copied to device (e.g., GPU).
cell2location/models/_cell2location_WTA_model.py:            # use data splitter which moves data to GPU once
cell2location/models/_cell2location_WTA_model.py:                use_gpu=use_gpu,
cell2location/models/_cell2location_WTA_model.py:                use_gpu=use_gpu,
cell2location/models/_cell2location_WTA_model.py:            use_gpu=use_gpu,
cell2location/models/_cell2location_WTA_model.py:                batch_size - data batch size (keep low enough to fit on GPU, default 2048).
cell2location/models/_cell2location_WTA_model.py:                use_gpu - use gpu for generating samples?
cell2location/models/reference/_reference_model.py:    use_gpu
cell2location/models/reference/_reference_model.py:        Use the GPU?
cell2location/models/reference/_reference_model.py:            data is copied to device (e.g., GPU).
cell2location/models/reference/_reference_model.py:                batch_size - data batch size (keep low enough to fit on GPU, default 2048).
cell2location/models/reference/_reference_model.py:                use_gpu - use gpu for generating samples?
cell2location/models/base/_pyro_mixin.py:        use_gpu
cell2location/models/base/_pyro_mixin.py:            Bool, use gpu?
cell2location/models/base/_pyro_mixin.py:        use_gpu
cell2location/models/base/_pyro_mixin.py:            Bool, use gpu?
cell2location/models/base/_pyro_mixin.py:        use_gpu
cell2location/models/base/_pyro_mixin.py:            Bool, use gpu?
cell2location/models/base/_pyro_mixin.py:        torch.cuda.empty_cache()
cell2location/models/base/_pyro_mixin.py:        torch.cuda.empty_cache()
cell2location/models/_cell2location_model.py:    use_gpu
cell2location/models/_cell2location_model.py:        Use the GPU?
cell2location/models/_cell2location_model.py:            data is copied to device (e.g., GPU).
cell2location/models/_cell2location_model.py:        use_gpu: Optional[Union[str, int, bool]] = None,
cell2location/models/_cell2location_model.py:        use_gpu
cell2location/models/_cell2location_model.py:            Use default GPU if available (if None or True), or index of GPU to use (if int),
cell2location/models/_cell2location_model.py:            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
cell2location/models/_cell2location_model.py:            data is copied to device (e.g., GPU).
cell2location/models/_cell2location_model.py:            # use data splitter which moves data to GPU once
cell2location/models/_cell2location_model.py:                use_gpu=use_gpu,
cell2location/models/_cell2location_model.py:            use_gpu=use_gpu,
cell2location/models/_cell2location_model.py:                batch_size - data batch size (keep low enough to fit on GPU, default 2048).
cell2location/models/_cell2location_model.py:                use_gpu - use gpu for generating samples?
cell2location/run_colocation.py:        "use_cuda": False,

```
