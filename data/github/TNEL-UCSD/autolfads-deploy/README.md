# https://github.com/TNEL-UCSD/autolfads-deploy

```console
ray/train_lfads.py:from lfads_tf2.utils import restrict_gpu_usage
ray/train_lfads.py:restrict_gpu_usage(gpu_ix=0)
ray/run_pbt.py:RESOURCES_PER_TRIAL = {"cpu": 2, "gpu": 0.5}
ray/ray_cluster_template.yaml:# NOTE: Modified the following commands to use the tf2-gpu environment
ray/run_random_search.py:    resources_per_trial={"cpu": 3, "gpu": 0.5},
images/lfads-tensorflow/cpu.Dockerfile:# Build tensorflow-addons so it doesn't force installation of tensorflow-gpu
images/lfads-tensorflow/cpu.Dockerfile:    && sed -i s/tensorflow-gpu\ ==\ 2.0.0-rc0/tensorflow\ ==\ ${TENSORFLOW_VERSION}/ setup.py \
images/lfads-tensorflow/cpu.Dockerfile:    && sed -i s/tensorflow-gpu==2.0.0/tensorflow/ /opt/lfads-tf2/setup.py \
images/lfads-tensorflow/gpu.Dockerfile:## docker build -t ucsdtnel/autolfads:latest-gpu -f gpu.Dockerfile .
images/lfads-tensorflow/gpu.Dockerfile:##    ucsdtnel/autolfads:latest-gpu
images/lfads-tensorflow/gpu.Dockerfile:FROM tensorflow/tensorflow:$TENSORFLOW_VERSION-gpu-py3
images/lfads-tensorflow/gpu.Dockerfile:RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
images/lfads-tensorflow/main.py:for gpu in tf.config.experimental.list_physical_devices("GPU"):
images/lfads-tensorflow/main.py:    tf.config.experimental.set_memory_growth(gpu, True)
README.md:**Prerequisites:** Container runtime (e.g. Docker - [Linux / Mac / Windows](https://docs.docker.com/get-docker/), Podman - [Linux / Mac / Windows](https://github.com/containers/podman/releases), containerD - [Linux / Windows](https://github.com/containerd/containerd/releases)) and the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) (GPU only).
README.md:1. Specify `latest` for CPU operation and `latest-gpu` for GPU compatible operation
README.md:    #   --gpus specifies which gpus to provide to the container
README.md:    # For GPU (Note: $TAG value should have a `-gpu` suffix`)
README.md:    docker run --rm --runtime=nvidia --gpus='"device=0"' -it -v $(pwd):/share ucsdtnel/autolfads:$TAG \
README.md:1. Install GPU specific packages
README.md:    conda install -c conda-forge cudatoolkit=10.0
paper/paper.md:When training models on a novel dataset, it is often helpful to probe hyperparameters and investigate model performance locally prior to conducting a more exhaustive, automated hyperparameter search. This need can be met by installing the LFADS package locally or in a virtual environment. Isolating the workflow from local computational environments, we provide a pair of reference container images targeting CPU and GPU architectures. This allows users to treat the bundled algorithm as a portable executable for which they simply provide the input neural data and desired LFADS model configuration to initiate model training. This approach eliminates the need for users to configure their environments with compatible interpreters and dependencies. Instead, the user installs a container runtime engine (e.g., Docker, Podman), which are generally well-supported cross-platform tools, to run the image based solution. In addition to streamlining configuration, this approach enables reproducibility as the software environment employed for computation is fully defined and version controlled.
examples/lorenz/container_run.sh:nvidia-smi > /dev/null 2>&1;
examples/lorenz/container_run.sh:	docker run --rm --runtime=nvidia --gpus='"device=0"' -it -v $(pwd):/share ucsdtnel/autolfads:${VERSION}-gpu \
examples/lorenz/ray_run.py:RESOURCES_PER_TRIAL = {"cpu": 2, "gpu": 0.5}
examples/lorenz/kubeflow_job.yaml:                # image: ucsdtnel/autolfads:latest-gpu
examples/lorenz/kubeflow_job.yaml:                    # nvidia.com/gpu: 1
examples/mc_maze/container_run.sh:nvidia-smi > /dev/null 2>&1;
examples/mc_maze/container_run.sh:	docker run --rm --runtime=nvidia --gpus='"device=0"' -it -v $(pwd):/share ucsdtnel/autolfads:${VERSION}-gpu \
examples/mc_maze/ray_run.py:RESOURCES_PER_TRIAL = {"cpu": 2, "gpu": 0.5}
examples/mc_maze/kubeflow_job.yaml:                # image: ucsdtnel/autolfads:latest-gpu
examples/mc_maze/kubeflow_job.yaml:                    # nvidia.com/gpu: 1
kubeflow/roles/nfs/files/deployment.yml:      # Allow scheduling on gpu nodes
kubeflow/roles/nfs/files/deployment.yml:        - key: nvidia.com/gpu
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/sample/README.md:- **ML Usage** GPU normally is required for deep learning task.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/sample/README.md:You may consider create **zero-sized GPU node-pool with autoscaling**.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/sample/README.md:Please reference [GPU Tutorial](/samples/tutorials/gpu/).
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pipeline/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/notebook-controller/upstream/crd/bases/kubeflow.org_notebooks.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/notebook-controller/upstream/crd/bases/kubeflow.org_notebooks.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/notebook-controller/upstream/crd/bases/kubeflow.org_notebooks.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/notebook-controller/upstream/crd/bases/kubeflow.org_notebooks.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/notebook-controller/upstream/crd/bases/kubeflow.org_notebooks.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/notebook-controller/upstream/crd/bases/kubeflow.org_notebooks.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/notebook-controller/upstream/crd/bases/kubeflow.org_notebooks.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/notebook-controller/upstream/crd/bases/kubeflow.org_notebooks.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/notebook-controller/upstream/crd/bases/kubeflow.org_notebooks.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/jupyter-web-app/upstream/base/configs/spawner_ui_config.yaml:    - kubeflownotebookswg/jupyter-pytorch-cuda-full:v1.8.0
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/jupyter-web-app/upstream/base/configs/spawner_ui_config.yaml:    - kubeflownotebookswg/jupyter-tensorflow-cuda-full:v1.8.0
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/jupyter-web-app/upstream/base/configs/spawner_ui_config.yaml:  # GPU/Device-Plugin Resources
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/jupyter-web-app/upstream/base/configs/spawner_ui_config.yaml:  gpus:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/jupyter-web-app/upstream/base/configs/spawner_ui_config.yaml:    # configs for gpu/device-plugin limits of the container
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/jupyter-web-app/upstream/base/configs/spawner_ui_config.yaml:    # https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/#using-device-plugins
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/jupyter-web-app/upstream/base/configs/spawner_ui_config.yaml:      - limitsKey: "nvidia.com/gpu"
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/jupyter-web-app/upstream/base/configs/spawner_ui_config.yaml:        uiName: "NVIDIA"
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/jupyter/jupyter-web-app/upstream/base/configs/spawner_ui_config.yaml:      - limitsKey: "amd.com/gpu"
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pvcviewer-controller/upstream/crd/bases/kubeflow.org_pvcviewers.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pvcviewer-controller/upstream/crd/bases/kubeflow.org_pvcviewers.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/pvcviewer-controller/upstream/crd/bases/kubeflow.org_pvcviewers.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mxjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_paddlejobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_tfjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_mpijobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                      cpu, gpu, int]. Deprecated: This API is deprecated in v1.7+
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                  cpu, gpu, int]. For more, https://github.com/pytorch/pytorch/blob/26f7f470df64d90e092081e39507e4ac751f55d6/torch/distributed/run.py#L629-L658.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_pytorchjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                          default is DefaultProcMount which uses the
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/training-operator/upstream/base/crds/kubeflow.org_xgboostjobs.yaml:                                          ProcMountType feature flag to be enabled.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/configmap/inferenceservice.yaml:            "defaultGpuImageVersion": "1.14.0-gpu",
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/configmap/inferenceservice.yaml:            "defaultGpuImageVersion": "v0.6.1-gpu",
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/configmap/inferenceservice.yaml:            "defaultGpuImageVersion": "0.4.0-gpu",
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/configmap/inferenceservice.yaml:            "image": "nvcr.io/nvidia/tritonserver",
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/crd/serving.kubeflow.org_inferenceservices.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/overlays/test/configmap/inferenceservice.yaml:            "defaultGpuImageVersion": "1.14.0-gpu",
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/overlays/test/configmap/inferenceservice.yaml:            "defaultGpuImageVersion": "v0.5.0-rc0-gpu",
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/overlays/test/configmap/inferenceservice.yaml:            "defaultGpuImageVersion": "0.3.0-gpu",
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfserving/upstream/overlays/test/configmap/inferenceservice.yaml:            "image": "nvcr.io/nvidia/tritonserver",
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/admission-webhook/upstream/base/crd.yaml:                        procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/admission-webhook/upstream/base/crd.yaml:                        procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/sample/README.md:- **ML Usage** GPU normally is required for deep learning task.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/sample/README.md:You may consider create **zero-sized GPU node-pool with autoscaling**.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/sample/README.md:Please reference [GPU Tutorial](/samples/tutorials/gpu/).
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/sample/README.md:- **ML Usage** GPU normally is required for deep learning task.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/sample/README.md:You may consider create **zero-sized GPU node-pool with autoscaling**.
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/sample/README.md:Please reference [GPU Tutorial](/samples/tutorials/gpu/).
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/v1/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_cronworkflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflowtasksets.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_clusterworkflowtemplates.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/apps/kfp-tekton/upstream/third-party/argo/upstream/manifests/base/crds/full/argoproj.io_workflows.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                        description: procMount denotes the type of proc mount to use
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                      description: procMount denotes the type of proc
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                      description: procMount denotes the type of proc
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                    procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                      description: procMount denotes the type of proc
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                        description: procMount denotes the type of
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                            description: procMount denotes the type of proc mount
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                        procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                        procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                        procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                            description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                            description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                            description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                            description: procMount denotes the type of proc mount
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                        procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                        procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                        procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                            description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                            description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                                            description: procMount denotes the type
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/ray/kuberay-operator/base/resources.yaml:                            numGpus:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                  gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                      gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                      gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                  gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                      gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                      gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                        gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                            gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                            gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                  gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                      gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                      gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                        gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                            gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                            gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                              description: procMount denotes the type of proc mount
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                                to use for the containers. The default is DefaultProcMount
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                                paths and masked paths. This requires the ProcMountType
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                      gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                      gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                                  procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                                    description: procMount denotes the type of proc
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                                      is DefaultProcMount which uses the container
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                                      paths. This requires the ProcMountType feature
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                            gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/bentoml/bentoml-yatai-stack/bases/yatai-deployment/resources.yaml:                            gpu:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/README.md:It encapsulates the complexity of autoscaling, networking, health checking, and server configuration to bring cutting edge serving features like GPU Autoscaling, Scale to Zero, and Canary Rollouts to your ML deployments. It enables a simple, pluggable, and complete story for Production ML Serving including prediction, pre-processing, post-processing and explainability. KServe is being [used across various organizations.](https://kserve.github.io/website/master/community/adopters/)
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/README.md:- Support modern **serverless inference workload** with **request based autoscaling including scale-to-zero** on **CPU and GPU**.
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                        procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                              procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve_kubeflow.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve-runtimes.yaml:    image: nvcr.io/nvidia/tritonserver:23.05-py3
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                        procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                      procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                          procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                            procMount:
kubeflow/roles/kubeflow/files/kubeflow/manifests/contrib/kserve/kserve/kserve.yaml:                        procMount:

```
