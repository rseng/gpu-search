# https://github.com/ci-for-research/self-hosted-runners

```console
windows-surf-hpc-cloud/README.md:[I](https://github.com/sverhoeven) used the [https://github.com/ci-for-science/example-gpu-houston](https://github.com/ci-for-science/example-gpu-houston) repo and am already started VM in the SURF HPC cloud to run a self hosted GitHub action runner. Through out this run-through I will use my account `sverhoeven` and `example-gpu-houston` as repo name, please replace with your account/repo for your own run-through. Below are screenshots of the run-through.
windows-surf-hpc-cloud/README.md:I [duplicated](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/duplicating-a-repository) the [https://github.com/ci-for-science/example-gpu-houston](https://github.com/ci-for-science/example-gpu-houston) repo to my own account and made it private.
windows-surf-hpc-cloud/README.md:Now I made a change (commit+push) to the repo ([https://github.com/sverhoeven/example-gpu-houston](https://github.com/sverhoeven/example-gpu-houston)).
windows-surf-hpc-cloud/README.md:Check in [https://github.com/sverhoeven/example-gpu-houston/settings/actions](https://github.com/sverhoeven/example-gpu-houston/settings/actions) (replace with your account/repo) for runner being active and it is.
.zenodo.json:      "GPU",
install-cuda/README.md:This short guide shows how to install Nvidia drivers and CUDA for GRID K2 hardware. For more information about GPUs on SURF HPC Cloud please visit [SURF HPC Documentation](https://doc.hpccloud.surfsara.nl/gpu-attach).
install-cuda/README.md:We have 2 methods to install Nvidia drivers and CUDA.
install-cuda/README.md:docker run --rm -ti -v $PWD:/data --workdir=/data ansible/ansible-runner ansible-playbook playbook-install-cuda-gridk2.yml
install-cuda/README.md:CUDA currently officially supports only two versions of Ubuntu: 18.04 and 16.04. This instructions were tested on Ubuntu 18.04.
install-cuda/README.md:GPU hardware information
install-cuda/README.md:lspci | grep -i nvidia
install-cuda/README.md:01:01.0 VGA compatible controller: NVIDIA Corporation GK104GL [GRID K2] (rev a1)
install-cuda/README.md:For Grid K2 card we will need CUDA 8.0. CUDA 8.0 only works with only gcc 5.0 so it should be installed before.
install-cuda/README.md:To decide what version of CUDA and Nvidia drivers you need, please check the links below.
install-cuda/README.md:[https://www.nvidia.com/Download/index.aspx?lang=en-us](https://www.nvidia.com/Download/index.aspx?lang=en-us)
install-cuda/README.md:[https://docs.nvidia.com/deploy/cuda-compatibility/index.html](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
install-cuda/README.md:### Install Nvidia drivers
install-cuda/README.md:Download Nvidia driver (version 367) and install it.
install-cuda/README.md:wget http://us.download.nvidia.com/XFree86/Linux-x86_64/367.134/NVIDIA-Linux-x86_64-367.134.run
install-cuda/README.md:sh ./NVIDIA-Linux-x86_64-367.134.run --accept-license  -s
install-cuda/README.md:### Install CUDA
install-cuda/README.md:Download CUDA 8.0 installer
install-cuda/README.md:# wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
install-cuda/README.md:wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda_8.0.61.2_linux-run
install-cuda/README.md:While installing CUDA, we had some issues related to Perl scripts.
install-cuda/README.md:See: [https://forums.developer.nvidia.com/t/cant-locate-installutils-pm-in-inc/46952/10](https://forums.developer.nvidia.com/t/cant-locate-installutils-pm-in-inc/46952/10)
install-cuda/README.md:sh ./cuda_8.0.61_375.26_linux-run  --tar mxvf
install-cuda/README.md:rm -rf InstallUtils.pm cuda-installer.pl run_files uninstall_cuda.pl
install-cuda/README.md:After fixing the Perl issue, we can install CUDA.
install-cuda/README.md:sh ./cuda_8.0.61_375.26_linux-run --silent --samples --toolkit --override --verbose
install-cuda/README.md:In order to be able to use CUDA, we need to change our environment variables.
install-cuda/README.md:export PATH=$PATH:/usr/local/cuda-8.0/bin
install-cuda/README.md:export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
install-cuda/README.md:### CUDA compiler
install-cuda/README.md:Check the CUDA compiler version.
install-cuda/README.md:nvcc: NVIDIA (R) Cuda compiler driver
install-cuda/README.md:Copyright (c) 2005-2017 NVIDIA Corporation
install-cuda/README.md:Cuda compilation tools, release 9.1, V9.1.85
install-cuda/README.md:See [gpu-houston](https://github.com/ci-for-research/example-gpu-houston) for a simple example code.
install-cuda/playbook-install-cuda-gridk2.yml:- name: Install Nvidia drivers and CUDA for Grid K2
install-cuda/playbook-install-cuda-gridk2.yml:    cuda_version_major: "8.0"
install-cuda/playbook-install-cuda-gridk2.yml:    cuda_version_minor: "61_375.26"
install-cuda/playbook-install-cuda-gridk2.yml:    nvidia_url: http://us.download.nvidia.com/XFree86/Linux-x86_64
install-cuda/playbook-install-cuda-gridk2.yml:    cuda_url: https://developer.nvidia.com/compute/cuda
install-cuda/playbook-install-cuda-gridk2.yml:  - name: Download Nvidia driver
install-cuda/playbook-install-cuda-gridk2.yml:      url: "{{ nvidia_url }}/{{ driver_version }}/NVIDIA-Linux-x86_64-{{ driver_version }}.run"
install-cuda/playbook-install-cuda-gridk2.yml:      dest: /tmp/installers/NVIDIA-Linux-x86_64-{{ driver_version }}.run
install-cuda/playbook-install-cuda-gridk2.yml:      - cuda
install-cuda/playbook-install-cuda-gridk2.yml:  - name: Install Nvidia driver
install-cuda/playbook-install-cuda-gridk2.yml:      sh ./NVIDIA-Linux-x86_64-367.134.run
install-cuda/playbook-install-cuda-gridk2.yml:  - name: Download CUDA
install-cuda/playbook-install-cuda-gridk2.yml:      url: "{{ cuda_url }}/{{ cuda_version_major }}/Prod2/local_installers/cuda_{{ cuda_version_major }}.{{ cuda_version_minor }}_linux-run"
install-cuda/playbook-install-cuda-gridk2.yml:      dest: /tmp/installers/cuda_{{ cuda_version_major }}.{{ cuda_version_minor }}_linux-run
install-cuda/playbook-install-cuda-gridk2.yml:      - cuda
install-cuda/playbook-install-cuda-gridk2.yml:    command: sh ./cuda_8.0.61_375.26_linux-run  --tar mxvf
install-cuda/playbook-install-cuda-gridk2.yml:    command: rm -rf InstallUtils.pm cuda-installer.pl run_files uninstall_cuda.pl
install-cuda/playbook-install-cuda-gridk2.yml:  - name: Install CUDA
install-cuda/playbook-install-cuda-gridk2.yml:    command: sh ./cuda_8.0.61_375.26_linux-run --silent --samples --toolkit --override --verbose
install-cuda/playbook-install-cuda-gridk2.yml:  - name: Make gcc-5 default compiler for CUDA
install-cuda/playbook-install-cuda-gridk2.yml:    command: ln -s /usr/bin/"{{ item.name }}" /usr/local/cuda/bin/"{{ item.dest }}"
install-cuda/playbook-install-cuda-gridk2.yml:      dest: "/etc/profile.d/cuda.sh"
install-cuda/playbook-install-cuda-gridk2.yml:        export PATH=$PATH:/usr/local/cuda-{{ cuda_version_major }}/bin
install-cuda/playbook-install-cuda-gridk2.yml:        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-{{ cuda_version_major }}/lib64
install-cuda/playbook-install-cuda-gridk2.yml:      marker: '# {mark} CUDA environment variables'
ubuntu-surf-hpc-cloud/README.md:- [Simple GPU example](https://github.com/ci-for-science/example-gpu-houston)
CITATION.cff:  - "GPU"
README.md:- it needs a GPU to run ([CUDA installation with ansible instructions](/install-cuda/README.md))
.mlc-config.json:            "pattern": "^https://github.com/sverhoeven/example-gpu-houston"
ubuntu-singularity/README.md:## GPU support
ubuntu-singularity/README.md:See [Singularity documentation](https://sylabs.io/guides/3.5/user-guide/gpu.html)

```
