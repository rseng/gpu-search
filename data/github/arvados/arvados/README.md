# https://github.com/arvados/arvados

```console
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:- name: cwltool:CUDARequirement
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:    Require support for NVIDA CUDA (GPU hardware acceleration).
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:      doc: 'cwltool:CUDARequirement'
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:    cudaVersionMin:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        Minimum CUDA version to run the software, in X.Y format.  This
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        corresponds to a CUDA SDK release.  When running directly on
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        CUDA SDK (matching the exact version, or, starting with CUDA
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        container image should provide the CUDA runtime, and the host
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        CUDA drivers are backwards compatible, it is possible to
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        See https://docs.nvidia.com/deploy/cuda-compatibility/ for
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:    cudaComputeCapability:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        CUDA hardware capability required to run the software, in X.Y
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:          compute capability.  GPUs with higher capability are also
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        * If it is an array value, then only select GPUs with compute
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:    cudaDeviceCountMin:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        Minimum number of GPU devices to request.  If not specified,
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        same as `cudaDeviceCountMax`.  If neither are specified,
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:    cudaDeviceCountMax:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        Maximum number of GPU devices to request.  If not specified,
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.2.yml:        same as `cudaDeviceCountMin`.
sdk/cwl/arvados_cwl/arvcontainer.py:        cuda_req, _ = self.get_requirement("http://commonwl.org/cwltool#CUDARequirement")
sdk/cwl/arvados_cwl/arvcontainer.py:        if cuda_req:
sdk/cwl/arvados_cwl/arvcontainer.py:            runtime_constraints["cuda"] = {
sdk/cwl/arvados_cwl/arvcontainer.py:                "device_count": resources.get("cudaDeviceCount", 1),
sdk/cwl/arvados_cwl/arvcontainer.py:                "driver_version": cuda_req["cudaVersionMin"],
sdk/cwl/arvados_cwl/arvcontainer.py:                "hardware_capability": aslist(cuda_req["cudaComputeCapability"])[0]
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:- name: cwltool:CUDARequirement
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:    Require support for NVIDA CUDA (GPU hardware acceleration).
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:      doc: 'cwltool:CUDARequirement'
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:    cudaVersionMin:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        Minimum CUDA version to run the software, in X.Y format.  This
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        corresponds to a CUDA SDK release.  When running directly on
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        CUDA SDK (matching the exact version, or, starting with CUDA
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        container image should provide the CUDA runtime, and the host
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        CUDA drivers are backwards compatible, it is possible to
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        See https://docs.nvidia.com/deploy/cuda-compatibility/ for
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:    cudaComputeCapability:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        CUDA hardware capability required to run the software, in X.Y
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:          compute capability.  GPUs with higher capability are also
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        * If it is an array value, then only select GPUs with compute
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:    cudaDeviceCountMin:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        Minimum number of GPU devices to request.  If not specified,
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        same as `cudaDeviceCountMax`.  If neither are specified,
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:    cudaDeviceCountMax:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        Maximum number of GPU devices to request.  If not specified,
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.0.yml:        same as `cudaDeviceCountMin`.
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:- name: cwltool:CUDARequirement
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:    Require support for NVIDA CUDA (GPU hardware acceleration).
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:      doc: 'cwltool:CUDARequirement'
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:    cudaVersionMin:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        Minimum CUDA version to run the software, in X.Y format.  This
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        corresponds to a CUDA SDK release.  When running directly on
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        CUDA SDK (matching the exact version, or, starting with CUDA
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        container image should provide the CUDA runtime, and the host
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        CUDA drivers are backwards compatible, it is possible to
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        See https://docs.nvidia.com/deploy/cuda-compatibility/ for
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:    cudaComputeCapability:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        CUDA hardware capability required to run the software, in X.Y
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:          compute capability.  GPUs with higher capability are also
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        * If it is an array value, then only select GPUs with compute
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:    cudaDeviceCountMin:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        Minimum number of GPU devices to request.  If not specified,
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        same as `cudaDeviceCountMax`.  If neither are specified,
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:    cudaDeviceCountMax:
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        Maximum number of GPU devices to request.  If not specified,
sdk/cwl/arvados_cwl/arv-cwl-schema-v1.1.yml:        same as `cudaDeviceCountMin`.
sdk/cwl/arvados_cwl/__init__.py:        "http://commonwl.org/cwltool#CUDARequirement",
sdk/cwl/tests/test_container.py:    "CUDA": {
sdk/cwl/tests/test_container.py:    def test_cuda_requirement(self, keepdocker):
sdk/cwl/tests/test_container.py:                "class": "http://commonwl.org/cwltool#CUDARequirement",
sdk/cwl/tests/test_container.py:                "cudaVersionMin": "11.0",
sdk/cwl/tests/test_container.py:                "cudaComputeCapability": "9.0",
sdk/cwl/tests/test_container.py:                "class": "http://commonwl.org/cwltool#CUDARequirement",
sdk/cwl/tests/test_container.py:                "cudaVersionMin": "11.0",
sdk/cwl/tests/test_container.py:                "cudaComputeCapability": "9.0",
sdk/cwl/tests/test_container.py:                "cudaDeviceCountMin": 2
sdk/cwl/tests/test_container.py:                "class": "http://commonwl.org/cwltool#CUDARequirement",
sdk/cwl/tests/test_container.py:                "cudaVersionMin": "11.0",
sdk/cwl/tests/test_container.py:                "cudaComputeCapability": ["4.0", "5.0"],
sdk/cwl/tests/test_container.py:                "cudaDeviceCountMin": 2
sdk/cwl/tests/test_container.py:                "baseCommand": "nvidia-smi",
sdk/cwl/tests/test_container.py:                            'cuda': test_arv_req[test_case]
sdk/cwl/tests/test_container.py:                        'command': ['nvidia-smi'],
sdk/ruby-google-api-client/lib/cacerts.pem:V2ccHsBqBt5ZtJot39wZhi4wNgYDVR0fBC8wLTAroCmgJ4YlaHR0cDovL2NybC5n
sdk/ruby-google-api-client/lib/cacerts.pem:IR9NmXmd4c8nnxCbHIgNsIpkQTG4DmyQJKSbXHGPurt+HBvbaoAPIbzp26a3QPSy
sdk/go/arvados/container.go:type CUDARuntimeConstraints struct {
sdk/go/arvados/container.go:	CUDA          CUDARuntimeConstraints `json:"cuda"`
sdk/go/arvados/config.go:type CUDAFeatures struct {
sdk/go/arvados/config.go:	CUDA            CUDAFeatures
sdk/go/arvados/config.go:		BsubCUDAArguments  []string
sdk/go/arvados/keep_cache_test.go:// BenchmarkOpenClose and BenchmarkKeepOpen can be used to measure the
sdk/go/arvados/keep_cache_test.go:func (s *fileOpsSuite) BenchmarkOpenClose(c *check.C) {
doc/user/cwl/cwl-style.html.textile.liquid:h3. Does your application support NVIDIA GPU acceleration?
doc/user/cwl/cwl-style.html.textile.liquid:Use "cwltool:CUDARequirement":cwl-extensions.html#CUDARequirement to request nodes with GPUs.
doc/user/cwl/cwl-run-options.html.textile.liquid:h3(#gpu). Use CUDA GPU instances
doc/user/cwl/cwl-run-options.html.textile.liquid:See "cwltool:CUDARequirement":cwl-extensions.html#CUDARequirement .
doc/user/cwl/cwl-extensions.html.textile.liquid:* "CUDARequirement":#CUDARequirement
doc/user/cwl/cwl-extensions.html.textile.liquid:  cwltool:CUDARequirement:
doc/user/cwl/cwl-extensions.html.textile.liquid:    cudaVersionMin: "11.0"
doc/user/cwl/cwl-extensions.html.textile.liquid:    cudaComputeCapability: "9.0"
doc/user/cwl/cwl-extensions.html.textile.liquid:    cudaDeviceCountMin: 1
doc/user/cwl/cwl-extensions.html.textile.liquid:    cudaDeviceCountMax: 1
doc/user/cwl/cwl-extensions.html.textile.liquid:h2(#CUDARequirement). cwltool:CUDARequirement
doc/user/cwl/cwl-extensions.html.textile.liquid:Request support for Nvidia CUDA GPU acceleration in the container.  Assumes that the CUDA runtime (SDK) is installed in the container, and the host will inject the CUDA driver libraries into the container (equal or later to the version requested).
doc/user/cwl/cwl-extensions.html.textile.liquid:|cudaVersionMin|string|Required.  The CUDA SDK version corresponding to the minimum driver version supported by the container (generally, the SDK version 'X.Y' the application was compiled against).|
doc/user/cwl/cwl-extensions.html.textile.liquid:|cudaComputeCapability|string|Required.  The minimum CUDA hardware capability (in 'X.Y' format) required by the application's PTX or C++ GPU code (will be JIT compiled for the available hardware).|
doc/user/cwl/cwl-extensions.html.textile.liquid:|cudaDeviceCountMin|integer|Minimum number of GPU devices to allocate on a single node. Required.|
doc/user/cwl/cwl-extensions.html.textile.liquid:|cudaDeviceCountMax|integer|Maximum number of GPU devices to allocate on a single node. Optional.  If not specified, same as @cudaDeviceCountMin@.|
doc/install/crunch2-lsf/install-dispatch.html.textile.liquid:%G number of GPU devices (@runtime_constraints.cuda.device_count@)
doc/install/crunch2-lsf/install-dispatch.html.textile.liquid:h3(#BsubCUDAArguments). Containers.LSF.BsubCUDAArguments
doc/install/crunch2-lsf/install-dispatch.html.textile.liquid:If the container requests access to GPUs (@runtime_constraints.cuda.device_count@ of the container request is greater than zero), the command line arguments in @BsubCUDAArguments@ will be added to the command line _after_ @BsubArgumentsList@.  This should consist of the additional @bsub@ flags your site requires to schedule the job on a node with GPU support.  Set @BsubCUDAArguments@ to an array of strings.  For example:
doc/install/crunch2-lsf/install-dispatch.html.textile.liquid:        <code class="userinput">BsubCUDAArguments: <b>["-gpu", "num=%G"]</b></code>
doc/install/crunch2-lsf/install-dispatch.html.textile.liquid:      gpu:
doc/install/crunch2-lsf/install-dispatch.html.textile.liquid:        CUDA:
doc/install/crunch2-cloud/install-dispatch-cloud.html.textile.liquid:h3(#GPUsupport). NVIDIA GPU support
doc/install/crunch2-cloud/install-dispatch-cloud.html.textile.liquid:To specify instance types with NVIDIA GPUs, "the compute image must be built with CUDA support":install-compute-node.html#nvidia , and you must include an additional @CUDA@ section:
doc/install/crunch2-cloud/install-dispatch-cloud.html.textile.liquid:        CUDA:
doc/install/crunch2-cloud/install-dispatch-cloud.html.textile.liquid:The @DriverVersion@ is the version of the CUDA toolkit installed in your compute image (in X.Y format, do not include the patchlevel).  The @HardwareCapability@ is the "CUDA compute capability of the GPUs available for this instance type":https://developer.nvidia.com/cuda-gpus.  The @DeviceCount@ is the number of GPU cores available for this instance type.
doc/install/crunch2-cloud/install-compute-node.html.textile.liquid:## "NVIDIA GPU support":#nvidia
doc/install/crunch2-cloud/install-compute-node.html.textile.liquid:  --nvidia-gpu-support
doc/install/crunch2-cloud/install-compute-node.html.textile.liquid:      Install all the necessary tooling for Nvidia GPU support (default: do not install Nvidia GPU support)
doc/install/crunch2-cloud/install-compute-node.html.textile.liquid:h3(#nvidia). NVIDIA GPU support
doc/install/crunch2-cloud/install-compute-node.html.textile.liquid:If you plan on using instance types with NVIDIA GPUs, add @--nvidia-gpu-support@ to the build command line.  Arvados uses the same compute image for both GPU and non-GPU instance types.  The GPU tooling is ignored when using the image with a non-GPU instance type.
doc/install/crunch2/install-compute-node-docker.html.textile.liquid:{% include 'install_cuda' %}
doc/install/crunch2/install-compute-node-singularity.html.textile.liquid:{% include 'install_cuda' %}
doc/_includes/_config_default_yml.liquid:        # %G number of GPU devices (runtime_constraints.cuda.device_count)
doc/_includes/_config_default_yml.liquid:        # runtime_constraints.cuda.device_count > 0
doc/_includes/_config_default_yml.liquid:        BsubCUDAArguments: ["-gpu", "num=%G"]
doc/_includes/_config_default_yml.liquid:        # Include this section if the node type includes GPU (CUDA) support
doc/_includes/_config_default_yml.liquid:        CUDA:
doc/_includes/_container_runtime_constraints.liquid:|cuda|object|Request CUDA GPU support, see below|Optional.|
doc/_includes/_container_runtime_constraints.liquid:h3. CUDA GPU support
doc/_includes/_container_runtime_constraints.liquid:|device_count|int|Number of GPUs to request.|Count greater than 0 enables CUDA GPU support.|
doc/_includes/_container_runtime_constraints.liquid:|driver_version|string|Minimum CUDA driver version, in "X.Y" format.|Required when device_count > 0|
doc/_includes/_container_runtime_constraints.liquid:|hardware_capability|string|Minimum CUDA hardware capability, in "X.Y" format.|Required when device_count > 0|
doc/_includes/_install_cuda.liquid:h2(#cuda). Install NVIDA CUDA Toolkit (optional)
doc/_includes/_install_cuda.liquid:If you want to use NVIDIA GPUs, "install the CUDA toolkit":https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html and the "NVIDIA Container Toolkit":https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.
doc/admin/upgrading.html.textile.liquid:h3. Support for NVIDIA CUDA GPUs
doc/admin/upgrading.html.textile.liquid:Arvados now supports requesting NVIDIA CUDA GPUs for cloud and LSF (Slurm is currently not supported).  To be able to request GPU nodes, some additional configuration is needed:
doc/admin/upgrading.html.textile.liquid:"Including GPU support in cloud compute node image":{{site.baseurl}}/install/crunch2-cloud/install-compute-node.html#nvidia
doc/admin/upgrading.html.textile.liquid:"Configure cloud dispatcher for GPU support":{{site.baseurl}}/install/crunch2-cloud/install-dispatch-cloud.html#GPUsupport
doc/admin/upgrading.html.textile.liquid:"LSF GPU configuration":{{site.baseurl}}/install/crunch2-lsf/install-dispatch.html
tools/compute-images/arvados-images-azure.json:    "nvidia_gpu_support": "",
tools/compute-images/arvados-images-azure.json:    "source": "scripts/etc-systemd-system-systemd-modules-load.service.d-detect-gpu.conf",
tools/compute-images/arvados-images-azure.json:    "destination": "{{user `workdir`}}/etc-systemd-system-systemd-modules-load.service.d-detect-gpu.conf"
tools/compute-images/arvados-images-azure.json:    "source": "scripts/usr-local-bin-detect-gpu.sh",
tools/compute-images/arvados-images-azure.json:    "destination": "{{user `workdir`}}/usr-local-bin-detect-gpu.sh"
tools/compute-images/arvados-images-azure.json:        "NVIDIA_GPU_SUPPORT={{user `nvidia_gpu_support`}}",
tools/compute-images/scripts/base.sh:if [ "$NVIDIA_GPU_SUPPORT" == "1" ]; then
tools/compute-images/scripts/base.sh:  # Install CUDA
tools/compute-images/scripts/base.sh:  NVIDIA_URL="https://developer.download.nvidia.com/compute/cuda/repos/$(echo "$DISTRO_ID$VERSION_ID" | tr -d .)/x86_64"
tools/compute-images/scripts/base.sh:  $SUDO apt-key adv --fetch-keys "$NVIDIA_URL/7fa2af80.pub"
tools/compute-images/scripts/base.sh:  $SUDO apt-key adv --fetch-keys "$NVIDIA_URL/3bf863cc.pub"
tools/compute-images/scripts/base.sh:  $SUDO add-apt-repository "deb $NVIDIA_URL/ /"
tools/compute-images/scripts/base.sh:  $SUDO apt-get -y install cuda
tools/compute-images/scripts/base.sh:  # Install libnvidia-container, the tooling for Docker/Singularity
tools/compute-images/scripts/base.sh:  curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
tools/compute-images/scripts/base.sh:  curl -fsSL "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" |
tools/compute-images/scripts/base.sh:    $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
tools/compute-images/scripts/base.sh:  $SUDO apt-get -y install libnvidia-container1 libnvidia-container-tools nvidia-container-toolkit
tools/compute-images/scripts/base.sh:  # state, if the system does not actually have an NVIDIA GPU. Configure the
tools/compute-images/scripts/base.sh:  if [[ -f /etc/modules-load.d/nvidia.conf ]]; then
tools/compute-images/scripts/base.sh:      $SUDO mv /etc/modules-load.d/nvidia.conf /etc/modules-load.d/nvidia.avail
tools/compute-images/scripts/base.sh:  $SUDO install "$WORKDIR/usr-local-bin-detect-gpu.sh" /usr/local/bin/detect-gpu.sh
tools/compute-images/scripts/base.sh:        "$WORKDIR/etc-systemd-system-systemd-modules-load.service.d-detect-gpu.conf" \
tools/compute-images/scripts/base.sh:        /etc/systemd/system/systemd-modules-load.service.d/detect-gpu.conf
tools/compute-images/scripts/base.sh:  # Instead rely on crunch-run's CUDA initialization.
tools/compute-images/scripts/base.sh:  $SUDO systemctl disable nvidia-persistenced.service
tools/compute-images/scripts/etc-apt-preferences.d-arvados.pref:Package: src:libnvidia-container
tools/compute-images/scripts/etc-apt-preferences.d-arvados.pref:Package: src:nvidia-graphics-drivers
tools/compute-images/scripts/etc-systemd-system-systemd-modules-load.service.d-detect-gpu.conf:ExecStartPre=/usr/local/bin/detect-gpu.sh enable
tools/compute-images/scripts/usr-local-bin-detect-gpu.sh:# Enumerate GPU devices on the host and output a standard "driver" name for
tools/compute-images/scripts/usr-local-bin-detect-gpu.sh:# each one found. Currently only detects `nvidia`.
tools/compute-images/scripts/usr-local-bin-detect-gpu.sh:detect_gpus() {
tools/compute-images/scripts/usr-local-bin-detect-gpu.sh:(tolower($2) ~ /^nvidia/) { print "nvidia"; }
tools/compute-images/scripts/usr-local-bin-detect-gpu.sh:        detect_gpus | while read driver; do
tools/compute-images/arvados-images-aws.json:    "nvidia_gpu_support": "",
tools/compute-images/arvados-images-aws.json:    "source": "scripts/etc-systemd-system-systemd-modules-load.service.d-detect-gpu.conf",
tools/compute-images/arvados-images-aws.json:    "destination": "{{user `workdir`}}/etc-systemd-system-systemd-modules-load.service.d-detect-gpu.conf"
tools/compute-images/arvados-images-aws.json:    "source": "scripts/usr-local-bin-detect-gpu.sh",
tools/compute-images/arvados-images-aws.json:    "destination": "{{user `workdir`}}/usr-local-bin-detect-gpu.sh"
tools/compute-images/arvados-images-aws.json:        "NVIDIA_GPU_SUPPORT={{user `nvidia_gpu_support`}}",
tools/compute-images/build.sh:  --nvidia-gpu-support
tools/compute-images/build.sh:      Install all the necessary tooling for Nvidia GPU support (default: do not install Nvidia GPU support)
tools/compute-images/build.sh:NVIDIA_GPU_SUPPORT=
tools/compute-images/build.sh:    help,json-file:,arvados-cluster-id:,aws-source-ami:,aws-profile:,aws-secrets-file:,aws-region:,aws-vpc-id:,aws-subnet-id:,aws-ebs-autoscale,aws-associate-public-ip:,aws-ena-support:,gcp-project-id:,gcp-account-file:,gcp-zone:,azure-secrets-file:,azure-resource-group:,azure-location:,azure-sku:,azure-cloud-environment:,ssh_user:,workdir:,resolver:,reposuffix:,pin-packages,no-pin-packages,public-key-file:,mksquashfs-mem:,nvidia-gpu-support,debug \
tools/compute-images/build.sh:        --nvidia-gpu-support)
tools/compute-images/build.sh:            NVIDIA_GPU_SUPPORT=1
tools/compute-images/build.sh:if [[ -n "$NVIDIA_GPU_SUPPORT" ]]; then
tools/compute-images/build.sh:  EXTRA2+=" -var nvidia_gpu_support=$NVIDIA_GPU_SUPPORT"
services/workbench2/cypress/e2e/process.cy.js:                cuda: {
services/workbench2/src/views-components/sharing-dialog/sharing-public-access-form-component.tsx:const sharingPublicAccessStyles: CustomStyleRulesCallback<'root'> = theme => ({
services/workbench2/src/views-components/sharing-dialog/sharing-public-access-form-component.tsx:const SharingPublicAccessForm = withStyles(sharingPublicAccessStyles)(
services/workbench2/src/views-components/sharing-dialog/sharing-public-access-form-component.tsx:const SharingPublicAccessFormComponent = ({ visibility, includePublic, onSave }: AccessProps) =>
services/workbench2/src/views-components/sharing-dialog/sharing-public-access-form-component.tsx:    <SharingPublicAccessForm {...{ visibility, includePublic, onSave }} />;
services/workbench2/src/views-components/sharing-dialog/sharing-public-access-form-component.tsx:export default SharingPublicAccessFormComponent;
services/workbench2/src/views-components/sharing-dialog/sharing-dialog.tsx:    getSharingPublicAccessFormData,
services/workbench2/src/views-components/sharing-dialog/sharing-dialog.tsx:        privateAccess: getSharingPublicAccessFormData(state)?.visibility === VisibilityLevel.PRIVATE,
services/workbench2/src/views-components/sharing-dialog/sharing-public-access-form.tsx:import SharingPublicAccessFormComponent from './sharing-public-access-form-component';
services/workbench2/src/views-components/sharing-dialog/sharing-public-access-form.tsx:import { getSharingPublicAccessFormData } from '../../store/sharing-dialog/sharing-dialog-types';
services/workbench2/src/views-components/sharing-dialog/sharing-public-access-form.tsx:export const SharingPublicAccessForm = compose(
services/workbench2/src/views-components/sharing-dialog/sharing-public-access-form.tsx:            const { visibility } = getSharingPublicAccessFormData(state) || { visibility: VisibilityLevel.PRIVATE };
services/workbench2/src/views-components/sharing-dialog/sharing-public-access-form.tsx:)(SharingPublicAccessFormComponent);
services/workbench2/src/views-components/sharing-dialog/sharing-dialog-component.tsx:import { SharingPublicAccessForm } from './sharing-public-access-form';
services/workbench2/src/views-components/sharing-dialog/sharing-dialog-component.tsx:                            <SharingPublicAccessForm onSave={onSave} />
services/workbench2/src/models/runtime-constraints.ts:export interface CUDAParameters {
services/workbench2/src/models/runtime-constraints.ts:    cuda?: CUDAParameters;
services/workbench2/src/common/config.ts:            CUDA?: {
services/workbench2/src/store/sharing-dialog/sharing-dialog-actions.ts:    SharingPublicAccessFormData,
services/workbench2/src/store/sharing-dialog/sharing-dialog-actions.ts:import { getSharingPublicAccessFormData } from './sharing-dialog-types';
services/workbench2/src/store/sharing-dialog/sharing-dialog-actions.ts:        let publicAccessFormData: SharingPublicAccessFormData;
services/workbench2/src/store/sharing-dialog/sharing-dialog-actions.ts:        const { permissionUuid, visibility, initialVisibility } = getSharingPublicAccessFormData(state);
services/workbench2/src/store/sharing-dialog/sharing-dialog-actions.ts:        const { visibility } = getSharingPublicAccessFormData(state);
services/workbench2/src/store/sharing-dialog/sharing-dialog-types.ts:export interface SharingPublicAccessFormData {
services/workbench2/src/store/sharing-dialog/sharing-dialog-types.ts:export const getSharingPublicAccessFormData = (state: any) =>
services/workbench2/src/store/sharing-dialog/sharing-dialog-types.ts:    getFormValues(SHARING_PUBLIC_ACCESS_FORM_NAME)(state) as SharingPublicAccessFormData;
services/workbench2/src/store/processes/process-copy-actions.cy.js:        cuda: { device_count: 0, driver_version: "", hardware_capability: "" },
services/workbench2/src/store/processes/process-copy-actions.cy.js:        cuda: {
services/workbench2/src/store/process-panel/process-panel.ts:export interface CUDAFeatures {
services/workbench2/src/store/process-panel/process-panel.ts:    CUDA: CUDAFeatures;
services/workbench2/src/views/instance-types-panel/instance-types-panel.cy.js:                    "gpuType" : {
services/workbench2/src/views/instance-types-panel/instance-types-panel.cy.js:                        ProviderType: "gpuProvider",
services/workbench2/src/views/instance-types-panel/instance-types-panel.cy.js:                        CUDA: {
services/workbench2/src/views/instance-types-panel/instance-types-panel.cy.js:            if (instanceType.CUDA && instanceType.CUDA.DeviceCount > 0) {
services/workbench2/src/views/instance-types-panel/instance-types-panel.cy.js:                cy.get('@item').contains(`CUDA GPUs${instanceType.CUDA.DeviceCount}`);
services/workbench2/src/views/instance-types-panel/instance-types-panel.cy.js:                cy.get('@item').contains(`Hardware capability${instanceType.CUDA.HardwareCapability}`);
services/workbench2/src/views/instance-types-panel/instance-types-panel.cy.js:                cy.get('@item').contains(`Driver version${instanceType.CUDA.DriverVersion}`);
services/workbench2/src/views/instance-types-panel/instance-types-panel.tsx:                                    {instanceType.CUDA && instanceType.CUDA.DeviceCount > 0 ?
services/workbench2/src/views/instance-types-panel/instance-types-panel.tsx:                                                <DetailsAttribute label="CUDA GPUs" value={instanceType.CUDA.DeviceCount} />
services/workbench2/src/views/instance-types-panel/instance-types-panel.tsx:                                                <DetailsAttribute label="Hardware capability" value={instanceType.CUDA.HardwareCapability} />
services/workbench2/src/views/instance-types-panel/instance-types-panel.tsx:                                                <DetailsAttribute label="Driver version" value={instanceType.CUDA.DriverVersion} />
services/workbench2/src/views/process-panel/process-resource-card.tsx:                                {process.container?.runtimeConstraints.cuda &&
services/workbench2/src/views/process-panel/process-resource-card.tsx:                                    process.container?.runtimeConstraints.cuda.device_count > 0 ?
services/workbench2/src/views/process-panel/process-resource-card.tsx:                                            <DetailsAttribute label="CUDA devices" value={process.container?.runtimeConstraints.cuda.device_count} />
services/workbench2/src/views/process-panel/process-resource-card.tsx:                                            <DetailsAttribute label="CUDA driver version" value={process.container?.runtimeConstraints.cuda.driver_version} />
services/workbench2/src/views/process-panel/process-resource-card.tsx:                                            <DetailsAttribute label="CUDA hardware capability" value={process.container?.runtimeConstraints.cuda.hardware_capability} />
services/workbench2/src/views/process-panel/process-resource-card.tsx:                                    {nodeInfo.CUDA && nodeInfo.CUDA.DeviceCount > 0 &&
services/workbench2/src/views/process-panel/process-resource-card.tsx:                                                <DetailsAttribute label="CUDA devices" value={nodeInfo.CUDA.DeviceCount} />
services/workbench2/src/views/process-panel/process-resource-card.tsx:                                                <DetailsAttribute label="CUDA driver version" value={nodeInfo.CUDA.DriverVersion} />
services/workbench2/src/views/process-panel/process-resource-card.tsx:                                                <DetailsAttribute label="CUDA hardware capability" value={nodeInfo.CUDA.HardwareCapability} />
services/api/app/models/container_request.rb:      if runtime_constraints['cuda']
services/api/app/models/container_request.rb:          v = runtime_constraints['cuda'][k]
services/api/app/models/container_request.rb:                       "[cuda.#{k}]=#{v.inspect} must be a positive or zero integer")
services/api/app/models/container_request.rb:          v = runtime_constraints['cuda'][k]
services/api/app/models/container_request.rb:          if !v.is_a?(String) || (runtime_constraints['cuda']['device_count'] > 0 && v.to_f == 0.0)
services/api/app/models/container_request.rb:                       "[cuda.#{k}]=#{v.inspect} must be a string in format 'X.Y'")
services/api/app/models/container.rb:    resolved_cuda = resolved_runtime_constraints['cuda']
services/api/app/models/container.rb:    if resolved_cuda.nil? or resolved_cuda['device_count'] == 0
services/api/app/models/container.rb:      runtime_constraint_variations[:cuda] = [
services/api/app/models/container.rb:        # Check for constraints without cuda
services/api/app/models/container.rb:        # The default "don't need CUDA" value
services/api/app/models/container.rb:        resolved_runtime_constraints.delete('cuda')
services/api/app/models/arvados_model.rb:      'cuda' => {
services/api/test/unit/container_test.rb:    runtime_constraints: {"vcpus" => 1, "ram" => 1, "cuda" => {"device_count":0, "driver_version": "", "hardware_capability": ""}},
services/api/test/unit/container_test.rb:    rc = {"vcpus" => 1, "ram" => 1, "keep_cache_ram" => 1, "keep_cache_disk" => 0, "API" => true, "cuda" => {"device_count":0, "driver_version": "", "hardware_capability": ""}}
services/api/test/unit/container_test.rb:  test "find_reusable method with cuda" do
services/api/test/unit/container_test.rb:    # No cuda
services/api/test/unit/container_test.rb:    no_cuda_attrs = REUSABLE_COMMON_ATTRS.merge({use_existing:false, priority:1, environment:{"var" => "queued"},
services/api/test/unit/container_test.rb:                                                                      "cuda" => {"device_count":0, "driver_version": "", "hardware_capability": ""}},})
services/api/test/unit/container_test.rb:    c1, _ = minimal_new(no_cuda_attrs)
services/api/test/unit/container_test.rb:    # has cuda
services/api/test/unit/container_test.rb:    cuda_attrs = REUSABLE_COMMON_ATTRS.merge({use_existing:false, priority:1, environment:{"var" => "queued"},
services/api/test/unit/container_test.rb:                                                                      "cuda" => {"device_count":1, "driver_version": "11.0", "hardware_capability": "9.0"}},})
services/api/test/unit/container_test.rb:    c2, _ = minimal_new(cuda_attrs)
services/api/test/unit/container_test.rb:    # should find the no cuda one
services/api/test/unit/container_test.rb:    reused = Container.find_reusable(no_cuda_attrs)
services/api/test/unit/container_test.rb:    # should find the cuda one
services/api/test/unit/container_test.rb:    reused = Container.find_reusable(cuda_attrs)
services/api/test/fixtures/containers.yml:    cuda:
services/api/test/fixtures/containers.yml:cuda_container:
services/api/test/fixtures/containers.yml:  uuid: zzzzz-dz642-cudagpcontainer
services/api/test/fixtures/containers.yml:    cuda:
services/api/test/fixtures/container_requests.yml:    cuda:
lib/crunchstat/crunchstat_test.go:		if m := regexp.MustCompile(`(?ms).*procmem \d+ init (\d+) test_process.*`).FindSubmatch(s.logbuf.Bytes()); len(m) > 0 {
lib/crunchstat/crunchstat.go:	procmounts, err := fs.ReadFile(r.FS, "proc/mounts")
lib/crunchstat/crunchstat.go:	for _, line := range bytes.Split(procmounts, []byte{'\n'}) {
lib/crunchstat/crunchstat.go:func (r *Reporter) doProcmemStats() {
lib/crunchstat/crunchstat.go:	procmem := ""
lib/crunchstat/crunchstat.go:		procmem += fmt.Sprintf(" %d %s", value, procname)
lib/crunchstat/crunchstat.go:	if procmem != "" {
lib/crunchstat/crunchstat.go:		r.Logger.Printf("procmem%s\n", procmem)
lib/crunchstat/crunchstat.go:	r.doProcmemStats()
lib/config/config.default.yml:        # %G number of GPU devices (runtime_constraints.cuda.device_count)
lib/config/config.default.yml:        # runtime_constraints.cuda.device_count > 0
lib/config/config.default.yml:        BsubCUDAArguments: ["-gpu", "num=%G"]
lib/config/config.default.yml:        # Include this section if the node type includes GPU (CUDA) support
lib/config/config.default.yml:        CUDA:
lib/config/load.go:			ldr.checkCUDAVersions(cc),
lib/config/load.go:func (ldr *Loader) checkCUDAVersions(cc arvados.Cluster) error {
lib/config/load.go:		if it.CUDA.DeviceCount == 0 {
lib/config/load.go:		_, err := strconv.ParseFloat(it.CUDA.DriverVersion, 64)
lib/config/load.go:			return fmt.Errorf("InstanceType %q has invalid CUDA.DriverVersion %q, expected format X.Y (%v)", it.Name, it.CUDA.DriverVersion, err)
lib/config/load.go:		_, err = strconv.ParseFloat(it.CUDA.HardwareCapability, 64)
lib/config/load.go:			return fmt.Errorf("InstanceType %q has invalid CUDA.HardwareCapability %q, expected format X.Y (%v)", it.Name, it.CUDA.HardwareCapability, err)
lib/lsf/dispatch_test.go:	crCUDARequest arvados.ContainerRequest
lib/lsf/dispatch_test.go:	err = arvados.NewClientFromEnv().RequestAndDecode(&s.crCUDARequest, "POST", "arvados/v1/container_requests", nil, map[string]interface{}{
lib/lsf/dispatch_test.go:				CUDA: arvados.CUDARuntimeConstraints{
lib/lsf/dispatch_test.go:			case s.crCUDARequest.ContainerUUID:
lib/lsf/dispatch_test.go:					"-J", s.crCUDARequest.ContainerUUID,
lib/lsf/dispatch_test.go:					"-gpu", "num=1"})
lib/lsf/dispatch.go:		"%G": fmt.Sprintf("%d", container.RuntimeConstraints.CUDA.DeviceCount),
lib/lsf/dispatch.go:	if container.RuntimeConstraints.CUDA.DeviceCount > 0 {
lib/lsf/dispatch.go:		argumentTemplate = append(argumentTemplate, disp.Cluster.Containers.LSF.BsubCUDAArguments...)
lib/crunchrun/cuda.go:// nvidiaModprobe makes sure all the nvidia kernel modules and devices
lib/crunchrun/cuda.go:// "CUDA_ERROR_UNKNOWN".
lib/crunchrun/cuda.go:func nvidiaModprobe(writer io.Writer) {
lib/crunchrun/cuda.go:	// directly on the host, the CUDA SDK will automatically
lib/crunchrun/cuda.go:	// https://sylabs.io/guides/3.7/user-guide/gpu.html#cuda-error-unknown-when-everything-seems-to-be-correctly-configured
lib/crunchrun/cuda.go:	// If we're running "nvidia-persistenced", it sets up most of
lib/crunchrun/cuda.go:	// However, it seems that doesn't include /dev/nvidia-uvm
lib/crunchrun/cuda.go:	// "nvidia-persistenced" or otherwise have the devices set up
lib/crunchrun/cuda.go:	// Running nvida-smi the first time loads the core 'nvidia'
lib/crunchrun/cuda.go:	// kernel module creates /dev/nvidiactl the per-GPU
lib/crunchrun/cuda.go:	// /dev/nvidia* devices
lib/crunchrun/cuda.go:	nvidiaSmi := exec.Command("nvidia-smi", "-L")
lib/crunchrun/cuda.go:	nvidiaSmi.Stdout = writer
lib/crunchrun/cuda.go:	nvidiaSmi.Stderr = writer
lib/crunchrun/cuda.go:	err := nvidiaSmi.Run()
lib/crunchrun/cuda.go:		fmt.Fprintf(writer, "Warning %v: %v\n", nvidiaSmi.Args, err)
lib/crunchrun/cuda.go:	// /dev/nvidia-modeset, /dev/nvidia-nvlink, /dev/nvidia-uvm
lib/crunchrun/cuda.go:	// and /dev/nvidia-uvm-tools (-m, -l and -u).  Annoyingly,
lib/crunchrun/cuda.go:	// Nvswitch devices are multi-GPU interconnects for up to 16
lib/crunchrun/cuda.go:	// GPUs.  The "-c0 -s" flag will create /dev/nvidia-nvswitch0.
lib/crunchrun/cuda.go:	// nvswitches (i.e. more than 16 GPUs) they'll have to ensure
lib/crunchrun/cuda.go:	// that all the /dev/nvidia-nvswitch* devices exist before
lib/crunchrun/cuda.go:		nvmodprobe := exec.Command("nvidia-modprobe", "-c0", opt)
lib/crunchrun/singularity.go:	if e.spec.CUDADeviceCount != 0 {
lib/crunchrun/singularity.go:	// Singularity always makes all nvidia devices visible to the
lib/crunchrun/singularity.go:	if cudaVisibleDevices := os.Getenv("CUDA_VISIBLE_DEVICES"); cudaVisibleDevices != "" {
lib/crunchrun/singularity.go:		env = append(env, "SINGULARITYENV_CUDA_VISIBLE_DEVICES="+cudaVisibleDevices)
lib/crunchrun/docker.go:	if spec.CUDADeviceCount != 0 {
lib/crunchrun/docker.go:		if cudaVisibleDevices := os.Getenv("CUDA_VISIBLE_DEVICES"); cudaVisibleDevices != "" {
lib/crunchrun/docker.go:			deviceIds = strings.Split(cudaVisibleDevices, ",")
lib/crunchrun/docker.go:		deviceCount := spec.CUDADeviceCount
lib/crunchrun/docker.go:		// capabilities "gpu" and "nvidia" but then there's
lib/crunchrun/docker.go:		// that are passed to nvidia-container-cli.
lib/crunchrun/docker.go:		// "compute" means include the CUDA libraries and
lib/crunchrun/docker.go:		// "utility" means include the CUDA utility programs
lib/crunchrun/docker.go:		// (like nvidia-smi).
lib/crunchrun/docker.go:		// https://github.com/moby/moby/blob/7b9275c0da707b030e62c96b679a976f31f929d3/daemon/nvidia_linux.go#L37
lib/crunchrun/docker.go:		// https://github.com/containerd/containerd/blob/main/contrib/nvidia/nvidia.go
lib/crunchrun/docker.go:			Driver:       "nvidia",
lib/crunchrun/docker.go:			Capabilities: [][]string{[]string{"gpu", "nvidia", "compute", "utility"}},
lib/crunchrun/singularity_test.go:		CUDADeviceCount: 3,
lib/crunchrun/crunchrun_test.go:		c.Check(s.executor.created.CUDADeviceCount, Equals, 0)
lib/crunchrun/crunchrun_test.go:func (s *TestSuite) TestEnableCUDADeviceCount(c *C) {
lib/crunchrun/crunchrun_test.go:    "runtime_constraints": {"cuda": {"device_count": 2}},
lib/crunchrun/crunchrun_test.go:	c.Check(s.executor.created.CUDADeviceCount, Equals, 2)
lib/crunchrun/crunchrun_test.go:func (s *TestSuite) TestEnableCUDAHardwareCapability(c *C) {
lib/crunchrun/crunchrun_test.go:    "runtime_constraints": {"cuda": {"hardware_capability": "foo"}},
lib/crunchrun/crunchrun_test.go:	c.Check(s.executor.created.CUDADeviceCount, Equals, 0)
lib/crunchrun/executor.go:	CUDADeviceCount int
lib/crunchrun/crunchrun.go:	if runner.Container.RuntimeConstraints.CUDA.DeviceCount > 0 {
lib/crunchrun/crunchrun.go:		nvidiaModprobe(runner.CrunchLog)
lib/crunchrun/crunchrun.go:		CUDADeviceCount: runner.Container.RuntimeConstraints.CUDA.DeviceCount,
lib/crunchrun/docker_test.go:		CUDADeviceCount: 3,
lib/crunchrun/docker_test.go:		Driver:       "nvidia",
lib/crunchrun/docker_test.go:		Capabilities: [][]string{{"gpu", "nvidia", "compute", "utility"}},
lib/dispatchcloud/node_size.go:		driverInsuff, driverErr := versionLess(it.CUDA.DriverVersion, ctr.RuntimeConstraints.CUDA.DriverVersion)
lib/dispatchcloud/node_size.go:		capabilityInsuff, capabilityErr := versionLess(it.CUDA.HardwareCapability, ctr.RuntimeConstraints.CUDA.HardwareCapability)
lib/dispatchcloud/node_size.go:		case it.CUDA.DeviceCount < ctr.RuntimeConstraints.CUDA.DeviceCount: // insufficient CUDA devices
lib/dispatchcloud/node_size.go:		case ctr.RuntimeConstraints.CUDA.DeviceCount > 0 && (driverInsuff || driverErr != nil): // insufficient driver version
lib/dispatchcloud/node_size.go:		case ctr.RuntimeConstraints.CUDA.DeviceCount > 0 && (capabilityInsuff || capabilityErr != nil): // insufficient hardware capability
lib/dispatchcloud/node_size_test.go:func (*NodeSizeSuite) TestChooseGPU(c *check.C) {
lib/dispatchcloud/node_size_test.go:		"costly":         {Price: 4.4, RAM: 4000000000, VCPUs: 8, Scratch: 2 * GiB, Name: "costly", CUDA: arvados.CUDAFeatures{DeviceCount: 2, HardwareCapability: "9.0", DriverVersion: "11.0"}},
lib/dispatchcloud/node_size_test.go:		"low_capability": {Price: 2.1, RAM: 2000000000, VCPUs: 4, Scratch: 2 * GiB, Name: "low_capability", CUDA: arvados.CUDAFeatures{DeviceCount: 1, HardwareCapability: "8.0", DriverVersion: "11.0"}},
lib/dispatchcloud/node_size_test.go:		"best":           {Price: 2.2, RAM: 2000000000, VCPUs: 4, Scratch: 2 * GiB, Name: "best", CUDA: arvados.CUDAFeatures{DeviceCount: 1, HardwareCapability: "9.0", DriverVersion: "11.0"}},
lib/dispatchcloud/node_size_test.go:		"low_driver":     {Price: 2.1, RAM: 2000000000, VCPUs: 4, Scratch: 2 * GiB, Name: "low_driver", CUDA: arvados.CUDAFeatures{DeviceCount: 1, HardwareCapability: "9.0", DriverVersion: "10.0"}},
lib/dispatchcloud/node_size_test.go:		"cheap_gpu":      {Price: 2.0, RAM: 2000000000, VCPUs: 4, Scratch: 2 * GiB, Name: "cheap_gpu", CUDA: arvados.CUDAFeatures{DeviceCount: 1, HardwareCapability: "8.0", DriverVersion: "10.0"}},
lib/dispatchcloud/node_size_test.go:		"invalid_gpu":    {Price: 1.9, RAM: 2000000000, VCPUs: 4, Scratch: 2 * GiB, Name: "invalid_gpu", CUDA: arvados.CUDAFeatures{DeviceCount: 1, HardwareCapability: "12.0.12", DriverVersion: "12.0.12"}},
lib/dispatchcloud/node_size_test.go:		"non_gpu":        {Price: 1.1, RAM: 2000000000, VCPUs: 4, Scratch: 2 * GiB, Name: "non_gpu"},
lib/dispatchcloud/node_size_test.go:	type GPUTestCase struct {
lib/dispatchcloud/node_size_test.go:		CUDA             arvados.CUDARuntimeConstraints
lib/dispatchcloud/node_size_test.go:	cases := []GPUTestCase{
lib/dispatchcloud/node_size_test.go:		GPUTestCase{
lib/dispatchcloud/node_size_test.go:			CUDA: arvados.CUDARuntimeConstraints{
lib/dispatchcloud/node_size_test.go:		GPUTestCase{
lib/dispatchcloud/node_size_test.go:			CUDA: arvados.CUDARuntimeConstraints{
lib/dispatchcloud/node_size_test.go:		GPUTestCase{
lib/dispatchcloud/node_size_test.go:			CUDA: arvados.CUDARuntimeConstraints{
lib/dispatchcloud/node_size_test.go:		GPUTestCase{
lib/dispatchcloud/node_size_test.go:			CUDA: arvados.CUDARuntimeConstraints{
lib/dispatchcloud/node_size_test.go:		GPUTestCase{
lib/dispatchcloud/node_size_test.go:			CUDA: arvados.CUDARuntimeConstraints{
lib/dispatchcloud/node_size_test.go:		GPUTestCase{
lib/dispatchcloud/node_size_test.go:			CUDA: arvados.CUDARuntimeConstraints{
lib/dispatchcloud/node_size_test.go:			SelectedInstance: "non_gpu",
lib/dispatchcloud/node_size_test.go:				CUDA:         tc.CUDA,

```
