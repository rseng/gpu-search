# https://github.com/iontorrent/TS

```console
pipeline/python/ion/reports/uploadMetrics.py:def getCurrentAnalysis(procMetrics, res):
pipeline/python/ion/reports/uploadMetrics.py:    if procMetrics != None:
pipeline/python/ion/reports/uploadMetrics.py:        res.processedflows = procMetrics.get("numFlows", 0)
pipeline/python/ion/reports/uploadMetrics.py:        res.processedCycles = procMetrics.get("cyclesProcessed", 0)
pipeline/python/ion/reports/uploadMetrics.py:        res.framesProcessed = procMetrics.get("framesProcessed", 0)
pipeline/python/ion/utils/pci_devices.py:NVIDIA_VENDOR_ID = "10de"
pipeline/python/ion/utils/pci_devices.py:class nvidia_pci_device(pci_device):
pipeline/python/ion/utils/pci_devices.py:    """This holds the meta data for any nvidia device"""
pipeline/python/ion/utils/pci_devices.py:    def get_nvidia_devices(cls):
pipeline/python/ion/utils/pci_devices.py:        nvidia_devices = list()
pipeline/python/ion/utils/pci_devices.py:            if device.vendor_id == NVIDIA_VENDOR_ID:
pipeline/python/ion/utils/pci_devices.py:                nvidia_devices.append(device)
pipeline/python/ion/utils/pci_devices.py:        return nvidia_devices
pipeline/python/ion/utils/pci_devices.py:class nvidia_gpu_pci_device(nvidia_pci_device):
pipeline/python/ion/utils/pci_devices.py:    """This will represent an nvidia GPU pci device"""
pipeline/python/ion/utils/pci_devices.py:    def get_nvidia_gpu_devices(cls):
pipeline/python/ion/utils/pci_devices.py:        nvidia_devices = nvidia_pci_device.get_nvidia_devices()
pipeline/python/ion/utils/pci_devices.py:        nvidia_gpu_devices = list()
pipeline/python/ion/utils/pci_devices.py:        for nvidia_device in nvidia_devices:
pipeline/python/ion/utils/pci_devices.py:            if nvidia_device.device_class == VGA_DEVICE_CLASS:
pipeline/python/ion/utils/pci_devices.py:                nvidia_gpu_devices.append(nvidia_device)
pipeline/python/ion/utils/pci_devices.py:        return nvidia_gpu_devices
pipeline/python/ion/utils/pci_devices.py:        """This will return the minimum bandwith of the GPU's"""
pipeline/python/ion/utils/pci_devices.py:        """This will return a safe boolean which tells us if the GPU is still alive."""
pipeline/python/ion/utils/TSversion.py:    "ion-gpu",
pipeline/CMakeLists.txt:install(PROGRAMS  "${PROJECT_SOURCE_DIR}/bin/ion_gpuinfo" DESTINATION bin)
pipeline/oia/oiad.py:        # 6 analysis jobs are too much for some GPU's # find . -name sigproc.log | xargs grep -i StreamResources
pipeline/oia/oiad.py:        # sigproc.log:CUDA 0 StreamManager: No StreamResources could be aquired!
pipeline/oia/oiad.py:        if self.pool.analysis_counter >= total_GPU_memory / 1000:
pipeline/oia/oiad.py:            #logger.debug("GPU memory reached (analysis)")
pipeline/oia/oiad.py:        # GPU limit
pipeline/oia/oiad.py:        #        total_GPU_memory = 4000
pipeline/oia/oiad.py:        #       if current_GPU_memory + block.GPU_memory_requirement_analysis >= 6G
pipeline/oia/oiad.py:        # GPU_memory_requirement_analysis = 1G
pipeline/oia/oiad.py:            "nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv | grep -v timestamp >> /var/log/gpu_util.log&"
pipeline/oia/oiad.py:                os.system("mv /var/log/gpu_util.log /var/log/gpu_util.bak")
pipeline/oia/oiad.py:                predicted_total_GPU_memory = 0
pipeline/oia/oiad.py:                        GPU_memory_requirement_analysis = config.getint(
pipeline/oia/oiad.py:                            run.exp_chipversion, "GPU_memory_requirement_analysis"
pipeline/oia/oiad.py:                        GPU_memory_requirement_analysis = config.getint(
pipeline/oia/oiad.py:                            "DefaultChip", "GPU_memory_requirement_analysis"
pipeline/oia/oiad.py:                    predicted_total_GPU_memory += (
pipeline/oia/oiad.py:                        int(GPU_memory_requirement_analysis) * an
pipeline/oia/oiad.py:                #        "HOST: {0} G   GPU: {1} G".format(
pipeline/oia/oiad.py:                #            predicted_total_GPU_memory / 1073741824,
pipeline/oia/oiad.py:        'nvidia-smi -pm 1; nvidia-smi -e 0; if [ "`nvidia-smi | grep \'Tesla K40c\'`" != "" ]; then nvidia-smi -ac 3004,875; fi'
pipeline/oia/oiad.py:    total_GPU_memory = 4000
pipeline/oia/oiad.py:    # retrieve GPU information
pipeline/oia/oiad.py:            total_GPU_memory = 0
pipeline/oia/oiad.py:                total_GPU_memory += memory_info.total / 1024 / 1024
pipeline/oia/oiad.py:        total_GPU_memory = 4000
pipeline/oia/install2_OIA.sh:GPU_PKG_VERSION=GPUVER
pipeline/oia/install2_OIA.sh:    # remove ion-analysis ion-gpu packages older than 5.1.5
pipeline/oia/install2_OIA.sh:    INSTALLED_GPU_PKG_VERSION=`dpkg -l | grep ion-gpu | awk '{print $3}'`
pipeline/oia/install2_OIA.sh:    if [ -z "${INSTALLED_GPU_PKG_VERSION}" ];
pipeline/oia/install2_OIA.sh:        INSTALLED_GPU_PKG_VERSION=1.0
pipeline/oia/install2_OIA.sh:    echo ${INSTALLED_GPU_PKG_VERSION}
pipeline/oia/install2_OIA.sh:    if dpkg --compare-versions ${INSTALLED_GPU_PKG_VERSION} lt 5.1.5 ||
pipeline/oia/install2_OIA.sh:        dpkg -r ion-analysis ion-gpu
pipeline/oia/install2_OIA.sh:    # make sure gpu installs first, Analysis might depend on a specific version
pipeline/oia/install2_OIA.sh:    if [ ! "ii ion-gpu $GPU_PKG_VERSION" = "`dpkg -l | grep ion-gpu | awk '{print $1, $2, $3}'`" ];
pipeline/oia/install2_OIA.sh:        # stop ganglia-monitor server (which might monitor the GPU)
pipeline/oia/install2_OIA.sh:        dpkg -i GPU_PKG_NAME
pipeline/oia/install2_OIA.sh:        # remove xorg.conf (installed by CUDA)
pipeline/oia/install2_OIA.sh:    rm -f GPU_PKG_NAME
pipeline/oia/install2_OIA.sh:    nvidia-smi -e 0
pipeline/oia/install2_OIA.sh:    if [ "ii ion-gpu $GPU_PKG_VERSION"           = "`dpkg -l | grep ion-gpu      | awk '{print $1, $2, $3}'`" ] &&
pipeline/oia/oiaTimingPlot.py:span_gpu = 10
pipeline/oia/oiaTimingPlot.py:# now, read in the gpu data
pipeline/oia/oiaTimingPlot.py:    "gpu_util.log", names=["systemtime", "percent"], sep=",", parse_dates=[0]
pipeline/oia/oiaTimingPlot.py:x_axis_gpu = np.zeros(len(data2), dtype="datetime64[s]")
pipeline/oia/oiaTimingPlot.py:y_axis_gpu = np.zeros(len(data))
pipeline/oia/oiaTimingPlot.py:y_axis_gpu_smoothed = np.zeros(len(data))
pipeline/oia/oiaTimingPlot.py:    x_axis_gpu[key] = np.datetime64((data2[key]["systemtime"]))
pipeline/oia/oiaTimingPlot.py:        y_axis_gpu[key] = int((data2[key]["percent"].replace(" ", "").replace("%", "")))
pipeline/oia/oiaTimingPlot.py:        y_axis_gpu[key] = 0
pipeline/oia/oiaTimingPlot.py:# print x_axis_gpu[0]
pipeline/oia/oiaTimingPlot.py:# print x_axis_gpu[len(x_axis_gpu)-1]
pipeline/oia/oiaTimingPlot.py:        sum_gpu = 0
pipeline/oia/oiaTimingPlot.py:            sum_gpu += y_axis_gpu[key2]
pipeline/oia/oiaTimingPlot.py:        y_axis_gpu_smoothed[key] = sum_gpu / (2 * span)
pipeline/oia/oiaTimingPlot.py:plt.plot(x_axis, y_axis_gpu, "#000000", linewidth=wl, label="% gpu")
pipeline/oia/oiaTimingPlot.py:plt.plot(x_axis, y_axis_gpu_smoothed, "#000000", linewidth=0.4, label="% gpu")
pipeline/oia/pkg_deb.txt:GPU_PKG
pipeline/oia/oia.config:GPU_memory_requirement_analysis  = 1073741824
pipeline/oia/oia.config:GPU_memory_requirement_analysis  = 1073741824
pipeline/oia/oia.config:GPU_memory_requirement_analysis  = 1073741824
pipeline/oia/oia.config:GPU_memory_requirement_analysis  = 1573741824
pipeline/oia/oia.config:GPU_memory_requirement_analysis  = 1573741824
pipeline/oia/oia.config:GPU_memory_requirement_analysis  = 1073741824
pipeline/bin/ion_gpuinfo:GPU_ERROR=0
pipeline/bin/ion_gpuinfo:# = Check the presence of GPU card on PCI slots =
pipeline/bin/ion_gpuinfo:echo "GPU in PCI Slots:"
pipeline/bin/ion_gpuinfo:out=$(lspci | grep -i 'controller: nVidia')
pipeline/bin/ion_gpuinfo:         echo 'ERROR: GPU not functional (rev ff)'
pipeline/bin/ion_gpuinfo:		 (( GPU_ERROR += 1 ))
pipeline/bin/ion_gpuinfo:     echo 'ERROR: GPU not found.'
pipeline/bin/ion_gpuinfo:	 (( GPU_ERROR += 1 ))
pipeline/bin/ion_gpuinfo:# = Check the presence of CUDA library =
pipeline/bin/ion_gpuinfo:echo "CUDA Library:"
pipeline/bin/ion_gpuinfo:CUDA_DIR='/usr/local/cuda/lib64'
pipeline/bin/ion_gpuinfo:if [[ -d $CUDA_DIR ]]; then
pipeline/bin/ion_gpuinfo:	ls -l $CUDA_DIR
pipeline/bin/ion_gpuinfo:	echo 'ERROR: CUDA library not found.'
pipeline/bin/ion_gpuinfo:	(( GPU_ERROR += 1 ))
pipeline/bin/ion_gpuinfo:# = Nvidia SMI output =
pipeline/bin/ion_gpuinfo:echo "nvidia-smi output:"
pipeline/bin/ion_gpuinfo:if (which nvidia-smi >/dev/null); then
pipeline/bin/ion_gpuinfo:	if [ $GPU_ERROR -eq 0 ]; then
pipeline/bin/ion_gpuinfo:		nvidia-smi -q -f /tmp/nvidia-smi.out
pipeline/bin/ion_gpuinfo:			cat /tmp/nvidia-smi.out
pipeline/bin/ion_gpuinfo:			"ERROR: problem with nvidia-smi"
pipeline/bin/ion_gpuinfo:			(( GPU_ERROR += 1 ))
pipeline/bin/ion_gpuinfo:		echo "Did not run nvidia-smi due to above errors"
pipeline/bin/ion_gpuinfo:# * add output of /opt/ion/gpu/deviceQuery
pipeline/bin/ion_gpuinfo:# * add output of /opt/ion/gpu/bandwidthTest
pipeline/bin/ion_gpuinfo:echo "Nvidia log message:"
pipeline/bin/ion_gpuinfo:exit $GPU_ERROR
buildTools/terms-of-use.txt:## Cuda Toolkit
buildTools/terms-of-use.txt:Important Notice READ CAREFULLY: This Software License Agreement ("Agreement") for NVIDIA CUDA Toolkit, including computer software and associated documentation ("Software"), is the Agreement which governs use of the SOFTWARE of NVIDIA Corporation and its subsidiaries ("NVIDIA") downloadable herefrom. By downloading, installing, copying, or otherwise using the SOFTWARE, You (as defined below) agree to be bound by the terms of this Agreement. If You do not agree to the terms of this Agreement, do not download the SOFTWARE. Recitals Use of NVIDIA's SOFTWARE requires three elements: the SOFTWARE, an NVIDIA GPU or application processor ("NVIDIA Hardware"), and a computer system. The SOFTWARE is protected by copyright laws and international copyright treaties, as well as other intellectual property laws and treaties. The SOFTWARE is not sold, and instead is only licensed for Your use, strictly in accordance with this Agreement. The NVIDIA Hardware is protected by various patents, and is sold, but this Agreement does not cover the sale or use of such hardware, since it may not necessarily be sold as a package with the SOFTWARE. This Agreement sets forth the terms and conditions of the SOFTWARE only.
buildTools/terms-of-use.txt:  1.1.3. Software "SOFTWARE" shall mean the deliverables provided pursuant to this Agreement. SOFTWARE may be provided in either source or binary form, at NVIDIA's discretion.
buildTools/terms-of-use.txt:  1.2.1. Rights and Limitations of Grant Provided that Licensee complies with the terms of this Agreement, NVIDIA hereby grants Licensee the following limited, non-exclusive, non-transferable, non-sublicensable (except as expressly permitted otherwise for Redistributable Software in Section 1.2.1.1 and Section 1.2.1.3 of this Agreement) right to use the SOFTWARE -- and, if the SOFTWARE is provided in source form, to compile the SOFTWARE -- with the following limitations:
buildTools/terms-of-use.txt:  1.2.1.1. Redistribution Rights Licensee may transfer, redistribute, and sublicense certain files of the Redistributable SOFTWARE, as defined in Attachment A of this Agreement, provided, however, that (a) the Redistributable SOFTWARE shall be distributed solely in binary form to Licensee's licensees ("Customers") only as a component of Licensee's own software products (each, a "Licensee Application"); (b) Licensee shall design the Licensee Application such that the Redistributable SOFTWARE files are installed only in a private (non-shared) directory location that is used only by the Licensee Application; (C) Licensee shall obtain each Customer's written or clickwrap agreement to the license terms under a written, legally enforceable agreement that has the effect of protecting the SOFTWARE and the rights of NVIDIA under terms no less restrictive than this Agreement.
buildTools/terms-of-use.txt:  1.2.1.3. Further Redistribution Rights Subject to the terms and conditions of the Agreement, Licensee may authorize Customers to further redistribute the Redistributable SOFTWARE that such Customers receive as part of the Licensee Application, solely in binary form, provided, however, that Licensee shall require in their standard software license agreements with Customers that all such redistributions must be made pursuant to a license agreement that has the effect of protecting the SOFTWARE and the rights of NVIDIA whose terms and conditions are at least as restrictive as those in the applicable Licensee software license agreement covering the Licensee Application. For avoidance of doubt, termination of this Agreement shall not affect rights previously granted by Licensee to its Customers under this Agreement to the extent validly granted to Customers under Section 1.2.1.1.
buildTools/terms-of-use.txt:1.3. Term and Termination This Agreement will continue in effect for two (2) years ("Initial Term") after Your initial download and use of the SOFTWARE, subject to the exclusive right of NVIDIA to terminate as provided herein. The term of this Agreement will automatically renew for successive one (1) year renewal terms after the Initial Term, unless either party provides to the other party at least three (3) months prior written notice of termination before the end of the applicable renewal term. This Agreement will automatically terminate if Licensee fails to comply with any of the terms and conditions hereof. In such event, Licensee must destroy all copies of the SOFTWARE and all of its component parts. Defensive Suspension If Licensee commences or participates in any legal proceeding against NVIDIA, then NVIDIA may, in its sole discretion, suspend or terminate all license grants and any other rights provided under this Agreement during the pendency of such legal proceedings.
buildTools/terms-of-use.txt:1.4. Copyright All rights, title, interest and copyrights in and to the SOFTWARE (including but not limited to all images, photographs, animations, video, audio, music, text, and other information incorporated into the SOFTWARE), the accompanying printed materials, and any copies of the SOFTWARE, are owned by NVIDIA, or its suppliers. The SOFTWARE is protected by copyright laws and international treaty provisions. Accordingly, Licensee is required to treat the SOFTWARE like any other copyrighted material, except as otherwise allowed pursuant to this Agreement and that it may make one copy of the SOFTWARE solely for backup or archive purposes. RESTRICTED RIGHTS NOTICE. Software has been developed entirely at private expense and is commercial computer software provided with RESTRICTED RIGHTS. Use, duplication or disclosure by the U.S. Government or a U.S. Government subcontractor is subject to the restrictions set forth in the Agreement under which Software was obtained pursuant to DFARS 227.7202-3(a) or as set forth in subparagraphs (C)(1) and (2) of the Commercial Computer Software - Restricted Rights clause at FAR 52.227-19, as applicable. Contractor/manufacturer is NVIDIA, 2701 San Tomas Expressway, Santa Clara, CA 95050.
buildTools/terms-of-use.txt:	1.6.1. No Warranties TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE SOFTWARE IS PROVIDED "AS IS" AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NONINFRINGEMENT.
buildTools/terms-of-use.txt:	1.6.2. No Liability for Consequential Damages TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE THE SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
buildTools/terms-of-use.txt:	1.6.3. No Support . NVIDIA has no obligation to support or to provide any updates of the Software.
buildTools/terms-of-use.txt:1.7.1. Feedback Notwithstanding any Non-Disclosure Agreement executed by and between the parties, the parties agree that in the event Licensee or NVIDIA provides Feedback (as defined below) to the other party on how to design, implement, or improve the SOFTWARE or Licensee's product(s) for use with the SOFTWARE, the following terms and conditions apply the Feedback:
buildTools/terms-of-use.txt:1.7.1.1. Exchange of Feedback Both parties agree that neither party has an obligation to give the other party any suggestions, comments or other feedback, whether verbally or in written or source code form, relating to (i) the SOFTWARE; (ii) Licensee's products; (iii) Licensee's use of the SOFTWARE; or (iv) optimization/interoperability of Licensee's product with the SOFTWARE (collectively defined as "Feedback"). In the event either party provides Feedback to the other party, the party receiving the Feedback may use any Feedback that the other party voluntarily provides to improve the (i) SOFTWARE or other related NVIDIA technologies, respectively for the benefit of NVIDIA; or (ii) Licensee's product or other related Licensee technologies, respectively for the benefit of Licensee. Accordingly, if either party provides Feedback to the other party, both parties agree that the other party and its respective licensees may freely use, reproduce, license, distribute, and otherwise commercialize the Feedback in the (i) SOFTWARE or other related technologies; or (ii) Licensee's products or other related technologies, respectively, without the payment of any royalties or fees.
buildTools/terms-of-use.txt:1.7.1.2. Residual Rights Licensee agrees that NVIDIA shall be free to use any general knowledge, skills and experience, (including, but not limited to, ideas, concepts, know-how, or techniques) ("Residuals"), contained in the (i) Feedback provided by Licensee to NVIDIA; (ii) Licensee's products shared or disclosed to NVIDIA in connection with the Feedback; or (C) Licensee's confidential information voluntarily provided to NVIDIA in connection with the Feedback, which are retained in the memories of NVIDIA's employees, agents, or contractors who have had access to such Residuals. Subject to the terms and conditions of this Agreement, NVIDIA's employees, agents, or contractors shall not be prevented from using Residuals as part of such employee's, agent's or contractor's general knowledge, skills, experience, talent, and/or expertise. NVIDIA shall not have any obligation to limit or restrict the assignment of such employees, agents or contractors or to pay royalties for any work resulting from the use of Residuals.
buildTools/terms-of-use.txt:1.7.2. Freedom of Action Licensee agrees that this Agreement is nonexclusive and NVIDIA may currently or in the future be developing software, other technology or confidential information internally, or receiving confidential information from other parties that maybe similar to the Feedback and Licensee's confidential information (as provided in Section 1.7.1.2 above), which may be provided to NVIDIA in connection with Feedback by Licensee. Accordingly, Licensee agrees that nothing in this Agreement will be construed as a representation or inference that NVIDIA will not develop, design, manufacture, acquire, market products, or have products developed, designed, manufactured, acquired, or marketed for NVIDIA, that compete with the Licensee's products or confidential information.
buildTools/terms-of-use.txt:1.7.3. No Implied Licenses Under no circumstances should anything in this Agreement be construed as NVIDIA granting by implication, estoppel or otherwise, (i) a license to any NVIDIA product or technology other than the SOFTWARE; or (ii) any additional license rights for the SOFTWARE other than the licenses expressly granted in this Agreement. If any provision of this Agreement is inconsistent with, or cannot be fully enforced under, the law, such provision will be construed as limited to the extent necessary to be consistent with and fully enforceable under the law. This Agreement is the final, complete and exclusive agreement between the parties relating to the subject matter hereof, and supersedes all prior or contemporaneous understandings and agreements relating to such subject matter, whether oral or written. This Agreement may only be modified in writing signed by an authorized officer of NVIDIA. Licensee agrees that it will not ship, transfer or export the SOFTWARE into any country, or use the SOFTWARE in any manner, prohibited by the United States Bureau of Industry and Security or any export laws, restrictions or regulations. The parties agree that the following sections of the Agreement will survive the termination of the License: Section 1.2.1.4, Section 1.4, Section 1.5, Section 1.6, and Section 1.7.
buildTools/terms-of-use.txt:1.8. Attachment A Redistributable Software In connection with Section 1.2.1.1 of this Agreement, the following files may be redistributed with software applications developed by Licensee, including certain variations of these files that have version number or architecture specific information NVIDIA CUDA Toolkit License Agreement www.nvidia.com End User License Agreements (EULA) DR-06739-001_v01_v8.0 | 9 embedded in the file name - as an example only, for release version 6.0 of the 64-bit Windows software, the file cudart64_60.dll is redistributable.
buildTools/terms-of-use.txt:Component : CUDA Runtime Windows : cudart.dll, cudart_static.lib, cudadevrt.lib Mac OSX : libcudart.dylib, libcudart_static.a, libcudadevrt.a Linux : libcudart.so, libcudart_static.a, libcudadevrt.a Android : libcudart.so, libcudart_static.a, libcudadevrt.a Component : CUDA FFT Library Windows : cufft.dll, cufftw.dll Mac OSX : libcufft.dylib, libcufft_static.a, libcufftw.dylib, libcufftw_static.a Linux : libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a Android : libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a Component : CUDA BLAS Library Windows : cublas.dll, cublas_device.lib Mac OSX : libcublas.dylib, libcublas_static.a, libcublas_device.a Linux : libcublas.so, libcublas_static.a, libcublas_device.a Android : libcublas.so, libcublas_static.a, libcublas_device.a Component : NVIDIA "Drop-in" BLAS Library Windows : nvblas.dll Mac OSX : libnvblas.dylib Linux : libnvblas.so Component : CUDA Sparse Matrix Library Windows : cusparse.dll Mac OSX : libcusparse.dylib, libcusparse_static.a Linux : libcusparse.so, libcusparse_static.a Android : libcusparse.so, libcusparse_static.a Component : CUDA Linear Solver Library Windows : cusolver.dll Mac OSX : libcusolver.dylib, libcusolver_static.a Linux : libcusolver.so, libcusolver_static.a Android : libcusolver.so, libcusolver_static.a Component : CUDA Random Number Generation Library Windows : curand.dll Mac OSX : libcurand.dylib, libcurand_static.a Linux : libcurand.so, libcurand_static.a Android : libcurand.so, libcurand_static.a Component : NVIDIA Performance Primitives Library Windows : nppc.dll, nppi.dll, npps.dll Mac OSX : libnppc.dylib, libnppi.dylib, libnpps.dylib, libnppc_static.a, libnpps_static.a, libnppi_static.a Linux : libnppc.so, libnppi.so, libnpps.so, libnppc_static.a, libnpps_static.a, libnppi_static.a Android : libnppc.so, libnppi.so, libnpps.so, libnppc_static.a, libnpps_static.a, libnppi_static.a Component : Internal common library required for statically linking to cuBLAS, cuSPARSE, cuFFT, cuRAND and NPP Mac OSX : libculibos.a Linux : libculibos.a Component : NVIDIA Runtime Compilation Library Windows : nvrtc.dll, nvrtc-builtins.dll Mac OSX : libnvrtc.dylib, libnvrtc-builtins.dylib Linux : libnvrtc.so, libnvrtc-builtins.so Component : NVIDIA Optimizing Compiler Library Windows : nvvm.dll Mac OSX : libnvvm.dylib Linux : libnvvm.so Component : NVIDIA Common Device Math Functions Library Windows : libdevice.compute_20.bc, libdevice.compute_30.bc, libdevice.compute_35.bc Mac OSX : libdevice.compute_20.bc, libdevice.compute_30.bc, libdevice.compute_35.bc Linux : libdevice.compute_20.bc, libdevice.compute_30.bc, libdevice.compute_35.bc Component : CUDA Occupancy Calculation Header Library All : cuda_occupancy.h Component : Profiling Tools Interface Library Windows : cupti.dll Mac OSX : libcupti.dylib Linux : libcupti.so
buildTools/terms-of-use.txt:1. Licensee's use of the GDB third party component is subject to the terms and conditions of GNU GPL v3: This product includes copyrighted third-party software licensed under the terms of the GNU General Public License v3 ("GPL v3"). All third-party software packages are copyright by their respective authors. GPL v3 terms and conditions are hereby incorporated into the Agreement by this reference: http://www.gnu.org/licenses/gpl.txt Consistent with these licensing requirements, the software listed below is provided under the terms of the specified open source software licenses. To obtain source code for software provided under licenses that require redistribution of source code, including the GNU General Public License (GPL) and GNU Lesser General Public License (LGPL), contact oss-requests@nvidia.com. This offer is valid for a period of three (3) years from the date of the distribution of this product by NVIDIA CORPORATION. Component License CUDA-GDB GPL v3
buildTools/terms-of-use.txt:Copyright (C) 2000-2020, Intel Corporation, all rights reserved. Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved. Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
buildTools/LICENSE.txt.in:## Cuda Toolkit
buildTools/LICENSE.txt.in:Important Notice READ CAREFULLY: This Software License Agreement ("Agreement") for NVIDIA CUDA Toolkit, including computer software and associated documentation ("Software"), is the Agreement which governs use of the SOFTWARE of NVIDIA Corporation and its subsidiaries ("NVIDIA") downloadable herefrom. By downloading, installing, copying, or otherwise using the SOFTWARE, You (as defined below) agree to be bound by the terms of this Agreement. If You do not agree to the terms of this Agreement, do not download the SOFTWARE. Recitals Use of NVIDIA's SOFTWARE requires three elements: the SOFTWARE, an NVIDIA GPU or application processor ("NVIDIA Hardware"), and a computer system. The SOFTWARE is protected by copyright laws and international copyright treaties, as well as other intellectual property laws and treaties. The SOFTWARE is not sold, and instead is only licensed for Your use, strictly in accordance with this Agreement. The NVIDIA Hardware is protected by various patents, and is sold, but this Agreement does not cover the sale or use of such hardware, since it may not necessarily be sold as a package with the SOFTWARE. This Agreement sets forth the terms and conditions of the SOFTWARE only.
buildTools/LICENSE.txt.in:  1.1.3. Software "SOFTWARE" shall mean the deliverables provided pursuant to this Agreement. SOFTWARE may be provided in either source or binary form, at NVIDIA's discretion.
buildTools/LICENSE.txt.in:  1.2.1. Rights and Limitations of Grant Provided that Licensee complies with the terms of this Agreement, NVIDIA hereby grants Licensee the following limited, non-exclusive, non-transferable, non-sublicensable (except as expressly permitted otherwise for Redistributable Software in Section 1.2.1.1 and Section 1.2.1.3 of this Agreement) right to use the SOFTWARE -- and, if the SOFTWARE is provided in source form, to compile the SOFTWARE -- with the following limitations:
buildTools/LICENSE.txt.in:  1.2.1.1. Redistribution Rights Licensee may transfer, redistribute, and sublicense certain files of the Redistributable SOFTWARE, as defined in Attachment A of this Agreement, provided, however, that (a) the Redistributable SOFTWARE shall be distributed solely in binary form to Licensee's licensees ("Customers") only as a component of Licensee's own software products (each, a "Licensee Application"); (b) Licensee shall design the Licensee Application such that the Redistributable SOFTWARE files are installed only in a private (non-shared) directory location that is used only by the Licensee Application; &copy; Licensee shall obtain each Customer's written or clickwrap agreement to the license terms under a written, legally enforceable agreement that has the effect of protecting the SOFTWARE and the rights of NVIDIA under terms no less restrictive than this Agreement.
buildTools/LICENSE.txt.in:  1.2.1.3. Further Redistribution Rights Subject to the terms and conditions of the Agreement, Licensee may authorize Customers to further redistribute the Redistributable SOFTWARE that such Customers receive as part of the Licensee Application, solely in binary form, provided, however, that Licensee shall require in their standard software license agreements with Customers that all such redistributions must be made pursuant to a license agreement that has the effect of protecting the SOFTWARE and the rights of NVIDIA whose terms and conditions are at least as restrictive as those in the applicable Licensee software license agreement covering the Licensee Application. For avoidance of doubt, termination of this Agreement shall not affect rights previously granted by Licensee to its Customers under this Agreement to the extent validly granted to Customers under Section 1.2.1.1.
buildTools/LICENSE.txt.in:1.3. Term and Termination This Agreement will continue in effect for two (2) years ("Initial Term") after Your initial download and use of the SOFTWARE, subject to the exclusive right of NVIDIA to terminate as provided herein. The term of this Agreement will automatically renew for successive one (1) year renewal terms after the Initial Term, unless either party provides to the other party at least three (3) months prior written notice of termination before the end of the applicable renewal term. This Agreement will automatically terminate if Licensee fails to comply with any of the terms and conditions hereof. In such event, Licensee must destroy all copies of the SOFTWARE and all of its component parts. Defensive Suspension If Licensee commences or participates in any legal proceeding against NVIDIA, then NVIDIA may, in its sole discretion, suspend or terminate all license grants and any other rights provided under this Agreement during the pendency of such legal proceedings.
buildTools/LICENSE.txt.in:1.4. Copyright All rights, title, interest and copyrights in and to the SOFTWARE (including but not limited to all images, photographs, animations, video, audio, music, text, and other information incorporated into the SOFTWARE), the accompanying printed materials, and any copies of the SOFTWARE, are owned by NVIDIA, or its suppliers. The SOFTWARE is protected by copyright laws and international treaty provisions. Accordingly, Licensee is required to treat the SOFTWARE like any other copyrighted material, except as otherwise allowed pursuant to this Agreement and that it may make one copy of the SOFTWARE solely for backup or archive purposes. RESTRICTED RIGHTS NOTICE. Software has been developed entirely at private expense and is commercial computer software provided with RESTRICTED RIGHTS. Use, duplication or disclosure by the U.S. Government or a U.S. Government subcontractor is subject to the restrictions set forth in the Agreement under which Software was obtained pursuant to DFARS 227.7202-3(a) or as set forth in subparagraphs &copy;(1) and (2) of the Commercial Computer Software - Restricted Rights clause at FAR 52.227-19, as applicable. Contractor/manufacturer is NVIDIA, 2701 San Tomas Expressway, Santa Clara, CA 95050.
buildTools/LICENSE.txt.in:	1.6.1. No Warranties TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE SOFTWARE IS PROVIDED "AS IS" AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NONINFRINGEMENT.
buildTools/LICENSE.txt.in:	1.6.2. No Liability for Consequential Damages TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE THE SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
buildTools/LICENSE.txt.in:	1.6.3. No Support . NVIDIA has no obligation to support or to provide any updates of the Software.
buildTools/LICENSE.txt.in:1.7.1. Feedback Notwithstanding any Non-Disclosure Agreement executed by and between the parties, the parties agree that in the event Licensee or NVIDIA provides Feedback (as defined below) to the other party on how to design, implement, or improve the SOFTWARE or Licensee's product(s) for use with the SOFTWARE, the following terms and conditions apply the Feedback:
buildTools/LICENSE.txt.in:1.7.1.1. Exchange of Feedback Both parties agree that neither party has an obligation to give the other party any suggestions, comments or other feedback, whether verbally or in written or source code form, relating to (i) the SOFTWARE; (ii) Licensee's products; (iii) Licensee's use of the SOFTWARE; or (iv) optimization/interoperability of Licensee's product with the SOFTWARE (collectively defined as "Feedback"). In the event either party provides Feedback to the other party, the party receiving the Feedback may use any Feedback that the other party voluntarily provides to improve the (i) SOFTWARE or other related NVIDIA technologies, respectively for the benefit of NVIDIA; or (ii) Licensee's product or other related Licensee technologies, respectively for the benefit of Licensee. Accordingly, if either party provides Feedback to the other party, both parties agree that the other party and its respective licensees may freely use, reproduce, license, distribute, and otherwise commercialize the Feedback in the (i) SOFTWARE or other related technologies; or (ii) Licensee's products or other related technologies, respectively, without the payment of any royalties or fees.
buildTools/LICENSE.txt.in:1.7.1.2. Residual Rights Licensee agrees that NVIDIA shall be free to use any general knowledge, skills and experience, (including, but not limited to, ideas, concepts, know-how, or techniques) ("Residuals"), contained in the (i) Feedback provided by Licensee to NVIDIA; (ii) Licensee's products shared or disclosed to NVIDIA in connection with the Feedback; or &copy; Licensee's confidential information voluntarily provided to NVIDIA in connection with the Feedback, which are retained in the memories of NVIDIA's employees, agents, or contractors who have had access to such Residuals. Subject to the terms and conditions of this Agreement, NVIDIA's employees, agents, or contractors shall not be prevented from using Residuals as part of such employee's, agent's or contractor's general knowledge, skills, experience, talent, and/or expertise. NVIDIA shall not have any obligation to limit or restrict the assignment of such employees, agents or contractors or to pay royalties for any work resulting from the use of Residuals.
buildTools/LICENSE.txt.in:1.7.2. Freedom of Action Licensee agrees that this Agreement is nonexclusive and NVIDIA may currently or in the future be developing software, other technology or confidential information internally, or receiving confidential information from other parties that maybe similar to the Feedback and Licensee's confidential information (as provided in Section 1.7.1.2 above), which may be provided to NVIDIA in connection with Feedback by Licensee. Accordingly, Licensee agrees that nothing in this Agreement will be construed as a representation or inference that NVIDIA will not develop, design, manufacture, acquire, market products, or have products developed, designed, manufactured, acquired, or marketed for NVIDIA, that compete with the Licensee's products or confidential information.
buildTools/LICENSE.txt.in:1.7.3. No Implied Licenses Under no circumstances should anything in this Agreement be construed as NVIDIA granting by implication, estoppel or otherwise, (i) a license to any NVIDIA product or technology other than the SOFTWARE; or (ii) any additional license rights for the SOFTWARE other than the licenses expressly granted in this Agreement. If any provision of this Agreement is inconsistent with, or cannot be fully enforced under, the law, such provision will be construed as limited to the extent necessary to be consistent with and fully enforceable under the law. This Agreement is the final, complete and exclusive agreement between the parties relating to the subject matter hereof, and supersedes all prior or contemporaneous understandings and agreements relating to such subject matter, whether oral or written. This Agreement may only be modified in writing signed by an authorized officer of NVIDIA. Licensee agrees that it will not ship, transfer or export the SOFTWARE into any country, or use the SOFTWARE in any manner, prohibited by the United States Bureau of Industry and Security or any export laws, restrictions or regulations. The parties agree that the following sections of the Agreement will survive the termination of the License: Section 1.2.1.4, Section 1.4, Section 1.5, Section 1.6, and Section 1.7.
buildTools/LICENSE.txt.in:1.8. Attachment A Redistributable Software In connection with Section 1.2.1.1 of this Agreement, the following files may be redistributed with software applications developed by Licensee, including certain variations of these files that have version number or architecture specific information NVIDIA CUDA Toolkit License Agreement www.nvidia.com End User License Agreements (EULA) DR-06739-001_v01_v8.0 | 9 embedded in the file name - as an example only, for release version 6.0 of the 64-bit Windows software, the file cudart64_60.dll is redistributable.
buildTools/LICENSE.txt.in:Component : CUDA Runtime Windows : cudart.dll, cudart_static.lib, cudadevrt.lib Mac OSX : libcudart.dylib, libcudart_static.a, libcudadevrt.a Linux : libcudart.so, libcudart_static.a, libcudadevrt.a Android : libcudart.so, libcudart_static.a, libcudadevrt.a Component : CUDA FFT Library Windows : cufft.dll, cufftw.dll Mac OSX : libcufft.dylib, libcufft_static.a, libcufftw.dylib, libcufftw_static.a Linux : libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a Android : libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a Component : CUDA BLAS Library Windows : cublas.dll, cublas_device.lib Mac OSX : libcublas.dylib, libcublas_static.a, libcublas_device.a Linux : libcublas.so, libcublas_static.a, libcublas_device.a Android : libcublas.so, libcublas_static.a, libcublas_device.a Component : NVIDIA "Drop-in" BLAS Library Windows : nvblas.dll Mac OSX : libnvblas.dylib Linux : libnvblas.so Component : CUDA Sparse Matrix Library Windows : cusparse.dll Mac OSX : libcusparse.dylib, libcusparse_static.a Linux : libcusparse.so, libcusparse_static.a Android : libcusparse.so, libcusparse_static.a Component : CUDA Linear Solver Library Windows : cusolver.dll Mac OSX : libcusolver.dylib, libcusolver_static.a Linux : libcusolver.so, libcusolver_static.a Android : libcusolver.so, libcusolver_static.a Component : CUDA Random Number Generation Library Windows : curand.dll Mac OSX : libcurand.dylib, libcurand_static.a Linux : libcurand.so, libcurand_static.a Android : libcurand.so, libcurand_static.a Component : NVIDIA Performance Primitives Library Windows : nppc.dll, nppi.dll, npps.dll Mac OSX : libnppc.dylib, libnppi.dylib, libnpps.dylib, libnppc_static.a, libnpps_static.a, libnppi_static.a Linux : libnppc.so, libnppi.so, libnpps.so, libnppc_static.a, libnpps_static.a, libnppi_static.a Android : libnppc.so, libnppi.so, libnpps.so, libnppc_static.a, libnpps_static.a, libnppi_static.a Component : Internal common library required for statically linking to cuBLAS, cuSPARSE, cuFFT, cuRAND and NPP Mac OSX : libculibos.a Linux : libculibos.a Component : NVIDIA Runtime Compilation Library Windows : nvrtc.dll, nvrtc-builtins.dll Mac OSX : libnvrtc.dylib, libnvrtc-builtins.dylib Linux : libnvrtc.so, libnvrtc-builtins.so Component : NVIDIA Optimizing Compiler Library Windows : nvvm.dll Mac OSX : libnvvm.dylib Linux : libnvvm.so Component : NVIDIA Common Device Math Functions Library Windows : libdevice.compute_20.bc, libdevice.compute_30.bc, libdevice.compute_35.bc Mac OSX : libdevice.compute_20.bc, libdevice.compute_30.bc, libdevice.compute_35.bc Linux : libdevice.compute_20.bc, libdevice.compute_30.bc, libdevice.compute_35.bc Component : CUDA Occupancy Calculation Header Library All : cuda_occupancy.h Component : Profiling Tools Interface Library Windows : cupti.dll Mac OSX : libcupti.dylib Linux : libcupti.so
buildTools/LICENSE.txt.in:1. Licensee's use of the GDB third party component is subject to the terms and conditions of GNU GPL v3: This product includes copyrighted third-party software licensed under the terms of the GNU General Public License v3 ("GPL v3"). All third-party software packages are copyright by their respective authors. GPL v3 terms and conditions are hereby incorporated into the Agreement by this reference: http://www.gnu.org/licenses/gpl.txt Consistent with these licensing requirements, the software listed below is provided under the terms of the specified open source software licenses. To obtain source code for software provided under licenses that require redistribution of source code, including the GNU General Public License (GPL) and GNU Lesser General Public License (LGPL), contact oss-requests@nvidia.com. This offer is valid for a period of three (3) years from the date of the distribution of this product by NVIDIA CORPORATION. Component License CUDA-GDB GPL v3
buildTools/LICENSE.txt.in:Copyright &copy; 2000-2020, Intel Corporation, all rights reserved. Copyright &copy; 2009-2011, Willow Garage Inc., all rights reserved. Copyright &copy; 2009-2016, NVIDIA Corporation, all rights reserved.
buildTools/build.sh:  gpu
buildTools/BUILD.txt:   MODULES="gpu Analysis" ./buildTools/build.sh
buildTools/BUILD.txt:   MODULES="gpu Analysis" ./buildTools/build.sh
buildTools/cmake/CMakeLists.installpath.txt:# override with: -DION_GPU_PREFIX
buildTools/cmake/CMakeLists.installpath.txt:set(ION_GPU_PREFIX "/opt/ion/gpu" CACHE PATH "Ion GPU Prefix")
buildTools/cmake/CMakeLists.dependencies.txt:option(ION_USE_SYSTEM_CUDA "Use CUDA system libraries" OFF)
buildTools/cmake/CMakeLists.dependencies.txt:mark_as_advanced(ION_USE_SYSTEM_CUDA)
buildTools/cmake/CMakeLists.dependencies.txt:option(ION_USE_CUDA "Compile CUDA code" ON)
buildTools/cmake/CMakeLists.dependencies.txt:mark_as_advanced(ION_USE_CUDA)
buildTools/cmake/CMakeLists.dependencies.txt:  set(cuda_proj_version "10.0.130-24817639")
buildTools/cmake/CMakeLists.dependencies.txt:  set(cuda_toolkit_tar_file "cuda-linux64-18.04-rel-${cuda_proj_version}.tar.gz")
buildTools/cmake/CMakeLists.dependencies.txt:  set(cuda_proj_version "8.0.44-21122537")
buildTools/cmake/CMakeLists.dependencies.txt:  set(cuda_toolkit_tar_file "cuda-linux64-16.04-rel-${cuda_proj_version}.tar.gz")
buildTools/cmake/CMakeLists.dependencies.txt:string(REGEX REPLACE "(.*)-[0-9]*" "\\1" CUDA_VERSION ${cuda_proj_version})
buildTools/cmake/CMakeLists.dependencies.txt:set(cuda_toolkit "cuda_toolkit")
buildTools/cmake/CMakeLists.dependencies.txt:set(cuda_toolkit_version "${cuda_toolkit}-${cuda_proj_version}")
buildTools/cmake/CMakeLists.dependencies.txt:if(ION_USE_CUDA)
buildTools/cmake/CMakeLists.dependencies.txt:    message(STATUS "BUILD with CUDA ${CUDA_VERSION}")
buildTools/cmake/CMakeLists.dependencies.txt:    add_definitions(-DION_COMPILE_CUDA)
buildTools/cmake/CMakeLists.dependencies.txt:    if (NOT ION_USE_SYSTEM_CUDA)
buildTools/cmake/CMakeLists.dependencies.txt:        ExternalProject_add(${cuda_toolkit}
buildTools/cmake/CMakeLists.dependencies.txt:            PREFIX ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}-prefix
buildTools/cmake/CMakeLists.dependencies.txt:            SOURCE_DIR ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}
buildTools/cmake/CMakeLists.dependencies.txt:            URL "http://${ION_UPDATE_SERVER}/updates/software/external/${cuda_toolkit_tar_file}"
buildTools/cmake/CMakeLists.dependencies.txt:            #PATCH_COMMAND patch -p1 -t -N < "${PROJECT_SOURCE_DIR}/../external/${cuda_toolkit_patch_file}"
buildTools/cmake/CMakeLists.dependencies.txt:        set(CUDA_TOOLKIT_ROOT_DIR "${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}")
buildTools/cmake/CMakeLists.dependencies.txt:        set(CUDA_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/include")
buildTools/cmake/CMakeLists.dependencies.txt:        set(CUDA_NVCC_EXECUTABLE "${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/bin/nvcc")
buildTools/cmake/CMakeLists.dependencies.txt:        set(CUDA_CUDART_LIBRARY "${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64/libcudart_static.a")
buildTools/cmake/CMakeLists.dependencies.txt:        set(CUDA_TOOLKIT_INCLUDE "${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/include")
buildTools/cmake/CMakeLists.dependencies.txt:        set(CUDA_cublas_LIBRARY "${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64/libcublas.so")
buildTools/cmake/CMakeLists.dependencies.txt:        set(CUDA_cufft_LIBRARY "${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64/libcufft.so")
buildTools/cmake/CMakeLists.dependencies.txt:        set(CUDA_VERBOSE_BUILD OFF)
buildTools/cmake/CMakeLists.dependencies.txt:        set(CUDA_64_BIT_DEVICE_CODE ON)
buildTools/cmake/CMakeLists.dependencies.txt:        include(${CMAKE_ROOT}/Modules/FindCUDA.cmake)
buildTools/cmake/CMakeLists.dependencies.txt:        find_package(CUDA REQUIRED)
buildTools/cmake/CMakeLists.dependencies.txt:    include_directories(${CUDA_INCLUDE_DIRS})
buildTools/cmake/CMakeLists.dependencies.txt:    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
buildTools/cmake/CMakeLists.dependencies.txt:    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
buildTools/cmake/CMakeLists.compiler.txt:set(CUDA_PROPAGATE_HOST_FLAGS OFF)
buildTools/cmake/CMakeLists.compiler.txt:  SET( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
buildTools/cmake/CMakeLists.compiler.txt:  SET( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
buildTools/cmake/CMakeLists.compiler.txt:                     "-Wno-deprecated-gpu-targets"
buildTools/cmake/CMakeLists.compiler.txt:    LIST(APPEND CUDA_NVCC_FLAGS --compiler-bindir $ENV{CXX})
buildTools/cmake/CMakeLists.compiler.txt:    add_definitions(-malign-double) ## See CUDA 4.0 Release Nodes
buildTools/cmake/Modules/FindCUDART.cmake:find_path(CUDART_INCLUDE_DIR
buildTools/cmake/Modules/FindCUDART.cmake:  NAMES cuda_runtime_api.h
buildTools/cmake/Modules/FindCUDART.cmake:  PATHS /usr/include /usr/local/cuda/include
buildTools/cmake/Modules/FindCUDART.cmake:find_library(CUDART_LIBRARY
buildTools/cmake/Modules/FindCUDART.cmake:  NAMES cudart
buildTools/cmake/Modules/FindCUDART.cmake:  PATHS /usr/lib /usr/lib64 /usr/lib/atlas /usr/lib64/atlas /usr/local/cuda/lib /usr/local/cuda/lib64
buildTools/cmake/Modules/FindCUDART.cmake:set(CUDART_PROCESS_INCLUDES CUDART_INCLUDE_DIR CUDART_INCLUDE_DIRS)
buildTools/cmake/Modules/FindCUDART.cmake:set(CUDART_PROCESS_LIBS CUDART_LIBRARY CUDART_LIBRARIES)
buildTools/cmake/Modules/FindCUDART.cmake:set(CUDART_FIND_REQUIRED 1)
buildTools/cmake/Modules/FindCUDART.cmake:libfind_process(CUDART)
buildTools/cmake/Modules/FindCUDA/parse_cubin.cmake:#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
buildTools/cmake/Modules/FindCUDA/parse_cubin.cmake:#  Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
buildTools/cmake/Modules/FindCUDA/parse_cubin.cmake:#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
buildTools/cmake/Modules/FindCUDA/parse_cubin.cmake:#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:#                               entries in CUDA_HOST_FLAGS. This is the build
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:set(CUDA_make2cmake "@CUDA_make2cmake@")
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:set(CUDA_parse_cubin "@CUDA_parse_cubin@")
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:set(CUDA_NVCC_EXECUTABLE "@CUDA_NVCC_EXECUTABLE@")
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:set(CUDA_NVCC_FLAGS "@CUDA_NVCC_FLAGS@;;@CUDA_WRAP_OPTION_NVCC_FLAGS@")
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:@CUDA_NVCC_FLAGS_CONFIG@
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:set(CUDA_NVCC_INCLUDE_ARGS "@CUDA_NVCC_INCLUDE_ARGS@")
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:# been chosen by FindCUDA.cmake.
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:@CUDA_HOST_FLAGS@
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:#message("CUDA_NVCC_HOST_COMPILER_FLAGS = ${CUDA_NVCC_HOST_COMPILER_FLAGS}")
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_${build_configuration}})
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:# cuda_execute_process - Executes a command with optional command echo and status message.
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:#   CUDA_result - return value from running the command
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:macro(cuda_execute_process status command)
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:    message(FATAL_ERROR "Malformed call to cuda_execute_process.  Missing COMMAND as second argument. (command = ${command})")
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:    set(cuda_execute_process_string)
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:        list(APPEND cuda_execute_process_string "\"${arg}\"")
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:        list(APPEND cuda_execute_process_string ${arg})
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:    execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${cuda_execute_process_string})
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  execute_process(COMMAND ${ARGN} RESULT_VARIABLE CUDA_result )
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  COMMAND "${CUDA_NVCC_EXECUTABLE}"
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  ${CUDA_NVCC_FLAGS}
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  ${CUDA_NVCC_INCLUDE_ARGS}
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:if(CUDA_result)
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  -P "${CUDA_make2cmake}"
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:if(CUDA_result)
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:if(CUDA_result)
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:if(CUDA_result)
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  COMMAND "${CUDA_NVCC_EXECUTABLE}"
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  ${CUDA_NVCC_FLAGS}
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  ${CUDA_NVCC_INCLUDE_ARGS}
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:if(CUDA_result)
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  cuda_execute_process(
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  cuda_execute_process(
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:    COMMAND "${CUDA_NVCC_EXECUTABLE}"
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:    ${CUDA_NVCC_FLAGS}
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:    ${CUDA_NVCC_INCLUDE_ARGS}
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:  cuda_execute_process(
buildTools/cmake/Modules/FindCUDA/run_nvcc.cmake:    -P "${CUDA_parse_cubin}"
buildTools/cmake/Modules/FindCUDA/make2cmake.cmake:#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
buildTools/cmake/Modules/FindCUDA/make2cmake.cmake:#  Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
buildTools/cmake/Modules/FindCUDA/make2cmake.cmake:#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
buildTools/cmake/Modules/FindCUDA/make2cmake.cmake:#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
buildTools/cmake/Modules/FindCUDA/make2cmake.cmake:  set(cuda_nvcc_depend "${cuda_nvcc_depend} \"${file}\"\n")
buildTools/cmake/Modules/FindCUDA/make2cmake.cmake:file(WRITE ${output_file} "# Generated by: make2cmake.cmake\nSET(CUDA_NVCC_DEPEND\n ${cuda_nvcc_depend})\n\n")
buildTools/cmake/CMakeLists.dependencies.TMAP.txt:#option(ION_USE_SYSTEM_CUDA "Use CUDA system libraries" OFF)
buildTools/cmake/CMakeLists.dependencies.TMAP.txt:#mark_as_advanced(ION_USE_SYSTEM_CUDA)
buildTools/cmake/CMakeLists.dependencies.TMAP.txt:#option(ION_USE_CUDA "Compile CUDA code" ON)
buildTools/cmake/CMakeLists.dependencies.TMAP.txt:#mark_as_advanced(ION_USE_CUDA)
Analysis/TMAP/scripts/standalone/buildTools/cmake/CMakeLists.compiler.txt:set(CUDA_PROPAGATE_HOST_FLAGS OFF)
Analysis/TMAP/scripts/standalone/buildTools/cmake/CMakeLists.compiler.txt:    #SET( CUDA_NVCC_FLAGS "-std=c++11")
Analysis/TMAP/scripts/standalone/buildTools/cmake/CMakeLists.compiler.txt:SET( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
Analysis/TMAP/scripts/standalone/buildTools/cmake/CMakeLists.compiler.txt:    LIST(APPEND CUDA_NVCC_FLAGS --compiler-bindir $ENV{CXX})
Analysis/TMAP/scripts/standalone/buildTools/cmake/CMakeLists.compiler.txt:    add_definitions(-malign-double) ## See CUDA 4.0 Release Nodes
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpuworkload"]["type"] = OT_DOUBLE;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpuworkload"]["value"] = 1.0;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpuworkload"]["min"] = 0.0;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpuworkload"]["max"] = 1.0;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-num-streams"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-num-streams"]["value"] = 2;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-num-streams"]["min"] = 1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-num-streams"]["max"] = 16;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-amp-guess"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-amp-guess"]["value"] = 1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-amp-guess"]["min"] = 0;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-amp-guess"]["max"] = 1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit"]["value"] = 1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit"]["min"] = 0;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit"]["max"] = 1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-blocksize"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-blocksize"]["value"] = -1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-blocksize"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-blocksize"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-l1config"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-l1config"]["value"] = -1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-l1config"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-l1config"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit"]["value"] = 1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit"]["min"] = 0;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit"]["max"] = 1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-blocksize"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-blocksize"]["value"] = 128;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-blocksize"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-blocksize"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-l1config"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-l1config"]["value"] = -1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-l1config"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-multi-flow-fit-l1config"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-type"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-type"]["value"] = 3;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-type"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-single-flow-fit-type"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-hybrid-fit-iter"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-hybrid-fit-iter"]["value"] = 3;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-hybrid-fit-iter"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-hybrid-fit-iter"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-partial-deriv-blocksize"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-partial-deriv-blocksize"]["value"] = 128;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-partial-deriv-blocksize"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-partial-deriv-blocksize"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-partial-deriv-l1config"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-partial-deriv-l1config"]["value"] = -1;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-partial-deriv-l1config"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-partial-deriv-l1config"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-use-all-devices"]["type"] = OT_BOOL;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-use-all-devices"]["value"] = false;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-use-all-devices"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-use-all-devices"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-verbose"]["type"] = OT_BOOL;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-verbose"]["value"] = false;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-verbose"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-verbose"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-device-ids"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-device-ids"]["value"] = Value(arrayValue);
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-device-ids"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-device-ids"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-fitting-only"]["type"] = OT_BOOL;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-fitting-only"]["value"] = true;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-fitting-only"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-fitting-only"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["post-fit-handshake-worker"]["type"] = OT_BOOL;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["post-fit-handshake-worker"]["value"] = true;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["post-fit-handshake-worker"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["post-fit-handshake-worker"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-flow-by-flow"]["type"] = OT_BOOL;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-flow-by-flow"]["value"] = false;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-flow-by-flow"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-flow-by-flow"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-switch-to-flow-by-flow-at"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-switch-to-flow-by-flow-at"]["value"] = 20;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-switch-to-flow-by-flow-at"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-switch-to-flow-by-flow-at"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-num-history-flows"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-num-history-flows"]["value"] = 10;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-num-history-flows"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-num-history-flows"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-force-multi-flow-fit"]["type"] = OT_BOOL;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-force-multi-flow-fit"]["value"] = false;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-force-multi-flow-fit"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-force-multi-flow-fit"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-memory-per-proc"]["type"] = OT_INT;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-memory-per-proc"]["value"] = 0;
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-memory-per-proc"]["min"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:    jsonBase["GpuControlOpts"]["gpu-memory-per-proc"]["max"] = "";
Analysis/TsInputUtil/tsInputUtil.cpp:                            if(*it1 == "chipType" || *it1 == "BkgModelControlOpts" || *it1 == "GlobalDefaultsForBkgModel" || *it1 == "GpuControlOpts"
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-amp-guess"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-device-ids"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-fitting-only"] = OT_BOOL;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-flow-by-flow"] = OT_BOOL;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-force-multi-flow-fit"] = OT_BOOL;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-hybrid-fit-iter"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-memory-per-proc"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-multi-flow-fit"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-multi-flow-fit-blocksize"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-multi-flow-fit-l1config"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-num-history-flows"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-num-streams"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-partial-deriv-blocksize"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-partial-deriv-l1config"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-single-flow-fit"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-single-flow-fit-blocksize"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-single-flow-fit-l1config"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-single-flow-fit-type"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-switch-to-flow-by-flow-at"] = OT_INT;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-use-all-devices"] = OT_BOOL;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpu-verbose"] = OT_BOOL;
Analysis/TsInputUtil/perChipJsonEditor.cpp:    mapOptType["gpuworkload"] = OT_DOUBLE;
Analysis/pynvml/pynvml_test.py:    print("Device %s: GPU Utilization: %s%%" % (i, util.gpu))
Analysis/pynvml/nvidia_smi_test.py:import nvidia_smi
Analysis/pynvml/nvidia_smi_test.py:print(nvidia_smi.XmlDeviceQuery())
Analysis/pynvml/nvidia_smi.py:# Copyright (c) 2011-2012, NVIDIA Corporation.  All rights reserved.
Analysis/pynvml/nvidia_smi.py:#    * Neither the name of the NVIDIA Corporation nor the names of its
Analysis/pynvml/nvidia_smi.py:# nvidia_smi
Analysis/pynvml/nvidia_smi.py:# nvml_bindings <at> nvidia <dot> com
Analysis/pynvml/nvidia_smi.py:# Sample code that attempts to reproduce the output of nvidia-smi -q- x
Analysis/pynvml/nvidia_smi.py:# >>> import nvidia_smi
Analysis/pynvml/nvidia_smi.py:# >>> print(nvidia_smi.XmlDeviceQuery())
Analysis/pynvml/nvidia_smi.py:        [nvmlClocksThrottleReasonGpuIdle, "clocks_throttle_reason_gpu_idle"],
Analysis/pynvml/nvidia_smi.py:        strResult += '<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v4.dtd">\n'
Analysis/pynvml/nvidia_smi.py:        strResult += "<nvidia_smi_log>\n"
Analysis/pynvml/nvidia_smi.py:        strResult += "  <attached_gpus>" + str(deviceCount) + "</attached_gpus>\n"
Analysis/pynvml/nvidia_smi.py:            strResult += '  <gpu id="%s">\n' % pciInfo.busId
Analysis/pynvml/nvidia_smi.py:            strResult += "    <gpu_operation_mode>\n"
Analysis/pynvml/nvidia_smi.py:                current = StrGOM(nvmlDeviceGetCurrentGpuOperationMode(handle))
Analysis/pynvml/nvidia_smi.py:                pending = StrGOM(nvmlDeviceGetPendingGpuOperationMode(handle))
Analysis/pynvml/nvidia_smi.py:            strResult += "    </gpu_operation_mode>\n"
Analysis/pynvml/nvidia_smi.py:            strResult += "      <pci_gpu_link_info>\n"
Analysis/pynvml/nvidia_smi.py:            strResult += "      </pci_gpu_link_info>\n"
Analysis/pynvml/nvidia_smi.py:                gpu_util = str(util.gpu) + " %"
Analysis/pynvml/nvidia_smi.py:                gpu_util = error
Analysis/pynvml/nvidia_smi.py:            strResult += "      <gpu_util>" + gpu_util + "</gpu_util>\n"
Analysis/pynvml/nvidia_smi.py:                    str(nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)) + " C"
Analysis/pynvml/nvidia_smi.py:            strResult += "      <gpu_temp>" + temp + "</gpu_temp>\n"
Analysis/pynvml/nvidia_smi.py:                    if p.usedGpuMemory == None:
Analysis/pynvml/nvidia_smi.py:                        mem = "%d MB" % (p.usedGpuMemory / 1024 / 1024)
Analysis/pynvml/nvidia_smi.py:            strResult += "  </gpu>\n"
Analysis/pynvml/nvidia_smi.py:        strResult += "</nvidia_smi_log>\n"
Analysis/pynvml/nvidia_smi.py:        strResult += "nvidia_smi.py: " + err.__str__() + "\n"
Analysis/pynvml/pynvml.py:# Copyright (c) 2011-2012, NVIDIA Corporation.  All rights reserved.
Analysis/pynvml/pynvml.py:#    * Neither the name of the NVIDIA Corporation nor the names of its
Analysis/pynvml/pynvml.py:NVML_TEMPERATURE_GPU = 0
Analysis/pynvml/pynvml.py:_nvmlGpuOperationMode_t = c_uint
Analysis/pynvml/pynvml.py:# On Windows with the WDDM driver, usedGpuMemory is reported as None
Analysis/pynvml/pynvml.py:# if (info.usedGpuMemory == None):
Analysis/pynvml/pynvml.py:#    print("Using %d MB of memory" % (info.usedGpuMemory / 1024 / 1024))
Analysis/pynvml/pynvml.py:    _fields_ = [("pid", c_uint), ("usedGpuMemory", c_ulonglong)]
Analysis/pynvml/pynvml.py:    _fields_ = [("gpu", c_uint), ("memory", c_uint)]
Analysis/pynvml/pynvml.py:nvmlClocksThrottleReasonGpuIdle = 0x0000000000000001
Analysis/pynvml/pynvml.py:    | nvmlClocksThrottleReasonGpuIdle
Analysis/pynvml/pynvml.py:                        nvmlLib = CDLL("libnvidia-ml.so")
Analysis/pynvml/pynvml.py:def nvmlDeviceGetGpuOperationMode(handle):
Analysis/pynvml/pynvml.py:    c_currState = _nvmlGpuOperationMode_t()
Analysis/pynvml/pynvml.py:    c_pendingState = _nvmlGpuOperationMode_t()
Analysis/pynvml/pynvml.py:    fn = _nvmlGetFunctionPointer("nvmlDeviceGetGpuOperationMode")
Analysis/pynvml/pynvml.py:def nvmlDeviceGetCurrentGpuOperationMode(handle):
Analysis/pynvml/pynvml.py:    return nvmlDeviceGetGpuOperationMode(handle)[0]
Analysis/pynvml/pynvml.py:def nvmlDeviceGetPendingGpuOperationMode(handle):
Analysis/pynvml/pynvml.py:    return nvmlDeviceGetGpuOperationMode(handle)[1]
Analysis/pynvml/pynvml.py:            if obj.usedGpuMemory == NVML_VALUE_NOT_AVAILABLE_ulonglong.value:
Analysis/pynvml/pynvml.py:                obj.usedGpuMemory = None
Analysis/pynvml/pynvml.py:def nvmlDeviceSetGpuOperationMode(handle, mode):
Analysis/pynvml/pynvml.py:    fn = _nvmlGetFunctionPointer("nvmlDeviceSetGpuOperationMode")
Analysis/pynvml/pynvml.py:    ret = fn(handle, _nvmlGpuOperationMode_t(mode))
Analysis/BkgModel/SignalProcessingMasterFitter.cpp://prototype GPU execution functions
Analysis/BkgModel/SignalProcessingMasterFitter.cpp:// ProcessImage had to be broken into two function, before and after GPUGenerateBeadTraces.
Analysis/BkgModel/SignalProcessingMasterFitter.cpp:bool SignalProcessingMasterFitter::InitProcessImageForGPU (
Analysis/BkgModel/SignalProcessingMasterFitter.cpp:  //for GPU execution call Prepare Load Flow
Analysis/BkgModel/SignalProcessingMasterFitter.cpp:  if ( region_data->PrepareLoadOneFlowGPU ( img,global_defaults, *region_data_extras.my_flow,
Analysis/BkgModel/SignalProcessingMasterFitter.cpp://prototype GPU execution functions
Analysis/BkgModel/SignalProcessingMasterFitter.cpp:// ProcessImage had to be broken into two function, before and after GPUGenerateBeadTraces.
Analysis/BkgModel/SignalProcessingMasterFitter.cpp:bool SignalProcessingMasterFitter::FinalizeProcessImageForGPU ( int flow_block_size )
Analysis/BkgModel/SignalProcessingMasterFitter.cpp:  if ( region_data->FinalizeLoadOneFlowGPU ( *region_data_extras.my_flow, flow_block_size ) )
Analysis/BkgModel/SignalProcessingMasterFitter.cpp:  // Lightweight friend object like the CUDA object holding a fitter
Analysis/BkgModel/SignalProcessingMasterFitter.cpp:    if (inception_state->bkg_control.gpuControl.gpuWorkLoad == 0)
Analysis/BkgModel/SignalProcessingMasterFitter.cpp:    GuessCrudeAmplitude (elapsed_time,fit_timer,global_defaults.signal_process_control.amp_guess_on_gpu, flow_block_size, flow_block_start);
Analysis/BkgModel/RegionalizedData.cpp://prototype GPU execution functions
Analysis/BkgModel/RegionalizedData.cpp:// UpdateTracesFromImage had to be broken into two function, before and after GPUGenerateBeadTraces.
Analysis/BkgModel/RegionalizedData.cpp:bool RegionalizedData::PrepareLoadOneFlowGPU (Image *img, 
Analysis/BkgModel/RegionalizedData.cpp:  // here we are setup for GPU execution
Analysis/BkgModel/RegionalizedData.cpp://Prototype GPU second half of UpdateTracesFromImage:
Analysis/BkgModel/RegionalizedData.cpp:bool RegionalizedData::FinalizeLoadOneFlowGPU ( FlowBufferInfo & my_flow, int flow_block_size )
Analysis/BkgModel/RegionalizedData.cpp:  //Do it all at once.. generate bead trace and rezero like it is done in the new GPU pipeline
Analysis/BkgModel/RegionalizedData.h:  bool PrepareLoadOneFlowGPU (Image *img, GlobalDefaultsForBkgModel &global_defaults, 
Analysis/BkgModel/RegionalizedData.h:  bool FinalizeLoadOneFlowGPU ( FlowBufferInfo & my_flow, int flow_block_size );
Analysis/BkgModel/BkgMagicDefines.h:// TODO: this is what I want for proton, but it breaks the GPU right now
Analysis/BkgModel/BkgMagicDefines.h:// to accommodate exponential tail fit large number of frames in GPU code
Analysis/BkgModel/BkgMagicDefines.h:#define MAX_COMPRESSED_FRAMES_GPU 61
Analysis/BkgModel/BkgMagicDefines.h:#define MAX_UNCOMPRESSED_FRAMES_GPU 110
Analysis/BkgModel/BkgMagicDefines.h:#define MAX_PREALLOC_COMPRESSED_FRAMES_GPU 48
Analysis/BkgModel/BkgMagicDefines.h:// Just for the GPU version, how many flows can be in a block (like numfb).
Analysis/BkgModel/BkgMagicDefines.h:#define MAX_NUM_FLOWS_IN_BLOCK_GPU 32
Analysis/BkgModel/CUDA/StreamManager.h:// cuda
Analysis/BkgModel/CUDA/StreamManager.h:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/StreamManager.h:#include "cuda_error.h"
Analysis/BkgModel/CUDA/StreamManager.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/StreamManager.h:enum cudaStreamState 
Analysis/BkgModel/CUDA/StreamManager.h:class cudaSimpleStreamExecutionUnit
Analysis/BkgModel/CUDA/StreamManager.h:  cudaStream_t _stream;
Analysis/BkgModel/CUDA/StreamManager.h:  cudaStreamState _state;
Analysis/BkgModel/CUDA/StreamManager.h:  cudaSimpleStreamExecutionUnit( streamResources * resources,  WorkerInfoQueueItem item );
Analysis/BkgModel/CUDA/StreamManager.h://  cudaStreamExecutionUnit();
Analysis/BkgModel/CUDA/StreamManager.h:  virtual ~cudaSimpleStreamExecutionUnit();
Analysis/BkgModel/CUDA/StreamManager.h:// followed by all async cuda calls needed to execute the job
Analysis/BkgModel/CUDA/StreamManager.h:  static cudaSimpleStreamExecutionUnit * makeExecutionUnit(streamResources * resources, WorkerInfoQueueItem item);
Analysis/BkgModel/CUDA/StreamManager.h:class cudaSimpleStreamManager
Analysis/BkgModel/CUDA/StreamManager.h:  cudaResourcePool * _resourcePool;
Analysis/BkgModel/CUDA/StreamManager.h:  vector<cudaSimpleStreamExecutionUnit *> _activeSEU;
Analysis/BkgModel/CUDA/StreamManager.h:  bool _GPUerror;
Analysis/BkgModel/CUDA/StreamManager.h:  bool executionComplete(cudaSimpleStreamExecutionUnit* seu );
Analysis/BkgModel/CUDA/StreamManager.h:  cudaSimpleStreamManager( WorkerInfoQueue * inQ, WorkerInfoQueue * fallbackQ );
Analysis/BkgModel/CUDA/StreamManager.h:  ~cudaSimpleStreamManager();
Analysis/BkgModel/CUDA/CudaConstDeclare.h:#ifndef CUDACONSTDECLARE_H 
Analysis/BkgModel/CUDA/CudaConstDeclare.h:#define CUDACONSTDECLARE_H 
Analysis/BkgModel/CUDA/CudaConstDeclare.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/CudaConstDeclare.h:#define USE_CUDA_ERF
Analysis/BkgModel/CUDA/CudaConstDeclare.h:#define USE_CUDA_EXP
Analysis/BkgModel/CUDA/CudaConstDeclare.h:#ifndef USE_CUDA_ERF
Analysis/BkgModel/CUDA/CudaConstDeclare.h:__constant__ static float ERF_APPROX_TABLE_CUDA[sizeof (ERF_APPROX_TABLE) ];
Analysis/BkgModel/CUDA/CudaConstDeclare.h:__constant__ static float * POISS_APPROX_TABLE_CUDA_BASE;
Analysis/BkgModel/CUDA/CudaConstDeclare.h:__constant__ static float4 * POISS_APPROX_LUT_CUDA_BASE;
Analysis/BkgModel/CUDA/CudaConstDeclare.h:#endif // CUDACONSTDECLARE_H
Analysis/BkgModel/CUDA/CudaDefines.h:#ifndef CUDADEFINES_H
Analysis/BkgModel/CUDA/CudaDefines.h:#define CUDADEFINES_H
Analysis/BkgModel/CUDA/CudaDefines.h:#define USE_CUDA_ERF
Analysis/BkgModel/CUDA/CudaDefines.h:#define USE_CUDA_EXP
Analysis/BkgModel/CUDA/CudaDefines.h:#endif // CUDADEFINES_H
Analysis/BkgModel/CUDA/CudaException.h:#ifndef CUDAEXCEPTION_H 
Analysis/BkgModel/CUDA/CudaException.h:#define CUDAEXCEPTION_H
Analysis/BkgModel/CUDA/CudaException.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/CudaException.h:class cudaException: public exception
Analysis/BkgModel/CUDA/CudaException.h:  cudaError_t err;
Analysis/BkgModel/CUDA/CudaException.h:  cudaError_t getCudaError() { return err; };
Analysis/BkgModel/CUDA/CudaException.h:    return "CUDA EXCEPTION: an Exception occurred" ;
Analysis/BkgModel/CUDA/CudaException.h:  cudaException(cudaError_t err):err(err)
Analysis/BkgModel/CUDA/CudaException.h:        << " | ** CUDA ERROR! ** " << endl                               
Analysis/BkgModel/CUDA/CudaException.h:        << " | Msg: " << cudaGetErrorString(err) << endl           
Analysis/BkgModel/CUDA/CudaException.h:class cudaExceptionDebug: public cudaException
Analysis/BkgModel/CUDA/CudaException.h:    return "CUDA EXCEPTION: an Exception occurred";
Analysis/BkgModel/CUDA/CudaException.h:  cudaExceptionDebug(cudaError_t err,  const char * file, int line):cudaException(err),file(file),line(line)
Analysis/BkgModel/CUDA/CudaException.h:        << " | ** CUDA ERROR! ** " << endl                               
Analysis/BkgModel/CUDA/CudaException.h:        << " | Msg: " << cudaGetErrorString(err) << endl           
Analysis/BkgModel/CUDA/CudaException.h:class cudaStreamCreationError: public cudaExceptionDebug
Analysis/BkgModel/CUDA/CudaException.h:    return "CUDA EXCEPTION: could not acquire stream resources";
Analysis/BkgModel/CUDA/CudaException.h:  cudaStreamCreationError( const char * file, int line):cudaExceptionDebug(cudaErrorUnknown,file,line) {};
Analysis/BkgModel/CUDA/CudaException.h:class cudaAllocationError: public cudaExceptionDebug
Analysis/BkgModel/CUDA/CudaException.h:    return "CUDA EXCEPTION: could not allocate memory";
Analysis/BkgModel/CUDA/CudaException.h:  cudaAllocationError(cudaError_t err, const char * file, int line):cudaExceptionDebug(err,file,line) {};
Analysis/BkgModel/CUDA/CudaException.h:class cudaNotEnoughMemForStream: public cudaExceptionDebug
Analysis/BkgModel/CUDA/CudaException.h:    return "CUDA EXCEPTION: Not enough memory for context and at least one stream!";
Analysis/BkgModel/CUDA/CudaException.h:  cudaNotEnoughMemForStream( const char * file, int line):cudaExceptionDebug(cudaErrorMemoryValueTooLarge,file,line) {};
Analysis/BkgModel/CUDA/CudaException.h:class cudaExecutionException: public cudaExceptionDebug
Analysis/BkgModel/CUDA/CudaException.h:    return "CUDA EXCEPTION: Error occured during job Execution!";
Analysis/BkgModel/CUDA/CudaException.h:  cudaExecutionException( cudaError_t err,  const char * file, int line):cudaExceptionDebug(err,file,line) {};
Analysis/BkgModel/CUDA/CudaException.h:#endif  // CUDAEXCEPTION_H
Analysis/BkgModel/CUDA/StreamingKernels.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/StreamingKernels.h:#include "ObsoleteCuda.h"
Analysis/BkgModel/CUDA/StreamingKernels.h:#include "cuda_error.h"
Analysis/BkgModel/CUDA/StreamingKernels.h:void copySingleFlowFitConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream);
Analysis/BkgModel/CUDA/StreamingKernels.h:void copyMultiFlowFitConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream);
Analysis/BkgModel/CUDA/StreamingKernels.h:void copyFittingConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream);
Analysis/BkgModel/CUDA/StreamingKernels.h:void copyXtalkConstParamAsync(ConstXtalkParams* ptr, int offset, cudaStream_t stream);
Analysis/BkgModel/CUDA/StreamingKernels.h:void  PerFlowGaussNewtonFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:void  PerFlowRelaxKmultGaussNewtonFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:void  PerFlowHybridFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:void  PerFlowLevMarFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:void PreSingleFitProcessing(dim3 grid, dim3 block, int smem, cudaStream_t stream,// Here FL stands for flows
Analysis/BkgModel/CUDA/StreamingKernels.h:  cudaStream_t stream,// Here FL stands for flows
Analysis/BkgModel/CUDA/StreamingKernels.h:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:void BuildMatrix( dim3 grid, dim3 block, int smem, cudaStream_t stream, 
Analysis/BkgModel/CUDA/StreamingKernels.h:void MultiFlowLevMarFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:  CpuStep* psteps, // we need a specific struct describing this config for this well fit for GPU
Analysis/BkgModel/CUDA/StreamingKernels.h:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.h:void transposeData(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, float *source, int width, int height);
Analysis/BkgModel/CUDA/StreamingKernels.h:void transposeDataToFloat(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, FG_BUFFER_TYPE *source, int width, int height);
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:#include "GpuMultiFlowFitControl.h"
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:unsigned int GpuMultiFlowFitControl::_maxBeads = 216*224;
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:unsigned int GpuMultiFlowFitControl::_maxFrames = MAX_COMPRESSED_FRAMES_GPU;
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:bool GpuMultiFlowFitControl::_gpuTraceXtalk = false;
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:GpuMultiFlowFitControl::GpuMultiFlowFitControl()
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:void GpuMultiFlowFitControl::SetFlowParams( int flow_key, int flow_block_size )
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:GpuMultiFlowFitControl::~GpuMultiFlowFitControl()
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:  // delete gpu matrix configs here
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:  map<MatrixIndex, GpuMultiFlowFitMatrixConfig*>::iterator it;
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:GpuMultiFlowFitMatrixConfig* GpuMultiFlowFitControl::createConfig(
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:  GpuMultiFlowFitMatrixConfig* config = 
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:    new GpuMultiFlowFitMatrixConfig(const_cast<master_fit_type_table*>(levMarSparseMatrices)->GetFitDescriptorByName(fitName.c_str()), 
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:void GpuMultiFlowFitControl::DetermineMaxSteps(int steps)
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:void GpuMultiFlowFitControl::DetermineMaxParams(int params)
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:GpuMultiFlowFitMatrixConfig* GpuMultiFlowFitControl::GetMatrixConfig(
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.cpp:  GpuMultiFlowFitMatrixConfig *config =  _allMatrixConfig[MatrixIndex( _activeFlowKey, _activeFlowMax, name)];
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/KernelIncludes/ImgRegParams.h:    printf("reading GPU data for new GPU fitting pipeline from file: %s\n", filename);
Analysis/BkgModel/CUDA/KernelIncludes/GpuPipelineDefines.h: * GpuPipelineDefines.h
Analysis/BkgModel/CUDA/KernelIncludes/GpuPipelineDefines.h:#ifndef GPUPIPELINEDEFINES_H_
Analysis/BkgModel/CUDA/KernelIncludes/GpuPipelineDefines.h:#define GPUPIPELINEDEFINES_H_
Analysis/BkgModel/CUDA/KernelIncludes/GpuPipelineDefines.h://#define GPU_NUM_FLOW_BUF 20
Analysis/BkgModel/CUDA/KernelIncludes/GpuPipelineDefines.h:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/KernelIncludes/GpuPipelineDefines.h:#endif /* GPUPIPELINEDEFINES_H_ */
Analysis/BkgModel/CUDA/KernelIncludes/HostParamDefines.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/KernelIncludes/HostParamDefines.h:      std::cout << "CUDA ERROR: requested XTalks span of " << xtalkSpanX << "," << xtalkSpanY << " is larger than defined MAX_WELL_XTALK_SPAN of " << MAX_WELL_XTALK_SPAN << "," << MAX_WELL_XTALK_SPAN << "!" << std::endl;
Analysis/BkgModel/CUDA/KernelIncludes/HostParamDefines.h:      std::cout << "CUDA ERROR: requested XTalks neighbours of " << this->numN << " is larger than defined MAX_XTALK_NEIGHBOURS of " << MAX_XTALK_NEIGHBOURS << "!" << endl;
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu://#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:#include "cuda_error.h"
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:  FG_BUFFER_TYPE fgTmp[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:  float traceTmp[MAX_UNCOMPRESSED_FRAMES_GPU+4];
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:  FG_BUFFER_TYPE fgTmp[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:  float traceTmp[MAX_UNCOMPRESSED_FRAMES_GPU+4];
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:  FG_BUFFER_TYPE fgTmp[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:  float traceTmp[MAX_UNCOMPRESSED_FRAMES_GPU+4];
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:  FG_BUFFER_TYPE fgTmp[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:  float traceTmp[MAX_UNCOMPRESSED_FRAMES_GPU+4];
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:    //    printf("GPU regId %u t0_sum %f t0_cnt: %d t0 avg: %f \n" , regId, t0Sum, t0Cnt, t0avgRegion);
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:    //    printf("GPU regId %u t0_sum %f t0_cnt: %d t0 avg: %f \n" , regId, t0Sum, t0Cnt, t0avgRegion);
Analysis/BkgModel/CUDA/KernelIncludes/GenerateBeadTraceKernels.cu:  float emptyTraceSum[MAX_UNCOMPRESSED_FRAMES_GPU] = {0};
Analysis/BkgModel/CUDA/KernelIncludes/FittingHelpers.cu:  const float4* ptr =  POISS_APPROX_LUT_CUDA_BASE + n * MAX_POISSON_TABLE_ROW;
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:#include "cuda_error.h"
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemcpyToSymbol( ConstFrmP, (void*) &ciP, sizeof(ConstantFrameParams), 0, cudaMemcpyHostToDevice);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemcpyToSymbol( ImgRegP, (void*) &irP, sizeof(ImgRegParamsConst), 0, cudaMemcpyHostToDevice);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemcpyToSymbol( ConstGlobalP, (void*) &pl, sizeof(ConstantParamsGlobal), 0, cudaMemcpyHostToDevice);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemcpyToSymbol( ConstFlowP, (void*) &fp, sizeof(PerFlowParamsGlobal), 0, cudaMemcpyHostToDevice);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemcpyToSymbol( ConfigP, (void*) &cp, sizeof(ConfigParams), 0, cudaMemcpyHostToDevice);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemcpyToSymbol( ConstXTalkP, (void*) &cXtP, sizeof(WellsLevelXTalkParamsConst<MAX_WELL_XTALK_SPAN,MAX_WELL_XTALK_SPAN>), 0, cudaMemcpyHostToDevice);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemcpyToSymbol( ConstTraceXTalkP, (void*) &cTlXtP, sizeof(XTalkNeighbourStatsConst<MAX_XTALK_NEIGHBOURS>), 0, cudaMemcpyHostToDevice);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemcpyToSymbol(ConstHistCol, (void*) &histConst, sizeof(HistoryCollectionConst), 0, cudaMemcpyHostToDevice);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemcpyToSymbol( ConstBoundRegP, (void*) &cp, sizeof(ConstantRegParamBounds), 0, cudaMemcpyHostToDevice);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaSetDevice(device);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMalloc(&devPtrLUT, poissTableSize); CUDA_ALLOC_CHECK(devPtrLUT);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemset(devPtrLUT, 0, poissTableSize); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:  cudaMemcpyToSymbol(POISS_APPROX_LUT_CUDA_BASE, &devPtrLUT  , sizeof (float4*)); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/KernelIncludes/DeviceSymbolCopy.cu:    cudaMemcpy(devPtrLUT, &pPoissLUT[i][0], sizeof(float4)*MAX_POISSON_TABLE_ROW, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:#include "cuda_error.h"
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu://#define __CUDA_ARCH__ 350
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:  float correctedTrace[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:  float fval[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:  float tmp_fval[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:  float err[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:  float jac[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:#if __CUDA_ARCH__ >= 300
Analysis/BkgModel/CUDA/KernelIncludes/SingleFlowFitKernels.cu:      float emptyTraceAvg[MAX_COMPRESSED_FRAMES_GPU] = {0};
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h: * RegionParamsGPU.h
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:#ifndef REGIONPARAMSGPU_H_
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:#define REGIONPARAMSGPU_H_
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:#include "cuda_error.h"
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:  int   interpolatedFrames[MAX_UNCOMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:  float interpolatedMult[MAX_UNCOMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:  float interpolatedDiv[MAX_UNCOMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:    printf("ConstantFrameParams\n GPU raw Image frames: %d, uncomp frames: %d, maxBkgFrames: %d \n", rawFrames, uncompFrames, maxCompFrames);
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:    printf("ConstantRegParamBounds GPU tmidNuc(min:%f, max:%f), ratioDrift(min:%f), copyDrift(min:%f, max:%f) \n", 
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:  int fineStart; //[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:  int coarseStart; //[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:  float darkness; //[MAX_NUM_FLOWS_IN_BLOCK_GPU] only [0] was used now single value
Analysis/BkgModel/CUDA/KernelIncludes/DeviceParamDefines.h:#endif /* REGIONPARAMSGPU_H_ */
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:#include "GpuPipelineDefines.h"
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float correctedTrace[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    float fineNucRise[ISIG_SUB_STEPS_SINGLE_FLOW * MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    float modelTrace[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    float err[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    float pdA[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    float pdKmult[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    float pdDeltaTmidNuc[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  __shared__ float smNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU]; 
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  __shared__ float tmpNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float correctedTrace[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float obsTrace[MAX_COMPRESSED_FRAMES_GPU]; // raw traces being written to
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float tmpTrace[MAX_COMPRESSED_FRAMES_GPU]; // raw traces being written to
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float purpleTrace[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float pdTmidNuc[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float pdRDR[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float pdPDR[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float yerr[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    printf("GPU before fitting...start: %d, tmidnuc: %f rdr: %f pdr: %f\n",
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    printf("====> GPU....tid: %d Ampl: %f\n", threadIdx.x, ampl);
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    printf("====> GPU....Before fitting residual: %f\n", curAvgRes);
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:      printf("====GPU REG Fitting...iter:%d, tmidNuc:%f, rdr:%f, pdr:%f\n", iter, tmidNuc, 
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:          printf("===GPU REG Params...iter:%d,delta0:%f,delta1:%f,delta2:%f\n", iter, deltas[0], deltas[1], deltas[2]);
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:          printf("===GPU REG Params...iter:%d,tmidnuc:%f,rdr:%f,pdr:%f,old_residual:%f,new_residual:%f\n", iter, new_tmidnuc, new_ratiodrift, new_copydrift, curRes, new_residual);
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  __shared__ float smNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float purpleTrace[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  __shared__ float smNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  __shared__ float smTmpNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float pdTmidNuc[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float pdRDR[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float pdPDR[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float yerr[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float oldTrace[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float newTrace[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:          printf("===GPU REG Params...iter:%d,delta0:%f,delta1:%f,delta2:%f,lambda:%f\n", iter, deltas[0], deltas[1], deltas[2],lambda);
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:          printf("===GPU REG Params...iter:%d,tmidnuc:%f,rdr:%f,pdr:%f,old_residual:%f,new_residual:%f\n", iter, new_tmidnuc, new_ratiodrift, new_copydrift, curAvgRes, newAvgRes);
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  __shared__ float smNucRise[ISIG_SUB_STEPS_MULTI_FLOW*MAX_COMPRESSED_FRAMES_GPU]; 
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float correctedTrace[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float obsTrace[MAX_COMPRESSED_FRAMES_GPU]; // raw traces being written to
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    printf("GPU before fitting...start: %d, tmidnuc: %f rdr: %f pdr: %f\n",
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:    printf("====> GPU....tid: %d Ampl: %f\n", threadIdx.x, ampl);
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  float AmplEst[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  const short* obsTracePtr[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  const float* emptyTracePtr[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/RegionalFittingKernels.cu:  const PerNucParamsRegion* nucRegParamsPtr[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/TraceLevelXTalk.cu:#include "cuda_error.h"
Analysis/BkgModel/CUDA/KernelIncludes/TraceLevelXTalk.cu:  float incorp_rise[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/TraceLevelXTalk.cu:  float lost_hydrogen[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/TraceLevelXTalk.cu:  float bulk_signal[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/TraceLevelXTalk.cu:  float xtalk[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/TraceLevelXTalk.cu:  float incorp_rise[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/TraceLevelXTalk.cu:  float lost_hydrogen[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/TraceLevelXTalk.cu:  float bulk_signal[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/TraceLevelXTalk.cu:  float xtalk[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/KernelIncludes/PostFitKernels.cu:         //printf("GPU %d %d, %f ,%d %d, %d %d, %f, %d %d, %d %f %f\n", rx,ry, *AmplCpy, c, r, tx, ty, default_signal, lx, ly, phase, ConstXTalkP.coeff(lx,ly,phase), sum);
Analysis/BkgModel/CUDA/KernelIncludes/PostFitKernels.cu:          //printf("GPU %d %d, %f ,%d %d, %d %d, %f, %d %d, %d %f %f\n",rx,ry, *AmplCpy, c, r, tx,ty , amplCopy, lx, ly, phase, ConstXTalkP.coeff(lx,ly,phase), sum);
Analysis/BkgModel/CUDA/KernelIncludes/PostFitKernels.cu:   // printf("regionSum GPU: %f / %d\n", sigSum, ImgRegP.getRegSize(regId) );
Analysis/BkgModel/CUDA/KernelIncludes/ConstantSymbolDeclare.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/KernelIncludes/ConstantSymbolDeclare.h:__constant__ static float4 * POISS_APPROX_LUT_CUDA_BASE;
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:#include "cuda_error.h"
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:#include "GpuMultiFlowFitControl.h"
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  cudaSimpleStreamExecutionUnit(res, item),
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  catch(cudaException &e)
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    cout << "GPUTracesAndParameters NewLayout("<< _myJob.getImgWidth()<<"," << _myJob.getImgHeight()<<",216,224,"<< _myJob.getMaxFrames() << ");" << endl;
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    //static GPUTracesAndParameters NewLayout(216,224,216,224,_myJob.getNumFrames());
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    static GPUTracesAndParameters NewLayout(_myJob.getImgWidth(),_myJob.getImgHeight(),216,224,_myJob.getMaxFrames());
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  catch(cudaException &e)
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:// ASYNC CUDA FUNCTIONS, KERNEL EXECUTION AND DATA HANDLING
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:        //static GPUTracesAndParameters NewLayout(216,224,216,224,_myJob.getNumFrames());
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:        //static GPUTracesAndParameters NewLayoutResults(_myJob.getImgWidth(),_myJob.getImgHeight(),216,224,_myJob.getMaxFrames());
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  catch(cudaException &e)
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    StreamingKernels::copyFittingConstParamAsync(_hConstP.getPtr(), getStreamId() ,_stream);CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:      StreamingKernels::copyXtalkConstParamAsync(_hConstXtalkP.getPtr(), getStreamId() ,_stream);CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  catch(cudaException &e)
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  //cudaMemcpyAsync( _h_pBeadParams, _d_pBeadParams, _copyOutSize , cudaMemcpyDeviceToHost, _stream); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  cudaMemcpy( _h_pBeadParams, _d_pBeadParams, _copyOutSize , cudaMemcpyDeviceToHost); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  cout << "CUDA SingleFitStream active and resources requested dev = "<< devAlloc/(1024.0*1024) << "MB ("<< (int)(deviceFraction*100)<<"%) host = " << hostAlloc/(1024.0*1024) << "MB" <<endl;
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  cudaResourcePool::requestDeviceMemory(devAlloc);
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  cudaResourcePool::requestHostMemory(hostAlloc);
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  if(GpuMultiFlowFitControl::doGPUTraceLevelXtalk()){
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  if(GpuMultiFlowFitControl::doGPUTraceLevelXtalk()){
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:  cout << "CUDA SingleFitStream SETTINGS: blocksize = " << _bpb  << " l1setting = " ;
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    cout << "cudaFuncCachePreferEqual" << endl;;
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    cout << "cudaFuncCachePreferShared" <<endl;
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    cout << "cudaFuncCachePreferL1" << endl;
Analysis/BkgModel/CUDA/SingleFitStream.old.cu:    cout << "GPU specific default" << endl;
Analysis/BkgModel/CUDA/MasterKernel.h:#include "GpuPipelineDefines.h"
Analysis/BkgModel/CUDA/SingleFitStream.cu:#include "cuda_error.h"
Analysis/BkgModel/CUDA/SingleFitStream.cu:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/SingleFitStream.cu:#include "GpuMultiFlowFitControl.h"
Analysis/BkgModel/CUDA/SingleFitStream.cu:      cudaSimpleStreamExecutionUnit(res, item),
Analysis/BkgModel/CUDA/SingleFitStream.cu:  catch(cudaException &e)
Analysis/BkgModel/CUDA/SingleFitStream.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/SingleFitStream.cu:  catch(cudaException &e)
Analysis/BkgModel/CUDA/SingleFitStream.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/SingleFitStream.cu:// ASYNC CUDA FUNCTIONS, KERNEL EXECUTION AND DATA HANDLING
Analysis/BkgModel/CUDA/SingleFitStream.cu:    catch(cudaException &e)
Analysis/BkgModel/CUDA/SingleFitStream.cu:      throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/SingleFitStream.cu:    StreamingKernels::copyFittingConstParamAsync(_hConstP.getPtr(), getStreamId() ,_stream);CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/SingleFitStream.cu:      StreamingKernels::copyXtalkConstParamAsync(_hConstXtalkP.getPtr(), getStreamId() ,_stream);CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/SingleFitStream.cu:  catch(cudaException &e)
Analysis/BkgModel/CUDA/SingleFitStream.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/SingleFitStream.cu:         cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/SingleFitStream.cu:      cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/SingleFitStream.cu:      cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/SingleFitStream.cu:  //cudaMemcpyAsync( _h_pBeadParams, _d_pBeadParams, _copyOutSize , cudaMemcpyDeviceToHost, _stream); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/SingleFitStream.cu:  cudaMemcpy( _h_pBeadParams, _d_pBeadParams, _copyOutSize , cudaMemcpyDeviceToHost); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/SingleFitStream.cu:  cout << "CUDA: SingleFitStream active and resources requested dev = "<< devAlloc/(1024.0*1024) << "MB ("<< (int)(deviceFraction*100)<<"%) host = " << hostAlloc/(1024.0*1024) << "MB" <<endl;
Analysis/BkgModel/CUDA/SingleFitStream.cu:  cudaResourcePool::requestDeviceMemory(devAlloc);
Analysis/BkgModel/CUDA/SingleFitStream.cu:  cudaResourcePool::requestHostMemory(hostAlloc);
Analysis/BkgModel/CUDA/SingleFitStream.cu:  if(GpuMultiFlowFitControl::doGPUTraceLevelXtalk()){
Analysis/BkgModel/CUDA/SingleFitStream.cu:  if(GpuMultiFlowFitControl::doGPUTraceLevelXtalk()){
Analysis/BkgModel/CUDA/SingleFitStream.cu:  cout << "CUDA: SingleFitStream SETTINGS: blocksize = " << _bpb  << " l1setting = " ;
Analysis/BkgModel/CUDA/SingleFitStream.cu:      cout << "cudaFuncCachePreferEqual" << endl;;
Analysis/BkgModel/CUDA/SingleFitStream.cu:      cout << "cudaFuncCachePreferShared" <<endl;
Analysis/BkgModel/CUDA/SingleFitStream.cu:      cout << "cudaFuncCachePreferL1" << endl;
Analysis/BkgModel/CUDA/SingleFitStream.cu:      cout << "GPU specific default" << endl;
Analysis/BkgModel/CUDA/dumper.cpp:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/dumper.cpp:size_t DumpBuffer::addCudaData(void *devData, size_t bytes)
Analysis/BkgModel/CUDA/dumper.cpp:  cudaMemcpy((void*)writePtr, devData, bytes, cudaMemcpyDeviceToHost );
Analysis/BkgModel/CUDA/dumper.cpp:bool DumpBuffer::CompareCuda(float * devData, float threshold, OutputFormat output)
Analysis/BkgModel/CUDA/dumper.cpp:  temp.addCudaData(devData,getSize());
Analysis/BkgModel/CUDA/dumper.cpp:bool DumpWrapper<T>::CompareCuda(float * devData, float * hostData, size_t size, float threshold, OutputFormat output)
Analysis/BkgModel/CUDA/dumper.cpp:  DumpBuffer tmpCuda(size,"CudaBuffer");
Analysis/BkgModel/CUDA/dumper.cpp:  tmpCuda.addCudaData(devData,size);
Analysis/BkgModel/CUDA/dumper.cpp:  return tmpCuda.CompareData(hostData,threshold, output);
Analysis/BkgModel/CUDA/MultiFitStream.cu:#include "cuda_error.h"
Analysis/BkgModel/CUDA/MultiFitStream.cu:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/MultiFitStream.cu:#include "GpuMultiFlowFitControl.h"
Analysis/BkgModel/CUDA/MultiFitStream.cu:int SimpleMultiFitStream::_l1type = -1;  // 0: SM=L1, 1: SM>L1,  2: L1>SM, -1:GPU default
Analysis/BkgModel/CUDA/MultiFitStream.cu:  // 0: SM=L1, 1: SM>L1,  2: L1>SM, -1:GPU default
Analysis/BkgModel/CUDA/MultiFitStream.cu:  cudaSimpleStreamExecutionUnit(res, item),
Analysis/BkgModel/CUDA/MultiFitStream.cu:  for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i)
Analysis/BkgModel/CUDA/MultiFitStream.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/MultiFitStream.cu:    for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i)
Analysis/BkgModel/CUDA/MultiFitStream.cu:    // we need a specific struct describing this config for this well fit for GPU
Analysis/BkgModel/CUDA/MultiFitStream.cu:  catch (cudaException &e)
Analysis/BkgModel/CUDA/MultiFitStream.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/MultiFitStream.cu:  catch (cudaException &e)
Analysis/BkgModel/CUDA/MultiFitStream.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/MultiFitStream.cu:  catch (cudaException &e)
Analysis/BkgModel/CUDA/MultiFitStream.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/MultiFitStream.cu:// ASYNC CUDA FUNCTIONS, KERNEL EXECUTION AND DATA HANDLING
Analysis/BkgModel/CUDA/MultiFitStream.cu:  //cout << "Copy data to GPU" << endl;
Analysis/BkgModel/CUDA/MultiFitStream.cu:    //  copyMultiFlowFitConstParamAsync(_HostConstP, getStreamId(),_stream);CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/MultiFitStream.cu:    StreamingKernels::copyFittingConstParamAsync(_hConstP.getPtr(), getStreamId(),_stream);CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/MultiFitStream.cu:  catch(cudaException &e)
Analysis/BkgModel/CUDA/MultiFitStream.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/MultiFitStream.cu:  //cout << "Copy data to GPU" << endl;
Analysis/BkgModel/CUDA/MultiFitStream.cu:  catch(cudaException &e)
Analysis/BkgModel/CUDA/MultiFitStream.cu:    throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/BkgModel/CUDA/MultiFitStream.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/MultiFitStream.cu:   CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/MultiFitStream.cu://  cudaThreadSynchronize();CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/MultiFitStream.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/MultiFitStream.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/MultiFitStream.cu:    _myJob.KeyNormalize();   // temporary call to key normalize till we put it into a GPU kernel
Analysis/BkgModel/CUDA/MultiFitStream.cu:    if(_fitNum < CUDA_MULTIFLOW_NUM_FIT){
Analysis/BkgModel/CUDA/MultiFitStream.cu:  cout << "CUDA: MultiFitStream active and resources requested dev = "<< devAlloc/(1024.0*1024) << "MB ("<< (int)(deviceFraction*100)<<"%) host = " << hostAlloc/(1024.0*1024) << "MB" << endl;
Analysis/BkgModel/CUDA/MultiFitStream.cu:  cudaResourcePool::requestDeviceMemory(devAlloc);
Analysis/BkgModel/CUDA/MultiFitStream.cu:  cudaResourcePool::requestHostMemory(hostAlloc);
Analysis/BkgModel/CUDA/MultiFitStream.cu:  for (int i=0; i<CUDA_MULTIFLOW_NUM_FIT; ++i)
Analysis/BkgModel/CUDA/MultiFitStream.cu:  cout << "CUDA: MultiFitStream SETTINGS: blocksize = " << _bpb << " l1setting = " ;
Analysis/BkgModel/CUDA/MultiFitStream.cu:      cout << "cudaFuncCachePreferEqual" << endl;;
Analysis/BkgModel/CUDA/MultiFitStream.cu:      cout << "cudaFuncCachePreferShared" <<endl;
Analysis/BkgModel/CUDA/MultiFitStream.cu:      cout << "cudaFuncCachePreferL1" << endl;
Analysis/BkgModel/CUDA/MultiFitStream.cu:     cout << " GPU specific default" << endl;;
Analysis/BkgModel/CUDA/MultiFitStream.cu:  cout << "CUDA: PartialDerivative SETTINGS: blocksize = " << _bpbPartialD << " l1setting = ";
Analysis/BkgModel/CUDA/MultiFitStream.cu:      cout << "cudaFuncCachePreferEqual" << endl;;
Analysis/BkgModel/CUDA/MultiFitStream.cu:      cout << "cudaFuncCachePreferShared" <<endl;
Analysis/BkgModel/CUDA/MultiFitStream.cu:      cout << "cudaFuncCachePreferL1" << endl;
Analysis/BkgModel/CUDA/MultiFitStream.cu:     cout << "GPU specific default" << endl;
Analysis/BkgModel/CUDA/MultiFitStream.cu:      StreamingKernels::copyFittingConstParamAsync(tmpConstP, getStreamId(),_stream);CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.h:#ifndef GPUMULTIFLOWFITMATRIXCONFIG_H
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.h:#define GPUMULTIFLOWFITMATRIXCONFIG_H
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.h:class GpuMultiFlowFitMatrixConfig
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.h:    GpuMultiFlowFitMatrixConfig(const std::vector<fit_descriptor>& fds, CpuStep*, int maxSteps, int flow_key, int flow_block_size);
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.h:    ~GpuMultiFlowFitMatrixConfig();
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.h:#endif // GPUMULTIFLOWFITMATRIXCONFIG_H
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:#include "cudaStreamTemplate.h"
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:#include "cuda_error.h"
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:int TemplateStream::_l1type = -1;  // 0: SM=L1, 1: SM>L1,  2: L1>SM, -1:GPU default
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:  // 0: SM=L1, 1: SM>L1,  2: L1>SM, -1:GPU default
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:TemplateStream::TemplateStream(streamResources * res, WorkerInfoQueueItem item ) : cudaSimpleStreamExecutionUnit(res, item)
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:   CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:  //any cuda calls that are not async have to happen here to keep things clean
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:  //the following 3 calls can only contain async cuda calls!!!
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:// trigger async copies, no more sync cuda calls from this [point on until handle results
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:  //cout << "Copy data to GPU" << endl;
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:  cout << "CUDA TemplateStream SETTINGS: blocksize " << _bpb << " l1setting " << _l1type;
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:      cout << " (cudaFuncCachePreferEqual" << endl;;
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:      cout << " (cudaFuncCachePreferShared)" <<endl;
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:      cout << " (cudaFuncCachePreferL1)" << endl;
Analysis/BkgModel/CUDA/cudaStreamTemplate.cu:     cout << " GPU specific default" << endl;;
Analysis/BkgModel/CUDA/StreamingKernels.cu:#include "CudaUtils.h" // for cuda < 5.0 this has to be included ONLY here!
Analysis/BkgModel/CUDA/StreamingKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/StreamingKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float fval[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float tmp_fval[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float fval[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float tmp_fval[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float fval[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float tmp_fval[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float fval[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float tmp_fval[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/StreamingKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/StreamingKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/StreamingKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/StreamingKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/StreamingKernels.cu:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/StreamingKernels.cu:  __shared__ float smBuffer[MAX_UNCOMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/StreamingKernels.cu:  for (int i=0; i<(MAX_UNCOMPRESSED_FRAMES_GPU); ++i) {
Analysis/BkgModel/CUDA/StreamingKernels.cu:  CpuStep* psteps, // we need a specific struct describing this config for this well fit for GPU
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float fval_L1[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float* ptrL = POISS_APPROX_TABLE_CUDA_BASE + MAX_POISSON_TABLE_ROW * ((event == 0)?(event):(event-1)) ;
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float* ptrR = POISS_APPROX_TABLE_CUDA_BASE + MAX_POISSON_TABLE_ROW * ((event < maxEvents-1)?(event):(event-1)) ;
Analysis/BkgModel/CUDA/StreamingKernels.cu:  float4* ptrLUT =  POISS_APPROX_LUT_CUDA_BASE + event * MAX_POISSON_TABLE_ROW + offset;
Analysis/BkgModel/CUDA/StreamingKernels.cu:void StreamingKernels::copyFittingConstParamAsync(ConstParams* ptr, int offset, cudaStream_t stream)
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaMemcpyToSymbolAsync ( CP, ptr, sizeof(ConstParams), offset*sizeof(ConstParams),cudaMemcpyHostToDevice, stream);
Analysis/BkgModel/CUDA/StreamingKernels.cu:void StreamingKernels::copyXtalkConstParamAsync(ConstXtalkParams* ptr, int offset, cudaStream_t stream)
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaMemcpyToSymbolAsync ( CP_XTALKPARAMS, ptr, sizeof(ConstXtalkParams), offset*sizeof(ConstXtalkParams),cudaMemcpyHostToDevice, stream);
Analysis/BkgModel/CUDA/StreamingKernels.cu:void  StreamingKernels::PerFlowGaussNewtonFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowGaussNewtonFit_k, cudaFuncCachePreferShared);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowGaussNewtonFit_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowGaussNewtonFit_k, cudaFuncCachePreferEqual);
Analysis/BkgModel/CUDA/StreamingKernels.cu:void  StreamingKernels::PerFlowHybridFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowHybridFit_k, cudaFuncCachePreferShared);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowHybridFit_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowHybridFit_k, cudaFuncCachePreferEqual);
Analysis/BkgModel/CUDA/StreamingKernels.cu:void  StreamingKernels::PerFlowLevMarFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowLevMarFit_k, cudaFuncCachePreferShared);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowLevMarFit_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowLevMarFit_k, cudaFuncCachePreferEqual);
Analysis/BkgModel/CUDA/StreamingKernels.cu:void  StreamingKernels::PerFlowRelaxKmultGaussNewtonFit(int l1type, dim3 grid, dim3 block, int smem, cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowRelaxedKmultGaussNewtonFit_k, cudaFuncCachePreferShared);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowRelaxedKmultGaussNewtonFit_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(PerFlowRelaxedKmultGaussNewtonFit_k, cudaFuncCachePreferEqual);
Analysis/BkgModel/CUDA/StreamingKernels.cu:void StreamingKernels::PreSingleFitProcessing(dim3 grid, dim3 block, int smem, cudaStream_t stream,// Here FL stands for flows
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:  CpuStep* psteps, // we need a specific struct describing this config for this well fit for GPU
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k, cudaFuncCachePreferShared);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(ComputePartialDerivativesForMultiFlowFitForWellsFlowByFlow_k, cudaFuncCachePreferEqual);
Analysis/BkgModel/CUDA/StreamingKernels.cu:    psteps, // we need a specific struct describing this config for this well fit for GPU
Analysis/BkgModel/CUDA/StreamingKernels.cu:void StreamingKernels::BuildMatrix( dim3 grid, dim3 block, int smem, cudaStream_t stream, 
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(BuildMatrixVec4_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(BuildMatrixVec2_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(BuildMatrix_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/StreamingKernels.cu:void StreamingKernels::MultiFlowLevMarFit(int l1type,  dim3 grid, dim3 block, int smem, cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(MultiFlowLevMarFit_k, cudaFuncCachePreferShared);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(MultiFlowLevMarFit_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/StreamingKernels.cu:      cudaFuncSetCacheConfig(MultiFlowLevMarFit_k, cudaFuncCachePreferEqual);
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaStream_t stream,// Here FL stands for flows
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaStream_t stream,// Here FL stands for flows
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaStream_t stream,
Analysis/BkgModel/CUDA/StreamingKernels.cu:void StreamingKernels::transposeData(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, float *source, int width, int height)
Analysis/BkgModel/CUDA/StreamingKernels.cu:void StreamingKernels::transposeDataToFloat(dim3 grid, dim3 block, int smem, cudaStream_t stream,float *dest, FG_BUFFER_TYPE *source, int width, int height)
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaSetDevice(device);
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaMalloc(&devPtr, poissTableSize); CUDA_ALLOC_CHECK(devPtr);
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaMemcpyToSymbol(POISS_APPROX_TABLE_CUDA_BASE , &devPtr  , sizeof (float*)); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/StreamingKernels.cu:    cudaMemcpy(devPtr, poiss_cdf[i], sizeof(float)*MAX_POISSON_TABLE_ROW, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/StreamingKernels.cu:#ifndef USE_CUDA_ERF
Analysis/BkgModel/CUDA/StreamingKernels.cu:    cudaMemcpyToSymbol (ERF_APPROX_TABLE_CUDA, ERF_APPROX_TABLE, sizeof (ERF_APPROX_TABLE)); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaSetDevice(device);
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaMalloc(&devPtrLUT, poissTableSize); CUDA_ALLOC_CHECK(devPtrLUT);
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaMemset(devPtrLUT, 0, poissTableSize); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaMemcpyToSymbol(POISS_APPROX_LUT_CUDA_BASE, &devPtrLUT  , sizeof (float4*)); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/StreamingKernels.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/StreamingKernels.cu:    cudaMemcpy(devPtrLUT, &pPoissLUT[i][0], sizeof(float4)*MAX_POISSON_TABLE_ROW, cudaMemcpyHostToDevice ); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaSetDevice(device);
Analysis/BkgModel/CUDA/StreamingKernels.cu:  cudaMemcpyFromSymbol (&basepointer,  POISS_APPROX_TABLE_CUDA_BASE , sizeof (float*)); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/StreamingKernels.cu:    cudaFree(basepointer); CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/MathOptimCuda.h:#ifndef MATHOPTIMCUDA_H
Analysis/BkgModel/CUDA/MathOptimCuda.h:#define MATHOPTIMCUDA_H
Analysis/BkgModel/CUDA/MathOptimCuda.h:// This file simply exposes the lookup tables in a header so the CUDA files can find them
Analysis/BkgModel/CUDA/MathOptimCuda.h:#endif // MATHOPTIMCUDA_H
Analysis/BkgModel/CUDA/BkgGpuPipeline.h: * BkgGpuPipeline.h
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:#ifndef BKGGPUPIPELINE_H_
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:#define BKGGPUPIPELINE_H_
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:#include "GpuPipelineDefines.h"
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:class cudaComputeVersion{
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:  cudaComputeVersion(int Major, int Minor){
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:  bool operator== (const cudaComputeVersion &other) const {
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:  bool operator> (const cudaComputeVersion &other) const {
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:  bool operator!= (const cudaComputeVersion &other) const {
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:  bool operator<= (const cudaComputeVersion &other) const {
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:  bool operator< (const cudaComputeVersion &other) const {
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:  bool operator>= (const cudaComputeVersion &other) const{
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:    cout << "CUDA: Device Memory allocated: " << accumBytes / (1024.0* 1024.0) << " MB " << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:    cout << "CUDA: Special Device Buffers Memory allocated: " << accumBytes / (1024.0* 1024.0) << " MB " << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:    cout << "CUDA: Host Trace Level XTalk Buffers Memory allocated: " << accumBytes / (1024.0* 1024.0) << " MB " << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:    cout << "CUDA: Device Trace Level XTalk Buffers Memory allocated: " << accumBytes / (1024.0* 1024.0) << " MB " << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:      cout << "CUDA: Device Trace Level XTalk Buffers increased by Dynamic Element from " << accumBytes / (1024.0* 1024.0) << " MB ";
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:    cout << "CUDA: Host Buffers Memory allocated: " << accumBytes / (1024.0* 1024.0) << " MB " << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:class GPUResultsBuffer
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:  GPUResultsBuffer(const ImgRegParams &ImgP, int numBuffers):
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:class BkgGpuPipeline
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:  BkgGpuPipeline(BkgModelWorkInfo* bkinfo, int deviceId, HistoryCollection * histCol = NULL);  //ToDO: add stream and device info
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:  ~BkgGpuPipeline();
Analysis/BkgModel/CUDA/BkgGpuPipeline.h:#endif /* BKGGPUPIPELINE_H_ */
Analysis/BkgModel/CUDA/ObsoleteCuda.h:#ifndef OBSOLETECUDA_H
Analysis/BkgModel/CUDA/ObsoleteCuda.h:#define OBSOLETECUDA_H
Analysis/BkgModel/CUDA/ObsoleteCuda.h://@TODO:  placeholder for things that CUDA believes it needs declared that the CPU code has evolved away from
Analysis/BkgModel/CUDA/ObsoleteCuda.h:// Lets us compile without having the CUDA code explode
Analysis/BkgModel/CUDA/ObsoleteCuda.h:#endif // OBSOLETECUDA_H
Analysis/BkgModel/CUDA/SingleFitStream.h:// cuda
Analysis/BkgModel/CUDA/SingleFitStream.h:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/SingleFitStream.h:#include "cuda_error.h"
Analysis/BkgModel/CUDA/SingleFitStream.h:class SimpleSingleFitStream : public cudaSimpleStreamExecutionUnit
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu: * BkgGpuPipeline.cu
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:#include "GpuPipelineDefines.h"
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:#include "BkgGpuPipeline.h"
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:BkgGpuPipeline::BkgGpuPipeline(
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaSetDevice( devId );
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaDeviceProp cuda_props;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaGetDeviceProperties( &cuda_props, devId );
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: Initiating Flow by Flow Pipeline on Device: "<< devId << "( " << cuda_props.name  << " v"<< cuda_props.major <<"."<< cuda_props.minor << ")" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::setSpatialParams()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: Chip offset x:" << loc->chip_offset_x   << " y:" <<  loc->chip_offset_y  << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::InitPipeline()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  }catch(cudaException &e){
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    throw cudaAllocationError(e.getCudaError(), __FILE__, __LINE__);
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:bool BkgGpuPipeline::firstFlow(){
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:size_t BkgGpuPipeline::checkAvailableDevMem()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaMemGetInfo( &free_byte, &total_byte );
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA " << devId << ": GPU memory usage: used = " << (total_byte-free_byte)/divMB<< ", free = " << free_byte/divMB<< " MB, total = "<< total_byte/divMB<<" MB" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::PrepareInputsForSetupKernel()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::PrepareSampleCollection()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA WARNING: BkgGpuPipeline: No HistoryCollection found! Creating new HistoryCollection initialized with latest available Regional Params!" <<endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::InitPersistentData()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: InitPersistentData: num Time-Compressed-Frames Per Region:" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::InitXTalk(){
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:BkgGpuPipeline::~BkgGpuPipeline()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: Starting cleanup flow by flow GPU pipeline" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: Cleanup flow by flow GPU pipeline completed" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::PerFlowDataUpdate(BkgModelWorkInfo* pbkinfo)
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:dim3 BkgGpuPipeline::matchThreadBlocksToRegionSize(int bx, int by)
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA WARNING: requested region height of " << ImgP.getRegH() << " does not allow optimal GPU threadblock height of 4 warps! Threadblock height corrected to " << correctBy << ". For optimal performance please choose a region height of a multiple of " << by << "." << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::ExecuteT0AvgNumLBeadKernel()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: executing GenerateT0AvgAndNumLBeads Kernel grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: GenerateT0AvgAndNumLBeads_New finalize" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: num live beads per region: " << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteT0AvgNumLBeadKernel: T0 avg per region: " << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  //cout << "CUDA: BkgGpuPipeline: BkgGpuPipeline: std deviation per region: "<< endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: Number of samples for regional fitting per Region:" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: starting offset for samples per Row (last entry is num samples)" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::ExecuteGenerateBeadTrace()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cudaFuncSetCacheConfig(GenerateAllBeadTraceEmptyFromMeta_k, cudaFuncCachePreferEqual);
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: CacheSetting: GenerateAllBeadTraceEmptyFromMeta_k cudaFuncCachePreferEqual" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cudaFuncSetCacheConfig(GenerateAllBeadTraceEmptyFromMeta_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: CacheSetting: GenerateAllBeadTraceEmptyFromMeta_k cudaFuncCachePreferL1" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cudaFuncSetCacheConfig(GenerateAllBeadTraceEmptyFromMeta_k, cudaFuncCachePreferShared);
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: CacheSetting: GenerateAllBeadTraceEmptyFromMeta_k cudaFuncCachePreferShared" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: executing GenerateAllBeadTraceFromMeta_k Kernel grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  // print bead traces post GPU GenBeadTracesKernel
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: GenerateAllBeadTraceEmptyFromMeta_k finalize" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: executing ReduceEmptyAverage_k Kernel grid(" << gridER.x << "," << gridER.y  << "), block(" << blockER.x << "," << blockER.y <<"), smem("<< smem <<")" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: ReduceEmptyAverage_k finalize" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: Region State: " << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteGenerateBeadTrace: Average Empty Traces:" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cout <<"DEBUG GPU EmptytraceAvg Current," << regId <<"," << GpFP.getRealFnum() << "," << HostEmptyTraceAvg->getCSVatReg<float>(regId,0,0,0,nf) << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::ExecuteTraceLevelXTalk()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: executing SimpleXTalkNeighbourContribution grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: SimpleXTalkNeighbourContribution finalize" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:        cout << "CUDA: Per Bead XTalk Contribution " <<endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: executing GenericXTalkAndNeighbourAccumulation grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")"<< endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: GenericXTalkAndNeighbourAccumulation finalize" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: executing GenericXTalkAccumulation grid(" << accumGrid.x << "," << accumGrid.y  << "), block(" << accumBlock.x << "," << accumBlock.y <<"), smem(0)" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: GenericXTalkAccumulation finalize" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteTraceLevelXTalk: executing SimpleXTalkNeighbourContributionAndAccumulation with: block(" << block.x << "," << block.y <<"), grid(" << grid.x << "," << grid.y  << ")  and smem: "<< smem << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: num GenericXTalkTraces per Region: " << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: Per Bead XTalk " <<endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::ExecuteSingleFlowFit()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cudaFuncSetCacheConfig(ExecuteThreadBlockPerRegion2DBlocksDense, cudaFuncCachePreferEqual);
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cout << "CUDA: BkgGpuPipeline: ExecuteSingleFlowFit: CacheSetting: ExecuteThreadBlockPerRegion2DBlocks cudaFuncCachePreferEqual" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cudaFuncSetCacheConfig(ExecuteThreadBlockPerRegion2DBlocksDense, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cout << "CUDA: BkgGpuPipeline: ExecuteSingleFlowFit: CacheSetting: ExecuteThreadBlockPerRegion2DBlocks cudaFuncCachePreferL1" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cudaFuncSetCacheConfig(ExecuteThreadBlockPerRegion2DBlocksDense, cudaFuncCachePreferShared);
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      cout << "CUDA: BkgGpuPipeline: ExecuteSingleFlowFit: CacheSetting: ExecuteThreadBlockPerRegion2DBlocks cudaFuncCachePreferShared" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteSingleFlowFit: executing ExecuteThreadBlockPerRegion2DBlocks Kernel (SingleFlowFit) grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteSingleFlowFit: ExecuteThreadBlockPerRegion2DBlocksDense finalize" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: Iteration counter: " << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:   cout << "CUDA: Region: " << reg << " numWarps: " << numWarpsPerRegRow*ImgP.getRegH(reg) <<  " max iter: ";
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu: cout << "CUDA: Max iterations within all " <<  numwarps << " warps: ";
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::HandleResults(RingBuffer<float> * ringbuffer)
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: Reinjecting results for flowblock containing flows "<< getFlowP().getRealFnum() - flowBlockSize << " to " << getFlowP().getRealFnum() << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: waiting on CPU Q ... ";
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::InitRegionalParamsAtFirstFlow()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      std::cout << "CUDA: BkgGpuPipeline: Starting Flow: " << startFlowNum << std::endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:            cout << "CUDA: BkgGpuPipeline: InitOldRegionalParamsAtFirstFlow: DEBUG regId " << regId << " PerFlowRegionParams,";
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::ReadRegionDataFromFileForBlockOf20()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:        if(i==0) cout << "CUDA: BkgGpuPipeline: ReadRegionDataFromFileForBlockOf20: updating GPU emphasis and nucRise" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ReadRegionDataFromFileForBlockOf20: updating Emphasis and NucRise on device for next block of " << flowBlockSize << " flows." << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::ExecuteRegionalFitting() {
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteRegionalFitting: executing PerformMultiFlowRegionalFitting Kernel grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem(0)" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteRegionalFitting: PerformMultiFlowRegionalFitting finalized" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::PrepareForRegionalFitting()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      if(i==0) cout << "CUDA: BkgGpuPipeline: PrepareForRegionalFitting: updating GPU crude emphasis" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::PrepareForSingleFlowFit()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      if(i==0) cout << "CUDA: BkgGpuPipeline: PrepareForSingleFlowFit: updating GPU fine emphasis" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::HandleRegionalFittingResults()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:      if(i==0) cout << "CUDA: BkgGpuPipeline: HandleRegionalFittingResults: updating reg params on host" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::ExecuteCrudeEmphasisGeneration() {
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteCrudeEmphasisGeneration: executing emphasis generation Kernel grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteCrudeEmphasisGeneration: GenerateEmphasis finalized" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::ExecuteFineEmphasisGeneration() {
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteFineEmphasisGeneration: executing emphasis generation Kernel grid(" << grid.x << "," << grid.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA: BkgGpuPipeline: ExecuteFineEmphasisGeneration: GenerateEmphasis finalized" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::ExecutePostFitSteps() {
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ExecutePostFitSteps: executing Wells XTalk Update Signal Map Kernel grid(" << gridBlockPerRegion.x << "," << gridBlockPerRegion.y  << "), block(" << block.x << "," << block.y <<"), smem("<< smem <<")" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ExecutePostFitSteps: UpdateSignalMap_k finalized" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ExecutePostFitSteps: executing post processing and corrections kernel grid(" << gridWarpPerRow.x << "," << gridWarpPerRow.y  << "), block(" << block.x << "," << block.y <<"), smem(0)" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cudaFuncSetCacheConfig(PostProcessingCorrections_k, cudaFuncCachePreferL1);
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cudaDeviceSynchronize();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA: BkgGpuPipeline: ExecutePostFitSteps: ProtonXTalk_k finalized" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::ApplyClonalFilter()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:        cout << "CUDA: Applying PolyClonal Filter after Flow: " << GpFP.getRealFnum() << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::CopySerializationDataFromDeviceToHost()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::DebugOutputDeviceBuffers(){
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA DEBUG: "; ImgP.print();
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA DEBUG:" << GpFP << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA DEBUG:" << ConfP << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA DEBUG:" << ConstFrmP << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  cout << "CUDA DEBUG:" << ConstGP << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA DEBUG:" <<  regId  << "," << Host->ConstRegP.refAtReg(regId) << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA DEBUG:" <<  regId  << "," << pHistCol->getHostPerFlowRegParams().refAtReg(regId) << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA DEBUG Sample traces:" << endl;
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:    cout << "CUDA DEBUG Empty Trace Avg:";
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::getDataForRawWells(RingBuffer<float> * ringbuffer)
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::getDataForPostFitStepsOnHost()
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::printBkgModelMaskEnum(){
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  std::cout << "CUDA: BkgModelMask flags: "<< std::endl
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:void BkgGpuPipeline::printRegionStateMask(){
Analysis/BkgModel/CUDA/BkgGpuPipeline.cu:  std::cout << "CUDA: BkgModelMask flags: "<< std::endl
Analysis/BkgModel/CUDA/CudaUtils.h:#ifndef CUDAUTILS_H
Analysis/BkgModel/CUDA/CudaUtils.h:#define CUDAUTILS_H
Analysis/BkgModel/CUDA/CudaUtils.h:#include "CudaConstDeclare.h"
Analysis/BkgModel/CUDA/CudaUtils.h:#ifdef USE_CUDA_ERF
Analysis/BkgModel/CUDA/CudaUtils.h:    ret = (1 - frac) * ERF_APPROX_TABLE_CUDA[left] + frac * ERF_APPROX_TABLE_CUDA[right];
Analysis/BkgModel/CUDA/CudaUtils.h:      ret = ERF_APPROX_TABLE_CUDA[0];
Analysis/BkgModel/CUDA/CudaUtils.h:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/CudaUtils.h:#if __CUDA_ARCH__ >= 350
Analysis/BkgModel/CUDA/CudaUtils.h:  const float* ptr = POISS_APPROX_TABLE_CUDA_BASE + n * MAX_POISSON_TABLE_ROW;
Analysis/BkgModel/CUDA/CudaUtils.h:  const float4* ptr =  POISS_APPROX_LUT_CUDA_BASE + n * MAX_POISSON_TABLE_ROW;
Analysis/BkgModel/CUDA/CudaUtils.h:#endif // CUDAUTILS_H
Analysis/BkgModel/CUDA/cuda_error.h:#ifndef CUDA_ERROR_H
Analysis/BkgModel/CUDA/cuda_error.h:#define CUDA_ERROR_H
Analysis/BkgModel/CUDA/cuda_error.h:#include "CudaException.h"
Analysis/BkgModel/CUDA/cuda_error.h://#define NO_CUDA_DEBUG
Analysis/BkgModel/CUDA/cuda_error.h:#ifndef NO_CUDA_DEBUG
Analysis/BkgModel/CUDA/cuda_error.h:#define CUDA_ERROR_CHECK()                                                                     \
Analysis/BkgModel/CUDA/cuda_error.h:    cudaError_t err = cudaGetLastError();                                                      \
Analysis/BkgModel/CUDA/cuda_error.h:    if ( err != cudaSuccess && err != cudaErrorSetOnActiveProcess ) {                          \
Analysis/BkgModel/CUDA/cuda_error.h:      if(err != cudaErrorMemoryAllocation){                                                    \
Analysis/BkgModel/CUDA/cuda_error.h:        throw cudaExceptionDebug(err,__FILE__, __LINE__);                                      \
Analysis/BkgModel/CUDA/cuda_error.h:      }else throw cudaAllocationError(err, __FILE__, __LINE__);                                                       \
Analysis/BkgModel/CUDA/cuda_error.h:                  << " | ** CUDA ERROR! ** " << std::endl                                      \
Analysis/BkgModel/CUDA/cuda_error.h:                  << " | Msg: " << cudaGetErrorString(err) << std::endl                        \
Analysis/BkgModel/CUDA/cuda_error.h:                  throw cudaExecutionException(err, __FILE__, __LINE__);                      \
Analysis/BkgModel/CUDA/cuda_error.h:      }else throw cudaAllocationError(err, __FILE__, __LINE__);                               \
Analysis/BkgModel/CUDA/cuda_error.h:#define CUDA_ERROR_CHECK() {}
Analysis/BkgModel/CUDA/cuda_error.h:#define CUDA_ALLOC_CHECK(a)  \
Analysis/BkgModel/CUDA/cuda_error.h:   cudaError_t err = cudaGetLastError();                                                      \
Analysis/BkgModel/CUDA/cuda_error.h:    if ( err != cudaSuccess && err != cudaErrorSetOnActiveProcess ) {                          \
Analysis/BkgModel/CUDA/cuda_error.h:      if(err != cudaErrorMemoryAllocation){                                                    \
Analysis/BkgModel/CUDA/cuda_error.h:        throw cudaExceptionDebug(err,__FILE__, __LINE__);                                      \
Analysis/BkgModel/CUDA/cuda_error.h:      }else throw cudaAllocationError(err, __FILE__, __LINE__);                                                       \
Analysis/BkgModel/CUDA/cuda_error.h:        throw cudaAllocationError(cudaErrorInvalidDevicePointer, __FILE__, __LINE__);  \
Analysis/BkgModel/CUDA/cuda_error.h:                  << " | ** CUDA ERROR! ** " << std::endl                                      \
Analysis/BkgModel/CUDA/cuda_error.h:                  << " | Msg: " << cudaGetErrorString(err) << std::endl                        \
Analysis/BkgModel/CUDA/cuda_error.h:/* cuda Error Codes:
Analysis/BkgModel/CUDA/cuda_error.h:  cudaSuccess                           =      0,   ///< No errors
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorMissingConfiguration         =      1,   ///< Missing configuration error
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorMemoryAllocation             =      2,   ///< Memory allocation error
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInitializationError          =      3,   ///< Initialization error
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorLaunchFailure                =      4,   ///< Launch failure
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorPriorLaunchFailure           =      5,   ///< Prior launch failure
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorLaunchTimeout                =      6,   ///< Launch timeout error
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorLaunchOutOfResources         =      7,   ///< Launch out of resources error
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidDeviceFunction        =      8,   ///< Invalid device function
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidConfiguration         =      9,   ///< Invalid configuration
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidDevice                =     10,   ///< Invalid device
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidValue                 =     11,   ///< Invalid value
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidPitchValue            =     12,   ///< Invalid pitch value
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidSymbol                =     13,   ///< Invalid symbol
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorMapBufferObjectFailed        =     14,   ///< Map buffer object failed
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorUnmapBufferObjectFailed      =     15,   ///< Unmap buffer object failed
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidHostPointer           =     16,   ///< Invalid host pointer
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidDevicePointer         =     17,   ///< Invalid device pointer
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidTexture               =     18,   ///< Invalid texture
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidTextureBinding        =     19,   ///< Invalid texture binding
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidChannelDescriptor     =     20,   ///< Invalid channel descriptor
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidMemcpyDirection       =     21,   ///< Invalid memcpy direction
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorAddressOfConstant            =     22,   ///< Address of constant error
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorTextureFetchFailed           =     23,   ///< Texture fetch failed
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorTextureNotBound              =     24,   ///< Texture not bound error
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorSynchronizationError         =     25,   ///< Synchronization error
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidFilterSetting         =     26,   ///< Invalid filter setting
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidNormSetting           =     27,   ///< Invalid norm setting
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorMixedDeviceExecution         =     28,   ///< Mixed device execution
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorCudartUnloading              =     29,   ///< CUDA runtime unloading
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorUnknown                      =     30,   ///< Unknown error condition
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorNotYetImplemented            =     31,   ///< Function not yet implemented
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorMemoryValueTooLarge          =     32,   ///< Memory value too large
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInvalidResourceHandle        =     33,   ///< Invalid resource handle
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorNotReady                     =     34,   ///< Not ready error
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorInsufficientDriver           =     35,   ///< CUDA runtime is newer than driver
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorSetOnActiveProcess           =     36,   ///< Set on active process error
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorNoDevice                     =     38,   ///< No available CUDA device
Analysis/BkgModel/CUDA/cuda_error.h:  cudaErrorStartupFailure               =   0x7f,   ///< Startup failure
Analysis/BkgModel/CUDA/cuda_error.h:#endif // CUDA_ERROR_H
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.h://translate function will translate the old layout into a Cube layout for the new gpu pipeline
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:// cuda
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:#include "cuda_error.h"
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:  cudaMemcpyKind getCopyKind(const MemSegment& src);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:  cudaMemcpyKind getCopyInKind();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:  cudaMemcpyKind getCopyOutKind();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:  void copyAsync (const MemSegment& src, cudaStream_t sid, size_t size = 0);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:  void memSetAsync(int value, cudaStream_t sid, size_t size);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:  void memSetAsync(int value, cudaStream_t sid, size_t offset, size_t size);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:  //returns segment pointer to be passed to GPU Kernel (should not be used for any other pointer arithmetics)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h://of equal size. if that is not the case an exception of type cudaAllocationError
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:  void copyToHostAsync (cudaStream_t sid, size_t size = 0);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:  void copyToDeviceAsync (cudaStream_t sid, size_t size = 0);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:  //returns segment device pointer to be passed to GPU Kernel (should not be used for any other pointer aritmetics
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h://throws cudaExecutionException if MemSegment is Device Memory
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:      cout << "CUDA Memory Manager Error: Access to Device Memory via [] operator not possible!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:      throw cudaExecutionException(cudaErrorInvalidHostPointer, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.h:        //throw cudaAllocationError()
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  cudaGetDevice(&_devId);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  cout << getLogHeader() << " acquiring cudaStream" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:    throw cudaException(cudaErrorMemoryAllocation);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  cudaStreamCreate(&_stream);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  cudaError_t err = cudaGetLastError();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  if(_stream ==NULL  || err != cudaSuccess){
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:    throw cudaStreamCreationError(__FILE__,__LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  if(_stream != NULL) cudaStreamDestroy(_stream); //CUDA_ERROR_CHECK(); 
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:cudaStream_t streamResources::getStream()
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  headerinfo << "CUDA " << _devId << " StreamResource " << _streamId << ":";
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:size_t cudaResourcePool::_SrequestedDeviceSize = 0;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:size_t cudaResourcePool::_SrequestedHostSize = 0;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:cudaResourcePool::cudaResourcePool(int numStreams)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  cudaGetDevice(&_devId);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  if(_sRes.empty()) throw   cudaStreamCreationError( __FILE__,__LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:cudaResourcePool::cudaResourcePool(size_t hostsize, size_t devicesize, int numStreams)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  cudaGetDevice(&_devId);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  if(_sRes.empty()) throw   cudaStreamCreationError( __FILE__,__LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:void cudaResourcePool::tryAddResource(unsigned int numStreams)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:      // remove print memory since we observed a segfault in the libcuda api call, see TS-7922
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:    catch(cudaException &e){
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:cudaResourcePool::~cudaResourcePool()
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:streamResources * cudaResourcePool::getResource()
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:void cudaResourcePool::releaseResource(streamResources *& res)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:void cudaResourcePool::poolCleaning()
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:int cudaResourcePool::getNumStreams()
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:string cudaResourcePool::getLogHeader()
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  headerinfo << "CUDA " << _devId << " ResourcePool:";
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:void cudaResourcePool::printMemoryUsage()
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  cudaMemGetInfo( &free_byte, &total_byte ) ;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:  cout << getLogHeader() << " GPU memory usage: used = " << used_db/divMB<< ", free = " << free_db/divMB<< " MB, total = "<< total_db/divMB<<" MB" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:size_t cudaResourcePool::requestDeviceMemory(size_t size)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:void cudaResourcePool::setDeviceMemory(size_t size)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu:size_t cudaResourcePool::requestHostMemory(size_t size)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA Memory Manager Warning: Segment accumulation failed first and last segment do not match type" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA Memory Manager Warning: Segment accumulation failed first and last base pointers are not in order" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA Memory Manager Warning: Appending Segment failed, segments do not match type" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA Memory Manager Warning: Appending Segment failed, first and last base pointers are not in order" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA Memory Manager Warning: Appended Segment is not the imidiate next segment, segments inbetween are added automatically!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA Memory Manager Warning: Segment rezise request to size <= 0, previouse size: " << _size << "!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cudaMemcpy(_basePointer,src,checkSize(size),getCopyInKind());
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cudaMemcpy(dst,_basePointer,checkSize(size),getCopyOutKind());
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cudaMemcpy(_basePointer + dstOffset, src, checkSize(dstOffset+size),getCopyInKind());
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  //cout << "cudaMemcpy( " << dst <<", " <<(void*)_basePointer << " + " <<  srcOffset << ", " <<  "checkSize(" << srcOffset << " + " <<  size  << " ),getCopyOutKind())" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cudaMemcpy(dst,(void*)(_basePointer + srcOffset) ,checkSize(srcOffset+size),getCopyOutKind());
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:void MemSegment::copyAsync (const MemSegment& src, cudaStream_t sid, size_t size)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  if(sid == 0) cout << "CUDA Memory Manager Warning: intended async-copy is using stream 0 turning it into non-async copy!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cudaMemcpyAsync(_basePointer, src._basePointer, checkSize(size), getCopyKind(src), sid);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cout << "CUDA Memory Manager Warning: intended async-copy is using non paged locked host memory turning it into a non-async copy!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cudaMemcpy(_basePointer, src._basePointer, checkSize(size), getCopyKind(src));
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA Memory Manager Warning: intended async-copy is using non paged locked host memory turning it into a non-async copy!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cudaMemcpy(_basePointer, src._basePointer, checkSize(size), getCopyKind(src));
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cout << "CUDA Memory Manager Warning: buffer size missmatch dst: ";
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cudaMemcpy(_basePointer, src._basePointer, checkSize(size), getCopyKind(src));
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  //cout << "cudaMemcpy(" << (void*)_basePointer << ", " << (void*)src._basePointer << ", " << checkSize(size) << ", " << getCopyKind(src) << ")" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cudaMemcpy(_basePointer + dstOffset, src._basePointer + srcOffset, copysize, getCopyKind(src));
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cudaMemset((void*)_basePointer, value, checkSize(size));
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:void MemSegment::memSetAsync(int value, cudaStream_t sid, size_t size)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cudaMemsetAsync((void*)_basePointer, value, checkSize(size), sid);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cout << "CUDA Memory Manager Warning: Asyncronouse Host Side memset not available!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cudaMemset((void*)tmpPtr, value,size);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:void MemSegment::memSetAsync(int value, cudaStream_t sid, size_t offset, size_t size)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cudaMemsetAsync((void*)tmpPtr, value, size, sid);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cout << "CUDA Memory Manager Warning: Asynchronous Host Side memset not available!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA Memory Manager Warning: tried to split segment at offset(" <<offset<<") >= segment size("<< _size << ")!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  //cout << "CUDA Memory Manager: splitting buffer of size " << _size << " into two buffers of sizes " << offset << " and " << _size-offset << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    //if(size < _size) cout << "CUDA Memory Manager Warning: copying smaller segment of size " << size <<" into large segment of size " << _size << "!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cout << "CUDA Memory Manager Warning: requested size (" << size <<") is larger than segment size (" << _size << ")!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  throw cudaAllocationError(cudaErrorInvalidValue, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:cudaMemcpyKind MemSegment::getCopyKind(const MemSegment& src)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:          return cudaMemcpyDeviceToDevice;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:          return cudaMemcpyHostToDevice;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:              return cudaMemcpyDeviceToHost;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:              return cudaMemcpyHostToHost;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:cudaMemcpyKind MemSegment::getCopyInKind()
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      return cudaMemcpyHostToDevice;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      return cudaMemcpyHostToHost;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:cudaMemcpyKind MemSegment::getCopyOutKind()
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      return cudaMemcpyDeviceToHost;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      return cudaMemcpyHostToHost;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cout << "CUDA Memory Manager Debug: MemSegment: " << getSize() <<", " << getVoidPtr() << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cout << "CUDA Memory Manager Error: Device-Host Segment size mismatch!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:void MemSegPair::copyToHostAsync (cudaStream_t sid, size_t size)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:void MemSegPair::copyToDeviceAsync (cudaStream_t sid, size_t size)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cout << "CUDA Memory Manager Debug: MemSegPair Host:" << _Host.getSize() <<", " << _Host.getVoidPtr() << " Device:" << _Device.getSize() <<", " << _Device.getVoidPtr() << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA MemoryManager: attempt to change pre-allocation setting for an already allocated buffer!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA MemoryManager: attempt to change pre-allocation setting for an already allocated buffer!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA MemoryManager: attempt to change pre-allocation setting for an already allocated buffer!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA MemoryManager: attempt to change pre-allocation setting for an already allocated buffer!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cout << "CUDA MemoryManager: attempt to change pre-allocation setting for an already allocated buffer!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    cout << "CUDA MemoryManager: attempt to allocate without providing type or size!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cout << "CUDA MemoryManager: attempt to allocate already allocated buffer!" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      throw cudaAllocationError(cudaErrorMemoryAllocation, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  catch(cudaException &e)
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    throw cudaAllocationError(e.getCudaError(), __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cudaError_t err = cudaSuccess;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:      cudaMalloc(&ptr, size);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  err = cudaGetLastError();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  if ( err != cudaSuccess || ptr == NULL){
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    throw cudaAllocationError(err, __FILE__, __LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:        cudaFreeHost(ptr); //CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:        cudaFree(ptr); //CUDA_ERROR_CHECK();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  // if(_type == UnDefined || _basePointer == NULL)   throw cudaNotEnoughMemForStream(__FILE__,__LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:        cout << "CUDA: getSegment Device " << segSizePadded <<" used:"  << memoryUsed() <<  " free:"  << memoryAvailable() << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:        cout << "CUDA: getSegment Host   " << segSizePadded <<" used:"  << memoryUsed() <<  " free:"  << memoryAvailable()  << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:    throw cudaNotEnoughMemForStream(__FILE__,__LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cudaMemGetInfo( &free_byte, &total_byte ) ;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:        throw cudaNotEnoughMemForStream(__FILE__,__LINE__);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  cudaGetDevice(&devId);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu:  headerinfo << "CUDA " << devId << ":";
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:#include "GpuPipelineDefines.h"
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:    printf("CUDA: HistoryCollection:\n collected Sample Flows: %d\n max Sample Flows: %d\n write Index: %d\n latest Sample Trace Buffer: %p\n  latest EmptyTrace Buffer: %p\n",collectedFlows,numFlows,writeBuffer,SmplAddress,EmptyAddress);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:  // to be executed during last flow before switching to GPU flow by flow pipeline.
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:    cout << "CUDA: HistoryCollection: serialization, loading non dynamic members" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:      cout << "CUDA: HistoryCollection: serialization, loading Regional Parameters" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:    cout << "CUDA: HistoryCollection: loading max sample Frames " << sampleFrames << " max avg empty frames "<< emptyFrames << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:      cout << "CUDA: HistoryCollection: serialization, loading Sample Traces and Empty Average History for " << collectedFlows << " flows" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:    cout << "CUDA: HistoryCollection: serialization, storing non dynamic members" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:      cout << "CUDA: HistoryCollection: serialization, storing Regional Parameters" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:      cout << "CUDA: HistoryCollection: max sample Frames " << sampleFrames << " max avg empty frames "<< emptyFrames << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:      cout << "CUDA: HistoryCollection: no history data for serialization available" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.h:      cout << "CUDA: HistoryCollection: no history data for serialization available" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.cu:#include "cuda_error.h"
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.cu:  //frames by region by param cuda
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.cu:  //frames by region by param cuda
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.cu:  if(CfP.getUncompFrames() > MAX_UNCOMPRESSED_FRAMES_GPU){
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.cu:         <<"CUDA WARNING: The number of uncompressed frames of "<< CfP.getUncompFrames() <<" for this block " << endl
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.cu:         <<"              exceeds the GPU frame buffer limit for a maximum of " << MAX_UNCOMPRESSED_FRAMES_GPU << " frames." <<endl
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.cu:         <<"              No more than "<< MAX_UNCOMPRESSED_FRAMES_GPU <<" uncompressed frames will used!!" <<endl
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.cu:    CfP.setUncompFrames(MAX_UNCOMPRESSED_FRAMES_GPU);
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.cu:    cout << "CUDA: HistoryCollection: creating Sample Bead History of " << numFlowsInHistory << " flows for Regional Parameter Fitting" <<endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.cu:        assert(numSamples < NUM_SAMPLES_RF && "GPU Flow by FLow Pipeline, Region Sample limit exceeded!");
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.cu:    cout << "DEBUG GPU EmptytraceAvg History," << regId <<"," << flow << ",";
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.cu:  cout << "DEBUG GPU Regional Param initialization regId " << regId << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.cu:  cout << "DEBUG: GPU Pipeline, collected samples for: " << collectedFlows << " flows for Regional Fitting" <<endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.cu:  cout << "DEBUG: GPU Pipeline, num Samples per region Host side:" <<endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.cu:  cout << "CUDA: HistoryCollection: InitDeviceBuffersAndSymbol: created " << DeviceSampleCompressedTraces.size() << " Device History Buffers (" << getSize()/(1024.0*1024.0) << "MB), and initialized Device control symbol" <<endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.cu:    cout << "CUDA: HistoryCollection: Copying history, regional param and polyclonal buffers from device to Host for serialization" << endl;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.h:// cuda
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.h:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.h:#include "cuda_error.h"
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.h:// Not Singleton anymore, one per GPU
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.h:  cudaStream_t _stream;
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.h:  cudaStream_t getStream();
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.h:class cudaResourcePool
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.h:  cudaResourcePool(int numStreams = MAX_ALLOWED_NUM_STREAMS );
Analysis/BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.h:  ~cudaResourcePool();
Analysis/BkgModel/CUDA/StreamManager.cu:#include "cuda_error.h"
Analysis/BkgModel/CUDA/StreamManager.cu:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/StreamManager.cu:bool cudaSimpleStreamExecutionUnit::_verbose = false;
Analysis/BkgModel/CUDA/StreamManager.cu:int cudaSimpleStreamExecutionUnit::_seuCnt = 0;
Analysis/BkgModel/CUDA/StreamManager.cu:bool cudaSimpleStreamManager::_verbose =false;
Analysis/BkgModel/CUDA/StreamManager.cu:int cudaSimpleStreamManager::_maxNumStreams = MAX_ALLOWED_NUM_STREAMS;
Analysis/BkgModel/CUDA/StreamManager.cu:cudaSimpleStreamExecutionUnit::cudaSimpleStreamExecutionUnit( streamResources * resources,  WorkerInfoQueueItem item )
Analysis/BkgModel/CUDA/StreamManager.cu:  if(_resource == NULL) throw cudaStreamCreationError(__FILE__,__LINE__);
Analysis/BkgModel/CUDA/StreamManager.cu:cudaSimpleStreamExecutionUnit::~cudaSimpleStreamExecutionUnit()
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamExecutionUnit::setName(std::string name)
Analysis/BkgModel/CUDA/StreamManager.cu:bool cudaSimpleStreamExecutionUnit::execute()
Analysis/BkgModel/CUDA/StreamManager.cu:void * cudaSimpleStreamExecutionUnit::getJobData()
Analysis/BkgModel/CUDA/StreamManager.cu:WorkerInfoQueueItem cudaSimpleStreamExecutionUnit::getItem()
Analysis/BkgModel/CUDA/StreamManager.cu:bool cudaSimpleStreamExecutionUnit::checkComplete()
Analysis/BkgModel/CUDA/StreamManager.cu:  cudaError_t ret;
Analysis/BkgModel/CUDA/StreamManager.cu:  ret = cudaStreamQuery(_stream);
Analysis/BkgModel/CUDA/StreamManager.cu:  if( ret == cudaErrorNotReady  ) return false;
Analysis/BkgModel/CUDA/StreamManager.cu:  if( ret == cudaSuccess) return true;
Analysis/BkgModel/CUDA/StreamManager.cu:  ret = cudaGetLastError();
Analysis/BkgModel/CUDA/StreamManager.cu:  throw cudaExecutionException(ret, __FILE__,__LINE__);
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamExecutionUnit::setVerbose(bool v)
Analysis/BkgModel/CUDA/StreamManager.cu:bool cudaSimpleStreamExecutionUnit::Verbose()
Analysis/BkgModel/CUDA/StreamManager.cu:string cudaSimpleStreamExecutionUnit::getName()
Analysis/BkgModel/CUDA/StreamManager.cu:string cudaSimpleStreamExecutionUnit::getLogHeader()
Analysis/BkgModel/CUDA/StreamManager.cu:  headerinfo << "CUDA " << _resource->getDevId() << ": SEU " << getSeuNum() << ": " << getName() << " SR " << getStreamId()<< ":";
Analysis/BkgModel/CUDA/StreamManager.cu:int cudaSimpleStreamExecutionUnit::getSeuNum()
Analysis/BkgModel/CUDA/StreamManager.cu:int cudaSimpleStreamExecutionUnit::getStreamId()
Analysis/BkgModel/CUDA/StreamManager.cu:bool cudaSimpleStreamExecutionUnit::InitJob() {
Analysis/BkgModel/CUDA/StreamManager.cu:cudaSimpleStreamExecutionUnit * cudaSimpleStreamExecutionUnit::makeExecutionUnit(streamResources * resources, WorkerInfoQueueItem item)
Analysis/BkgModel/CUDA/StreamManager.cu:  cudaSimpleStreamExecutionUnit * tmpSeu = NULL;
Analysis/BkgModel/CUDA/StreamManager.cu:  headerinfo << "CUDA " << resources->getDevId() << ": SEU Factory SR "<< resources->getStreamId() << ":";
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamExecutionUnit::setCompute(int compute)
Analysis/BkgModel/CUDA/StreamManager.cu:int cudaSimpleStreamExecutionUnit::getCompute()
Analysis/BkgModel/CUDA/StreamManager.cu:int cudaSimpleStreamExecutionUnit::getNumFrames()
Analysis/BkgModel/CUDA/StreamManager.cu:int cudaSimpleStreamExecutionUnit::getNumBeads()
Analysis/BkgModel/CUDA/StreamManager.cu:cudaSimpleStreamManager::cudaSimpleStreamManager( 
Analysis/BkgModel/CUDA/StreamManager.cu:  cudaGetDevice( &_devId );
Analysis/BkgModel/CUDA/StreamManager.cu:  cudaDeviceProp deviceProp;
Analysis/BkgModel/CUDA/StreamManager.cu:  cudaGetDeviceProperties(&deviceProp, _devId);
Analysis/BkgModel/CUDA/StreamManager.cu:  _GPUerror = false;
Analysis/BkgModel/CUDA/StreamManager.cu:cudaSimpleStreamManager::~cudaSimpleStreamManager()
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamManager::allocateResources()
Analysis/BkgModel/CUDA/StreamManager.cu:  //size_t maxDeviceSize = getMaxDeviceSize(MAX_PREALLOC_COMPRESSED_FRAMES_GPU);
Analysis/BkgModel/CUDA/StreamManager.cu:    _resourcePool = new cudaResourcePool(_maxNumStreams); // throws cudaException
Analysis/BkgModel/CUDA/StreamManager.cu:    _GPUerror = false;
Analysis/BkgModel/CUDA/StreamManager.cu:    _GPUerror = true;
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamManager::freeResources()
Analysis/BkgModel/CUDA/StreamManager.cu:int cudaSimpleStreamManager::getNumStreams()
Analysis/BkgModel/CUDA/StreamManager.cu:int cudaSimpleStreamManager::availableResources()
Analysis/BkgModel/CUDA/StreamManager.cu:size_t cudaSimpleStreamManager::getMaxHostSize(int flow_block_size)
Analysis/BkgModel/CUDA/StreamManager.cu:size_t cudaSimpleStreamManager::getMaxDeviceSize(int maxFrames, int maxBeads, int flow_block_size)
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamManager::moveToCPU()
Analysis/BkgModel/CUDA/StreamManager.cu:  //get jobs and hand them over to the CPU Q after GPU error was encountered
Analysis/BkgModel/CUDA/StreamManager.cu:        cout << getLogHeader()<< " managed to acquire streamResources, switching execution back to GPU!" << endl;
Analysis/BkgModel/CUDA/StreamManager.cu:        _GPUerror = false;
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamManager::getJob()
Analysis/BkgModel/CUDA/StreamManager.cu:// new GPU Jobs have to be added to this switch/case statement
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamManager::addSEU()
Analysis/BkgModel/CUDA/StreamManager.cu:    cudaSimpleStreamExecutionUnit * tmpSeu = NULL;
Analysis/BkgModel/CUDA/StreamManager.cu:      tmpSeu = cudaSimpleStreamExecutionUnit::makeExecutionUnit(_resourcePool->getResource(), _item);
Analysis/BkgModel/CUDA/StreamManager.cu:    catch(cudaException &e){
Analysis/BkgModel/CUDA/StreamManager.cu:        _GPUerror = true;
Analysis/BkgModel/CUDA/StreamManager.cu:        cout << " *** ERROR DURING STREAM UNIT CREATION, retry on GPU" << endl;
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamManager::executeSEU()
Analysis/BkgModel/CUDA/StreamManager.cu:      catch(cudaAllocationError &e){
Analysis/BkgModel/CUDA/StreamManager.cu:          cout << getLogHeader() << "*** CUDA RESOURCE POOL EMPTY , handing incomplete Job back to CPU for retry" << endl;
Analysis/BkgModel/CUDA/StreamManager.cu:          _GPUerror = true;
Analysis/BkgModel/CUDA/StreamManager.cu:          cout << getLogHeader() << "*** CUDA STREAM RESOURCE COULD NOT BE ALLOCATED, " << getNumStreams() << " StreamResources still avaiable, retry pending" << endl;
Analysis/BkgModel/CUDA/StreamManager.cu:      catch(cudaException &e){
Analysis/BkgModel/CUDA/StreamManager.cu:        if(e.getCudaError() == cudaErrorLaunchFailure)
Analysis/BkgModel/CUDA/StreamManager.cu:          cout << getLogHeader() << "encountered Kernel Launch Failure. Stop retrying, set GPU error state" << endl;
Analysis/BkgModel/CUDA/StreamManager.cu:          _GPUerror = true;
Analysis/BkgModel/CUDA/StreamManager.cu:            cudaSimpleStreamExecutionUnit::setVerbose(true);
Analysis/BkgModel/CUDA/StreamManager.cu:            cout << getLogHeader() << "encountered " << MAX_EXECUTION_ERRORS << " errors. Stop retrying, set GPU error state" << endl;
Analysis/BkgModel/CUDA/StreamManager.cu:            cudaSimpleStreamExecutionUnit::setVerbose(false);
Analysis/BkgModel/CUDA/StreamManager.cu:            _GPUerror = true;
Analysis/BkgModel/CUDA/StreamManager.cu:bool cudaSimpleStreamManager::DoWork()
Analysis/BkgModel/CUDA/StreamManager.cu:    if(_GPUerror ){
Analysis/BkgModel/CUDA/StreamManager.cu:        _GPUerror = true;
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamManager::recordBeads(int n)
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamManager::recordFrames(int n)
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamManager::setNumMaxStreams(int numMaxStreams)
Analysis/BkgModel/CUDA/StreamManager.cu:    cout << "CUDA: tried to set number of streams to " << numMaxStreams << ", correcting to allowed maximum of " << MAX_ALLOWED_NUM_STREAMS <<  " streams " << endl;
Analysis/BkgModel/CUDA/StreamManager.cu:int cudaSimpleStreamManager::getNumMaxStreams()
Analysis/BkgModel/CUDA/StreamManager.cu:void cudaSimpleStreamManager::setVerbose(bool v)
Analysis/BkgModel/CUDA/StreamManager.cu:string cudaSimpleStreamManager::getLogHeader()
Analysis/BkgModel/CUDA/StreamManager.cu:  headerinfo << "CUDA " << _devId << ": StreamManager:";
Analysis/BkgModel/CUDA/StreamManager.cu:bool cudaSimpleStreamManager::checkItem()
Analysis/BkgModel/CUDA/StreamManager.cu:bool cudaSimpleStreamManager::isFinishItem()
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.h:#include "GpuMultiFlowFitControl.h"
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.h:#define CUDA_MULTIFLOW_NUM_FIT 2 
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.h:  GpuMultiFlowFitControl _multiFlowFitControl;
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.h:  GpuMultiFlowFitMatrixConfig* _fd[CUDA_MULTIFLOW_NUM_FIT];
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.h:  void putJobToGPU(WorkerInfoQueueItem item);
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:#include "GpuMultiFlowFitControl.h"
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:  maxFrames =  (_maxFrames != 0)?(_maxFrames):(GpuMultiFlowFitControl::GetMaxFrames());
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:  if ( _flow_block_size > MAX_NUM_FLOWS_IN_BLOCK_GPU )
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:      "GPU acceleration requires that the number of flows in a block be less than %d.\n"
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:      "This limit is set at compile time in MAX_NUM_FLOWS_IN_BLOCK_GPU.\n", 
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:      MAX_NUM_FLOWS_IN_BLOCK_GPU );
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:  int maxBeads = (_maxBeads != 0)?(_maxBeads):(GpuMultiFlowFitControl::GetMaxBeads());
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:  return getMaxBeads();// GpuMultiFlowFitControl::GetMaxBeads();
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:void WorkSet::putJobToGPU(WorkerInfoQueueItem item)
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:  //_info->pq->GetGpuQueue()->PutItem(item);
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:  _info->QueueControl->GetGpuQueue()->PutItem(item);
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:    << " | max beads: " << GpuMultiFlowFitControl::GetMaxBeads() << " max frames: " << GpuMultiFlowFitControl::GetMaxFrames() << endl
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:  return _info->inception_state->bkg_control.gpuControl.postFitHandshakeWorker;
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:    return _info->bkgObj->getGlobalDefaultsForBkgModel().signal_process_control.amp_guess_on_gpu;
Analysis/BkgModel/CUDA/HostDataWrapper/JobWrapper.cu:  return MAX_UNCOMPRESSED_FRAMES_GPU;
Analysis/BkgModel/CUDA/CudaUtils.cu:#include "CudaUtils.h"
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.cpp:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.cpp:#include "GpuMultiFlowFitMatrixConfig.h"
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.cpp:GpuMultiFlowFitMatrixConfig::GpuMultiFlowFitMatrixConfig(const std::vector<fit_descriptor>& fds, CpuStep* Steps, int maxSteps, int flow_key, int flow_block_size)
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.cpp:GpuMultiFlowFitMatrixConfig::~GpuMultiFlowFitMatrixConfig()
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.cpp:void GpuMultiFlowFitMatrixConfig::CreatePartialDerivStepsVector(const std::vector<fit_descriptor>& fds, CpuStep* Steps, int maxSteps)
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.cpp:void GpuMultiFlowFitMatrixConfig::CreateAffectedFlowsVector(
Analysis/BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.cpp:void GpuMultiFlowFitMatrixConfig::CreateBitMapForJTJMatrixComputation()
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:#ifndef GPUMULTIFLOWFITCONTROL_H
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:#define GPUMULTIFLOWFITCONTROL_H
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:#include "GpuMultiFlowFitMatrixConfig.h"
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:class GpuMultiFlowFitControl
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:    GpuMultiFlowFitControl(GpuMultiFlowFitControl const&);  
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:    GpuMultiFlowFitControl& operator=(GpuMultiFlowFitControl const&);
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:    GpuMultiFlowFitControl();
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:    ~GpuMultiFlowFitControl();
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:    GpuMultiFlowFitMatrixConfig* GetMatrixConfig(const string &name, 
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:    static void SetChemicalXtalkCorrectionForPGM(bool doXtalk) { _gpuTraceXtalk = doXtalk; }
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:    static bool doGPUTraceLevelXtalk() { return _gpuTraceXtalk; }
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:    GpuMultiFlowFitMatrixConfig* createConfig(
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:    std::map<MatrixIndex, GpuMultiFlowFitMatrixConfig* > _allMatrixConfig;
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:    static bool _gpuTraceXtalk;
Analysis/BkgModel/CUDA/GpuMultiFlowFitControl.h:#endif // GPUMULTIFLOWFITCONTROL_H
Analysis/BkgModel/CUDA/MultiFitStream.h:// cuda
Analysis/BkgModel/CUDA/MultiFitStream.h:#include "cuda_runtime.h"
Analysis/BkgModel/CUDA/MultiFitStream.h:#include "cuda_error.h"
Analysis/BkgModel/CUDA/MultiFitStream.h:  TMemSegment<CpuStep> Steps; // we need a specific struct describing this config for this well fit for GPU
Analysis/BkgModel/CUDA/MultiFitStream.h:class SimpleMultiFitStream : public cudaSimpleStreamExecutionUnit
Analysis/BkgModel/CUDA/MultiFitStream.h:  float _lambda_start[CUDA_MULTIFLOW_NUM_FIT];
Analysis/BkgModel/CUDA/MultiFitStream.h:  int _fit_training_level[CUDA_MULTIFLOW_NUM_FIT];
Analysis/BkgModel/CUDA/MultiFitStream.h:  int _fit_iterations[CUDA_MULTIFLOW_NUM_FIT];
Analysis/BkgModel/CUDA/MultiFitStream.h:  int _clonal_restriction[CUDA_MULTIFLOW_NUM_FIT];
Analysis/BkgModel/CUDA/MultiFitStream.h:  float _restrict_clonal[CUDA_MULTIFLOW_NUM_FIT];
Analysis/BkgModel/CUDA/MultiFitStream.h:  MultiFitData _HostDeviceFitData[CUDA_MULTIFLOW_NUM_FIT];
Analysis/BkgModel/CUDA/cudaStreamTemplate.h:#ifndef CUDASTREAMTEMPLATE_H
Analysis/BkgModel/CUDA/cudaStreamTemplate.h:#define CUDASTREAMTEMPLATE_H
Analysis/BkgModel/CUDA/cudaStreamTemplate.h:class TemplateStream : public cudaSimpleStreamExecutionUnit
Analysis/BkgModel/CUDA/cudaStreamTemplate.h:#endif // CUDASTREAMTEMPLATE_H
Analysis/BkgModel/CUDA/dumper.h://size_t addCudaData(void * devData, size_t bytes);
Analysis/BkgModel/CUDA/dumper.h://bool CompareCuda(float * devData, float threshold, OutputFormat output =MIN); 
Analysis/BkgModel/CUDA/dumper.h://static bool CompareCuda(float * devData, float * hostData, size_t size, float threshold, OutputFormat output =MIN);
Analysis/BkgModel/CUDA/ParamStructs.h:#include "CudaDefines.h"
Analysis/BkgModel/CUDA/ParamStructs.h:  int coarse_nuc_start[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/CUDA/ParamStructs.h:  int fine_nuc_start[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/CUDA/ParamStructs.h:  float deltaFrames[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/ParamStructs.h:  float frameNumber[MAX_COMPRESSED_FRAMES_GPU];
Analysis/BkgModel/CUDA/ParamStructs.h:  int flowIdxMap[MAX_NUM_FLOWS_IN_BLOCK_GPU]; 
Analysis/BkgModel/SignalProcessingMasterFitter.h:class BkgModelCuda;
Analysis/BkgModel/SignalProcessingMasterFitter.h:    // Forward declaration of the CUDA model which needs private access to the CPU model
Analysis/BkgModel/SignalProcessingMasterFitter.h:    friend class BkgModelCuda;
Analysis/BkgModel/SignalProcessingMasterFitter.h:    // ProcessImage had to be broken into two function, before and after GPUGenerateBeadTraces.
Analysis/BkgModel/SignalProcessingMasterFitter.h:    bool InitProcessImageForGPU ( Image *img, int raw_flow, int flow_buffer_index );
Analysis/BkgModel/SignalProcessingMasterFitter.h:    bool FinalizeProcessImageForGPU ( int flow_block_size );
Analysis/BkgModel/SignalProcessingMasterFitter.h:    // allow GPU code to trigger PCA Dark Matter Calculation on CPU
Analysis/BkgModel/SignalProcessingMasterFitter.h:    void SetFittersIfNeeded(); // temporarily made public for new GPU pipeline ToDo: find beter way
Analysis/BkgModel/SignalProcessingMasterFitter.h:    /* Relevant functions to integrate GPU multi flow fitting into signal processing pipeline*/
Analysis/BkgModel/MathModel/PoissonCdf.h:#if defined( __SSE__ ) && !defined( __CUDACC__ )
Analysis/BkgModel/MathModel/PoissonCdf.h:#ifdef ION_COMPILE_CUDA
Analysis/BkgModel/MathModel/PoissonCdf.h:    #include <cuda_runtime.h>		// for __host__ and __device__
Analysis/BkgModel/MathModel/PoissonCdf.h:// or the code from Alan Kaminsky's clear recopying of said code for CUDA.
Analysis/BkgModel/MathModel/PoissonCdf.h:#ifdef ION_COMPILE_CUDA
Analysis/BkgModel/MathModel/PoissonCdf.h:#ifdef ION_COMPILE_CUDA
Analysis/BkgModel/MathModel/PoissonCdf.h:#ifdef ION_COMPILE_CUDA
Analysis/BkgModel/MathModel/PoissonCdf.h:#ifdef ION_COMPILE_CUDA
Analysis/BkgModel/MathModel/MathOptim.h:#if defined( __SSE__ ) && !defined( __CUDACC__ )
Analysis/BkgModel/MathModel/MathOptim.h:#if !defined( __SSE3__ ) || defined( __CUDACC__ )
Analysis/BkgModel/MathModel/MathOptim.h:#if defined( __SSE3__ ) && !defined( __CUDACC__ )
Analysis/BkgModel/MathModel/MathOptim.h:        (void) occ_l; (void) occ_r; //stub for cuda
Analysis/BkgModel/GlobalDefaultsForBkgModel.cpp:  amp_guess_on_gpu = false;
Analysis/BkgModel/Fitters/RefineFit.h:// make this code look >just< like the GPU option
Analysis/BkgModel/Fitters/Complex/MultiLevMar.cpp:// strong candidate for export to the GPU
Analysis/BkgModel/Fitters/Complex/MultiLevMar.h:// and look just like the cuda code :-)
Analysis/BkgModel/Bookkeeping/RegionParams.cpp:  for (int flow=0; flow<MAX_NUM_FLOWS_IN_BLOCK_GPU; ++flow) {
Analysis/BkgModel/Bookkeeping/RegionParams.cpp:  for (int flow=0; flow<MAX_NUM_FLOWS_IN_BLOCK_GPU; ++flow) {
Analysis/BkgModel/Bookkeeping/RegionParams.cpp:  for ( int i=0; i<MAX_NUM_FLOWS_IN_BLOCK_GPU; i++ )
Analysis/BkgModel/Bookkeeping/RegionParams.cpp:  for ( int i=0; i<MAX_NUM_FLOWS_IN_BLOCK_GPU; i++ )
Analysis/BkgModel/Bookkeeping/BeadParams.h:// CPU oriented:  fix up for GPU required to avoid too much data volume passed
Analysis/BkgModel/Bookkeeping/BeadParams.h:  float mean_residual_error[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/Bookkeeping/BeadParams.h:  float tauB[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // save for output trace.h5
Analysis/BkgModel/Bookkeeping/BeadParams.h:  float etbR[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // save for output trace.h5
Analysis/BkgModel/Bookkeeping/BeadParams.h:  float bkg_leakage[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // save for output
Analysis/BkgModel/Bookkeeping/BeadParams.h:  int fit_type[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // save for output
Analysis/BkgModel/Bookkeeping/BeadParams.h:  bool converged[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // save for output
Analysis/BkgModel/Bookkeeping/BeadParams.h:  float initA[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/Bookkeeping/BeadParams.h:  float initkmult[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/Bookkeeping/BeadParams.h:  float t_sigma_actual[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/Bookkeeping/BeadParams.h:  float t_mid_nuc_actual[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/Bookkeeping/BeadParams.h:    for( size_t i = 0 ; i < MAX_NUM_FLOWS_IN_BLOCK_GPU ; ++i ){
Analysis/BkgModel/Bookkeeping/BeadParams.h:  //int hits_by_flow[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // temporary tracker
Analysis/BkgModel/Bookkeeping/BeadParams.h:  // DOING SO WILL KILL GPU DATA ACCESS AND CAUSE ASSERTIONS IN SingleFitStream.cu
Analysis/BkgModel/Bookkeeping/BeadParams.h:  float Ampl[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // homopolymer length mixture
Analysis/BkgModel/Bookkeeping/BeadParams.h:  float kmult[MAX_NUM_FLOWS_IN_BLOCK_GPU];  // individual flow multiplier to rate of enzyme action
Analysis/BkgModel/Bookkeeping/BeadParams.h:  // DOING SO WILL KILL GPU DATA ACCESS AND CAUSE ASSERTIONS IN SingleFitStream.cu
Analysis/BkgModel/Bookkeeping/TimeCompression.h:#if defined( __SSE__ ) && !defined( __CUDACC__ )
Analysis/BkgModel/Bookkeeping/XtalkCurry.cpp:// for each bead on GPU.Can be used for CPU too. Just one time exercise when
Analysis/BkgModel/Bookkeeping/BeadParams.cpp: /* for (int j=0; j<MAX_NUM_FLOWS_IN_BLOCK_GPU; j++)
Analysis/BkgModel/Bookkeeping/BeadParams.cpp:  for (int j=0;j<MAX_NUM_FLOWS_IN_BLOCK_GPU;j++)
Analysis/BkgModel/Bookkeeping/RegionParams.h:#ifdef ION_COMPILE_CUDA
Analysis/BkgModel/Bookkeeping/RegionParams.h:  #include <cuda_runtime.h>   // for __host__ and __device__
Analysis/BkgModel/Bookkeeping/RegionParams.h:// MAX_NUM_FLOWS_IN_BLOCK_GPU. In an ideal world, these could be dynamically sized.
Analysis/BkgModel/Bookkeeping/RegionParams.h:// In this world, however, these structures are copied over to the GPU in raw form.
Analysis/BkgModel/Bookkeeping/RegionParams.h:  float t_mid_nuc[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/Bookkeeping/RegionParams.h:  float t_mid_nuc_shift_per_flow[MAX_NUM_FLOWS_IN_BLOCK_GPU]; // note how this is redundant(!)
Analysis/BkgModel/Bookkeeping/RegionParams.h:  #ifdef ION_COMPILE_CUDA
Analysis/BkgModel/Bookkeeping/RegionParams.h:// MAX_NUM_FLOWS_IN_BLOCK_GPU. In an ideal world, these could be dynamically sized.
Analysis/BkgModel/Bookkeeping/RegionParams.h:// In this world, however, these structures are copied over to the GPU in raw form.
Analysis/BkgModel/Bookkeeping/RegionParams.h:  float darkness[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/Bookkeeping/RegionParams.h:  float Ampl[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/Bookkeeping/RegionParams.h:  float copy_multiplier[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/Bookkeeping/RegionParams.h:  float darkness[MAX_NUM_FLOWS_IN_BLOCK_GPU];
Analysis/BkgModel/GlobalDefaultsForBkgModel.h:  bool amp_guess_on_gpu;
Analysis/BkgModel/GlobalDefaultsForBkgModel.h:        & amp_guess_on_gpu
Analysis/config/args_GX7v1_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_GX7v1_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_GX7v1_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_GX7v1_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_GX7v1_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_GX7v1_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_GX7v1_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_GX7v1_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_GX7v1_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_GX7v1_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_GX7v1_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_GX7v1_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_GX7v1_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_GX7v1_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_GX7v1_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_GX7v1_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_GX7v1_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_GX7v1_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_GX7v1_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_GX7v1_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_GX7v1_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_900_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_900_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_900_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_900_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_900_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_900_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_900_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_900_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_900_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_900_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_900_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_900_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_900_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_900_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_900_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_900_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_900_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_900_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_900_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_900_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_900_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_316v2_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_316v2_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_316v2_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_316v2_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_316v2_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_316v2_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_316v2_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_316v2_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_316v2_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_316v2_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_316v2_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_316v2_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_316v2_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_316v2_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_316v2_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_316v2_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_316v2_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_316v2_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_316v2_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_316v2_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_316v2_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_541_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_541_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_541_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_541_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_541_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_541_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_541_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_541_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_541_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_541_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_541_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_541_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_541_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_541_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_541_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_541_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_541_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_541_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_541_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_541_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_541_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_541_analysis.json:      "gpuworkload" : 0
Analysis/config/args_530_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_530_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_530_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_530_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_530_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_530_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_530_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_530_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_530_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_530_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_530_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_530_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_530_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_530_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_530_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_530_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_530_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_530_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_530_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_530_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_530_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_P2.1.1_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_P2.1.1_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_P2.1.1_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_P2.1.1_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_P2.1.1_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_P2.2.1_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_P2.2.1_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_P2.2.1_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_P2.2.1_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_P2.2.1_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_541v2_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_541v2_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_541v2_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_541v2_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_541v2_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_541v2_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_541v2_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_541v2_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_541v2_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_541v2_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_541v2_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_541v2_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_541v2_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_541v2_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_541v2_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_541v2_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_541v2_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_541v2_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_541v2_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_541v2_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_541v2_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_P1.1.541_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_P1.1.541_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_P1.1.541_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_P1.1.541_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_P1.1.541_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_540_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_540_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_540_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_540_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_540_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_540_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_540_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_540_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_540_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_540_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_540_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_540_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_540_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_540_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_540_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_540_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_540_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_540_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_540_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_540_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_540_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_550_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_550_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_550_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_550_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_550_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_550_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_550_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_550_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_550_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_550_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_550_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_550_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_550_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_550_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_550_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_550_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_550_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_550_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_550_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_550_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_550_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_P1.1.17_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_P1.1.17_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_P1.1.17_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_P1.1.17_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_P1.1.17_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_521_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_521_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_521_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_521_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_521_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_521_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_521_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_521_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_521_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_521_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_521_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_521_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_521_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_521_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_521_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_521_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_521_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_521_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_521_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_521_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_521_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_P2.2.2_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_P2.2.2_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_P2.2.2_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_P2.2.2_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_P2.2.2_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_318select_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_318select_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_318select_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_318select_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_318select_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_318select_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_318select_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_318select_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_318select_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_318select_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_318select_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_318select_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_318select_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_318select_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_318select_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_318select_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_318select_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_318select_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_318select_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_318select_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_318select_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_316_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_316_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_316_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_316_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_316_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_316_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_316_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_316_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_316_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_316_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_316_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_316_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_316_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_316_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_316_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_316_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_316_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_316_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_316_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_316_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_316_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_314_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_314_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_314_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_314_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_314_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_314_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_314_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_314_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_314_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_314_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_314_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_314_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_314_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_314_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_314_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_314_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_314_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_314_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_314_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_314_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_314_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_GX5v2_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_GX5v2_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_GX5v2_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_GX5v2_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_GX5v2_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_GX5v2_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_GX5v2_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_GX5v2_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_GX5v2_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_GX5v2_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_GX5v2_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_GX5v2_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_GX5v2_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_GX5v2_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_GX5v2_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_GX5v2_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_GX5v2_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_GX5v2_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_GX5v2_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_GX5v2_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_GX5v2_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_522_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_522_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_522_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_522_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_522_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_522_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_522_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_522_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_522_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_522_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_522_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_522_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_522_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_522_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_522_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_522_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_522_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_522_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_522_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_522_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_522_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_318D_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_318D_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_318D_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_318D_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_318D_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_318D_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_318D_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_318D_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_318D_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_318D_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_318D_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_318D_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_318D_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_318D_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_318D_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_318D_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_318D_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_318D_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_318D_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_318D_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_318D_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_520_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_520_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_520_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_520_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_520_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_520_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_520_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_520_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_520_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_520_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_520_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_520_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_520_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_520_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_520_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_520_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_520_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_520_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_520_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_520_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_520_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_GX9v1_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_GX9v1_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_GX9v1_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_GX9v1_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_GX9v1_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_GX9v1_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_GX9v1_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_GX9v1_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_GX9v1_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_GX9v1_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_GX9v1_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_GX9v1_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_GX9v1_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_GX9v1_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_GX9v1_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_GX9v1_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_GX9v1_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_GX9v1_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_GX9v1_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_GX9v1_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_GX9v1_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_GX9v1_analysis.json:      "gpuworkload" : 0
Analysis/config/args_P2.3.1_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_P2.3.1_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_P2.3.1_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_P2.3.1_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_P2.3.1_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_P2.3.1_analysis.json:      "gpuworkload" : 0.0,
Analysis/config/args_P1.0.19_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_P1.0.19_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_P1.0.19_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_P1.0.19_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_P1.0.19_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_P1.2.18_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_P1.2.18_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_P1.2.18_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_P1.2.18_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_P1.2.18_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_541M_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_541M_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_541M_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_541M_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_541M_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_541M_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_541M_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_541M_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_541M_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_541M_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_541M_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_541M_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_541M_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_541M_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_541M_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_541M_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_541M_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_541M_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_541M_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_541M_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_541M_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_318_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_318_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_318_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_318_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_318_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_318_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_318_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_318_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_318_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_318_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_318_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_318_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_318_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_318_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_318_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_318_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_318_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_318_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_318_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_318_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_318_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_P2.0.1_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_P2.0.1_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_P2.0.1_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_P2.0.1_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_P2.0.1_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_560_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_560_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_560_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_560_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_560_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_560_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_560_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_560_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_560_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_560_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_560_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_560_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_560_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_560_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_560_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_560_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_560_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_560_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_560_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_560_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_560_analysis.json:      "gpuworkload" : 1.0,
Analysis/config/args_560_analysis.json:      "gpuworkload" : 0
Analysis/config/args_P1.0.20_analysis.json:   "GpuControlOpts" : {
Analysis/config/args_P1.0.20_analysis.json:      "gpu-amp-guess" : 1,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-device-ids" : [],
Analysis/config/args_P1.0.20_analysis.json:      "gpu-fitting-only" : true,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-flow-by-flow" : false,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-hybrid-fit-iter" : 3,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-multi-flow-fit" : 1,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-multi-flow-fit-blocksize" : 128,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-multi-flow-fit-l1config" : -1,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-num-history-flows" : 10,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-num-streams" : 2,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-partial-deriv-blocksize" : 128,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-partial-deriv-l1config" : -1,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-single-flow-fit" : 1,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-single-flow-fit-blocksize" : -1,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-single-flow-fit-l1config" : -1,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-single-flow-fit-type" : 3,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-switch-to-flow-by-flow-at" : 20,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-use-all-devices" : false,
Analysis/config/args_P1.0.20_analysis.json:      "gpu-verbose" : false,
Analysis/config/args_P1.0.20_analysis.json:      "gpuworkload" : 1.0,
Analysis/Util/WorkerInfoQueue.h:class DynamicWorkQueueGpuCpu {
Analysis/Util/WorkerInfoQueue.h:  DynamicWorkQueueGpuCpu(int _depth);
Analysis/Util/WorkerInfoQueue.h:  WorkerInfoQueueItem GetGpuItem();
Analysis/Util/WorkerInfoQueue.h:  int getGpuReadIndex();
Analysis/Util/WorkerInfoQueue.h:  ~DynamicWorkQueueGpuCpu();
Analysis/Util/WorkerInfoQueue.h:    int gpuRdIdx;
Analysis/Util/WorkerInfoQueue.cpp:DynamicWorkQueueGpuCpu::DynamicWorkQueueGpuCpu(int _depth)
Analysis/Util/WorkerInfoQueue.cpp:  gpuRdIdx = 0;
Analysis/Util/WorkerInfoQueue.cpp:WorkerInfoQueueItem DynamicWorkQueueGpuCpu::GetGpuItem() {
Analysis/Util/WorkerInfoQueue.cpp:  //printf("Acquiring lock gpu\n");    
Analysis/Util/WorkerInfoQueue.cpp:  if (gpuRdIdx == cpuRdIdx)
Analysis/Util/WorkerInfoQueue.cpp:  //printf("Getting Gpu Item, GpuIdx: %d CpuIdx: %d, start: %d\n", gpuRdIdx, cpuRdIdx, start); 
Analysis/Util/WorkerInfoQueue.cpp:  item = qlist[gpuRdIdx++];
Analysis/Util/WorkerInfoQueue.cpp:  //printf("Releasing lock gpu\n");    
Analysis/Util/WorkerInfoQueue.cpp:WorkerInfoQueueItem DynamicWorkQueueGpuCpu::GetCpuItem() {
Analysis/Util/WorkerInfoQueue.cpp:  if (cpuRdIdx == gpuRdIdx)
Analysis/Util/WorkerInfoQueue.cpp:  //printf("Getting Cpu Item, GpuIdx: %d CpuIdx: %d, start: %d\n", gpuRdIdx, cpuRdIdx, start); 
Analysis/Util/WorkerInfoQueue.cpp:void DynamicWorkQueueGpuCpu::PutItem(WorkerInfoQueueItem &new_item) {
Analysis/Util/WorkerInfoQueue.cpp:void DynamicWorkQueueGpuCpu::WaitTillDone(void)
Analysis/Util/WorkerInfoQueue.cpp:void DynamicWorkQueueGpuCpu::DecrementDone(void)
Analysis/Util/WorkerInfoQueue.cpp:void DynamicWorkQueueGpuCpu::ResetIndices() {
Analysis/Util/WorkerInfoQueue.cpp:  gpuRdIdx = 0;
Analysis/Util/WorkerInfoQueue.cpp:int DynamicWorkQueueGpuCpu::getGpuReadIndex() {
Analysis/Util/WorkerInfoQueue.cpp:  return gpuRdIdx;
Analysis/Util/WorkerInfoQueue.cpp:DynamicWorkQueueGpuCpu::~DynamicWorkQueueGpuCpu()
Analysis/Util/DataCube.h://#ifndef __CUDACC__
Analysis/Util/DataCube.h://#ifndef __CUDA_ARCH__
Analysis/Util/DataCube.h:// Note: restricting index types to size_t to get around complaint from cuda compiler.
Analysis/xtalk_sim/DiffEqModel.cpp:#define USE_CUDA
Analysis/xtalk_sim/DiffEqModel.cpp:#ifndef USE_CUDA
Analysis/xtalk_sim/DiffEqModel.cpp:  cudaModel = new DelsqCUDA(ni,nj,nk,incorp_inject_cnt);
Analysis/xtalk_sim/DiffEqModel.cpp:#ifndef USE_CUDA
Analysis/xtalk_sim/DiffEqModel.cpp:  cudaModel->setParams(dx,dy,dz,dcoeff,dt);
Analysis/xtalk_sim/DiffEqModel.cpp:  cudaModel->setInput(cmatrix,buffer_effect,correction_factor, layer_step_frac, index_array, weight_array);
Analysis/xtalk_sim/DiffEqModel.cpp:  cudaModel->copyIn();
Analysis/xtalk_sim/DiffEqModel.cpp:		cudaModel->DoWork();
Analysis/xtalk_sim/DiffEqModel.cpp:		cudaModel->DoIncorp(dt*GetIncorpFlux(simTime)*16.9282f);
Analysis/xtalk_sim/DiffEqModel.cpp:  cudaModel->setOutput(cmatrix);
Analysis/xtalk_sim/DiffEqModel.cpp:  cudaModel->copyOut();
Analysis/xtalk_sim/DiffEqModel.cpp:#ifndef USE_CUDA
Analysis/xtalk_sim/DiffEqModel.cpp:  delete cudaModel;
Analysis/xtalk_sim/xtalk_sim.cpp:	pModel->SetupIncorporationSignalInjectionOnGPU();
Analysis/xtalk_sim/DelsqCUDA.cu:#include "DelsqCUDA.h"
Analysis/xtalk_sim/DelsqCUDA.cu:DelsqCUDA::DelsqCUDA(size_t x, size_t y, size_t z, int inject_cnt, int deviceId)
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaSetDevice(devId);
Analysis/xtalk_sim/DelsqCUDA.cu:  createCudaBuffers();
Analysis/xtalk_sim/DelsqCUDA.cu:DelsqCUDA::~DelsqCUDA()
Analysis/xtalk_sim/DelsqCUDA.cu:  destroyCudaBuffers(); 
Analysis/xtalk_sim/DelsqCUDA.cu:void DelsqCUDA::createCudaBuffers()
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMalloc( &Dsrc, size()); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMalloc( &Ddst, size()); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMalloc( &Dbuffer_effect, size()); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMalloc( &Dcorrection_factor, size()); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMalloc( &Dlayer_step_frac, sizeZ()); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMalloc( &Dindex_array, my_inject_cnt*sizeof(size_t)); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMalloc( &Dweight_array, my_inject_cnt*sizeof(DATA_TYPE)); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:void DelsqCUDA::destroyCudaBuffers()
Analysis/xtalk_sim/DelsqCUDA.cu:  if(Dsrc != NULL) cudaFree(Dsrc); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  if(Ddst != NULL) cudaFree(Ddst); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  if(Dbuffer_effect != NULL) cudaFree(Dbuffer_effect); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  if(Dcorrection_factor != NULL) cudaFree(Dcorrection_factor); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  if(Dlayer_step_frac != NULL) cudaFree(Dlayer_step_frac); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  if(Dindex_array != NULL) cudaFree(Dindex_array); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  if(Dweight_array != NULL) cudaFree(Dweight_array); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:void DelsqCUDA::setParams( DATA_TYPE dx, DATA_TYPE dy,DATA_TYPE dz,DATA_TYPE dcoeff, DATA_TYPE dt)
Analysis/xtalk_sim/DelsqCUDA.cu:    cudaMemcpyToSymbol( CP, &cParams, sizeof(ConstStruct)); CUDA_ERROR_CHECK();
Analysis/xtalk_sim/DelsqCUDA.cu:void DelsqCUDA::setInput( DATA_TYPE * cmatrix, 
Analysis/xtalk_sim/DelsqCUDA.cu:void DelsqCUDA::copyIn()
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMemcpy( Dsrc, Hcmatrix, size(), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMemcpy( Dbuffer_effect, Hbuffer_effect, size(), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMemcpy( Dcorrection_factor, Hcorrection_factor, size(), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMemcpy( Dlayer_step_frac, Hlayer_step_frac, sizeZ(), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:	cudaMemcpy( Dindex_array, Hindex_array, my_inject_cnt*sizeof(size_t), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:	cudaMemcpy( Dweight_array, Hweight_array, my_inject_cnt*sizeof(DATA_TYPE), cudaMemcpyHostToDevice); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:void DelsqCUDA::setOutput( DATA_TYPE * dst )
Analysis/xtalk_sim/DelsqCUDA.cu:void DelsqCUDA::copyOut()
Analysis/xtalk_sim/DelsqCUDA.cu:  cudaMemcpy( Hdst, dOutput , size(), cudaMemcpyDeviceToHost); CUDA_ERROR_CHECK(); 
Analysis/xtalk_sim/DelsqCUDA.cu:void DelsqCUDA::DoWork()
Analysis/xtalk_sim/DelsqCUDA.cu:  delsq_kernel<<< grid, block >>>(Dsrc, Ddst, Dlayer_step_frac, Dbuffer_effect, Dcorrection_factor); CUDA_ERROR_CHECK();
Analysis/xtalk_sim/DelsqCUDA.cu:void DelsqCUDA::DoIncorp( DATA_TYPE incorp_signal )
Analysis/xtalk_sim/DelsqCUDA.cu:	incorp_sig_kernel<<< grid, block >>>(Dsrc, Dindex_array, Dweight_array, incorp_signal); CUDA_ERROR_CHECK();
Analysis/xtalk_sim/cuda_error.h:#ifndef CUDA_ERROR_H
Analysis/xtalk_sim/cuda_error.h:#define CUDA_ERROR_H
Analysis/xtalk_sim/cuda_error.h:#ifndef NO_CUDA_DEBUG
Analysis/xtalk_sim/cuda_error.h:#define CUDA_ERROR_CHECK()                                                                     \
Analysis/xtalk_sim/cuda_error.h:    cudaError_t err = cudaGetLastError();                                                      \
Analysis/xtalk_sim/cuda_error.h:    if ( err != cudaSuccess && err != cudaErrorSetOnActiveProcess ) {                          \
Analysis/xtalk_sim/cuda_error.h:                  << " | ** CUDA ERROR! ** " << std::endl                                      \
Analysis/xtalk_sim/cuda_error.h:                  << " | Msg: " << cudaGetErrorString(err) << std::endl                        \
Analysis/xtalk_sim/cuda_error.h:#define CUDA_ERROR_CHECK() {}
Analysis/xtalk_sim/cuda_error.h:/* cuda Error Codes:
Analysis/xtalk_sim/cuda_error.h:  cudaSuccess                           =      0,   ///< No errors
Analysis/xtalk_sim/cuda_error.h:  cudaErrorMissingConfiguration         =      1,   ///< Missing configuration error
Analysis/xtalk_sim/cuda_error.h:  cudaErrorMemoryAllocation             =      2,   ///< Memory allocation error
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInitializationError          =      3,   ///< Initialization error
Analysis/xtalk_sim/cuda_error.h:  cudaErrorLaunchFailure                =      4,   ///< Launch failure
Analysis/xtalk_sim/cuda_error.h:  cudaErrorPriorLaunchFailure           =      5,   ///< Prior launch failure
Analysis/xtalk_sim/cuda_error.h:  cudaErrorLaunchTimeout                =      6,   ///< Launch timeout error
Analysis/xtalk_sim/cuda_error.h:  cudaErrorLaunchOutOfResources         =      7,   ///< Launch out of resources error
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidDeviceFunction        =      8,   ///< Invalid device function
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidConfiguration         =      9,   ///< Invalid configuration
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidDevice                =     10,   ///< Invalid device
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidValue                 =     11,   ///< Invalid value
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidPitchValue            =     12,   ///< Invalid pitch value
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidSymbol                =     13,   ///< Invalid symbol
Analysis/xtalk_sim/cuda_error.h:  cudaErrorMapBufferObjectFailed        =     14,   ///< Map buffer object failed
Analysis/xtalk_sim/cuda_error.h:  cudaErrorUnmapBufferObjectFailed      =     15,   ///< Unmap buffer object failed
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidHostPointer           =     16,   ///< Invalid host pointer
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidDevicePointer         =     17,   ///< Invalid device pointer
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidTexture               =     18,   ///< Invalid texture
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidTextureBinding        =     19,   ///< Invalid texture binding
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidChannelDescriptor     =     20,   ///< Invalid channel descriptor
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidMemcpyDirection       =     21,   ///< Invalid memcpy direction
Analysis/xtalk_sim/cuda_error.h:  cudaErrorAddressOfConstant            =     22,   ///< Address of constant error
Analysis/xtalk_sim/cuda_error.h:  cudaErrorTextureFetchFailed           =     23,   ///< Texture fetch failed
Analysis/xtalk_sim/cuda_error.h:  cudaErrorTextureNotBound              =     24,   ///< Texture not bound error
Analysis/xtalk_sim/cuda_error.h:  cudaErrorSynchronizationError         =     25,   ///< Synchronization error
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidFilterSetting         =     26,   ///< Invalid filter setting
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidNormSetting           =     27,   ///< Invalid norm setting
Analysis/xtalk_sim/cuda_error.h:  cudaErrorMixedDeviceExecution         =     28,   ///< Mixed device execution
Analysis/xtalk_sim/cuda_error.h:  cudaErrorCudartUnloading              =     29,   ///< CUDA runtime unloading
Analysis/xtalk_sim/cuda_error.h:  cudaErrorUnknown                      =     30,   ///< Unknown error condition
Analysis/xtalk_sim/cuda_error.h:  cudaErrorNotYetImplemented            =     31,   ///< Function not yet implemented
Analysis/xtalk_sim/cuda_error.h:  cudaErrorMemoryValueTooLarge          =     32,   ///< Memory value too large
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInvalidResourceHandle        =     33,   ///< Invalid resource handle
Analysis/xtalk_sim/cuda_error.h:  cudaErrorNotReady                     =     34,   ///< Not ready error
Analysis/xtalk_sim/cuda_error.h:  cudaErrorInsufficientDriver           =     35,   ///< CUDA runtime is newer than driver
Analysis/xtalk_sim/cuda_error.h:  cudaErrorSetOnActiveProcess           =     36,   ///< Set on active process error
Analysis/xtalk_sim/cuda_error.h:  cudaErrorNoDevice                     =     38,   ///< No available CUDA device
Analysis/xtalk_sim/cuda_error.h:  cudaErrorStartupFailure               =   0x7f,   ///< Startup failure
Analysis/xtalk_sim/cuda_error.h:#endif // CUDA_ERROR_H
Analysis/xtalk_sim/DiffEqModel.h:#include "DelsqCUDA.h"
Analysis/xtalk_sim/DiffEqModel.h:	// build matricies for single well signal injection that are usable by the GPU for injection in GPU code
Analysis/xtalk_sim/DiffEqModel.h:	void SetupIncorporationSignalInjectionOnGPU(void);
Analysis/xtalk_sim/DiffEqModel.h:	size_t    *index_array;		// indicies used by GPU for incorporation signal injection
Analysis/xtalk_sim/DiffEqModel.h:	DATA_TYPE *weight_array;	// weights used by GPU for incorporation signal injection
Analysis/xtalk_sim/DiffEqModel.h:// CUDA execution
Analysis/xtalk_sim/DiffEqModel.h:  DelsqCUDA * cudaModel;
Analysis/xtalk_sim/DiffEqModel_Init.cpp:// that are used by the GPU code to do signal injection on the GPU instead of on the CPU.
Analysis/xtalk_sim/DiffEqModel_Init.cpp:void DiffEqModel::SetupIncorporationSignalInjectionOnGPU()
Analysis/xtalk_sim/DelsqCUDA.h:#ifndef DELSQCUDA_H
Analysis/xtalk_sim/DelsqCUDA.h:#define DELSQCUDA_H
Analysis/xtalk_sim/DelsqCUDA.h:#include "cuda_runtime.h"
Analysis/xtalk_sim/DelsqCUDA.h:#include "cuda_error.h"
Analysis/xtalk_sim/DelsqCUDA.h:class DelsqCUDA
Analysis/xtalk_sim/DelsqCUDA.h:  DelsqCUDA(size_t x, size_t y, size_t z, int inject_cnt, int deviceId = DEVICE_ID);
Analysis/xtalk_sim/DelsqCUDA.h:  ~DelsqCUDA();
Analysis/xtalk_sim/DelsqCUDA.h:  void createCudaBuffers();
Analysis/xtalk_sim/DelsqCUDA.h:  void destroyCudaBuffers();
Analysis/xtalk_sim/DelsqCUDA.h:#endif // DELSQCUDA_H
Analysis/CMakeLists.txt:if(ION_USE_CUDA)
Analysis/CMakeLists.txt:    #if(NOT ION_USE_SYSTEM_CUDA)
Analysis/CMakeLists.txt:     #   message(STATUS "BUILD with CUDA ${CUDA_VERSION}")
Analysis/CMakeLists.txt:        #install(DIRECTORY ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib
Analysis/CMakeLists.txt:        #        DESTINATION /usr/local/cuda)
Analysis/CMakeLists.txt:        #install(DIRECTORY ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64
Analysis/CMakeLists.txt:        #        DESTINATION /usr/local/cuda)
Analysis/CMakeLists.txt:        #install(PROGRAMS ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/bin/nvcc
Analysis/CMakeLists.txt:        #        DESTINATION /usr/local/cuda/bin)
Analysis/CMakeLists.txt:        #string(REGEX REPLACE "(.*)\\.[0-9]*" "\\1" CUDA_SHORT_VERSION ${CUDA_VERSION})
Analysis/CMakeLists.txt:        #install(FILES ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64/libcudart.so
Analysis/CMakeLists.txt:        #              ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64/libcudart.so.${CUDA_SHORT_VERSION}
Analysis/CMakeLists.txt:        #              ${PROJECT_BINARY_DIR}/../${cuda_toolkit_version}/lib64/libcudart.so.${CUDA_VERSION}
Analysis/CMakeLists.txt:	#	        DESTINATION /usr/local/cuda/lib64)
Analysis/CMakeLists.txt:    # from https://pypi.python.org/pypi/nvidia-ml-py/
Analysis/CMakeLists.txt:    install(FILES pynvml/nvidia_smi.py             DESTINATION ${PYTHON_LOCAL_SITE_PACKAGES})
Analysis/CMakeLists.txt:    install(FILES pynvml/nvidia_smi.pyc            DESTINATION ${PYTHON_LOCAL_SITE_PACKAGES})
Analysis/CMakeLists.txt:    install(PROGRAMS pynvml/nvidia_smi_test.py     DESTINATION bin)
Analysis/CMakeLists.txt:include_directories("${PROJECT_SOURCE_DIR}/BkgModel/CUDA")
Analysis/CMakeLists.txt:include_directories("${PROJECT_SOURCE_DIR}/BkgModel/CUDA/HostDataWrapper")
Analysis/CMakeLists.txt:include_directories("${PROJECT_SOURCE_DIR}/BkgModel/CUDA/HosteDeviceDataCubes")
Analysis/CMakeLists.txt:include_directories("${PROJECT_SOURCE_DIR}/BkgModel/CUDA/KernelIncludes")
Analysis/CMakeLists.txt:# CUDA Files
Analysis/CMakeLists.txt:if(ION_USE_CUDA)
Analysis/CMakeLists.txt:  CUDA_COMPILE(CUDA_TEMP_FILES 
Analysis/CMakeLists.txt:      BkgModel/CUDA/HosteDeviceDataCubes/MemoryManager.cu
Analysis/CMakeLists.txt:      BkgModel/CUDA/HosteDeviceDataCubes/ResourcePool.cu
Analysis/CMakeLists.txt:      BkgModel/CUDA/HosteDeviceDataCubes/LayoutTranslator.cu  
Analysis/CMakeLists.txt:      BkgModel/CUDA/HosteDeviceDataCubes/SampleHistory.cu
Analysis/CMakeLists.txt:      BkgModel/CUDA/HostDataWrapper/JobWrapper.cu 
Analysis/CMakeLists.txt:      BkgModel/CUDA/BkgGpuPipeline.cu
Analysis/CMakeLists.txt:      BkgModel/CUDA/SingleFitStream.cu
Analysis/CMakeLists.txt:      BkgModel/CUDA/StreamManager.cu 
Analysis/CMakeLists.txt:      BkgModel/CUDA/MultiFitStream.cu
Analysis/CMakeLists.txt:      BkgModel/CUDA/StreamingKernels.cu
Analysis/CMakeLists.txt:      BkgModel/CUDA/MasterKernel.cu      
Analysis/CMakeLists.txt:    SET( CUDA_CPP_FILES
Analysis/CMakeLists.txt:      BkgModel/CUDA/HosteDeviceDataCubes/PerBeadDataCubes.cpp
Analysis/CMakeLists.txt:      BkgModel/CUDA/HostDataWrapper/ClonalFilterWrapper.cpp
Analysis/CMakeLists.txt:      BkgModel/CUDA/KernelIncludes/DeviceParamDefines.cpp
Analysis/CMakeLists.txt:    AnalysisOrg/IO/GpuControlOpts.cpp
Analysis/CMakeLists.txt:    AnalysisOrg/cudaWrapper.cpp
Analysis/CMakeLists.txt:    BkgModel/CUDA/GpuMultiFlowFitControl.cpp
Analysis/CMakeLists.txt:    BkgModel/CUDA/GpuMultiFlowFitMatrixConfig.cpp
Analysis/CMakeLists.txt:	${CUDA_CPP_FILES}
Analysis/CMakeLists.txt:    ${CUDA_TEMP_FILES}
Analysis/CMakeLists.txt:if(ION_USE_CUDA)
Analysis/CMakeLists.txt:  if(NOT ION_USE_SYSTEM_CUDA)
Analysis/CMakeLists.txt:    add_dependencies(ion-analysis ${cuda_toolkit})
Analysis/CMakeLists.txt:    if (ION_USE_CUDA)
Analysis/CMakeLists.txt:        target_link_libraries(Analysis ion-analysis pthread dl ${CUDA_LIBRARIES})
Analysis/CMakeLists.txt:#    if (ION_USE_CUDA)
Analysis/CMakeLists.txt:#        target_link_libraries(bkgFit ion-analysis pthread ${CUDA_LIBRARIES} ${ION_ARMADILLO_LIBS})
Analysis/Image/ChipIdDecoder.h:    static bool BigEnoughForGPU();
Analysis/Image/ChipIdDecoder.cpp:bool ChipIdDecoder::BigEnoughForGPU(){
Analysis/AnalysisOrg/BkgFitterTracker.cpp:#include "GpuMultiFlowFitControl.h"
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  //no longer allows for heterogeneous execution only all CPU or all GPU
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::UnSpinGPUThreads ()
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  if (analysis_queue.GetGpuQueue())
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    for (int i=0;i < analysis_compute_plan.numBkgWorkers_gpu;i++)
Analysis/AnalysisOrg/BkgFitterTracker.cpp:      analysis_queue.GetGpuQueue()->PutItem (item);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    analysis_queue.GetGpuQueue()->WaitTillDone();
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    delete analysis_queue.GetGpuQueue();
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    analysis_queue.SetGpuQueue(NULL);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::UnSpinMultiFlowFitGpuThreads ()
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  if (analysis_queue.GetMultiFitGpuQueue())
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    for (int i=0;i < analysis_compute_plan.numMultiFlowFitGpuWorkers;i++)
Analysis/AnalysisOrg/BkgFitterTracker.cpp:      analysis_queue.GetMultiFitGpuQueue()->PutItem (item);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    analysis_queue.GetMultiFitGpuQueue()->WaitTillDone();
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    delete analysis_queue.GetMultiFitGpuQueue();
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    printf("Deleting multi fit gpu queue\n");
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    analysis_queue.SetMultiFitGpuQueue(NULL);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  //ampEstBufferForGPU = NULL;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  //  if (ampEstBufferForGPU)
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  //    delete ampEstBufferForGPU;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::UpdateAndCheckGPUCommandlineOptions (CommandLineOpts &inception_state)
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  // shut off gpu multi flow fit if starting flow is not in first 20
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    inception_state.bkg_control.gpuControl.gpuMultiFlowFit = 0;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  // shut off gpu multifit if regional sampling is not enabled
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    inception_state.bkg_control.gpuControl.gpuMultiFlowFit = 0;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  //force gpuMultiFlowFit, just for Vadim:
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  if( inception_state.bkg_control.gpuControl.gpuForceMultiFlowFit)
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    inception_state.bkg_control.gpuControl.gpuMultiFlowFit = 1;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  assert( "GPU ASSERT FAILED: Flow by Flow pipeline currently only works if regional sampling is turned on" && !((global_defaults.signal_process_control.regional_sampling == false) && (inception_state.bkg_control.gpuControl.gpuFlowByFlowExecution == true)) );
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::SetUpCpuAndGpuPipelines (BkgModelControlOpts &bkg_control )
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  cout << "Analysis Pipeline: configuring GPU queue and GPU if available" << endl;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  GpuQueueControl.configureGpu(bkg_control);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  GpuQueueControl.createQueue(numFitters);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  CpuQueueControl.setGpuQueue(GpuQueueControl.getQueue());
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::UpdateGPUPipelineExecutionConfiguration(CommandLineOpts & inception_state){
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  // tweaking global defaults for bkg model if GPU is used
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  if (useGpuAcceleration()) {
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    global_defaults.signal_process_control.amp_guess_on_gpu = GpuQueueControl.ampGuessOnGpu();
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::SpinnUpGpuThreads()
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  //if gpu spinup fails gpu object will be in error state the gpu quueue will be NULL
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  GpuQueueControl.SpinUpThreads(CpuQueueControl.GetQueue());
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  if (useGpuAcceleration())
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    fprintf (stdout, "Number of GPU threads for background model: %d\n", GpuQueueControl.getNumWorkers());
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    fprintf (stdout, "No GPU threads created. proceeding with CPU only execution\n");
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  //provide Cpu Queue to gpu threads as fall back in error state
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::UnSpinGpuThreads ()
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  GpuQueueControl.UnSpinThreads();
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::checkAndInitGPUPipelineSwitch(
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  if(GpuQueueControl.checkIfInitFlowByFlow(flow,restart))
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    cout << "CUDA: cleaning up GPU pipeline and queuing system used for first 20 flows!" <<endl;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    UnSpinGpuThreads();
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    cout << "CUDA: initiating flow by flow pipeline" << endl;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    if (inception_state.bkg_control.gpuControl.postFitHandshakeWorker)
Analysis/AnalysisOrg/BkgFitterTracker.cpp:      GpuQueueControl.setUpAndStartFlowByFlowHandshakeWorker(  inception_state, my_image_spec, &signal_proc_fitters, packQueue, writeQueue, rawWells,flow);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  GpuMultiFlowFitControl::SetMaxFrames(maxFrames);
Analysis/AnalysisOrg/BkgFitterTracker.cpp://void BkgFitterTracker::SpinUpGPUThreads()
Analysis/AnalysisOrg/BkgFitterTracker.cpp://  analysis_queue.SpinUpGPUThreads( analysis_compute_plan );
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::DetermineAndSetGPUAllocationAndKernelParams(
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    GpuMultiFlowFitControl::SetMaxFrames(maxFrames);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    GpuMultiFlowFitControl::SetMaxBeads(maxBeads);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  GpuMultiFlowFitControl::SetChemicalXtalkCorrectionForPGM(bkg_control.enable_trace_xtalk_correction);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  cout << "CUDA: worst case per region beads: "<< maxBeads << " frames: " << maxFrames << endl;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  GpuQueueControl.configureKernelExecution(global_max_flow_key, global_max_flow_max);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:// GPU Block level signal processing
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::ExecuteGPUFlowByFlowSignalProcessing(
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  GpuMultiFlowFitControl::SetMaxFrames(maxFrames);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    //bkinfo[r].gpuAmpEstPerFlow =   ampEstBufferForGPU;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  //if (!ProcessProtonBlockImageOnGPU(bkinfo, flow_block_size,deviceId)) {
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  if(!GpuQueueControl.fullBlockSignalProcessing(bkinfo)){
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    std::cout << "GPU block processing  failed at flow " << flow << std::endl;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:void BkgFitterTracker::CollectSampleWellsForGPUFlowByFlowSignalProcessing(
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  GpuMultiFlowFitControl::SetMaxFrames(maxFrames);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    //bkinfo[r].gpuAmpEstPerFlow =   ampEstBufferForGPU;
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  //if (!ProcessProtonBlockImageOnGPU(bkinfo, flow_block_size,deviceId)) {
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  //GpuQueueControl.collectHistroyForRegionalFitting(bkinfo,flow_block_size,20);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  GpuQueueControl.collectHistroyForRegionalFitting(bkinfo,flow_block_size,inception_state->bkg_control.gpuControl.gpuNumHistoryFlows);
Analysis/AnalysisOrg/BkgFitterTracker.cpp:  if (!ampEstBufferForGPU)
Analysis/AnalysisOrg/BkgFitterTracker.cpp:    ampEstBufferForGPU = new RingBuffer<float>(numBuffers, bufSize);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:      //if (info->pq->GetGpuQueue() && info->pq->performGpuMultiFlowFitting())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:      //  info->pq->GetGpuQueue()->PutItem(item);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:      //if (info->pq->GetGpuQueue() && info->pq->performGpuSingleFlowFitting())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:      //  info->pq->GetGpuQueue()->PutItem(item);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  //if (info->pq->GetGpuQueue() && info->pq->performGpuSingleFlowFitting())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  //  info->pq->GetGpuQueue()->PutItem(item);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  // This will override gpuWorkLoad=1 and will only use GPU for chips which are allowed in the following function
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  my_compute_plan.use_gpu_acceleration = UseGpuAcceleration(bkg_control.gpuControl.gpuWorkLoad);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  my_compute_plan.gpu_work_load = bkg_control.gpuControl.gpuWorkLoad;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  // Option to use all GPUs in system (including display devices). If set to true, will only use the
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  my_compute_plan.use_all_gpus = bkg_control.gpuControl.gpuUseAllDevices;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  // force to run on user supplied gpu device id's
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (bkg_control.gpuControl.gpuDeviceIds.size() > 0) {
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    my_compute_plan.valid_devices = bkg_control.gpuControl.gpuDeviceIds;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (configureGpu (
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:          my_compute_plan.use_gpu_acceleration, 
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:          my_compute_plan.use_all_gpus,
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:          my_compute_plan.numBkgWorkers_gpu))
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    my_compute_plan.use_gpu_only_fitting = bkg_control.gpuControl.doGpuOnlyFitting;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    my_compute_plan.gpu_multiflow_fit = bkg_control.gpuControl.gpuMultiFlowFit;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    my_compute_plan.gpu_singleflow_fit = bkg_control.gpuControl.gpuSingleFlowFit;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    printf ("use_gpu_acceleration: %d\n", my_compute_plan.use_gpu_acceleration);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    my_compute_plan.use_gpu_acceleration = false;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    my_compute_plan.gpu_work_load = 0;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    bkg_control.gpuControl.gpuFlowByFlowExecution = false;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    //my_compute_plan.numBkgWorkers = my_compute_plan.use_gpu_acceleration ? numCores() 
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (analysis_compute_plan.use_gpu_acceleration) {
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    if(analysis_compute_plan.numBkgWorkers_gpu) {
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:      my_queue.AllocateGpuInfo(analysis_compute_plan.numBkgWorkers_gpu);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:      my_queue.SetGpuQueue(new WorkerInfoQueue (numRegions*analysis_compute_plan.numBkgWorkers_gpu+1));
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  // decide on whether to use both CPU and GPU for bkg model fitting jobs
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (analysis_compute_plan.use_gpu_only_fitting) {
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (!analysis_compute_plan.gpu_multiflow_fit) {
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    my_queue.turnOffGpuMultiFlowFitting();
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (!analysis_compute_plan.gpu_singleflow_fit) {
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    my_queue.turnOffGpuSingleFlowFitting();
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (analysis_compute_plan.use_gpu_acceleration)
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    fprintf (stdout, "Number of GPU threads for background model: %d\n", analysis_compute_plan.numBkgWorkers_gpu);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (analysis_queue.GetGpuQueue())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    analysis_queue.GetGpuQueue()->WaitTillDone();
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (analysis_queue.GetGpuQueue())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    analysis_queue.GetGpuQueue()->WaitTillDone();
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp://  if (analysis_queue.GetSingleFitGpuQueue())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp://    analysis_queue.GetSingleFitGpuQueue()->WaitTillDone();
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (analysis_compute_plan.use_gpu_acceleration)
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:void ProcessorQueue::SpinUpGPUThreads(ComputationPlanner &analysis_compute_plan )
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (analysis_compute_plan.use_gpu_acceleration) {
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    // create gpu thread for multi flow fit
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    CreateGpuThreadsForFitType(GetGpuInfo(),
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:        GetGpuQueue(),
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:     		analysis_compute_plan.numBkgWorkers_gpu,
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:void ProcessorQueue::CreateGpuThreadsForFitType(
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    std::vector<BkgFitWorkerGpuInfo> &gpuInfo, 
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    std::vector<int> &gpus
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  int threadsPerDevice = numWorkers / gpus.size();
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    gpuInfo[i].gpu_index = gpus[deviceId];
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    gpuInfo[i].queue = (void*) q;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    gpuInfo[i].fallbackQueue = (void*) fallbackQ;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    // Spawn GPU workers pulling items from either the combined queue (dynamic)
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    // or a separate GPU queue (static)
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    int t = pthread_create (&work_thread, NULL, BkgFitWorkerGpu, &gpuInfo[i]);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:      fprintf (stderr, "Error starting GPU thread\n");
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:bool UseGpuAcceleration(float useGpuFlag) {
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (useGpuFlag) {
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    if (ChipIdDecoder::BigEnoughForGPU())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:      printf("GPU acceleration suppressed on small chips\n");
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (GetGpuQueue())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    GetGpuQueue()->WaitTillDone();
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (GetGpuQueue())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    GetGpuQueue()->WaitTillDone();
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp://  if (analysis_queue.GetSingleFitGpuQueue())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp://    analysis_queue.GetSingleFitGpuQueue()->WaitTillDone();
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (GetGpuQueue())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    //my_compute_plan.numBkgWorkers = my_compute_plan.use_gpu_acceleration ? numCores()
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  // queue control with regards to gpu jobs
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  gpuMultiFlowFitting = bkg_control.gpuControl.gpuMultiFlowFit;  
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  gpuSingleFlowFitting = bkg_control.gpuControl.gpuSingleFlowFit;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:void ProcessorQueue::CreateGpuThreadsForFitType(
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    std::vector<BkgFitWorkerGpuInfo> &gpuInfo,
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    std::vector<int> &gpus
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  int threadsPerDevice = numWorkers / gpus.size();
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    gpuInfo[i].gpu_index = gpus[deviceId];
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    gpuInfo[i].queue = (void*) q;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    gpuInfo[i].fallbackQueue = (void*) fallbackQ;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    // Spawn GPU workers pulling items from either the combined queue (dynamic)
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    // or a separate GPU queue (static)
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    int t = pthread_create (&work_thread, NULL, BkgFitWorkerGpu, &gpuInfo[i]);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:      fprintf (stderr, "Error starting GPU thread\n");
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (GetGpuQueue() && performGpuMultiFlowFitting())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    GetGpuQueue()->PutItem(item);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  if (GetGpuQueue() && performGpuSingleFlowFitting())
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    GetGpuQueue()->PutItem(item);
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:  /*if heterogeneous execution try GPU Q if cpu queue was empty*/
Analysis/AnalysisOrg/SignalProcessingFitterQueue.cpp:    pQ = GetGpuQueue();
Analysis/AnalysisOrg/ProcessImageToWell.cpp:#include "GpuMultiFlowFitControl.h"
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  GPUFlowByFlowPipelineInfo &info)
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  info.startingFlow = inception_state.bkg_control.gpuControl.switchToFlowByFlowAt;
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  if( GlobalFitter.GpuQueueControl.useFlowByFlowExecution())
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  if (inception_state.bkg_control.gpuControl.postFitHandshakeWorker) {
Analysis/AnalysisOrg/ProcessImageToWell.cpp:    if(GlobalFitter.GpuQueueControl.handshakeCreated()){
Analysis/AnalysisOrg/ProcessImageToWell.cpp:      GlobalFitter.GpuQueueControl.joinFlowByFlowHandshakeWorker();
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  GlobalFitter.UnSpinGpuThreads();
Analysis/AnalysisOrg/ProcessImageToWell.cpp:    GlobalFitter.GpuQueueControl.mirrorDeviceBuffersToHostForSerialization();
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  GlobalFitter.UpdateAndCheckGPUCommandlineOptions(inception_state);
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  GlobalFitter.SetUpCpuAndGpuPipelines(inception_state.bkg_control);
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  GlobalFitter.UpdateGPUPipelineExecutionConfiguration(inception_state);
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  //GlobalFitter.SetUpGpuPipeline(inception_state.bkg_control, getFlowBlockSeq().HasFlowInFirstFlowBlock( inception_state.flow_context.startingFlow ));
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  // Get the GPU ready, if we're using it.
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  GlobalFitter.DetermineAndSetGPUAllocationAndKernelParams( inception_state.bkg_control, KEY_LEN, getFlowBlockSeq().MaxFlowsInAnyFlowBlock() );
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  GlobalFitter.SpinnUpGpuThreads();
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  //GlobalFitter.SpinUpGPUThreads(); //now is done within pipelinesetup
Analysis/AnalysisOrg/ProcessImageToWell.cpp:  if (GlobalFitter.GpuQueueControl.useFlowByFlowExecution())
Analysis/AnalysisOrg/ProcessImageToWell.cpp:      if ( ! GlobalFitter.GpuQueueControl.isCurrentFlowExecutedAsFlowByFlow(flow))
Analysis/AnalysisOrg/ProcessImageToWell.cpp:    if(GlobalFitter.GpuQueueControl.isCurrentFlowExecutedAsFlowByFlow(flow))
Analysis/AnalysisOrg/ProcessImageToWell.cpp:      GlobalFitter.checkAndInitGPUPipelineSwitch(inception_state,
Analysis/AnalysisOrg/ProcessImageToWell.cpp:      GlobalFitter.ExecuteGPUFlowByFlowSignalProcessing(
Analysis/AnalysisOrg/ProcessImageToWell.cpp:    if(inception_state.bkg_control.gpuControl.gpuFlowByFlowExecution){
Analysis/AnalysisOrg/ProcessImageToWell.cpp:      if(flow == (inception_state.bkg_control.gpuControl.switchToFlowByFlowAt -1) ){
Analysis/AnalysisOrg/ProcessImageToWell.cpp:        GlobalFitter.CollectSampleWellsForGPUFlowByFlowSignalProcessing(
Analysis/AnalysisOrg/ProcessImageToWell.cpp:    if (!GlobalFitter.GpuQueueControl.handshakeCreated() && ! GlobalFitter.GpuQueueControl.isCurrentFlowExecutedAsFlowByFlow(flow))
Analysis/AnalysisOrg/ProcessImageToWell.cpp:    if ( ! GlobalFitter.GpuQueueControl.isCurrentFlowExecutedAsFlowByFlow(flow))
Analysis/AnalysisOrg/ProcessImageToWell.cpp:    } //post fit steps of bkgmodel in separate thread so as not to slow down GPU thread
Analysis/AnalysisOrg/cudaWrapper.cpp:#include "cudaWrapper.h"
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:#include "BkgGpuPipeline.h"
Analysis/AnalysisOrg/cudaWrapper.cpp:#define MIN_CUDA_COMPUTE_VERSION 20
Analysis/AnalysisOrg/cudaWrapper.cpp:#define CUDA_WRONG_TIME_AND_PLACE  \
Analysis/AnalysisOrg/cudaWrapper.cpp:       std::cout << "CUDAWRAPPER: WE SHOULD NOT EVEN BE HERE, THIS PART OF THE CODE SHOULD BE UNREACHABLE WITHOUT GPU: " << __FILE__ << " " << __LINE__ <<std::endl;  \
Analysis/AnalysisOrg/cudaWrapper.cpp://gpuDeviceConfig Class
Analysis/AnalysisOrg/cudaWrapper.cpp:int gpuDeviceConfig::getVersion(int devId)
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaDeviceProp dev_props;
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaError_t err = cudaGetDeviceProperties( &dev_props, devId );
Analysis/AnalysisOrg/cudaWrapper.cpp:  if (err != cudaSuccess) return 0;
Analysis/AnalysisOrg/cudaWrapper.cpp:  cout << "CUDA: gpuDeviceConfig: device added for evaluation: " << devId << ":" << dev_props.name <<" v" << dev_props.major <<"." << dev_props.minor << " " << dev_props.totalGlobalMem/(1024.0*1024.0*1024.0) << "GB" << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  CUDA_WRONG_TIME_AND_PLACE
Analysis/AnalysisOrg/cudaWrapper.cpp:void gpuDeviceConfig::applyMemoryConstraint(size_t minBytes)
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaDeviceProp dev_props;
Analysis/AnalysisOrg/cudaWrapper.cpp:    cudaGetDeviceProperties( &dev_props, validDevices[i] );
Analysis/AnalysisOrg/cudaWrapper.cpp:    cudaGetDeviceProperties( &dev_props, *it);
Analysis/AnalysisOrg/cudaWrapper.cpp:  CUDA_WRONG_TIME_AND_PLACE
Analysis/AnalysisOrg/cudaWrapper.cpp:void gpuDeviceConfig::applyComputeConstraint(int minVersion)
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaDeviceProp dev_props;
Analysis/AnalysisOrg/cudaWrapper.cpp:    cudaGetDeviceProperties( &dev_props, *it);
Analysis/AnalysisOrg/cudaWrapper.cpp:  CUDA_WRONG_TIME_AND_PLACE
Analysis/AnalysisOrg/cudaWrapper.cpp:void gpuDeviceConfig::initDeviceContexts(){
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:      cout << "CUDA "<< *it << ": gpuDeviceConfig::initDeviceContexts: Creating Context and Constant memory on device with id: "<<  *it<< endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:    catch(cudaException &e) {
Analysis/AnalysisOrg/cudaWrapper.cpp:      //throw cudaExecutionException(e.getCudaError(),__FILE__,__LINE__);
Analysis/AnalysisOrg/cudaWrapper.cpp:      cout << "CUDA "<< *it << ": gpuDeviceConfig::initDeviceContexts: Context could not be created. removing device with id: "<<  *it << " from valid device list" << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  CUDA_WRONG_TIME_AND_PLACE
Analysis/AnalysisOrg/cudaWrapper.cpp:gpuDeviceConfig::gpuDeviceConfig():maxComputeVersion(0),minComputeVersion(0){ };
Analysis/AnalysisOrg/cudaWrapper.cpp:bool gpuDeviceConfig::setValidDevices(std::vector<int> &CmdlineDeviceList, bool useMaxComputeVersionOnly)
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  int numGPUs = 0;
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaError_t err = cudaGetDeviceCount( &numGPUs );
Analysis/AnalysisOrg/cudaWrapper.cpp:  if (err != cudaSuccess) {
Analysis/AnalysisOrg/cudaWrapper.cpp:    printf("CUDA: gpuDeviceConfig: No GPU device available. Defaulting to CPU only computation (return code %d: %s) &\n", err , cudaGetErrorString(err));
Analysis/AnalysisOrg/cudaWrapper.cpp:        printf("CUDA WARNING: gpuDeviceConfig: Device with device id %d provided through the command line could not be found and will be ignored!\n", *itDevId);
Analysis/AnalysisOrg/cudaWrapper.cpp:      printf("CUDA WARNING: gpuDeviceConfig: THE DEVICE LIST PROVIDED TO THE COMMAND LINE DID NOT CONTAIN ANY VALID DEVICES!\n");
Analysis/AnalysisOrg/cudaWrapper.cpp:    for ( int dev = 0; dev < numGPUs;  dev++ ){
Analysis/AnalysisOrg/cudaWrapper.cpp:    int minCompVersion = (useMaxComputeVersionOnly)?(maxComputeVersion):(MIN_CUDA_COMPUTE_VERSION);
Analysis/AnalysisOrg/cudaWrapper.cpp:    printf("CUDA: gpuDeviceConfig: minimum compute version used for pipeline: %.1f\n", (float)minCompVersion/10.0 );
Analysis/AnalysisOrg/cudaWrapper.cpp:    err = cudaGetLastError();
Analysis/AnalysisOrg/cudaWrapper.cpp:    printf("CUDA: gpuDeviceConfig: No GPU device available or device not valid. Defaulting to CPU only computation (return code %d: %s) &\n", err , cudaGetErrorString(err));
Analysis/AnalysisOrg/cudaWrapper.cpp:    cout << "CUDA: gpuDeviceConfig: no context could be created, defaulting to CPU only execution" << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:    cudaSetValidDevices( &validDevices[0], int( validDevices.size()));
Analysis/AnalysisOrg/cudaWrapper.cpp:// cudaFlowByFlowHandShaker class
Analysis/AnalysisOrg/cudaWrapper.cpp:  std::cout << "CUDA: flowByFlowHandshaker: Started GPU-CPU handshake worker" << std::endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  cout << "CUDA: flowByFlowHandshaker: " << workers.size() << " worker threads joined." <<endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:// gpuBkgFitWorker class
Analysis/AnalysisOrg/cudaWrapper.cpp:gpuBkgFitWorker::gpuBkgFitWorker(int devId):devId(devId),q(NULL),errorQueue(NULL){};
Analysis/AnalysisOrg/cudaWrapper.cpp:void gpuBkgFitWorker::createStreamManager(){
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  cout << "CUDA " << devId << ": gpuBkgFitWorker: Creating GPU StreamManager" << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaDeviceProp cuda_props;
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaGetDeviceProperties( &cuda_props, devId );
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaSimpleStreamManager  sM( q, errorQueue );
Analysis/AnalysisOrg/cudaWrapper.cpp:  cout << "CUDA " <<  devId <<": gpuBkgFitWorker: Created GPU BkgModel worker...  ("
Analysis/AnalysisOrg/cudaWrapper.cpp:              << cuda_props.name
Analysis/AnalysisOrg/cudaWrapper.cpp:              << " v"<< cuda_props.major <<"."<< cuda_props.minor << ")" << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  cout << "CUDA " << devId << ": gpuBkgFitWorker: Destroying GPU StreamManager" << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  CUDA_WRONG_TIME_AND_PLACE
Analysis/AnalysisOrg/cudaWrapper.cpp:void gpuBkgFitWorker::InternalThreadFunction(){
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  // Unpack GPU worker info
Analysis/AnalysisOrg/cudaWrapper.cpp:  // Wrapper to create a GPU worker and set GPU
Analysis/AnalysisOrg/cudaWrapper.cpp:  //cout << "GPU_INDEX " << devId << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaSetDevice( devId );
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaError_t err = cudaGetLastError();
Analysis/AnalysisOrg/cudaWrapper.cpp:  if ( err == cudaSuccess )
Analysis/AnalysisOrg/cudaWrapper.cpp:  cout << "CUDA: gpuBkgFitWorker: Failed to initialize GPU worker... (" << devId <<": " << cudaGetErrorString(err)<<")" << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  CUDA_WRONG_TIME_AND_PLACE
Analysis/AnalysisOrg/cudaWrapper.cpp:bool gpuBkgFitWorker::start(){
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  //cudaSetDevice( devId );
Analysis/AnalysisOrg/cudaWrapper.cpp:  //cudaError_t err = cudaGetLastError();
Analysis/AnalysisOrg/cudaWrapper.cpp:  //if ( err == cudaSuccess ){
Analysis/AnalysisOrg/cudaWrapper.cpp:  //cout << "CUDA: Failed to initialize GPU worker... (" << devId <<": " << cudaGetErrorString(err)<<")" << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  CUDA_WRONG_TIME_AND_PLACE
Analysis/AnalysisOrg/cudaWrapper.cpp:void gpuBkgFitWorker::join(){
Analysis/AnalysisOrg/cudaWrapper.cpp:// cudaWrapper class
Analysis/AnalysisOrg/cudaWrapper.cpp:cudaWrapper::cudaWrapper(){
Analysis/AnalysisOrg/cudaWrapper.cpp:  useGpu = false;
Analysis/AnalysisOrg/cudaWrapper.cpp:  useAllGpus = false;
Analysis/AnalysisOrg/cudaWrapper.cpp:  #ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  GpuPipeline = NULL;
Analysis/AnalysisOrg/cudaWrapper.cpp:cudaWrapper::~cudaWrapper(){
Analysis/AnalysisOrg/cudaWrapper.cpp:    cout << "CUDA WARNING: cudaWrapper: destructor is called while there are still " << BkgWorkerThreads.size() << " Worker Threads in active state!" <<endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  if(GpuPipeline!=NULL)
Analysis/AnalysisOrg/cudaWrapper.cpp:    delete GpuPipeline;
Analysis/AnalysisOrg/cudaWrapper.cpp:  GpuPipeline = NULL;
Analysis/AnalysisOrg/cudaWrapper.cpp:bool cudaWrapper::checkChipSize()
Analysis/AnalysisOrg/cudaWrapper.cpp:  if (! ChipIdDecoder::BigEnoughForGPU())
Analysis/AnalysisOrg/cudaWrapper.cpp:    printf("CUDA: cudaWrapper: GPU acceleration suppressed on small chips\n");
Analysis/AnalysisOrg/cudaWrapper.cpp:    useGpu = false;
Analysis/AnalysisOrg/cudaWrapper.cpp:void cudaWrapper::configureGpu(BkgModelControlOpts &bkg_control)
Analysis/AnalysisOrg/cudaWrapper.cpp:  configOpts = &bkg_control.gpuControl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  // This will override gpuWorkLoad=1 and will only use GPU for chips which are allowed in the following function
Analysis/AnalysisOrg/cudaWrapper.cpp:  useGpu = (configOpts->gpuWorkLoad > 0);
Analysis/AnalysisOrg/cudaWrapper.cpp:  //only perform next steps if chiop is large enough and gpu compute turned on
Analysis/AnalysisOrg/cudaWrapper.cpp:  if( useGpu && checkChipSize())
Analysis/AnalysisOrg/cudaWrapper.cpp:    useAllGpus = false; //ToDo add comandline param to overwrite
Analysis/AnalysisOrg/cudaWrapper.cpp:    //configure actual GPUs. if compile without CUDA this is a NoOp
Analysis/AnalysisOrg/cudaWrapper.cpp:    useGpu = deviceConfig.setValidDevices(configOpts->gpuDeviceIds,useMaxComputeVersion);
Analysis/AnalysisOrg/cudaWrapper.cpp:  if (!useGpu)
Analysis/AnalysisOrg/cudaWrapper.cpp:    configOpts->gpuFlowByFlowExecution = false;
Analysis/AnalysisOrg/cudaWrapper.cpp:  cout << "CUDA: useGpuAcceleration: "<< useGpuAcceleration() << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:void cudaWrapper::createQueue(int numRegions)
Analysis/AnalysisOrg/cudaWrapper.cpp:  if(useGpuAcceleration()){ //assume that we will have one worker per valid device
Analysis/AnalysisOrg/cudaWrapper.cpp:void cudaWrapper::destroyQueue()
Analysis/AnalysisOrg/cudaWrapper.cpp:WorkerInfoQueue * cudaWrapper::getQueue()
Analysis/AnalysisOrg/cudaWrapper.cpp:bool cudaWrapper::checkIfInitFlowByFlow(int currentFlow, bool restart){
Analysis/AnalysisOrg/cudaWrapper.cpp: #ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:      cout << "CUDA: cudaWrapper: flow " << switchAtFlow() << " reached, switching from old block of 20 flows to NEW flow by flow GPU pipeline!" <<endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:    if(restart && GpuPipeline == NULL){
Analysis/AnalysisOrg/cudaWrapper.cpp:      cout << "CUDA: cudaWrapper: Initiating flow by flow GPU pipeline after restart!" <<endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:bool cudaWrapper::SpinUpThreads( WorkerInfoQueue* fallbackCPUQ)
Analysis/AnalysisOrg/cudaWrapper.cpp:  if(useGpuAcceleration()){
Analysis/AnalysisOrg/cudaWrapper.cpp:      gpuBkgFitWorker * worker = new gpuBkgFitWorker(*cit);
Analysis/AnalysisOrg/cudaWrapper.cpp:        cout << "CUDA: cudaWrapper::SpinUpThreads: " << BkgWorkerThreads.size() << " GPU Bkg worker threads created" << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:        cout << "CUDA: cudaWrapper::SpinUpThreads: failed to create worker a thread" << endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:    //no threads created! ToDo: initiate fallback to gpu in caller
Analysis/AnalysisOrg/cudaWrapper.cpp:    /*error state! no GPU workers could be created*/
Analysis/AnalysisOrg/cudaWrapper.cpp:    useGpu = false;
Analysis/AnalysisOrg/cudaWrapper.cpp:    cout << "CUDA ERROR: cudaWrapper::SpinUpThreads: Failed to create any GPU worker threads. cleaning up GPU pipeline and fall back to CPU only Execution!";
Analysis/AnalysisOrg/cudaWrapper.cpp:void cudaWrapper::UnSpinThreads()
Analysis/AnalysisOrg/cudaWrapper.cpp:  size_t numgputhreads = BkgWorkerThreads.size();
Analysis/AnalysisOrg/cudaWrapper.cpp:  for(vector<gpuBkgFitWorker*>::iterator it = BkgWorkerThreads.begin(); it != BkgWorkerThreads.end(); )
Analysis/AnalysisOrg/cudaWrapper.cpp:  if(numgputhreads > 0 ) cout << "CUDA: cudaWrapper::UnSpinThreads: all " << numgputhreads << " GPU worker threads are joined." <<endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:void cudaWrapper::setUpAndStartFlowByFlowHandshakeWorker( const CommandLineOpts &inception_state,
Analysis/AnalysisOrg/cudaWrapper.cpp:void cudaWrapper::joinFlowByFlowHandshakeWorker()
Analysis/AnalysisOrg/cudaWrapper.cpp:    cout << "CUDA: cudaWrapper::joinFlowByFlowHandshakeWorker: GPU-CPU handshake worker thread joined." <<endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:bool cudaWrapper::fullBlockSignalProcessing(BkgModelWorkInfo* bkinfo)
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:    cudaSetDevice(deviceConfig.getFirstValidDevice());
Analysis/AnalysisOrg/cudaWrapper.cpp:    // create static GpuPipeline Object
Analysis/AnalysisOrg/cudaWrapper.cpp:    if(GpuPipeline == NULL)
Analysis/AnalysisOrg/cudaWrapper.cpp:      GpuPipeline = new BkgGpuPipeline(bkinfo, deviceConfig.getFirstValidDevice(),RegionalFitHistory );
Analysis/AnalysisOrg/cudaWrapper.cpp:     GpuPipeline->PerFlowDataUpdate(bkinfo);
Analysis/AnalysisOrg/cudaWrapper.cpp:     //GpuPipeline->InitRegionalParamsAtFirstFlow();
Analysis/AnalysisOrg/cudaWrapper.cpp:     //GpuPipeline->DebugOutputDeviceBuffers();
Analysis/AnalysisOrg/cudaWrapper.cpp:     GpuPipeline->ExecuteGenerateBeadTrace();
Analysis/AnalysisOrg/cudaWrapper.cpp:     GpuPipeline->ExecuteCrudeEmphasisGeneration();
Analysis/AnalysisOrg/cudaWrapper.cpp:     GpuPipeline->ExecuteRegionalFitting();
Analysis/AnalysisOrg/cudaWrapper.cpp:     GpuPipeline->HandleRegionalFittingResults();
Analysis/AnalysisOrg/cudaWrapper.cpp:     GpuPipeline->ExecuteFineEmphasisGeneration();
Analysis/AnalysisOrg/cudaWrapper.cpp:     GpuPipeline->ExecuteTraceLevelXTalk();
Analysis/AnalysisOrg/cudaWrapper.cpp:     GpuPipeline->ExecuteSingleFlowFit();
Analysis/AnalysisOrg/cudaWrapper.cpp:     GpuPipeline->ExecutePostFitSteps();
Analysis/AnalysisOrg/cudaWrapper.cpp:     GpuPipeline->HandleResults(Handshaker->getRingBuffer()); // copy reg_params and single flow fit results to host
Analysis/AnalysisOrg/cudaWrapper.cpp:       std::cout << "CUDA: cudaWrapper: New pipeline encountered issue during" << bkinfo->flow << ". Exiting with error code for retry!" << std::endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:  CUDA_WRONG_TIME_AND_PLACE
Analysis/AnalysisOrg/cudaWrapper.cpp:void cudaWrapper::collectHistroyForRegionalFitting(BkgModelWorkInfo* bkinfo, int flowBlockSize, int extractNumFlows)
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  CUDA_WRONG_TIME_AND_PLACE
Analysis/AnalysisOrg/cudaWrapper.cpp:void cudaWrapper::mirrorDeviceBuffersToHostForSerialization()
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  if(GpuPipeline != NULL){
Analysis/AnalysisOrg/cudaWrapper.cpp:    GpuPipeline->CopySerializationDataFromDeviceToHost();
Analysis/AnalysisOrg/cudaWrapper.cpp:void cudaWrapper::configureKernelExecution(
Analysis/AnalysisOrg/cudaWrapper.cpp:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.cpp:  if(configOpts->gpuMultiFlowFit)
Analysis/AnalysisOrg/cudaWrapper.cpp:    SimpleMultiFitStream::setBeadsPerBlockMultiF(configOpts->gpuThreadsPerBlockMultiFit);
Analysis/AnalysisOrg/cudaWrapper.cpp:    SimpleMultiFitStream::setL1SettingMultiF(configOpts->gpuL1ConfigMultiFit);
Analysis/AnalysisOrg/cudaWrapper.cpp:    SimpleMultiFitStream::setBeadsPerBlockPartialD(configOpts->gpuThreadsPerBlockPartialD);
Analysis/AnalysisOrg/cudaWrapper.cpp:    SimpleMultiFitStream::setL1SettingPartialD(configOpts->gpuL1ConfigPartialD);
Analysis/AnalysisOrg/cudaWrapper.cpp:  if(configOpts->gpuSingleFlowFit)
Analysis/AnalysisOrg/cudaWrapper.cpp:    SimpleSingleFitStream::setBeadsPerBlock(configOpts->gpuThreadsPerBlockSingleFit);
Analysis/AnalysisOrg/cudaWrapper.cpp:    SimpleSingleFitStream::setL1Setting(configOpts->gpuL1ConfigSingleFit);
Analysis/AnalysisOrg/cudaWrapper.cpp:    SimpleSingleFitStream::setFitType(configOpts->gpuSingleFlowFitType);
Analysis/AnalysisOrg/cudaWrapper.cpp:    //SimpleSingleFitStream::setHybridIter(configOpts->gpuHybridIterations);
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaSimpleStreamManager::setNumMaxStreams(configOpts->gpuNumStreams);
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaSimpleStreamManager::setVerbose(configOpts->gpuVerbose);
Analysis/AnalysisOrg/cudaWrapper.cpp:  cudaSimpleStreamExecutionUnit::setVerbose(configOpts->gpuVerbose);
Analysis/AnalysisOrg/cudaWrapper.cpp:  if(configOpts->gpuDevMemoryPerProc > 0){
Analysis/AnalysisOrg/cudaWrapper.cpp:    size_t memToRequest = (size_t) configOpts->gpuDevMemoryPerProc * (1024.0*1024)/configOpts->gpuNumStreams;
Analysis/AnalysisOrg/cudaWrapper.cpp:    if( memToRequest < cudaResourcePool::getRequestDeviceMemory())
Analysis/AnalysisOrg/cudaWrapper.cpp:      cout << "CUDA WARNING: memory provided for fixed allocation " <<  memToRequest/(1024*1024)<< "MB is less than minimum required memory determined via dynamic allocator: " << cudaResourcePool::getRequestDeviceMemory()/(1024*1024) << "MB. This might lead to memory reallocation during runtime." <<endl;
Analysis/AnalysisOrg/cudaWrapper.cpp:    cudaResourcePool::setDeviceMemory(memToRequest);
Analysis/AnalysisOrg/cudaWrapper.cpp:  CUDA_WRONG_TIME_AND_PLACE
Analysis/AnalysisOrg/BkgFitterTracker.h:#include "cudaWrapper.h"
Analysis/AnalysisOrg/BkgFitterTracker.h:  //RingBuffer<float> *ampEstBufferForGPU;
Analysis/AnalysisOrg/BkgFitterTracker.h:  cudaWrapper GpuQueueControl;
Analysis/AnalysisOrg/BkgFitterTracker.h:  //RingBuffer<float>* getRingBuffer() const { return ampEstBufferForGPU; }
Analysis/AnalysisOrg/BkgFitterTracker.h:  void SetUpCpuAndGpuPipelines (BkgModelControlOpts &bkg_control );
Analysis/AnalysisOrg/BkgFitterTracker.h:  void UpdateGPUPipelineExecutionConfiguration(CommandLineOpts & inception_state);
Analysis/AnalysisOrg/BkgFitterTracker.h:  void UpdateAndCheckGPUCommandlineOptions(CommandLineOpts & inception_state);
Analysis/AnalysisOrg/BkgFitterTracker.h:  void SpinnUpGpuThreads();
Analysis/AnalysisOrg/BkgFitterTracker.h:  void UnSpinGpuThreads();
Analysis/AnalysisOrg/BkgFitterTracker.h:  //  void UnSpinMultiFlowFitGpuThreads();
Analysis/AnalysisOrg/BkgFitterTracker.h:  //checks system state and accordingly switches to different GPU pipeline execution mode
Analysis/AnalysisOrg/BkgFitterTracker.h:  void checkAndInitGPUPipelineSwitch(
Analysis/AnalysisOrg/BkgFitterTracker.h:  bool useGpuAcceleration() { return GpuQueueControl.useGpuAcceleration(); }
Analysis/AnalysisOrg/BkgFitterTracker.h:  void ExecuteGPUFlowByFlowSignalProcessing(
Analysis/AnalysisOrg/BkgFitterTracker.h:  void CollectSampleWellsForGPUFlowByFlowSignalProcessing(
Analysis/AnalysisOrg/BkgFitterTracker.h:  void DetermineAndSetGPUAllocationAndKernelParams( BkgModelControlOpts &bkg_control,
Analysis/AnalysisOrg/BkgFitterTracker.h:	    GpuQueueControl;
Analysis/AnalysisOrg/BkgFitterTracker.h:  GpuQueueControl;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:struct BkgFitWorkerGpuInfo
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  int gpu_index;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    GPU_QUEUE,
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    gpuMultiFlowFitting = true;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    gpuSingleFlowFitting = true;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void SetGpuQueue(WorkerInfoQueue* q) { fitting_queues[GPU_QUEUE] = q;}
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void AllocateGpuInfo(int n) { gpu_info.resize(n); }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void turnOnGpuMultiFlowFitting() { gpuMultiFlowFitting = true; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void turnOffGpuMultiFlowFitting() { gpuMultiFlowFitting = false; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void turnOnGpuSingleFlowFitting() { gpuSingleFlowFitting = true; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void turnOffGpuSingleFlowFitting() { gpuSingleFlowFitting = false; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool performGpuMultiFlowFitting() { return gpuMultiFlowFitting; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool performGpuSingleFlowFitting() { return gpuSingleFlowFitting; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  std::vector<BkgFitWorkerGpuInfo>& GetGpuInfo() { return gpu_info; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void SpinUpGPUThreads( struct ComputationPlanner &analysis_compute_plan );
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  static void CreateGpuThreadsForFitType(
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    std::vector<BkgFitWorkerGpuInfo> &gpuInfo,
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    std::vector<int> &gpus
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  WorkerInfoQueue* GetGpuQueue() { return fitting_queues[GPU_QUEUE]; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  // Create array to hold gpu_info
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  std::vector<BkgFitWorkerGpuInfo> gpu_info;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool heterogeneous_computing; // use both gpu and cpu
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool gpuMultiFlowFitting;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool gpuSingleFlowFitting;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    //this is just a handle to the gpu queue so jobs can be handed to this queue if needed
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    WorkerInfoQueue * gpuQueue;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    bool heterogeneousComputing; // use both gpu and cpu
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    bool gpuMultiFlowFitting;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    bool gpuSingleFlowFitting;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    void CreateGpuThreadsForFitType(
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:        std::vector<BkgFitWorkerGpuInfo> &gpuInfo,
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:        std::vector<int> &gpus );
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    gpuQueue = NULL;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    gpuMultiFlowFitting = true;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    gpuSingleFlowFitting = true;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void setGpuQueue(WorkerInfoQueue * GpuQ){gpuQueue = GpuQ;}
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void turnOnGpuMultiFlowFitting() { gpuMultiFlowFitting = true; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void turnOffGpuMultiFlowFitting() { gpuMultiFlowFitting = false; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void turnOnGpuSingleFlowFitting() { gpuSingleFlowFitting = true; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  void turnOffGpuSingleFlowFitting() { gpuSingleFlowFitting = false; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool performGpuMultiFlowFitting() { return gpuMultiFlowFitting; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool performGpuSingleFlowFitting() { return gpuSingleFlowFitting; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  WorkerInfoQueue* GetGpuQueue() { return gpuQueue; }
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  //RingBuffer<float> *gpuAmpEstPerFlow;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h://prototype GPU trace generation whole block on GPU
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:struct BkgModelImgToTraceInfoGPU
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:// between GPU and CPU for writing amplitudes estimates to rawwell buffers
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:/*struct GPUFlowByFlowPipelineInfo
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  int numBkgWorkers_gpu;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  int numGpuWorkers;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool use_gpu_acceleration;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  float gpu_work_load;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool use_all_gpus;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool use_gpu_only_fitting;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool gpu_multiflow_fit;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  bool gpu_singleflow_fit;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    numBkgWorkers_gpu = 0;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    numGpuWorkers = 0;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    use_gpu_acceleration = false;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    gpu_work_load = 0;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    use_all_gpus = false;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    use_gpu_only_fitting = true;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    gpu_multiflow_fit = true;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:    gpu_singleflow_fit = true;
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  & p.numBkgWorkers_gpu
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  & p.use_gpu_acceleration
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  & p.gpu_work_load
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h:  & p.use_all_gpus
Analysis/AnalysisOrg/SignalProcessingFitterQueue.h://bool UseGpuAcceleration(float useGpuFlag);
Analysis/AnalysisOrg/cudaWrapper.h:#ifndef CUDAWRAPPER_H
Analysis/AnalysisOrg/cudaWrapper.h:#define CUDAWRAPPER_H
Analysis/AnalysisOrg/cudaWrapper.h:#include "GpuControlOpts.h"
Analysis/AnalysisOrg/cudaWrapper.h://#define ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.h:class BkgGpuPipeline;
Analysis/AnalysisOrg/cudaWrapper.h:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.h:* class gpuDeviceConfig
Analysis/AnalysisOrg/cudaWrapper.h:* manages the available cuda devices in the system
Analysis/AnalysisOrg/cudaWrapper.h:* allows for specific configurations/selection of GPU resources
Analysis/AnalysisOrg/cudaWrapper.h:* creates CUDA context on selected devices by initializing
Analysis/AnalysisOrg/cudaWrapper.h:class gpuDeviceConfig
Analysis/AnalysisOrg/cudaWrapper.h:  gpuDeviceConfig();
Analysis/AnalysisOrg/cudaWrapper.h:  bool  setValidDevices(std::vector<int> &CmdlineDeviceList, bool useMaxComputeVersionOnly);  //if useMaxComputeVersionOnly == false:  any CUDA compatible device will be used without checking for copute version
Analysis/AnalysisOrg/cudaWrapper.h:* between flow by flow Gpu pipeline and wells file writing
Analysis/AnalysisOrg/cudaWrapper.h:* class gpuBkgFitWorker
Analysis/AnalysisOrg/cudaWrapper.h:* one GPU Worker Thread for the block of 20 flow
Analysis/AnalysisOrg/cudaWrapper.h:* per region GPU pipeline
Analysis/AnalysisOrg/cudaWrapper.h:* the GPU the thread is working on
Analysis/AnalysisOrg/cudaWrapper.h:class gpuBkgFitWorker: protected pThreadWrapper{
Analysis/AnalysisOrg/cudaWrapper.h:  gpuBkgFitWorker(int devId);
Analysis/AnalysisOrg/cudaWrapper.h:  void setFallBackQueue(WorkerInfoQueue * fallbackQueue){errorQueue = fallbackQueue;} //provide queue to put job into if gpu execution fails
Analysis/AnalysisOrg/cudaWrapper.h:* class cudaWrapper
Analysis/AnalysisOrg/cudaWrapper.h:* manages the interface between CPU code and GPU code
Analysis/AnalysisOrg/cudaWrapper.h:* handles gpu configuration and resources
Analysis/AnalysisOrg/cudaWrapper.h:* provides interface for GPU pipelines
Analysis/AnalysisOrg/cudaWrapper.h:class cudaWrapper{
Analysis/AnalysisOrg/cudaWrapper.h:  bool useGpu;
Analysis/AnalysisOrg/cudaWrapper.h:  bool useAllGpus;
Analysis/AnalysisOrg/cudaWrapper.h:  gpuDeviceConfig deviceConfig;
Analysis/AnalysisOrg/cudaWrapper.h:  std::vector<gpuBkgFitWorker*> BkgWorkerThreads;
Analysis/AnalysisOrg/cudaWrapper.h:  GpuControlOpts * configOpts;
Analysis/AnalysisOrg/cudaWrapper.h:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.h:  BkgGpuPipeline * GpuPipeline;
Analysis/AnalysisOrg/cudaWrapper.h:  cudaWrapper();
Analysis/AnalysisOrg/cudaWrapper.h:  ~cudaWrapper();
Analysis/AnalysisOrg/cudaWrapper.h:  //configures the devices and creates contexts through gpuDeviceConfig class
Analysis/AnalysisOrg/cudaWrapper.h:  void configureGpu(BkgModelControlOpts &bkg_control);
Analysis/AnalysisOrg/cudaWrapper.h:  int getNumWorkers(){ return BkgWorkerThreads.size();} //returns actual number of currently active gpu workers
Analysis/AnalysisOrg/cudaWrapper.h:  bool useGpuAcceleration() { return useGpu; }
Analysis/AnalysisOrg/cudaWrapper.h:  bool useFlowByFlowExecution() { return configOpts->gpuFlowByFlowExecution; };
Analysis/AnalysisOrg/cudaWrapper.h:  bool ampGuessOnGpu(){ return (configOpts->gpuSingleFlowFit && configOpts->gpuAmpGuess); }
Analysis/AnalysisOrg/cudaWrapper.h:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.h:#ifdef ION_COMPILE_CUDA
Analysis/AnalysisOrg/cudaWrapper.h:     cout << "STORE STORE STORE cudaWrapper " <<  endl;
Analysis/AnalysisOrg/cudaWrapper.h:#endif // CUDAWRAPPER_H
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:#include "GpuControlOpts.h"
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:void GpuControlOpts::DefaultGpuControl()
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuWorkLoad = 1.0;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuNumStreams = 2;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    //gpuSharedByNumProcs = 0; //0 default nothing done. if provided GPU memory (minus some padding) will be divided by this number
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuDevMemoryPerProc = 0;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuForceMultiFlowFit=false;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuMultiFlowFit = 1;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuThreadsPerBlockMultiFit = 128;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuL1ConfigMultiFit = -1;  // actual default is set hardware specific in MultiFitStream.cu
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuThreadsPerBlockPartialD = 128;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuL1ConfigPartialD = -1;  // actual default is set hardware specific in MultiFitStream.cu
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuSingleFlowFit = 1;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuThreadsPerBlockSingleFit = -1; // actual default is set hardware specific in SingleFitStream.cu
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuL1ConfigSingleFit = -1; // actual default is set hardware specific in SingleFitStream.cu
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    // 0: GaussNewton, 1: LevMar 2:Hybrid (gpuHybridIterations Gauss Newton, then rest LevMar)
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuSingleFlowFitType = 3;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuHybridIterations = 3;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    doGpuOnlyFitting = 1;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuAmpGuess = 1;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuUseAllDevices=false;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuVerbose = false;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuFlowByFlowExecution = false;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuNumHistoryFlows = 10;
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:void GpuControlOpts::PrintHelp()
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     GpuControlOpts\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  printf ("     --gpuworkload           FLOAT             gpu work load [1.0]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  printf ("     --gpuWorkLoad           FLOAT             same as --gpuworkload [1.0]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-verbose           BOOL              gpu verbose [false]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-use-all-devices   BOOL              forces use of all available GPUs passing minimum requirements. may cause non-deterministic results if GPUs have varying compute version [false]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-fitting-only      BOOL              do gpu only fitting [true]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-flow-by-flow      BOOL              use new flow by flow GPU pipeline [false]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-num-history-flows    INT            number of history flows to perform regional fitting\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  printf ("     --gpu-tmidnuc-shift-per-flow     BOOL     do t_mid_nuc shift per flow [true]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-device-ids        INT               gpu device ids []\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-num-streams       INT               gpu num streams [2]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	//printf ("     --gpu-shared-by-num-procs        INT      number of processes sharing the gpu, if provided gpu memory per process will be fixed as: gpu memory/num procs\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-memory-per-proc   INT               request a minimum of <int>MB of device memory per process, if provided overwrites dynamic memory allocation\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-amp-guess         INT               gpu amp guess [1]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-hybrid-fit-iter   INT               gpu hybrid fit iteration [3]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-single-flow-fit   INT               gpu single flow fit [1]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-multi-flow-fit    INT               gpu multi flow fit [1]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-force-multi-flow-fit          BOOL  force multi flow fit execution despite any other setting (for Vadim's use only)\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-single-flow-fit-blocksize     INT   gpu threads per block single fit []\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-multi-flow-fit-blocksize      INT   gpu threads per block multi fit [128]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-single-flow-fit-l1config      INT   gpu L1 config single fit []\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-multi-flow-fit-l1config       INT   gpu L1 config multi fit []\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-single-flow-fit-type          INT   gpu single flow fit type [3]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-partial-deriv-blocksize       INT   gpu threads per block partialD [128]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:	printf ("     --gpu-partial-deriv-l1config        INT   gpu L1 config partialD []\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  printf ("     --gpu-switch-to-flow-by-flow-at     INT   if using flow by flow pipeline switch at flow [20]\n");
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:void GpuControlOpts::SetOpts(OptArgs &opts, Json::Value& json_params)
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuWorkLoad = RetrieveParameterFloat(opts, json_params, '-', "gpuworkload", 1.0);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  if ( ( gpuWorkLoad > 1 ) || ( gpuWorkLoad < 0 ) )
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    fprintf ( stderr, "Option Error: gpuworkload must specify a value between 0 and 1 (%f invalid).\n", gpuWorkLoad );
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuNumStreams = RetrieveParameterInt(opts, json_params, '-', "gpu-num-streams", 2);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  if ( ( gpuNumStreams < 1 ) && ( gpuNumStreams > 16 ) )
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    fprintf ( stderr, "Option Error: gpu-num-streams must specify a value between 1 and 16 (%d invalid).\n", gpuNumStreams );
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuAmpGuess = RetrieveParameterInt(opts, json_params, '-', "gpu-amp-guess", 1);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  if ( gpuAmpGuess != 0 && gpuAmpGuess != 1 )
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    fprintf ( stderr, "Option Error: gpu-amp-guess must be either 0 or 1 (%d invalid).\n",gpuAmpGuess );
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp://  gpuSharedByNumProcs = RetrieveParameterInt(opts, json_params, '-', "gpu-shared-by-num-procs", 0);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp://  if ( gpuSharedByNumProcs < 0  )
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp://    fprintf ( stderr, "Option Error: gpu-shared-by-num-procs must be >= 0, where 0 will do dynamic memory allocation (%d invalid).\n",gpuSharedByNumProcs );
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuDevMemoryPerProc = RetrieveParameterInt(opts, json_params, '-', "gpu-memory-per-proc", 0);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  if ( gpuDevMemoryPerProc < 0  )
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    fprintf ( stderr, "Option Error: gpu-memory-per-proc must be >= 0, where 0 will do dynamic memory allocation (%d invalid).\n",gpuDevMemoryPerProc );
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuSingleFlowFit = RetrieveParameterInt(opts, json_params, '-', "gpu-single-flow-fit", 1);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  if ( gpuSingleFlowFit != 0 && gpuSingleFlowFit != 1 )
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    fprintf ( stderr, "Option Error: gpu-single-flow-fit must be either 0 or 1 (%d invalid).\n", gpuSingleFlowFit );
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuThreadsPerBlockSingleFit = RetrieveParameterInt(opts, json_params, '-', "gpu-single-flow-fit-blocksize", -1);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  if(gpuThreadsPerBlockSingleFit >= 0)
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    if ( gpuThreadsPerBlockSingleFit <= 0 )
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:      fprintf ( stderr, "Option Error: gpu-single-flow-fit-blocksize must be > 0 (%d invalid).\n", gpuThreadsPerBlockSingleFit );
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuL1ConfigSingleFit = RetrieveParameterInt(opts, json_params, '-', "gpu-single-flow-fit-l1config", -1);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuMultiFlowFit = RetrieveParameterInt(opts, json_params, '-', "gpu-multi-flow-fit", 1);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuForceMultiFlowFit = RetrieveParameterBool(opts, json_params, '-', "gpu-force-multi-flow-fit", false);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  if ( gpuMultiFlowFit != 0 && gpuMultiFlowFit != 1 )
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    fprintf ( stderr, "Option Error: gpu-multi-flow-fit must be either 0 or 1 (%d invalid).\n", gpuMultiFlowFit );
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuThreadsPerBlockMultiFit = RetrieveParameterInt(opts, json_params, '-', "gpu-multi-flow-fit-blocksize", 128);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  if ( gpuThreadsPerBlockMultiFit <= 0 )
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    fprintf ( stderr, "Option Error: gpu-multi-flow-fit-blocksize must be > 0 (%d invalid).\n", gpuThreadsPerBlockMultiFit );
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuL1ConfigMultiFit = RetrieveParameterInt(opts, json_params, '-', "gpu-multi-flow-fit-l1config", -1);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuSingleFlowFitType = RetrieveParameterInt(opts, json_params, '-', "gpu-single-flow-fit-type", 3);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuHybridIterations = RetrieveParameterInt(opts, json_params, '-', "gpu-hybrid-fit-iter", 3);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuThreadsPerBlockPartialD = RetrieveParameterInt(opts, json_params, '-', "gpu-partial-deriv-blocksize", 128);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  if ( gpuThreadsPerBlockPartialD <= 0 )
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    fprintf ( stderr, "Option Error: gpu-partial-deriv-blocksize must be > 0 (%d invalid).\n", gpuThreadsPerBlockPartialD );
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuL1ConfigPartialD = RetrieveParameterInt(opts, json_params, '-', "gpu-partial-deriv-l1config", -1);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuUseAllDevices = RetrieveParameterBool(opts, json_params, '-', "gpu-use-all-devices", false);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuVerbose = RetrieveParameterBool(opts, json_params, '-', "gpu-verbose", false);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  RetrieveParameterVectorInt(opts, json_params, '-', "gpu-device-ids", "", deviceIds);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuDeviceIds.push_back(deviceIds[i]);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    std::sort(gpuDeviceIds.begin(), gpuDeviceIds.end());
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  doGpuOnlyFitting = RetrieveParameterBool(opts, json_params, '-', "gpu-fitting-only", true);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  gpuFlowByFlowExecution = RetrieveParameterBool(opts, json_params, '-', "gpu-flow-by-flow", false);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:  if (gpuFlowByFlowExecution) {
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    switchToFlowByFlowAt = RetrieveParameterInt(opts, json_params, '-', "gpu-switch-to-flow-by-flow-at", 20);
Analysis/AnalysisOrg/IO/GpuControlOpts.cpp:    gpuNumHistoryFlows = RetrieveParameterInt(opts, json_params, '-', "gpu-num-history-flows", 10);
Analysis/AnalysisOrg/IO/BkgControlOpts.h:#include "GpuControlOpts.h"
Analysis/AnalysisOrg/IO/BkgControlOpts.h:    GpuControlOpts gpuControl;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:#ifndef GPUCONTROLOPTS_H
Analysis/AnalysisOrg/IO/GpuControlOpts.h:#define GPUCONTROLOPTS_H
Analysis/AnalysisOrg/IO/GpuControlOpts.h:class GpuControlOpts{
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    // commandline options for GPU for background model computation
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    float gpuWorkLoad;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuNumStreams;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    //int gpuSharedByNumProcs;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuDevMemoryPerProc;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    bool gpuForceMultiFlowFit;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuMultiFlowFit;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuThreadsPerBlockMultiFit;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuL1ConfigMultiFit;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuThreadsPerBlockPartialD;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuL1ConfigPartialD;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuSingleFlowFit;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuThreadsPerBlockSingleFit;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuL1ConfigSingleFit;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuSingleFlowFitType;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuHybridIterations;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int doGpuOnlyFitting;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuAmpGuess;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    bool gpuUseAllDevices;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    bool gpuVerbose;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    bool gpuFlowByFlowExecution;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    int gpuNumHistoryFlows;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    // hold the device ids for the gpus to be used for computation
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    std::vector<int> gpuDeviceIds;
Analysis/AnalysisOrg/IO/GpuControlOpts.h:    void DefaultGpuControl(void);
Analysis/AnalysisOrg/IO/GpuControlOpts.h:#endif // GPUCONTROLOPTS_H
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	// GpuControlOpts
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpuworkload"] = VT_FLOAT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpuWorkLoad"] = VT_FLOAT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-num-streams"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-memory-per-proc"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-amp-guess"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-single-flow-fit"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-single-flow-fit-blocksize"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-single-flow-fit-l1config"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-multi-flow-fit"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-force-multi-flow-fit"] = VT_BOOL;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-multi-flow-fit-blocksize"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-multi-flow-fit-l1config"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-single-flow-fit-type"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-hybrid-fit-iter"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-partial-deriv-blocksize"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-partial-deriv-l1config"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-use-all-devices"] = VT_BOOL;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-verbose"] = VT_BOOL;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-device-ids"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-fitting-only"] = VT_BOOL;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-tmidnuc-shift-per-flow"] = VT_BOOL;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-flow-by-flow"] = VT_BOOL;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-switch-to-flow-by-flow-at"] = VT_INT;
Analysis/AnalysisOrg/IO/CommandLineOpts.cpp:	m_opts["gpu-num-history-flows"] = VT_INT;
Analysis/AnalysisOrg/IO/BkgControlOpts.cpp:  gpuControl.DefaultGpuControl();
Analysis/AnalysisOrg/IO/BkgControlOpts.cpp:    gpuControl.PrintHelp();
Analysis/AnalysisOrg/IO/BkgControlOpts.cpp:	gpuControl.SetOpts(opts, json_params);
dbReports/iondb/media/resources/bootstrap-2.1.1-j2/less/responsive-767px-max.less:    -webkit-transform: translate3d(0, 0, 0); // activate the GPU
dbReports/iondb/media/resources/bootstrap/less/responsive-767px-max.less:    -webkit-transform: translate3d(0, 0, 0); // activate the GPU
dbReports/iondb/rundb/migrations/0188_data_update_plan_cmdline_args.py:                        "analysisargs"            : "Analysis --from-beadfind --clonal-filter-bkgmodel on --region-size=216x224 --bkg-bfmask-update off --gpuWorkLoad 1 --total-timeout 600",
dbReports/iondb/rundb/migrations/0188_data_update_plan_cmdline_args.py:                        "thumbnailanalysisargs"   : "Analysis --from-beadfind --clonal-filter-bkgmodel on --region-size=100x100 --bkg-bfmask-update off --gpuWorkLoad 1 --bkg-debug-param 1 --beadfind-thumbnail 1",
dbReports/iondb/rundb/tasks.py:def checkLspciForGpu():
dbReports/iondb/rundb/tasks.py:    errorFileName = "/var/spool/ion/gpuErrors"
dbReports/iondb/rundb/tasks.py:    gpuFound = False
dbReports/iondb/rundb/tasks.py:    # find all the lines containing "nvidia" (case insensitive) and get the rev
dbReports/iondb/rundb/tasks.py:        revNum, startIndex = findNvidiaInLspci(lspciStr, startIndex)
dbReports/iondb/rundb/tasks.py:        # if we didn't find a line containing nvidia, bail
dbReports/iondb/rundb/tasks.py:        gpuFound = True
dbReports/iondb/rundb/tasks.py:        if revNum == "ff":  # When rev == ff, we have lost GPU connection
dbReports/iondb/rundb/tasks.py:    writeError(errorFileName, gpuFound, revsValid)
dbReports/iondb/rundb/tasks.py:    return gpuFound and revsValid
dbReports/iondb/rundb/tasks.py:def findNvidiaInLspci(lspciStr, startIndex):
dbReports/iondb/rundb/tasks.py:    # find the line with the NVIDIA controller information
dbReports/iondb/rundb/tasks.py:    idx = lowStr.find("controller: nvidia", startIndex)
dbReports/iondb/rundb/tasks.py:    # truncate the line with the NVIDIA info
dbReports/iondb/rundb/tasks.py:        nvidiaLine = lspciStr[idx:newline]
dbReports/iondb/rundb/tasks.py:    # extract the rev number from the NVIDIA line
dbReports/iondb/rundb/tasks.py:    beg = nvidiaLine.find(token) + len(token)
dbReports/iondb/rundb/tasks.py:    end = nvidiaLine.find(")", beg)
dbReports/iondb/rundb/tasks.py:    return nvidiaLine[beg:end], newline + 1
dbReports/iondb/rundb/tasks.py:def writeError(errorFileName, gpuFound, allRevsValid):
dbReports/iondb/rundb/tasks.py:        f.write(json.dumps({"gpuFound": gpuFound, "allRevsValid": allRevsValid}))
dbReports/iondb/product_integration/metrics.py:    # GPU errors: /var/spool/ion/gpuErrors
dbReports/iondb/product_integration/metrics.py:    metrics["systemGpuErrors"] = None
dbReports/iondb/product_integration/metrics.py:        with open("/var/spool/ion/gpuErrors") as fp:
dbReports/iondb/product_integration/metrics.py:            metrics["systemGpuErrors"] = json.load(fp)
dbReports/iondb/product_integration/metrics.py:            "Could not read /var/spool/ion/gpuErrors when collecting metrics!"
dbReports/iondb/utils/default_chip_args.py:            "analysisArgs": "Analysis --from-beadfind --clonal-filter-bkgmodel on --region-size=216x224 --bkg-bfmask-update off --gpuWorkLoad 1 --total-timeout 600",
dbReports/iondb/utils/default_chip_args.py:            "thumbnailAnalysisArgs": "Analysis --from-beadfind --clonal-filter-bkgmodel on --region-size=100x100 --bkg-bfmask-update off --gpuWorkLoad 1 --bkg-debug-param 1 --beadfind-thumbnail 1",
dbReports/iondb/bin/startanalysis_batch.py:def generate_report_name(exp, timestring, ebr, gpu, note, analysis_arg):
dbReports/iondb/bin/startanalysis_batch.py:    """ report name: <exp name>_<build num>_<time stamp>_<ebr>_<gpu>_<note>"""
dbReports/iondb/bin/startanalysis_batch.py:        gpu,
dbReports/iondb/bin/startanalysis_batch.py:def generate_post(run_name, timestamp, ebr_opt, gpu_opt, note_opt):
dbReports/iondb/bin/startanalysis_batch.py:    if int(gpu_opt) == 0:
dbReports/iondb/bin/startanalysis_batch.py:        gpu_arg = " --gpuWorkLoad 0"
dbReports/iondb/bin/startanalysis_batch.py:        gpu_str = "noGPU"
dbReports/iondb/bin/startanalysis_batch.py:    elif int(gpu_opt) == 1:
dbReports/iondb/bin/startanalysis_batch.py:        gpu_arg = " --gpuWorkLoad 1"
dbReports/iondb/bin/startanalysis_batch.py:        gpu_str = "GPU"
dbReports/iondb/bin/startanalysis_batch.py:    elif int(gpu_opt) == 2:
dbReports/iondb/bin/startanalysis_batch.py:        gpu_arg = " --sigproc-compute-flow 20,20:1 --gpu-flow-by-flow true --num-regional-samples 200 --gpuWorkLoad 1"
dbReports/iondb/bin/startanalysis_batch.py:        gpu_str = "GPU_newPipeline"
dbReports/iondb/bin/startanalysis_batch.py:        gpu_arg = ""
dbReports/iondb/bin/startanalysis_batch.py:        gpu_str = ""
dbReports/iondb/bin/startanalysis_batch.py:    m = re.search("--gpuWorkLoad.{2}", analysisargs)
dbReports/iondb/bin/startanalysis_batch.py:        amended_analysisargs = re.sub(m.group(0), gpu_arg, analysisargs)
dbReports/iondb/bin/startanalysis_batch.py:        amended_analysisargs = analysisargs + gpu_arg
dbReports/iondb/bin/startanalysis_batch.py:        exp, timestamp, ebr_str, gpu_str, note_opt, amended_analysisargs
dbReports/iondb/bin/startanalysis_batch.py:        "--gpu",
dbReports/iondb/bin/startanalysis_batch.py:        help="enable GPU for pipeline. This is often use to modify the analysis arguments",
dbReports/iondb/bin/startanalysis_batch.py:        cmd_args["gpu"],
plugin/FieldSupport/rndplugins/GBU_HBU_Analysis/run_gbu_calc.sh:  -F <name> Output File name prefix. Default: 'GPU'
plugin/FieldSupport/rndplugins/GBU_HBU_Analysis/run_gbu_calc.sh:FILESTEM="GPU"
plugin/CustomerSupportArchive/autoCal/tools/instrument.py:    #flowscript = 'Script_WT_Pressurize-Before-OpenClamp'
plugin/CustomerSupportArchive/autoCal/tools/instrument.py:    #expname = 'Pressurize-Before-OpenClamp'
plugin/CustomerSupportArchive/autoCal/tools/instrument.py:    cmdcontrol( 'OpenClamp', 1 )
plugin/CustomerSupportArchive/autoCal/tools/explog.py:            valk_match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pc1>[0-9\.\-]+) (?P<pc2>[0-9\.\-]+) (?P<pc3>[0-9\.\-]+) (?P<pc4>[0-9\.\-]+) (?P<pm1>[0-9\.\-]+) (?P<pm2>[0-9\.\-]+) (?P<pm3>[0-9\.\-]+) (?P<pm4>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)(?P<na999>.+Vref=)(?P<vref>[0-9\.\-]+)'
plugin/CustomerSupportArchive/autoCal/tools/explog.py:            valk_string_2 = r'(?P<flow>.+?): Pressure=(?P<p0>[\d.-]+) (?P<p1>[\d.-]+) Temp=(?P<t0>[\d.-]+) (?P<t1>[\d.-]+) (?P<t2>[\d.-]+) (?P<t3>[\d.-]+) dac_start_sig=(?P<dac>[\d.-]+) avg=(?P<dc>[\d.-]+) time=(?P<t>[\d:]+) fpgaTemp=(?P<t4>[\d.-]+) (?P<t5>[\d.-]+) chipTemp=(?P<t6>[\d.-]+) (?P<t7>[\d.-]+) (?P<t8>[\d.-]+) (?P<t9>[\d.-]+) (?P<t10>[\d.-]+)[\w\s=]+cpuTemp=(?P<t11>[\d.-]+) (?P<t12>[\d.-]+) heater=[\d.-]* cooler=[\d.-]* gpuTemp=(?P<t13>[\d.-]+) diskPerFree=(?P<dpf>[\d.-]*) FACC_Offset=(?P<faccOff>[\d.-]*).+FACC=(?P<facc>[\d.-]*).+Pinch=(?P<pc1>[\d.-]*) (?P<pc2>[\d.-]*) (?P<pc3>[\d.-]*) (?P<pc4>[\d.-]*) (?P<pm1>[\d.-]*) (?P<pm2>[\d.-]*) (?P<pm3>[\d.-]*) (?P<pm4>[\d.-]*).+FR=(?P<flowrate>[\d.-]*).+FTemp=(?P<flowtemp>[\d.-]*).+Vref=(?P<vref>[\d.-]*)'
plugin/CustomerSupportArchive/autoCal/tools/explog.py:                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pinch1>[0-9\.\-]+) (?P<pinch2>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)'
plugin/CustomerSupportArchive/autoCal/tools/explog.py:                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+) (?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)'
plugin/CustomerSupportArchive/autoCal/tools/explog.py:            gpuT    = [] # gpu temperature
plugin/CustomerSupportArchive/autoCal/tools/explog.py:                    gpuT.append  (   int( m['t13'] ) )
plugin/CustomerSupportArchive/autoCal/tools/explog.py:            self.gpuT  = np.array( gpuT  , np.int16 )
plugin/CustomerSupportArchive/autoCal/tools/explog.py:            self.calc_flow_metrics( self.gpuT  , 'GPUTemperature'     )
plugin/CustomerSupportArchive/autoCal/tools/explog.py:            plt.plot   ( self.flowax , self.gpuT  , label='GPU'   )
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/instrument.py:    #flowscript = 'Script_WT_Pressurize-Before-OpenClamp'
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/instrument.py:    #expname = 'Pressurize-Before-OpenClamp'
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/instrument.py:    cmdcontrol( 'OpenClamp', 1 )
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/explog.py:            valk_match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pc1>[0-9\.\-]+) (?P<pc2>[0-9\.\-]+) (?P<pc3>[0-9\.\-]+) (?P<pc4>[0-9\.\-]+) (?P<pm1>[0-9\.\-]+) (?P<pm2>[0-9\.\-]+) (?P<pm3>[0-9\.\-]+) (?P<pm4>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)(?P<na999>.+Vref=)(?P<vref>[0-9\.\-]+)'
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/explog.py:            valk_string_2 = r'(?P<flow>.+?): Pressure=(?P<p0>[\d.-]+) (?P<p1>[\d.-]+) Temp=(?P<t0>[\d.-]+) (?P<t1>[\d.-]+) (?P<t2>[\d.-]+) (?P<t3>[\d.-]+) dac_start_sig=(?P<dac>[\d.-]+) avg=(?P<dc>[\d.-]+) time=(?P<t>[\d:]+) fpgaTemp=(?P<t4>[\d.-]+) (?P<t5>[\d.-]+) chipTemp=(?P<t6>[\d.-]+) (?P<t7>[\d.-]+) (?P<t8>[\d.-]+) (?P<t9>[\d.-]+) (?P<t10>[\d.-]+)[\w\s=]+cpuTemp=(?P<t11>[\d.-]+) (?P<t12>[\d.-]+) heater=[\d.-]* cooler=[\d.-]* gpuTemp=(?P<t13>[\d.-]+) diskPerFree=(?P<dpf>[\d.-]*) FACC_Offset=(?P<faccOff>[\d.-]*).+FACC=(?P<facc>[\d.-]*).+Pinch=(?P<pc1>[\d.-]*) (?P<pc2>[\d.-]*) (?P<pc3>[\d.-]*) (?P<pc4>[\d.-]*) (?P<pm1>[\d.-]*) (?P<pm2>[\d.-]*) (?P<pm3>[\d.-]*) (?P<pm4>[\d.-]*).+FR=(?P<flowrate>[\d.-]*).+FTemp=(?P<flowtemp>[\d.-]*).+Vref=(?P<vref>[\d.-]*)'
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/explog.py:                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pinch1>[0-9\.\-]+) (?P<pinch2>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)'
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/explog.py:                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+) (?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)'
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/explog.py:            gpuT    = [] # gpu temperature
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/explog.py:                    gpuT.append  (   int( m['t13'] ) )
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/explog.py:            self.gpuT  = np.array( gpuT  , np.int16 )
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/explog.py:            self.calc_flow_metrics( self.gpuT  , 'GPUTemperature'     )
plugin/CustomerSupportArchive/ValkyrieWorkflow/tools/explog.py:            plt.plot   ( self.flowax , self.gpuT  , label='GPU'   )
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/instrument.py:    #flowscript = 'Script_WT_Pressurize-Before-OpenClamp'
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/instrument.py:    #expname = 'Pressurize-Before-OpenClamp'
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/instrument.py:    cmdcontrol( 'OpenClamp', 1 )
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/explog.py:            valk_match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pc1>[0-9\.\-]+) (?P<pc2>[0-9\.\-]+) (?P<pc3>[0-9\.\-]+) (?P<pc4>[0-9\.\-]+) (?P<pm1>[0-9\.\-]+) (?P<pm2>[0-9\.\-]+) (?P<pm3>[0-9\.\-]+) (?P<pm4>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)(?P<na999>.+Vref=)(?P<vref>[0-9\.\-]+)'
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/explog.py:            valk_string_2 = r'(?P<flow>.+?): Pressure=(?P<p0>[\d.-]+) (?P<p1>[\d.-]+) Temp=(?P<t0>[\d.-]+) (?P<t1>[\d.-]+) (?P<t2>[\d.-]+) (?P<t3>[\d.-]+) dac_start_sig=(?P<dac>[\d.-]+) avg=(?P<dc>[\d.-]+) time=(?P<t>[\d:]+) fpgaTemp=(?P<t4>[\d.-]+) (?P<t5>[\d.-]+) chipTemp=(?P<t6>[\d.-]+) (?P<t7>[\d.-]+) (?P<t8>[\d.-]+) (?P<t9>[\d.-]+) (?P<t10>[\d.-]+)[\w\s=]+cpuTemp=(?P<t11>[\d.-]+) (?P<t12>[\d.-]+) heater=[\d.-]* cooler=[\d.-]* gpuTemp=(?P<t13>[\d.-]+) diskPerFree=(?P<dpf>[\d.-]*) FACC_Offset=(?P<faccOff>[\d.-]*).+FACC=(?P<facc>[\d.-]*).+Pinch=(?P<pc1>[\d.-]*) (?P<pc2>[\d.-]*) (?P<pc3>[\d.-]*) (?P<pc4>[\d.-]*) (?P<pm1>[\d.-]*) (?P<pm2>[\d.-]*) (?P<pm3>[\d.-]*) (?P<pm4>[\d.-]*).+FR=(?P<flowrate>[\d.-]*).+FTemp=(?P<flowtemp>[\d.-]*).+Vref=(?P<vref>[\d.-]*)'
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/explog.py:                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pinch1>[0-9\.\-]+) (?P<pinch2>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)'
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/explog.py:                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+) (?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)'
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/explog.py:            gpuT    = [] # gpu temperature
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/explog.py:                    gpuT.append  (   int( m['t13'] ) )
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/explog.py:            self.gpuT  = np.array( gpuT  , np.int16 )
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/explog.py:            self.calc_flow_metrics( self.gpuT  , 'GPUTemperature'     )
plugin/CustomerSupportArchive/NucStepSpatialV2/tools/explog.py:            plt.plot   ( self.flowax , self.gpuT  , label='GPU'   )
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/instrument.py:    #flowscript = 'Script_WT_Pressurize-Before-OpenClamp'
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/instrument.py:    #expname = 'Pressurize-Before-OpenClamp'
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/instrument.py:    cmdcontrol( 'OpenClamp', 1 )
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/explog.py:            valk_match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pc1>[0-9\.\-]+) (?P<pc2>[0-9\.\-]+) (?P<pc3>[0-9\.\-]+) (?P<pc4>[0-9\.\-]+) (?P<pm1>[0-9\.\-]+) (?P<pm2>[0-9\.\-]+) (?P<pm3>[0-9\.\-]+) (?P<pm4>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)(?P<na999>.+Vref=)(?P<vref>[0-9\.\-]+)'
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/explog.py:            valk_string_2 = r'(?P<flow>.+?): Pressure=(?P<p0>[\d.-]+) (?P<p1>[\d.-]+) Temp=(?P<t0>[\d.-]+) (?P<t1>[\d.-]+) (?P<t2>[\d.-]+) (?P<t3>[\d.-]+) dac_start_sig=(?P<dac>[\d.-]+) avg=(?P<dc>[\d.-]+) time=(?P<t>[\d:]+) fpgaTemp=(?P<t4>[\d.-]+) (?P<t5>[\d.-]+) chipTemp=(?P<t6>[\d.-]+) (?P<t7>[\d.-]+) (?P<t8>[\d.-]+) (?P<t9>[\d.-]+) (?P<t10>[\d.-]+)[\w\s=]+cpuTemp=(?P<t11>[\d.-]+) (?P<t12>[\d.-]+) heater=[\d.-]* cooler=[\d.-]* gpuTemp=(?P<t13>[\d.-]+) diskPerFree=(?P<dpf>[\d.-]*) FACC_Offset=(?P<faccOff>[\d.-]*).+FACC=(?P<facc>[\d.-]*).+Pinch=(?P<pc1>[\d.-]*) (?P<pc2>[\d.-]*) (?P<pc3>[\d.-]*) (?P<pc4>[\d.-]*) (?P<pm1>[\d.-]*) (?P<pm2>[\d.-]*) (?P<pm3>[\d.-]*) (?P<pm4>[\d.-]*).+FR=(?P<flowrate>[\d.-]*).+FTemp=(?P<flowtemp>[\d.-]*).+Vref=(?P<vref>[\d.-]*)'
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/explog.py:                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pinch1>[0-9\.\-]+) (?P<pinch2>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)'
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/explog.py:                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+) (?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)'
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/explog.py:            gpuT    = [] # gpu temperature
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/explog.py:                    gpuT.append  (   int( m['t13'] ) )
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/explog.py:            self.gpuT  = np.array( gpuT  , np.int16 )
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/explog.py:            self.calc_flow_metrics( self.gpuT  , 'GPUTemperature'     )
plugin/CustomerSupportArchive/Lane_Diagnostics/tools/explog.py:            plt.plot   ( self.flowax , self.gpuT  , label='GPU'   )
plugin/CustomerSupportArchive/chipDiagnostics/tools/instrument.py:    #flowscript = 'Script_WT_Pressurize-Before-OpenClamp'
plugin/CustomerSupportArchive/chipDiagnostics/tools/instrument.py:    #expname = 'Pressurize-Before-OpenClamp'
plugin/CustomerSupportArchive/chipDiagnostics/tools/instrument.py:    cmdcontrol( 'OpenClamp', 1 )
plugin/CustomerSupportArchive/chipDiagnostics/tools/explog.py:            valk_match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pc1>[0-9\.\-]+) (?P<pc2>[0-9\.\-]+) (?P<pc3>[0-9\.\-]+) (?P<pc4>[0-9\.\-]+) (?P<pm1>[0-9\.\-]+) (?P<pm2>[0-9\.\-]+) (?P<pm3>[0-9\.\-]+) (?P<pm4>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)(?P<na999>.+Vref=)(?P<vref>[0-9\.\-]+)'
plugin/CustomerSupportArchive/chipDiagnostics/tools/explog.py:            valk_string_2 = r'(?P<flow>.+?): Pressure=(?P<p0>[\d.-]+) (?P<p1>[\d.-]+) Temp=(?P<t0>[\d.-]+) (?P<t1>[\d.-]+) (?P<t2>[\d.-]+) (?P<t3>[\d.-]+) dac_start_sig=(?P<dac>[\d.-]+) avg=(?P<dc>[\d.-]+) time=(?P<t>[\d:]+) fpgaTemp=(?P<t4>[\d.-]+) (?P<t5>[\d.-]+) chipTemp=(?P<t6>[\d.-]+) (?P<t7>[\d.-]+) (?P<t8>[\d.-]+) (?P<t9>[\d.-]+) (?P<t10>[\d.-]+)[\w\s=]+cpuTemp=(?P<t11>[\d.-]+) (?P<t12>[\d.-]+) heater=[\d.-]* cooler=[\d.-]* gpuTemp=(?P<t13>[\d.-]+) diskPerFree=(?P<dpf>[\d.-]*) FACC_Offset=(?P<faccOff>[\d.-]*).+FACC=(?P<facc>[\d.-]*).+Pinch=(?P<pc1>[\d.-]*) (?P<pc2>[\d.-]*) (?P<pc3>[\d.-]*) (?P<pc4>[\d.-]*) (?P<pm1>[\d.-]*) (?P<pm2>[\d.-]*) (?P<pm3>[\d.-]*) (?P<pm4>[\d.-]*).+FR=(?P<flowrate>[\d.-]*).+FTemp=(?P<flowtemp>[\d.-]*).+Vref=(?P<vref>[\d.-]*)'
plugin/CustomerSupportArchive/chipDiagnostics/tools/explog.py:                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pinch1>[0-9\.\-]+) (?P<pinch2>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)'
plugin/CustomerSupportArchive/chipDiagnostics/tools/explog.py:                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+) (?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)'
plugin/CustomerSupportArchive/chipDiagnostics/tools/explog.py:            gpuT    = [] # gpu temperature
plugin/CustomerSupportArchive/chipDiagnostics/tools/explog.py:                    gpuT.append  (   int( m['t13'] ) )
plugin/CustomerSupportArchive/chipDiagnostics/tools/explog.py:            self.gpuT  = np.array( gpuT  , np.int16 )
plugin/CustomerSupportArchive/chipDiagnostics/tools/explog.py:            self.calc_flow_metrics( self.gpuT  , 'GPUTemperature'     )
plugin/CustomerSupportArchive/chipDiagnostics/tools/explog.py:            plt.plot   ( self.flowax , self.gpuT  , label='GPU'   )
tsconfig/ion_tsconfig/TSconfig.py:    "ion-gpu",
tsconfig/ts_functions:#postfix shared/procmail boolean false
tsconfig/ts_functions:    # qsub -l nodes=1:ppn=4:gpus=1
tsconfig/iontorrent_master.in:    - ion-gpu
tsconfig/bin/grp_configuration_test:        ion-gpu
tsconfig/bin/ion:ion-gpu
tsconfig/bin/grp_validate.sh:$PKG_VER_CMD "ion-gpu|tail -1" 2>> ${ERROR_LOG} 1>> ${LOG_LOG}
tsconfig/ansible/roles/common/tasks/main.yml:- name: solves the falsely reported DBE from GPU driver
tsconfig/ansible/group_vars/iontorrent_computes:    - ion-gpu
LICENSE.txt:## Cuda Toolkit
LICENSE.txt:Important Notice READ CAREFULLY: This Software License Agreement ("Agreement") for NVIDIA CUDA Toolkit, including computer software and associated documentation ("Software"), is the Agreement which governs use of the SOFTWARE of NVIDIA Corporation and its subsidiaries ("NVIDIA") downloadable herefrom. By downloading, installing, copying, or otherwise using the SOFTWARE, You (as defined below) agree to be bound by the terms of this Agreement. If You do not agree to the terms of this Agreement, do not download the SOFTWARE. Recitals Use of NVIDIA's SOFTWARE requires three elements: the SOFTWARE, an NVIDIA GPU or application processor ("NVIDIA Hardware"), and a computer system. The SOFTWARE is protected by copyright laws and international copyright treaties, as well as other intellectual property laws and treaties. The SOFTWARE is not sold, and instead is only licensed for Your use, strictly in accordance with this Agreement. The NVIDIA Hardware is protected by various patents, and is sold, but this Agreement does not cover the sale or use of such hardware, since it may not necessarily be sold as a package with the SOFTWARE. This Agreement sets forth the terms and conditions of the SOFTWARE only.
LICENSE.txt:  1.1.3. Software "SOFTWARE" shall mean the deliverables provided pursuant to this Agreement. SOFTWARE may be provided in either source or binary form, at NVIDIA's discretion.
LICENSE.txt:  1.2.1. Rights and Limitations of Grant Provided that Licensee complies with the terms of this Agreement, NVIDIA hereby grants Licensee the following limited, non-exclusive, non-transferable, non-sublicensable (except as expressly permitted otherwise for Redistributable Software in Section 1.2.1.1 and Section 1.2.1.3 of this Agreement) right to use the SOFTWARE -- and, if the SOFTWARE is provided in source form, to compile the SOFTWARE -- with the following limitations:
LICENSE.txt:  1.2.1.1. Redistribution Rights Licensee may transfer, redistribute, and sublicense certain files of the Redistributable SOFTWARE, as defined in Attachment A of this Agreement, provided, however, that (a) the Redistributable SOFTWARE shall be distributed solely in binary form to Licensee's licensees ("Customers") only as a component of Licensee's own software products (each, a "Licensee Application"); (b) Licensee shall design the Licensee Application such that the Redistributable SOFTWARE files are installed only in a private (non-shared) directory location that is used only by the Licensee Application; (C) Licensee shall obtain each Customer's written or clickwrap agreement to the license terms under a written, legally enforceable agreement that has the effect of protecting the SOFTWARE and the rights of NVIDIA under terms no less restrictive than this Agreement.
LICENSE.txt:  1.2.1.3. Further Redistribution Rights Subject to the terms and conditions of the Agreement, Licensee may authorize Customers to further redistribute the Redistributable SOFTWARE that such Customers receive as part of the Licensee Application, solely in binary form, provided, however, that Licensee shall require in their standard software license agreements with Customers that all such redistributions must be made pursuant to a license agreement that has the effect of protecting the SOFTWARE and the rights of NVIDIA whose terms and conditions are at least as restrictive as those in the applicable Licensee software license agreement covering the Licensee Application. For avoidance of doubt, termination of this Agreement shall not affect rights previously granted by Licensee to its Customers under this Agreement to the extent validly granted to Customers under Section 1.2.1.1.
LICENSE.txt:1.3. Term and Termination This Agreement will continue in effect for two (2) years ("Initial Term") after Your initial download and use of the SOFTWARE, subject to the exclusive right of NVIDIA to terminate as provided herein. The term of this Agreement will automatically renew for successive one (1) year renewal terms after the Initial Term, unless either party provides to the other party at least three (3) months prior written notice of termination before the end of the applicable renewal term. This Agreement will automatically terminate if Licensee fails to comply with any of the terms and conditions hereof. In such event, Licensee must destroy all copies of the SOFTWARE and all of its component parts. Defensive Suspension If Licensee commences or participates in any legal proceeding against NVIDIA, then NVIDIA may, in its sole discretion, suspend or terminate all license grants and any other rights provided under this Agreement during the pendency of such legal proceedings.
LICENSE.txt:1.4. Copyright All rights, title, interest and copyrights in and to the SOFTWARE (including but not limited to all images, photographs, animations, video, audio, music, text, and other information incorporated into the SOFTWARE), the accompanying printed materials, and any copies of the SOFTWARE, are owned by NVIDIA, or its suppliers. The SOFTWARE is protected by copyright laws and international treaty provisions. Accordingly, Licensee is required to treat the SOFTWARE like any other copyrighted material, except as otherwise allowed pursuant to this Agreement and that it may make one copy of the SOFTWARE solely for backup or archive purposes. RESTRICTED RIGHTS NOTICE. Software has been developed entirely at private expense and is commercial computer software provided with RESTRICTED RIGHTS. Use, duplication or disclosure by the U.S. Government or a U.S. Government subcontractor is subject to the restrictions set forth in the Agreement under which Software was obtained pursuant to DFARS 227.7202-3(a) or as set forth in subparagraphs (C)(1) and (2) of the Commercial Computer Software - Restricted Rights clause at FAR 52.227-19, as applicable. Contractor/manufacturer is NVIDIA, 2701 San Tomas Expressway, Santa Clara, CA 95050.
LICENSE.txt:	1.6.1. No Warranties TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE SOFTWARE IS PROVIDED "AS IS" AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NONINFRINGEMENT.
LICENSE.txt:	1.6.2. No Liability for Consequential Damages TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE THE SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
LICENSE.txt:	1.6.3. No Support . NVIDIA has no obligation to support or to provide any updates of the Software.
LICENSE.txt:1.7.1. Feedback Notwithstanding any Non-Disclosure Agreement executed by and between the parties, the parties agree that in the event Licensee or NVIDIA provides Feedback (as defined below) to the other party on how to design, implement, or improve the SOFTWARE or Licensee's product(s) for use with the SOFTWARE, the following terms and conditions apply the Feedback:
LICENSE.txt:1.7.1.1. Exchange of Feedback Both parties agree that neither party has an obligation to give the other party any suggestions, comments or other feedback, whether verbally or in written or source code form, relating to (i) the SOFTWARE; (ii) Licensee's products; (iii) Licensee's use of the SOFTWARE; or (iv) optimization/interoperability of Licensee's product with the SOFTWARE (collectively defined as "Feedback"). In the event either party provides Feedback to the other party, the party receiving the Feedback may use any Feedback that the other party voluntarily provides to improve the (i) SOFTWARE or other related NVIDIA technologies, respectively for the benefit of NVIDIA; or (ii) Licensee's product or other related Licensee technologies, respectively for the benefit of Licensee. Accordingly, if either party provides Feedback to the other party, both parties agree that the other party and its respective licensees may freely use, reproduce, license, distribute, and otherwise commercialize the Feedback in the (i) SOFTWARE or other related technologies; or (ii) Licensee's products or other related technologies, respectively, without the payment of any royalties or fees.
LICENSE.txt:1.7.1.2. Residual Rights Licensee agrees that NVIDIA shall be free to use any general knowledge, skills and experience, (including, but not limited to, ideas, concepts, know-how, or techniques) ("Residuals"), contained in the (i) Feedback provided by Licensee to NVIDIA; (ii) Licensee's products shared or disclosed to NVIDIA in connection with the Feedback; or (C) Licensee's confidential information voluntarily provided to NVIDIA in connection with the Feedback, which are retained in the memories of NVIDIA's employees, agents, or contractors who have had access to such Residuals. Subject to the terms and conditions of this Agreement, NVIDIA's employees, agents, or contractors shall not be prevented from using Residuals as part of such employee's, agent's or contractor's general knowledge, skills, experience, talent, and/or expertise. NVIDIA shall not have any obligation to limit or restrict the assignment of such employees, agents or contractors or to pay royalties for any work resulting from the use of Residuals.
LICENSE.txt:1.7.2. Freedom of Action Licensee agrees that this Agreement is nonexclusive and NVIDIA may currently or in the future be developing software, other technology or confidential information internally, or receiving confidential information from other parties that maybe similar to the Feedback and Licensee's confidential information (as provided in Section 1.7.1.2 above), which may be provided to NVIDIA in connection with Feedback by Licensee. Accordingly, Licensee agrees that nothing in this Agreement will be construed as a representation or inference that NVIDIA will not develop, design, manufacture, acquire, market products, or have products developed, designed, manufactured, acquired, or marketed for NVIDIA, that compete with the Licensee's products or confidential information.
LICENSE.txt:1.7.3. No Implied Licenses Under no circumstances should anything in this Agreement be construed as NVIDIA granting by implication, estoppel or otherwise, (i) a license to any NVIDIA product or technology other than the SOFTWARE; or (ii) any additional license rights for the SOFTWARE other than the licenses expressly granted in this Agreement. If any provision of this Agreement is inconsistent with, or cannot be fully enforced under, the law, such provision will be construed as limited to the extent necessary to be consistent with and fully enforceable under the law. This Agreement is the final, complete and exclusive agreement between the parties relating to the subject matter hereof, and supersedes all prior or contemporaneous understandings and agreements relating to such subject matter, whether oral or written. This Agreement may only be modified in writing signed by an authorized officer of NVIDIA. Licensee agrees that it will not ship, transfer or export the SOFTWARE into any country, or use the SOFTWARE in any manner, prohibited by the United States Bureau of Industry and Security or any export laws, restrictions or regulations. The parties agree that the following sections of the Agreement will survive the termination of the License: Section 1.2.1.4, Section 1.4, Section 1.5, Section 1.6, and Section 1.7.
LICENSE.txt:1.8. Attachment A Redistributable Software In connection with Section 1.2.1.1 of this Agreement, the following files may be redistributed with software applications developed by Licensee, including certain variations of these files that have version number or architecture specific information NVIDIA CUDA Toolkit License Agreement www.nvidia.com End User License Agreements (EULA) DR-06739-001_v01_v8.0 | 9 embedded in the file name - as an example only, for release version 6.0 of the 64-bit Windows software, the file cudart64_60.dll is redistributable.
LICENSE.txt:Component : CUDA Runtime Windows : cudart.dll, cudart_static.lib, cudadevrt.lib Mac OSX : libcudart.dylib, libcudart_static.a, libcudadevrt.a Linux : libcudart.so, libcudart_static.a, libcudadevrt.a Android : libcudart.so, libcudart_static.a, libcudadevrt.a Component : CUDA FFT Library Windows : cufft.dll, cufftw.dll Mac OSX : libcufft.dylib, libcufft_static.a, libcufftw.dylib, libcufftw_static.a Linux : libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a Android : libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a Component : CUDA BLAS Library Windows : cublas.dll, cublas_device.lib Mac OSX : libcublas.dylib, libcublas_static.a, libcublas_device.a Linux : libcublas.so, libcublas_static.a, libcublas_device.a Android : libcublas.so, libcublas_static.a, libcublas_device.a Component : NVIDIA "Drop-in" BLAS Library Windows : nvblas.dll Mac OSX : libnvblas.dylib Linux : libnvblas.so Component : CUDA Sparse Matrix Library Windows : cusparse.dll Mac OSX : libcusparse.dylib, libcusparse_static.a Linux : libcusparse.so, libcusparse_static.a Android : libcusparse.so, libcusparse_static.a Component : CUDA Linear Solver Library Windows : cusolver.dll Mac OSX : libcusolver.dylib, libcusolver_static.a Linux : libcusolver.so, libcusolver_static.a Android : libcusolver.so, libcusolver_static.a Component : CUDA Random Number Generation Library Windows : curand.dll Mac OSX : libcurand.dylib, libcurand_static.a Linux : libcurand.so, libcurand_static.a Android : libcurand.so, libcurand_static.a Component : NVIDIA Performance Primitives Library Windows : nppc.dll, nppi.dll, npps.dll Mac OSX : libnppc.dylib, libnppi.dylib, libnpps.dylib, libnppc_static.a, libnpps_static.a, libnppi_static.a Linux : libnppc.so, libnppi.so, libnpps.so, libnppc_static.a, libnpps_static.a, libnppi_static.a Android : libnppc.so, libnppi.so, libnpps.so, libnppc_static.a, libnpps_static.a, libnppi_static.a Component : Internal common library required for statically linking to cuBLAS, cuSPARSE, cuFFT, cuRAND and NPP Mac OSX : libculibos.a Linux : libculibos.a Component : NVIDIA Runtime Compilation Library Windows : nvrtc.dll, nvrtc-builtins.dll Mac OSX : libnvrtc.dylib, libnvrtc-builtins.dylib Linux : libnvrtc.so, libnvrtc-builtins.so Component : NVIDIA Optimizing Compiler Library Windows : nvvm.dll Mac OSX : libnvvm.dylib Linux : libnvvm.so Component : NVIDIA Common Device Math Functions Library Windows : libdevice.compute_20.bc, libdevice.compute_30.bc, libdevice.compute_35.bc Mac OSX : libdevice.compute_20.bc, libdevice.compute_30.bc, libdevice.compute_35.bc Linux : libdevice.compute_20.bc, libdevice.compute_30.bc, libdevice.compute_35.bc Component : CUDA Occupancy Calculation Header Library All : cuda_occupancy.h Component : Profiling Tools Interface Library Windows : cupti.dll Mac OSX : libcupti.dylib Linux : libcupti.so
LICENSE.txt:1. Licensee's use of the GDB third party component is subject to the terms and conditions of GNU GPL v3: This product includes copyrighted third-party software licensed under the terms of the GNU General Public License v3 ("GPL v3"). All third-party software packages are copyright by their respective authors. GPL v3 terms and conditions are hereby incorporated into the Agreement by this reference: http://www.gnu.org/licenses/gpl.txt Consistent with these licensing requirements, the software listed below is provided under the terms of the specified open source software licenses. To obtain source code for software provided under licenses that require redistribution of source code, including the GNU General Public License (GPL) and GNU Lesser General Public License (LGPL), contact oss-requests@nvidia.com. This offer is valid for a period of three (3) years from the date of the distribution of this product by NVIDIA CORPORATION. Component License CUDA-GDB GPL v3
LICENSE.txt:Copyright (C) 2000-2020, Intel Corporation, all rights reserved. Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved. Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
gpu/bandwidthTest.cu: * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
gpu/bandwidthTest.cu: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/bandwidthTest.cu: * This is a simple test program to measure the memcopy bandwidth of the GPU.
gpu/bandwidthTest.cu:// CUDA runtime
gpu/bandwidthTest.cu:#include <cuda_runtime.h>
gpu/bandwidthTest.cu:#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
gpu/bandwidthTest.cu:#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
gpu/bandwidthTest.cu:#include <cuda.h>
gpu/bandwidthTest.cu:static const char *sSDKsample = "CUDA Bandwidth Test";
gpu/bandwidthTest.cu:static bool bDontUseGPUTiming;
gpu/bandwidthTest.cu:        checkCudaErrors(cudaSetDevice(0));
gpu/bandwidthTest.cu:    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");
gpu/bandwidthTest.cu:        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
gpu/bandwidthTest.cu:        if (error_id != cudaSuccess)
gpu/bandwidthTest.cu:            printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
gpu/bandwidthTest.cu:                printf("\n!!!!!Invalid GPU number %d given hence default gpu %d will be used !!!!!\n", startDevice,0);
gpu/bandwidthTest.cu:        cudaDeviceProp deviceProp;
gpu/bandwidthTest.cu:        cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, currentDevice);
gpu/bandwidthTest.cu:        if (error_id == cudaSuccess)
gpu/bandwidthTest.cu:            if (deviceProp.computeMode == cudaComputeModeProhibited)
gpu/bandwidthTest.cu:                fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
gpu/bandwidthTest.cu:                checkCudaErrors(cudaSetDevice(currentDevice));
gpu/bandwidthTest.cu:            printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
gpu/bandwidthTest.cu:            checkCudaErrors(cudaSetDevice(currentDevice));
gpu/bandwidthTest.cu:#if CUDART_VERSION >= 2020
gpu/bandwidthTest.cu:        bDontUseGPUTiming = true;
gpu/bandwidthTest.cu:    // Ensure that we reset all CUDA Devices in question
gpu/bandwidthTest.cu:        cudaSetDevice(nDevice);
gpu/bandwidthTest.cu:        cudaSetDevice(currentDevice);
gpu/bandwidthTest.cu:        cudaSetDevice(currentDevice);
gpu/bandwidthTest.cu:    cudaEvent_t start, stop;
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventCreate(&start));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventCreate(&stop));
gpu/bandwidthTest.cu:#if CUDART_VERSION >= 2020
gpu/bandwidthTest.cu:        checkCudaErrors(cudaHostAlloc((void **)&h_idata, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
gpu/bandwidthTest.cu:        checkCudaErrors(cudaHostAlloc((void **)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
gpu/bandwidthTest.cu:        checkCudaErrors(cudaMallocHost((void **)&h_idata, memSize));
gpu/bandwidthTest.cu:        checkCudaErrors(cudaMallocHost((void **)&h_odata, memSize));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaMalloc((void **) &d_idata, memSize));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaMemcpy(d_idata, h_idata, memSize,
gpu/bandwidthTest.cu:                               cudaMemcpyHostToDevice));
gpu/bandwidthTest.cu:    //copy data from GPU to Host
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventRecord(start, 0));
gpu/bandwidthTest.cu:            checkCudaErrors(cudaMemcpyAsync(h_odata, d_idata, memSize,
gpu/bandwidthTest.cu:                                            cudaMemcpyDeviceToHost, 0));
gpu/bandwidthTest.cu:            checkCudaErrors(cudaMemcpy(h_odata, d_idata, memSize,
gpu/bandwidthTest.cu:                                       cudaMemcpyDeviceToHost));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventRecord(stop, 0));
gpu/bandwidthTest.cu:    // make sure GPU has finished copying
gpu/bandwidthTest.cu:    checkCudaErrors(cudaDeviceSynchronize());
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
gpu/bandwidthTest.cu:    if (PINNED != memMode || bDontUseGPUTiming)
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventDestroy(stop));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventDestroy(start));
gpu/bandwidthTest.cu:        checkCudaErrors(cudaFreeHost(h_idata));
gpu/bandwidthTest.cu:        checkCudaErrors(cudaFreeHost(h_odata));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaFree(d_idata));
gpu/bandwidthTest.cu:    cudaEvent_t start, stop;
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventCreate(&start));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventCreate(&stop));
gpu/bandwidthTest.cu:#if CUDART_VERSION >= 2020
gpu/bandwidthTest.cu:        checkCudaErrors(cudaHostAlloc((void **)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
gpu/bandwidthTest.cu:        checkCudaErrors(cudaMallocHost((void **)&h_odata, memSize));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaMalloc((void **) &d_idata, memSize));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventRecord(start, 0));
gpu/bandwidthTest.cu:            checkCudaErrors(cudaMemcpyAsync(d_idata, h_odata, memSize,
gpu/bandwidthTest.cu:                                            cudaMemcpyHostToDevice, 0));
gpu/bandwidthTest.cu:            checkCudaErrors(cudaMemcpy(d_idata, h_odata, memSize,
gpu/bandwidthTest.cu:                                       cudaMemcpyHostToDevice));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventRecord(stop, 0));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaDeviceSynchronize());
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
gpu/bandwidthTest.cu:    if (PINNED != memMode || bDontUseGPUTiming)
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventDestroy(stop));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventDestroy(start));
gpu/bandwidthTest.cu:        checkCudaErrors(cudaFreeHost(h_odata));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaFree(d_idata));
gpu/bandwidthTest.cu:    cudaEvent_t start, stop;
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventCreate(&start));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventCreate(&stop));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaMalloc((void **) &d_idata, memSize));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaMalloc((void **) &d_odata, memSize));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaMemcpy(d_idata, h_idata, memSize,
gpu/bandwidthTest.cu:                               cudaMemcpyHostToDevice));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventRecord(start, 0));
gpu/bandwidthTest.cu:        checkCudaErrors(cudaMemcpy(d_odata, d_idata, memSize,
gpu/bandwidthTest.cu:                                   cudaMemcpyDeviceToDevice));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventRecord(stop, 0));
gpu/bandwidthTest.cu:    //cudaDeviceSynchronize() is required in order to get
gpu/bandwidthTest.cu:    checkCudaErrors(cudaDeviceSynchronize());
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
gpu/bandwidthTest.cu:    if (bDontUseGPUTiming)
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventDestroy(stop));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaEventDestroy(start));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaFree(d_idata));
gpu/bandwidthTest.cu:    checkCudaErrors(cudaFree(d_odata));
gpu/bandwidthTest.cu:#if CUDART_VERSION >= 2020
gpu/deviceQuery.cpp: * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
gpu/deviceQuery.cpp: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/deviceQuery.cpp:/* This sample queries the properties of the CUDA devices present in the system
gpu/deviceQuery.cpp: * via CUDA Runtime API. */
gpu/deviceQuery.cpp:#include <cuda_runtime.h>
gpu/deviceQuery.cpp:#include <helper_cuda.h>
gpu/deviceQuery.cpp:#if CUDART_VERSION < 5000
gpu/deviceQuery.cpp:// CUDA-C includes
gpu/deviceQuery.cpp:#include <cuda.h>
gpu/deviceQuery.cpp:// This function wraps the CUDA Driver API into a template function
gpu/deviceQuery.cpp:inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
gpu/deviceQuery.cpp:  if (CUDA_SUCCESS != error) {
gpu/deviceQuery.cpp:#endif /* CUDART_VERSION < 5000 */
gpu/deviceQuery.cpp:      " CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");
gpu/deviceQuery.cpp:  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
gpu/deviceQuery.cpp:  if (error_id != cudaSuccess) {
gpu/deviceQuery.cpp:    printf("cudaGetDeviceCount returned %d\n-> %s\n",
gpu/deviceQuery.cpp:           static_cast<int>(error_id), cudaGetErrorString(error_id));
gpu/deviceQuery.cpp:  // This function call returns 0 if there are no CUDA capable devices.
gpu/deviceQuery.cpp:    printf("There are no available device(s) that support CUDA\n");
gpu/deviceQuery.cpp:    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
gpu/deviceQuery.cpp:    cudaSetDevice(dev);
gpu/deviceQuery.cpp:    cudaDeviceProp deviceProp;
gpu/deviceQuery.cpp:    cudaGetDeviceProperties(&deviceProp, dev);
gpu/deviceQuery.cpp:    cudaDriverGetVersion(&driverVersion);
gpu/deviceQuery.cpp:    cudaRuntimeGetVersion(&runtimeVersion);
gpu/deviceQuery.cpp:    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
gpu/deviceQuery.cpp:    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
gpu/deviceQuery.cpp:    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
gpu/deviceQuery.cpp:        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
gpu/deviceQuery.cpp:#if CUDART_VERSION >= 5000
gpu/deviceQuery.cpp:    // This is supported in CUDA 5.0 (runtime API device properties)
gpu/deviceQuery.cpp:    // This only available in CUDA 4.0-4.2 (but these were only exposed in the
gpu/deviceQuery.cpp:    // CUDA Driver API)
gpu/deviceQuery.cpp:    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
gpu/deviceQuery.cpp:    getCudaAttribute<int>(&memBusWidth,
gpu/deviceQuery.cpp:    getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);
gpu/deviceQuery.cpp:    printf("  Integrated GPU sharing Host Memory:            %s\n",
gpu/deviceQuery.cpp:    printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
gpu/deviceQuery.cpp:#if CUDART_VERSION > 8000
gpu/deviceQuery.cpp:        "Default (multiple host threads can use ::cudaSetDevice() with device "
gpu/deviceQuery.cpp:        "::cudaSetDevice() with this device)",
gpu/deviceQuery.cpp:        "Prohibited (no host thread can use ::cudaSetDevice() with this "
gpu/deviceQuery.cpp:        "::cudaSetDevice() with this device)",
gpu/deviceQuery.cpp:  // If there are 2 or more GPUs, query to determine whether RDMA is supported
gpu/deviceQuery.cpp:    cudaDeviceProp prop[64];
gpu/deviceQuery.cpp:    int gpuid[64];  // we want to find the first two GPUs that can support P2P
gpu/deviceQuery.cpp:    int gpu_p2p_count = 0;
gpu/deviceQuery.cpp:      checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));
gpu/deviceQuery.cpp:        // This is an array of P2P capable GPUs
gpu/deviceQuery.cpp:        gpuid[gpu_p2p_count++] = i;
gpu/deviceQuery.cpp:    // Show all the combinations of support P2P GPUs
gpu/deviceQuery.cpp:    if (gpu_p2p_count >= 2) {
gpu/deviceQuery.cpp:      for (int i = 0; i < gpu_p2p_count; i++) {
gpu/deviceQuery.cpp:        for (int j = 0; j < gpu_p2p_count; j++) {
gpu/deviceQuery.cpp:          if (gpuid[i] == gpuid[j]) {
gpu/deviceQuery.cpp:          checkCudaErrors(
gpu/deviceQuery.cpp:              cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
gpu/deviceQuery.cpp:          printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
gpu/deviceQuery.cpp:                 prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
gpu/deviceQuery.cpp:  // exe and CUDA driver name
gpu/deviceQuery.cpp:  std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
gpu/deviceQuery.cpp:  sProfileString += ", CUDA Driver Version = ";
gpu/deviceQuery.cpp:  sProfileString += ", CUDA Runtime Version = ";
gpu/common/inc/helper_cuda.h: * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/helper_cuda.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/helper_cuda.h:// These are CUDA Helper functions for initialization and error checking
gpu/common/inc/helper_cuda.h:#ifndef COMMON_HELPER_CUDA_H_
gpu/common/inc/helper_cuda.h:#define COMMON_HELPER_CUDA_H_
gpu/common/inc/helper_cuda.h:// files, please refer the CUDA examples for examples of the needed CUDA
gpu/common/inc/helper_cuda.h:// headers, which may change depending on which CUDA functions are used.
gpu/common/inc/helper_cuda.h:// CUDA Runtime error messages
gpu/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(cudaError_t error) {
gpu/common/inc/helper_cuda.h:  return cudaGetErrorName(error);
gpu/common/inc/helper_cuda.h:#ifdef CUDA_DRIVER_API
gpu/common/inc/helper_cuda.h:// CUDA Driver API errors
gpu/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(CUresult error) {
gpu/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(cublasStatus_t error) {
gpu/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(cufftResult error) {
gpu/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(cusparseStatus_t error) {
gpu/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(cusolverStatus_t error) {
gpu/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(curandStatus_t error) {
gpu/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(NppStatus error) {
gpu/common/inc/helper_cuda.h:    // These are for CUDA 5.5 or higher
gpu/common/inc/helper_cuda.h:    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
gpu/common/inc/helper_cuda.h:      return "NPP_CUDA_KERNEL_EXECUTION_ERROR";
gpu/common/inc/helper_cuda.h:#define DEVICE_RESET cudaDeviceReset();
gpu/common/inc/helper_cuda.h:    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
gpu/common/inc/helper_cuda.h:            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
gpu/common/inc/helper_cuda.h:    // Make sure we call CUDA Device Reset before exiting
gpu/common/inc/helper_cuda.h:// This will output the proper CUDA error strings in the event
gpu/common/inc/helper_cuda.h:// that a CUDA host call returns an error
gpu/common/inc/helper_cuda.h:#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
gpu/common/inc/helper_cuda.h:// This will output the proper error string when calling cudaGetLastError
gpu/common/inc/helper_cuda.h:#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
gpu/common/inc/helper_cuda.h:inline void __getLastCudaError(const char *errorMessage, const char *file,
gpu/common/inc/helper_cuda.h:  cudaError_t err = cudaGetLastError();
gpu/common/inc/helper_cuda.h:  if (cudaSuccess != err) {
gpu/common/inc/helper_cuda.h:            "%s(%i) : getLastCudaError() CUDA error :"
gpu/common/inc/helper_cuda.h:            cudaGetErrorString(err));
gpu/common/inc/helper_cuda.h:// This will only print the proper error string when calling cudaGetLastError
gpu/common/inc/helper_cuda.h:#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)
gpu/common/inc/helper_cuda.h:inline void __printLastCudaError(const char *errorMessage, const char *file,
gpu/common/inc/helper_cuda.h:  cudaError_t err = cudaGetLastError();
gpu/common/inc/helper_cuda.h:  if (cudaSuccess != err) {
gpu/common/inc/helper_cuda.h:            "%s(%i) : getLastCudaError() CUDA error :"
gpu/common/inc/helper_cuda.h:            cudaGetErrorString(err));
gpu/common/inc/helper_cuda.h:// Beginning of GPU Architecture definitions
gpu/common/inc/helper_cuda.h:  // Defines for GPU Architecture types (using the SM version to determine
gpu/common/inc/helper_cuda.h:  sSMtoCores nGpuArchCoresPerSM[] = {
gpu/common/inc/helper_cuda.h:  while (nGpuArchCoresPerSM[index].SM != -1) {
gpu/common/inc/helper_cuda.h:    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
gpu/common/inc/helper_cuda.h:      return nGpuArchCoresPerSM[index].Cores;
gpu/common/inc/helper_cuda.h:      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
gpu/common/inc/helper_cuda.h:  return nGpuArchCoresPerSM[index - 1].Cores;
gpu/common/inc/helper_cuda.h:  // end of GPU Architecture definitions
gpu/common/inc/helper_cuda.h:#ifdef __CUDA_RUNTIME_H__
gpu/common/inc/helper_cuda.h:// General GPU Device CUDA Initialization
gpu/common/inc/helper_cuda.h:inline int gpuDeviceInit(int devID) {
gpu/common/inc/helper_cuda.h:  checkCudaErrors(cudaGetDeviceCount(&device_count));
gpu/common/inc/helper_cuda.h:            "gpuDeviceInit() CUDA error: "
gpu/common/inc/helper_cuda.h:            "no devices supporting CUDA.\n");
gpu/common/inc/helper_cuda.h:    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
gpu/common/inc/helper_cuda.h:            ">> gpuDeviceInit (-device=%d) is not a valid"
gpu/common/inc/helper_cuda.h:            " GPU device. <<\n",
gpu/common/inc/helper_cuda.h:  cudaDeviceProp deviceProp;
gpu/common/inc/helper_cuda.h:  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
gpu/common/inc/helper_cuda.h:  if (deviceProp.computeMode == cudaComputeModeProhibited) {
gpu/common/inc/helper_cuda.h:            "Prohibited>, no threads can use cudaSetDevice().\n");
gpu/common/inc/helper_cuda.h:    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
gpu/common/inc/helper_cuda.h:  checkCudaErrors(cudaSetDevice(devID));
gpu/common/inc/helper_cuda.h:  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);
gpu/common/inc/helper_cuda.h:// This function returns the best GPU (with maximum GFLOPS)
gpu/common/inc/helper_cuda.h:inline int gpuGetMaxGflopsDeviceId() {
gpu/common/inc/helper_cuda.h:  cudaDeviceProp deviceProp;
gpu/common/inc/helper_cuda.h:  checkCudaErrors(cudaGetDeviceCount(&device_count));
gpu/common/inc/helper_cuda.h:            "gpuGetMaxGflopsDeviceId() CUDA error:"
gpu/common/inc/helper_cuda.h:            " no devices supporting CUDA.\n");
gpu/common/inc/helper_cuda.h:  // Find the best CUDA capable GPU device
gpu/common/inc/helper_cuda.h:    cudaGetDeviceProperties(&deviceProp, current_device);
gpu/common/inc/helper_cuda.h:    // If this GPU is not running on Compute Mode prohibited,
gpu/common/inc/helper_cuda.h:    if (deviceProp.computeMode != cudaComputeModeProhibited) {
gpu/common/inc/helper_cuda.h:            "gpuGetMaxGflopsDeviceId() CUDA error:"
gpu/common/inc/helper_cuda.h:// Initialization code to find the best CUDA Device
gpu/common/inc/helper_cuda.h:inline int findCudaDevice(int argc, const char **argv) {
gpu/common/inc/helper_cuda.h:  cudaDeviceProp deviceProp;
gpu/common/inc/helper_cuda.h:      devID = gpuDeviceInit(devID);
gpu/common/inc/helper_cuda.h:    devID = gpuGetMaxGflopsDeviceId();
gpu/common/inc/helper_cuda.h:    checkCudaErrors(cudaSetDevice(devID));
gpu/common/inc/helper_cuda.h:    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
gpu/common/inc/helper_cuda.h:    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
gpu/common/inc/helper_cuda.h:inline int findIntegratedGPU() {
gpu/common/inc/helper_cuda.h:  cudaDeviceProp deviceProp;
gpu/common/inc/helper_cuda.h:  checkCudaErrors(cudaGetDeviceCount(&device_count));
gpu/common/inc/helper_cuda.h:    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
gpu/common/inc/helper_cuda.h:  // Find the integrated GPU which is compute capable
gpu/common/inc/helper_cuda.h:    cudaGetDeviceProperties(&deviceProp, current_device);
gpu/common/inc/helper_cuda.h:    // If GPU is integrated and is not running on Compute Mode prohibited,
gpu/common/inc/helper_cuda.h:    // then cuda can map to GLES resource
gpu/common/inc/helper_cuda.h:        (deviceProp.computeMode != cudaComputeModeProhibited)) {
gpu/common/inc/helper_cuda.h:      checkCudaErrors(cudaSetDevice(current_device));
gpu/common/inc/helper_cuda.h:      checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
gpu/common/inc/helper_cuda.h:      printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
gpu/common/inc/helper_cuda.h:            "CUDA error:"
gpu/common/inc/helper_cuda.h:            " No GLES-CUDA Interop capable GPU found.\n");
gpu/common/inc/helper_cuda.h:// General check for CUDA GPU SM Capabilities
gpu/common/inc/helper_cuda.h:inline bool checkCudaCapabilities(int major_version, int minor_version) {
gpu/common/inc/helper_cuda.h:  cudaDeviceProp deviceProp;
gpu/common/inc/helper_cuda.h:  checkCudaErrors(cudaGetDevice(&dev));
gpu/common/inc/helper_cuda.h:  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
gpu/common/inc/helper_cuda.h:        "  No GPU device was found that can support "
gpu/common/inc/helper_cuda.h:        "CUDA compute capability %d.%d.\n",
gpu/common/inc/helper_cuda.h:  // end of CUDA Helper Functions
gpu/common/inc/helper_cuda.h:#endif  // COMMON_HELPER_CUDA_H_
gpu/common/inc/GL/glew.h:/* --------------------------- GL_ARB_gpu_shader5 -------------------------- */
gpu/common/inc/GL/glew.h:#ifndef GL_ARB_gpu_shader5
gpu/common/inc/GL/glew.h:#define GL_ARB_gpu_shader5 1
gpu/common/inc/GL/glew.h:#define GLEW_ARB_gpu_shader5 GLEW_GET_VAR(__GLEW_ARB_gpu_shader5)
gpu/common/inc/GL/glew.h:#endif /* GL_ARB_gpu_shader5 */
gpu/common/inc/GL/glew.h:/* ------------------------- GL_ARB_gpu_shader_fp64 ------------------------ */
gpu/common/inc/GL/glew.h:#ifndef GL_ARB_gpu_shader_fp64
gpu/common/inc/GL/glew.h:#define GL_ARB_gpu_shader_fp64 1
gpu/common/inc/GL/glew.h:#define GLEW_ARB_gpu_shader_fp64 GLEW_GET_VAR(__GLEW_ARB_gpu_shader_fp64)
gpu/common/inc/GL/glew.h:#endif /* GL_ARB_gpu_shader_fp64 */
gpu/common/inc/GL/glew.h:#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
gpu/common/inc/GL/glew.h:/* --------------------- GL_EXT_gpu_program_parameters --------------------- */
gpu/common/inc/GL/glew.h:#ifndef GL_EXT_gpu_program_parameters
gpu/common/inc/GL/glew.h:#define GL_EXT_gpu_program_parameters 1
gpu/common/inc/GL/glew.h:#define GLEW_EXT_gpu_program_parameters GLEW_GET_VAR(__GLEW_EXT_gpu_program_parameters)
gpu/common/inc/GL/glew.h:#endif /* GL_EXT_gpu_program_parameters */
gpu/common/inc/GL/glew.h:/* --------------------------- GL_EXT_gpu_shader4 -------------------------- */
gpu/common/inc/GL/glew.h:#ifndef GL_EXT_gpu_shader4
gpu/common/inc/GL/glew.h:#define GL_EXT_gpu_shader4 1
gpu/common/inc/GL/glew.h:#define GLEW_EXT_gpu_shader4 GLEW_GET_VAR(__GLEW_EXT_gpu_shader4)
gpu/common/inc/GL/glew.h:#endif /* GL_EXT_gpu_shader4 */
gpu/common/inc/GL/glew.h:/* --------------------------- GL_NV_gpu_program4 -------------------------- */
gpu/common/inc/GL/glew.h:#ifndef GL_NV_gpu_program4
gpu/common/inc/GL/glew.h:#define GL_NV_gpu_program4 1
gpu/common/inc/GL/glew.h:#define GLEW_NV_gpu_program4 GLEW_GET_VAR(__GLEW_NV_gpu_program4)
gpu/common/inc/GL/glew.h:#endif /* GL_NV_gpu_program4 */
gpu/common/inc/GL/glew.h:/* -------------------------- GL_NV_gpu_program4_1 ------------------------- */
gpu/common/inc/GL/glew.h:#ifndef GL_NV_gpu_program4_1
gpu/common/inc/GL/glew.h:#define GL_NV_gpu_program4_1 1
gpu/common/inc/GL/glew.h:#define GLEW_NV_gpu_program4_1 GLEW_GET_VAR(__GLEW_NV_gpu_program4_1)
gpu/common/inc/GL/glew.h:#endif /* GL_NV_gpu_program4_1 */
gpu/common/inc/GL/glew.h:/* --------------------------- GL_NV_gpu_program5 -------------------------- */
gpu/common/inc/GL/glew.h:#ifndef GL_NV_gpu_program5
gpu/common/inc/GL/glew.h:#define GL_NV_gpu_program5 1
gpu/common/inc/GL/glew.h:#define GLEW_NV_gpu_program5 GLEW_GET_VAR(__GLEW_NV_gpu_program5)
gpu/common/inc/GL/glew.h:#endif /* GL_NV_gpu_program5 */
gpu/common/inc/GL/glew.h:/* ------------------------- GL_NV_gpu_program_fp64 ------------------------ */
gpu/common/inc/GL/glew.h:#ifndef GL_NV_gpu_program_fp64
gpu/common/inc/GL/glew.h:#define GL_NV_gpu_program_fp64 1
gpu/common/inc/GL/glew.h:#define GLEW_NV_gpu_program_fp64 GLEW_GET_VAR(__GLEW_NV_gpu_program_fp64)
gpu/common/inc/GL/glew.h:#endif /* GL_NV_gpu_program_fp64 */
gpu/common/inc/GL/glew.h:/* --------------------------- GL_NV_gpu_shader5 --------------------------- */
gpu/common/inc/GL/glew.h:#ifndef GL_NV_gpu_shader5
gpu/common/inc/GL/glew.h:#define GL_NV_gpu_shader5 1
gpu/common/inc/GL/glew.h:#define GLEW_NV_gpu_shader5 GLEW_GET_VAR(__GLEW_NV_gpu_shader5)
gpu/common/inc/GL/glew.h:#endif /* GL_NV_gpu_shader5 */
gpu/common/inc/GL/glew.h:#define GL_BUFFER_GPU_ADDRESS_NV 0x8F1D
gpu/common/inc/GL/glew.h:#define GL_GPU_ADDRESS_NV 0x8F34
gpu/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_ARB_gpu_shader5;
gpu/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_ARB_gpu_shader_fp64;
gpu/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_EXT_gpu_program_parameters;
gpu/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_EXT_gpu_shader4;
gpu/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_program4;
gpu/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_program4_1;
gpu/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_program5;
gpu/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_program_fp64;
gpu/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_shader5;
gpu/common/inc/GL/glext.h: Copyright NVIDIA Corporation 2006
gpu/common/inc/GL/glext.h: *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
gpu/common/inc/GL/glext.h: NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
gpu/common/inc/GL/glext.h: THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
gpu/common/inc/GL/glext.h:#ifndef GL_EXT_gpu_shader4
gpu/common/inc/GL/glext.h:#ifndef GL_NV_gpu_program4
gpu/common/inc/GL/glext.h:#ifndef GL_EXT_gpu_shader4
gpu/common/inc/GL/glext.h:#define GL_EXT_gpu_shader4 1
gpu/common/inc/GL/glext.h:#ifndef GL_NV_gpu_program4
gpu/common/inc/GL/glext.h:#define GL_NV_gpu_program4 1
gpu/common/inc/nvQuaternion.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/nvQuaternion.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/nvQuaternion.h:// Copyright (c) NVIDIA Corporation. All rights reserved.
gpu/common/inc/nvQuaternion.h:    Copyright (c) 2000 NVIDIA Corporation
gpu/common/inc/rendercheck_gl.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/rendercheck_gl.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/helper_cusolver.h: * Copyright 2015 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/helper_cusolver.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/helper_cusolver.h:#include <cuda_runtime.h>
gpu/common/inc/nvVector.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/nvVector.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/nvVector.h:// Copyright (c) NVIDIA Corporation. All rights reserved.
gpu/common/inc/nvVector.h:    Copyright (c) 2000 NVIDIA Corporation
gpu/common/inc/nvMath.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/nvMath.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/nvMath.h:// Copyright (c) NVIDIA Corporation. All rights reserved.
gpu/common/inc/nvMath.h:    Copyright (c) 2000 NVIDIA Corporation
gpu/common/inc/helper_gl.h: * Copyright 2014 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/helper_gl.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/rendercheck_gles.h: * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/rendercheck_gles.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/helper_functions.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/helper_functions.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/helper_cuda_drvapi.h: * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/helper_cuda_drvapi.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/helper_cuda_drvapi.h:// Helper functions for CUDA Driver API error handling (make sure that CUDA_H is
gpu/common/inc/helper_cuda_drvapi.h:#ifndef COMMON_HELPER_CUDA_DRVAPI_H_
gpu/common/inc/helper_cuda_drvapi.h:#define COMMON_HELPER_CUDA_DRVAPI_H_
gpu/common/inc/helper_cuda_drvapi.h:#ifndef COMMON_HELPER_CUDA_H_
gpu/common/inc/helper_cuda_drvapi.h:// These are CUDA Helper functions
gpu/common/inc/helper_cuda_drvapi.h:// add a level of protection to the CUDA SDK samples, let's force samples to
gpu/common/inc/helper_cuda_drvapi.h:// explicitly include CUDA.H
gpu/common/inc/helper_cuda_drvapi.h:#ifdef __cuda_cuda_h__
gpu/common/inc/helper_cuda_drvapi.h:// This will output the proper CUDA error strings in the event that a CUDA host
gpu/common/inc/helper_cuda_drvapi.h:#ifndef checkCudaErrors
gpu/common/inc/helper_cuda_drvapi.h:#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
gpu/common/inc/helper_cuda_drvapi.h:inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
gpu/common/inc/helper_cuda_drvapi.h:  if (CUDA_SUCCESS != err) {
gpu/common/inc/helper_cuda_drvapi.h:            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
gpu/common/inc/helper_cuda_drvapi.h:            err, getCudaDrvErrorString(err), file, line);
gpu/common/inc/helper_cuda_drvapi.h:#ifdef getLastCudaDrvErrorMsg
gpu/common/inc/helper_cuda_drvapi.h:#undef getLastCudaDrvErrorMsg
gpu/common/inc/helper_cuda_drvapi.h:#define getLastCudaDrvErrorMsg(msg) \
gpu/common/inc/helper_cuda_drvapi.h:  __getLastCudaDrvErrorMsg(msg, __FILE__, __LINE__)
gpu/common/inc/helper_cuda_drvapi.h:inline void __getLastCudaDrvErrorMsg(const char *msg, const char *file,
gpu/common/inc/helper_cuda_drvapi.h:  if (CUDA_SUCCESS != err) {
gpu/common/inc/helper_cuda_drvapi.h:    fprintf(stderr, "getLastCudaDrvErrorMsg -> %s", msg);
gpu/common/inc/helper_cuda_drvapi.h:            "getLastCudaDrvErrorMsg -> cuCtxSynchronize API error = %04d "
gpu/common/inc/helper_cuda_drvapi.h:            err, getCudaDrvErrorString(err), file, line);
gpu/common/inc/helper_cuda_drvapi.h:// This function wraps the CUDA Driver API into a template function
gpu/common/inc/helper_cuda_drvapi.h:inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
gpu/common/inc/helper_cuda_drvapi.h:  if (error_result != CUDA_SUCCESS) {
gpu/common/inc/helper_cuda_drvapi.h:           static_cast<int>(error_result), getCudaDrvErrorString(error_result));
gpu/common/inc/helper_cuda_drvapi.h:// Beginning of GPU Architecture definitions
gpu/common/inc/helper_cuda_drvapi.h:  // Defines for GPU Architecture types (using the SM version to determine the #
gpu/common/inc/helper_cuda_drvapi.h:  sSMtoCores nGpuArchCoresPerSM[] = {
gpu/common/inc/helper_cuda_drvapi.h:  while (nGpuArchCoresPerSM[index].SM != -1) {
gpu/common/inc/helper_cuda_drvapi.h:    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
gpu/common/inc/helper_cuda_drvapi.h:      return nGpuArchCoresPerSM[index].Cores;
gpu/common/inc/helper_cuda_drvapi.h:      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
gpu/common/inc/helper_cuda_drvapi.h:  return nGpuArchCoresPerSM[index - 1].Cores;
gpu/common/inc/helper_cuda_drvapi.h:  // end of GPU Architecture definitions
gpu/common/inc/helper_cuda_drvapi.h:#ifdef __cuda_cuda_h__
gpu/common/inc/helper_cuda_drvapi.h:// General GPU Device CUDA Initialization
gpu/common/inc/helper_cuda_drvapi.h:inline int gpuDeviceInitDRV(int ARGC, const char **ARGV) {
gpu/common/inc/helper_cuda_drvapi.h:  if (CUDA_SUCCESS == err) {
gpu/common/inc/helper_cuda_drvapi.h:    checkCudaErrors(cuDeviceGetCount(&deviceCount));
gpu/common/inc/helper_cuda_drvapi.h:    fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
gpu/common/inc/helper_cuda_drvapi.h:    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
gpu/common/inc/helper_cuda_drvapi.h:            ">> cudaDeviceInit (-device=%d) is not a valid GPU device. <<\n",
gpu/common/inc/helper_cuda_drvapi.h:  checkCudaErrors(cuDeviceGet(&cuDevice, dev));
gpu/common/inc/helper_cuda_drvapi.h:  getCudaAttribute<int>(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);
gpu/common/inc/helper_cuda_drvapi.h:            "threads can use this CUDA Device.\n");
gpu/common/inc/helper_cuda_drvapi.h:    printf("gpuDeviceInitDRV() Using CUDA Device [%d]: %s\n", dev, name);
gpu/common/inc/helper_cuda_drvapi.h:// This function returns the best GPU based on performance
gpu/common/inc/helper_cuda_drvapi.h:inline int gpuGetMaxGflopsDeviceIdDRV() {
gpu/common/inc/helper_cuda_drvapi.h:  checkCudaErrors(cuDeviceGetCount(&device_count));
gpu/common/inc/helper_cuda_drvapi.h:            "gpuGetMaxGflopsDeviceIdDRV error: no devices supporting CUDA\n");
gpu/common/inc/helper_cuda_drvapi.h:  // Find the best CUDA capable GPU device
gpu/common/inc/helper_cuda_drvapi.h:    checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/helper_cuda_drvapi.h:    checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/helper_cuda_drvapi.h:    checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/helper_cuda_drvapi.h:    checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/helper_cuda_drvapi.h:    getCudaAttribute<int>(&computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
gpu/common/inc/helper_cuda_drvapi.h:            "gpuGetMaxGflopsDeviceIdDRV error: all devices have compute mode "
gpu/common/inc/helper_cuda_drvapi.h:// General initialization call to pick the best CUDA Device
gpu/common/inc/helper_cuda_drvapi.h:inline CUdevice findCudaDeviceDRV(int argc, const char **argv) {
gpu/common/inc/helper_cuda_drvapi.h:    devID = gpuDeviceInitDRV(argc, argv);
gpu/common/inc/helper_cuda_drvapi.h:    devID = gpuGetMaxGflopsDeviceIdDRV();
gpu/common/inc/helper_cuda_drvapi.h:    checkCudaErrors(cuDeviceGet(&cuDevice, devID));
gpu/common/inc/helper_cuda_drvapi.h:    printf("> Using CUDA Device [%d]: %s\n", devID, name);
gpu/common/inc/helper_cuda_drvapi.h:inline CUdevice findIntegratedGPUDrv() {
gpu/common/inc/helper_cuda_drvapi.h:  checkCudaErrors(cuDeviceGetCount(&device_count));
gpu/common/inc/helper_cuda_drvapi.h:    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
gpu/common/inc/helper_cuda_drvapi.h:  // Find the integrated GPU which is compute capable
gpu/common/inc/helper_cuda_drvapi.h:    checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/helper_cuda_drvapi.h:    checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/helper_cuda_drvapi.h:    // If GPU is integrated and is not running on Compute Mode prohibited use
gpu/common/inc/helper_cuda_drvapi.h:      checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/helper_cuda_drvapi.h:      checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/helper_cuda_drvapi.h:      checkCudaErrors(cuDeviceGetName(deviceName, 256, current_device));
gpu/common/inc/helper_cuda_drvapi.h:      printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
gpu/common/inc/helper_cuda_drvapi.h:    fprintf(stderr, "CUDA error: No Integrated CUDA capable GPU found.\n");
gpu/common/inc/helper_cuda_drvapi.h:// General check for CUDA GPU SM Capabilities
gpu/common/inc/helper_cuda_drvapi.h:inline bool checkCudaCapabilitiesDRV(int major_version, int minor_version,
gpu/common/inc/helper_cuda_drvapi.h:  checkCudaErrors(cuDeviceGet(&cuDevice, devID));
gpu/common/inc/helper_cuda_drvapi.h:  checkCudaErrors(cuDeviceGetName(name, 100, cuDevice));
gpu/common/inc/helper_cuda_drvapi.h:  checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/helper_cuda_drvapi.h:  checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/helper_cuda_drvapi.h:        "No GPU device was found that can support CUDA compute capability "
gpu/common/inc/helper_cuda_drvapi.h:  // end of CUDA Helper Functions
gpu/common/inc/helper_cuda_drvapi.h:#endif  // COMMON_HELPER_CUDA_DRVAPI_H_
gpu/common/inc/nvMatrix.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/nvMatrix.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/nvMatrix.h:// Copyright (c) NVIDIA Corporation. All rights reserved.
gpu/common/inc/nvMatrix.h:    Copyright (c) 2000 NVIDIA Corporation
gpu/common/inc/multithreading.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/multithreading.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/nvrtc_helper.h: * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/nvrtc_helper.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/nvrtc_helper.h:#include <cuda.h>
gpu/common/inc/nvrtc_helper.h:#include <helper_cuda_drvapi.h>
gpu/common/inc/nvrtc_helper.h:  // Picks the best CUDA device available
gpu/common/inc/nvrtc_helper.h:  CUdevice cuDevice = findCudaDeviceDRV(argc, (const char **)argv);
gpu/common/inc/nvrtc_helper.h:  checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/nvrtc_helper.h:  checkCudaErrors(cuDeviceGetAttribute(
gpu/common/inc/nvrtc_helper.h:  checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
gpu/common/inc/nvrtc_helper.h:  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
gpu/common/inc/nvrtc_helper.h:  checkCudaErrors(cuInit(0));
gpu/common/inc/nvrtc_helper.h:  checkCudaErrors(cuDeviceGet(&cuDevice, 0));
gpu/common/inc/nvrtc_helper.h:  checkCudaErrors(cuCtxCreate(&context, 0, cuDevice));
gpu/common/inc/nvrtc_helper.h:  checkCudaErrors(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
gpu/common/inc/paramgl.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/paramgl.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/drvapi_error_string.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/drvapi_error_string.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/drvapi_error_string.h:#ifdef __cuda_cuda_h__  // check to see if CUDA_H is included above
gpu/common/inc/drvapi_error_string.h:} s_CudaErrorStr;
gpu/common/inc/drvapi_error_string.h:static s_CudaErrorStr sCudaDrvErrorString[] = {
gpu/common/inc/drvapi_error_string.h:    {"CUDA_SUCCESS", 0},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_INVALID_VALUE", 1},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_OUT_OF_MEMORY", 2},
gpu/common/inc/drvapi_error_string.h:     * This indicates that the CUDA driver has not been initialized with
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NOT_INITIALIZED", 3},
gpu/common/inc/drvapi_error_string.h:     * This indicates that the CUDA driver is in the process of shutting down.
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_DEINITIALIZED", 4},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_PROFILER_DISABLED", 5},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_PROFILER_NOT_INITIALIZED", 6},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_PROFILER_ALREADY_STARTED", 7},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_PROFILER_ALREADY_STOPPED", 8},
gpu/common/inc/drvapi_error_string.h:     * This indicates that no CUDA-capable devices were detected by the
gpu/common/inc/drvapi_error_string.h:     * installed CUDA driver.
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NO_DEVICE (no CUDA-capable devices were detected)", 100},
gpu/common/inc/drvapi_error_string.h:     * correspond to a valid CUDA device.
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_INVALID_DEVICE (device specified is not a valid CUDA device)",
gpu/common/inc/drvapi_error_string.h:     * indicate an invalid CUDA module.
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_INVALID_IMAGE", 200},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_INVALID_CONTEXT", 201},
gpu/common/inc/drvapi_error_string.h:     * This error return is deprecated as of CUDA 3.2. It is no longer an
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_CONTEXT_ALREADY_CURRENT", 202},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_MAP_FAILED", 205},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_UNMAP_FAILED", 206},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_ARRAY_IS_MAPPED", 207},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_ALREADY_MAPPED", 208},
gpu/common/inc/drvapi_error_string.h:     * options for a particular CUDA source file that do not include the
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NO_BINARY_FOR_GPU", 209},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_ALREADY_ACQUIRED", 210},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NOT_MAPPED", 211},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NOT_MAPPED_AS_ARRAY", 212},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NOT_MAPPED_AS_POINTER", 213},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_ECC_UNCORRECTABLE", 214},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_UNSUPPORTED_LIMIT", 215},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_CONTEXT_ALREADY_IN_USE", 216},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_PEER_ACCESS_UNSUPPORTED", 217},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_INVALID_PTX", 218},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_INVALID_GRAPHICS_CONTEXT", 219},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NVLINK_UNCORRECTABLE", 220},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_JIT_COMPILER_NOT_FOUND", 221},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_INVALID_SOURCE", 300},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_FILE_NOT_FOUND", 301},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", 302},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_SHARED_OBJECT_INIT_FAILED", 303},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_OPERATING_SYSTEM", 304},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_INVALID_HANDLE", 400},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NOT_FOUND", 500},
gpu/common/inc/drvapi_error_string.h:     * indicated differently than ::CUDA_SUCCESS (which indicates completion).
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NOT_READY", 600},
gpu/common/inc/drvapi_error_string.h:     * This leaves the process in an inconsistent state and any further CUDA
gpu/common/inc/drvapi_error_string.h:     * work will return the same error. To continue using CUDA, the process must
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_ILLEGAL_ADDRESS", 700},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", 701},
gpu/common/inc/drvapi_error_string.h:     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
gpu/common/inc/drvapi_error_string.h:     * continue using CUDA.
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_LAUNCH_TIMEOUT", 702},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING", 703},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED", 704},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_PEER_ACCESS_NOT_ENABLED", 705},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE", 708},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_CONTEXT_IS_DESTROYED", 709},
gpu/common/inc/drvapi_error_string.h:     * reconstructed if the program is to continue using CUDA.
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_ASSERT", 710},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_TOO_MANY_PEERS", 711},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED", 712},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED", 713},
gpu/common/inc/drvapi_error_string.h:     * This leaves the process in an inconsistent state and any further CUDA
gpu/common/inc/drvapi_error_string.h:     * work will return the same error. To continue using CUDA, the process must
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_HARDWARE_STACK_ERROR", 714},
gpu/common/inc/drvapi_error_string.h:     * This leaves the process in an inconsistent state and any further CUDA
gpu/common/inc/drvapi_error_string.h:     * work will return the same error. To continue using CUDA, the process must
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_ILLEGAL_INSTRUCTION", 715},
gpu/common/inc/drvapi_error_string.h:     * process in an inconsistent state and any further CUDA work will return
gpu/common/inc/drvapi_error_string.h:     * the same error. To continue using CUDA, the process must be terminated
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_MISALIGNED_ADDRESS", 716},
gpu/common/inc/drvapi_error_string.h:     * This leaves the process in an inconsistent state and any further CUDA
gpu/common/inc/drvapi_error_string.h:     * work will return the same error. To continue using CUDA, the process must
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_INVALID_ADDRESS_SPACE", 717},
gpu/common/inc/drvapi_error_string.h:     * CUDA work will return the same error. To continue using CUDA, the process
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_INVALID_PC", 718},
gpu/common/inc/drvapi_error_string.h:     * reconstructed if the program is to continue using CUDA.
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_LAUNCH_FAILED", 719},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE", 720},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NOT_PERMITTED", 800},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_NOT_SUPPORTED", 801},
gpu/common/inc/drvapi_error_string.h:    {"CUDA_ERROR_UNKNOWN", 999},
gpu/common/inc/drvapi_error_string.h:inline const char *getCudaDrvErrorString(CUresult error_id) {
gpu/common/inc/drvapi_error_string.h:  while (sCudaDrvErrorString[index].error_id != error_id &&
gpu/common/inc/drvapi_error_string.h:         sCudaDrvErrorString[index].error_id != -1) {
gpu/common/inc/drvapi_error_string.h:  if (sCudaDrvErrorString[index].error_id == error_id)
gpu/common/inc/drvapi_error_string.h:    return (const char *)sCudaDrvErrorString[index].error_string;
gpu/common/inc/drvapi_error_string.h:    return (const char *)"CUDA_ERROR not found!";
gpu/common/inc/drvapi_error_string.h:#endif  // __cuda_cuda_h__
gpu/common/inc/rendercheck_d3d9.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/rendercheck_d3d9.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/helper_timer.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/helper_timer.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/exception.h: * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/exception.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/exception.h:/* CUda UTility Library */
gpu/common/inc/rendercheck_d3d10.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/rendercheck_d3d10.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/dynlink_d3d10.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/dynlink_d3d10.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/timer.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/timer.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/param.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/param.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/param.h: sgreen@nvidia.com 4/2001
gpu/common/inc/helper_string.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/helper_string.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/helper_string.h:// CUDA Utility Helper Functions
gpu/common/inc/helper_string.h:// This function wraps the CUDA Driver API into a template function
gpu/common/inc/helper_string.h:      "./7_CUDALibraries/",             // "/7_CUDALibraries/" subdir
gpu/common/inc/helper_string.h:      "./7_CUDALibraries/<executable_name>/",  // "/7_CUDALibraries/<executable_name>/"
gpu/common/inc/helper_string.h:      "./7_CUDALibraries/<executable_name>/data/",  // "/7_CUDALibraries/<executable_name>/data/"
gpu/common/inc/helper_string.h:      "../7_CUDALibraries/<executable_name>/data/",  // up 1 in tree,
gpu/common/inc/helper_string.h:                                                     // "/7_CUDALibraries/<executable_name>/"
gpu/common/inc/helper_string.h:      "../../7_CUDALibraries/<executable_name>/data/",  // up 2 in tree,
gpu/common/inc/helper_string.h:                                                        // "/7_CUDALibraries/<executable_name>/"
gpu/common/inc/helper_string.h:      "../../../7_CUDALibraries/<executable_name>/data/",  // up 3 in tree,
gpu/common/inc/helper_string.h:                                                           // "/7_CUDALibraries/<executable_name>/"
gpu/common/inc/helper_string.h:      "../../../7_CUDALibraries/<executable_name>/",  // up 3 in tree,
gpu/common/inc/helper_string.h:                                                      // "/7_CUDALibraries/<executable_name>/"
gpu/common/inc/helper_string.h:      "../../../../7_CUDALibraries/<executable_name>/data/",  // up 4 in tree,
gpu/common/inc/helper_string.h:                                                              // "/7_CUDALibraries/<executable_name>/"
gpu/common/inc/helper_string.h:      "../../../../7_CUDALibraries/<executable_name>/",  // up 4 in tree,
gpu/common/inc/helper_string.h:                                                         // "/7_CUDALibraries/<executable_name>/"
gpu/common/inc/helper_string.h:      "../../../../../7_CUDALibraries/<executable_name>/data/",  // up 5 in
gpu/common/inc/helper_string.h:                                                                 // "/7_CUDALibraries/<executable_name>/"
gpu/common/inc/helper_image.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/helper_image.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/nvShaderUtils.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/nvShaderUtils.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/nvShaderUtils.h: * Copyright (c) NVIDIA Corporation. All rights reserved.
gpu/common/inc/helper_math.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/helper_math.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/helper_math.h: *  (float3, float4 etc.) since these are not provided as standard by CUDA.
gpu/common/inc/helper_math.h:#include "cuda_runtime.h"
gpu/common/inc/helper_math.h:#ifndef __CUDACC__
gpu/common/inc/helper_math.h:// host implementations of CUDA functions
gpu/common/inc/dynlink_d3d11.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/dynlink_d3d11.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/common/inc/rendercheck_d3d11.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
gpu/common/inc/rendercheck_d3d11.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/CMakeLists.txt:project (ion-gpu)
gpu/CMakeLists.txt:set(PROJECT_DESCRIPTION "Ion Torrent GPU Libraries")
gpu/CMakeLists.txt:# Where to install - override with: -DION_GPU_PREFIX
gpu/CMakeLists.txt:set(CMAKE_INSTALL_PREFIX "${ION_GPU_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)
gpu/CMakeLists.txt:set(CPACK_PACKAGING_INSTALL_PREFIX ${ION_GPU_PREFIX})
gpu/CMakeLists.txt:  set(gpu_devdriver_version "460.80")
gpu/CMakeLists.txt:  set(gpu_devdriver_file "devdriver_NVIDIA-Linux-x86_64-460.80.run")
gpu/CMakeLists.txt:  set(gpu_devdriver_version "375.26")
gpu/CMakeLists.txt:  set(gpu_devdriver_file "devdriver_NVIDIA-Linux-x86_64-375.26.run")
gpu/CMakeLists.txt:set(gpu_devdriver "gpu_devdriver")
gpu/CMakeLists.txt:set(gpu_devdriver_version "${gpu_devdriver}-${gpu_devdriver_version}")
gpu/CMakeLists.txt:ExternalProject_add(${gpu_devdriver}
gpu/CMakeLists.txt:    PREFIX ${PROJECT_BINARY_DIR}/../${gpu_devdriver_version}-prefix
gpu/CMakeLists.txt:    SOURCE_DIR ${PROJECT_BINARY_DIR}/../${gpu_devdriver_version}
gpu/CMakeLists.txt:    URL "http://${ION_UPDATE_SERVER}/updates/software/external/${gpu_devdriver_file}.tar.gz" # cmake doesn't download .run files
gpu/CMakeLists.txt:install(PROGRAMS ${PROJECT_BINARY_DIR}/../${gpu_devdriver_version}/${gpu_devdriver_file}
gpu/CMakeLists.txt:install(PROGRAMS create_nvidia_files
gpu/CMakeLists.txt:message(STATUS "BUILD with CUDA ${CUDA_VERSION}")
gpu/CMakeLists.txt:set(COMMON_LIBS pthread dl rt ${CUDA_LIBRARIES})
gpu/CMakeLists.txt:CUDA_ADD_EXECUTABLE(bandwidthTest bandwidthTest.cu)
gpu/CMakeLists.txt:CUDA_ADD_EXECUTABLE(deviceQuery deviceQuery.cpp)
gpu/CMakeLists.txt:CUDA_ADD_EXECUTABLE(matrixMul matrixMul.cu)
gpu/CMakeLists.txt:add_dependencies(bandwidthTest cuda_toolkit)
gpu/CMakeLists.txt:add_dependencies(deviceQuery cuda_toolkit)
gpu/CMakeLists.txt:add_dependencies(matrixMul cuda_toolkit)
gpu/CMakeLists.txt:# Compiling GPU kernel code requires
gpu/CMakeLists.txt:set(CPACK_PACKAGE_DESCRIPTION "This package contains the NVIDIA device driver for the Torrent Server and instruments.")
gpu/create_nvidia_files:/sbin/modprobe nvidia
gpu/create_nvidia_files:    # Count the number of NVIDIA controllers found.
gpu/create_nvidia_files:    NVDEVS=`lspci | grep -i NVIDIA`
gpu/create_nvidia_files:        mknod -m 666 /dev/nvidia$i c 195 $i
gpu/create_nvidia_files:    mknod -m 666 /dev/nvidiactl c 195 255
gpu/matrixMul.cu: * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
gpu/matrixMul.cu: * Please refer to the NVIDIA end user license agreement (EULA) associated
gpu/matrixMul.cu: * It has been written for clarity of exposition to illustrate various CUDA programming
gpu/matrixMul.cu: * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
gpu/matrixMul.cu:// CUDA runtime
gpu/matrixMul.cu:#include <cuda_runtime.h>
gpu/matrixMul.cu:// Helper functions and utilities to work with CUDA
gpu/matrixMul.cu:#include <helper_cuda.h>
gpu/matrixMul.cu: * Matrix multiplication (CUDA Kernel) on the device: C = A * B
gpu/matrixMul.cu:template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
gpu/matrixMul.cu: * Run a simple test of matrix multiplication using CUDA
gpu/matrixMul.cu:    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
gpu/matrixMul.cu:    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
gpu/matrixMul.cu:    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
gpu/matrixMul.cu:    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
gpu/matrixMul.cu:    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
gpu/matrixMul.cu:    printf("Computing result using CUDA Kernel...\n");
gpu/matrixMul.cu:    // Performs warmup operation using matrixMul CUDA kernel
gpu/matrixMul.cu:        MatrixMulCUDA<16> <<< grid, threads >>>(d_C, d_A, d_B,
gpu/matrixMul.cu:        MatrixMulCUDA<32> <<< grid, threads >>>(d_C, d_A, d_B,
gpu/matrixMul.cu:    cudaDeviceSynchronize();
gpu/matrixMul.cu:    // Allocate CUDA events that we'll use for timing
gpu/matrixMul.cu:    cudaEvent_t start;
gpu/matrixMul.cu:    checkCudaErrors(cudaEventCreate(&start));
gpu/matrixMul.cu:    cudaEvent_t stop;
gpu/matrixMul.cu:    checkCudaErrors(cudaEventCreate(&stop));
gpu/matrixMul.cu:    checkCudaErrors(cudaEventRecord(start, NULL));
gpu/matrixMul.cu:            MatrixMulCUDA<16> <<< grid, threads >>>(d_C, d_A, d_B,
gpu/matrixMul.cu:            MatrixMulCUDA<32> <<< grid, threads >>>(d_C, d_A, d_B,
gpu/matrixMul.cu:    checkCudaErrors(cudaEventRecord(stop, NULL));
gpu/matrixMul.cu:    checkCudaErrors(cudaEventSynchronize(stop));
gpu/matrixMul.cu:    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
gpu/matrixMul.cu:    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
gpu/matrixMul.cu:    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
gpu/matrixMul.cu:    checkCudaErrors(cudaFree(d_A));
gpu/matrixMul.cu:    checkCudaErrors(cudaFree(d_B));
gpu/matrixMul.cu:    checkCudaErrors(cudaFree(d_C));
gpu/matrixMul.cu:    printf("\nNOTE: The CUDA Samples are not meant for performance"\
gpu/matrixMul.cu:           "measurements. Results may vary when GPU Boost is enabled.\n");
gpu/matrixMul.cu:    printf("[Matrix Multiply Using CUDA] - Starting...\n");
gpu/matrixMul.cu:    // This will pick the best possible CUDA capable device, otherwise
gpu/matrixMul.cu:    int dev = findCudaDevice(argc, (const char **)argv);
gpu/debian/preinst:	# test to see if running GPU on PROTON, S5 or DX
gpu/debian/preinst:			# stop ganglia-monitor server (which might monitor the GPU)
gpu/debian/preinst:    	rm -rf @ION_GPU_PREFIX@/devdriver
gpu/debian/preinst:        rm -rf /etc/modprobe.d/nvidia-installer-disable-nouveau.conf
gpu/debian/preinst:            sed -i "/create_nvidia_files/d" /etc/rc.local
gpu/debian/prerm:# To delete NVIDIA drivers, ???
gpu/debian/prerm:# To delete cuda toolkit, delete directory /usr/local/cuda
gpu/debian/prerm:        if [ -d @ION_GPU_PREFIX@/devdriver ]; then
gpu/debian/prerm:            cd @ION_GPU_PREFIX@/devdriver
gpu/debian/prerm:            ./nvidia-installer --uninstall --ui=none --no-questions
gpu/debian/prerm:    	rm -rf @ION_GPU_PREFIX@/devdriver
gpu/debian/prerm:        rm -rf /etc/modprobe.d/nvidia-installer-disable-nouveau.conf
gpu/debian/prerm:            sed -i "/create_nvidia_files/d" /etc/rc.local
gpu/debian/prerm:        if [ -d @ION_GPU_PREFIX@/devdriver ]; then
gpu/debian/prerm:            cd @ION_GPU_PREFIX@/devdriver
gpu/debian/prerm:            ./nvidia-installer --uninstall --ui=none --no-questions
gpu/debian/postrm:		rm -rf @ION_GPU_PREFIX@/devdriver
gpu/debian/postrm:        rm -rf /etc/modprobe.d/nvidia-installer-disable-nouveau.conf
gpu/debian/postrm:		    sed -i "/create_nvidia_files/d" /etc/rc.local
gpu/debian/postinst:#            echo "X server is running.  nVidia device drivers will"
gpu/debian/postinst:#            echo "not be installed.  If you want the GPU enabled,"
gpu/debian/postinst:        # Test for nvidia graphics device
gpu/debian/postinst:        if ! lspci|grep -i "nvidia"|egrep '(3D controller|VGA compatible controller)';then
gpu/debian/postinst:            echo "No compatible nVidia hardware detected."
gpu/debian/postinst:            echo "nVidia device driver will not be installed."
gpu/debian/postinst:        if lspci|grep -i "nvidia"|egrep '(3D controller|VGA compatible controller)'|grep 'rev ff';then
gpu/debian/postinst:            echo "nVidia hardware is detected but the device code indicates a hardware problem."
gpu/debian/postinst:            echo "Try reinstalling the nVidia card and then reinstalling this package."
gpu/debian/postinst:            echo `lspci|grep -i "nvidia"|egrep '(3D controller|VGA compatible controller)'`
gpu/debian/postinst:        rm -rf @ION_GPU_PREFIX@/devdriver
gpu/debian/postinst:        bash @ION_GPU_PREFIX@/devdriver_*.run \
gpu/debian/postinst:            --target @ION_GPU_PREFIX@/devdriver > /tmp/nvidiadriver_install.log
gpu/debian/postinst:        echo -e "# generated by nvidia-installer\nblacklist nouveau\noptions nouveau modeset=0" > /etc/modprobe.d/nvidia-installer-disable-nouveau.conf
gpu/debian/postinst:        echo "Install Nvidia Device Driver (see /tmp/nvidiadriver_install.log)"
gpu/debian/postinst:        cd @ION_GPU_PREFIX@/devdriver >/dev/null
gpu/debian/postinst:        ./nvidia-installer \
gpu/debian/postinst:            --run-nvidia-xconfig \
gpu/debian/postinst:	    --no-cc-version-check >& /tmp/nvidiadriver_install.log
gpu/debian/postinst:        nvidia_installer_log="/var/log/nvidia-installer.log"
gpu/debian/postinst:        elif [ -e $nvidia_installer_log ]; then
gpu/debian/postinst:          if grep -E "NVIDIA Linux graphics driver will ignore this GPU" $nvidia_installer_log >> /dev/null; then
gpu/debian/postinst:            echo "Following NVIDIA GPU's are no longer supported on this operating system."
gpu/debian/postinst:            echo `lspci|grep -i "nvidia"|egrep '(3D controller|VGA compatible controller)'`
gpu/debian/postinst:            echo "Nvidia driver installation failed. Look for more details in /var/log/nvidia-installer.log"
gpu/debian/postinst:        # create nvidia device files
gpu/debian/postinst:            @ION_GPU_PREFIX@/create_nvidia_files
gpu/debian/postinst:	        sed -i "/create_nvidia_files/d" /etc/rc.local
gpu/debian/postinst:                sed -i "s:^exit 0:@ION_GPU_PREFIX@/create_nvidia_files\nexit 0:" /etc/rc.local
gpu/debian/postinst:		# remove xorg.conf (installed by CUDA)
gpu/debian/postinst:		if [ "Enabled" == "`nvidia-smi -a | sed -e"/Ecc Mode/{x;N;p};d" | sed -e ':a;N;$!ba;s/\n/ /g' | sed -e 's/.*: //'`" ]; then
gpu/debian/postinst:		    nvidia-smi -e 0
gpu/debian/postinst:    	rm -rf @ION_GPU_PREFIX@/devdriver
gpu/debian/postinst:        rm -rf /etc/modprobe.d/nvidia-installer-disable-nouveau.conf
gpu/debian/postinst:            sed -i "/create_nvidia_files/d" /etc/rc.local

```
