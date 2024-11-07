# https://github.com/thorstone25/qups

```console
teardown.m:% setup CUDA cache; % one-time setup
kern/das_spec.m:% y = DAS_SPEC(..., 'device', device, ...) forces selection of a gpuDevice.
kern/das_spec.m:%   -1 - use a CUDAKernel on the current GPU
kern/das_spec.m:%   +n - reset gpuDevice n and use a CUDAKernel (caution: this will clear
kern/das_spec.m:%        all of your gpuArray variables!
kern/das_spec.m:% intepolation. On the gpu, method can be one of 
kern/das_spec.m:% [y, k, PRE_ARGS, POST_ARGS] = DAS_SPEC(...) when the CUDA ptx is used returns
kern/das_spec.m:% the parallel.gpu.CUDAKernel k as well as the arguments for calling the
kern/das_spec.m:% The data x{f} must be in dimensions T x N x M. If x{f} is a gpuArray, 
kern/das_spec.m:% parallel.gpu.CUDAKernel k. This is useful for processing many identical
kern/das_spec.m:% parallel.gpu.CUDAKernel k, an illegal address error may occur, requiring
kern/das_spec.m:isType = @(c,T) isa(c, T) || isa(c, 'gpuArray') && strcmp(classUnderlying(c), T);
kern/das_spec.m:if gpuDeviceCount && any(cellfun(@(v)isa(v, 'gpuArray'), {Pi, Pr, Pv, Nv, x}))
kern/das_spec.m:    device = -1; % GPU
kern/das_spec.m:gdev = (exist('gpuDeviceCount','file') && gpuDeviceCount() && device <= gpuDeviceCount() && logical(exist('bf.ptx', 'file'))); ... % PTX track available
kern/das_spec.m:odev = (exist('oclDeviceCount','file') && oclDeviceCount() && device <= oclDeviceCount() && logical(exist('bf.cl' , 'file'))); ... OpenCL kernel available
kern/das_spec.m:    % reselect gpu device and copy over inputs if necessary
kern/das_spec.m:    if device > 0 && gdev && getfield(gpuDevice(), 'Index') ~= device
kern/das_spec.m:        % g = gpuDevice(device); 
kern/das_spec.m:    % send constant data to GPU
kern/das_spec.m:    if gdev, ptypefun = @(x) gpuArray(ptypefun(x)); end
kern/das_spec.m:        if gdev, ptypefun = @(x) gpuArray(single(x)); end
kern/das_spec.m:        g = gpuDevice();
kern/das_spec.m:        k = parallel.gpu.CUDAKernel(...
kern/das_spec.m:            ceil(g.AvailableMemory / (2^8) / (prod(osize{:}) * k.ThreadBlockSize(1))),... max cause GPU memory reqs (empirical)
kern/das_spec.m:        % expand inputs to 4D - in  OpenCL, all 3D vectors interpreted are 4D underlying
kern/rayPaths.m:% [...] = RAYPATHS(..., 'gpu', tf) chooses whether to use a default. The
kern/rayPaths.m:    kwargs.gpu (1,1) logical = logical(gpuDeviceCount())
kern/rayPaths.m:            if kwargs.gpu && exist('wbilerp.ptx', 'file') % call the GPU version
kern/rayPaths.m:            % use no parpool on GPU, default parpool on CPU, 
kern/rayPaths.m:            if kwargs.gpu, pul = 0; x0 = gpuArray(x0); z0 = gpuArray(z0);
kern/rayPaths.m:[w{:}] = gather(w{:}); % enforce on CPU - sparse gpu doesn't support 'cat'
kern/convd.m:% CONVD - GPU-enabled Convolution in one dimension
kern/convd.m:% C = CONVD(..., 'gpu', true) selects whether to use a gpu. A ptx-file will 
kern/convd.m:% be used if compiled. The default is true if x or y is a gpuArray.
kern/convd.m:% C = CONVD(..., 'ocl', true) selects whether to use an OpenCL device if
kern/convd.m:% OpenCL support is available. If the currently selected device does not 
kern/convd.m:% is true if OpenCL support is available via Matlab-OpenCL.
kern/convd.m:    kwargs.gpu (1,1) logical = isa(x, 'gpuArray') || isa(y, 'gpuArray')
kern/convd.m:gpu_type    = isa(x, 'gpuArray') || isa(y, 'gpuArray');
kern/convd.m:if gpu_type, To = gpuArray(To); end
kern/convd.m:% whether/how to operate with CUDA/OpenCL
kern/convd.m:use_gdev = kwargs.gpu && exist('convd.ptx', 'file') && exist('convd.cu', 'file');
kern/convd.m:        kern = parallel.gpu.CUDAKernel( ...
kern/convd.m:        % datatype here: for NVIDIA gpus, size_t <-> uint64
kern/convd.m:    % move to GPU if requested
kern/convd.m:    if kwargs.gpu, [x, y] = deal(gpuArray(x), gpuArray(y)); end
kern/convd.m:    % Use the given parallel environment if not on a GPU
kern/convd.m:    if isempty(clu) || isa(x, 'gpuArray') || isa(y, 'gpuArray'), clu = 0; end
kern/wbilerpg.m:% WBILERPG - Weights for bilinear interpolation (GPU/OpenCL-enabled)
kern/wbilerpg.m:% Note: This function requires that either a CUDA-enabled gpu or a working
kern/wbilerpg.m:% implementation of OpenCL via Matlab-OpenCL be available.
kern/wbilerpg.m:if uclass == "gpuArray", uclass = classUnderlying(xa); end
kern/wbilerpg.m:use_gdev = logical(gpuDeviceCount()) && exist('wbilerp.ptx', 'file');
kern/wbilerpg.m:    kern = parallel.gpu.CUDAKernel('wbilerp.ptx', 'wbilerp.cu', "wbilerp" + suffix);
kern/wbilerpg.m:        "wbilerpg() requires either a supported CUDA enabled GPU " + ...
kern/wbilerpg.m:        "or a working OpenCL implementation via Matlab-OpenCL. Use wbilerp instead." ...
kern/wsinterpd.m:% WSINTERPD GPU-enabled weight-and-sum interpolation in one dimension
kern/wsinterpd.m:isftype = @(x,T) strcmp(class(x), T) || any(arrayfun(@(c)isa(x,c),["tall", "gpuArray"])) && strcmp(classUnderlying(x), T);
kern/wsinterpd.m:        && ( isa(x, 'gpuArray') || isa(t, 'gpuArray') || isa(x, 'halfT') && x.gtype) ...
kern/wsinterpd.m:% use ptx on gpu if available or use native MATLAB
kern/wsinterpd.m:        suffix = "h"; prc = 16; [x,t,w] = dealfun(@(x)gpuArray(halfT(x)), x, t, w); % custom type
kern/wsinterpd.m:        warning("Datatype " + class(x) + " not recognized as a GPU compatible type.");
kern/wsinterpd.m:    % enforce complex type for ocl or gpu data
kern/wsinterpd.m:    if use_odev || isa(x, 'gpuArray') || (isa(x,'halfT') && isa(x.val, 'gpuArray')), x = complex(x); end
kern/wsinterpd.m:    if use_odev || isa(w, 'gpuArray') || (isa(w,'halfT') && isa(w.val, 'gpuArray')), w = complex(w); end
kern/wsinterpd.m:            y = complex(gpuArray(halfT(zeros(osz))));
kern/wsinterpd.m:        d = gpuDevice();
kern/wsinterpd.m:        k = parallel.gpu.CUDAKernel('interpd.ptx', 'interpd.cu', 'wsinterpd' + suffix);
kern/wsinterpd.m:    if any(cellfun(@(x)isa(x,'gpuArray'), {x,t,w})), parenv = 0; % no parenv for gpuArrays
kern/xiaolinwu_k_scaled.m:%     (1:K)); % arrayfun computes this in parallel if on a gpuArray
kern/xiaolinwu_k_scaled.m:% parallel on GPU
kern/slsc.m:% z = SLSC(..., 'ocl', dev) selects the openCL device with index dev. A
kern/slsc.m:% Note: OpenCL and MATLAB differ in their compute precision, leading to
kern/slsc.m:    kwargs.ocl (1,1) logical = exist('oclDevice','file') && oclDeviceCount(); % OpenCL device index
kern/slsc.m:% don't use any pool for gpu data
kern/slsc.m:        && ~isa(x, 'gpuArray') && ~isa(x, 'tall')
kern/slsc.m:% OCL or MATLAB (TODO: CUDA kernel?)
kern/slsc.m:    % native MATLAB with native gpu support
kern/wsinterpd2.m:% WSINTERPD2 GPU-enabled weight-and-sum interpolation in one dimension with
kern/wsinterpd2.m:isftype = @(x,T) strcmp(class(x), T) || any(arrayfun(@(c)isa(x,c),["tall", "gpuArray"])) && strcmp(classUnderlying(x), T);
kern/wsinterpd2.m:        && ( isa(x, 'gpuArray') || isa(t1, 'gpuArray') || isa(t2, 'gpuArray') || isa(x, 'halfT') && x.gtype) ...
kern/wsinterpd2.m:% use ptx on gpu if available or use native MATLAB
kern/wsinterpd2.m:        suffix = "h"; prc = 16; [x,t1,t2,w] = dealfun(@(x)gpuArray(halfT(x)), x, t1, t2, w); % custom type
kern/wsinterpd2.m:        warning("Datatype " + class(x) + " not recognized as a GPU compatible type.");
kern/wsinterpd2.m:    % enforce complex type for gpu data
kern/wsinterpd2.m:    if use_odev || isa(x, 'gpuArray') || isa(x, 'halfT'), x = complex(x); end
kern/wsinterpd2.m:    if use_odev || isa(w, 'gpuArray') || isa(w, 'halfT'), w = complex(w); end
kern/wsinterpd2.m:            y = complex(gpuArray(halfT(zeros(osz))));
kern/wsinterpd2.m:    % use a cached kernel (reduces likelihood of overloading OpenCL)?
kern/wsinterpd2.m:    use_cached = isscalar(k) && isvalid(k) && existsOnGPU(k) && (prc == prc0) ...
kern/wsinterpd2.m:            && (( use_gdev && isa(k,'parallel.gpu.CUDAKernel')) ...
kern/wsinterpd2.m:        k = parallel.gpu.CUDAKernel('interpd.ptx', 'interpd.cu', 'wsinterpd2' + suffix);
kern/wsinterpd2.m:    if use_gdev, d = gpuDevice(); cargs = {flagnum, imag(omega)}; else, d = k.Device; cargs = {}; end 
kern/wsinterpd2.m:    if any(cellfun(@(x)isa(x,'gpuArray'), {x,t1,t2,w})), parenv = 0; % no parenv for gpuArrays
kern/interpd.m:% INTERPD GPU-enabled interpolation in one dimension
kern/interpd.m:k = parallel.gpu.CUDAKernel.empty; % default CUDAKernel output argument
kern/interpd.m:% move the input data to the proper dimensions for the GPU kernel
kern/interpd.m:isftype = @(x,T) strcmp(class(x), T) || any(arrayfun(@(c)isa(x,c),["tall", "gpuArray"])) && strcmp(classUnderlying(x), T);
kern/interpd.m:        && ( isa(x, 'gpuArray') || isa(t, 'gpuArray') || isa(x, 'halfT') && x.gtype) ...
kern/interpd.m:% use ptx on gpu if available or use native MATLAB
kern/interpd.m:        suffix = "h"; prc = 16; [x,t] = dealfun(@(x)gpuArray(halfT(x)), x, t); % custom type
kern/interpd.m:        warning("Datatype " + class(x) + " not recognized as a GPU compatible type.");
kern/interpd.m:            y = complex(gpuArray(halfT(repelem(extrapval,osz))));
kern/interpd.m:        k = parallel.gpu.CUDAKernel('interpd.ptx', 'interpd.cu', 'interpd' + suffix);
kern/interpd.m:        d = gpuDevice();
test/BFTest.m:            % reset gpu
test/BFTest.m:        function resetGPU(test)
test/BFTest.m:            if gpuDeviceCount()
test/BFTest.m:                reselectgpu();
test/BFTest.m:                    wait(parfevalOnAll(gcp(), @reselectgpu, 0));
test/BFTest.m:            function reselectgpu()
test/BFTest.m:                id = gpuDevice().Index; gpuDevice([]); gpuDevice(id); 
test/BFTest.m:            % move data to GPU if requested
test/BFTest.m:            if gdev, chd = gpuArray(chd); else, chd = gather(chd); end 
test/BFTest.m:% test GPU devices if we can
test/BFTest.m:if gpuDeviceCount, s.dev = -1; end
test/BFTest.m:function s = getgpu()
test/BFTest.m:if gpuDeviceCount, s.ptx = {'CUDA'}; end
test/ParTest.m:    % PARTEST - Test parallel execution functionality (GPU & parallel.Pool)
test/ParTest.m:            if gpuDeviceCount
test/ParTest.m:                dev = [dev, arrayfun(@gpuDevice, 1:gpuDeviceCount(), 'UniformOutput', false)];
test/ParTest.m:                case "parallel.gpu.CUDADevice", dtype = "gpu";  gpuDevice(dev.Index);
test/ParTest.m:            odevs = setdiff(["gpu", "ocl"], dtype);
test/ParTest.m:            tst.assumeFalse(isMATLABReleaseOlderThan("R2023a")) % compilation via mexcuda
test/ParTest.m:            defs = [tst.us.getDASConstCudaDef(tst.cd), tst.us.getGreensConstCudaDef(tst.sct)];
test/ParTest.m:            tst.us.recompileCUDA(defs, "compute_60")
test/interpTest.m:        dev = {'CPU', 'GPU'} % gpu/cpu
test/interpTest.m:    	    if dev == "GPU"
test/interpTest.m:        		test.assumeTrue(gpuDeviceCount > 0);
test/interpTest.m:            % gpu/cpu
test/interpTest.m:                case "GPU", x0_ = gpuArray(x0_);
test/interpTest.m:                case "parallel.gpu.CUDADevice", dtype = "gpu";  gpuDevice(dev.Index);
test/interpTest.m:            odevs = setdiff(["gpu", "ocl"], dtype);
test/SimTest.m:        function resetGPU(test)
test/SimTest.m:            if gpuDeviceCount()
test/SimTest.m:                reselectgpu();
test/SimTest.m:                    wait(parfevalOnAll(gcp(), @reselectgpu, 0));
test/SimTest.m:            function reselectgpu()
test/SimTest.m:                id = gpuDevice().Index; gpuDevice([]); gpuDevice(id); 
test/SimTest.m:                case 'kWave',         if(gpuDeviceCount && (clu == 0 || isa(clu, 'parallel.Cluster'))), dtype = 'gpuArray-single'; else, dtype = 'single'; end % data type for k-Wave
test/SimTest.m:function s = getgpu()
test/SimTest.m:if gpuDeviceCount, s.ptx = {'CUDA'}; end
test/USTest.m:                        if canUseGPU(), d = [0 gpuDevice().Index]; else, d = 0; end
test/KernTest.m:                case "parallel.gpu.CUDADevice", dtype = "gpu";  gpuDevice(dev.Index);
test/KernTest.m:            odevs = setdiff(["gpu", "ocl"], dtype);
test/KernTest.m:            isgpu = isa(dev, 'parallel.gpu.CUDADevice');
test/KernTest.m:            if isgpu, [A,B] = dealfun(@gpuArray, A,B); end
test/KernTest.m:                z2 = gather(convd(shiftdim(A,1-d),shiftdim(B,1-d),d,s,'gpu', isgpu, 'ocl', isocl));
test/KernTest.m:            if isa(dev, 'parallel.gpu.CUDADevice'), [x, t, tau] = dealfun(@gpuArray, x, t, tau); end
test/KernTest.m:            if isa(dev, 'parallel.gpu.CUDADevice'), interpd(x, tau, 1, "lanczos3"); end
test/KernTest.m:            if isa(dev, 'parallel.gpu.CUDADevice'), b = gpuArray(b); end
test/KernTest.m:            if isa(dev, 'parallel.gpu.CUDADevice'), x = gpuArray(x); end
test/KernTest.m:            % implicit move to gpu
test/KernTest.m:            if isa(dev, 'parallel.gpu.CUDADevice')
test/KernTest.m:                [xa, ya, xb, xa] = dealfun(@gpuArray, xa, ya, xb, xa); 
test/KernTest.m:                if canUseGPU()
test/KernTest.m:                    test.assertEqual(y, gather(sel(gpuArray(x),(ix))));
test/KernTest.m:                    test.assertEqual(y, gather(sel((x),gpuArray(ix))));
test/KernTest.m:                    test.assertEqual(y, gather(sel(gpuArray(x),gpuArray(ix))));
test/KernTest.m:% test GPU/OpenCL devices if we can
test/KernTest.m:if gpuDeviceCount
test/KernTest.m:    dev = [dev, arrayfun(@gpuDevice, 1:gpuDeviceCount(), 'UniformOutput', false)];
test/ExampleTest.m:                'mex',      "recompile" + ["","Mex"], ... mex binaries required ... compilation (optional, requires CUDA)
test/ExampleTest.m:                'gpu',      ["wbilerpg","feval"], ... CUDAKernel or oclKernel support required
test/ExampleTest.m:                'comp',     ("recompile" + ["","CUDA"]), ... compilations setup required
test/ExampleTest.m:            kerns = ("wbilerp")+".ptx"; % require GPU binaries (pre-compiled)
test/ExampleTest.m:                bl_fcn(bl_fcn(:,1) == "gpu",:) = []; 
test/ExampleTest.m:            % filter by gpu compiler availability (system not available on thread pools)
test/ExampleTest.m:            if gpuDeviceCount()
test/ExampleTest.m:            if gpuDeviceCount() || (exist('oclDeviceCount', 'file') && oclDeviceCount())
test/ExampleTest.m:                devs = sortrows(devs, ["Type", "MaxComputeUnits"], "ascend"); % prefer gpu, most CUs
test/ExampleTest.m:            if test.run_file, try gpuDevice([]); end, end %#ok<TRYNC> % clear the gpu if there
README.md:    - Hardware acceleration via CUDA (Nvidia) or OpenCL (AMD, Apple, Intel, Nvidia), or natively via the [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html)
README.md:2. (optional) Install and patch [extension](##Extensions) packages and compile mex and CUDA binaries (failures can be safely ignored)
README.md:| [Matlab-OpenCL](https://github.com/thorstone25/Matlab-OpenCL) | hardware acceleration | (see [README](https://github.com/thorstone25/Matlab-OpenCL/blob/main/README.md)) | [website](https://github.com/IANW-Projects/MatCL?tab=readme-ov-file#reference) (via MatCL) |
README.md:| [CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) | hardware acceleration | (see [CUDA Support](#cuda-support)) | |
README.md:setup parallel CUDA cache; % setup the environment with any available acceleration
README.md:chd = greens(us, scat); % create channel data using a shifted Green's function (CUDA/OpenCL-enabled)
README.md:% chd = kspaceFirstOrder(us, scat); % ... or with k-Wave (CUDA-enabled)
README.md:% chd = simus(us, scat); %            ... or with MUST   (CUDA-enabled)
README.md:* [MatCL](https://github.com/IANW-Projects/MatCL?tab=readme-ov-file#reference) (via Matlab-OpenCL)
README.md:### [Matlab-OpenCL](github.com/thorstone25/Matlab-OpenCL)
README.md:OpenCL support is provided via [Matlab-OpenCL](github.com/thorstone25/Matlab-OpenCL), but is only tested on Linux. This package relies on [MatCL](https://github.com/IANW-Projects/MatCL), but the underlying OpenCL installation is platform and OS specific. The following packages and references may be helpful, but are not tested for compatability.
README.md:| `sudo apt install opencl-headers`                    | Compilation header files (req'd for all devices)|
README.md:| `sudo apt install pocl-opencl-icd`                   | Most CPU devices |
README.md:| `sudo apt install intel-opencl-icd`                  | Intel Graphics devices |
README.md:| `sudo apt install nvidia-driver-xxx`                 | Nvidia Graphics devices (included with the driver) |
README.md:| `sudo apt install ./amdgpu-install_x.x.x-x_all.deb`  | AMD Discrete Graphics devices (see [here](https://docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) or [here](https://docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html))|
README.md:### CUDA Support
README.md:Starting in R2023a, CUDA support is provided by default within MATLAB via [`mexcuda`](https://www.mathworks.com/help/parallel-computing/mexcuda.html).
README.md:Otherwise, for CUDA to work, `nvcc` must succesfully run from the MATLAB environment. If a Nvidia GPU is available and `setup CUDA cache` completes with no warnings, you're all set! If you have difficulty getting nvcc to work in MATLAB, you may need to figure out which environment paths are required for _your_ CUDA installation. Running `setup CUDA` will attempt to do this for you, but may fail if you have a custom installation.
README.md:First, be sure you can run `nvcc` from a terminal or command-line interface per [CUDA installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Then set the `MW_NVCC_PATH` environmental variable within MATLAB by running `setenv('MW_NVCC_PATH', YOUR_NVCC_BIN_PATH);` prior to running `setup CUDA`. You can run `which nvcc` within a terminal to locate the installation directory. For example, if `which nvcc` returns `/opt/cuda/bin/nvcc`, then run `setenv('MW_NVCC_PATH', '/opt/cuda/bin');`.
README.md:First, setup your system for CUDA per [CUDA installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). On Windows you must set the path for both CUDA and the _correct_ MSVC compiler for C/C++. Start a PowerShell terminal within Visual Studio. Run `echo %CUDA_PATH%` to find the base CUDA_PATH and run `echo %VCToolsInstallDir%` to find the MSVC path. Then, in MATLAB, set these paths with `setenv('MW_NVCC_PATH', YOUR_CUDA_BIN_PATH); setenv('VCToolsInstallDir', YOUR_MSVC_PATH);`, where `YOUR_CUDA_BIN_PATH` is the path to the `bin` folder in the `CUDA_PATH` folder. Finally, run `setup CUDA`. From here the proper paths should be added.
README.md:### GPU management
README.md:Some QUPS functions use the currently selected `parallel.gpu.CUDADevice`, or select a `parallel.gpu.CUDADevice` by default. Use [`gpuDevice`](https://www.mathworks.com/help/parallel-computing/parallel.gpu.gpudevice.html) to manually select the gpu. Within a parallel pool, each worker can have a unique selection. By default, GPUs are spread evenly across all workers.
README.md:If [CUDA support](#cuda-support) is enabled, [ptx-files](https://www.mathworks.com/help/parallel-computing/parallel.gpu.gpudevice.html) will be compiled to target the currently selected device. If the currently selected device changes, you may need to recompile binares using `UltrasoundSystem.recompileCUDA` or `setup cache`, particularly if the computer contains GPUs from different [virtual architectures](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list) or have different compute capabilities.
README.md:Beamforming and simulation routines tend to require a trade-off between performance and memory usage. While QUPS attempts to balance this, you can still run into OOM errors if the GPU is almost full or if the problem is too large for either the GPU or CPU. There are several options to mitigate this such as:
README.md:* (GPU OOM) use `x = gather(x);` to move some variables from the GPU (device) to the CPU (host) - this works on `ChannelData` objects too.
README.md:* (GPU OOM) use `gpuDevice().reset()` to reset the currently selected GPU - NOTE: this will clear any variables that were not moved from the GPU to the CPU with `gather` first
README.md:* (GPU OOM) for `UltrasoundSystem.DAS` and `UltrasoundSystem.greens`, set the `device` argument to `0` to avoid using a GPU device.
cheat_sheet.m:% send data to the GPU (incompatible with tall arrays)
cheat_sheet.m:if gpuDeviceCount, chd = gpuArray(chd); end
cheat_sheet.m:% make data a tall type (incompatible with gpuArrays)
cheat_sheet.m:chd = greens(us, scat); % NOTE: setting us.fs to single precision is much faster on the GPU.
build/coverage.xml:            <method branch-rate="NaN" line-rate="1" name="gpuArray" signature="chd = gpuArray(chd)">
build/coverage.xml:            <method branch-rate="0.75" line-rate="0.66667" name="recompileCUDA" signature="com = recompileCUDA(us, defs, arch, kwargs)">
build/coverage.xml:            <method branch-rate="0.375" line-rate="0.91176" name="getDASConstCudaDef" signature="def = getDASConstCudaDef(us, chd, varargin, kwargs)">
build/coverage.xml:            <method branch-rate="0.5" line-rate="1" name="getGreensConstCudaDef" signature="def = getGreensConstCudaDef(us, scat)">
build/coverage.xml:            <method branch-rate="0.92857" line-rate="1" name="genCUDAdefs" signature="defs = genCUDAdefs(name)">
build/compileCUDABinaries.m:function [s, com] = compileCUDABinaries(ofl)
build/compileCUDABinaries.m:% compileCUDABinaries - Compile CUDA binaries for the oldest supported CC.
build/compileCUDABinaries.m:% s = compileCUDABinaries(outdir) compiles the binaries and places them in
build/compileCUDABinaries.m:% [s, com] = compileCUDABinaries(...) additionally returns the compilation
build/compileCUDABinaries.m:% See also UltrasoundSystem.genCUDAdefs UltrasoundSystem.getSrcFolder
build/compileCUDABinaries.m:defs = UltrasoundSystem.genCUDAdefs(); % definition structs
build/compileCUDABinaries.m:    "-arch=" + arch + " ", ... compile for active gpu
setup.m:% SETUP cache - recompiles CUDA and mex files and creates a bin folder to 
setup.m:% SETUP CUDA - adds the default CUDA installation paths to the system
setup.m:% attempt to find an installation of CUDA with nvcc and add it to the
setup.m:% use that version of CUDA first. On Windows, if the 'VCToolsInstallDir'
setup.m:% Note: In MATLAB R2023a+ ptx compilation in provided via mexcuda, which
setup.m:% SETUP disable-gpu - disables gpu support by shadowing `gpuDeviceCount` so
setup.m:% that it always returns 0. This prevents implicit gpu usage by some
setup.m:% SETUP enable-gpu undoes the effects of the above.
setup.m:% SETUP disable-ocl - disables gpu support by shadowing `oclDeviceCount` so
setup.m:% that it always returns 0, similar to the disable-gpu option. This 
setup.m:    opts (1,1) string {mustBeMember(opts, ["CUDA", "cache", "parallel", "disable-gpu", "disable-ocl", "enable-gpu", "enable-ocl", "no-path"])}
setup.m:        case 'CUDA' % add CUDA to the path
setup.m:                    p = "/usr/local/cuda/bin"; % linux default nvcc path
setup.m:                    l = arrayfun(@(d) {dir(fullfile(d + ":\Program Files\NVIDIA GPU Computing Toolkit\CUDA","**","nvcc*"))}, wdrvs); % search for nvcc
setup.m:                % join nvcc and CUDA paths (if they exist)
setup.m:                warning('CUDA compilation paths undefined if not unix or pc.');
setup.m:        case {"disable-gpu", "disable-ocl", "enable-gpu", "enable-ocl"}
setup.m:                    % shadow gpu/ocl support by setting the device count to 0
resources/project/yftfCaPSE4KYJt7apxNIMR-w4YU/_4WwaR_MIizJJfHJHCST-sgYS3Yp.xml:<Info location="compileCUDABinaries.m" type="File"/>
example_.m:% CUDA hardware acceleration
example_.m:gpu = ~ismac && canUseGPU(); % set to false to remain on the cpu
example_.m:if gpu, setup CUDA; end % add default CUDA installation paths on Linux / Windows devices
example_.m:% OpenCL hardware acceleration if Matlab-OpenCL is installed
example_.m:    oclDevice(1); % select the first OpenCL device
example_.m:    case 'Greens' , chd0 = greens(us, scat); % use a Greens function with a GPU if available!su-vpn.stanford.edu
example_.m:if gpu, chd = gpuArray(chd); end % move data onto a GPU if one is available
examples/simulation/multilayer_media.m:    gpus = gpuDeviceCount(); % number of gpus available locally
examples/simulation/multilayer_media.m:    if gpus
examples/simulation/multilayer_media.m:        clu.NumWorkers = 2 * gpus; % 2 sims per gpu
examples/simulation/multilayer_media.m:        clu.NumThreads = 2; % 2 threads per gpu
examples/simulation/multilayer_media.m:if gpuDeviceCount(), chd = gpuArray(chd); end % move to gpu if available
utils/animate.m:    x {mustBeA(x, ["cell","gpuArray","double","single","logical","int64","int32","int16","int8","uint64","uint32","uint16","uint8"])} % data
utils/sel.m:if isa(x,'gpuArray') || isa(ind, 'gpuArray'), [i{:}] = dealfun(@gpuArray, i{:}); end % cast to GPU
utils/rsqrt.m:% 1 ./ sqrt(x), while handling MATLAB casting issues for gpuArrays or
utils/rsqrt.m:% gpuArray types are cast to complex if x contains negative values. 
utils/rsqrt.m:if any(x < 0,'all'),x = 1 ./ sqrt(complex(x)); % must be done manually for gpuArrays
buildfile.m:ext_nms = ["FieldII" "kWave" "OpenCL", "MUST", "USTB"];
buildfile.m:plan(  "install_OpenCL" ).Outputs = fullfile(base, "../Matlab-OpenCL");
buildfile.m:plan("uninstall_OpenCL" ).Inputs  = plan("install_OpenCL" ).Outputs;
buildfile.m:plan("patch_OpenCL").Inputs = plan("install_OpenCL" ).Outputs;
buildfile.m:% get CUDA files
buildfile.m:plan("compile_CUDA").Inputs  = fullfile(fls);
buildfile.m:plan("compile_CUDA").Outputs = fullfile(ofls);
buildfile.m:    "Dependencies", "compile_"+["CUDA", "mex"] ...
buildfile.m:function compile_CUDATask(context, arch)
buildfile.m:% compile CUDA kernels
buildfile.m:    warning("QUPS:build:compileCUDA:unsupportedArchitecture", ...
buildfile.m:defs = UltrasoundSystem.genCUDAdefs(); % matching definition structs
buildfile.m:try % via mexcuda
buildfile.m:    cellfun(@(args) mexcuda(args{:}), args);
buildfile.m:        join(["Unable to compile via mexcuda:",...
buildfile.m:    % Compile CUDA kernels
buildfile.m:    setup CUDA no-path; % add CUDA and US to path
buildfile.m:    us.recompileCUDA(defs, arch(end)); % compile
buildfile.m:function   install_OpenCLTask( context),   installer(context, "OpenCL" ); end % Download and install Matlab-OpenCL
buildfile.m:function uninstall_OpenCLTask( context), uninstaller(context, "OpenCL" ); end % Uninstall Matlab-OpenCL
buildfile.m:    ext (1,1) string {mustBeMember(ext, ["FieldII", "kWave", "MUST", "USTB", "OpenCL"])}
buildfile.m:        case "OpenCL"
buildfile.m:            url = "https://github.com/thorstone25/Matlab-OpenCL.git";
buildfile.m:function patch_OpenCLTask(context)
buildfile.m:% if recording pressure vs. time on gpu, keep storage on CPU when
buildfile.m:    "        if isa(sensor_data.p, 'gpuArray'), sensor_data.p = gather(sensor_data.p); end % always retain on CPU", ...
src/wbilerp.cu:#if (__CUDA_ARCH__ >= 530)
src/wbilerp.cu:#if (__CUDA_ARCH__ >= 530)
src/slsc.cl:#pragma OPENCL EXTENSION cl_khr_fp64 : enable
src/slsc.cl:#pragma OPENCL EXTENSION cl_khr_fp16 : enable
src/interpd.cl:// #prgma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
src/convd.cl:#pragma OPENCL EXTENSION cl_khr_fp16 : enable // must enable half precision
src/convd.cl:#pragma OPENCL EXTENSION cl_khr_fp64 : enable // must enable double precision
src/ChannelData.m:% gpuArray or tall type. The time axes is also cast to the corresponding
src/ChannelData.m:        function chd = gpuArray(chd), chd = applyFun2Data(chd, @gpuArray); end
src/ChannelData.m:        % cast underlying type to gpuArray
src/ChannelData.m:            %    **   GPU support is enabled via interpd
src/ChannelData.m:            %    ***  GPU support is enabled via interp1
src/ChannelData.m:            %    **** GPU support is native
src/ChannelData.m:            %    **   GPU support is enabled via interpd
src/ChannelData.m:            %    ***  GPU support is enabled via interp1
src/interpolators.cl: @file gpuBF/interp1d.cuh
src/interpolators.cl:// Modified for use as a stand-alone OpenCL file (Thurston Brevett <tbrevett@stanford.edu>)
src/interpolators.cl:#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable 
src/interpolators.cl:#pragma OPENCL EXTENSION cl_khr_fp64 : enable // must enable double precision
src/interpolators.cl:#pragma OPENCL EXTENSION cl_khr_fp16 : enable // must enable half precision
src/UltrasoundSystem.m:% * greens - simulate point scatterers via QUPS (GPU-enabled)
src/UltrasoundSystem.m:% * kspaceFirstOrder - simulate a medium via K-wave (GPU-enabled)
src/UltrasoundSystem.m:% Multiple beamformers are provided, all of which are GPU enabled, either
src/UltrasoundSystem.m:            % recompile mex and CUDA files for the UltrasoundSystem object.
src/UltrasoundSystem.m:            if gpuDeviceCount % only do CUDA stuff if there's a MATLAB-compatible GPU
src/UltrasoundSystem.m:                defs = us.genCUDAdefs();
src/UltrasoundSystem.m:                if opts.recompile && (~opts.copybin || any(~s)), us.recompileCUDA(); end % attempt to recompile code
src/UltrasoundSystem.m:            % [...] = GREENS(..., 'device', index) uses the gpuDevice with ID
src/UltrasoundSystem.m:            % avoids using a gpuDevice. The default is -1 if a gpu is
src/UltrasoundSystem.m:            % time on consumer GPUs which may have a performance ratio of
src/UltrasoundSystem.m:                kwargs.device (1,1) {mustBeInteger} = -1 * (logical(gpuDeviceCount()) || (exist('oclDeviceCount','file') && logical(oclDeviceCount())))
src/UltrasoundSystem.m:            use_gdev = use_dev && exist('greens.ptx', 'file') && gpuDeviceCount(); % use the GPU kernel
src/UltrasoundSystem.m:            use_odev = use_dev && exist('oclDeviceCount','file') && oclDeviceCount(); % use the OpenCL kernel
src/UltrasoundSystem.m:            if use_gdev, ps = gpuArray(ps); end
src/UltrasoundSystem.m:                isftype = @(x,T) strcmp(class(x), T) || any(arrayfun(@(c)isa(x,c),["tall", "gpuArray"])) && strcmp(classUnderlying(x), T);
src/UltrasoundSystem.m:                else,   error("Datatype " + class(kern) + " not recognized as a GPU compatible type.");
src/UltrasoundSystem.m:                    k = parallel.gpu.CUDAKernel('greens.ptx', 'greens.cu', 'greens' + suffix);
src/UltrasoundSystem.m:                % TODO: set data types | cast to GPU/tall? | reset GPU?
src/UltrasoundSystem.m:                if use_gdev, gpuDevice(kwargs.device); end
src/UltrasoundSystem.m:                    [ps, pn, pv, tvec, kern_] = dealfun(@gpuArray, ...
src/UltrasoundSystem.m:                % move back to GPU if requested
src/UltrasoundSystem.m:                if use_gdev, x = gpuArray(gather(x)); end
src/UltrasoundSystem.m:            x = reshape(x, size(x,[2:ndims(x) 1])); % same as shiftdim(x,1), but without memory copies on GPU
src/UltrasoundSystem.m:            % or parallel.Pool. Use 0 when operating on a GPU or if memory 
src/UltrasoundSystem.m:            % GPU or if memory usage explodes on a parallel.ProcessPool.
src/UltrasoundSystem.m:            % GPU or if memory usage explodes on a parallel.ProcessPool.
src/UltrasoundSystem.m:            % cluster with multiple GPUs, clu.SubmitArguments should
src/UltrasoundSystem.m:            % contain ' --gpus=1' so that 1 GPU is requested per pulse.
src/UltrasoundSystem.m:            % multiple GPUs, clu.SubmitArguments should contain the number
src/UltrasoundSystem.m:            % of GPUs desired for execution e.g. ' --gpus=4' and
src/UltrasoundSystem.m:            % simulataneous pulses to use e.g. 8. If more CPUs than GPUs
src/UltrasoundSystem.m:            % are requested, GPUs will be shared between multiple CPUs.
src/UltrasoundSystem.m:            % Sharing GPUs between multiple CPUs can be faster, but
src/UltrasoundSystem.m:            % requires sufficient GPU RAM.
src/UltrasoundSystem.m:            % the CPU version and 'G' for the GPU version. The selection is
src/UltrasoundSystem.m:            % determined by whether the 'DataCast' option represent a GPU 
src/UltrasoundSystem.m:                kwargs.gpu (1,1) logical = logical(gpuDeviceCount()) % whether to employ gpu during pre-processing
src/UltrasoundSystem.m:                kwave_args.DataCast (1,1) string = [repmat('gpuArray-',[1,logical(gpuDeviceCount())]), 'single']
src/UltrasoundSystem.m:                    if kwargs.gpu, [pg, pn] = dealfun(@gpuArray, pg, pn); end
src/UltrasoundSystem.m:                if contains(kwave_args.DataCast, "gpuArray")
src/UltrasoundSystem.m:                    usegpu = isa(u, 'gpuArray');
src/UltrasoundSystem.m:                            rx_sig, 2, 'full', 'gpu', usegpu ...
src/UltrasoundSystem.m:            setgpu = isa(W, 'parallel.Cluster'); % on implicit pools, gpu must be set explicitly
src/UltrasoundSystem.m:                % configure gpu selection
src/UltrasoundSystem.m:                if gpuDeviceCount() % if gpus exist
src/UltrasoundSystem.m:                    % get reference to the desired gpuDevice
src/UltrasoundSystem.m:                    if setgpu % set the current gpuDevice on implicit pools
src/UltrasoundSystem.m:                        % TODO: allow input to specify gpu dispatch
src/UltrasoundSystem.m:                        g = gpuDevice(1+mod(tsk.ID-1, gpuDeviceCount())); % even split across tasks 
src/UltrasoundSystem.m:                        g = gpuDevice();
src/UltrasoundSystem.m:                    % specify the device input explicitly if using a GPU binary
src/UltrasoundSystem.m:            % OpenCL kernels.
src/UltrasoundSystem.m:            % GPU. The default is 'cubic'.
src/UltrasoundSystem.m:            % b = DAS(..., 'device', index) uses the gpuDevice with ID
src/UltrasoundSystem.m:            % avoids using a gpuDevice. The default is -1 if a gpu is
src/UltrasoundSystem.m:            % available or 0 if no gpu is available.
src/UltrasoundSystem.m:            % [b, k, PRE_ARGS, POST_ARGS] = DAS(...) when the CUDA ptx is
src/UltrasoundSystem.m:            % used returns the parallel.gpu.CUDAKernel k as well as the
src/UltrasoundSystem.m:            % is a gpuArray, it must have the same type as was used to
src/UltrasoundSystem.m:            % create the parallel.gpu.CUDAKernel k. This is useful for
src/UltrasoundSystem.m:            % the parallel.gpu.CUDAKernel k, an illegal address error may
src/UltrasoundSystem.m:                kwargs.device (1,1) {mustBeInteger} = -1 * (logical(gpuDeviceCount()) || (exist('oclDeviceCount','file') && logical(oclDeviceCount())))
src/UltrasoundSystem.m:                % request the CUDA kernel?
src/UltrasoundSystem.m:                % use a thread pool if activated and no gpu variables in use
src/UltrasoundSystem.m:                if isa(gcp('nocreate'), 'parallel.ThreadPool') && ~isa(chd.data, 'gpuArray')
src/UltrasoundSystem.m:            % GPU or if memory usage explodes on a parallel.ProcessPool.
src/UltrasoundSystem.m:        function [mcom, nvcom] = recompile(us), mcom = recompileMex(us); if gpuDeviceCount, nvcom = recompileCUDA(us); else, nvcom = string.empty; end, end
src/UltrasoundSystem.m:        % RECOMPILE - Recompile mex and CUDA files
src/UltrasoundSystem.m:        % RECOMPILE(us) recompiles all mex binaries and CUDA files and 
src/UltrasoundSystem.m:        % GPUs, it does not attempt to recompile CUDA files.
src/UltrasoundSystem.m:        % See also ULTRASOUNDSYSTEM.RECOMPILECUDA ULTRASOUNDSYSTEM.RECOMPILEMEX
src/UltrasoundSystem.m:        function com = recompileCUDA(us, defs, arch, kwargs)
src/UltrasoundSystem.m:            % RECOMPILECUDA - Recompile CUDA ptx files
src/UltrasoundSystem.m:            % RECOMPILECUDA(us) recompiles all CUDA files to ptx and stores
src/UltrasoundSystem.m:            % RECOMPILECUDA(us, defs) compiles for the compiler 
src/UltrasoundSystem.m:            % definitions returned by UltrasoundSystem.genCUDAdefs().
src/UltrasoundSystem.m:            % RECOMPILECUDA(us, defs, arch) controls the architecture
src/UltrasoundSystem.m:            % The default is the architecture of the current GPU device.
src/UltrasoundSystem.m:            % RECOMPILECUDA(..., 'mex', true) uses `mexcuda` rather than
src/UltrasoundSystem.m:            % and other compiler options are not available via mexcuda. The
src/UltrasoundSystem.m:            % nvcom = RECOMPILECUDA(...) returns the command sent to nvcc.
src/UltrasoundSystem.m:            % nvcom = us.recompileCUDA(); % recompile CUDA files
src/UltrasoundSystem.m:            % % recompile the 3rd CUDA file manually
src/UltrasoundSystem.m:            % % mexcuda(nvcom{3}{:}); % via mexcuda
src/UltrasoundSystem.m:            % ULTRASOUNDSYSTEM.GENCUDADEFS
src/UltrasoundSystem.m:                defs (1,:) struct = UltrasoundSystem.genCUDAdefs();
src/UltrasoundSystem.m:                    "-arch=" + arch + " ", ... compile for active gpu
src/UltrasoundSystem.m:                else % via mexcuda
src/UltrasoundSystem.m:                    ... "-arch=" + arch + " ", ... compile for active gpu
src/UltrasoundSystem.m:                    comp = @(s) mexcuda(s{1}{:});
src/UltrasoundSystem.m:        function def = getDASConstCudaDef(us, chd, varargin, kwargs)
src/UltrasoundSystem.m:            % GETDASCONSTCUDADEF - Constant size compilation definition for DAS
src/UltrasoundSystem.m:            % def = GETDASCONSTCUDADEF(us) creates a compilation
src/UltrasoundSystem.m:            % definition for the CUDA executables used by
src/UltrasoundSystem.m:            % us.recompileCUDA() to reset the binaries to handle variable sizes. 
src/UltrasoundSystem.m:            % def = GETDASCONSTCUDADEF(us, chd) additionally uses a fixed
src/UltrasoundSystem.m:            % def = GETDASCONSTCUDADEF(us, chd, a1, a2, ..., an)
src/UltrasoundSystem.m:            % def = GETDASCONSTCUDADEF(..., 'keep_tx', true) preserves the transmit
src/UltrasoundSystem.m:            % def = GETDASCONSTCUDADEF(..., 'keep_rx', true) preserves the receive
src/UltrasoundSystem.m:            % def = GETDASCONSTCUDADEF(..., 'interp', method) specifies the method for
src/UltrasoundSystem.m:            % def = GETDASCONSTCUDADEF(..., 'interp', 'none') avoids specifying the 
src/UltrasoundSystem.m:            % def = GETDASCONSTCUDADEF(..., 'no_apod', true) avoids
src/UltrasoundSystem.m:            % def = getDASConstCudaDef(us, chd, apod, 'keep_rx', true);
src/UltrasoundSystem.m:            % us.recompileCUDA(def);
src/UltrasoundSystem.m:            % See also ULTRASOUNDSYSTEM.DAS ULTRASOUNDSYSTEM.GENCUDADEFS 
src/UltrasoundSystem.m:            def = UltrasoundSystem.genCUDAdefs('beamform');
src/UltrasoundSystem.m:        function def = getGreensConstCudaDef(us, scat)
src/UltrasoundSystem.m:            % GETGREENSCONSTCUDADEF - Constant size compilation definition for greens
src/UltrasoundSystem.m:            % def = GETGREENSCONSTCUDADEF(us) creates a compilation
src/UltrasoundSystem.m:            % definition for the CUDA executables used by
src/UltrasoundSystem.m:            % UltrasoundSystem.recompileCUDA to reset the binaries to
src/UltrasoundSystem.m:            % def = GETGREENSCONSTCUDADEF(us, scat) additionally uses a
src/UltrasoundSystem.m:            % See also ULTRASOUNDSYSTEM.GREENS ULTRASOUNDSYSTEM.GENCUDADEFS 
src/UltrasoundSystem.m:            def = UltrasoundSystem.genCUDAdefs('greens');
src/UltrasoundSystem.m:        function defs = genCUDAdefs(name)
src/UltrasoundSystem.m:            % GENCUDADEFS - Generate CUDA compilation definitions.
src/UltrasoundSystem.m:            % defs = GENCUDADEFS() returns a struct array of definition
src/UltrasoundSystem.m:            % structs defs to compile all CUDA kernels used by QUPS.
src/UltrasoundSystem.m:            % defs = GENCUDADEFS(name) returns the kernels specified within
src/UltrasoundSystem.m:            % defs = GENCUDADEFS("all") returns all available kernels. The 
src/UltrasoundSystem.m:            % Use with UltrasoundSystem.recompileCUDA to compile the
src/UltrasoundSystem.m:            % defs = UltrasoundSystem.genCUDAdefs(["interpd", "convd"]);
src/UltrasoundSystem.m:            % us.recompileCUDA(defs);
src/UltrasoundSystem.m:            % See also ULTRASOUNDSYSTEM.RECOMPILECUDA ULTRASOUNDSYSTEM.GETDASCONSTCUDADEF 
src/UltrasoundSystem.m:            % ULTRASOUNDSYSTEM.GETGREENSCONSTCUDADEF 
src/UltrasoundSystem.m:                        'no-deprecated-gpu-targets'...
src/UltrasoundSystem.m:                        'no-deprecated-gpu-targets'...
src/UltrasoundSystem.m:                        'no-deprecated-gpu-targets'...
src/UltrasoundSystem.m:                        'no-deprecated-gpu-targets'...
src/UltrasoundSystem.m:                        ... 'conv_cuda.cu', ...
src/UltrasoundSystem.m:                        'no-deprecated-gpu-targets'...
src/UltrasoundSystem.m:% NVARCH - compute capability architecture string for the current gpu
src/UltrasoundSystem.m:    g = gpuDevice();
src/UltrasoundSystem.m:    arch = "compute_" + replace(g.ComputeCapability,'.',''); % use the current GPU's CC number
src/UltrasoundSystem.m:    warning("Unable to access GPU.");
src/UltrasoundSystem.m:    error("Nvidia architectures must start with 'compute_'.");
src/wbilerp.cl:#pragma OPENCL EXTENSION cl_khr_fp64 : enable // must enable double precision
src/wbilerp.cl:#pragma OPENCL EXTENSION cl_khr_fp16 : enable // must enable half precision
src/bf.cu:#if (__CUDA_ARCH__ >= 530)
src/convd.cu:#if (__CUDA_ARCH__ >= 530)
src/convd.cu:#define __CUDA_NO_HALF2_OPERATORS__ // block half2 vector math operators
src/convd.cu:#include <cuda_fp16.h> // define half/half2 types, without half2 operators
src/convd.cu:#if (__CUDA_ARCH__ >= 530)
src/convd.cu:#if (__CUDA_ARCH__ >= 530)
src/convd.cu:#if (__CUDA_ARCH__ >= 530)
src/convd.cu:#if (__CUDA_ARCH__ >= 530)
src/convd.cu:#if (__CUDA_ARCH__ >= 530)
src/convd.cu:#if (__CUDA_ARCH__ >= 530)
src/convd.cu:#if (__CUDA_ARCH__ >= 530)
src/greens.cu:#if (__CUDA_ARCH__ >= 530)
src/helper_math.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
src/helper_math.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
src/helper_math.h: *  (float3, float4 etc.) since these are not provided as standard by CUDA.
src/helper_math.h:#include "cuda_runtime.h"
src/helper_math.h:#if (__CUDA_ARCH__ >= 530)
src/helper_math.h:#define __CUDA_NO_HALF2_OPERATORS__ // block half2 vector math operators
src/helper_math.h:#include <cuda_fp16.h> // define half/half2 types, without half2 operators
src/helper_math.h:#ifndef __CUDACC__
src/helper_math.h:// host implementations of CUDA functions
src/helper_math.h:#if (__CUDA_ARCH__ >= 530)
src/helper_math.h:#if (__CUDA_ARCH__ >= 530)
src/helper_math.h:#if (__CUDA_ARCH__ >= 530)
src/helper_math.h:#if (__CUDA_ARCH__ >= 530)
src/interpd.cu: @file gpuBF/interp1d.cuh
src/interpd.cu:#if (__CUDA_ARCH__ >= 530)
src/interpd.cu:#if (__CUDA_ARCH__ >= 530)
src/interpd.cu:#if __CUDA_ARCH__ < 600
src/interpd.cu:#if (__CUDA_ARCH__ >= 530) 
src/interpd.cu:#if (__CUDA_ARCH__ >= 530)
src/interpd.cu:#if (__CUDA_ARCH__ >= 530)
src/interpd.cu:#if (__CUDA_ARCH__ >= 530)
src/interpd.cu:#if (__CUDA_ARCH__ >= 530)

```
