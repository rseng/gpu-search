# https://github.com/DLR-RM/BlenderProc

```console
blenderproc/python/utility/Initializer.py:        # cpu thread means GPU-only rendering)
blenderproc/python/loader/AMASSLoader.py:                    # use GPU to accelerate mesh calculations
blenderproc/python/loader/AMASSLoader.py:                    comp_device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
blenderproc/python/loader/AMASSLoader.py:        comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blenderproc/python/renderer/RendererUtility.py:def set_render_devices(use_only_cpu: bool = False, desired_gpu_device_type: Union[str, List[str]] = None,
blenderproc/python/renderer/RendererUtility.py:                       desired_gpu_ids: Union[int, List[int]] = None):
blenderproc/python/renderer/RendererUtility.py:    :param desired_gpu_device_type: One or multiple GPU device types to consider. If multiple are given,
blenderproc/python/renderer/RendererUtility.py:                                    the first available is used. Possible choices are ["OPTIX", "CUDA",
blenderproc/python/renderer/RendererUtility.py:                                    "METAL", "HIP"]. Default is ["OPTIX", "CUDA", "HIP"] on linux/windows and
blenderproc/python/renderer/RendererUtility.py:    :param desired_gpu_ids: One or multiple GPU ids to specifically use. If none is given, all suitable GPUs are used.
blenderproc/python/renderer/RendererUtility.py:    if desired_gpu_device_type is None:
blenderproc/python/renderer/RendererUtility.py:        # If no gpu types are specified, use the default types based on the OS
blenderproc/python/renderer/RendererUtility.py:                desired_gpu_device_type = ["METAL"]
blenderproc/python/renderer/RendererUtility.py:                desired_gpu_device_type = []
blenderproc/python/renderer/RendererUtility.py:            desired_gpu_device_type = ["OPTIX", "CUDA", "HIP"]
blenderproc/python/renderer/RendererUtility.py:    elif not isinstance(desired_gpu_device_type, list):
blenderproc/python/renderer/RendererUtility.py:        desired_gpu_device_type = [desired_gpu_device_type]
blenderproc/python/renderer/RendererUtility.py:    # Make sure desired_gpu_device_type is a list
blenderproc/python/renderer/RendererUtility.py:    if desired_gpu_ids is not None and not isinstance(desired_gpu_ids, list):
blenderproc/python/renderer/RendererUtility.py:        desired_gpu_ids = [desired_gpu_ids]
blenderproc/python/renderer/RendererUtility.py:    # Decide between gpu and cpu rendering
blenderproc/python/renderer/RendererUtility.py:    if not desired_gpu_device_type or use_only_cpu:
blenderproc/python/renderer/RendererUtility.py:        # Use GPU
blenderproc/python/renderer/RendererUtility.py:        bpy.context.scene.cycles.device = "GPU"
blenderproc/python/renderer/RendererUtility.py:        for device_type in desired_gpu_device_type:
blenderproc/python/renderer/RendererUtility.py:                    # Only use gpus with specified ids
blenderproc/python/renderer/RendererUtility.py:                    if desired_gpu_ids is None or i in desired_gpu_ids:
blenderproc/python/renderer/RendererUtility.py:                    raise RuntimeError(f"The specified gpu ids lead to no selected gpu at all. Valid gpu ids are "
blenderproc/python/camera/LensDistortionUtility.py:        # - use torch.nn.functional.grid_sample() instead to do it on the GPU (even in batches)
blenderproc/external/vhacd/decompose.py:    You can turn of the usage of OpenCL by setting the environment variable NO_OPENCL to "1".
blenderproc/external/vhacd/decompose.py:        if "NO_OPENCL" in os.environ and os.environ["NO_OPENCL"] == "1":
blenderproc/external/vhacd/decompose.py:                      os.path.join(vhacd_path, "v-hacd") + " -DNO_OPENCL=ON")
blenderproc/external/vhacd/decompose.py:                      os.path.join(vhacd_path, "v-hacd") + " -DNO_OPENCL=OFF")
paper.bib:note= {\url{https://github.com/NVIDIA/Dataset_Synthesizer}},
paper.bib:title = {{NDDS}: {NVIDIA} Deep Learning Dataset Synthesizer},
README_BlenderProc4BOP.md:Objects from the selected BOP dataset are arranged inside an empty room, with objects from other BOP datasets used as distractors. To achieve a rich spectrum of generated images, a random PBR material from the [CC0 Textures](https://cc0textures.com/) library is assigned to the walls of the room, and light with a random strength and color is emitted from the room ceiling and from a randomly positioned point light source. This simple setup keeps the computational load low (1-4 seconds per image; 50K images can be rendered on 5 GPU's overnight).
change_log.md:- it can now be specified which GPU to use for rendering, if multiple are available
change_log.md:  - add support for Apple Silicon and GPU on Mac OS 12.3 
change_log.md:- switch to blender 2.93, with that textures are now stored on the GPU between different frames increasing the speed drastically
change_log.md:- AMASS now also supports `torch=1.8.1+cu111`, to better support modern GPUs
change_log.md:- added MacOS support (but only for CPUs, GPU support on MacOS is not available)
examples/datasets/bop_challenge/README.md:Tip: If you have access to multiple GPUs, you can speedup the process by dividing the 2000 scenes into multiples of 40 scenes (40 scenes * 25 images make up one chunk of 1000 images). Therefore run the script in parallel with different output folders. At the end, rename and merge the scenes in a joint folder. For example, if you have 10 GPUs, set `--num_scenes=200` and run the script 10 times with different output folders.

```
