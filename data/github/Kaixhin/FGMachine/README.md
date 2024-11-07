# https://github.com/Kaixhin/FGMachine

```console
CHANGELOG.md:* read GPUs on OS X ([7c3b87a](https://github.com/Kaixhin/FGMachine/commit/7c3b87a)), closes [#6](https://github.com/Kaixhin/FGMachine/issues/6)
README.md:To launch [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) containers, use the following:
README.md:Note that `--net=host` is passed to allow access to the NVIDIA Docker API. When launching a sibling container, you will need to run `` `curl -s localhost:3476/docker/cli` `` and manually add the arguments to the project implementation in the container, with `docker` as the command (do not use `nvidia-docker`).
README.md:#### GPU capacity support
README.md:In order to handle projects, which require GPUs to perform a task, you need to add two parameters for each project in `projects.json` file:
README.md:  "gpu_capacity": "<gpu capacity needed (as a fraction of one GPU capacity, e.g. 0.5)>",
README.md:  "gpu_command": "<option to pass to script to identify card number, including command line option style (e.g. -gpu)>",
README.md:Note that `gpu_capacity` represents (the inverse of) instances of the program the FGMachine host system can run on one GPU; for example a machine with 4 GPUs will be able to run 8 instances of the program with `capacity` 0.1 and `gpu_capacity` 0.5. However, if the `capacity` was 0.25 in the previous example, the machine would only be able to run 4 instances of the program.
README.md:`gpu_capacity` automatically assigns a GPU for experiments, which makes it easier to run batch experiments. Note that like `nvidia-smi`, GPU IDs passed via `gpu-command` are **0-indexed**. For manual control, it is recommended to use a GPU flag as part of the experiment hyperparameters in the project schema.
machine.js:    gpus: []
machine.js:  // GPU models
machine.js:    var grep = spawnSync("grep", ["-i", "nvidia"], {input: grep_vga.stdout});
machine.js:    var gpuStrings = grep.stdout.toString().split("\n");
machine.js:    for (var i = 0; i < gpuStrings.length - 1; i++) {
machine.js:      specs.gpus.push(gpuStrings[i].replace(/.*controller: /g, ""));
machine.js:        specs.gpus.push(profilerStrings[j].replace(/Chipset Model: /g, ""));
machine.js:	if ("gpus" in specs) {
machine.js:	  for (var i = 0; i < specs.gpus.length; i++) {
machine.js:	    GPUsCapacity.push(1);
machine.js:	  GPUCapacity = GPUsCapacity.reduce((a, b) => a + b, 0);
machine.js:var GPUsCapacity = [];
machine.js:var GPUCapacity = 0;
machine.js:// Checks if GPU resources are required
machine.js:var isGPURequired = function(projId) {
machine.js:  return "gpu_capacity" in projects[projId];
machine.js:    if (isGPURequired(projId)) {
machine.js:      // if we have any GPUs 
machine.js:      if (("gpus" in specs) && (specs.gpus.length > 0)) {
machine.js:        var gpu_available_capacity = GPUsCapacity.reduce((a, b) => a + b, 0);
machine.js:        var gpu_capacity = Math.floor(gpu_available_capacity / projects[projId].gpu_capacity);
machine.js:        capacity = Math.min(gpu_capacity, capacity);
machine.js:  var gpuRequired = isGPURequired(req.params.id);
machine.js:  var assignedGPUId = 0;
machine.js:  // Control GPU capacity
machine.js:  if (gpuRequired) {
machine.js:    var freeGPUId = 0;
machine.js:    for (; freeGPUId < specs.gpus.length; freeGPUId++) {
machine.js:      if (GPUsCapacity[freeGPUId] >= project.gpu_capacity) {
machine.js:    GPUsCapacity[freeGPUId] = Math.max(GPUsCapacity[freeGPUId] - project.gpu_capacity, 0); // Reduce GPU capacity
machine.js:    assignedGPUId = freeGPUId;
machine.js:    args.push(project.gpu_command);
machine.js:    args.push(freeGPUId);
machine.js:    if (gpuRequired) {
machine.js:      GPUsCapacity[assignedGPUId] = Math.min(GPUsCapacity[assignedGPUId] + project.gpu_capacity, 1); 
machine.js:  for (var i = 0; i < specs.gpus.length; i++) {
machine.js:    GPUsCapacity[i] = 1;
examples/Recurrent-Attention-Model/Dockerfile_cuda_v6_5:# Start with CUDA Torch base image
examples/Recurrent-Attention-Model/Dockerfile_cuda_v6_5:FROM kaixhin/cuda-torch:6.5
examples/Recurrent-Attention-Model/main.lua:cmd:option('-cuda', false, 'Use CUDA')
examples/Recurrent-Attention-Model/main.lua:-- CUDA conversion
examples/Recurrent-Attention-Model/main.lua:if opt.cuda then
examples/Recurrent-Attention-Model/main.lua:  dataset.train.X = dataset.train.X:cuda()
examples/Recurrent-Attention-Model/main.lua:  dataset.train.Y = dataset.train.Y:cuda()
examples/Recurrent-Attention-Model/main.lua:  dataset.test.X = dataset.test.X:cuda()
examples/Recurrent-Attention-Model/main.lua:  dataset.test.Y = dataset.test.Y:cuda()
examples/Recurrent-Attention-Model/main.lua:  agent:cuda()
examples/Recurrent-Attention-Model/main.lua:  criterion:cuda()
examples/Recurrent-Attention-Model/README.md:For more information on Docker usage, including CUDA capabilities, please see the [source repo](https://github.com/Kaixhin/dockerfiles).
examples/Recurrent-Attention-Model/README.md:For CUDA capabilities, use the `kaixhin/cuda-torch` base image can be used. Dockerfiles that build from this image are also included. For example, to use CUDA 7.5, run `sudo docker build -t ram -f Dockerfile_cuda_v7_5 .`. For more information, see the notes on [CUDA in Docker](https://github.com/Kaixhin/dockerfiles#cuda).
examples/Recurrent-Attention-Model/Dockerfile_cuda_v7_5:# Start with CUDA Torch base image
examples/Recurrent-Attention-Model/Dockerfile_cuda_v7_5:FROM kaixhin/cuda-torch:latest
examples/Recurrent-Attention-Model/Dockerfile_cuda_v7_0:# Start with CUDA Torch base image
examples/Recurrent-Attention-Model/Dockerfile_cuda_v7_0:FROM kaixhin/cuda-torch:7.0

```
