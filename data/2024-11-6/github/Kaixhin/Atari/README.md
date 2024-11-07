# https://github.com/Kaixhin/Atari

```console
run.sh:  th main.lua -gpu 0 -zoom 4 -hiddenSize 32 -optimiser adam -steps 500000 -learnStart 50000 -tau 4 -memSize 50000 -epsilonSteps 10000 -valFreq 10000 -valSteps 6000 -bootstraps 0 -memPriority rank -PALpha 0 "$@"
test/testExperience.lua:  gpu = false,
Experience.lua:  self.gpu = opt.gpu
README.md:To run experiments based on hyperparameters specified in the individual papers, use `./run.sh <paper> <game> <args>`. `<args>` can be used to overwrite arguments specified earlier (in the script); for more details see the script itself. By default the code trains on a demo environment called Catch - use `./run.sh demo` to run the demo with good default parameters. Note that this code uses CUDA if available, but the Catch network is small enough that it runs faster on CPU. If cuDNN is available, it can be enabled using `-cudnn true`; note that by default cuDNN is nondeterministic, and its deterministic modes are slower than cutorch.
README.md:Requires [Torch7](http://torch.ch/), and can use CUDA and cuDNN if available. Also requires the following extra luarocks packages:
Agent.lua:      aIndex = torch.mode(QHeadsMaxInds:float(), 1)[1][1] -- TODO: Torch.CudaTensor:mode is missing
Agent.lua:  torch.save(path, self.theta:float()) -- Do not save as CudaTensor to increase compatibility
Model.lua:  self.gpu = opt.gpu
Model.lua:  local frame = observation:type(self.tensorType) -- Convert from CudaTensor if necessary
Model.lua:  -- GPU conversion
Model.lua:  if self.gpu > 0 then
Model.lua:    net:cuda()
Setup.lua:  -- Tensor creation function for removing need to cast to CUDA if GPU is enabled
Setup.lua:  -- GPU setup
Setup.lua:  if self.opt.gpu > 0 then
Setup.lua:    log.info('Setting up GPU')
Setup.lua:    cutorch.setDevice(self.opt.gpu)
Setup.lua:      return torch.CudaTensor(...)
Setup.lua:  -- Detect and use GPU 1 by default
Setup.lua:  local cuda = pcall(require, 'cutorch')
Setup.lua:  cmd:option('-gpu', cuda and 1 or 0, 'GPU device ID (0 to disable)')
Setup.lua:  if opt.async then opt.gpu = 0 end -- Asynchronous agents are CPU-only

```
