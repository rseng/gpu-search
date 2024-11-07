# https://github.com/Kaixhin/Autoencoders

```console
models/Seq2SeqAE.lua:  -- Create CUDA wrapper
models/Seq2SeqAE.lua:  function self.autoencoder:cuda()
models/Seq2SeqAE.lua:    self.parent.encoder:cuda()
models/Seq2SeqAE.lua:    self.parent.decoder:cuda()
main.lua:local cuda = pcall(require, 'cutorch') -- Use CUDA if available
main.lua:cmd:option('-cpu', false, 'CPU only (useful if GPU memory is too low)')
main.lua:  cuda = false
main.lua:if cuda then
main.lua:if cuda then
main.lua:  XTrain = XTrain:cuda()
main.lua:  XTest = XTest:cuda()
main.lua:if cuda then
main.lua:  autoencoder:cuda()
main.lua:  if cuda then
main.lua:    adversary:cuda()
main.lua:if cuda then
main.lua:  criterion:cuda()
main.lua:  softmax:cuda()

```
