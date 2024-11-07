# https://github.com/zhangzzk/forklens

```console
examples/cnn_tests/config.py:    'device': 'cuda:0',
examples/cnn_tests/config.py:    'gpu_number': 2,
src/networks.py:    def __init__(self, nFeatures, BatchSize, GPUs=1):
src/networks.py:        self.GPUs = GPUs
src/networks.py:        x = x.view(int(self.batch/self.GPUs),-1)
src/networks.py:        y = y.view(int(self.batch/self.GPUs),-1)
src/train.py:        self.nGPUs = config.train['gpu_number']
src/train.py:        model = ForkCNN(self.features, self.batch_size, self.nGPUs)
src/train.py:        if self.nGPUs > 1:
src/train.py:            model = nn.DataParallel(model, device_ids=range(self.nGPUs))
src/train.py:        # self.model = ForkCNN(self.features, self.batch_size, self.nGPUs)
src/train.py:        # if self.nGPUs > 1:
src/train.py:        #     self.model = nn.DataParallel(self.model, device_ids=range(self.nGPUs))
src/train.py:        self.nGPUs = config.train['gpu_number']
src/train.py:        if self.nGPUs > 1:
src/train.py:            model = nn.DataParallel(model, device_ids=range(self.nGPUs))

```
