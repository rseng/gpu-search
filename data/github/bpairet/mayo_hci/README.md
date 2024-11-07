# https://github.com/bpairet/mayo_hci

```console
mayonnaise.py:        #Check if we have a gpu, otherwise, set self.device to cpu:
mayonnaise.py:        if torch.cuda.is_available():
mayonnaise.py:            gpu_memory_map = get_gpu_memory_map()
mayonnaise.py:            if gpu_memory_map[0] < gpu_memory_map[1]:
mayonnaise.py:                self.device = torch.device('cuda:0')
mayonnaise.py:                self.device = torch.device('cuda:1')
algo_utils.py:def get_gpu_memory_map():
algo_utils.py:    """Get the current gpu usage.
algo_utils.py:            'nvidia-smi', '--query-gpu=memory.used',
algo_utils.py:    gpu_memory = [int(x) for x in result.strip().split('\n')]
algo_utils.py:    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
algo_utils.py:    return gpu_memory_map

```
