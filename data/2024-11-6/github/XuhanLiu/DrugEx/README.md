# https://github.com/XuhanLiu/DrugEx

```console
designer.py:    # torch.cuda.set_device(0)
designer.py:    os.environ["CUDA_VISIBLE_DEVICES"] = OPT['-g'] if '-g' in OPT else "0, 1, 2, 3"
train_smiles.py:    os.environ["CUDA_VISIBLE_DEVICES"] = OPT['-g'] if '-g' in OPT else "0,1,2,3"
train_smiles.py:    torch.cuda.set_device(0)
train_smiles.py:    os.environ["CUDA_VISIBLE_DEVICES"] = OPT.get('-g', "0,1,2,3")
README.md:        $ conda install pytorch torchvision cudatoolkit=x.x -c pytorch 
README.md:        Note: it depends on the GPU device and CUDA tookit 
README.md:              (x.x is the version of CUDA)
environ.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_graph.py:    torch.cuda.set_device(utils.devices[0])
train_graph.py:    os.environ["CUDA_VISIBLE_DEVICES"] = devs
utils/objective.py:    def __init__(self, config_path, exe='vina-gpu', n_thread=3):
utils/objective.py:        assert exe in ['vina', 'vina-gpu']
utils/objective.py:            ranks = similarity_sort(preds, fps, is_gpu=True)
utils/objective.py:            ranks = nsgaii_sort(preds, is_gpu=True)
utils/__init__.py:dev = torch.device('cuda')
utils/nsgaii.py:def gpu_non_dominated_sort(swarm: torch.Tensor):
utils/nsgaii.py:def nsgaii_sort(array, is_gpu=False):
utils/nsgaii.py:    if is_gpu:
utils/nsgaii.py:        fronts = gpu_non_dominated_sort(array)
utils/nsgaii.py:def similarity_sort(array, fps, is_gpu=False):
utils/nsgaii.py:    if is_gpu:
utils/nsgaii.py:        fronts = gpu_non_dominated_sort(array)

```
