# https://github.com/Orin-beep/PlasGO

```console
preprocessing.py:parser.add_argument('--device', help="device utilized for generating original per-protein embeddings with ProtT5 ('gpu' or 'cpu'), default: 'gpu'", type=str, default = 'gpu')
preprocessing.py:parser.add_argument('--batch_size', help="batch size for protein embedding with the ProtT5 model. If your GPU is out of memory, you can try to reduce this parameter to 1, default: 8", type=int, default = 8)
preprocessing.py:        [--device DEVICE] device utilized for generating original per-protein embeddings with ProtT5 ('gpu' or 'cpu'), default: 'gpu'
preprocessing.py:        [--batch_size BATCH_SIZE] batch size for protein embedding with the ProtT5 model. If your GPU is out of memory, you can try to reduce this parameter to 1, default: 8
preprocessing.py:device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessing.py:if(device==torch.device("cuda")):
preprocessing.py:    print("GPU detected. Running protein embedding with GPU ...")
preprocessing.py:        print("GPU not detected. Running protein embedding with CPU ...")
preprocessing.py:        print("GPU not detected. We highly recommend you to run this script with GPU because of the large size of the ProtT5 model. If you still want to run with CPU, please specify the option '--device cpu'")
prot_t5_embed.py:parser.add_argument('--device', help="device utilized for protein embedding ('gpu' or 'cpu'), default: 'gpu'", type=str, default = 'gpu')
prot_t5_embed.py:parser.add_argument('--batch_size', help="batch size used for protein embedding. If your GPU is out of memory, you can try to reduce this parameter, default: 16", type=int, default = 16)
prot_t5_embed.py:        [--device DEVICE]   device utilized for protein embedding ('gpu' or 'cpu'), default: 'gpu'
prot_t5_embed.py:        [--batch_size BATCH_SIZE]   batch size used for protein embedding. If your GPU is out of memory, you can try to reduce this parameter, default: 16
prot_t5_embed.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prot_t5_embed.py:    if(device==torch.device("cuda")):
prot_t5_embed.py:        print("GPU detected. Running protein embedding with GPU ...")        
prot_t5_embed.py:        print("GPU not detected. Running protein embedding with CPU ...")
README.md:If you want to use GPU to accelerate the program:
README.md:* CUDA
README.md:* PyTorch-GPU
README.md:* For GPU version PyTorch: search [PyTorch](https://pytorch.org/get-started/previous-versions/) to find the correct CUDA version according to your computer
README.md:You can easily prepare the environment by using Anaconda to install ```plasgo.yaml```. This will install all packages you need in GPU mode (make sure you have installed CUDA on your system to use the GPU version. Otherwise, PlasGO will run in CPU mode). The installing command is: 
README.md:        [--device DEVICE] device utilized for generating original per-protein embeddings with ProtT5 ('gpu' or 'cpu'), default: 'gpu'
README.md:        [--batch_size BATCH_SIZE] batch size for protein embedding with the ProtT5 model. If your GPU is out of memory, you can try to reduce this parameter to 1, default: 8
README.md:        [--device DEVICE] device utilized for GO term prediction ('gpu' or 'cpu'), default: 'gpu'
README.md:        [--batch_size BATCH_SIZE] batch size (plasmid sentence count in a batch) for GO term prediction. If your GPU is out of memory during prediction, you can try to reduce this parameter, default: 32
README.md:        [--device DEVICE]   device utilized for training ('gpu' or 'cpu'), default: 'gpu'
README.md:        [--threads THREADS] number of threads utilized for training if 'cpu' is detected ('cuda' not found), default: 2
README.md:        [--device DEVICE]   device utilized for protein embedding ('gpu' or 'cpu'), default: 'gpu'
README.md:        [--batch_size BATCH_SIZE]   batch size used for protein embedding. If your GPU is out of memory, you can try to reduce this parameter, default: 16
plasgo_predict.py:parser.add_argument('--device', help="device utilized for GO term prediction ('gpu' or 'cpu'), default: 'gpu'", type=str, default = 'gpu')
plasgo_predict.py:parser.add_argument('--batch_size', help="batch size (plasmid sentence count in a batch) for GO term prediction. If your GPU is out of memory during prediction, you can try to reduce this parameter, default: 32", type=int, default = 32)
plasgo_predict.py:        [--device DEVICE] device utilized for GO term prediction ('gpu' or 'cpu'), default: 'gpu'
plasgo_predict.py:        [--batch_size BATCH_SIZE] batch size (plasmid sentence count in a batch) for GO term prediction. If your GPU is out of memory during prediction, you can try to reduce this parameter, default: 32
plasgo_predict.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plasgo_predict.py:    if(device==torch.device("cuda")):
plasgo_predict.py:        print("GPU detected. Running GO term prediction with GPU ...")        
plasgo_predict.py:        print("GPU not detected. Running GO term prediction with CPU ...")
library/import_utils.py:                "tensorflow-gpu",
library/import_utils.py:                "tf-nightly-gpu",
library/import_utils.py:                "tf-nightly-rocm",
library/import_utils.py:                "tensorflow-rocm",
library/import_utils.py:def is_torch_cuda_available():
library/import_utils.py:        return torch.cuda.is_available()
library/import_utils.py:def is_torch_bf16_gpu_available():
library/import_utils.py:    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
library/import_utils.py:    # the original bf16 check was for gpu only, but later a cpu/bf16 combo has emerged so this util
library/import_utils.py:        "The util is_torch_bf16_available is deprecated, please use is_torch_bf16_gpu_available "
library/import_utils.py:        "or is_torch_bf16_cpu_available instead according to whether it's used with cpu or gpu",
library/import_utils.py:    return is_torch_bf16_gpu_available()
library/import_utils.py:    if device == "cuda":
library/import_utils.py:        return is_torch_bf16_gpu_available()
library/import_utils.py:    if not torch.cuda.is_available() or torch.version.cuda is None:
library/import_utils.py:    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
library/import_utils.py:    if int(torch.version.cuda.split(".")[0]) < 11:
library/import_utils.py:    # bitsandbytes throws an error if cuda is not available
library/import_utils.py:    return _bitsandbytes_available and torch.cuda.is_available()
library/import_utils.py:    # Let's add an extra check to see if cuda is available
library/import_utils.py:    return _flash_attn_2_available and torch.cuda.is_available()
library/import_utils.py:`pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com`
train_plasgo.py:parser.add_argument('--device', help="device utilized for training ('gpu' or 'cpu'), default: 'gpu'", type=str, default = 'gpu')
train_plasgo.py:parser.add_argument('--threads', help="number of threads utilized for training if 'cpu' is detected ('cuda' not found), default: 2", type=int, default=2)
train_plasgo.py:        [--device DEVICE]   device utilized for training ('gpu' or 'cpu'), default: 'gpu'
train_plasgo.py:        [--threads THREADS] number of threads utilized for training if 'cpu' is detected ('cuda' not found), default: 2

```
