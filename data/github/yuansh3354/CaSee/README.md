# https://github.com/yuansh3354/CaSee

```console
GSE131907_example/Step-06.number_of_expression_genes_in_cancer_normal_cells.r:library(ggpubr)
configs/CaSee_Model_configs.yaml:  batch_size: 128  # if GPU not enough, plz set sutiable number  
configs/CaSee_Model_configs.yaml:  gpu: True
GSE131907_example.yaml:  batch_size: 128  # if GPU not enough, plz set sutiable number  
GSE131907_example.yaml:  gpu: True
CaSee.py:gpu = config_dict['trainig_loop']['gpu']
CaSee.py:        if gpu == True:
CaSee.py:            trainer = pl.Trainer(gpus=-1,
CaSee.py:        if gpu == True:
CaSee.py:            trainer = pl.Trainer(gpus=-1)
CaSee.py:    if gpu == True:
CaSee.py:        trainer = pl.Trainer(gpus=-1)
README.md:- NVIDIA GeForce RTX 3090 24GB 384bit 1695MHz 19500MHz 
README.md:  batch_size: 128  # if GPU not enough, plz set sutiable number  
README.md:  gpu:True 
sup_script/BscModel_V4_configs.yaml:  batch_size: 128  # if GPU not enough, plz set sutiable number  
Time_log/CaSee-nohup.out:GPU available: True, used: True
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:GPU available: True, used: True
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:GPU available: True, used: True
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
Time_log/CaSee-nohup.out:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Time_log/CaSee-nohup.out:
utils/myexptorch.py:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
utils/myexptorch.py:    if (torch.cuda.is_available() & x == 'cuda'):
utils/myexptorch.py:        device = torch.device('cuda')

```