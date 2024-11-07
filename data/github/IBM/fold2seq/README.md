# https://github.com/IBM/fold2seq

```console
data/pdb_lists/train_domain.txt:1gpuA03-544-680
data/pdb_lists/train_domain.txt:2cudA00-1-79
data/pdb_lists/train_domain.txt:3gpuA01-2-133
data/pdb_lists/train_domain.txt:3gpuA02-134-274
environment.yml:cudatoolkit=10.2.89=hfd86e86_1
environment.yml:pytorch=1.7.1=py3.8_cuda10.2.89_cudnn7.6.5_0
src/fold_encoder.py:	if torch. cuda. is_available() == True and 'K40' not in torch.cuda.get_device_name(0):
src/fold_encoder.py:		device = 'cuda'
src/fold_encoder.py:	if torch. cuda. is_available() == True and 'K40' not in torch.cuda.get_device_name(0):
src/fold_encoder.py:		device = 'cuda'
src/inference.py:	if torch. cuda. is_available() == True and 'K40' not in torch.cuda.get_device_name(0):
src/inference.py:		device = 'cuda'
src/inference.py:	if torch. cuda. is_available() == True and 'K40' not in torch.cuda.get_device_name(0):
src/inference.py:		device = 'cuda'
src/inference.py:	if torch.cuda.device_count() > 1:
src/inference.py:		print("Let's use ",torch.cuda.device_count()," GPUs!")
src/inference.py:		args.batch_size*=torch.cuda.device_count()
src/train.py:    if torch. cuda. is_available() == True and 'K40' not in torch.cuda.get_device_name(0):
src/train.py:        device = 'cuda'
src/train.py:    if torch.cuda.device_count() > 1:
src/train.py:       print("Let's use ",torch.cuda.device_count()," GPUs!")
src/train.py:         f.write("Let's use "+str(torch.cuda.device_count())+" GPUs!")
src/train.py:       args.batch_size*=torch.cuda.device_count()

```
