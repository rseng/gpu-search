# https://github.com/ylxu05/HN-PPISP

```console
predict.py:os.environ["CUDA_VISIBLE_DEVICES"] = "0"
predict.py:            if torch.cuda.is_available():
predict.py:                seq_var = torch.autograd.Variable(seq_data.cuda().float())
predict.py:                pssm_var = torch.autograd.Variable(pssm_data.cuda().float())
predict.py:                dssp_var = torch.autograd.Variable(dssp_data.cuda().float())
predict.py:                local_var = torch.autograd.Variable(local_data.cuda().float())
predict.py:                target_var = torch.autograd.Variable(label.cuda().float())
predict.py:                                              sampler=test_samples, pin_memory=(torch.cuda.is_available()),
predict.py:    model = model.cuda()
train.py:os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train.py:            if torch.cuda.is_available():
train.py:                seq_var = torch.autograd.Variable(seq_data.cuda().float())
train.py:                pssm_var = torch.autograd.Variable(pssm_data.cuda().float())
train.py:                dssp_var = torch.autograd.Variable(dssp_data.cuda().float())
train.py:                local_var = torch.autograd.Variable(local_data.cuda().float())
train.py:                target_var = torch.autograd.Variable(label.cuda().float())
train.py:        loss = torch.nn.functional.binary_cross_entropy(output, target_var).cuda()
train.py:            if torch.cuda.is_available():
train.py:                seq_var = torch.autograd.Variable(seq_data.cuda().float())
train.py:                pssm_var = torch.autograd.Variable(pssm_data.cuda().float())
train.py:                dssp_var = torch.autograd.Variable(dssp_data.cuda().float())
train.py:                local_var = torch.autograd.Variable(local_data.cuda().float())
train.py:                target_var = torch.autograd.Variable(label.cuda().float())
train.py:        loss = torch.nn.functional.binary_cross_entropy(output, target_var).cuda()
train.py:                                               sampler=train_samples, pin_memory=(torch.cuda.is_available()),
train.py:                                               sampler=eval_samples, pin_memory=(torch.cuda.is_available()),
train.py:    # Model on cuda
train.py:    if torch.cuda.is_available():
train.py:        model = model.cuda()
train.py:    # Wrap model for multi-GPUs, if necessary

```
