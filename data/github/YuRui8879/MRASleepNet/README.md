# https://github.com/YuRui8879/MRASleepNet

```console
Algorithm/Algorithm.py:    def __init__(self,model_save_path,log_save_path,split_data_file_path,batch_size,learning_rate,epochs,cuda_device,reg_para,parallel = True):
Algorithm/Algorithm.py:        self.parallel = parallel # Whether to use multi GPU calculation
Algorithm/Algorithm.py:        self.cuda_device = cuda_device # Designated Training GPU
Algorithm/Algorithm.py:            self.model.cuda()
Algorithm/Algorithm.py:            self.model.cuda(self.cuda_device)
Algorithm/Algorithm.py:            weight = torch.FloatTensor(train_adapter.calc_class_weight()).cuda()
Algorithm/Algorithm.py:            weight = torch.FloatTensor(train_adapter.calc_class_weight()).cuda(self.cuda_device)
Algorithm/Algorithm.py:                inputs,labels = data[0].squeeze().cuda(),data[1].squeeze().cuda()
Algorithm/Algorithm.py:                inputs,labels = data[0].squeeze().cuda(self.cuda_device),data[1].squeeze().cuda(self.cuda_device)
Algorithm/Algorithm.py:            model.cuda()
Algorithm/Algorithm.py:            model.cuda(self.cuda_device)
Algorithm/Algorithm.py:                inputs,labels = data[0].squeeze().cuda(),data[1].squeeze().cuda()
Algorithm/Algorithm.py:                inputs,labels = data[0].squeeze().cuda(self.cuda_device),data[1].squeeze().cuda(self.cuda_device)
README.md:--parallel        Whether to use multi-GPU training
README.md:--cuda_device     If you do not use multi-GPU training, you need to specify the GPU
main.py:parser.add_argument('-p', '--parallel', type=bool, default=False, help = 'Whether to use multi-GPU training')
main.py:parser.add_argument('-c', '--cuda_device', type=int, default=0, help = 'If you do not use multi-GPU training, you need to specify the GPU')
main.py:cuda_device = args.cuda_device
main.py:                cuda_device = cuda_device,reg_para = reg_parameter, parallel = parallel)

```
