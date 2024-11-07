# https://github.com/TingMAC/FrMLNet

```console
train_QB.py:os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
train_QB.py:dtype = torch.cuda.FloatTensor
train_QB.py:    CNN = nn.DataParallel(CNN,device_ids=devicesList).cuda()  
test_QB.py:os.environ['CUDA_VISIBLE_DEVICES']='2'
test_QB.py:dtype = torch.cuda.FloatTensor
test_QB.py:    CNN = nn.DataParallel(CNN,device_ids=devicesList).cuda()

```
