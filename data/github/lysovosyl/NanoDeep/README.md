# https://github.com/lysovosyl/NanoDeep

```console
README.md:The code here has been tested using MinION and NVIDIA RTX3080 on live sequencing runs and NVIDIA P620 using playback on a simulated run.You are strongly advised to test your setup prior to running (see below for example tests) as this code does affect sequencing output. Please note that running this code is at your own risk.
README.md:nanodeep_trainmodel -data_path path_to_your_data/ -lable_path path_to_your_label/ -save_path ./save -model_name Nanodeep -device cuda:0 --save_best --load_to_mem -signal_length 4000 -epochs 100 -batch_size 50
README.md:nanodeep_testmodel -data_path path_to_your_data/ -lable_path path_to_your_label/ -save_path ./save --weight_path ../path_to_your_label/your_model_name.pth -model_name Nanodeep -device cuda:0  --load_to_mem -signal_length 4000  -batch_size 50
README.md:nanodeep_adaptivesample --filter_type deplete --filter_which 0  --model_name nanodeep --weight_path ./save/nanodeep.pth --first_channel 1 --last_channel 256 --compute_device cuda:0 
nanodeep/nanodeep_adaptivesample.py:def check_mem(cuda_device):
nanodeep/nanodeep_adaptivesample.py:        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
nanodeep/nanodeep_adaptivesample.py:    total, used = devices_info[int(cuda_device[-1])].split(',')
nanodeep/nanodeep_adaptivesample.py:def occumpy_mem(cuda_device):
nanodeep/nanodeep_adaptivesample.py:    total, used = check_mem(cuda_device)
nanodeep/nanodeep_adaptivesample.py:    x = torch.cuda.FloatTensor(256, 1024, block_mem)
nanodeep/nanodeep_adaptivesample.py:    parser.add_argument("--device", default='cuda:0', type=str, help="use which device")
nanodeep/nanodeep_trainmodel.py:    parser.add_argument('-device',default='cuda:0',help='The device use to train model')
nanodeep/nanodeep_trainmodel.py:            torch.cuda.set_device(opt.device)
nanodeep/nanodeep_trainmodel.py:            nanopore_gpu = rt_deep(model, opt.signal_length,opt.device)
nanodeep/nanodeep_trainmodel.py:            nanopore_gpu.load_data(data_path=opt.data_path,
nanodeep/nanodeep_trainmodel.py:            nanopore_gpu.load_data(data_path=opt.data_path,
nanodeep/nanodeep_trainmodel.py:            nanopore_gpu.train(
nanodeep/nanodeep_trainmodel.py:                    nanopore_gpu.load_data(data_path=opt.data_path,
nanodeep/nanodeep_trainmodel.py:                    nanopore_gpu.load_data(data_path=opt.assign_testset_data,
nanodeep/nanodeep_trainmodel.py:                confmat_data,acc_value,lable_all,predict_proba = nanopore_gpu.test_model(opt.batch_size)
nanodeep/nanodeep_trainmodel.py:                nanopore_gpu.draw_ROC(lable_all,predict_proba,roc_save_path)
nanodeep/nanodeep_trainmodel.py:        torch.cuda.set_device(opt.device)
nanodeep/nanodeep_trainmodel.py:        nanopore_gpu = rt_deep(model, opt.signal_length,opt.device)
nanodeep/nanodeep_trainmodel.py:        nanopore_gpu.load_data(data_path=opt.data_path,
nanodeep/nanodeep_trainmodel.py:        nanopore_gpu.load_data(data_path=opt.data_path,
nanodeep/nanodeep_trainmodel.py:        nanopore_gpu.train(
nanodeep/nanodeep_trainmodel.py:                nanopore_gpu.load_data(data_path=opt.data_path,
nanodeep/nanodeep_trainmodel.py:                nanopore_gpu.load_data(data_path=opt.assign_testset_data,
nanodeep/nanodeep_trainmodel.py:            confmat_data, acc_value, lable_all, predict_proba = nanopore_gpu.test_model(opt.batch_size)
nanodeep/nanodeep_trainmodel.py:            nanopore_gpu.draw_ROC(lable_all, predict_proba, roc_save_path)
nanodeep/nanodeep_testmodel.py:    parser.add_argument('-device', default='cuda:0', help='The device use to train model')
nanodeep/nanodeep_testmodel.py:    torch.cuda.set_device(opt.device)
nanodeep/nanodeep_testmodel.py:    nanopore_gpu = rt_deep(model, opt.signal_length,opt.device)
nanodeep/nanodeep_testmodel.py:    nanopore_gpu.load_the_model_weights(opt.model_path)
nanodeep/nanodeep_testmodel.py:    nanopore_gpu.load_data(data_path=opt.data_path,
nanodeep/nanodeep_testmodel.py:    confmat_data, acc_value, lable_all, predict_proba = nanopore_gpu.test_model(batch_size = opt.batch_size)
nanodeep/nanodeep_testmodel.py:    nanopore_gpu.draw_ROC(lable_all, predict_proba, roc_save_path)
read_deep/model/DeepSelectNet.py:# signal = signal.to('cuda:1')
read_deep/model/DeepSelectNet.py:# mymodel = mymodel.to('cuda:1')

```
