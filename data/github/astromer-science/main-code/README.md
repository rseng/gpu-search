# https://github.com/astromer-science/main-code

```console
Dockerfile:FROM tensorflow/tensorflow:2.8.0-gpu
Dockerfile:RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
Dockerfile:RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
run_container.sh:echo 'Looking for GPUs (ETA: 10 seconds)'
run_container.sh:gpu=$(lspci | grep -i '.* vga .* nvidia .*')
run_container.sh:if [[ $gpu == *' nvidia '* ]]; then
run_container.sh:  echo GPU found
run_container.sh:    --gpus all \
README.md:Automatically, the script recognizes if there are GPUs, making them visible inside the container.
presentation/scripts/testing.py:    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
presentation/scripts/testing.py:    parser.add_argument('--gpu', default='-1', type=str,
presentation/scripts/testing.py:                        help='GPU number to be used')
presentation/scripts/testing_script.py:gpu = sys.argv[1]
presentation/scripts/testing_script.py:                                    --gpu {}'.format(ds_name, fold_n, dataset,
presentation/scripts/testing_script.py:                                                     gpu)
presentation/scripts/create_records.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
presentation/scripts/classify.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
presentation/scripts/classify.py:    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
presentation/scripts/classify.py:    parser.add_argument('--gpu', default='0', type=str,
presentation/scripts/classify.py:                        help='GPU number to be used')
presentation/scripts/train.py:    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
presentation/scripts/train.py:    parser.add_argument('--gpu', default='0', type=str,
presentation/scripts/train.py:                        help='GPU to use')
presentation/scripts/finetuning.py:    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
presentation/scripts/finetuning.py:    parser.add_argument('--gpu', default='0', type=str,
presentation/scripts/finetuning.py:                        help='GPU to use')
presentation/scripts/ft_script.py:gpu = sys.argv[1]
presentation/scripts/ft_script.py:                   --gpu {}\
presentation/scripts/ft_script.py:                                    gpu,
presentation/scripts/lstm_script.py:GPU = sys.argv[1]
presentation/scripts/lstm_script.py:os.environ["CUDA_VISIBLE_DEVICES"]= GPU
presentation/scripts/clf_script.py:gpu = sys.argv[1]
presentation/scripts/clf_script.py:                                    --gpu {}'.format(ds_name, fold_n, dataset,
presentation/scripts/clf_script.py:                                                     gpu)
presentation/experiments/clf/evaluate.py:os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]

```
