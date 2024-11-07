# https://github.com/li-shaochuan/SEDIM

```console
imputeByBBO.py:gpu_id = '2,0,1,3'
imputeByBBO.py:os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
imputeByBBO.py:os.system('echo $CUDA_VISIBLE_DEVICES')
imputeByBBO.py:config.gpu_options.allocator_type = 'BFC'
imputeByBBO.py:config.gpu_options.per_process_gpu_memory_fraction =1
imputeByBBO.py:config.gpu_options.allow_growth = True
README.md:By default, we provide the origin code. SEDIM is written in python 3.7, and tensorflow-gpu==1.14.
main.py:gpu_id = '1,3'
main.py:os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
main.py:os.system('echo $CUDA_VISIBLE_DEVICES')
main.py:config.gpu_options.allocator_type = 'BFC'
main.py:config.gpu_options.per_process_gpu_memory_fraction =1
main.py:config.gpu_options.allow_growth = True

```
