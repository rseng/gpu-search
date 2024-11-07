# https://github.com/haiyang1986/TVAR

```console
tvar.yml:  - _tflow_select=2.1.0=gpu
tvar.yml:  - cudatoolkit=9.2=0
tvar.yml:  - cudnn=7.6.5=cuda9.2_0
tvar.yml:  - tensorflow=1.11.0=gpu_py36h9c9050a_0
tvar.yml:  - tensorflow-base=1.11.0=gpu_py36had579c0_0
tvar.yml:  - tensorflow-gpu=1.11.0=h0d30ee6_0
run.sh:#python TVar_gpu.py -m cv
README.md:TVAR is a tissue-specific functional annotation tool of non-coding variants based on multi-label deep learning. The framework's input is the vcf profiles of non-coding variants. The output is the corresponding functional score vectors across the GTEx 49 tissues. TVAR is mainly divided into two components: Feature extraction and result analysis module running on the CPU (TVAR-CPU). 2. Model training and scoring module running on GPU (TVAR-GPU).
README.md:python TVar_gpu.py -m train  -i ./input/input.gz
README.md:python TVar_gpu.py -m cv  -i ./input/input.gz
README.md:python TVar_gpu.py -m score -i ./input/input.vcf
README.md:TVAR was based on open-source Python 3.6 libraries. The deep learning network's implementation was based on Numpy 1.15.4, Scipy 1.1.0, Tensorlayer 1.11.1 (GPU version) and Tensorflow 1.11.0 (GPU version). After the testing, TVAR has been working correctly on Ubuntu Linux release 20.04. We used the NVIDIA Tesla T4 for model training and testing.
run_eval_all.py:    gpu_path = './'
run_eval_all.py:    #     cmd1 = "python TVar_gpu.py -m cv"
run_eval_all.py:        cmd1 = "python TVar_gpu.py -m train"
run_eval_all.py:            cmd = "python TVar_gpu.py -m score -i %s" % (line.rstrip())
run_eval_all.py:    #         cmd = "python TVar_gpu.py -m rare -n %s -i %s" % (tissue, line.rstrip())
run_eval_all.py:    #         cmd = "python TVar_gpu.py -m gwas -n %s -i %s" % (tissue, line.rstrip())
readme.md:TVAR is a tissue-specific functional annotation tool of non-coding variants based on multi-label deep learning. The framework's input is the vcf profiles of non-coding variants. The output is the corresponding functional score vectors across the GTEx 49 tissues. TVAR is mainly divided into two components: Feature extraction and result analysis module running on the CPU (TVAR-CPU). 2. Model training and scoring module running on GPU (TVAR-GPU).
readme.md:python TVar_gpu.py -m train  -i ./input/input.gz
readme.md:python TVar_gpu.py -m cv  -i ./input/input.gz
readme.md:python TVar_gpu.py -m score -i ./input/input.vcf
readme.md:TVAR was based on open-source Python 3.6 libraries. The deep learning network's implementation was based on Numpy 1.15.4, Scipy 1.1.0, Tensorlayer 1.11.1 (GPU version) and Tensorflow 1.11.0 (GPU version). After the testing, TVAR has been working correctly on Ubuntu Linux release 20.04. We used the NVIDIA Tesla T4 for model training and testing.

```
