# https://github.com/exosports/MARGE

```console
environment.yml:  - tensorflow-gpu==1.13.1
doc/MARGE_user_manual/MARGE_user_manual.tex:\item CUDA 9.1.85
doc/MARGE_user_manual/MARGE_user_manual.tex:\noindent Users that do not have Nvidia GPU drivers installed will need to 
doc/MARGE_user_manual/MARGE_user_manual.tex:remove the tensorflow-gpu package:
doc/MARGE_user_manual/MARGE_user_manual.tex:conda remove -n marge tensorflow-gpu
doc/MARGE_user_manual/MARGE_user_manual.tex:\item GPU with \textgreater= 4 GB RAM
doc/MARGE_user_manual/MARGE_user_manual.tex:observations.  On GPUs, calculations will be optimized for 2\^{}N batch 
example/README:Optional    : GPU with >= 4 GB RAM
example/README:CPU/GPU used.
README:80NSSC20K0682.  We gratefully thank Nvidia Corporation for the Titan Xp GPU 
README:that it uses tensorflow-gpu, which assumes the user has Nvidia drivers 
README:installed.  If this is not the case, after building the environment, non-GPU 
README:    conda remove -n marge tensorflow-gpu
README:and follow the prompts.  This will remove the requirement for Nvidia drivers.
README: - CUDA 9.1.85

```
