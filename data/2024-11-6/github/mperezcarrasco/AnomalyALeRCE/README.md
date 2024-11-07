# https://github.com/mperezcarrasco/AnomalyALeRCE

```console
Dockerfile:FROM tensorflow/tensorflow:2.8.0-gpu
Dockerfile:RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
Dockerfile:RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
run_container.sh:echo 'Looking for GPUs (ETA: 10 seconds)'
run_container.sh:gpu=$(lspci | grep -i '.* vga .* nvidia .*')
run_container.sh:if [[ $gpu == *' nvidia '* ]]; then
run_container.sh:  echo GPU found
run_container.sh:    --gpus all \
README.md:This script automatically finds the `anomalydetector` container and runs it on top of your kernel. If GPUs are available, the script makes them visible inside the container.
main.py:    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
main.py:    print('You are using GPU', args.device)
src/models/main.py:    #if torch.cuda.is_available():
src/models/main.py:    #    if torch.cuda.device_count() > 1:
src/evaluate.py:    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

```
