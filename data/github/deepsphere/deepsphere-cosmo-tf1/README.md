# https://github.com/deepsphere/deepsphere-cosmo-tf1

```console
requirements_py27.txt:#tensorflow-gpu==1.6.0
launch_cscs.py:#SBATCH --constraint=gpu
launch_cscs.py:module load daint-gpu
launch_cscs.py:module load TensorFlow/1.7.0-CrayGNU-17.12-cuda-8.0-python3
launch_cscs_2dcnn.py:#SBATCH --constraint=gpu
launch_cscs_2dcnn.py:module load daint-gpu
launch_cscs_2dcnn.py:module load TensorFlow/1.7.0-CrayGNU-18.08-cuda-9.1-python3
README.md:   **Note**: if you will be working with a GPU, comment the
README.md:   `tensorflow-gpu==1.6.0` line.
requirements.txt:#tensorflow-gpu==1.6.0
deepsphere/models.py:        Batch evaluation saves memory and enables this to run on smaller GPUs.

```
