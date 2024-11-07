# https://github.com/bioAI-Oslo/Spikeometric

```console
gpu_environment.yml:name: gpu-snn-glm-simulator
gpu_environment.yml:    - nvidia
gpu_environment.yml:        - cudatoolkit=11.7
gpu_environment.yml:        - pytorch-cuda=11.7
docs/index.rst:Make sure you get the cuda versions if you are planning to use a GPU.
docs/benchmarks/benchmarks.rst:`GitHub repository <https://github.com/bioAI-Oslo/Spikeometric>`_. The GPU used for the experiments in this section is the Nvidia V100 with 16GB of memory
docs/benchmarks/benchmarks.rst:CPU vs GPU
docs/benchmarks/benchmarks.rst:Spikeometric is designed to work well with both CPU and GPU architectures, but utilizing a GPU
docs/benchmarks/benchmarks.rst:For networks with up to 10 000 synapses, the CPU is faster due to overhead on the GPU, but while time per iteration remains
docs/benchmarks/benchmarks.rst:constant at about 0.3 ms up to about 2 500 000 synapses on the GPU, it increases from 0.15 ms at 1000 synapses to 30 ms per iteration at 2 500 000 synapses on the CPU.
docs/benchmarks/benchmarks.rst:.. figure:: ../_static/cpu_gpu.png
tests/test_device.py:def test_simulates_on_gpu_if_available(bernoulli_glm, example_data, stimulus):
tests/test_device.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tests/test_device.py:    assert spikes.is_cuda == torch.cuda.is_available()
tests/test_device.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tests/test_device.py:        assert bernoulli_glm.state_dict()[parameter].is_cuda == torch.cuda.is_available()
tests/test_device.py:    assert example_data.W0.is_cuda == torch.cuda.is_available()
tests/test_device.py:    assert example_data.edge_index.is_cuda == torch.cuda.is_available()

```
