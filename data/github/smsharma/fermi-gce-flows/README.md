# https://github.com/smsharma/fermi-gce-flows

```console
sbi/inference/snpe/snpe.py:            gpus=[0],  # Hard-coded
sbi/inference/base.py:            device: torch device on which to compute, e.g. gpu or cpu.
sbi/utils/torchutils.py:    """Set and return the default device to cpu or gpu."""
sbi/utils/torchutils.py:        if device == "gpu":
sbi/utils/torchutils.py:            device = "cuda"
sbi/utils/torchutils.py:            #     """GPU was selected as a device for training the neural network. Note
sbi/utils/torchutils.py:            #        default architectures we provide. Using the GPU will be effective
sbi/utils/torchutils.py:            #        GPU, e.g., for a CNN or RNN `embedding_net`."""
environment.yml:  - cudatoolkit=10.2.89=h8f6ccaa_9
environment.yml:  - pytorch=1.9.1=py3.9_cuda10.2_cudnn7.6.5_0
paper/ml4ps/fermi-gce-flows-ml4ps.tex:	\item Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)?
paper/ml4ps/neurips_2021.tex:	\item Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)?
paper/ml4ps-camera-ready/fermi-gce-flows-ml4ps.tex:Normalizing flows allow for tractable density evaluation, and $\log \hat p_\phi(\theta\mid s_\varphi(x))$ is used as the training objective to simultaneously optimize parameters $\{\phi,\varphi\}$ associated with the convolution and flow neural networks, respectively. $10^6$ samples from the forward model are produced, with 15\% of these held out for validation. The model is trained for up to 30 epochs with early stopping, using a batch size of 256. The \texttt{AdamW} optimizer~\cite{DBLP:journals/corr/KingmaB14,DBLP:conf/iclr/LoshchilovH19} is used with initial learning rate $10^{-3}$ and weight decay $10^{-5}$, with learning rate decayed through cosine annealing. Experiments were performed on RTX8000 GPUs at the NYU \emph{Greene} computing cluster.
paper/ml4ps-camera-ready/fermi-gce-flows-ml4ps.tex:	\item Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)?
paper/ml4ps-camera-ready/neurips_2021.tex:	\item Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)?
scripts/run_jupyter.sh:#SBATCH --gres=gpu:1
scripts/train.sh:#SBATCH --gres=gpu:rtx8000
scripts/combine_samples.sh:# #SBATCH --gres=gpu:1
scripts/simulate.sh:# #SBATCH --gres=gpu:1
scripts/submit_train.py:#SBATCH --gres=gpu:1
train.py:        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

```
