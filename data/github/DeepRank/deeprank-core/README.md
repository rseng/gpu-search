# https://github.com/DeepRank/deeprank-core

```console
docs/source/installation.md:If you want to use the GPUs, choose a specific python version (note that at the moment we support python 3.10 only), are a MacOS user, or if the YML installation was not successful, you can install the package manually. We advise to do this inside a [conda virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
docs/source/installation.md:You can first create a copy of the `deeprank2.yml` file, place it in your current directory, and remove the packages that cannot be installed properly, or the ones that you want to install differently (e.g., pytorch-related packages if you wish to install the CUDA version), and then proceed with the environment creation by using the edited YML file: `conda env create -f deeprank2.yml` or `mamba env create -f deeprank2.yml`, if you have [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) installed. Then activate the environment, and proceed with installing the missing packages, which might fall into the following list. If you have any issues during installation of dependencies, please refer to the official documentation for each package (linked below), as our instructions may be out of date (last tested on 19 Feb 2024):
docs/source/installation.md:  - We support torch's CPU library as well as CUDA.
docs/source/conf.py:    "torch.cuda",
joss/paper.md:In the past decades, a variety of experimental methods (e.g., X-ray crystallography, nuclear magnetic resonance, cryogenic electron microscopy) have determined and accumulated a large number of atomic-resolution 3D structures of proteins and protein-protein complexes [@schwede_protein_2013]. Since experimental determination of structures is a tedious and expensive process, several computational prediction methods have been developed over the past decades, exploiting classical molecular modelling [@rosetta; @modeller; @haddock], and, more recently, DL [@alphafold_2021; @alphafold_multi]. The large amount of data available makes it possible to use DL to leverage 3D structures and learn their complex patterns. Unlike other machine learning (ML) techniques, deep neural networks hold the promise of learning from millions of data points without reaching a performance plateau quickly, which is made computationally feasible by hardware accelerators (i.e., GPUs, TPUs) and parallel file system technologies.
joss/paper.md:DeepRank2 also provides convenient pre-implemented modules for training simple [PyTorch](https://pytorch.org/)-based GNNs and CNNs using the data generated in the previous step. Alternatively, users can implement custom PyTorch networks in the DeepRank package (or export the data to external software). Data can be loaded across multiple CPUs, and the training can be run on GPUs. The data stored within the HDF5 files are read into customized datasets, and the user-friendly API allows for selection of individual features (from those generated above), definition of the targets, and the predictive task (classification or regression), among other settings. Then the datasets can be used for training, validating, and testing the chosen neural network. The final model and results can be saved using built-in data exporter modules.
joss/paper.md:This work was supported by the [Netherlands eScience Center](https://www.esciencecenter.nl/) under grant number NLESC.OEC.2021.008, and [SURF](https://www.surf.nl/en) infrastructure, and was developed in collaboration with the [Department of Medical BioSciences](https://www.radboudumc.nl/en/research/departments/medical-biosciences) at RadboudUMC (Hypatia Fellowship, Rv819.52706). This work was also supported from NVIDIA Academic Award.
tests/test_trainer.py:    use_cuda: bool = False,
tests/test_trainer.py:        cuda=use_cuda,
tests/test_trainer.py:    if use_cuda:
tests/test_trainer.py:        _log.debug("cuda is available, testing that the model is cuda")
tests/test_trainer.py:            assert parameter.is_cuda, f"{parameter} is not cuda"
tests/test_trainer.py:    def test_cuda(self) -> None:  # test_ginet, but with cuda
tests/test_trainer.py:        if torch.cuda.is_available():
tests/test_trainer.py:            warnings.warn("CUDA is not available; test_cuda was skipped")
tests/test_trainer.py:            _log.info("CUDA is not available; test_cuda was skipped")
tests/test_trainer.py:    def test_invalid_cuda_ngpus(self) -> None:
tests/test_trainer.py:                ngpu=2,
tests/test_trainer.py:    def test_invalid_no_cuda_available(self) -> None:
tests/test_trainer.py:        if not torch.cuda.is_available():
tests/test_trainer.py:                    cuda=True,
tests/test_trainer.py:            warnings.warn("CUDA is available; test_invalid_no_cuda_available was skipped")
tests/test_trainer.py:            _log.info("CUDA is available; test_invalid_no_cuda_available was skipped")
README.md:If you want to use the GPUs, choose a specific python version (note that at the moment we support python 3.10 only), are a MacOS user, or if the YML installation was not successful, you can install the package manually. We advise to do this inside a [conda virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
README.md:You can first create a copy of the `deeprank2.yml` file, place it in your current directory, and remove the packages that cannot be installed properly, or the ones that you want to install differently (e.g., pytorch-related packages if you wish to install the CUDA version), and then proceed with the environment creation by using the edited YML file: `conda env create -f deeprank2.yml` or `mamba env create -f deeprank2.yml`, if you have [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) installed. Then activate the environment, and proceed with installing the missing packages, which might fall into the following list. If you have any issues during installation of dependencies, please refer to the official documentation for each package (linked below), as our instructions may be out of date (last tested on 19 Feb 2024):
README.md:  - We support torch's CPU library as well as CUDA.
deeprank2/trainer.py:        cuda: Whether to use CUDA. Defaults to False.
deeprank2/trainer.py:        ngpu: Number of GPU to be used. Defaults to 0.
deeprank2/trainer.py:        cuda: bool = False,
deeprank2/trainer.py:        ngpu: int = 0,
deeprank2/trainer.py:        self.cuda = cuda
deeprank2/trainer.py:        self.ngpu = ngpu
deeprank2/trainer.py:        if self.cuda and torch.cuda.is_available():
deeprank2/trainer.py:            self.device = torch.device("cuda")
deeprank2/trainer.py:            if self.ngpu == 0:
deeprank2/trainer.py:                self.ngpu = 1
deeprank2/trainer.py:                _log.info("CUDA detected. Setting number of GPUs to 1.")
deeprank2/trainer.py:        elif self.cuda and not torch.cuda.is_available():
deeprank2/trainer.py:                --> CUDA not detected: Make sure that CUDA is installed
deeprank2/trainer.py:                    and that you are running on GPUs.\n
deeprank2/trainer.py:                --> To turn CUDA off set cuda=False in Trainer.\n
deeprank2/trainer.py:                --> CUDA not detected: Make sure that CUDA is installed
deeprank2/trainer.py:                    and that you are running on GPUs.\n
deeprank2/trainer.py:                --> To turn CUDA off set cuda=False in Trainer.\n
deeprank2/trainer.py:            if self.ngpu > 0:
deeprank2/trainer.py:                    --> CUDA not detected.
deeprank2/trainer.py:                        Set cuda=True in Trainer to turn CUDA on.\n
deeprank2/trainer.py:                    --> CUDA not detected.
deeprank2/trainer.py:                        Set cuda=True in Trainer to turn CUDA on.\n
deeprank2/trainer.py:        if self.device.type == "cuda":
deeprank2/trainer.py:            _log.info(f"CUDA device name is {torch.cuda.get_device_name(0)}.")
deeprank2/trainer.py:            _log.info(f"Number of GPUs set to {self.ngpu}.")
deeprank2/trainer.py:        self.test_loader = DataLoader(self.dataset_test, pin_memory=self.cuda)
deeprank2/trainer.py:        # multi-gpu
deeprank2/trainer.py:        if self.ngpu > 1:
deeprank2/trainer.py:            ids = list(range(self.ngpu))
deeprank2/trainer.py:            pin_memory=self.cuda,
deeprank2/trainer.py:                pin_memory=self.cuda,
deeprank2/trainer.py:            if self.cuda:
deeprank2/trainer.py:            if self.cuda:
deeprank2/trainer.py:                pin_memory=self.cuda,
deeprank2/trainer.py:        if torch.cuda.is_available():
deeprank2/trainer.py:        self.cuda = state["cuda"]
deeprank2/trainer.py:        self.ngpu = state["ngpu"]
deeprank2/trainer.py:            "cuda": self.cuda,
deeprank2/trainer.py:            "ngpu": self.ngpu,
deeprank2/dataset.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deeprank2/dataset.py:                if torch.cuda.is_available():

```
