# https://github.com/anuradhawick/LRBinner

```console
lrbinner.py:    common_arg_parser.add_argument('--cuda',
lrbinner.py:                        help='Whether to use CUDA if available.'
lrbinner.py:            Dimension reduced reads are then clustered. Minimum RAM requirement is 9GB (4GB GPU if cuda used).""", add_help=True)
lrbinner.py:    cuda = args.cuda
lrbinner.py:    if cuda:
lrbinner.py:        if torch.cuda.is_available():
lrbinner.py:            cuda = True
lrbinner.py:            logger.info("CUDA found in system")
lrbinner.py:            cuda = False
lrbinner.py:            logger.info("CUDA not found in system")
Dockerfile:RUN conda install -y numpy scipy seaborn h5py hdbscan gcc openmp tqdm biopython fraggenescan hmmer tabulate pytorch pytorch-cuda=11.7 -c pytorch -c nvidia -c bioconda c conda-forge biopython
Dockerfile:# docker run  --rm -it --gpus all -v `pwd`:`pwd` -u `id -u`:`id -g`  anuradhawick/lrbinner
Dockerfile:# docker run  --rm -it --gpus all -v `pwd`:`pwd` -u `id -u`:`id -g` --entrypoint /bin/bash  anuradhawick/lrbinner
mbcclr_utils/ae_utils.py:def make_data_loader(covs, profs, *, batch_size=1024, drop_last=True, shuffle=True, cuda=False):
mbcclr_utils/ae_utils.py:    n_workers = 4 if cuda else 1
mbcclr_utils/ae_utils.py:                      shuffle=shuffle, pin_memory=cuda, num_workers=n_workers)
mbcclr_utils/ae_utils.py:def vae_encode(output, latent_dims, hidden_layers, epochs, constraints, cuda):
mbcclr_utils/ae_utils.py:    if cuda:
mbcclr_utils/ae_utils.py:        device = "cuda"
mbcclr_utils/ae_utils.py:    dloader = make_data_loader(cov_profiles, comp_profiles, cuda=cuda)
mbcclr_utils/ae_utils.py:        cov_profiles, comp_profiles, drop_last=False, shuffle=False, cuda=cuda)
mbcclr_utils/pipelines.py:    cuda = args.cuda
mbcclr_utils/pipelines.py:    cuda = args.cuda
mbcclr_utils/pipelines.py:            cuda)
mbcclr_utils/pipelines.py:    cuda = args.cuda
mbcclr_utils/pipelines.py:            cuda)
mbcclr_utils/cluster_utils.py:def calc_densities(histogram, cuda=False, pdf=_NORMALPDF):
mbcclr_utils/cluster_utils.py:    if cuda:
README.md:conda create -n lrbinner -y python=3.10 numpy scipy seaborn h5py hdbscan gcc openmp tqdm biopython fraggenescan hmmer tabulate pytorch pytorch-cuda=11.7 -c pytorch -c nvidia -c bioconda
README.md:python LRBinner reads -r reads.fasta -bc 10 -bs 32 -o lrb --resume --cuda -mbs 5000 --ae-dims 4 --ae-epochs 200 -bit 0 -t 32
README.md:clustered. Minimum RAM requirement is 9GB (4GB GPU if cuda used).
README.md:                      [--cuda] [--resume] --output <DEST> [--version]
README.md:  --cuda                Whether to use CUDA if available.
README.md:                        [--separate] [--cuda] [--resume] --output <DEST>
README.md:  --cuda                Whether to use CUDA if available.

```
