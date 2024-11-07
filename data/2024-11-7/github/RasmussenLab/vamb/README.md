# https://github.com/RasmussenLab/vamb

```console
workflow_avamb/README.md:## Using a GPU to speed up Avamb
workflow_avamb/README.md:Using a GPU can speed up Avamb considerably - especially when you are binning millions of contigs. In order to enable it you need to make a couple of changes to the configuration file. Basically we need to add `--cuda` to the `avamb_params` to tell Avamb to use the GPU. Then if you are using the `--cluster` option, you also need to update `avamb_ppn` accordingly - e.g. on our system (qsub) we exchange `"avamb_ppn": "10"` to `"avamb_ppn": "10:gpus=1"`. Therefore the `config.json` file looks like this if I want to use GPU acceleration:
workflow_avamb/README.md:   "avamb_ppn": "10:gpus=1",
workflow_avamb/README.md:   "avamb_params": " --model vae-aae  -o C --minfasta 500000 --cuda",
workflow_avamb/README.md:   "avamb_preload": "module load cuda/toolkit/10.2.89;",
workflow_avamb/README.md:Note that I could not get `avamb` to work with `cuda` on our cluster when installing from bioconda. Therefore I added a line to preload cuda toolkit to the configuration file that will load this module when running `avamb`. 
workflow_avamb/avamb.snake.conda.smk:AVAMB_PPN = get_config("avamb_ppn", "10", r"[1-9]\d*(:gpus=[1-9]\d*)?$")
workflow_avamb/avamb.snake.conda.smk:# parse if GPUs is needed #
workflow_avamb/avamb.snake.conda.smk:avamb_threads, sep, avamb_gpus = AVAMB_PPN.partition(":gpus=")
workflow_avamb/avamb.snake.conda.smk:CUDA = len(avamb_gpus) > 0
workflow_avamb/avamb.snake.conda.smk:        cuda="--cuda" if CUDA else ""
workflow_avamb/avamb.snake.conda.smk:        vamb --outdir {output.outdir_avamb} --fasta {input.contigs} -p {threads} --abundance {input.abundance} -m {MIN_CONTIG_SIZE} --minfasta {MIN_BIN_SIZE}  {params.cuda}  {AVAMB_PARAMS}
vamb/encode.py:    cuda: bool = False,
vamb/encode.py:        cuda: Pagelock memory of dataloader (use when using GPU acceleration)
vamb/encode.py:    n_workers = 4 if cuda else 1
vamb/encode.py:        pin_memory=cuda,
vamb/encode.py:        cuda: Use CUDA (GPU accelerated training) [False]
vamb/encode.py:        cuda: bool = False,
vamb/encode.py:        self.usecuda = cuda
vamb/encode.py:        if cuda:
vamb/encode.py:            self.cuda()
vamb/encode.py:        if self.usecuda:
vamb/encode.py:            epsilon = epsilon.cuda()
vamb/encode.py:            if self.usecuda:
vamb/encode.py:                depths_in = depths_in.cuda()
vamb/encode.py:                tnf_in = tnf_in.cuda()
vamb/encode.py:                abundance_in = abundance_in.cuda()
vamb/encode.py:                weights = weights.cuda()
vamb/encode.py:                # Move input to GPU if requested
vamb/encode.py:                if self.usecuda:
vamb/encode.py:                    depths = depths.cuda()
vamb/encode.py:                    tnf = tnf.cuda()
vamb/encode.py:                    ab = ab.cuda()
vamb/encode.py:                if self.usecuda:
vamb/encode.py:        cls, path: Union[IO[bytes], str], cuda: bool = False, evaluate: bool = True
vamb/encode.py:            cuda: If network should work on GPU [False]
vamb/encode.py:        # Forcably load to CPU even if model was saves as GPU model
vamb/encode.py:        vae = cls(nsamples, nhiddens, nlatent, alpha, beta, dropout, cuda)
vamb/encode.py:        if cuda:
vamb/encode.py:            vae.cuda()
vamb/encode.py:        logger.info(f"\tCUDA: {self.usecuda}")
vamb/semisupervised_encode.py:    rpkm, tnf, lengths, batchsize: int = 256, destroy: bool = False, cuda: bool = False
vamb/semisupervised_encode.py:    n_workers = 4 if cuda else 1
vamb/semisupervised_encode.py:    dataloader = _encode.make_dataloader(rpkm, tnf, lengths, batchsize, destroy, cuda)
vamb/semisupervised_encode.py:        cuda,
vamb/semisupervised_encode.py:    cuda: bool = False,
vamb/semisupervised_encode.py:        cuda,
vamb/semisupervised_encode.py:        rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda
vamb/semisupervised_encode.py:        pin_memory=cuda,
vamb/semisupervised_encode.py:    cuda: bool = False,
vamb/semisupervised_encode.py:    _, _, _, _, batchsize, n_workers, cuda = _make_dataset(
vamb/semisupervised_encode.py:        rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda
vamb/semisupervised_encode.py:        pin_memory=cuda,
vamb/semisupervised_encode.py:        cuda: Use CUDA (GPU accelerated training) [False]
vamb/semisupervised_encode.py:        cuda: bool = False,
vamb/semisupervised_encode.py:            cuda=cuda,
vamb/semisupervised_encode.py:        if self.usecuda:
vamb/semisupervised_encode.py:            logsigma = logsigma.cuda()
vamb/semisupervised_encode.py:            if self.usecuda:
vamb/semisupervised_encode.py:                labels_in = labels_in.cuda()
vamb/semisupervised_encode.py:                # Move input to GPU if requested
vamb/semisupervised_encode.py:                if self.usecuda:
vamb/semisupervised_encode.py:                    labels = labels.cuda()
vamb/semisupervised_encode.py:                if self.usecuda:
vamb/semisupervised_encode.py:        logger.info(f"\tCUDA: {self.usecuda}")
vamb/semisupervised_encode.py:        cuda: Use CUDA (GPU accelerated training) [False]
vamb/semisupervised_encode.py:        cuda: bool = False,
vamb/semisupervised_encode.py:            cuda=cuda,
vamb/semisupervised_encode.py:        if self.usecuda:
vamb/semisupervised_encode.py:            logsigma = logsigma.cuda()
vamb/semisupervised_encode.py:            if self.usecuda:
vamb/semisupervised_encode.py:                depths_in = depths_in.cuda()
vamb/semisupervised_encode.py:                tnf_in = tnf_in.cuda()
vamb/semisupervised_encode.py:                abundance_in = abundance_in.cuda()
vamb/semisupervised_encode.py:                weights = weights.cuda()
vamb/semisupervised_encode.py:                labels_in = labels_in.cuda()
vamb/semisupervised_encode.py:                # Move input to GPU if requested
vamb/semisupervised_encode.py:                if self.usecuda:
vamb/semisupervised_encode.py:                    depths = depths.cuda()
vamb/semisupervised_encode.py:                    tnf = tnf.cuda()
vamb/semisupervised_encode.py:                    ab = ab.cuda()
vamb/semisupervised_encode.py:                    labels = labels.cuda()
vamb/semisupervised_encode.py:                if self.usecuda:
vamb/semisupervised_encode.py:        cuda: Use CUDA (GPU accelerated training) [False]
vamb/semisupervised_encode.py:        cuda: bool = False,
vamb/semisupervised_encode.py:            cuda=cuda,
vamb/semisupervised_encode.py:            cuda=cuda,
vamb/semisupervised_encode.py:            cuda=cuda,
vamb/semisupervised_encode.py:            if self.VAEVamb.usecuda:
vamb/semisupervised_encode.py:                depths_in_sup = depths_in_sup.cuda()
vamb/semisupervised_encode.py:                tnf_in_sup = tnf_in_sup.cuda()
vamb/semisupervised_encode.py:                abundance_in_sup = abundance_in_sup.cuda()
vamb/semisupervised_encode.py:                weights_in_sup = weights_in_sup.cuda()
vamb/semisupervised_encode.py:                labels_in_sup = labels_in_sup.cuda()
vamb/semisupervised_encode.py:                depths_in_unsup = depths_in_unsup.cuda()
vamb/semisupervised_encode.py:                tnf_in_unsup = tnf_in_unsup.cuda()
vamb/semisupervised_encode.py:                abundance_in_unsup = abundance_in_unsup.cuda()
vamb/semisupervised_encode.py:                weights_in_unsup = weights_in_unsup.cuda()
vamb/semisupervised_encode.py:                labels_in_unsup = labels_in_unsup.cuda()
vamb/semisupervised_encode.py:            if self.usecuda:
vamb/semisupervised_encode.py:                logsigma_vamb_unsup = logsigma_vamb_unsup.cuda()
vamb/semisupervised_encode.py:            if self.usecuda:
vamb/semisupervised_encode.py:                logsigma_vamb_sup_s = logsigma_vamb_sup_s.cuda()
vamb/semisupervised_encode.py:        logger.info(f"\tCUDA: {self.VAEVamb.usecuda}")
vamb/semisupervised_encode.py:    def load(cls, path, cuda=False, evaluate=True):
vamb/semisupervised_encode.py:            cuda: If network should work on GPU [False]
vamb/semisupervised_encode.py:        # Forcably load to CPU even if model was saves as GPU model
vamb/semisupervised_encode.py:        vae = cls(nsamples, nlabels, nhiddens, nlatent, alpha, beta, dropout, cuda)
vamb/semisupervised_encode.py:        if cuda:
vamb/semisupervised_encode.py:            vae.VAEVamb.cuda()
vamb/semisupervised_encode.py:            vae.VAELabels.cuda()
vamb/semisupervised_encode.py:            vae.VAEJoint.cuda()
vamb/cluster.py:        cuda: Accelerate clustering with GPU [False]
vamb/cluster.py:        # Whether this clusterer runs on GPU
vamb/cluster.py:        "cuda",
vamb/cluster.py:        # but if not on GPU, we delete used rows of the matrix (and indices) to speed up subsequent computations.
vamb/cluster.py:        # On GPU, deleting rows is not feasable because that requires us to copy the matrix from and to the GPU,
vamb/cluster.py:  CUDA:         {self.cuda}
vamb/cluster.py:        if self.cuda:
vamb/cluster.py:            kept_mask = kept_mask.cuda()
vamb/cluster.py:        cuda: bool = False,
vamb/cluster.py:        # Move to GPU
vamb/cluster.py:        if cuda:
vamb/cluster.py:            torch_matrix = torch_matrix.cuda()
vamb/cluster.py:            torch_lengths = torch_lengths.cuda()
vamb/cluster.py:        self.cuda: bool = cuda
vamb/cluster.py:        if cuda:
vamb/cluster.py:            self.lengths = self.lengths.cuda()
vamb/cluster.py:        # distance calculation by having fewer points. Worth it on CPU, not on GPU
vamb/cluster.py:        if not self.cuda:
vamb/cluster.py:        if self.cuda:
vamb/cluster.py:            ).cuda()
vamb/cluster.py:            # When running on GPU, we also take this (rare-ish) chance to pack the on-GPU matrix.
vamb/cluster.py:            # moving the matrix from and to GPU, so is slow.
vamb/cluster.py:                if self.cuda:
vamb/cluster.py:                or (self.cuda and not self.kept_mask[new_index].item())
vamb/cluster.py:        # We need to make a histogram of only the unclustered distances - when run on GPU
vamb/cluster.py:        if self.cuda:
vamb/cluster.py:        # Currently, this function does not run on GPU. This means we must
vamb/cluster.py:        # If the issue is resolved, there can be large speedups on GPU
vamb/cluster.py:        if self.cuda:
vamb/cluster.py:                        distances, self.kept_mask, _DEFAULT_RADIUS, self.cuda
vamb/cluster.py:                    distances, self.kept_mask, threshold, self.cuda
vamb/cluster.py:        if self.cuda:
vamb/cluster.py:    tensor: _Tensor, kept_mask: _Tensor, threshold: float, cuda: bool
vamb/cluster.py:    # If it's on GPU, we remove the already clustered points at this step
vamb/cluster.py:    if cuda:
vamb/__main__.py:        "cuda",
vamb/__main__.py:            typeasserted(args.cuda, bool),
vamb/__main__.py:        cuda: bool,
vamb/__main__.py:        if cuda and not torch.cuda.is_available():
vamb/__main__.py:                "Cuda is not available on your PyTorch installation"
vamb/__main__.py:        self.cuda = cuda
vamb/__main__.py:        cuda=vamb_options.cuda,
vamb/__main__.py:        _cuda=vamb_options.cuda,
vamb/__main__.py:    cuda: bool,
vamb/__main__.py:    logger.info(f"\tUse CUDA for clustering: {cuda}")
vamb/__main__.py:        cuda=cuda,
vamb/__main__.py:        cuda=opt.common.general.cuda,
vamb/__main__.py:        opt.common.general.cuda,
vamb/__main__.py:        cuda=opt.common.general.cuda,
vamb/__main__.py:        opt.common.general.cuda,
vamb/__main__.py:    cuda: bool,
vamb/__main__.py:        cuda=cuda,
vamb/__main__.py:        cuda=opt.general.cuda,
vamb/__main__.py:            cuda=opt.common.general.cuda,
vamb/__main__.py:        cuda=opt.common.general.cuda,
vamb/__main__.py:        cuda=opt.common.general.cuda,
vamb/__main__.py:        cuda=opt.common.general.cuda,
vamb/__main__.py:        cuda=opt.common.general.cuda,
vamb/__main__.py:        cuda=opt.common.general.cuda,
vamb/__main__.py:        opt.common.general.cuda,
vamb/__main__.py:        "--cuda", help="use GPU to train & cluster [False]", action="store_true"
vamb/taxvamb_encode.py:    cuda=False,
vamb/taxvamb_encode.py:    _, _, _, _, batchsize, n_workers, cuda = _semisupervised_encode._make_dataset(
vamb/taxvamb_encode.py:        rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda
vamb/taxvamb_encode.py:        pin_memory=cuda,
vamb/taxvamb_encode.py:    cuda: bool = False,
vamb/taxvamb_encode.py:        cuda,
vamb/taxvamb_encode.py:        rpkm, tnf, lengths, batchsize=batchsize, destroy=destroy, cuda=cuda
vamb/taxvamb_encode.py:        pin_memory=cuda,
vamb/taxvamb_encode.py:    cuda=False,
vamb/taxvamb_encode.py:        pin_memory=cuda,
vamb/taxvamb_encode.py:        cuda: Use CUDA (GPU accelerated training) [False]
vamb/taxvamb_encode.py:        cuda: bool = False,
vamb/taxvamb_encode.py:            cuda=cuda,
vamb/taxvamb_encode.py:        if self.usecuda:
vamb/taxvamb_encode.py:            self.loss_fn = self.loss_fn.cuda()
vamb/taxvamb_encode.py:            self.pred_helper = self.pred_helper.cuda()
vamb/taxvamb_encode.py:        logger.info(f"\tCUDA: {self.usecuda}")
vamb/taxvamb_encode.py:        cuda: Use CUDA (GPU accelerated training) [False]
vamb/taxvamb_encode.py:        cuda: bool = False,
vamb/taxvamb_encode.py:            cuda=cuda,
vamb/taxvamb_encode.py:        if self.usecuda:
vamb/taxvamb_encode.py:            self.loss_fn = self.loss_fn.cuda()
vamb/taxvamb_encode.py:            self.pred_helper = self.pred_helper.cuda()
vamb/taxvamb_encode.py:        cuda: Use CUDA (GPU accelerated training) [False]
vamb/taxvamb_encode.py:        cuda: bool = False,
vamb/taxvamb_encode.py:        self.usecuda = cuda
vamb/taxvamb_encode.py:            cuda=cuda,
vamb/taxvamb_encode.py:            cuda=cuda,
vamb/taxvamb_encode.py:            cuda=cuda,
vamb/taxvamb_encode.py:    def load(cls, path, nodes, table_parent, cuda=False, evaluate=True):
vamb/taxvamb_encode.py:            cuda: If network should work on GPU [False]
vamb/taxvamb_encode.py:        # Forcably load to CPU even if model was saves as GPU model
vamb/taxvamb_encode.py:            cuda=cuda,
vamb/taxvamb_encode.py:        if cuda:
vamb/taxvamb_encode.py:            vae.VAEVamb.cuda()
vamb/taxvamb_encode.py:            vae.VAELabels.cuda()
vamb/taxvamb_encode.py:            vae.VAEJoint.cuda()
vamb/taxvamb_encode.py:        cuda: Use CUDA (GPU accelerated training) [False]
vamb/taxvamb_encode.py:        cuda: bool = False,
vamb/taxvamb_encode.py:        self.usecuda = cuda
vamb/taxvamb_encode.py:        if self.usecuda:
vamb/taxvamb_encode.py:            self.loss_fn = self.loss_fn.cuda()
vamb/taxvamb_encode.py:            self.pred_helper = self.pred_helper.cuda()
vamb/taxvamb_encode.py:        if cuda:
vamb/taxvamb_encode.py:            self.cuda()
vamb/taxvamb_encode.py:                # Move input to GPU if requested
vamb/taxvamb_encode.py:                if self.usecuda:
vamb/taxvamb_encode.py:                    depths = depths.cuda()
vamb/taxvamb_encode.py:                    tnf = tnf.cuda()
vamb/taxvamb_encode.py:                    abundances = abundances.cuda()
vamb/taxvamb_encode.py:                    weights = weights.cuda()
vamb/taxvamb_encode.py:                    if self.usecuda:
vamb/taxvamb_encode.py:        cls, path: Union[IO[bytes], str], cuda: bool = False, evaluate: bool = True
vamb/taxvamb_encode.py:            if self.usecuda:
vamb/taxvamb_encode.py:                depths_in = depths_in.cuda()
vamb/taxvamb_encode.py:                tnf_in = tnf_in.cuda()
vamb/taxvamb_encode.py:                abundances_in = abundances_in.cuda()
vamb/taxvamb_encode.py:                weights = weights.cuda()
vamb/taxvamb_encode.py:                labels_in = labels_in.cuda()
vamb/taxvamb_encode.py:        logger.info(f"\tCUDA: {self.usecuda}")
vamb/aamb_encode.py:        _cuda: bool,
vamb/aamb_encode.py:        torch.cuda.manual_seed(seed)
vamb/aamb_encode.py:        self.usecuda = _cuda
vamb/aamb_encode.py:        if _cuda:
vamb/aamb_encode.py:            self.cuda()
vamb/aamb_encode.py:        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
vamb/aamb_encode.py:        if self.usecuda:
vamb/aamb_encode.py:            sampled_z = sampled_z.cuda()
vamb/aamb_encode.py:        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
vamb/aamb_encode.py:        logger.info(f"\tCUDA: {self.usecuda}")
vamb/aamb_encode.py:        if self.usecuda:
vamb/aamb_encode.py:            adversarial_loss.cuda()
vamb/aamb_encode.py:                if self.usecuda:
vamb/aamb_encode.py:                    z_prior = torch.cuda.FloatTensor(nrows, self.ld).normal_()
vamb/aamb_encode.py:                    z_prior.cuda()
vamb/aamb_encode.py:                        torch.tensor([T], device="cuda"),
vamb/aamb_encode.py:                        torch.ones([nrows, self.y_len], device="cuda"),
vamb/aamb_encode.py:                    y_prior = y_prior.cuda()
vamb/aamb_encode.py:                if self.usecuda:
vamb/aamb_encode.py:                    depths_in = depths_in.cuda()
vamb/aamb_encode.py:                    tnfs_in = tnfs_in.cuda()
vamb/aamb_encode.py:        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
vamb/aamb_encode.py:                if self.usecuda:
vamb/aamb_encode.py:                    z_prior = torch.cuda.FloatTensor(nrows, self.ld).normal_()
vamb/aamb_encode.py:                    z_prior.cuda()
vamb/aamb_encode.py:                        torch.tensor([0.15], device="cuda"),
vamb/aamb_encode.py:                        torch.ones([nrows, self.y_len], device="cuda"),
vamb/aamb_encode.py:                    y_prior = y_prior.cuda()
vamb/aamb_encode.py:                if self.usecuda:
vamb/aamb_encode.py:                    depths_in = depths_in.cuda()
vamb/aamb_encode.py:                    tnfs_in = tnfs_in.cuda()
vamb/aamb_encode.py:                if self.usecuda:
test/test_semisupervised_encode.py:            cuda=False,
test/test_semisupervised_encode.py:            cuda=False,
test/test_semisupervised_encode.py:            cuda=False,
test/test_semisupervised_encode.py:            cuda=False,
test/test_semisupervised_encode.py:            cuda=False,
CHANGELOG.md:  However, GPU clustering may be significantly slower. (#198)

```
