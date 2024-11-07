# https://github.com/LeonSong1995/gsMap

```console
src/gsMap/find_latent_representation.py:    if torch.cuda.is_available():
src/gsMap/find_latent_representation.py:        logger.info('Using GPU for computations.')
src/gsMap/find_latent_representation.py:        torch.cuda.manual_seed(seed_value)
src/gsMap/find_latent_representation.py:        torch.cuda.manual_seed_all(seed_value)
src/gsMap/GNN/train.py:        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src/gsMap/utils/make_annotations.py:    pool = cp.cuda.MemoryPool(cp.cuda.malloc_async)
src/gsMap/utils/make_annotations.py:    cp.cuda.set_allocator(pool.malloc)
src/gsMap/utils/make_annotations.py:    logger.warning('Cupy not found, will not use GPU to compute LD score')
src/gsMap/utils/make_annotations.py:    use_gpu: bool = False
src/gsMap/utils/make_annotations.py:        self.use_gpu = make_annotation_config.use_gpu
src/gsMap/utils/make_annotations.py:        if self.use_gpu:
src/gsMap/utils/make_annotations.py:            logger.debug('Using GPU to compute LD score')
src/gsMap/utils/make_annotations.py:    parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU to compute LD score')

```
