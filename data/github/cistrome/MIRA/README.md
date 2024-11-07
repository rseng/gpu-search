# https://github.com/cistrome/MIRA

```console
README.rst:Installing with GPU support
README.rst:Training on a GPU reduces the training time of MIRA topic models.
README.rst:To install MIRA with PyTorch compiled with GPU support, first install MIRA, as above. Then, follow instructions 
docs/source/getting_started.rst:* (optional) CUDA-enabled GPU
docs/source/getting_started.rst:Installing with GPU support
docs/source/getting_started.rst:Training on a GPU reduces the training time of MIRA topic models.
docs/source/getting_started.rst:To install MIRA with PyTorch compiled with GPU support, first install MIRA, as above. Then, follow instructions 
docs/source/_templates/atac_topic_model.rst:    :members: get_learning_rate_bounds, set_learning_rates, trim_learning_rate_bounds, plot_learning_rate_bounds, instantiate_model, fit, predict, get_hierarchical_umap_features, get_umap_features, score, impute, save, to_gpu, to_cpu, set_device, load, rank_peaks, get_enriched_TFs, get_motif_scores, get_enrichments, plot_compare_topic_enrichments
docs/source/_templates/topic_model.rst:    :members: get_learning_rate_bounds, set_learning_rates, trim_learning_rate_bounds, plot_learning_rate_bounds, instantiate_model, fit, predict, get_hierarchical_umap_features, get_umap_features, score, impute, save, to_gpu, to_cpu, set_device, load, rank_genes, rank_modules, get_top_genes, post_topic, post_topics, fetch_topic_enrichments, fetch_enrichments, get_enrichments, plot_enrichments
docs/source/topicmodeling/mira.topics.AccessibilityTopicModel.rst:    :members: get_learning_rate_bounds, set_learning_rates, trim_learning_rate_bounds, plot_learning_rate_bounds, instantiate_model, fit, predict, get_hierarchical_umap_features, get_umap_features, score, impute, save, to_gpu, to_cpu, set_device, load, rank_peaks, get_enriched_TFs, get_motif_scores, get_enrichments, plot_compare_topic_enrichments
docs/source/topicmodeling/mira.topics.ExpressionTopicModel.rst:    :members: get_learning_rate_bounds, set_learning_rates, trim_learning_rate_bounds, plot_learning_rate_bounds, instantiate_model, fit, predict, get_hierarchical_umap_features, get_umap_features, score, impute, save, to_gpu, to_cpu, set_device, load, rank_genes, rank_modules, get_top_genes, post_topic, post_topics, fetch_topic_enrichments, fetch_enrichments, get_enrichments, plot_enrichments
mira/topic_model/hyperparameter_optim/trainer.py:        if not torch.cuda.is_available():
mira/topic_model/hyperparameter_optim/trainer.py:            logger.warn('GPU is not available, will not speed up training.')
mira/topic_model/model_factory.py:from torch.cuda import is_available as gpu_available
mira/topic_model/model_factory.py:    use_cuda : boolean, default=True
mira/topic_model/model_factory.py:        Try using CUDA GPU speedup while training.
mira/topic_model/model_factory.py:        However, this model is pretty much impossible to train on CPU. If instantiated without GPU,
mira/topic_model/model_factory.py:        not gpu_available() and not instance.atac_encoder == 'light':
mira/topic_model/model_factory.py:        logger.error('If a GPU is unavailable, one cannot use the "skipDAN" or "DAN" encoders for the ATAC model since training will be impossibly slow.'
mira/topic_model/model_factory.py:                     'Use a GPU, or switch the "atac_encoder" option to "light", which does not require a GPU.'
mira/topic_model/CODAL/covariate_model.py:            use_cuda = True,
mira/topic_model/CODAL/covariate_model.py:        self.use_cuda = use_cuda
mira/topic_model/CODAL/covariate_model.py:    def _get_weights(self, on_gpu = True, inference_mode = False,*,
mira/topic_model/CODAL/covariate_model.py:            on_gpu=on_gpu, 
mira/topic_model/base.py:            use_cuda = True,
mira/topic_model/base.py:        self.use_cuda = use_cuda
mira/topic_model/base.py:    def _get_weights(self, on_gpu = True, inference_mode = False,*,
mira/topic_model/base.py:        torch.cuda.empty_cache()
mira/topic_model/base.py:        assert(isinstance(self.use_cuda, bool))
mira/topic_model/base.py:        use_cuda = torch.cuda.is_available() and self.use_cuda and on_gpu
mira/topic_model/base.py:        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
mira/topic_model/base.py:        if not use_cuda:
mira/topic_model/base.py:                logger.warn('Cuda unavailable. Will not use GPU speedup while training.')
mira/topic_model/base.py:            on_gpu=True, inference_mode=False,
mira/topic_model/base.py:        self._get_weights(on_gpu = False, inference_mode = True,
mira/topic_model/base.py:    def to_gpu(self):
mira/topic_model/base.py:        Move topic model to GPU device "cuda:0", if available.
mira/topic_model/base.py:        self.set_device('cuda:0')
mira/rp_model/optim.py:                if torch.cuda.is_available():
mira/rp_model/optim.py:                    F_prev = torch.tensor(np.nan, dtype=dtype).cuda()
mira/rp_model/optim.py:                if(torch.cuda.is_available()):
mira/rp_model/optim.py:                    F_b = torch.tensor(np.nan, dtype=dtype).cuda()
mira/rp_model/optim.py:                    g_b = torch.tensor(np.nan, dtype=dtype).cuda()
mira/rp_model/optim.py:                        if torch.cuda.is_available():
mira/rp_model/optim.py:                            g_b = torch.tensor(np.nan, dtype=dtype).cuda()
miscdocs/github_install_tutorial.md:To enter the environment. For analysis, I recommend adding jupyter and scanpy. If you will be working with a GPU, also install the requesite CUDA toolkit:
miscdocs/github_install_tutorial.md:(kladi) $ conda install cudatoolkit==<version>

```
