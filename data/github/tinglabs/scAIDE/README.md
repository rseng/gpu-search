# https://github.com/tinglabs/scAIDE

```console
baseline/script/DCA_/run.py:	"""Note: for 'Shekhar_mouse_retina', run with CPU to avoid GPU's memory error;
baseline/script/DCA_/run.py:		for other data in get_all_small_data_names(), run with GPU.
baseline/script/DCA_/run.py:	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
baseline/script/DCA_/run.py:	# d02 baseline2 32 small gpu 0
baseline/script/DCA_/run.py:	# d02 baseline 256 small gpu 1
baseline/script/scDeepCluster_/run_default.sh:GPU_USE=1
baseline/script/scDeepCluster_/run_default.sh:  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/10X_PBMC.h5 --n_clusters 8 --gpu $GPU_USE --layers $LAYERS
baseline/script/scDeepCluster_/run_default.sh:  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/mouse_bladder_cell.h5 --n_clusters 16 --gpu $GPU_USE --layers $LAYERS
baseline/script/scDeepCluster_/run_default.sh:  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/mouse_ES_cell.h5 --n_clusters 4 --gpu $GPU_USE --layers $LAYERS
baseline/script/scDeepCluster_/run_default.sh:  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/worm_neuron_cell.h5 --n_clusters 10 --gpu $GPU_USE --layers $LAYERS
baseline/script/scDeepCluster_/run_default.sh:  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/large_real_datasets/Shekhar_mouse_retina_raw_data.h5 --n_clusters 19 --gpu $GPU_USE --layers $LAYERS
baseline/script/scDeepCluster_/run_default.sh:  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/large_real_datasets/PBMC_68k.h5 --n_clusters 10 --gpu $GPU_USE --layers $LAYERS
baseline/script/scDeepCluster_/run_default.sh:  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/large_real_datasets/sc_brain.h5 --n_clusters 7 --pretrain_epochs 40 --gpu $GPU_USE --layers $LAYERS
baseline/script/scDeepCluster_/run_default.sh:#    python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/sim_sparsity/${DATA_NAME}.h5 --n_clusters 10 --gpu $GPU_USE --layers $LAYERS
baseline/script/scDeepCluster_/run_default.sh:#    python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/1M_neurons_samples/${DATA_NAME}.h5 --n_clusters 10 --gpu $GPU_USE --layers $LAYERS
baseline/script/scScope_/run.py:			latent_dim, T=T, batch_size=64, max_epoch=100, num_gpus=1,
baseline/script/scScope_/run.py:			X, latent_dim, T=T, batch_size=64, max_epoch=100, num_gpus=1,
baseline/script/scScope_/run.py:	parser.add_argument('--gpu', type=str, default='0')
baseline/script/scScope_/run.py:	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
baseline/script/scScope_/run.py:	# d04 baseline3 50 large gpu 0
baseline/script/scScope_/run.py:	# d04 baseline4 256 large gpu 1
baseline/script/scVI_/run.py:	parser.add_argument('--gpu', type=str, default='0')
baseline/script/scVI_/run.py:	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
baseline/script/scVI_/run.py:	# d04 baseline n_latent 10 gpu 2 large
baseline/script/scVI_/run.py:	# d04 baseline2 n_latent 256 gpu 3 middle
baseline/script/SAUCIE_/run.py:	saucie = SAUCIE(X.shape[1], lambda_c=lambda_c, lambda_d=lambda_d, layers=layers, limit_gpu_fraction=1.0)
baseline/script/SAUCIE_/run.py:	parser.add_argument('--gpu', type=str, default='0')
baseline/script/SAUCIE_/run.py:	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sc_cluster/src/dim_reduce_clt_eval.py:	# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sc_cluster/src/dim_reduce_clt_eval.py:	# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sc_cluster/src/dim_reduce_clt_eval.py:	# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

```
