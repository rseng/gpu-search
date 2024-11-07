# https://github.com/OPUS-MaLab/opus_fold3

```console
dock_sidechain.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
fold_sidechain.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
dock_backbone.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
scripts/script_dock_sidechain.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
scripts/script_dock_backbone.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
scripts/script_fold_sidechain.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
scripts/script_fold_backbone.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
fold_backbone.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
refine_sidechain_em.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
refine_sidechain_em.py:gpus = tf.config.experimental.list_physical_devices('GPU')
refine_sidechain_em.py:for gpu in gpus:
refine_sidechain_em.py:        tf.config.experimental.set_memory_growth(gpu, True)

```
