# https://github.com/devanshkv/fetch

```console
README.md:To use fetch, you would first have to create candidates. Use [`your`](https://thepetabyteproject.github.io/your/) for this purpose, [this notebook](https://thepetabyteproject.github.io/your/ipynb/Candidate/) explains the whole process. Your also comes with a command line script [`your_candmaker.py`](https://thepetabyteproject.github.io/your/bin/your_candmaker/) which allows you to use CPU or single/multiple GPUs. 
requirements.txt:tensorflow[and-cuda]>=2.12
bin/predict.py:        "--gpu_id",
bin/predict.py:        help="GPU ID (use -1 for CPU)",
bin/predict.py:    if args.gpu_id >= 0:
bin/predict.py:        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
bin/predict.py:        os.environ["CUDA_VISIBLE_DEVICES"] = ""
bin/train.py:        "-g", "--gpu_id", help="GPU ID", type=int, required=False, default=0
bin/train.py:    if args.gpu_id:
bin/train.py:        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

```
