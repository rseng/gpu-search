# https://github.com/kkjawz/coref-ee

```console
modeling.py:            it is much faster if this is True, on the CPU or GPU, it is faster if
modeling.py:    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
setup_pretrained.sh:curl -O http://lsz-gpu-01.cs.washington.edu/resources/coref/char_vocab.english.txt
setup_pretrained.sh:curl -O http://lsz-gpu-01.cs.washington.edu/resources/coref/$ckpt_file
experiments.conf:two_local_gpus {
experiments.conf:  gpus = [0, 1, 2, 3]
experiments.conf:  cluster = ${two_local_gpus}
experiments.conf:  multi_gpu = false
worker.py:  util.set_gpus(cluster_config["gpus"][task_index])
extract_features.py:flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
extract_features.py:    # or GPU.
README.md:* It does not use GPUs by default. Instead, it looks for the `GPU` environment variable, which the code treats as shorthand for `CUDA_VISIBLE_DEVICES`.
requirements.txt:tensorflow-gpu==1.10.1
setup_all.sh:curl -O http://lsz-gpu-01.cs.washington.edu/resources/glove_50_300_2.txt
util.py:    # two_local_gpus {
util.py:    #   gpus = [0, 1, 2, 3]
util.py:    if "GPUS" not in os.environ:
util.py:        raise ValueError("Need to set GPU environment variable")
util.py:    gpus = list(map(int, os.environ["GPUS"].split(',')))
util.py:    workers = ['localhost:{}'.format(port) for port in range(2223, 2223 + len(gpus))]
util.py:                      'gpus': gpus}
util.py:    if "GPU" in os.environ:
util.py:        set_gpus(int(os.environ["GPU"]))
util.py:        set_gpus()
util.py:def set_gpus(*gpus):
util.py:    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
util.py:    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
ps.py:  util.set_gpus()
train_mgpu.sh:TRAIN_GPUS=$1
train_mgpu.sh:EVAL_GPU=$2
train_mgpu.sh:IFS=',' read -r -a TRAIN_GPUS_ARRAY <<< "$TRAIN_GPUS"
train_mgpu.sh:tmux send-keys -t "$NAME:ps" "GPUS=$TRAIN_GPUS python ps.py $NAME" Enter
train_mgpu.sh:for GPU in ${TRAIN_GPUS_ARRAY[@]}
train_mgpu.sh:	tmux send-keys -t "$NAME:worker $I" "GPUS=$TRAIN_GPUS TASK=$I python worker.py $NAME" Enter
train_mgpu.sh:tmux send-keys -t "$NAME:eval" "GPU=$EVAL_GPU python continuous_evaluate.py $NAME" Enter
prepare_bert_data.py:flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

```
