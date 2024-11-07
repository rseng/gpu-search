# https://github.com/Filter-Bubble/e2e-Dutch

```console
test/test_coref_model.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
e2edutch/cfg/models.conf:two_local_gpus {
e2edutch/cfg/models.conf:  gpus = [0, 1]
e2edutch/cfg/models.conf:  cluster = ${two_local_gpus}
e2edutch/cfg/test.conf:two_local_gpus {
e2edutch/cfg/test.conf:  gpus = [0, 1]
e2edutch/stanza.py:    def __init__(self, config, pipeline, use_gpu):
e2edutch/stanza.py:        # Make e2edutch follow Stanza's GPU settings:
e2edutch/stanza.py:        # set the environment value for GPU, so that initialize_from_env picks it up.
e2edutch/stanza.py:        # if use_gpu:
e2edutch/stanza.py:        #    os.environ['GPU'] = ' '.join(tf.config.experimental.list_physical_devices('GPU'))
e2edutch/stanza.py:        #    if 'GPU' in os.environ['GPU'] :
e2edutch/stanza.py:        #        os.environ.pop('GPU')
e2edutch/util.py:    Configure Tensorflow to use a gpu or cpu based on the environment values of GPU.
e2edutch/util.py:    if "GPU" in os.environ:
e2edutch/util.py:        set_gpus(int(os.environ["GPU"]))
e2edutch/util.py:        set_gpus()
e2edutch/util.py:def set_gpus(*gpus):
e2edutch/util.py:    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
e2edutch/util.py:    logger.info("Setting CUDA_VISIBLE_DEVICES to: {}".format(
e2edutch/util.py:        os.environ["CUDA_VISIBLE_DEVICES"]))
e2edutch/util.py:    for gpu in tf.config.experimental.list_physical_devices('GPU'):
e2edutch/util.py:        tf.config.experimental.set_memory_growth(gpu, True)
README.md:- If you want to enable the use of a GPU, set the environment variable:
README.md:export GPU=0

```
