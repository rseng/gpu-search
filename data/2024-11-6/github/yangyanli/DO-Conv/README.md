# https://github.com/yangyanli/DO-Conv

```console
sample_pt_with_fusion.py:print(torch.cuda.is_available())
sample_pt_with_fusion.py:    "cuda") if torch.cuda.is_available() else torch.device("cpu")
sample_pt.py:print(torch.cuda.is_available())
sample_pt.py:    "cuda") if torch.cuda.is_available() else torch.device("cpu")
README.md:In thie repo, we provide reference implementation of DO-Conv in <a href="https://www.tensorflow.org/" target="_blank">Tensorflow</a> (tensorflow-gpu==2.2.0), <a href="https://pytorch.org/" target="_blank">PyTorch</a> (pytorch==1.4.0, torchvision==0.5.0) and <a href="https://gluon-cv.mxnet.io/contents.html" target="_blank">GluonCV</a> (mxnet-cu100==1.5.1.post0, gluoncv==0.6.0), as replacement to <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D" target="_blank">tf.keras.layers.Conv2D</a>, <a href="https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html" target="_blank">torch.nn.Conv2d</a> and <a href="https://beta.mxnet.io/api/gluon/_autogen/mxnet.gluon.nn.Conv2D.html" target="_blank">mxnet.gluon.nn.Conv2D</a>, respectively. Please see the code for more details.
README.md:## Example Usage: Tensorflow (tensorflow-gpu==2.2.0)
sample_gluoncv.py:# set the context on GPU is available otherwise CPU
sample_gluoncv.py:ctx = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]

```
