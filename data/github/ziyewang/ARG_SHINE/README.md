# https://github.com/ziyewang/ARG_SHINE

```console
argshine_env.yaml:  - cudatoolkit=10.2.89=hfd86e86_1
argshine_env.yaml:  - pytorch=1.6.0=py3.7_cuda10.2.89_cudnn7.6.5_0
ARG_CNN/ARG_CNN_predict.py:torch.cuda.manual_seed_all(1)
ARG_CNN/ARG_CNN_predict.py:        y_hat = model(x.cuda())
ARG_CNN/ARG_CNN_predict.py:    model = model.cuda()

```
