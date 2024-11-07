# https://github.com/fzi-forschungszentrum-informatik/TSInterpret

```console
ClassificationModels/CNN_T.py:    device = torch.device( "cpu")#"cuda:0" if torch.cuda.is_available() else
ClassificationModels/CNN_T.py:    device = torch.device("cpu")#"cuda:0" if torch.cuda.is_available() else "cpu")
ClassificationModels/CNN.py:        if not tf.test.is_gpu_available:
ClassificationModels/LSTM_T.py:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ClassificationModels/ResNet.py:        if not tf.test.is_gpu_available:

```
