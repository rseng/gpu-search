# https://github.com/adammoss/supernovae

```console
paper/sn.tex:For each network we perform 5 randomised runs over the training data to obtain the classifier metrics. We define the loss function as the categorical cross-entropy between the predictions and test data. The network weights were trained using back-propogation with the `Adam' updater~\cite{2014arXiv1412.6980K}. Mini-batches of size 10\footnote{If training with a GPU larger mini-batches are recommended to make use of the GPU cores.} were used throughout, and each model was trained over approximately 200 {\em epochs}, where each epoch is a full pass over the training data.
readme.md:After 200 epochs this should have an AUC of around 0.986, an accuracy of 94.8% and an F1 score of 0.64. The training loss should be just below the test loss. To run with a GPU (note here it is better to run with a larger batch size)
readme.md:THEANO_FLAGS=device=gpu,floatX=float32 python run.py -f test.ini

```
