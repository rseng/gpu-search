# https://github.com/Ayuei/DeBEIR

```console
README.md:* GPU hardware compatible with pytorch is encouraged
benchmark/README.md:We don't see a noticeable difference, which shows that during the encode stage that calls to the GPU model is not the
src/debeir/models/colbert.py:        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src/debeir/models/colbert.py:        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src/debeir/training/losses/contrastive.py:        device = (torch.device('cuda')
src/debeir/training/losses/contrastive.py:                  if features.is_cuda
src/debeir/training/train_sentence_encoder.py:                                      per_gpu_train_batch_size=train_batch_size,

```
