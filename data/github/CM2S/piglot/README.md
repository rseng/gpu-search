# https://github.com/CM2S/piglot

```console
piglot/optimisers/botorch/bayes.py:            # Generate candidates: catch CUDA OOM errors and fall back to CPU
piglot/optimisers/botorch/bayes.py:            except torch.cuda.OutOfMemoryError:
piglot/optimisers/botorch/bayes.py:                warnings.warn('CUDA out of memory: falling back to CPU')

```
