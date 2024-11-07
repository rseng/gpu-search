# https://github.com/simon-prunet/defringe

```console
randlin.py:  GPU = True
randlin.py:  print("cupy not installed. This probably means there is no GPU on this host.")
randlin.py:  GPU = False
randlin.py:def gpu_random_svd(X_gpu,k,s,q=0):
randlin.py:  Computes approximate SVD of X_gpu (which is of size m x n)
randlin.py:  if (GPU and not isinstance(X_gpu,cp.ndarray)):
randlin.py:  n = X_gpu.shape[1]
randlin.py:  O_gpu = cp.random.uniform(low=-1.0,high=1.0,size=(n,l),dtype=X_gpu.dtype)
randlin.py:  Y_gpu = cp.dot(X_gpu,O_gpu)
randlin.py:  Q_gpu, R_gpu = cp.linalg.qr(Y_gpu,'reduced')
randlin.py:    Yt_gpu = cp.dot(X_gpu.T,Q_gpu)
randlin.py:    Qt_gpu, Rt_gpu = cp.linalg.qr(Yt_gpu,'reduced')
randlin.py:    Y_gpu = cp.dot(X_gpu,Qt_gpu)
randlin.py:    Q_gpu, R_gpu = cp.linalg.qr(Y_gpu,'reduced')
randlin.py:  B_gpu = cp.dot(Q_gpu.T,X_gpu)
randlin.py:  M_gpu, D_gpu, VT_gpu = cp.linalg.svd(B_gpu,full_matrices=False)
randlin.py:  U_gpu = cp.dot(Q_gpu,M_gpu)
randlin.py:  return (U_gpu,D_gpu,VT_gpu)
images.py:  GPU = True
images.py:  print("cupy not installed. This probably means there is no GPU on this host.")
images.py:  GPU = False
README.md:CCD infrared image defringing code in python, using cupy for GPU acceleration. Based on matrix completion of noisy matrix with low-rank regularization via nuclear norm. See https://arxiv.org/abs/2109.02562 for details.
algorithms.py:  GPU = True
algorithms.py:  print("cupy not installed. This probably means there is no GPU on this host.")
algorithms.py:  GPU = False
algorithms.py:    if (GPU):
algorithms.py:        U,D,VT = randlin.gpu_random_svd(Z,*random)
algorithms.py:      U,D,VT = randlin.gpu_random_svd(X,*random)
algorithms.py:      U,D,VT=randlin.gpu_random_svd(X,*random)
algorithms.py:      U,D,VT = randlin.gpu_random_svd(Z,*random)

```
