# https://github.com/vortex-exoplanet/VIP

```console
README.rst:- VIP offers the possibility of computing SVDs on GPU by using ``CuPy`` (starting from version 0.8.0) or ``PyTorch`` (from version 0.9.2). These remain as optional requirements, to be installed by the user, as well as a proper CUDA environment (and a decent GPU card).
docs/source/Installation-and-dependencies.rst:- VIP offers the possibility of computing SVDs on GPU by using ``CuPy`` (starting from version 0.8.0) or ``PyTorch`` (from version 0.9.2). These remain as optional requirements, to be installed by the user, as well as a proper CUDA environment (and a decent GPU card).
vip_hci/greedy/ipca_fullfr.py:        ``cupy``: uses the Cupy library for GPU computation of the SVD as in
vip_hci/greedy/ipca_fullfr.py:        but on GPU (through Cupy).
vip_hci/greedy/ipca_fullfr.py:        where all the computations are done on a GPU (through Cupy). `
vip_hci/greedy/ipca_fullfr.py:        `pytorch``: uses the Pytorch library for GPU computation of the SVD.
vip_hci/greedy/ipca_fullfr.py:        option but on GPU (through Pytorch).
vip_hci/greedy/ipca_fullfr.py:        where all the linear algebra computations are done on a GPU
vip_hci/greedy/inmf_fullfr.py:        ``cupy``: uses the Cupy library for GPU computation of the SVD as in
vip_hci/greedy/inmf_fullfr.py:        but on GPU (through Cupy).
vip_hci/greedy/inmf_fullfr.py:        where all the computations are done on a GPU (through Cupy). `
vip_hci/greedy/inmf_fullfr.py:        `pytorch``: uses the Pytorch library for GPU computation of the SVD.
vip_hci/greedy/inmf_fullfr.py:        option but on GPU (through Pytorch).
vip_hci/greedy/inmf_fullfr.py:        where all the linear algebra computations are done on a GPU
vip_hci/config/paramenum.py:    * ``CUPY``: uses the Cupy library for GPU computation of the SVD as in
vip_hci/config/paramenum.py:    but on GPU (through Cupy).
vip_hci/config/paramenum.py:    where all the computations are done on a GPU (through Cupy). `
vip_hci/config/paramenum.py:    * ``PYTORCH``: uses the Pytorch library for GPU computation of the SVD.
vip_hci/config/paramenum.py:    option but on GPU (through Pytorch).
vip_hci/config/paramenum.py:    where all the linear algebra computations are done on a GPU
vip_hci/psfsub/svd.py:    msg = "Cupy not found. Do you have a GPU? Consider setting up a CUDA "
vip_hci/psfsub/svd.py:    msg = "Pytorch not found. Do you have a GPU? Consider setting up a CUDA "
vip_hci/psfsub/svd.py:        * ``cupy``: uses the Cupy library for GPU computation of the SVD as in
vip_hci/psfsub/svd.py:          but on GPU (through Cupy).
vip_hci/psfsub/svd.py:          where all the computations are done on a GPU (through Cupy). `
vip_hci/psfsub/svd.py:        * ``pytorch``: uses the Pytorch library for GPU computation of the SVD.
vip_hci/psfsub/svd.py:          option but on GPU (through Pytorch).
vip_hci/psfsub/svd.py:          where all the linear algebra computations are done on a GPU
vip_hci/psfsub/svd.py:    """ Wrapper for different SVD libraries (CPU and GPU).
vip_hci/psfsub/svd.py:        ``cupy``: uses the Cupy library for GPU computation of the SVD as in
vip_hci/psfsub/svd.py:        but on GPU (through Cupy).
vip_hci/psfsub/svd.py:        where all the computations are done on a GPU (through Cupy). `
vip_hci/psfsub/svd.py:        `pytorch``: uses the Pytorch library for GPU computation of the SVD.
vip_hci/psfsub/svd.py:        option but on GPU (through Pytorch).
vip_hci/psfsub/svd.py:        where all the linear algebra computations are done on a GPU
vip_hci/psfsub/svd.py:        If True (by default) the arrays computed in GPU are transferred from
vip_hci/psfsub/svd.py:        a_gpu = cupy.array(matrix)
vip_hci/psfsub/svd.py:        a_gpu = cupy.asarray(a_gpu)  # move the data to the current device
vip_hci/psfsub/svd.py:        u_gpu, s_gpu, vh_gpu = cupy.linalg.svd(a_gpu, full_matrices=True,
vip_hci/psfsub/svd.py:        V = vh_gpu[:ncomp]
vip_hci/psfsub/svd.py:            S = s_gpu[:ncomp]
vip_hci/psfsub/svd.py:            U = u_gpu[:, :ncomp]
vip_hci/psfsub/svd.py:            print('Done SVD/PCA with cupy (GPU)')
vip_hci/psfsub/svd.py:        U, S, V = randomized_svd_gpu(matrix, ncomp, n_iter=2, lib='cupy')
vip_hci/psfsub/svd.py:            print('Done randomized SVD/PCA with cupy (GPU)')
vip_hci/psfsub/svd.py:        a_gpu = cupy.array(matrix)
vip_hci/psfsub/svd.py:        a_gpu = cupy.asarray(a_gpu)     # move the data to the current device
vip_hci/psfsub/svd.py:        C = cupy.dot(a_gpu, a_gpu.T)    # covariance matrix
vip_hci/psfsub/svd.py:        pc = cupy.dot(EV.T, a_gpu)      # using a compact trick when cov is MM'
vip_hci/psfsub/svd.py:            print('Done PCA with cupy eigh function (GPU)')
vip_hci/psfsub/svd.py:        a_gpu = torch.Tensor.cuda(torch.from_numpy(matrix.astype('float32').T))
vip_hci/psfsub/svd.py:        u_gpu, s_gpu, vh_gpu = torch.svd(a_gpu)
vip_hci/psfsub/svd.py:        V = vh_gpu[:ncomp]
vip_hci/psfsub/svd.py:        S = s_gpu[:ncomp]
vip_hci/psfsub/svd.py:        U = torch.transpose(u_gpu, 0, 1)[:ncomp]
vip_hci/psfsub/svd.py:            print('Done SVD/PCA with pytorch (GPU)')
vip_hci/psfsub/svd.py:        a_gpu = torch.Tensor.cuda(torch.from_numpy(matrix.astype('float32')))
vip_hci/psfsub/svd.py:        C = torch.mm(a_gpu, torch.transpose(a_gpu, 0, 1))
vip_hci/psfsub/svd.py:        V = torch.mm(torch.transpose(EV, 0, 1), a_gpu)
vip_hci/psfsub/svd.py:        U, S, V = randomized_svd_gpu(matrix, ncomp, n_iter=2, lib='pytorch')
vip_hci/psfsub/svd.py:            print('Done randomized SVD/PCA with randomized pytorch (GPU)')
vip_hci/psfsub/svd.py:def randomized_svd_gpu(M, n_components, n_oversamples=10, n_iter='auto',
vip_hci/psfsub/svd.py:    """Compute a truncated randomized SVD on GPU - adapted from Sklearn.
vip_hci/psfsub/svd.py:        Chooses the GPU library to be used.
vip_hci/psfsub/svd.py:        M_gpu = torch.Tensor.cuda(torch.from_numpy(M.astype('float32')))
vip_hci/psfsub/svd.py:        Q = torch.cuda.FloatTensor(M_gpu.shape[1], n_random).normal_()
vip_hci/psfsub/svd.py:            Q = torch.mm(M_gpu, Q)
vip_hci/psfsub/svd.py:            Q = torch.mm(torch.transpose(M_gpu, 0, 1), Q)
vip_hci/psfsub/svd.py:        Q, _ = torch.qr(torch.mm(M_gpu, Q))
vip_hci/psfsub/svd.py:        B = torch.mm(torch.transpose(Q, 0, 1), M_gpu)
vip_hci/psfsub/utils_pca.py:        * ``cupy``: uses the Cupy library for GPU computation of the SVD as in
vip_hci/psfsub/utils_pca.py:          but on GPU (through Cupy).
vip_hci/psfsub/utils_pca.py:          where all the computations are done on a GPU (through Cupy). `
vip_hci/psfsub/utils_pca.py:        * ``pytorch``: uses the Pytorch library for GPU computation of the SVD.
vip_hci/psfsub/utils_pca.py:          option but on GPU (through Pytorch).
vip_hci/psfsub/utils_pca.py:          where all the linear algebra computations are done on a GPU
vip_hci/var/iuwt.py:                       core_count=1, store_on_gpu=False, smoothed_array=None):
vip_hci/var/iuwt.py:    mode                (default='ser')     Implementation of the IUWT to be used - 'ser', 'mp' or 'gpu'.
vip_hci/var/iuwt.py:    store_on_gpu        (default=False):    Boolean specifier for whether the decomposition is stored on the gpu or not.

```
