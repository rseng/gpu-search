# https://github.com/axgoujon/convex_ridge_regularizers

```console
denoising/validation_proximal_denoisers.py:    parser.add_argument('-d', '--device', default="cuda", type=str,
denoising/test_t-step_denoisers_ReLU.py:    parser.add_argument('-d', '--device', default="cuda", type=str,
denoising/test_t-step_denoisers.py:    parser.add_argument('-d', '--device', default="cuda", type=str,
denoising/test_proximal_denoisers.py:    parser.add_argument('-d', '--device', default="cuda", type=str,
models/utils.py:def load_model(name, device='cuda:0', epoch=None):
models/utils.py:    checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})
README.md:* (optional) CUDA
tutorial/README.md:* (optional) CUDA
training/train_loop.py:        torch.cuda.manual_seed(seed)
training/train_loop.py:    parser.add_argument('-d', '--device', default="cuda", type=str,
training/README.md:To launch the training on, e.g., GPU \#0:
training/README.md:~/convex_ridge_regularizers/training$ python train.py --device cuda:0
training/train.py:# CUDA_LAUNCH_BLOCKING=1
training/train.py:    torch.cuda.manual_seed(seed)
training/train.py:    parser.add_argument('-d', '--device', default="cuda", type=str,
inverse_problems/ct/test_acr.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/ct/test_pnp.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/ct/validation_tv.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/ct/test_crr.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/ct/validation_acr.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/ct/test_tv.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/ct/data/make_data_sets.py:device = "cuda"
inverse_problems/ct/validation_pnp.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/ct/validation_wcrr.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/ct/utils_ct/ct_forward_utils.py:    fwd_op_odl = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
inverse_problems/ct/utils_ct/ct_forward_utils.py:    # !!!! the ray transform for odl and astra_cuda is known to have scaling issues (related to the volume of the discretization cells)
inverse_problems/ct/utils_ct/ct_forward_utils.py:    #get_norm_HtH(fwd_op, adjoint_op, device = 'cuda')
inverse_problems/ct/utils_ct/torch_wrapper.py:        # TODO(kohr-h): use GPU memory directly when possible
inverse_problems/ct/utils_ct/torch_wrapper.py:            # TODO: implement directly for GPU data
inverse_problems/ct/test_wcrr.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/ct/validation_crr.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/utils_inverse_problems/batch_wrapper.py:    def __init__(self, sample_reconstruction_map, data_loader, n_hyperparameters=2, modality="ct", device='cuda:0', **kwargs):
inverse_problems/utils_inverse_problems/get_reconstruction_map.py:def get_reconstruction_map(method, modality, device="cuda:0", **opts):
inverse_problems/mri/test_pnp.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/mri/validation_tv.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/mri/test_crr.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/mri/test_tv.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/mri/data/baseline_reconstructions.py:device = 'cuda:0'
inverse_problems/mri/validation_pnp.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/mri/validation_wcrr.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/mri/test_wcrr.py:    parser.add_argument('--device', type=str, default="cuda:0")
inverse_problems/mri/validation_crr.py:    parser.add_argument('--device', type=str, default="cuda:0")
others/averaged_cnn/model_averaged/utils_averaged_cnn.py:def load_model(sigma=5, device="cuda:0"):
others/wcrr/model_wcrr/utils.py:def load_model(name, device='cuda:0', epoch=None):
others/wcrr/model_wcrr/utils.py:    checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})
others/tv/tv_prox.py:import pylops_gpu
others/tv/tv_prox.py:        self.Dop_0 = pylops_gpu.FirstDerivative(self.H*self.W, dims=(self.H, self.W), dir=0, device=device, togpu=(True, True))
others/tv/tv_prox.py:        self.Dop_1 = pylops_gpu.FirstDerivative(self.H*self.W, dims=(self.H, self.W), dir=1, device=device, togpu=(True, True))
others/acr/env_tomo.yml:  - cudatoolkit=9.2=0
others/acr/env_tomo.yml:  - pytorch=1.4.0=py3.5_cuda9.2.148_cudnn7.6.3_0

```
