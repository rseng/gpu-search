# https://github.com/basp-group/SARA-dimensionality-reduction

```console
realdata_averaging_allvisibs_pdfb.m:run_pdfb_bpcon_par_sim_rescaled_gpu = 0; % flag
realdata_averaging_allvisibs_pdfb.m:nufft_param.use_fft_on_gpu = 0; % gpu FFT
realdata_allvisibs_pdfb.m:run_pdfb_bpcon_par_sim_rescaled_gpu = 0; % flag
realdata_allvisibs_pdfb.m:nufft_param.use_fft_on_gpu = 0; % gpu FFT
realdata_rgrid_pdfb.m:run_pdfb_bpcon_par_sim_rescaled_gpu = 0; % flag
realdata_rgrid_pdfb.m:param_nufft.use_fft_on_gpu = 0; % gpu FFT
realdata_rsing_pdfb.m:run_pdfb_bpcon_par_sim_rescaled_gpu = 0; % flag
realdata_rsing_pdfb.m:param_nufft.use_fft_on_gpu = 0; % gpu FFT
lib/so_fft2_adj_gpu.m:function x = so_fft2_adj_gpu(X, N, No, scale)
lib/so_fft2_adj_gpu.m:x = gather(ifft2(gpuArray(X)));
lib/op_nu_so_fft2.m:function [A, At] = op_nu_so_fft2(N, No, scale, use_fft_on_gpu)
lib/op_nu_so_fft2.m:if ~exist('use_fft_on_gpu', 'var')
lib/op_nu_so_fft2.m:    use_fft_on_gpu = 0;
lib/op_nu_so_fft2.m:if use_fft_on_gpu == 0
lib/op_nu_so_fft2.m:    A = @(x) so_fft2_gpu(x, No, scale);
lib/op_nu_so_fft2.m:    At = @(x) so_fft2_adj_gpu(x, N, No, scale);
lib/so_fft2_gpu.m:function X = so_fft2_gpu(x, No, scale)
lib/so_fft2_gpu.m:X = gather(fft2(gpuArray(x), No(1), No(2)));
lib/script_run_all_tests_serial.m:%% GPU fft
lib/script_run_all_tests_serial.m:if run_pdfb_bpcon_par_sim_rescaled_gpu
lib/script_run_all_tests_serial.m:        fprintf(' Running pdfb_bpcon_par_sim_rescaled_gpu\n');
lib/script_run_all_tests_serial.m:            = pdfb_bpcon_par_sim_rescaled_gpu(yT{i}, epsilonT{i}, epsilonTs{i}, epsilon{i}, A, At, T, W, Psi, Psit, param_pdfb);
lib/script_run_all_tests_serial.m:        fprintf(' pdfb_bpcon_par_sim_rescaled_gpu runtime: %ds\n\n', ceil(tend));

```
