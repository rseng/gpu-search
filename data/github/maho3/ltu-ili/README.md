# https://github.com/maho3/ltu-ili

```console
tests/test_sbi.py:device = 'cuda' if torch.cuda.is_available() else 'cpu'
tests/test_lampe.py:device = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG.md:Lastly, the **train_args** are used to configure the training optimizer, the early stopping criterion, and the number of rounds of inference (for Sequential models). All engines use the Adam optimizer. Lastly, **device** specifies whether to use Pytorch's `cpu` or `cuda` backend, and **out_dir** specifies where to save your models after they are done training.
CONFIG.md: * Newly added `sbi` training engines, such as `MNLE`, don't yet work in our framework. Also, the `nsf` and `made` architectures unfortunately don't work if you're using `cuda` for GPU acceleration (yet!).
paper/fig9_dust/sbi_example.py:    (N_samps,), x=torch.Tensor(x_o).to('cuda'))
paper/fig9_dust/sbi_example.py:        np.median(np.array(posterior_samples), axis=0), device='cuda')
paper/fig9_dust/sbi_example.py:    [ax.plot(bins, np.log10(generate_hmf(torch.tensor(posterior_samples[i], device='cuda'))),
paper/fig9_dust/sbi_example_gamma.py:# , device='cuda')
paper/fig6_quijote/quijote_sbi_MAF_SNLE_9_val.yaml:device : 'cuda'
paper/fig6_quijote/quijote_sbi_MAF_SNLE_9_val.yaml:        device : 'cuda'
paper/fig6_quijote/quijote_sbi_MAF_SNLE_9_val.yaml:        device : 'cuda'
paper/fig6_quijote/quijote_sbi_MAF_SNLE_9_infer.yaml:    device: 'cuda'
paper/fig6_quijote/quijote_sbi_MAF_SNLE_9_infer.yaml:device: 'cuda'
paper/fig5_xray/inf_npe.yaml:device: 'cuda'
ili/dataloaders/loaders.py:        # Get device returns -1 for cpu, integers for CUDA tensors

```
