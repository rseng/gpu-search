# https://github.com/changhoonhahn/haloflow

```console
bin/della.py:python script to deploy jobs on della-gpu
bin/della.py:def train_NDE_optuna(obs, nf_model='maf', hr=12, gpu=True, mig=True): 
bin/della.py:        ['', "#SBATCH --gres=gpu:1"][gpu], 
bin/della.py:    train_NDE_optuna('mags', nf_model='maf', hr=4, gpu=False, mig=False) 
bin/della.py:    train_NDE_optuna('mags_morph', nf_model='maf', hr=4, gpu=False, mig=False) 
bin/della.py:    train_NDE_optuna('mags_morph_satlum_all', nf_model='maf', hr=4, gpu=False, mig=False) 
bin/della.py:    train_NDE_optuna('mags_morph_satlum_all_rich_all', nf_model='maf', hr=4, gpu=False, mig=False) 
bin/della.py:    train_NDE_optuna('mags_morph_satlum_mrlim', nf_model='maf', hr=4, gpu=False, mig=False) 
bin/della.py:    train_NDE_optuna('mags_morph_satlum_mrliml_rich_mrlim', nf_model='maf', hr=4, gpu=False, mig=False) 
bin/nde.py:cuda = torch.cuda.is_available()
bin/nde.py:device = ("cuda:0" if cuda else "cpu")

```
