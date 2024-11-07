# https://github.com/biomedia-mira/deepscm

```console
Readme.md:python -m deepscm.experiments.morphomnist.trainer -e SVIExperiment -m {IndependentVISEM, ConditionalDecoderVISEM, ConditionalVISEM} --data_dir /path/to/data --default_root_dir /path/to/checkpoints --decoder_type fixed_var {--gpus 0}
Readme.md:python -m deepscm.experiments.medical.trainer -e SVIExperiment -m ConditionalVISEM --default_root_dir /path/to/checkpoints --downsample 3 --decoder_type fixed_var --train_batch_size 256 {--gpus 0}
deepscm/experiments/medical/trainer.py:    if args.gpus is not None and isinstance(args.gpus, int):
deepscm/experiments/medical/trainer.py:        # Make sure that it only uses a single GPU..
deepscm/experiments/medical/trainer.py:        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
deepscm/experiments/medical/trainer.py:        args.gpus = 1
deepscm/experiments/medical/tester.py:    if args.gpus is not None and isinstance(args.gpus, int):
deepscm/experiments/medical/tester.py:        # Make sure that it only uses a single GPU..
deepscm/experiments/medical/tester.py:        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
deepscm/experiments/medical/tester.py:        args.gpus = 1
deepscm/experiments/medical/base_experiment.py:        self.torch_device = self.trainer.root_gpu if self.trainer.on_gpu else self.trainer.root_device
deepscm/experiments/medical/base_experiment.py:        if self.trainer.on_gpu:
deepscm/experiments/medical/ukbb/readme.md:python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --downsample 3 --decoder_type fixed_var --train_batch_size 256 --gpus 0
deepscm/experiments/medical/ukbb/readme.md:python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --downsample 3 --decoder_type fixed_var --train_batch_size 256 --gpus 0 --max_epochs 100
deepscm/experiments/medical/ukbb/readme.md:python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --downsample 3 --decoder_type independent_gaussian --train_batch_size 256 --gpus 0
deepscm/experiments/medical/ukbb/readme.md:python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --decoder_type fixed_var --gpus 0
deepscm/experiments/medical/ukbb/readme.md:python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --decoder_type independent_gaussian --gpus 0
deepscm/experiments/medical/ukbb/readme.md:python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --decoder_type fixed_var --latent_dim 256 --gpus 0
deepscm/experiments/medical/ukbb/readme.md:python experiments/medical/trainer.py -e SVIExperiment -m ConditionalVISEM --default_root_dir /vol/biomedic2/np716/logdir/gemini/ukbb/ --decoder_type independent_gaussian --latent_dim 256 --gpus 0
deepscm/experiments/morphomnist/trainer.py:    if args.gpus is not None and isinstance(args.gpus, int):
deepscm/experiments/morphomnist/trainer.py:        # Make sure that it only uses a single GPU..
deepscm/experiments/morphomnist/trainer.py:        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
deepscm/experiments/morphomnist/trainer.py:        args.gpus = 1
deepscm/experiments/morphomnist/tester.py:    if args.gpus is not None and isinstance(args.gpus, int):
deepscm/experiments/morphomnist/tester.py:        # Make sure that it only uses a single GPU..
deepscm/experiments/morphomnist/tester.py:        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
deepscm/experiments/morphomnist/tester.py:        args.gpus = 1
deepscm/experiments/morphomnist/base_experiment.py:        self.torch_device = self.trainer.root_gpu if self.trainer.on_gpu else self.trainer.root_device
deepscm/experiments/morphomnist/base_experiment.py:        if self.trainer.on_gpu:
deepscm/experiments/plotting/morphomnist/plotting_helper.py:os.environ['CUDA_VISIBLE_DEVICES'] = ''
deepscm/experiments/plotting/ukbb/plotting_helper.py:os.environ['CUDA_VISIBLE_DEVICES'] = ''

```
