# https://github.com/smsharma/mining-for-substructure-lens

```console
inference/trainer.py:    def __init__(self, model, run_on_gpu=True, double_precision=False):
inference/trainer.py:        self.run_on_gpu = run_on_gpu and torch.cuda.is_available()
inference/trainer.py:        self.device = torch.device("cuda" if self.run_on_gpu else "cpu")
inference/trainer.py:            "GPU" if self.run_on_gpu else "CPU",
inference/trainer.py:                pin_memory=self.run_on_gpu,
inference/trainer.py:                pin_memory=self.run_on_gpu,
inference/trainer.py:                pin_memory=self.run_on_gpu,
inference/trainer.py:    def __init__(self, model, run_on_gpu=True, double_precision=False):
inference/trainer.py:            model, run_on_gpu, double_precision
inference/estimator.py:        trainer = RatioTrainer(self.model, run_on_gpu=True)
inference/estimator.py:        run_on_gpu=True,
inference/estimator.py:                run_on_gpu,
inference/estimator.py:        run_on_gpu,
inference/estimator.py:        # CPU or GPU?
inference/estimator.py:        run_on_gpu = run_on_gpu and torch.cuda.is_available()
inference/estimator.py:        device = torch.device("cuda" if run_on_gpu else "cpu")
inference/estimator.py:            if run_on_gpu:
inference/estimator.py:                if run_on_gpu:
scripts/eval_sgd.sh:#SBATCH --gres=gpu:1
scripts/train_alpha.sh:#SBATCH --gres=gpu:1
scripts/eval_other.sh:#SBATCH --gres=gpu:1
scripts/train_alices.sh:#SBATCH --gres=gpu:1
scripts/train_sgd.sh:#SBATCH --gres=gpu:1
scripts/eval_alpha.sh:#SBATCH --gres=gpu:1
scripts/eval_calibration_ref.sh:#SBATCH --gres=gpu:1
scripts/train_lr.sh:#SBATCH --gres=gpu:1
scripts/eval_lr.sh:#SBATCH --gres=gpu:1
scripts/eval_alices.sh:#SBATCH --gres=gpu:1
scripts/train_carl.sh:#SBATCH --gres=gpu:1
scripts/simulate_train.sh:# #SBATCH --gres=gpu:1
scripts/train_other.sh:#SBATCH --gres=gpu:1
scripts/simulate_test.sh:# #SBATCH --gres=gpu:1
scripts/eval_calibration.sh:#SBATCH --gres=gpu:1
scripts/eval_carl.sh:#SBATCH --gres=gpu:1

```
