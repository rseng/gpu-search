# https://github.com/Nic5472K/ScientificData2021_HealthGym

```console
Academic/mimic_III_sepsis/D005_Models.py:        Q_k = torch.zeros(x0.shape[0], self.HD).cuda()
Academic/mimic_III_sepsis/D005_Models.py:        S_k = torch.zeros(x0.shape[0], self.HD).cuda()
Academic/mimic_III_sepsis/B002_Wgan_Train.py:torch.cuda.manual_seed(     seed)
Academic/mimic_III_sepsis/D004_WGAN_GP.py:        self.correlation_real = correlation_real.cuda()
Academic/mimic_III_sepsis/D004_WGAN_GP.py:        if torch.cuda.is_available():
Academic/mimic_III_sepsis/D004_WGAN_GP.py:            self.CUDA = True
Academic/mimic_III_sepsis/D004_WGAN_GP.py:            self.CUDA = False
Academic/mimic_III_sepsis/D004_WGAN_GP.py:        if self.CUDA:
Academic/mimic_III_sepsis/D004_WGAN_GP.py:            self.G = self.G.cuda()
Academic/mimic_III_sepsis/D004_WGAN_GP.py:            self.D = self.D.cuda()
Academic/mimic_III_sepsis/D004_WGAN_GP.py:        z = torch.rand((num_samples, seq_len, self.ID)).cuda()
Academic/mimic_III_sepsis/D004_WGAN_GP.py:        alpha = torch.rand((self.batch_size, 1, 1)).cuda()
Academic/mimic_III_sepsis/D004_WGAN_GP.py:            grad_outputs=torch.ones_like(prob_interpolated).cuda(),
Academic/mimic_III_sepsis/D004_WGAN_GP.py:                    data_real = data_real.cuda()
Academic/mimic_III_hypotension/synthetic_data_generation/lib/utils.py:    if torch.cuda.is_available():
Academic/mimic_III_hypotension/synthetic_data_generation/lib/utils.py:        device = torch.device("cuda:0")
Demo/A001_Others/B003_WganGp.py:        if torch.cuda.is_available():
Demo/A001_Others/B003_WganGp.py:            self.CUDA = True
Demo/A001_Others/B003_WganGp.py:            self.CUDA = False
Demo/A001_Others/B003_WganGp.py:        if self.CUDA:
Demo/A001_Others/B003_WganGp.py:            self.G = self.G.cuda()
Demo/A001_Others/B003_WganGp.py:            self.D = self.D.cuda()
Demo/A001_Others/B003_WganGp.py:            self.correlation_real = self.correlation_real.cuda()
Demo/A001_Others/B003_WganGp.py:        z = torch.rand((num_samples, seq_len, self.ID)).cuda()
Demo/A001_Others/B003_WganGp.py:        alpha = torch.rand((self.batch_size, 1, 1)).cuda()
Demo/A001_Others/B003_WganGp.py:            grad_outputs=torch.ones_like(prob_interpolated).cuda(),
Demo/A001_Others/B003_WganGp.py:                    data_real = data_real.cuda()
Demo/A001_Others/B003zC001_Models.py:        Q_k = torch.zeros(x0.shape[0], self.HD).cuda()
Demo/A001_Others/B003zC001_Models.py:        S_k = torch.zeros(x0.shape[0], self.HD).cuda()
Demo/README.md: 	- This will take some time and we presume you are using CUDA.
Demo/A003_Main.py:torch.cuda.manual_seed(     seed)

```
