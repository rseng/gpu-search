# https://github.com/Guo-Jian-Wang/colfi

```console
README.rst:* `CUDA <https://developer.nvidia.com/cuda-downloads>`_ (optional, but suggested)
docs/source/index.rst:As a general method of parameter estimation, CoLFI can be used for research in many scientific fields. The code colfi is available for free from `GitHub <https://github.com/Guo-Jian-Wang/colfi>`_. It can be executed on GPUs or CPUs.
docs/source/installation.rst:* `CUDA <https://developer.nvidia.com/cuda-downloads>`_ (optional, but suggested)
examples/pantheon/train_pantheon_fwCDM_nde_steps.py:os.environ["CUDA_VISIBLE_DEVICES"] = "0"
examples/pantheon/_train_pantheon_fwCDM_nde.py:os.environ["CUDA_VISIBLE_DEVICES"] = "0"
colfi/models_mdn.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_mdn.py:                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_mdn.py:        if self.use_multiGPU:
colfi/models_mdn.py:            #     noise_obj = ds.AddGaussianNoise(xx,params=yy,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU)
colfi/models_mdn.py:            noise_obj = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU)
colfi/models_mdn.py:                noise_obj_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU)
colfi/models_mdn.py:        if self.use_multiGPU:
colfi/models_mdn.py:            noise_obj = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU)
colfi/models_mdn.py:                noise_obj_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU)
colfi/models_mdn.py:        if self.use_multiGPU:
colfi/models_mdn.py:    def _predict(self, obs, chain_leng=10000, use_GPU=False, in_type='torch'):
colfi/models_mdn.py:        use_GPU : bool
colfi/models_mdn.py:            If True, calculate using GPU, otherwise, calculate using CPU.
colfi/models_mdn.py:        if use_GPU:
colfi/models_mdn.py:            self.net = self.net.cuda()
colfi/models_mdn.py:                obs = dp.numpy2cuda(obs)
colfi/models_mdn.py:                obs = dp.torch2cuda(obs)
colfi/models_mdn.py:        if use_GPU:
colfi/models_mdn.py:            pred = dp.cuda2numpy(pred.data)
colfi/models_mdn.py:    #change the branch net & trunck net (contain training) to use multiple GPUs ???
colfi/models_mdn.py:            _inputs, _target = ds.AddGaussianNoise(self.inputs,params=self.target,obs_errors=self.error,cholesky_factor=self.cholesky_f,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_mdn.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_mdn.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_mdn.py:                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_mdn.py:        if self.use_multiGPU:
colfi/models_mdn.py:    def _predict(self, obs, chain_leng=10000, use_GPU=False, in_type='torch'):
colfi/models_mdn.py:        use_GPU : bool
colfi/models_mdn.py:            If True, calculate using GPU, otherwise, calculate using CPU.
colfi/models_mdn.py:        if use_GPU:
colfi/models_mdn.py:            self.net = self.net.cuda()
colfi/models_mdn.py:                obs = [dp.numpy2cuda(obs[i]) for i in range(len(obs))]
colfi/models_mdn.py:                obs = [dp.torch2cuda(obs[i]) for i in range(len(obs))]
colfi/models_mdn.py:        if use_GPU:
colfi/models_mdn.py:            pred = dp.cuda2numpy(pred.data)
colfi/models_ann.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_ann.py:                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_ann.py:        if self.use_multiGPU:
colfi/models_ann.py:    def _predict(self, inputs, use_GPU=False, in_type='torch'):
colfi/models_ann.py:        use_GPU : bool, optional
colfi/models_ann.py:            If True, calculate using GPU, otherwise, calculate using CPU.
colfi/models_ann.py:        if use_GPU:
colfi/models_ann.py:            self.net = self.net.cuda()
colfi/models_ann.py:                inputs = dp.numpy2cuda(inputs)
colfi/models_ann.py:                inputs = dp.torch2cuda(inputs)
colfi/models_ann.py:        if use_GPU:
colfi/models_ann.py:            pred = dp.cuda2numpy(pred.data)
colfi/models_ann.py:    #update this to ensure simulate and predict on GPU when use_GPU=True?
colfi/models_ann.py:    def predict(self, obs, use_GPU=False, in_type='numpy'):
colfi/models_ann.py:        self.pred_params = self._predict(obs, use_GPU=use_GPU, in_type=in_type)
colfi/models_ann.py:    def predict_chain(self, obs_data, cov_matrix=None, chain_leng=10000, use_GPU=False):
colfi/models_ann.py:        self.obs_best_multi = ds.AddGaussianNoise(self.obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisyObs()
colfi/models_ann.py:        self.chain = self.predict(self.obs_best_multi, use_GPU=use_GPU, in_type='numpy')
colfi/models_ann.py:    def predict_params(self, sim_obs, use_GPU=False):
colfi/models_ann.py:        params = self.predict(sim_obs, use_GPU=use_GPU, in_type='numpy')
colfi/models_ann.py:            if self.use_multiGPU:
colfi/models_ann.py:            self.net.cuda()
colfi/models_ann.py:    #change the branch net & trunk net (contain training) to use multiple GPUs ???
colfi/models_ann.py:        if self.use_GPU:
colfi/models_ann.py:                exec('self.branch_net%s = self.branch_net%s.cuda(device)'%(i,i))
colfi/models_ann.py:#        if self.use_GPU:
colfi/models_ann.py:#            self.inputs = dp.numpy2cuda(self.inputs, device=device)
colfi/models_ann.py:#            self.target = dp.numpy2cuda(self.target, device=device)
colfi/models_ann.py:#            self.error = dp.numpy2cuda(self.error, device=device)
colfi/models_ann.py:            _inputs, _target = ds.AddGaussianNoise(self.inputs,params=self.target,obs_errors=self.error,cholesky_factor=self.cholesky_f,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_ann.py:        #this means that the branch networks can only be trained on 1 GPU, how to train them on muliple GPUs?
colfi/models_ann.py:        #Note: all networks should be transfered to GPU when using "mp.spawn" to train the branch networks
colfi/models_ann.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_ann.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_ann.py:                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_ann.py:        if self.use_multiGPU:
colfi/models_ann.py:            _inputs, _target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_ann.py:            _inputs, _target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_ann.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_ann.py:        if self.use_multiGPU:
colfi/models_ann.py:        obs_best_multi = ds.AddGaussianNoise(obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisyObs()
colfi/models_ann.py:#             self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_ann.py:#         if self.use_multiGPU:
colfi/models_ann.py:#         obs_best_multi = ds.AddGaussianNoise(obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisyObs()
colfi/element.py:    #here 'inplace=True' is used to save GPU memory
colfi/models_g.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_g.py:                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_g.py:        if self.use_multiGPU:
colfi/models_g.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=False)
colfi/models_g.py:                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_g.py:        if self.use_multiGPU:
colfi/models_g.py:    def _predict(self, obs, use_GPU=False, in_type='torch'):
colfi/models_g.py:        use_GPU : bool
colfi/models_g.py:            If True, calculate using GPU, otherwise, calculate using CPU.
colfi/models_g.py:        if use_GPU:
colfi/models_g.py:            self.net = self.net.cuda()
colfi/models_g.py:                obs = dp.numpy2cuda(obs)
colfi/models_g.py:                obs = dp.torch2cuda(obs)
colfi/models_g.py:        if use_GPU:
colfi/models_g.py:            pred = dp.cuda2numpy(pred.data)
colfi/models_g.py:    #change the branch net & trunck net (contain training) to use multiple GPUs ???
colfi/models_g.py:            _inputs, _target = ds.AddGaussianNoise(self.inputs,params=self.target,obs_errors=self.error,cholesky_factor=self.cholesky_f,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_g.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_g.py:            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_g.py:                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
colfi/models_g.py:        if self.use_multiGPU:
colfi/models_g.py:    def _predict(self, obs, use_GPU=False, in_type='numpy'):
colfi/models_g.py:        use_GPU : bool
colfi/models_g.py:            If True, calculate using GPU, otherwise, calculate using CPU.
colfi/models_g.py:        if use_GPU:
colfi/models_g.py:            self.net = self.net.cuda()
colfi/models_g.py:                obs = [dp.numpy2cuda(obs[i]) for i in range(len(obs))]
colfi/models_g.py:                obs = [dp.torch2cuda(obs[i]) for i in range(len(obs))]
colfi/models_g.py:        if use_GPU:
colfi/models_g.py:            pred = dp.cuda2numpy(pred.data)
colfi/data_simulator.py:    use_GPU : bool, optional
colfi/data_simulator.py:        If True, the noise will be generated by GPU, otherwise, it will be generated by CPU. Default: True
colfi/data_simulator.py:                 noise_type='multiNormal', factor_sigma=0.2, multi_noise=5, use_GPU=True):
colfi/data_simulator.py:        self.use_GPU = use_GPU
colfi/data_simulator.py:        if use_GPU:
colfi/data_simulator.py:            self.epsilon = torch.cuda.FloatTensor([1e-20])
colfi/data_simulator.py:        # Note 1: "torch.FloatTensor(ell_num).normal_().cuda()"(a) is equivalent to "torch.randn(ell_num).cuda()"(b)
colfi/data_simulator.py:        # and equivalent to "torch.cuda.FloatTensor(ell_num).normal_()"(c), in which (c) is faster than (a) and (b)
colfi/data_simulator.py:        # Note 2: in the method of cudaErr, if the input 'data' is in torch.cuda.FloatTensor type,
colfi/data_simulator.py:        # def cudaErr(data):
colfi/data_simulator.py:        #         (ii):  m=MultivariateNormal(torch.zeros(measurement_leng).cuda(), cov_cuda)
colfi/data_simulator.py:        #         (iii): m=MultivariateNormal(torch.cuda.FloatTensor(measurement_leng).zero_(), cov_cuda)
colfi/data_simulator.py:            if self.use_GPU:
colfi/data_simulator.py:                noise = torch.cuda.FloatTensor(measurement.size()).normal_(0,1) * obs_error
colfi/data_simulator.py:            if self.use_GPU:
colfi/data_simulator.py:                mean = torch.zeros(cholesky_factor.size(-1)).cuda()
colfi/data_simulator.py:            if self.use_GPU:
colfi/data_simulator.py:                error_factor = torch.abs(torch.cuda.FloatTensor(measurement.size(1)).normal_(0, factor_sigma)) #A !!!+, use this
colfi/data_simulator.py:                # error_factor = torch.abs(torch.cuda.FloatTensor(measurement.size(0), 1).normal_(0, factor_sigma))
colfi/data_simulator.py:                # error_factor = torch.abs(torch.cuda.FloatTensor(measurement.size()).normal_(0, factor_sigma)) !!!
colfi/data_simulator.py:                # error_factor = torch.abs(torch.cuda.FloatTensor(1).normal_(0, factor_sigma)) #test for map-->params
colfi/data_simulator.py:                # error_factor = Beta(1, 5).sample((measurement.size(1),)).cuda()
colfi/data_simulator.py:                # error_factor = Beta(1, 10).sample((measurement.size(1),)).cuda()
colfi/data_simulator.py:                # error_factor = Beta(1, 3).sample((measurement.size(1),)).cuda()
colfi/data_simulator.py:                # error_factor = torch.cuda.FloatTensor(measurement.size(1),).exponential_(6)
colfi/data_simulator.py:                # error_factor = torch.abs(Laplace(0, 0.2).sample((measurement.size(1),)).cuda()) #not good
colfi/data_simulator.py:                # error_factor = torch.abs(Laplace(0, 0.3).sample((measurement.size(1),)).cuda())
colfi/data_simulator.py:                # error_factor = torch.abs(Laplace(0, 0.1).sample((measurement.size(1),)).cuda())
colfi/data_simulator.py:            if self.use_GPU:
colfi/data_simulator.py:                error_factor = torch.abs(torch.cuda.FloatTensor(1).normal_(0, factor_sigma)) + self.epsilon #A !!, use this
colfi/data_simulator.py:                # mean = torch.zeros(cholesky_factor.size(-1)).cuda()
colfi/data_simulator.py:                # error_factor = torch.abs(torch.cuda.FloatTensor(measurement.size(0), 1).normal_(0, factor_sigma)) #!, less good
colfi/data_simulator.py:                # # error_factor = torch.abs(torch.cuda.FloatTensor(measurement.size(1)).normal_(0, factor_sigma)) #?
colfi/data_simulator.py:                # # error_factor = torch.abs(torch.cuda.FloatTensor(1).normal_(0, factor_sigma)) #?
colfi/data_simulator.py:                # error_factor = torch.abs(torch.cuda.FloatTensor(measurement.size(1)).normal_(0, factor_sigma)) + 1e-4 #self.epsilon #A !!, use this
colfi/data_simulator.py:                # error_factor_2 = torch.abs(torch.cuda.FloatTensor(measurement.size(1)).normal_(1, factor_sigma_2))
colfi/data_simulator.py:                # error_factor = torch.abs(torch.cuda.FloatTensor(1).normal_(0, factor_sigma)) + self.epsilon #A !!, use this
colfi/data_simulator.py:                # error_factor_2 = torch.abs(torch.cuda.FloatTensor(measurement.size(1),measurement.size(1)).normal_(1, factor_sigma_2))
colfi/data_simulator.py:                # error_factor = torch.abs(torch.cuda.FloatTensor(1).normal_(0, factor_sigma)) + self.epsilon #A !!, use this
colfi/data_processor.py:def numpy2cuda(data, device=None, dtype=torch.cuda.FloatTensor):
colfi/data_processor.py:    """ Transfer data from the numpy array (on CPU) to the torch tensor (on GPU). """
colfi/data_processor.py:        # dtype = torch.cuda.FloatTensor
colfi/data_processor.py:        data = torch2cuda(data, device=device)
colfi/data_processor.py:def torch2cuda(data, device=None):
colfi/data_processor.py:    """ Transfer data (torch tensor) from CPU to GPU. """
colfi/data_processor.py:    return data.cuda(device=device)
colfi/data_processor.py:def cuda2torch(data):
colfi/data_processor.py:    """ Transfer data (torch tensor) from GPU to CPU. """
colfi/data_processor.py:def cuda2numpy(data):
colfi/data_processor.py:    """ Transfer data from the torch tensor (on GPU) to the numpy array (on CPU). """
colfi/data_processor.py:def cpu2cuda(data):
colfi/data_processor.py:    """Transfer data from CPU to GPU.
colfi/data_processor.py:        return numpy2cuda(data)
colfi/data_processor.py:        return torch2cuda(data)
colfi/data_processor.py:    def check_GPU(self):
colfi/data_processor.py:        if torch.cuda.is_available():
colfi/data_processor.py:            device_ids = list(range(torch.cuda.device_count()))
colfi/data_processor.py:    def call_GPU(self, prints=True):
colfi/data_processor.py:        if torch.cuda.is_available():
colfi/data_processor.py:            self.use_GPU = True
colfi/data_processor.py:            gpu_num = torch.cuda.device_count()
colfi/data_processor.py:            if gpu_num > 1:
colfi/data_processor.py:                self.use_multiGPU = True
colfi/data_processor.py:                self._prints('\nTraining the network using {} GPUs'.format(gpu_num), prints=prints)
colfi/data_processor.py:                self.use_multiGPU = False
colfi/data_processor.py:                self._prints('\nTraining the network using 1 GPU', prints=prints)
colfi/data_processor.py:            self.use_GPU = False
colfi/data_processor.py:            self.use_multiGPU = False
colfi/data_processor.py:        self.call_GPU(prints=prints)
colfi/data_processor.py:        if self.use_GPU:
colfi/data_processor.py:            self.net = self.net.cuda(device=device)
colfi/data_processor.py:            if self.use_multiGPU:
colfi/data_processor.py:        if self.use_GPU:
colfi/data_processor.py:            self.obs_base_torch = numpy2cuda(self.obs_base)
colfi/data_processor.py:            self.params_base_torch = numpy2cuda(self.params_base)
colfi/data_processor.py:        if self.use_GPU:
colfi/data_processor.py:            self.obs = numpy2cuda(self.obs)
colfi/data_processor.py:            self.params = numpy2cuda(self.params)
colfi/data_processor.py:                self.obs_base_torch = numpy2cuda(self.obs_base)
colfi/data_processor.py:                self.params_base_torch = numpy2cuda(self.params_base)
colfi/data_processor.py:        if self.use_GPU:
colfi/data_processor.py:                self.obs_vali = numpy2cuda(self.obs_vali)
colfi/data_processor.py:                self.params_vali = numpy2cuda(self.params_vali)
colfi/data_processor.py:        if self.use_GPU:
colfi/data_processor.py:            self.obs = numpy2cuda(self.obs)
colfi/data_processor.py:            self.params = numpy2cuda(self.params)
colfi/data_processor.py:                self.obs_errors = numpy2cuda(self.obs_errors)
colfi/data_processor.py:                self.cholesky_factor = numpy2cuda(self.cholesky_factor)
colfi/data_processor.py:            self.obs_base_torch = numpy2cuda(self.obs_base)
colfi/data_processor.py:            self.params_base_torch = numpy2cuda(self.params_base)
colfi/data_processor.py:                self.obs_vali = numpy2cuda(self.obs_vali)
colfi/data_processor.py:                self.params_vali = numpy2cuda(self.params_vali)
colfi/data_processor.py:        if self.use_GPU:
colfi/data_processor.py:            self.obs = [numpy2cuda(self.obs[i]) for i in range(self.branch_n)]
colfi/data_processor.py:            self.params = numpy2cuda(self.params)
colfi/data_processor.py:            self.obs_base_torch = [numpy2cuda(self.obs_base[i]) for i in range(self.branch_n)]
colfi/data_processor.py:            self.params_base_torch = numpy2cuda(self.params_base)
colfi/data_processor.py:        if self.use_GPU:
colfi/data_processor.py:                self.obs_vali = [numpy2cuda(self.obs_vali[i]) for i in range(self.branch_n)]
colfi/data_processor.py:                self.params_vali = numpy2cuda(self.params_vali)
colfi/data_processor.py:        if self.use_GPU:
colfi/data_processor.py:            self.obs = [numpy2cuda(self.obs[i]) for i in range(self.branch_n)]
colfi/data_processor.py:            self.params = numpy2cuda(self.params)
colfi/data_processor.py:                    self.obs_errors[i] = numpy2cuda(self.obs_errors[i])
colfi/data_processor.py:                    self.cholesky_factor[i] = numpy2cuda(self.cholesky_factor[i])
colfi/data_processor.py:            self.obs_base_torch = [numpy2cuda(self.obs_base[i]) for i in range(self.branch_n)]
colfi/data_processor.py:            self.params_base_torch = numpy2cuda(self.params_base)
colfi/data_processor.py:                self.obs_vali = [numpy2cuda(self.obs_vali[i]) for i in range(self.branch_n)]
colfi/data_processor.py:                self.params_vali = numpy2cuda(self.params_vali)
colfi/data_processor.py:    def statistic_torch(self, use_GPU=True):
colfi/data_processor.py:        if use_GPU:
colfi/data_processor.py:                st[e] = numpy2cuda(st[e])
colfi/data_processor.py:                self.obs_statistic_torch = Statistic(self.obs[:max_idx]/self.obs_base, dim=self.statistic_dim_obs).statistic_torch(use_GPU=self.use_GPU)
colfi/data_processor.py:                self.obs_statistic_torch = Statistic(self.obs[:max_idx], dim=self.statistic_dim_obs).statistic_torch(use_GPU=self.use_GPU)
colfi/data_processor.py:                self.params_statistic_torch = Statistic(self.params_tot/self.params_base, dim=self.statistic_dim_params).statistic_torch(use_GPU=self.use_GPU)
colfi/data_processor.py:                self.params_statistic_torch = Statistic(self.params_tot, dim=self.statistic_dim_params).statistic_torch(use_GPU=self.use_GPU)
colfi/data_processor.py:                self.obs_statistic_torch = [Statistic(self.obs[i]/self.obs_base[i], dim=self.statistic_dim_obs).statistic_torch(use_GPU=self.use_GPU) for i in range(len(self.obs))]
colfi/data_processor.py:                self.obs_statistic_torch = [Statistic(self.obs[i], dim=self.statistic_dim_obs).statistic_torch(use_GPU=self.use_GPU) for i in range(len(self.obs))]
colfi/data_processor.py:                self.params_statistic_torch = Statistic(self.params_tot/self.params_base, dim=self.statistic_dim_params).statistic_torch(use_GPU=self.use_GPU)
colfi/data_processor.py:                self.params_statistic_torch = Statistic(self.params_tot, dim=self.statistic_dim_params).statistic_torch(use_GPU=self.use_GPU)
colfi/models_mg.py:        self.obs_best_multi = ds.AddGaussianNoise(self.obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisyObs()
colfi/models_mg.py:        obs_best_multi = ds.AddGaussianNoise(obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisyObs()

```
