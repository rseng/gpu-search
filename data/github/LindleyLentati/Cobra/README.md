# https://github.com/LindleyLentati/Cobra

```console
README.rst:GPU accelerated Bayesian pulsar search pipeline
DatClass.py:import pycuda.autoinit
DatClass.py:import pycuda.gpuarray as gpuarray
DatClass.py:from pycuda.compiler import SourceModule
DatClass.py:import pycuda.cumath as cumath
DatClass.py:from pycuda.elementwise import ElementwiseKernel
DatClass.py:import pycuda.driver as drv
DatClass.py:import skcuda.fft as fft
DatClass.py:import skcuda.linalg as cula
DatClass.py:		self.gpu_fft_data = None
DatClass.py:		self.gpu_time = None
DatClass.py:		self.gpu_pulsar_signal = None
DatClass.py:		self.gpu_pulsar_fft = None
DatClass.py:			gpu_Data = gpuarray.to_gpu(np.float64(self.Data))
DatClass.py:			self.gpu_fft_data  = gpuarray.zeros(self.NSamps/2+1, np.complex128)
DatClass.py:			fft.fft(gpu_Data, self.gpu_fft_data, self.Plan) 
DatClass.py:			self.gpu_fft_data = self.gpu_fft_data[1:-1]
DatClass.py:			gpu_Data.gpudata.free()
DatClass.py:#			self.Real = gpuarray.empty(self.NSamps/2-1, np.float64)
DatClass.py:#			self.Imag = gpuarray.empty(self.NSamps/2-1, np.float64)
DatClass.py:#			self.Real = gpuarray.to_gpu(np.float64(gpu_fftData.real[1:-1].get()))
DatClass.py:#			self.Imag = gpuarray.to_gpu(np.float64(gpu_fftData.imag[1:-1].get()))
DatClass.py:			self.FSamps=len(self.gpu_fft_data)
DatClass.py:			#self.SampleFreqs = gpuarray.empty(self.FSamps, np.float64)
DatClass.py:			#self.SampleFreqs = gpuarray.to_gpu(2.0*np.pi*np.float64(np.arange(1,self.FSamps+1))/self.TObs)
DatClass.py:			self.gpu_time = gpuarray.to_gpu(np.float64(self.BaseTime))
DatClass.py:			self.gpu_pulsar_signal = gpuarray.empty(self.NSamps, np.float64)
DatClass.py:			self.gpu_pulsar_fft = gpuarray.empty(self.NSamps/2+1, np.complex128)
DatClass.py:		OComp = self.gpu_fft_data.get()
DatClass.py:		self.gpu_fft_data = gpuarray.to_gpu(np.complex128(NComp))
DatClass.py:		self.Real = gpuarray.to_gpu(np.float64(NComp.real))
DatClass.py:                self.Imag = gpuarray.to_gpu(np.float64(NComp.imag))
DatClass.py:				self.gpu_fft_data[start:stop] /= noise
DatClass.py:				fftdataR = (self.gpu_fft_data.get()).real
DatClass.py:				fftdataI = (self.gpu_fft_data.get()).imag
DatClass.py:				fftdata = self.gpu_fft_data.get()
DatClass.py:				self.gpu_fft_data = gpuarray.to_gpu(np.complex128(fftdata))
DatClass.py:			fftD = self.gpu_fft_data.get()
DatClass.py:                        self.gpu_fft_data = self.gpu_fft_data / noise
DatClass.py:		#self.Noise = gpuarray.empty(self.FSamps, np.float64)
DatClass.py:		self.Noise = gpuarray.to_gpu(np.float64(noisevec))
DatClass.py:		sig=self.gpu_pulsar_signal.get()
Cobra.py:import pycuda.autoinit
Cobra.py:import pycuda.gpuarray as gpuarray
Cobra.py:from pycuda.compiler import SourceModule
Cobra.py:import pycuda.cumath as cumath
Cobra.py:from pycuda.elementwise import ElementwiseKernel
Cobra.py:import pycuda.driver as drv
Cobra.py:import skcuda.fft as fft
Cobra.py:import skcuda.linalg as cula
Cobra.py:import skcuda.cublas as cublas
Cobra.py:			"pycuda::complex<double> *a, double *b",
Cobra.py:			self.CosOrbit = gpuarray.empty(self.InterpBinarySteps+1, np.float64)
Cobra.py:                        self.SinOrbit = gpuarray.empty(self.InterpBinarySteps+1, np.float64)
Cobra.py:			self.CosOrbit = gpuarray.to_gpu(np.float64(self.CPUCosOrbit))
Cobra.py:                        self.SinOrbit = gpuarray.to_gpu(np.float64(self.CPUSinOrbit))
Cobra.py:				self.CosOrbit.append(gpuarray.empty(self.InterpBinarySteps+1, np.float64))
Cobra.py:				self.SinOrbit.append(gpuarray.empty(self.InterpBinarySteps+1, np.float64))
Cobra.py:				self.CosOrbit[i]  = gpuarray.to_gpu(np.float64(self.CPUCosOrbit[i]))
Cobra.py:				self.SinOrbit[i]  = gpuarray.to_gpu(np.float64(self.CPUSinOrbit[i]))
Cobra.py:                                self.CosOrbit.append(gpuarray.empty(self.InterpBinarySteps+1, np.float64))
Cobra.py:                                self.SinOrbit.append(gpuarray.empty(self.InterpBinarySteps+1, np.float64))
Cobra.py:                                self.CosOrbit[i]  = gpuarray.to_gpu(np.float64(self.CPUCosOrbit[i]))
Cobra.py:                                self.SinOrbit[i]  = gpuarray.to_gpu(np.float64(self.CPUSinOrbit[i]))
Cobra.py:				self.TrueAnomaly.append(gpuarray.empty(self.InterpBinarySteps+1, np.float64))
Cobra.py:				self.TrueAnomaly[i] = gpuarray.to_gpu(np.float64(self.CPUTrueAnomaly[i]))
Cobra.py:	def gaussGPULike(self, x):
Cobra.py:				self.addInterpEccBinary(self.DatFiles[i].gpu_pulsar_signal,  self.DatFiles[i].gpu_time, self.CosOrbit[EccBin], self.SinOrbit[EccBin], BinaryPeriod, BinaryPhase, BinaryAmp, CosOmega, SinOmega, Ecc, phase,  period, width**2, BinaryAmp*blin, Alpha, Beta, grid=(self.DatFiles[i].Tblocks,1), block=(self.DatFiles[i].block_size,1,1))
Cobra.py:                                self.addInterpGRBinary(self.DatFiles[i].gpu_pulsar_signal,  self.DatFiles[i].gpu_time, self.CosOrbit[EccBin], self.SinOrbit[EccBin], self.TrueAnomaly[EccBin], BinaryPeriod, BinaryPhase, BinaryAmp, Omega, Ecc, M2, OMDot, SINI, Gamma, PBDot, SqEcc_th, Ecc_r, arr, ar, phase,  period, width**2, BinaryAmp*blin, self.pepoch, grid=(self.DatFiles[i].Tblocks,1), block=(self.DatFiles[i].block_size,1,1))
Cobra.py:				self.addInterpCircBinary(self.DatFiles[i].gpu_pulsar_signal,  self.DatFiles[i].gpu_time, self.CosOrbit, self.SinOrbit, BinaryPeriod, BinaryPhase, BinaryAmp, phase,  period, width**2, BinaryAmp*blin,  eta, Beta, H2Beta, grid=(self.DatFiles[i].Tblocks,1), block=(self.DatFiles[i].block_size,1,1))
Cobra.py:				self.AddAcceleration(self.DatFiles[i].gpu_pulsar_signal,  self.DatFiles[i].gpu_time, Acceleration, period, phase, width**2,  grid=(self.DatFiles[i].Tblocks,1), block=(self.DatFiles[i].block_size,1,1))
Cobra.py:				self.MakeSignal(self.DatFiles[i].gpu_pulsar_signal, self.DatFiles[i].gpu_time, period, width**2, phase, grid=(self.DatFiles[i].Tblocks,1), block=(self.DatFiles[i].block_size,1,1))
Cobra.py:			fft.fft(self.DatFiles[i].gpu_pulsar_signal, self.DatFiles[i].gpu_pulsar_fft, self.DatFiles[i].Plan) 
Cobra.py:			self.MultNoise(self.DatFiles[i].gpu_pulsar_fft[1:-1], self.DatFiles[i].Noise)
Cobra.py:			mcdot=cublas.cublasZdotc(h, self.DatFiles[i].FSamps, (self.DatFiles[i].gpu_pulsar_fft[1:-1]).gpudata, 1, (self.DatFiles[i].gpu_pulsar_fft[1:-1]).gpudata, 1).real
Cobra.py:			cdot = cublas.cublasZdotc(h, self.DatFiles[i].FSamps, self.DatFiles[i].gpu_fft_data.gpudata, 1,(self.DatFiles[i].gpu_pulsar_fft[1:-1]).gpudata, 1).real
Cobra.py:				#np.savetxt(self.ChainRoot+"Real_"+str(i)+".dat", zip(self.DatFiles[i].SampleFreqs.get(), self.DatFiles[i].Real.get(), (MLAmp*self.DatFiles[i].gpu_pulsar_fft.get()).real[1:-1]))
Cobra.py:				#np.savetxt(self.ChainRoot+"Imag_"+str(i)+".dat", zip(self.DatFiles[i].SampleFreqs.get(), self.DatFiles[i].Imag.get(), (MLAmp*self.DatFiles[i].gpu_pulsar_fft.get()).imag[1:-1]))
Cobra.py:				#self.DatFiles[i].gpu_pulsar_signal = self.DatFiles[i].gpu_time - phase*period
Cobra.py:			#	self.GetPhaseBins(self.DatFiles[i].gpu_pulsar_signal, period, grid=(self.DatFiles[i].Tblocks,1), block=(self.DatFiles[i].block_size,1,1))
Cobra.py:				#phasebins=self.DatFiles[i].gpu_pulsar_signal.get()
Cobra.py:			self.DatFiles[i].gpu_pulsar_signal = self.DatFiles[i].gpu_time - 0*period
Cobra.py:			self.MakeSignal(self.DatFiles[i].gpu_pulsar_signal, period, ((period*width)**2), grid=(self.DatFiles[i].Tblocks,1), block=(self.DatFiles[i].block_size,1,1))
Cobra.py:			s = self.DatFiles[i].gpu_pulsar_signal.get()
Cobra.py:                        fft.fft(self.DatFiles[i].gpu_pulsar_signal, self.DatFiles[i].gpu_pulsar_fft, self.DatFiles[i].Plan)
Cobra.py:			ranPhases = np.random.uniform(0,1, len(self.DatFiles[i].gpu_pulsar_fft))
Cobra.py:			OComp = self.DatFiles[i].gpu_pulsar_fft.get()
Cobra.py:	def GaussGPULikeWrap(self, cube, ndim, nparams):
Cobra.py:		like, dp =  self.gaussGPULike(x)
Cobra.py:		self.gaussGPULike(self.ML)
Cobra.py:			pymultinest.run(self.GaussGPULikeWrap, self.MNprior, self.Cand.n_dims, n_params = self.Cand.n_params, importance_nested_sampling = False, resume = resume, verbose = True, sampling_efficiency = efr, multimodal=False, const_efficiency_mode = ceff, n_live_points = nlive, outputfiles_basename=self.ChainRoot, wrapped_params=self.Cand.wrapped)

```
