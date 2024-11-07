# https://github.com/NLESC-JCER/QMCTorch

```console
docs/index.rst:   notebooks/gpu
docs/conf.py:#     'torch.cuda',
docs/conf.py:    'torch.cuda',
docs/example/backflow/backflow.py:    def __init__(self, mol, cuda, size=16):
docs/example/backflow/backflow.py:        super().__init__(mol, cuda)
docs/example/gpu/h2.py:                   cuda=True)
docs/example/gpu/h2.py:                     cuda=True)
docs/example/horovod/h2.py:use_cuda = torch.cuda.is_available()
docs/example/horovod/h2.py:if use_cuda:
docs/example/horovod/h2.py:    torch.cuda.set_device(hvd.rank())
docs/example/horovod/h2.py:                   cuda=use_cuda)
docs/example/horovod/h2.py:                     cuda=use_cuda)
tests/wavefunction/orbitals/backflow/test_backflow_kernel_generic_pyscf.py:    def __init__(self, mol, cuda=False):
tests/wavefunction/orbitals/backflow/test_backflow_kernel_generic_pyscf.py:        super().__init__(mol, cuda)
paper/paper.md:in a physically-motivated neural network. The use of `PyTorch` as a backend to perform the optimization, allows leveraging automatic differentiation and GPU computing to accelerate the development and deployment of QMC simulations. `QMCTorch` supports the use of both Gaussian and Slater type orbitals via interface to popular quantum chemistry packages `pyscf` and `ADF`.
paper/paper.md:`QMCTorch` is a Python package using `PyTorch` [@pytorch] as a backend to perform Quantum Monte-Carlo (QMC) simulations, namely Variational Monte-Carlo,  of molecular systems. Many software such as `QMCPack`[@qmcpack], `QMC=Chem` [@qmcchem], `CHAMP` [@champ] provide high-quality implementation of advanced QMC methodologies in low-level languages (C++/Fortran).  Python implementations of QMC such as `PAUXY` [@pauxy] and `PyQMC` [@pyqmc] have also been proposed to facilitate the use and development of QMC techniques. Large efforts have been made to leverage recent development of deep learning techniques for QMC simulations with for example the creation of neural-network based wave-function ansatz [@paulinet; @ferminet; @choo_fermionic_2020; @HAN2019108929; @ANN_QMC; @detfree_nn; @fixed_node; @Lin_2023; @ANN_WF] that have lead to very interesting results. `QMCTorch` allows to perform QMC simulations using physically motivated neural network architectures that closely follow the wave function ansatz used by QMC practitioners. Its architecture allows to rapidly explore new functional forms of some key elements of the wave function ansatz. Users do not need to derive analytical expressions for the gradients of the total energy w.r.t. the variational parameters, that are simply obtained via automatic differentiation. This includes for example the parameters of the atomic orbitals that can be variationally optimized and the atomic coordinates that allows `QMCTorch` to perform geometry optimization of molecular structures. In addition, the GPU capabilities offered by `PyTorch` combined with the parallelization over multiple computing nodes obtained via `Horovod` [@horovod], allow to deploy the simulations on large heterogeneous computing architectures. In summary, `QMCTorch` provides QMC practitioners a framework to rapidly prototype new ideas and to test them using modern computing resources.
paper/paper.md:The snippet of code above shows a typical example of `QMCTorch` script. A `Molecule` object is first created by specifying the atomic positions and the calculator required to run the HF or DFT calculations (here `pyscf` using  a `sto-3g` basis set). This molecule is then used to create a `SlaterJastrow` wave function ansatz. Other options, such as the required Jastrow kernel, active space, and the use of GPUs can also be specified here. A sampler and optimizer are then defined that are then used with the wave function to instantiate the solver. This solver can then be used to optimize the variational parameters, that is done here through 50 epochs. 
qmctorch/wavefunction/slater_jastrow_base.py:                 cuda=False,
qmctorch/wavefunction/slater_jastrow_base.py:            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
qmctorch/wavefunction/slater_jastrow_base.py:            mol.nelec, 3, kinetic, cuda)
qmctorch/wavefunction/slater_jastrow_base.py:        # check for cuda
qmctorch/wavefunction/slater_jastrow_base.py:        if not torch.cuda.is_available and self.cuda:
qmctorch/wavefunction/slater_jastrow_base.py:            raise ValueError('Cuda not available, use cuda=False')
qmctorch/wavefunction/slater_jastrow_base.py:        self.ao = AtomicOrbitals(mol, cuda)
qmctorch/wavefunction/slater_jastrow_base.py:        if self.cuda:
qmctorch/wavefunction/slater_jastrow_base.py:        if self.cuda:
qmctorch/wavefunction/slater_jastrow_base.py:                                  self.configs, mol, cuda)
qmctorch/wavefunction/slater_jastrow_base.py:        if self.cuda:
qmctorch/wavefunction/slater_jastrow_base.py:        if self.cuda:
qmctorch/wavefunction/slater_jastrow_base.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/slater_jastrow_base.py:        log.info('  Cuda support        : {0}', self.cuda)
qmctorch/wavefunction/slater_jastrow_base.py:        if self.cuda:
qmctorch/wavefunction/slater_jastrow_base.py:                '  GPU                 : {0}', torch.cuda.get_device_name(0))
qmctorch/wavefunction/slater_jastrow_base.py:                              cuda=self.cuda,
qmctorch/wavefunction/slater_jastrow.py:                 cuda=False,
qmctorch/wavefunction/slater_jastrow.py:            cuda (bool, optional): turns GPU ON/OFF  Defaults to Fals   e.
qmctorch/wavefunction/slater_jastrow.py:        super().__init__(mol, configs, kinetic, cuda, include_all_mo)
qmctorch/wavefunction/slater_jastrow.py:                kernel_kwargs=jastrow_kernel_kwargs, cuda=cuda)
qmctorch/wavefunction/slater_jastrow.py:            if self.cuda:
qmctorch/wavefunction/jastrows/elec_nuclei/jastrow_factor_electron_nuclei.py:                 cuda=False):
qmctorch/wavefunction/jastrows/elec_nuclei/jastrow_factor_electron_nuclei.py:            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
qmctorch/wavefunction/jastrows/elec_nuclei/jastrow_factor_electron_nuclei.py:        self.cuda = cuda
qmctorch/wavefunction/jastrows/elec_nuclei/jastrow_factor_electron_nuclei.py:        if self.cuda:
qmctorch/wavefunction/jastrows/elec_nuclei/jastrow_factor_electron_nuclei.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/jastrows/elec_nuclei/jastrow_factor_electron_nuclei.py:                                             atomic_pos, cuda,
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/pade_jastrow_kernel.py:    def __init__(self, nup, ndown, atomic_pos, cuda, w=1.):
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/pade_jastrow_kernel.py:            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/pade_jastrow_kernel.py:        super().__init__(nup, ndown, atomic_pos, cuda)
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/jastrow_kernel_electron_nuclei_base.py:    def __init__(self, nup, ndown, atomic_pos, cuda, **kwargs):
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/jastrow_kernel_electron_nuclei_base.py:            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/jastrow_kernel_electron_nuclei_base.py:        self.cuda = cuda
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/jastrow_kernel_electron_nuclei_base.py:        if self.cuda:
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/jastrow_kernel_electron_nuclei_base.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/fully_connected_jastrow_kernel.py:    def __init__(self, nup, ndown, atomic_pos, cuda, w=1.):
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/fully_connected_jastrow_kernel.py:            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
qmctorch/wavefunction/jastrows/elec_nuclei/kernels/fully_connected_jastrow_kernel.py:        super().__init__(nup, ndown, atomic_pos, cuda)
qmctorch/wavefunction/jastrows/elec_elec/jastrow_factor_electron_electron.py:                 cuda=False):
qmctorch/wavefunction/jastrows/elec_elec/jastrow_factor_electron_electron.py:            cuda (bool, optional): use cuda. Defaults to False.
qmctorch/wavefunction/jastrows/elec_elec/jastrow_factor_electron_electron.py:        self.cuda = cuda
qmctorch/wavefunction/jastrows/elec_elec/jastrow_factor_electron_electron.py:        if self.cuda:
qmctorch/wavefunction/jastrows/elec_elec/jastrow_factor_electron_electron.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/jastrows/elec_elec/jastrow_factor_electron_electron.py:                nup, ndown, number_of_orbitals, cuda, jastrow_kernel, kernel_kwargs)
qmctorch/wavefunction/jastrows/elec_elec/jastrow_factor_electron_electron.py:                nup, ndown, cuda, **kernel_kwargs)
qmctorch/wavefunction/jastrows/elec_elec/kernels/pade_jastrow_kernel.py:    def __init__(self, nup, ndown, cuda, w=1.):
qmctorch/wavefunction/jastrows/elec_elec/kernels/pade_jastrow_kernel.py:            cuda (bool): Turns GPU ON/OFF.
qmctorch/wavefunction/jastrows/elec_elec/kernels/pade_jastrow_kernel.py:        super().__init__(nup, ndown, cuda)
qmctorch/wavefunction/jastrows/elec_elec/kernels/fully_connected_jastrow_kernel.py:    def __init__(self,  nup, ndown, cuda,
qmctorch/wavefunction/jastrows/elec_elec/kernels/fully_connected_jastrow_kernel.py:        super().__init__(nup, ndown, cuda)
qmctorch/wavefunction/jastrows/elec_elec/kernels/pade_jastrow_polynomial_kernel.py:    def __init__(self, nup, ndown, cuda,
qmctorch/wavefunction/jastrows/elec_elec/kernels/pade_jastrow_polynomial_kernel.py:            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
qmctorch/wavefunction/jastrows/elec_elec/kernels/pade_jastrow_polynomial_kernel.py:        super().__init__(nup, ndown, cuda)
qmctorch/wavefunction/jastrows/elec_elec/kernels/jastrow_kernel_electron_electron_base.py:    def __init__(self, nup, ndown, cuda, **kwargs):
qmctorch/wavefunction/jastrows/elec_elec/kernels/jastrow_kernel_electron_electron_base.py:            cuda (bool, optional): [description]. Defaults to False.
qmctorch/wavefunction/jastrows/elec_elec/kernels/jastrow_kernel_electron_electron_base.py:        self.cuda = cuda
qmctorch/wavefunction/jastrows/elec_elec/kernels/jastrow_kernel_electron_electron_base.py:        if self.cuda:
qmctorch/wavefunction/jastrows/elec_elec/kernels/jastrow_kernel_electron_electron_base.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/jastrows/elec_elec/orbital_dependent_jastrow_kernel.py:    def __init__(self, nup, ndown, nmo, cuda,
qmctorch/wavefunction/jastrows/elec_elec/orbital_dependent_jastrow_kernel.py:            cuda (bool): use GPUs
qmctorch/wavefunction/jastrows/elec_elec/orbital_dependent_jastrow_kernel.py:        super().__init__(nup, ndown, cuda)
qmctorch/wavefunction/jastrows/elec_elec/orbital_dependent_jastrow_kernel.py:            [jastrow_kernel(nup, ndown, cuda, **kernel_kwargs) for _ in range(self.nmo)])
qmctorch/wavefunction/jastrows/jastrow_factor_combined_terms.py:                 cuda=False):
qmctorch/wavefunction/jastrows/jastrow_factor_combined_terms.py:            cuda (bool, optional): [description]. Defaults to False.
qmctorch/wavefunction/jastrows/jastrow_factor_combined_terms.py:        self.cuda = cuda
qmctorch/wavefunction/jastrows/jastrow_factor_combined_terms.py:                                                                    cuda=cuda))
qmctorch/wavefunction/jastrows/jastrow_factor_combined_terms.py:                                                                  cuda=cuda))
qmctorch/wavefunction/jastrows/jastrow_factor_combined_terms.py:                                                                          cuda=cuda))
qmctorch/wavefunction/jastrows/elec_elec_nuclei/jastrow_factor_electron_electron_nuclei.py:                 cuda=False):
qmctorch/wavefunction/jastrows/elec_elec_nuclei/jastrow_factor_electron_electron_nuclei.py:            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
qmctorch/wavefunction/jastrows/elec_elec_nuclei/jastrow_factor_electron_electron_nuclei.py:        self.cuda = cuda
qmctorch/wavefunction/jastrows/elec_elec_nuclei/jastrow_factor_electron_electron_nuclei.py:        if self.cuda:
qmctorch/wavefunction/jastrows/elec_elec_nuclei/jastrow_factor_electron_electron_nuclei.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/jastrows/elec_elec_nuclei/jastrow_factor_electron_electron_nuclei.py:                                             cuda,
qmctorch/wavefunction/jastrows/elec_elec_nuclei/jastrow_factor_electron_electron_nuclei.py:        self.device = torch.device('cuda')
qmctorch/wavefunction/jastrows/elec_elec_nuclei/kernels/jastrow_kernel_electron_electron_nuclei_base.py:    def __init__(self, nup, ndown, atomic_pos, cuda, **kwargs):
qmctorch/wavefunction/jastrows/elec_elec_nuclei/kernels/jastrow_kernel_electron_electron_nuclei_base.py:            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
qmctorch/wavefunction/jastrows/elec_elec_nuclei/kernels/jastrow_kernel_electron_electron_nuclei_base.py:        self.cuda = cuda
qmctorch/wavefunction/jastrows/elec_elec_nuclei/kernels/jastrow_kernel_electron_electron_nuclei_base.py:        if self.cuda:
qmctorch/wavefunction/jastrows/elec_elec_nuclei/kernels/jastrow_kernel_electron_electron_nuclei_base.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/jastrows/elec_elec_nuclei/kernels/fully_connected_jastrow_kernel.py:    def __init__(self, nup, ndown, atomic_pos, cuda):
qmctorch/wavefunction/jastrows/elec_elec_nuclei/kernels/fully_connected_jastrow_kernel.py:        super().__init__(nup, ndown, atomic_pos, cuda)
qmctorch/wavefunction/jastrows/elec_elec_nuclei/kernels/boys_handy_jastrow_kernel.py:    def __init__(self, nup, ndown, atomic_pos, cuda, nterm=5):
qmctorch/wavefunction/jastrows/elec_elec_nuclei/kernels/boys_handy_jastrow_kernel.py:        super().__init__(nup, ndown, atomic_pos, cuda)
qmctorch/wavefunction/slater_orbital_dependent_jastrow.py:                 cuda=False,
qmctorch/wavefunction/slater_orbital_dependent_jastrow.py:            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
qmctorch/wavefunction/slater_orbital_dependent_jastrow.py:        super().__init__(mol, configs, kinetic, cuda, include_all_mo)
qmctorch/wavefunction/slater_orbital_dependent_jastrow.py:            cuda=self.cuda)
qmctorch/wavefunction/slater_orbital_dependent_jastrow.py:        if self.cuda:
qmctorch/wavefunction/pooling/slater_pooling.py:    def __init__(self, config_method, configs, mol, cuda=False):
qmctorch/wavefunction/pooling/slater_pooling.py:            cuda (bool, optional): Turns GPU ON/OFF. Defaults to False.
qmctorch/wavefunction/pooling/slater_pooling.py:        self.orb_proj = OrbitalProjector(configs, mol, cuda=cuda)
qmctorch/wavefunction/pooling/slater_pooling.py:                                       cuda=cuda)
qmctorch/wavefunction/pooling/slater_pooling.py:        if cuda:
qmctorch/wavefunction/pooling/slater_pooling.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/pooling/orbital_projector.py:    def __init__(self, configs, mol, cuda=False):
qmctorch/wavefunction/pooling/orbital_projector.py:            cuda (bool): use cuda or not
qmctorch/wavefunction/pooling/orbital_projector.py:        if cuda:
qmctorch/wavefunction/pooling/orbital_projector.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/pooling/orbital_projector.py:    def __init__(self, unique_excitations, mol, max_orb, cuda=False):
qmctorch/wavefunction/pooling/orbital_projector.py:            cuda (bool): use cuda or not
qmctorch/wavefunction/pooling/orbital_projector.py:        if cuda:
qmctorch/wavefunction/pooling/orbital_projector.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/slater_jastrow_backflow.py:                 cuda=False,
qmctorch/wavefunction/slater_jastrow_backflow.py:            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
qmctorch/wavefunction/slater_jastrow_backflow.py:        super().__init__(mol, configs, kinetic, cuda, include_all_mo)
qmctorch/wavefunction/slater_jastrow_backflow.py:                mol, backflow_kernel, backflow_kernel_kwargs, cuda)
qmctorch/wavefunction/slater_jastrow_backflow.py:                mol, backflow_kernel, backflow_kernel_kwargs, cuda)
qmctorch/wavefunction/slater_jastrow_backflow.py:            kernel_kwargs=jastrow_kernel_kwargs, cuda=cuda)
qmctorch/wavefunction/slater_jastrow_backflow.py:        if self.cuda:
qmctorch/wavefunction/slater_combined_jastrow.py:                 cuda=False,
qmctorch/wavefunction/slater_combined_jastrow.py:            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
qmctorch/wavefunction/slater_combined_jastrow.py:        super().__init__(mol, configs, kinetic, None, {}, cuda, include_all_mo)
qmctorch/wavefunction/slater_combined_jastrow.py:                cuda=cuda)
qmctorch/wavefunction/slater_combined_jastrow.py:            if self.cuda:
qmctorch/wavefunction/wf_base.py:    def __init__(self, nelec, ndim, kinetic='auto', cuda=False):
qmctorch/wavefunction/wf_base.py:        self.cuda = cuda
qmctorch/wavefunction/wf_base.py:        if self.cuda:
qmctorch/wavefunction/wf_base.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/slater_combined_jastrow_backflow.py:                 cuda=False,
qmctorch/wavefunction/slater_combined_jastrow_backflow.py:            cuda (bool, optional): turns GPU ON/OFF  Defaults to False.
qmctorch/wavefunction/slater_combined_jastrow_backflow.py:        super().__init__(mol, configs, kinetic, None, {}, cuda, include_all_mo)
qmctorch/wavefunction/slater_combined_jastrow_backflow.py:                mol, backflow_kernel, backflow_kernel_kwargs, cuda)
qmctorch/wavefunction/slater_combined_jastrow_backflow.py:                mol, backflow_kernel, backflow_kernel_kwargs, cuda)
qmctorch/wavefunction/slater_combined_jastrow_backflow.py:        if self.cuda:
qmctorch/wavefunction/slater_combined_jastrow_backflow.py:                cuda=cuda)
qmctorch/wavefunction/slater_combined_jastrow_backflow.py:            if self.cuda:
qmctorch/wavefunction/orbitals/atomic_orbitals_backflow.py:    def __init__(self, mol, backflow_kernel, backflow_kernel_kwargs={}, cuda=False):
qmctorch/wavefunction/orbitals/atomic_orbitals_backflow.py:            cuda (bool, optional): Turn GPU ON/OFF Defaults to False.
qmctorch/wavefunction/orbitals/atomic_orbitals_backflow.py:        super().__init__(mol, cuda)
qmctorch/wavefunction/orbitals/atomic_orbitals_backflow.py:                                                     cuda=cuda)
qmctorch/wavefunction/orbitals/atomic_orbitals.py:    def __init__(self, mol, cuda=False):
qmctorch/wavefunction/orbitals/atomic_orbitals.py:            cuda (bool, optional): Turn GPU ON/OFF Defaults to False.
qmctorch/wavefunction/orbitals/atomic_orbitals.py:                cuda=cuda)
qmctorch/wavefunction/orbitals/atomic_orbitals.py:                cuda=cuda)
qmctorch/wavefunction/orbitals/atomic_orbitals.py:        self.cuda = cuda
qmctorch/wavefunction/orbitals/atomic_orbitals.py:        if self.cuda:
qmctorch/wavefunction/orbitals/atomic_orbitals.py:        self.device = torch.device('cuda')
qmctorch/wavefunction/orbitals/backflow/backflow_transformation.py:    def __init__(self, mol, backflow_kernel, backflow_kernel_kwargs={}, cuda=False):
qmctorch/wavefunction/orbitals/backflow/backflow_transformation.py:                                               cuda,
qmctorch/wavefunction/orbitals/backflow/backflow_transformation.py:        self.cuda = cuda
qmctorch/wavefunction/orbitals/backflow/backflow_transformation.py:        if self.cuda:
qmctorch/wavefunction/orbitals/backflow/backflow_transformation.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/orbitals/backflow/orbital_dependent_backflow_kernel.py:    def __init__(self, backflow_kernel, backflow_kernel_kwargs, mol, cuda):
qmctorch/wavefunction/orbitals/backflow/orbital_dependent_backflow_kernel.py:            [backflow_kernel(mol, cuda, **backflow_kernel_kwargs) for iao in range(self.nao)])
qmctorch/wavefunction/orbitals/backflow/orbital_dependent_backflow_kernel.py:        self.cuda = cuda
qmctorch/wavefunction/orbitals/backflow/orbital_dependent_backflow_kernel.py:        if self.cuda:
qmctorch/wavefunction/orbitals/backflow/orbital_dependent_backflow_kernel.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_inverse.py:    def __init__(self, mol, cuda=False):
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_inverse.py:        super().__init__(mol, cuda)
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_base.py:    def __init__(self, mol, cuda):
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_base.py:        self.cuda = cuda
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_base.py:        if self.cuda:
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_base.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_square.py:    def __init__(self, mol, cuda=False):
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_square.py:        super().__init__(mol, cuda)
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_fully_connected.py:    def __init__(self, mol, cuda):
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_fully_connected.py:        super().__init__(mol, cuda)
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_power_sum.py:    def __init__(self, mol, cuda, order=2):
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_power_sum.py:        super().__init__(mol, cuda)
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_autodiff_inverse.py:    def __init__(self, mol, cuda, order=2):
qmctorch/wavefunction/orbitals/backflow/kernels/backflow_kernel_autodiff_inverse.py:        super().__init__(mol, cuda)
qmctorch/wavefunction/orbitals/backflow/orbital_dependent_backflow_transformation.py:    def __init__(self, mol, backflow_kernel, backflow_kernel_kwargs={}, cuda=False):
qmctorch/wavefunction/orbitals/backflow/orbital_dependent_backflow_transformation.py:            backflow_kernel, backflow_kernel_kwargs, mol, cuda)
qmctorch/wavefunction/orbitals/backflow/orbital_dependent_backflow_transformation.py:        self.cuda = cuda
qmctorch/wavefunction/orbitals/backflow/orbital_dependent_backflow_transformation.py:        if self.cuda:
qmctorch/wavefunction/orbitals/backflow/orbital_dependent_backflow_transformation.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/orbitals/spherical_harmonics.py:            cuda (bool): use cuda (defaults False)
qmctorch/wavefunction/orbitals/spherical_harmonics.py:        # check if we need cuda
qmctorch/wavefunction/orbitals/spherical_harmonics.py:        if 'cuda' not in kwargs:
qmctorch/wavefunction/orbitals/spherical_harmonics.py:            cuda = False
qmctorch/wavefunction/orbitals/spherical_harmonics.py:            cuda = kwargs['cuda']
qmctorch/wavefunction/orbitals/spherical_harmonics.py:        if cuda:
qmctorch/wavefunction/orbitals/spherical_harmonics.py:            self.device = torch.device('cuda')
qmctorch/wavefunction/orbitals/atomic_orbitals_orbital_dependent_backflow.py:    def __init__(self, mol, backflow_kernel, backflow_kernel_kwargs={}, cuda=False):
qmctorch/wavefunction/orbitals/atomic_orbitals_orbital_dependent_backflow.py:            cuda (bool, optional): Turn GPU ON/OFF Defaults to False.
qmctorch/wavefunction/orbitals/atomic_orbitals_orbital_dependent_backflow.py:        super().__init__(mol, cuda)
qmctorch/wavefunction/orbitals/atomic_orbitals_orbital_dependent_backflow.py:                                                                     cuda=cuda)
qmctorch/sampler/generalized_metropolis.py:                 cuda=False):
qmctorch/sampler/generalized_metropolis.py:            cuda (bool, optional): use cuda. Defaults to False.
qmctorch/sampler/generalized_metropolis.py:                             cuda)
qmctorch/sampler/hamiltonian.py:                 cuda: bool = False):
qmctorch/sampler/hamiltonian.py:            cuda (bool, optional): turn CUDA ON/OFF. Defaults to False.
qmctorch/sampler/hamiltonian.py:                             nelec, ndim, init, cuda)
qmctorch/sampler/walkers.py:                 init: Union[Dict, None] = None, cuda: bool = False):
qmctorch/sampler/walkers.py:            cuda (bool, optional): turn cuda ON/OFF. Defaults to False
qmctorch/sampler/walkers.py:        self.cuda = cuda
qmctorch/sampler/walkers.py:        if cuda:
qmctorch/sampler/walkers.py:            self.device = torch.device('cuda')
qmctorch/sampler/walkers.py:        if self.cuda:
qmctorch/sampler/walkers.py:            self.device = torch.device('cuda')
qmctorch/sampler/metropolis.py:                 cuda: bool = False):
qmctorch/sampler/metropolis.py:            cuda (bool, optional): turn CUDA ON/OFF. Defaults to False.
qmctorch/sampler/metropolis.py:                             nelec, ndim, init, cuda)
qmctorch/sampler/sampler_base.py:                 cuda):
qmctorch/sampler/sampler_base.py:            cuda ([type]): [description]
qmctorch/sampler/sampler_base.py:        self.cuda = cuda
qmctorch/sampler/sampler_base.py:        if cuda:
qmctorch/sampler/sampler_base.py:            self.device = torch.device('cuda')
qmctorch/sampler/sampler_base.py:            nwalkers=nwalkers, nelec=nelec, ndim=ndim, init=init, cuda=cuda)
qmctorch/solver/solver_mpi.py:        if self.cuda:
qmctorch/solver/solver_mpi.py:            if self.wf.cuda and pos.device.type == 'cpu':
qmctorch/solver/solver_base.py:        self.cuda = False
qmctorch/solver/solver_base.py:        # handles GPU availability
qmctorch/solver/solver_base.py:        if self.wf.cuda:
qmctorch/solver/solver_base.py:            self.device = torch.device('cuda')
qmctorch/solver/solver_base.py:            self.sampler.cuda = True
qmctorch/solver/solver_base.py:            self.sampler.walkers.cuda = True
qmctorch/solver/solver_base.py:        if self.wf.cuda and pos.device.type == 'cpu':
qmctorch/solver/solver_base.py:            #  get the position and put to gpu if necessary
qmctorch/solver/solver_base.py:            if self.wf.cuda and pos.device.type == 'cpu':
tests_hvd/test_h2_hvd.py:                                cuda=False)

```
