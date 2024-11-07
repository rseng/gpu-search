# https://github.com/TeamCOMPAS/COMPAS

```console
setup.py:    gpu=["cupy"],
compas_python_utils/cosmic_integration/binned_cosmic_integrator/cosmological_model.py:from .gpu_utils import xp
compas_python_utils/cosmic_integration/binned_cosmic_integrator/conversions.py:from .gpu_utils import xp
compas_python_utils/cosmic_integration/binned_cosmic_integrator/bbh_population.py:from .gpu_utils import xp
compas_python_utils/cosmic_integration/binned_cosmic_integrator/gpu_utils.py:    gpu_available = True
compas_python_utils/cosmic_integration/binned_cosmic_integrator/gpu_utils.py:    gpu_available = False
compas_python_utils/cosmic_integration/binned_cosmic_integrator/gpu_utils.py:if gpu_available:
compas_python_utils/cosmic_integration/binned_cosmic_integrator/bin_2d_data.py:from .gpu_utils import xp
compas_python_utils/cosmic_integration/binned_cosmic_integrator/snr_grid.py:from .gpu_utils import xp
compas_python_utils/cosmic_integration/binned_cosmic_integrator/snr_grid.py:        # TODO: SNRInterpolator can be GPU-ized
compas_python_utils/cosmic_integration/binned_cosmic_integrator/detection_matrix.py:from .gpu_utils import xp
compas_python_utils/cosmic_integration/binned_cosmic_integrator/detection_rate_computer.py:from .gpu_utils import xp
compas_python_utils/cosmic_integration/binned_cosmic_integrator/detection_rate_computer.py:    If the GPU is available, this function moves the np.ndarray objects to the GPU and perform the computation there.
compas_python_utils/cosmic_integration/binned_cosmic_integrator/detection_rate_computer.py:    If the GPU is not available, this function will perform the computation on the CPU.
compas_python_utils/detailed_evolution_plotter/van_den_heuvel_figures/vanDenHeuvelPlot.eps:/eopenclosed 16#029a
compas_python_utils/detailed_evolution_plotter/van_den_heuvel_figures/vanDenHeuvelPlot.eps:0a\Jg)3LI.&%Ds@d\j(7W,]%mcH)u%is]^FKb2KJm22E)jG)%83GPu*^f1V2ed?1T$V]+ZTQ1:1
online-docs/pages/User guide/Post-processing/notebooks/CosmicIntegration.py:# ## GPU usage
online-docs/pages/User guide/Post-processing/notebooks/CosmicIntegration.py:# If you have a CUDA-enabled GPU, the cosmic-integrator will automatically use it to speed up the calculation. To check if your GPU is used, you can run the following
online-docs/pages/User guide/Post-processing/notebooks/CosmicIntegration.py:from compas_python_utils.cosmic_integration.binned_cosmic_integrator.gpu_utils import gpu_available
online-docs/pages/User guide/Post-processing/notebooks/CosmicIntegration.py:print(f"GPU available: {gpu_available}")

```
