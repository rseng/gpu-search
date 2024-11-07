# https://github.com/NiallJeffrey/DeepMass

```console
DES_mass_maps_demo/original_run_scripts/splinter_singularity.sh:#PBS -q gpu
CMB_foreground_demo/wph_synthesis_script.py:print('GPU count: ' + str(torch.cuda.device_count()) + '\n')
CMB_foreground_demo/wph_synthesis_script.py:print(torch.cuda.device(device))
CMB_foreground_demo/foregrounds_utils.py:print(tf.config.list_physical_devices('GPU'))

```
