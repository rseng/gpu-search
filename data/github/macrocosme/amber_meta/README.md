# https://github.com/macrocosme/amber_meta

```console
amber_meta/amber_options.py:     'opencl_platform',
amber_meta/amber_options.py:     'opencl_device',
amber_meta/amber_options.py:    options_base = ['print', 'opencl_platform', 'opencl_device', 'device_name', 'sync', 'padding_file', 'zapped_channels', 'integration_steps', 'integration_file', 'compact_results', 'output', 'dms', 'dm_first', 'dm_step', 'threshold',] #'debug',
amber_meta/amber_run.py:        CPU id for process and GPU.
amber_meta/amber_run.py:        # First add the option with a dash (e.g. -opencl_platform)
amber_meta/amber_run.py:                   base_name='tuning_halfrate_3GPU_goodcentralfreq',
amber_meta/amber_run.py:                       'tuning_halfrate_3GPU_goodcentralfreq_step1',
amber_meta/amber_run.py:                       'tuning_halfrate_3GPU_goodcentralfreq_step2',
amber_meta/amber_run.py:                       'tuning_halfrate_3GPU_goodcentralfreq_step3'
amber_meta/amber_run.py:              base_name='tuning_halfrate_3GPU_goodcentralfreq',
output_frb190709_rfim/configuration/amber.conf:opencl_platform=0
output_frb190709_rfim/configuration/amber.conf:# number of GPU to use
output_frb190709_rfim/configuration/amber.conf:opencl_device=[1, 2, 3]
output_frb190709_rfim/scenario/tuning_1.sh:## OpenCL platform ID
output_frb190709_rfim/scenario/tuning_1.sh:OPENCL_PLATFORM="0"
output_frb190709_rfim/scenario/tuning_1.sh:## OpenCL device ID
output_frb190709_rfim/scenario/tuning_1.sh:OPENCL_DEVICE="0"
output_frb190709_rfim/scenario/tuning_1.sh:## Name of OpenCL device, used for configuration files
output_frb190709_rfim/scenario/tuning_1.sh:## Size, in bytes, of the OpenCL device's cache line
output_frb190709_rfim/scenario/tuning_1.sh:## Number of OpenCL work-items running simultaneously
output_frb190709_rfim/scenario/tuning_1.sh:## Maximum number of work-items in OpenCL dimension 0; dedispersion specific
output_frb190709_rfim/scenario/tuning_1.sh:## Maximum number of work-items in OpenCL dimension 1; dedispersion specific
output_frb190709_rfim/scenario/tuning_1.sh:## Maximum number of variables in OpenCL dimension 0; dedispersion specific
output_frb190709_rfim/scenario/tuning_1.sh:## Maximum number of variables in OpenCL dimension 1; dedispersion specific
output_frb190709_rfim/scenario/tuning_2.sh:## OpenCL platform ID
output_frb190709_rfim/scenario/tuning_2.sh:OPENCL_PLATFORM="0"
output_frb190709_rfim/scenario/tuning_2.sh:## OpenCL device ID
output_frb190709_rfim/scenario/tuning_2.sh:OPENCL_DEVICE="1"
output_frb190709_rfim/scenario/tuning_2.sh:## Name of OpenCL device, used for configuration files
output_frb190709_rfim/scenario/tuning_2.sh:## Size, in bytes, of the OpenCL device's cache line
output_frb190709_rfim/scenario/tuning_2.sh:## Number of OpenCL work-items running simultaneously
output_frb190709_rfim/scenario/tuning_2.sh:## Maximum number of work-items in OpenCL dimension 0; dedispersion specific
output_frb190709_rfim/scenario/tuning_2.sh:## Maximum number of work-items in OpenCL dimension 1; dedispersion specific
output_frb190709_rfim/scenario/tuning_2.sh:## Maximum number of variables in OpenCL dimension 0; dedispersion specific
output_frb190709_rfim/scenario/tuning_2.sh:## Maximum number of variables in OpenCL dimension 1; dedispersion specific
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:## OpenCL platform ID
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:OPENCL_PLATFORM="0"
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:## OpenCL device ID
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:OPENCL_DEVICE="2"
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:## Name of OpenCL device, used for configuration files
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:## Size, in bytes, of the OpenCL device's cache line
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:## Number of OpenCL work-items running simultaneously
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:## Maximum number of work-items in OpenCL dimension 0; dedispersion specific
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:## Maximum number of work-items in OpenCL dimension 1; dedispersion specific
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:## Maximum number of variables in OpenCL dimension 0; dedispersion specific
output_frb190709_rfim/scenario/tuning_3_nodownsamp.sh:## Maximum number of variables in OpenCL dimension 1; dedispersion specific
output_frb190709_rfim/scenario/tuning_3.sh:## OpenCL platform ID
output_frb190709_rfim/scenario/tuning_3.sh:OPENCL_PLATFORM="0"
output_frb190709_rfim/scenario/tuning_3.sh:## OpenCL device ID
output_frb190709_rfim/scenario/tuning_3.sh:OPENCL_DEVICE="2"
output_frb190709_rfim/scenario/tuning_3.sh:## Name of OpenCL device, used for configuration files
output_frb190709_rfim/scenario/tuning_3.sh:## Size, in bytes, of the OpenCL device's cache line
output_frb190709_rfim/scenario/tuning_3.sh:## Number of OpenCL work-items running simultaneously
output_frb190709_rfim/scenario/tuning_3.sh:## Maximum number of work-items in OpenCL dimension 0; dedispersion specific
output_frb190709_rfim/scenario/tuning_3.sh:## Maximum number of work-items in OpenCL dimension 1; dedispersion specific
output_frb190709_rfim/scenario/tuning_3.sh:## Maximum number of variables in OpenCL dimension 0; dedispersion specific
output_frb190709_rfim/scenario/tuning_3.sh:## Maximum number of variables in OpenCL dimension 1; dedispersion specific

```
