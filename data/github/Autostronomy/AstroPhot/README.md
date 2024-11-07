# https://github.com/Autostronomy/AstroPhot

```console
docs/source/configfile_interface.rst:                       be one of: cpu, gpu
CITATION.cff:  AstroPhot incorporates automatic differentiation and GPU
CITATION.cff:  or GPU; across images that are large, multi-band,
astrophot/fit/oldlm.py:            torch.cuda.empty_cache()
astrophot/fit/oldlm.py:            torch.cuda.empty_cache()
astrophot/fit/oldlm.py:    expression and performs far faster on GPU since no communication
astrophot/AP_config.py:ap_device = "cuda:0" if torch.cuda.is_available() else "cpu"
astrophot/__init__.py:    - `--device`: set the device for AstroPhot to use for computations. Must be one of: cpu, gpu.
astrophot/__init__.py:        choices=["cpu", "gpu"],
astrophot/__init__.py:        help="set the device for AstroPhot to use for computations. Must be one of: cpu, gpu",
astrophot/__init__.py:        AP_config.device = "cpu" if args.device == "cpu" else "cuda:0"
astrophot/image/image_object.py:        # # Check that image data and header are in agreement (this requires talk back from GPU to CPU so is only used for testing)

```
