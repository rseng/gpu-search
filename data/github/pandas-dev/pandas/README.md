# https://github.com/pandas-dev/pandas

```console
web/pandas/config.yml:  - name: "Nvidia"
web/pandas/config.yml:    url: https://www.nvidia.com
web/pandas/config.yml:    logo: static/img/partners/nvidia.svg
web/pandas/community/blog/extension-arrays.md:nested data, data with units, geo data, GPU arrays. Keep an eye on the
doc/source/user_guide/enhancingperf.rst:Numba supports compilation of Python to run on either CPU or GPU hardware and is designed to integrate with the Python scientific software stack.
pandas/core/interchange/dataframe_protocol.py:    CUDA = 2
pandas/core/interchange/dataframe_protocol.py:    OPENCL = 4
pandas/core/interchange/dataframe_protocol.py:    ROCM = 10
pandas/tests/io/sas/data/DRXFCD_G.csv:26103110,"BARRACUDA, COOKED, NS AS TO COOKING METHOD","Barracuda, cooked, NS as to cooking method"
pandas/tests/io/sas/data/DRXFCD_G.csv:26103120,"BARRACUDA, BAKED OR BROILED, FAT ADDED IN COOKING","Barracuda, baked or broiled, fat added in cooking"
pandas/tests/io/sas/data/DRXFCD_G.csv:26103121,"BARRACUDA, BAKED OR BROILED, FAT NOT ADDED IN COOKING","Barracuda, baked or broiled, fat not added in cooking"
pandas/tests/io/sas/data/DRXFCD_G.csv:26103130,"BARRACUDA, COATED, BAKED OR BROILED, FAT ADDED IN COOKING","Barracuda, coated, baked or broiled, fat added in cooking"
pandas/tests/io/sas/data/DRXFCD_G.csv:26103131,"BARRACUDA, COATED, BAKED OR BROILED, FAT NOT ADDED IN COOKIN","Barracuda, coated, baked or broiled, fat not added in cooking"
pandas/tests/io/sas/data/DRXFCD_G.csv:26103140,"BARRACUDA, COATED, FRIED","Barracuda, coated, fried"
pandas/tests/io/sas/data/DRXFCD_G.csv:26103160,"BARRACUDA, STEAMED OR POACHED","Barracuda, steamed or poached"
pandas/io/clipboard/__init__.py:    OpenClipboard = windll.user32.OpenClipboard
pandas/io/clipboard/__init__.py:    OpenClipboard.argtypes = [HWND]
pandas/io/clipboard/__init__.py:    OpenClipboard.restype = BOOL
pandas/io/clipboard/__init__.py:            success = OpenClipboard(hwnd)
pandas/io/clipboard/__init__.py:            raise PyperclipWindowsException("Error calling OpenClipboard")
pandas/io/clipboard/__init__.py:            # If an application calls OpenClipboard with hwnd set to NULL,
typings/numba.pyi:    target: Literal["cpu", "gpu", "npyufunc", "cuda"] = ...,  # deprecated

```
