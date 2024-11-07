# https://github.com/benlansdell/ethome

```console
tests/test_analysis.py:    os.environ["CUDA_VISIBLE_DEVICES"] = ""
tests/test_analysis.py:    os.environ["CUDA_VISIBLE_DEVICES"] = ""
Makefile:		CUDA_VISIBLE_DEVICES= pytest
Makefile:		CUDA_VISIBLE_DEVICES= coverage run -m pytest
Makefile:		CUDA_VISIBLE_DEVICES= python examples/sample_workflow.py
examples/sample_workflow.py:# More reliable to not use GPU here. It's only doing inference with a small net, doesn't take long
examples/sample_workflow.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""

```
