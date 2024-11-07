# https://github.com/scverse/scanpy

```console
docs/news.md:### `rapids-singlecell` brings scanpy to the GPU! {small}`2024-03-18`
docs/news.md:{doc}`rapids-singlecell <rapids_singlecell:index>` by Severin Dicks provides a scanpy-like API with accelerated operations implemented on GPU.
docs/release-notes/1.4.5.md:- run neighbors on a GPU using rapids {pr}`830` {smaller}`T White`
pyproject.toml:rapids = ["cudf>=0.9", "cuml>=0.9", "cugraph>=0.9"]  # GPU accelerated calculation of neighbors
pyproject.toml:    "gpu: tests that use a GPU (currently unused, but needs to be specified here as we import anndata.tests.helpers, which uses it)",
src/scanpy/tools/_louvain.py:            GPU accelerated implementation.
src/scanpy/tools/_umap.py:            GPU accelerated implementation.

```
