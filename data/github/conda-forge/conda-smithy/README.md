# https://github.com/conda-forge/conda-smithy

```console
tests/test_variant_algebra.py:                - cuda_compiler_version     # ["linux-64"]
tests/test_variant_algebra.py:                - cuda_compiler_version     # ["linux-64"]
tests/test_configure_feedstock.py:def test_cuda_enabled_render(cuda_enabled_recipe, jinja_env):
tests/test_configure_feedstock.py:    forge_config = copy.deepcopy(cuda_enabled_recipe.config)
tests/test_configure_feedstock.py:    has_env = "CF_CUDA_ENABLED" in os.environ
tests/test_configure_feedstock.py:        old_val = os.environ["CF_CUDA_ENABLED"]
tests/test_configure_feedstock.py:        del os.environ["CF_CUDA_ENABLED"]
tests/test_configure_feedstock.py:        assert "CF_CUDA_ENABLED" not in os.environ
tests/test_configure_feedstock.py:            forge_dir=cuda_enabled_recipe.recipe,
tests/test_configure_feedstock.py:        assert os.environ["CF_CUDA_ENABLED"] == "True"
tests/test_configure_feedstock.py:        matrix_dir = os.path.join(cuda_enabled_recipe.recipe, ".ci_support")
tests/test_configure_feedstock.py:            os.environ["CF_CUDA_ENABLED"] = old_val
tests/test_configure_feedstock.py:            if "CF_CUDA_ENABLED" in os.environ:
tests/test_configure_feedstock.py:                del os.environ["CF_CUDA_ENABLED"]
tests/test_configure_feedstock.py:                    ("cuda_compiler_version_min", ["11.2"]),
tests/test_configure_feedstock.py:                    ("nccl", ["2"]),
tests/test_configure_feedstock.py:                            "quay.io/condaforge/linux-anvil-cuda:11.8",
tests/test_configure_feedstock.py:                            "quay.io/condaforge/linux-anvil-cuda:11.2",
tests/test_configure_feedstock.py:                    ("cuda_compiler", ("cuda-nvcc", "nvcc", "nvcc", "None")),
tests/test_configure_feedstock.py:                        "cuda_compiler_version",
tests/test_configure_feedstock.py:                                "cuda_compiler",
tests/test_configure_feedstock.py:                                "cuda_compiler_version",
tests/test_configure_feedstock.py:                    ("cuda_compiler_version_min", ["11.2"]),
tests/test_configure_feedstock.py:                    ("nccl", ["2"]),
tests/test_configure_feedstock.py:                            "quay.io/condaforge/linux-anvil-cuda:11.2",
tests/test_configure_feedstock.py:                    ("cuda_compiler", ("nvcc", "None")),
tests/test_configure_feedstock.py:                    ("cuda_compiler_version", ("11.2", "None")),
tests/test_configure_feedstock.py:                                "cuda_compiler",
tests/test_configure_feedstock.py:                                "cuda_compiler_version",
tests/test_configure_feedstock.py:                "cuda_compiler",
tests/test_configure_feedstock.py:                "cuda_compiler_version",
tests/test_configure_feedstock.py:                "cuda_compiler": ["nvcc", "None"],
tests/test_configure_feedstock.py:                "cuda_compiler_version": ["11.2", "None"],
tests/test_configure_feedstock.py:                    "quay.io/condaforge/linux-anvil-cuda:11.2",
tests/test_configure_feedstock.py:                        "cuda_compiler",
tests/test_configure_feedstock.py:                        "cuda_compiler_version",
tests/test_cli.py:def test_init_cuda_docker_images(testing_workdir):
tests/test_cli.py:    recipe = os.path.join(_thisdir, "recipes", "cuda_docker_images")
tests/test_cli.py:        testing_workdir, "cuda_docker_images-feedstock"
tests/test_cli.py:            matrix_dir, f"linux_64_cuda_compiler_version{v}.yaml"
tests/test_cli.py:        assert config["cuda_compiler"] == ["nvcc"]
tests/test_cli.py:        assert config["cuda_compiler_version"] == [f"{v}"]
tests/test_cli.py:            docker_image = f"condaforge/linux-anvil-cuda:{v}"
tests/recipes/cuda_docker_images/meta.yaml:  name: test_cuda_docker_images
tests/recipes/cuda_docker_images/meta.yaml:    - {{ compiler("cuda") }}  # [cuda_compiler_version != "None"]
tests/recipes/cuda_docker_images/conda_build_config.yaml:cuda_compiler:                          # [linux64]
tests/recipes/cuda_docker_images/conda_build_config.yaml:cuda_compiler_version:                  # [linux64]
tests/recipes/cuda_docker_images/conda_build_config.yaml:  - condaforge/linux-anvil-cuda:9.2     # [linux64]
tests/recipes/cuda_docker_images/conda_build_config.yaml:  - condaforge/linux-anvil-cuda:10.0    # [linux64]
tests/recipes/cuda_docker_images/conda_build_config.yaml:  - condaforge/linux-anvil-cuda:10.1    # [linux64]
tests/recipes/cuda_docker_images/conda_build_config.yaml:  - condaforge/linux-anvil-cuda:10.2    # [linux64]
tests/recipes/cuda_docker_images/conda_build_config.yaml:  - condaforge/linux-anvil-cuda:11.0    # [linux64]
tests/recipes/cuda_docker_images/conda_build_config.yaml:    - cuda_compiler_version             # [linux64]
tests/recipes/cuda_recipes/meta.yaml:    skip: True   # [os.environ.get("CF_CUDA_ENABLED") != "True"]
tests/recipes/cuda_recipes/meta.yaml:        - {{ compiler('cuda') }}
tests/recipes/cuda_recipes/recipe.yaml:      - env.get('CF_CUDA_ENABLED') != "True"
tests/recipes/cuda_recipes/recipe.yaml:        - ${{ compiler('cuda') }}
tests/conftest.py:def cuda_enabled_recipe(config_yaml: ConfigYAML):
tests/conftest.py:        cuda_recipe_path = os.path.abspath(
tests/conftest.py:                "cuda_recipes",
tests/conftest.py:        content = Path(cuda_recipe_path).read_text()
.authors.yml:  - leof@nvidia.com
conda_smithy/data/conda-forge.json:          "description": "List of sources to find new versions (i.e. the strings like 1.2.3) for the package.\n\nThe following sources are available:\n- `cran`: Update from CRAN\n- `github`: Update from the GitHub releases RSS feed (includes pre-releases)\n- `githubreleases`: Get the latest version by following the redirect of\n`https://github.com/{owner}/{repo}/releases/latest` (excludes pre-releases)\n- `incrementalpharawurl`: If this source is run for a specific small selection of feedstocks, it acts like\nthe `rawurl` source but also increments letters in the version string (e.g. 2024a -> 2024b). If the source\nis run for other feedstocks (even if selected manually), it does nothing.\n- `librariesio`: Update from Libraries.io RSS feed\n- `npm`: Update from the npm registry\n- `nvidia`: Update from the NVIDIA download page\n- `pypi`: Update from the PyPI registry\n- `rawurl`: Update from a raw URL by trying to bump the version number in different ways and\nchecking if the URL exists (e.g. 1.2.3 -> 1.2.4, 1.3.0, 2.0.0, etc.)\n- `rosdistro`: Update from a ROS distribution\n\nCommon issues:\n- If you are using a GitHub-based source in your recipe and the bot issues PRs for pre-releases, restrict\nthe sources to `githubreleases` to avoid pre-releases.\n- If you use source tarballs that are uploaded manually by the maintainers a significant time after a\nGitHub release, you may want to restrict the sources to `rawurl` to avoid the bot attempting to update\nthe recipe before the tarball is uploaded.",
conda_smithy/data/conda-forge.json:        "nvidia",
conda_smithy/data/conda-forge.json:      "description": "This parameter allows for conda-smithy to run chocoloatey installs on Windows\nwhen additional system packages are needed. This is a list of strings that\nrepresent package names and any additional parameters. For example,\n\n```yaml\nchoco:\n    # install a package\n    - nvidia-display-driver\n\n    # install a package with a specific version\n    - cuda --version=11.0.3\n```\n\nThis is currently only implemented for Azure Pipelines. The command that is run is\n`choco install {entry} -fdv -y --debug`.  That is, `choco install` is executed\nwith a standard set of additional flags that are useful on CI.",
conda_smithy/configure_feedstock.py:        # detect if `compiler('cuda')` is used in meta.yaml,
conda_smithy/configure_feedstock.py:        # looking for `compiler('cuda')` with both quote variants;
conda_smithy/configure_feedstock.py:        pat = re.compile(r"^[^\#]*compiler\((\"cuda\"|\'cuda\')\).*")
conda_smithy/configure_feedstock.py:                os.environ["CF_CUDA_ENABLED"] = "True"
conda_smithy/configure_feedstock.py:        data["gha_with_gpu"] = False
conda_smithy/configure_feedstock.py:                if "gpu" in label.lower():
conda_smithy/configure_feedstock.py:                    data["gha_with_gpu"] = True
conda_smithy/schema.py:    NVIDIA = "nvidia"
conda_smithy/schema.py:            - `nvidia`: Update from the NVIDIA download page
conda_smithy/schema.py:            - nvidia-display-driver
conda_smithy/schema.py:            - cuda --version=11.0.3
conda_smithy/templates/github-actions.yml.tmpl:          {%- if data.gha_with_gpu %}
conda_smithy/templates/github-actions.yml.tmpl:            CONDA_FORGE_DOCKER_RUN_ARGS: "--gpus all"
.mailmap:Leo Fang <leofang@bnl.gov> Leo Fang <leof@nvidia.com>
CHANGELOG.rst:* When a label is added that has the string with `gpu` or `GPU` for a self-hosted
CHANGELOG.rst:  runner, the docker build will pass the GPUs to the docker instance.
CHANGELOG.rst:* Conda smithy will now detect if a recipe uses ``compiler('cuda')``
CHANGELOG.rst:and set the ``CF_CUDA_ENABLED`` environment variable to ``True`` if
CHANGELOG.rst:for builds with or without GPUs in ``conda_build_config.yaml``.

```
