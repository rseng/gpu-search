# https://github.com/RichardObi/medigan

```console
config/global.json:                        "gpu_id": 0
config/global.json:                "gpu_id: default=0, help=the gpu to run the model on.",
config/global.json:                        "gpu_id": 0
config/global.json:                "gpu_id: default=0 help=the gpu to run the model."
config/global.json:                        "gpu_id": null,
config/global.json:                "gpu_id: type=int, default=None, help=0 is the first gpu, 1 is the second gpu, etc.",
config/global.json:                        "gpu_id": null
config/global.json:                "gpu_id: type=int, default=None, help=0 is the first gpu, 1 is the second gpu, etc.",
config/global.json:                        "gpu_id": 0,
config/global.json:                "gpu_id: default=0, help=the gpu to run the model on.",
config/global.json:                        "gpu_id": 0,
config/global.json:                "gpu_id: default=0, help=the gpu to run the model on.",
config/global.json:                        "gpu_id": 0,
config/global.json:                "gpu_id: default=0, help=the gpu to run the model on.",
config/global.json:                        "gpu_id": 0,
config/global.json:                "gpu_id: default=0, help=the gpu to run the model on.",
config/global.json:                        "gpu_id": "0",
config/global.json:                "gpu_id: default=0, help=the gpu to run the model on.",
config/global.json:                     "gpu_id": "0"
config/global.json:                "gpu_id: default=0, help=the gpu to run the model on."
docs/source/model_documentation.md:    gpu_id=0,
docs/source/model_documentation.md:    "gpu_id: default=0, help=the gpu to run the model on.",
docs/source/model_documentation.md:    gpu_id=0,
docs/source/model_documentation.md:    "gpu_id: default=0 help=the gpu to run the model.",
docs/source/model_documentation.md:    gpu_id=None, 
docs/source/model_documentation.md:    "gpu_id: type=int, default=None, help=0 is the first gpu, 1 is the second gpu, etc.",
docs/source/model_documentation.md:    gpu_id=None, 
docs/source/model_documentation.md:    "gpu_id: type=int, default=None, help=0 is the first gpu, 1 is the second gpu, etc.",
docs/source/model_documentation.md:    gpu_id=0, 
docs/source/model_documentation.md:    "gpu_id: default=0, help=the gpu to run the model on.",
docs/source/model_documentation.md:    gpu_id=0, 
docs/source/model_documentation.md:    "gpu_id: default=0, help=the gpu to run the model on.",
docs/source/model_documentation.md:    gpu_id=0, 
docs/source/model_documentation.md:    "gpu_id: default=0, help=the gpu to run the model on.",
docs/source/model_documentation.md:    gpu_id=0, 
docs/source/model_documentation.md:    "gpu_id: default=0, help=the gpu to run the model on.",
docs/source/model_documentation.md:    gpu_id=0,
docs/source/model_documentation.md:    "gpu_id: default=0, help=the gpu to run the model on.",
docs/source/model_documentation.md:    gpu_id=0,
docs/source/model_documentation.md:    "gpu_id: default=0, help=the gpu to run the model on.",
docs/source/adding_models.rst:        #. Please make sure your model is able to run both on gpu and on cpu and your code automatically detects on which one to run.
docs/source/adding_models.rst:                - ``gpu_id``: int, if a user has various GPUs available, the user can specify which one of them to use to run your generative model.
docs/source/adding_models.rst:* Also, the model should do simple error handling, run flexibly on either gpu or cpu, use ``logging`` instead of ``prints``, and create some sort of synthetic data.
tests/model_integration_test_manual.py:    gpu_id=0,
tests/model_integration_test_manual.py:    gpu_id=0,
tests/test_model_executor.py:            "gpu_id": 0,
tests/test_model_executor.py:            "gpu_id": 0,
templates/examples/__init__.py:        ngpu: int,
templates/examples/__init__.py:        self.ngpu = ngpu
templates/examples/__init__.py:        ngpu: int,
templates/examples/__init__.py:            ngpu=ngpu,
templates/examples/__init__.py:def image_generator(model_path, device, nz, ngf, nc, ngpu, num_samples):
templates/examples/__init__.py:        ngpu=ngpu,
templates/examples/__init__.py:    if device.type == "cuda":
templates/examples/__init__.py:        netG.cuda()
templates/examples/__init__.py:        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
templates/examples/__init__.py:        ngpu = 0
templates/examples/__init__.py:        if device == "cuda":
templates/examples/__init__.py:            ngpu = 1
templates/examples/__init__.py:        image_list = image_generator(model_file, device, 100, 64, 1, ngpu, num_samples)
src/medigan/generators.py:                into CUDA pinned memory before returning them.  If your data elements

```
