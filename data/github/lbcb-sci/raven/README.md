# https://github.com/lbcb-sci/raven

```console
RavenExe/src/main.cc:#ifdef CUDA_ENABLED
RavenExe/src/main.cc:    {"cuda-poa-batches", optional_argument, nullptr, 'c'},
RavenExe/src/main.cc:    {"cuda-banded-alignment", no_argument, nullptr, 'b'},
RavenExe/src/main.cc:    {"cuda-alignment-batches", required_argument, nullptr, 'a'},
RavenExe/src/main.cc:#ifdef CUDA_ENABLED
RavenExe/src/main.cc:         "    -c, --cuda-poa-batches <int>\n"
RavenExe/src/main.cc:         "      number of batches for CUDA accelerated polishing\n"
RavenExe/src/main.cc:         "    -b, --cuda-banded-alignment\n"
RavenExe/src/main.cc:         "      use banding approximation for polishing on GPU\n"
RavenExe/src/main.cc:         "    -a, --cuda-alignment-batches <int>\n"
RavenExe/src/main.cc:         "      number of batches for CUDA accelerated alignment\n"
RavenExe/src/main.cc:  std::uint32_t cuda_poa_batches = 0;
RavenExe/src/main.cc:  std::uint32_t cuda_alignment_batches = 0;
RavenExe/src/main.cc:  bool cuda_banded_alignment = false;
RavenExe/src/main.cc:#ifdef CUDA_ENABLED
RavenExe/src/main.cc:#ifdef CUDA_ENABLED
RavenExe/src/main.cc:        cuda_poa_batches = 1;
RavenExe/src/main.cc:          cuda_poa_batches = std::atoi(argv[optind++]);
RavenExe/src/main.cc:          cuda_poa_batches = std::atoi(optarg);
RavenExe/src/main.cc:        cuda_banded_alignment = true;
RavenExe/src/main.cc:        cuda_alignment_batches = std::atoi(optarg);
RavenExe/src/main.cc:          .cuda_cfg =
RavenExe/src/main.cc:              raven::CudaCfg{.poa_batches = cuda_poa_batches,
RavenExe/src/main.cc:                             .alignment_batches = cuda_alignment_batches,
RavenExe/src/main.cc:                             .banded_alignment = cuda_banded_alignment},
RavenExe/Exe.cmake:    if (racon_enable_cuda)
RavenExe/Exe.cmake:        target_compile_definitions(raven_exe PRIVATE CUDA_ENABLED)
README.md:  only available when built with CUDA:
README.md:    -c, --cuda-poa-batches <int>
README.md:      number of batches for CUDA accelerated polishing
README.md:    -b, --cuda-banded-alignment
README.md:      use banding approximation for polishing on GPU
README.md:    -a, --cuda-alignment-batches <int>
README.md:      number of batches for CUDA accelerated alignment
README.md:- `racon_enable_cuda`: build with NVIDIA CUDA support
PythonLib/example.py:        ravenpy.CudaCfg(0, 0, False),
PythonLib/src/ravenpy.cc:  py::class_<raven::CudaCfg>(m, "CudaCfg")
PythonLib/src/ravenpy.cc:      .def_readwrite("poa_batches", &raven::CudaCfg::poa_batches)
PythonLib/src/ravenpy.cc:      .def_readwrite("alignment_batches", &raven::CudaCfg::alignment_batches)
PythonLib/src/ravenpy.cc:      .def_readwrite("banded_alignment", &raven::CudaCfg::banded_alignment);
PythonLib/src/ravenpy.cc:      .def(py::init<raven::AlignCfg, raven::CudaCfg, std::uint32_t>())
PythonLib/src/ravenpy.cc:      .def_readwrite("cuda_cfg", &raven::PolishCfg::cuda_cfg)
RavenLib/include/raven/graph/polish.hpp:struct CudaCfg {
RavenLib/include/raven/graph/polish.hpp:  CudaCfg cuda_cfg;
RavenLib/src/polish.cc:      cfg.cuda_cfg.poa_batches, cfg.cuda_cfg.banded_alignment,
RavenLib/src/polish.cc:      cfg.cuda_cfg.alignment_batches);

```
