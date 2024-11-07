# https://github.com/PMBio/spatialde2-paper

```console
benchmarks/segmentation_speed_benchmark.py:            with tf.device("/device:GPU:0"):
benchmarks/segmentation_speed_benchmark.py:        spatialde_gpu_res = timeit.repeat(spde, setup="import tensorflow as tf", repeat=1, number=number)[0]
benchmarks/segmentation_speed_benchmark.py:                                    "time_seconds": spatialde_gpu_res / number,
benchmarks/segmentation_speed_benchmark.py:                                    "ncores": "gpu",
benchmarks/figure_1.Rmd:gpu <- readRDS("tests_runtime_gpu.rds") %>%
benchmarks/figure_1.Rmd:    mutate(cores="GPU")
benchmarks/figure_1.Rmd:speed_bench <- bind_rows(single, multi, gpu) %>%
benchmarks/figure_1.Rmd:mutate(clusterspeed, method=recode(method, leiden="Leiden", spatialde2="SpatialDE2"), cores=recode(ncores, `1`="1 core", `10`="10 cores", gpu="GPU")) %>%
benchmarks/spatially_variable_genes_benchmark_speed_gpu.R.R:saveRDS(times, paste0("tests_runtime_gpu.rds"))

```
