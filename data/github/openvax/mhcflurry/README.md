# https://github.com/openvax/mhcflurry

```console
downloads-generation/models_class1_processing/make_train_data.py:    worker_pool_with_gpu_assignments_from_args,
downloads-generation/models_class1_processing/make_train_data.py:        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
downloads-generation/models_class1_processing/cluster_submit_script_header.mssm_hpc.lsf:#BSUB -q gpu # queue
downloads-generation/models_class1_processing/cluster_submit_script_header.mssm_hpc.lsf:#BSUB -gpu "num=1:mode=exclusive_process:mps=no:j_exclusive=yes"
downloads-generation/models_class1_processing/cluster_submit_script_header.mssm_hpc.lsf:#export TF_GPU_ALLOCATOR=cuda_malloc_async
downloads-generation/models_class1_processing/cluster_submit_script_header.mssm_hpc.lsf:module add cuda/11.8.0 cudnn/8.9.5-11
downloads-generation/models_class1_processing/cluster_submit_script_header.mssm_hpc.lsf:nvidia-smi
downloads-generation/models_class1_processing/cluster_submit_script_header.mssm_hpc.lsf:export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH
downloads-generation/models_class1_processing/cluster_submit_script_header.mssm_hpc.lsf:python -c 'import tensorflow as tf ; print("GPU AVAILABLE" if tf.test.is_gpu_available() else "GPU NOT AVAILABLE")'
downloads-generation/models_class1_processing/GENERATE.sh:    GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/models_class1_processing/GENERATE.sh:    echo "Detected GPUS: $GPUS"
downloads-generation/models_class1_processing/GENERATE.sh:    if [ "$GPUS" -eq "0" ]; then
downloads-generation/models_class1_processing/GENERATE.sh:        NUM_JOBS=${NUM_JOBS-$GPUS}
downloads-generation/models_class1_processing/GENERATE.sh:    PARALLELISM_ARGS+=" --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1"
downloads-generation/data_predictions/GENERATE.sh:    GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/data_predictions/GENERATE.sh:    echo "Detected GPUS: $GPUS"
downloads-generation/data_predictions/GENERATE.sh:    if [ "$GPUS" -eq "0" ]; then
downloads-generation/data_predictions/GENERATE.sh:        NUM_JOBS=${NUM_JOBS-$GPUS}
downloads-generation/data_predictions/GENERATE.sh:    EXTRA_ARGS+=" --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1"
downloads-generation/data_predictions/GENERATE.sh:        --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.nogpu.lsf \
downloads-generation/data_predictions/GENERATE.sh:            --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.nogpu.lsf \
downloads-generation/data_predictions/GENERATE.sh:    #        --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.gpu.lsf \
downloads-generation/data_predictions/cluster_submit_script_header.mssm_hpc.gpu.lsf:#BSUB -q gpu # queue
downloads-generation/data_predictions/cluster_submit_script_header.mssm_hpc.gpu.lsf:#BSUB -R rusage[ngpus_excl_p=1]  # 1 exclusive GPU
downloads-generation/data_predictions/cluster_submit_script_header.mssm_hpc.gpu.lsf:module add cuda/10.0.130 cudnn/7.1.1
downloads-generation/data_predictions/cluster_submit_script_header.mssm_hpc.gpu.lsf:# python -c 'import tensorflow as tf ; print("GPU AVAILABLE" if tf.test.is_gpu_available() else "GPU NOT AVAILABLE")'
downloads-generation/data_predictions/run_predictors.py:    worker_pool_with_gpu_assignments_from_args,
downloads-generation/data_predictions/run_predictors.py:        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
downloads-generation/models_class1_kim_benchmark/GENERATE.sh:GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/models_class1_kim_benchmark/GENERATE.sh:echo "Detected GPUS: $GPUS"
downloads-generation/models_class1_kim_benchmark/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50
downloads-generation/models_class1_kim_benchmark/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 5
downloads-generation/models_class1_kim_benchmark/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50
downloads-generation/models_class1_pan/cluster_submit_script_header.mssm_hpc.lsf:#BSUB -q gpu # queue
downloads-generation/models_class1_pan/cluster_submit_script_header.mssm_hpc.lsf:#BSUB -gpu "num=1:mode=exclusive_process:mps=no:j_exclusive=yes"
downloads-generation/models_class1_pan/cluster_submit_script_header.mssm_hpc.lsf:#export TF_GPU_ALLOCATOR=cuda_malloc_async
downloads-generation/models_class1_pan/cluster_submit_script_header.mssm_hpc.lsf:module add cuda/11.8.0 cudnn/8.9.5-11
downloads-generation/models_class1_pan/cluster_submit_script_header.mssm_hpc.lsf:nvidia-smi
downloads-generation/models_class1_pan/cluster_submit_script_header.mssm_hpc.lsf:export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH
downloads-generation/models_class1_pan/cluster_submit_script_header.mssm_hpc.lsf:python -c 'import tensorflow as tf ; print("GPU AVAILABLE" if tf.test.is_gpu_available() else "GPU NOT AVAILABLE")'
downloads-generation/models_class1_pan/GENERATE.sh:    GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/models_class1_pan/GENERATE.sh:    echo "Detected GPUS: $GPUS"
downloads-generation/models_class1_pan/GENERATE.sh:    if [ "$GPUS" -eq "0" ]; then
downloads-generation/models_class1_pan/GENERATE.sh:        NUM_JOBS=${NUM_JOBS-$GPUS}
downloads-generation/models_class1_pan/GENERATE.sh:    PARALLELISM_ARGS+=" --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1"
downloads-generation/models_class1_unselected/GENERATE.sh:GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/models_class1_unselected/GENERATE.sh:echo "Detected GPUS: $GPUS"
downloads-generation/models_class1_unselected/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50
downloads-generation/models_class1_minimal/GENERATE.sh:GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/models_class1_minimal/GENERATE.sh:echo "Detected GPUS: $GPUS"
downloads-generation/models_class1_minimal/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 1
downloads-generation/models_class1_minimal/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50
downloads-generation/models_class1/GENERATE.sh:GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/models_class1/GENERATE.sh:echo "Detected GPUS: $GPUS"
downloads-generation/models_class1/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 1
downloads-generation/models_class1/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50
downloads-generation/models_class1_pan_variants/GENERATE.sh:    GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/models_class1_pan_variants/GENERATE.sh:    echo "Detected GPUS: $GPUS"
downloads-generation/models_class1_pan_variants/GENERATE.sh:    if [ "$GPUS" -eq "0" ]; then
downloads-generation/models_class1_pan_variants/GENERATE.sh:        NUM_JOBS=${NUM_JOBS-$GPUS}
downloads-generation/models_class1_pan_variants/GENERATE.sh:    PARALLELISM_ARGS+=" --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1"
downloads-generation/models_class1_pan_variants/GENERATE.sh:    PARALLELISM_ARGS+=" --cluster-parallelism --cluster-max-retries 3 --cluster-submit-command bsub --cluster-results-workdir $HOME/mhcflurry-scratch --cluster-script-prefix-path $SCRIPT_DIR/cluster_submit_script_header.mssm_hpc.gpu.lsf"
downloads-generation/models_class1_pan_variants/cluster_submit_script_header.mssm_hpc.gpu.lsf:#BSUB -q gpu # queue
downloads-generation/models_class1_pan_variants/cluster_submit_script_header.mssm_hpc.gpu.lsf:#BSUB -R rusage[ngpus_excl_p=1]  # 1 exclusive GPU
downloads-generation/models_class1_pan_variants/cluster_submit_script_header.mssm_hpc.gpu.lsf:module add cuda/10.1.105 cudnn/7.6.5
downloads-generation/models_class1_pan_variants/cluster_submit_script_header.mssm_hpc.gpu.lsf:# python -c 'import tensorflow as tf ; print("GPU AVAILABLE" if tf.test.is_gpu_available() else "GPU NOT AVAILABLE")'
downloads-generation/models_class1_trained_with_mass_spec/GENERATE.sh:GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/models_class1_trained_with_mass_spec/GENERATE.sh:echo "Detected GPUS: $GPUS"
downloads-generation/models_class1_trained_with_mass_spec/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 1
downloads-generation/models_class1_trained_with_mass_spec/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50
downloads-generation/models_class1_selected_no_mass_spec/GENERATE.sh:GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/models_class1_selected_no_mass_spec/GENERATE.sh:echo "Detected GPUS: $GPUS"
downloads-generation/models_class1_selected_no_mass_spec/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 5
downloads-generation/models_class1_selected_no_mass_spec/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50
downloads-generation/analysis_predictor_info/cluster_submit_script_header.mssm_hpc.lsf:#BSUB -q gpu # queue
downloads-generation/analysis_predictor_info/cluster_submit_script_header.mssm_hpc.lsf:#BSUB -R rusage[ngpus_excl_p=1]  # 1 exclusive GPU
downloads-generation/analysis_predictor_info/cluster_submit_script_header.mssm_hpc.lsf:module add cuda/10.0.130
downloads-generation/analysis_predictor_info/cluster_submit_script_header.mssm_hpc.lsf:export CUDNN_HOME=/hpc/users/odonnt02/oss/cudnn/cuda
downloads-generation/analysis_predictor_info/cluster_submit_script_header.mssm_hpc.lsf:python -c 'import tensorflow as tf ; print("GPU AVAILABLE" if tf.test.is_gpu_available() else "GPU NOT AVAILABLE")'
downloads-generation/analysis_predictor_info/generate_artifacts.py:    worker_pool_with_gpu_assignments_from_args,
downloads-generation/analysis_predictor_info/generate_artifacts.py:        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
downloads-generation/analysis_predictor_info/predict_on_model_selection_data.py:    worker_pool_with_gpu_assignments_from_args,
downloads-generation/analysis_predictor_info/predict_on_model_selection_data.py:        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
downloads-generation/analysis_predictor_info/GENERATE.sh:    GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/analysis_predictor_info/GENERATE.sh:    echo "Detected GPUS: $GPUS"
downloads-generation/analysis_predictor_info/GENERATE.sh:    if [ "$GPUS" -eq "0" ]; then
downloads-generation/analysis_predictor_info/GENERATE.sh:        NUM_JOBS=${NUM_JOBS-$GPUS}
downloads-generation/analysis_predictor_info/GENERATE.sh:    PARALLELISM_ARGS+=" --num-jobs $NUM_JOBS --max-tasks-per-worker 1 --gpus $GPUS --max-workers-per-gpu 1"
downloads-generation/models_class1_unselected_with_mass_spec/GENERATE.sh:GPUS=$(nvidia-smi -L 2> /dev/null | wc -l) || GPUS=0
downloads-generation/models_class1_unselected_with_mass_spec/GENERATE.sh:echo "Detected GPUS: $GPUS"
downloads-generation/models_class1_unselected_with_mass_spec/GENERATE.sh:    --num-jobs $(expr $PROCESSORS \* 2) --gpus $GPUS --max-workers-per-gpu 2 --max-tasks-per-worker 50
docs/intro.rst:neural network library via either the Tensorflow or Theano backends. GPUs may
docs/commandline_tutorial.rst:    dozen GPUs over a period of about two days. If you model select over fewer
test/test_train_and_related_commands.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
test/test_train_pan_allele_models_command.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
test/test_doctest.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
test/test_calibrate_percentile_ranks_command.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
test/test_train_processing_models_command.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
mhcflurry/select_allele_specific_models_command.py:from .local_parallelism import worker_pool_with_gpu_assignments_from_args, add_local_parallelism_args
mhcflurry/select_allele_specific_models_command.py:    worker_pool = worker_pool_with_gpu_assignments_from_args(args)
mhcflurry/train_allele_specific_models_command.py:    worker_pool_with_gpu_assignments_from_args,
mhcflurry/train_allele_specific_models_command.py:    worker_pool = worker_pool_with_gpu_assignments_from_args(args)
mhcflurry/select_pan_allele_models_command.py:    worker_pool_with_gpu_assignments_from_args,
mhcflurry/select_pan_allele_models_command.py:        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
mhcflurry/select_processing_models_command.py:    worker_pool_with_gpu_assignments_from_args,
mhcflurry/select_processing_models_command.py:        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
mhcflurry/train_processing_models_command.py:    worker_pool_with_gpu_assignments_from_args,
mhcflurry/train_processing_models_command.py:        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
mhcflurry/calibrate_percentile_ranks_command.py:    worker_pool_with_gpu_assignments_from_args,
mhcflurry/calibrate_percentile_ranks_command.py:        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
mhcflurry/train_pan_allele_models_command.py:    worker_pool_with_gpu_assignments_from_args,
mhcflurry/train_pan_allele_models_command.py:        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
mhcflurry/train_pan_allele_models_command.py:        if tf.config.list_physical_devices('GPU'):
mhcflurry/train_pan_allele_models_command.py:            mem = tf.config.experimental.get_memory_info('GPU:0')['current'] / 10**9
mhcflurry/train_pan_allele_models_command.py:            print("Current used GPU memory: ", mem, "gb")
mhcflurry/local_parallelism.py:        choices=("tensorflow-gpu", "tensorflow-cpu", "tensorflow-default"),
mhcflurry/local_parallelism.py:        "--gpus",
mhcflurry/local_parallelism.py:        help="Number of GPUs to attempt to parallelize across. Requires running "
mhcflurry/local_parallelism.py:        "--max-workers-per-gpu",
mhcflurry/local_parallelism.py:        help="Maximum number of workers to assign to a GPU. Additional tasks will "
mhcflurry/local_parallelism.py:def worker_pool_with_gpu_assignments_from_args(args):
mhcflurry/local_parallelism.py:    Create a multiprocessing.Pool where each worker uses its own GPU.
mhcflurry/local_parallelism.py:    Uses commandline arguments. See `worker_pool_with_gpu_assignments`.
mhcflurry/local_parallelism.py:    return worker_pool_with_gpu_assignments(
mhcflurry/local_parallelism.py:        num_gpus=args.gpus,
mhcflurry/local_parallelism.py:        max_workers_per_gpu=args.max_workers_per_gpu,
mhcflurry/local_parallelism.py:def worker_pool_with_gpu_assignments(
mhcflurry/local_parallelism.py:        num_gpus=0,
mhcflurry/local_parallelism.py:        max_workers_per_gpu=1,
mhcflurry/local_parallelism.py:    Create a multiprocessing.Pool where each worker uses its own GPU.
mhcflurry/local_parallelism.py:    num_gpus : int
mhcflurry/local_parallelism.py:    max_workers_per_gpu : int
mhcflurry/local_parallelism.py:    if num_gpus:
mhcflurry/local_parallelism.py:        print("Attempting to round-robin assign each worker a GPU.")
mhcflurry/local_parallelism.py:        gpu_assignments_remaining = dict((
mhcflurry/local_parallelism.py:            (gpu, max_workers_per_gpu) for gpu in range(num_gpus)
mhcflurry/local_parallelism.py:            if gpu_assignments_remaining:
mhcflurry/local_parallelism.py:                # Use a GPU
mhcflurry/local_parallelism.py:                gpu_num = sorted(
mhcflurry/local_parallelism.py:                    gpu_assignments_remaining,
mhcflurry/local_parallelism.py:                    key=lambda key: gpu_assignments_remaining[key])[0]
mhcflurry/local_parallelism.py:                gpu_assignments_remaining[gpu_num] -= 1
mhcflurry/local_parallelism.py:                if not gpu_assignments_remaining[gpu_num]:
mhcflurry/local_parallelism.py:                    del gpu_assignments_remaining[gpu_num]
mhcflurry/local_parallelism.py:                gpu_assignment = [gpu_num]
mhcflurry/local_parallelism.py:                gpu_assignment = []
mhcflurry/local_parallelism.py:                'gpu_device_nums': gpu_assignment,
mhcflurry/local_parallelism.py:            print("Worker %d assigned GPUs: %s" % (
mhcflurry/local_parallelism.py:                worker_num, gpu_assignment))
mhcflurry/local_parallelism.py:    this feature is to support allocating each worker to a (different) GPU.
mhcflurry/local_parallelism.py:def worker_init(keras_backend=None, gpu_device_nums=None, worker_log_dir=None):
mhcflurry/local_parallelism.py:    if keras_backend or gpu_device_nums:
mhcflurry/local_parallelism.py:        print("WORKER pid=%d assigned GPU devices: %s" % (
mhcflurry/local_parallelism.py:            os.getpid(), gpu_device_nums))
mhcflurry/local_parallelism.py:            keras_backend, gpu_device_nums=gpu_device_nums)
mhcflurry/common.py:def configure_tensorflow(backend=None, gpu_device_nums=None, num_threads=None):
mhcflurry/common.py:    Configure Keras backend to use GPU or CPU.
mhcflurry/common.py:        one of 'tensorflow-default', 'tensorflow-cpu', 'tensorflow-gpu'
mhcflurry/common.py:    gpu_device_nums : list of int, optional
mhcflurry/common.py:        GPU devices to potentially use
mhcflurry/common.py:    # turn on selected GPUs with memory growth enabled
mhcflurry/common.py:    if gpu_device_nums is not None:
mhcflurry/common.py:        physical_devices = tf.config.list_physical_devices("GPU")
mhcflurry/common.py:            [physical_devices[idx] for idx in gpu_device_nums], "GPU"
mhcflurry/common.py:        for gpu in physical_devices:
mhcflurry/common.py:            tf.config.experimental.set_memory_growth(gpu, True)

```
