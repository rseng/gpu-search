# https://github.com/HKU-BAL/Clair3

```console
docs/full_alignment_training.md:- A high-end GPU (have tested in RTX Titan, and RTX 2080Ti)
docs/full_alignment_training.md:# A single GPU is used for model training
docs/full_alignment_training.md:export CUDA_VISIBLE_DEVICES="0"
docs/full_alignment_training.md:export CUDA_VISIBLE_DEVICES="0"
docs/full_alignment_training_r1.md:- A high-end GPU (have tested on RTX Titan, and RTX 2080Ti)
docs/full_alignment_training_r1.md:# A single GPU is used for model training
docs/full_alignment_training_r1.md:export CUDA_VISIBLE_DEVICES="0"
docs/full_alignment_training_r1.md:export CUDA_VISIBLE_DEVICES="0"
docs/pileup_training.md:- A high-end GPU (have tested in RTX Titan, RTX 2080Ti, and GTX 1080Ti)
docs/pileup_training.md:# A single GPU is used for model training
docs/pileup_training.md:export CUDA_VISIBLE_DEVICES="0"
docs/pileup_training.md:export CUDA_VISIBLE_DEVICES="0"
scripts/clair3_c_impl.sh:no_phasing_for_fa::,pileup_model_prefix::,fa_model_prefix::,call_snp_only::,remove_intermediate_dir::,enable_phasing::,enable_long_indel::,use_gpu::,longphase_for_phasing::,longphase::,base_err::,gq_bin_size:: -n 'run_clair3.sh' -- "$@"`
scripts/clair3_c_impl.sh:    --use_gpu ) USE_GPU="$2"; shift 2 ;;
scripts/clair3_c_impl.sh:    --use_gpu ${USE_GPU}" :::: ${OUTPUT_FOLDER}/tmp/CHUNK_LIST |& tee ${LOG_PATH}/1_call_var_bam_pileup.log
scripts/clair3_c_impl.sh:    --use_gpu ${USE_GPU} \
scripts/clair3.sh:no_phasing_for_fa::,pileup_model_prefix::,fa_model_prefix::,call_snp_only::,remove_intermediate_dir::,enable_phasing::,enable_long_indel::,use_gpu::,longphase_for_phasing::,longphase::,base_err::,gq_bin_size:: -n 'run_clair3.sh' -- "$@"`
scripts/clair3.sh:    --use_gpu ) USE_GPU="$2"; shift 2 ;;
scripts/clair3.sh:export CUDA_VISIBLE_DEVICES=""
run_clair3.sh:remove_intermediate_dir,no_phasing_for_fa,call_snp_only,enable_phasing,enable_long_indel,use_gpu,longphase_for_phasing,disable_c_impl,help,version -n 'run_clair3.sh' -- "$@"`
run_clair3.sh:USE_GPU=False
run_clair3.sh:    --use_gpu ) USE_GPU=True; shift 1 ;;
run_clair3.sh:    --use_gpu=${USE_GPU} \
clair3/CallVariants.py:    use_gpu = args.use_gpu
clair3/CallVariants.py:    if use_gpu:
clair3/CallVariants.py:        gpus = tf.config.experimental.list_physical_devices('GPU')
clair3/CallVariants.py:        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
clair3/CallVariants.py:        os.environ["CUDA_VISIBLE_DEVICES"] = ""
clair3/CallVariants.py:    use_gpu = args.use_gpu
clair3/CallVariants.py:    if use_gpu:
clair3/CallVariants.py:        gpus = tf.config.experimental.list_physical_devices('GPU')
clair3/CallVariants.py:        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
clair3/CallVariants.py:        os.environ["CUDA_VISIBLE_DEVICES"] = ""
clair3/CallVariants.py:    parser.add_argument('--use_gpu', type=str2bool, default=False,
clair3/CallVariants.py:                        help="DEBUG: Use GPU for calling. Speed up is mostly insignficiant. Only use this for building your own pipeline")
clair3/CallVariantsFromCffi.py:    if not args.use_gpu:
clair3/CallVariantsFromCffi.py:    use_gpu = args.use_gpu
clair3/CallVariantsFromCffi.py:    if use_gpu:
clair3/CallVariantsFromCffi.py:        os.environ["CUDA_VISIBLE_DEVICES"] = ""
clair3/CallVariantsFromCffi.py:        if use_gpu:
clair3/CallVariantsFromCffi.py:        if use_gpu:
clair3/CallVariantsFromCffi.py:    if not use_gpu:
clair3/CallVariantsFromCffi.py:            if use_gpu:
clair3/CallVariantsFromCffi.py:    parser.add_argument('--use_gpu', type=str2bool, default=False,
clair3/CallVariantsFromCffi.py:                        help="DEBUG: Use GPU for calling. Speed up is mostly insignficiant. Only use this for building your own pipeline")
clair3/CallVarBam.py:    parser.add_argument('--use_gpu', type=str2bool, default=False,
clair3/CallVarBam.py:                        help="DEBUG: Use GPU for calling. Speed up is mostly insignificant. Only use this for building your own pipeline")
clair3/utils.py:    which can meet the requirement of gpu utilization. lz4hc decompression allows speed up training array decompression 4~5x compared
clair3/utils.py:    with tensorflow tfrecord file format, current gpu utilization could reach over 85% with only 10G memory.

```
