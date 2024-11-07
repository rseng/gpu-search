# https://github.com/YangLab/RNAlight

```console
Repeat_Work.md:* GPU: NVIDIA Tesla V100 PCIe 32GB, Driver Version: 440.31, CUDA Version: 10.2
Repeat_Work.md:      tensorflow-gpu==2.0.0
Repeat_Work.md:      cudnn==7.6.5=cuda10.0_0
Repeat_Work.md:      cudatoolkit==10.0.130
Other_Encoding_Strategies_for_DL/mRNA/word2vec/01_CNN_9000nt_word2vec_3_mer.py:# GPU Device
Other_Encoding_Strategies_for_DL/mRNA/word2vec/01_CNN_9000nt_word2vec_3_mer.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/mRNA/word2vec/01_CNN_9000nt_word2vec_4_mer.py:# GPU Device
Other_Encoding_Strategies_for_DL/mRNA/word2vec/01_CNN_9000nt_word2vec_4_mer.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/mRNA/word2vec/01_CNN_9000nt_word2vec_5_mer.py:# GPU Device
Other_Encoding_Strategies_for_DL/mRNA/word2vec/01_CNN_9000nt_word2vec_5_mer.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/mRNA/5_prime_padding/CNN_RNN_9000nt.py:# GPU Device
Other_Encoding_Strategies_for_DL/mRNA/5_prime_padding/CNN_RNN_9000nt.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/mRNA/5_prime_padding/CNN_9000nt.py:# GPU Device
Other_Encoding_Strategies_for_DL/mRNA/5_prime_padding/CNN_9000nt.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/lncRNA/word2vec/01_CNN_4000nt_word2vec_5_mer.py:# GPU Device
Other_Encoding_Strategies_for_DL/lncRNA/word2vec/01_CNN_4000nt_word2vec_5_mer.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/lncRNA/word2vec/01_CNN_4000nt_word2vec_4_mer.py:# GPU Device
Other_Encoding_Strategies_for_DL/lncRNA/word2vec/01_CNN_4000nt_word2vec_4_mer.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/lncRNA/word2vec/01_CNN_4000nt_word2vec_3_mer.py:# GPU Device
Other_Encoding_Strategies_for_DL/lncRNA/word2vec/01_CNN_4000nt_word2vec_3_mer.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/lncRNA/5_prime_padding/CNN_RNN_4000nt.py:# GPU Device
Other_Encoding_Strategies_for_DL/lncRNA/5_prime_padding/CNN_RNN_4000nt.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/lncRNA/5_prime_padding/CNN_4000nt.py:# GPU Device
Other_Encoding_Strategies_for_DL/lncRNA/5_prime_padding/CNN_4000nt.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_5_mer_time_mem_assum.py:def gpu_memory_usage(gpu_id):
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_5_mer_time_mem_assum.py:    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv"
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_5_mer_time_mem_assum.py:# GPU Device
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_5_mer_time_mem_assum.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_5_mer_time_mem_assum.py:memory_consumption = gpu_memory_usage(0)
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_3_mer_time_mem_assum.py:def gpu_memory_usage(gpu_id):
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_3_mer_time_mem_assum.py:    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv"
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_3_mer_time_mem_assum.py:# GPU Device
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_3_mer_time_mem_assum.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_3_mer_time_mem_assum.py:memory_consumption = gpu_memory_usage(0)
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_one_hot_time_mem_assum.py:def gpu_memory_usage(gpu_id):
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_one_hot_time_mem_assum.py:    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv"
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_one_hot_time_mem_assum.py:# GPU Device
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_one_hot_time_mem_assum.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_one_hot_time_mem_assum.py:memory_consumption = gpu_memory_usage(0)
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_4_mer_time_mem_assum.py:def gpu_memory_usage(gpu_id):
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_4_mer_time_mem_assum.py:    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv"
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_4_mer_time_mem_assum.py:# GPU Device
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_4_mer_time_mem_assum.py:CONFIG.gpu_options.allow_growth = True
Other_Encoding_Strategies_for_DL/lncRNA/time_memory_consumption/CNN_4000nt_word2vec_4_mer_time_mem_assum.py:memory_consumption = gpu_memory_usage(0)

```
