# https://github.com/inpefess/gym-saturation

```console
tableaux2023-paper/gym-saturation.bib:url={https://developer.nvidia.com/blog/nlu-with-tensorrt-bert/},
tableaux2023-paper/gym-saturation.bib:@misc{nvidia-blog,
tableaux2023-paper/gym-saturation.bib:title={{Optimizing T5 and GPT-2 for Real-Time Inference with NVIDIA TensorRT}},
tableaux2023-paper/gym-saturation.bib:url={https://developer.nvidia.com/blog/optimizing-t5-and-gpt-2-for-real-time-inference-with-tensorrt/},
tableaux2023-paper/gym-saturation.tex:Looking at Figure~\ref{fig:ast2vec}, one might wonder how efficient is such an architecture. The average response time observed in our experiments was $2ms$ (with a $150ms$ maximum). A typical natural language processing model which embeds whole texts has a latency from $40ms$ to more than $600ms$~\cite{nvidia-blog} (depending on the model complexity and the length of a text to embed) when run on CPU, so there is no reason to believe that \texttt{ast2vec} is too slow. When evaluating a prover, one usually fixes the time limit: for example, $60s$ is the default value for Vampire. Being written in C++ and with a cornucopia of optimisation tweaks, Vampire can generate around a million clauses during this relatively short timeframe. Thus, to be on par with Vampire, a representation service must have latency around $60\mu s$ (orders of magnitude faster than we have). There can be several ways to lower the latency:
tableaux2023-paper/gym-saturation.tex:\item use GPU. NVIDIA reports around 20x improvement vs CPU~\cite{nlu-with-tensorrt-bert}. However, throwing more GPUs won't be as efficient without batch inference from the previous point
tableaux2023-paper/mean-reward.eps: G^Gpu\&a54,`+7Z81(gi:Ao))LE+#9N)]-]=9D?[hI9c(7)1\+f_#B`t5&,@K+SJ$GO@Xi=
tableaux2023-paper/mean-reward.eps: I#jU_/\7NIW-c"L,caGMH%qe]<jt2fIRN[m+04I//jI6h]Des5&"_G4FS2K*DcGpu$c`D;$

```
