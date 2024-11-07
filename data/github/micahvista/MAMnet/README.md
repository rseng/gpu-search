# https://github.com/micahvista/MAMnet

```console
MAMnet.py:#tf.config.set_visible_devices([], 'GPU')
MAMnet.py:def call_sv(feedgpuqueue, calledqueue, step, window_size, weightpathdict, meanvalue, workdir, genotype = False, mc = False, Hi = 200):
MAMnet.py:        if(len(filelist) == 0 and feedgpuqueue.empty() == False):
MAMnet.py:            tmplist = feedgpuqueue.get()
MAMnet.py:    feedgpuqueue = Queue()
MAMnet.py:    p = multiprocessing.Process(target=call_sv, args=(feedgpuqueue, calledqueue, step, window_size, weightpathdict, meanvalue, workdir, genotype, mc, Hi, ))
MAMnet.py:            feedgpuqueue.put(1) 

```
