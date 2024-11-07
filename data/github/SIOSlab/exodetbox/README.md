# https://github.com/SIOSlab/exodetbox

```console
exodetbox/Brown2010DynamicCompletenessReplication.py:#IF USING GPU
exodetbox/Brown2010DynamicCompletenessReplication.py:import pycuda.autoinit
exodetbox/Brown2010DynamicCompletenessReplication.py:import pycuda.gpuarray as gpuarray
exodetbox/Brown2010DynamicCompletenessReplication.py:import skcuda.linalg as linalg
exodetbox/Brown2010DynamicCompletenessReplication.py:#GPU example
exodetbox/Brown2010DynamicCompletenessReplication.py:# x_gpu = gpuarray.to_gpu(x)
exodetbox/Brown2010DynamicCompletenessReplication.py:# y_gpu = gpuarray.to_gpu(y)
exodetbox/Brown2010DynamicCompletenessReplication.py:# z_gpu = linalg.multiply(x_gpu, y_gpu)
exodetbox/Brown2010DynamicCompletenessReplication.py:#np.allclose(x*y, z_gpu.get())
exodetbox/Brown2010DynamicCompletenessReplication.py:def dynamicCompleteness(tssStart,tssEnd,planetIsVisibleBool2,startTimes,tpast_startTimes,periods,planetTypeInds=None, GPU=False):
exodetbox/Brown2010DynamicCompletenessReplication.py:    if not GPU:
exodetbox/Brown2010DynamicCompletenessReplication.py:    else: #GPU is true
exodetbox/Brown2010DynamicCompletenessReplication.py:        planetIsVisibleBool2_gpu = gpuarray.to_gpu(planetIsVisibleBool2)
exodetbox/Brown2010DynamicCompletenessReplication.py:        # x_gpu = gpuarray.to_gpu(x)
exodetbox/Brown2010DynamicCompletenessReplication.py:        # y_gpu = gpuarray.to_gpu(y)
exodetbox/Brown2010DynamicCompletenessReplication.py:        # z_gpu = linalg.multiply(x_gpu, y_gpu)
exodetbox/Brown2010DynamicCompletenessReplication.py:        ptypeBool_gpu = gpuarray.to_gpu(planetTypeBool)
exodetbox/Brown2010DynamicCompletenessReplication.py:        planetIsVisibleBool2_gpu = linalg.multiply(planetIsVisibleBool2_gpu,planetTypeBool_gpu) #Here we remove the planets that are not the desired type
exodetbox/Brown2010DynamicCompletenessReplication.py:        startTimes_gpu = gpuarray.to_gpu(np.tile(tobs1,(7,1)).T) #startTime into properly sized array
exodetbox/Brown2010DynamicCompletenessReplication.py:        ts2_gpu = gpuarray.to_gpu(ts2)
exodetbox/Brown2010DynamicCompletenessReplication.py:        planetDetectedBools_times_gpu = linalg.multiply(ts2_gpu[:,:-1] < startTimes_gpu,linalg.multiply(ts2_gpu[:,1:] > startTimes_gpu,planetIsVisibleBool2_gpu)) #multiply time window bools by planetIsVisibleBool2. For revisit Completeness
exodetbox/Brown2010DynamicCompletenessReplication.py:        planetDetectedBools2_times_gpu = linalg.multiply(ts2_gpu[:,:-1] < tobs2,linalg.multiply(ts2_gpu[:,1:] > tobs2,planetIsVisibleBool2_gpu)) #is the planet visible at this time segment in time 2?
exodetbox/Brown2010DynamicCompletenessReplication.py:        planetDetectedthenDetected = linalg.multiply(planetDetectedBools_gpu,planetDetectedBools2_gpu) #each planet detected at time 1 and time 2 #planets detected and still in visible region    
exodetbox/Brown2010DynamicCompletenessReplication.py:        planetNotDetectedThenDetected = linalg.multiply(planetNotDetectedBools_gpu,planetDetectedBools2_gpu) #each planet NOT detected at time 1 and detected at time 2 #planet not detected and now in visible region
exodetbox/Brown2010DynamicCompletenessReplication.py:# #### Running time trials GPU
exodetbox/Brown2010DynamicCompletenessReplication.py:# gputimeList = list()
exodetbox/Brown2010DynamicCompletenessReplication.py:#     gputimeList.append(timingStop-timingStart)
exodetbox/Brown2010DynamicCompletenessReplication.py:# #### Running time trials NO GPU
exodetbox/Brown2010DynamicCompletenessReplication.py:#### Running time trials NO GPU

```
