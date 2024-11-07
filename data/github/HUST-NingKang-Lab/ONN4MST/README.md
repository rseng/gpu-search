# https://github.com/HUST-NingKang-Lab/ONN4MST

```console
README.md:The model can run either in GPU  or CPU mode, we have provided an option `-g` to indicate that. See [Usage](https://github.com/HUST-NingKang-Lab/ONN4MST#usage) for details.
README.md:usage: searching.py [-h] [-g {0,1}] [-gid GPU_CORE_ID] [-s {0,1}] [-t TREE]
README.md:The `-m`  and `-t` arguments for `src/searching.py` are used to specify model (".json" file, see release page) and biome ontology (".tree" file under `config`). If you want ONN4MST to run in GPU mode, use `-g 1`.  And the model based on selected features can be accessed by using `-m config/model_sf.json` with `-s 1`. 
README.md:- using model based on all features, in GPU mode.
README.md:- using model based on selected features, in GPU mode.
src/testing.py:os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
src/testing.py:os.environ["CUDA_VISIBLE_DEVICES"]="0"
src/searching.py:  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
src/searching.py:  os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gid)
src/searching.py:  gpus, ifn, ofn, ontology, mdl, threshold, sf, gid, mp, ofmt = args.gpus, args.ifn, args.ofn, args.tree, args.model, args.threshold, args.selfea, args.gpu_core_id, args.mapping, args.outfmt
src/searching.py:  if(gpus == 1):
src/searching.py:    print("gpu mode...")
src/searching.py:  Model = Modelrecv(mdl, matrices_size, label_size, gpus)
src/graph_builder.py:  def __init__(self, feature=None, feature_size=None, label=None, label_size=None, lr=1e-4, is_training=True, reuse=False, gpu_mode=0):
src/graph_builder.py:    myconfig.gpu_options.per_process_gpu_memory_fraction = 0.9
src/graph_builder.py:      if(gpu_mode == 0):
src/graph_builder.py:      elif(gpu_mode == 1):
src/graph_builder.py:        tf.logging.info('model using gpu!')
src/predicting.py:#set seeable gpu core id for device
src/predicting.py:  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
src/predicting.py:  os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gid)
src/predicting.py:def Modelrecv(mdl, feature_size, label_size, gpus):
src/predicting.py:  Model = model(feature_size = feature_size, label_size = label_size, gpu_mode = gpus)
src/training.py:os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
src/training.py:os.environ["CUDA_VISIBLE_DEVICES"]="0"
src/utils.py:  parser.add_argument('-g', '--gpus', type=int, choices = [0,1], default = 0, help="This parameter means whether run this program on gpu devices, 0 means on cpu, 1 means on gpu. Default is 0.")
src/utils.py:  parser.add_argument('-gid', '--gpu_core_id', type=str, default = '0', help="If you set the \'-g\'=1, then you should indicate which gpu core you want to use. For example, '0,1,4,6' means these four gpu cores is useable for the program. Default is \'0\'.")
src/Document.md:The function of "graph_builder.py" is to encode the Ontology-aware Neural Network. Specifically, in this script, we can know how many layers the ontological neural network has, how many neurons each layer has, and how the different layers are connected. In addition, the script defines some hyperparameters, such as the memory invocation ratio of the GPU.

```
