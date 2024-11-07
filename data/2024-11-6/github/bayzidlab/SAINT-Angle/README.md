# https://github.com/bayzidlab/SAINT-Angle

```console
saint_angle/saint_angle.py:            prottrans = generate_prottrans(pseq=protein_pseqs[protein_name], use_gpu=args.use_gpu)
saint_angle/saint_angle.py:            prottrans = generate_prottrans(pseq=protein_pseqs[protein_name], use_gpu=args.use_gpu)
saint_angle/saint_angle.py:    parser.add_argument("--gpu", action="store_true", help="Enables gpu usage for ProtTrans features generation (Default: False)", dest="use_gpu")
saint_angle/saint_angle.py:    parser.set_defaults(use_gpu=False)
saint_angle/utils/features.py:def generate_prottrans(pseq, use_gpu=False):
saint_angle/utils/features.py:    device = torch.device("cuda:0") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
README.md:Then, create a separate ***Conda*** environment for running **SAINT-Angle** with ***Python*** version `3.7`. Install ***TensforFlow*** version `2.6.0`, ***Keras*** version `2.6.0`, and ***pandas*** version `1.3.5`. We tested our method with aforementioned versions of ***TensorFlow*** and ***Keras***. We recommend you install *GPU* version of ***TensorFlow*** for faster inference. Besides, run the following command in a terminal to install ***SentencePiece*** and ***transformers***.

```
