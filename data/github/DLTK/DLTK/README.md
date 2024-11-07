# https://github.com/DLTK/DLTK

```console
setup.py:          '`pip install tensorflow-gpu` if you have a CUDA-enabled GPU or with '
setup.py:          '`pip install tensorflow` without GPU support.')
README.md:2. Install TensorFlow (>=1.4.0) (preferred: with GPU support) for your system
README.md:   pip install "tensorflow-gpu>=1.4.0"
README.md:We would like to thank [NVIDIA GPU Computing](http://www.nvidia.com/) for providing us with hardware for our research. 
requirements.txt:tensorflow-gpu==1.3.0
examples/applications/MRBrainS13_tissue_segmentation/README.md:                    [--cuda_devices CUDA_DEVICES] [--model_path MODEL_PATH]
examples/applications/MRBrainS13_tissue_segmentation/deploy.py:    parser.add_argument('--cuda_devices', '-c', default='0')
examples/applications/MRBrainS13_tissue_segmentation/deploy.py:    # GPU allocation options
examples/applications/MRBrainS13_tissue_segmentation/deploy.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
examples/applications/MRBrainS13_tissue_segmentation/train.py:    parser.add_argument('--cuda_devices', '-c', default='0')
examples/applications/MRBrainS13_tissue_segmentation/train.py:    # GPU allocation options
examples/applications/MRBrainS13_tissue_segmentation/train.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
examples/applications/IXI_HH_sex_classification_resnet/README.md:                    [--cuda_devices CUDA_DEVICES] [--model_path MODEL_PATH]
examples/applications/IXI_HH_sex_classification_resnet/deploy.py:    parser.add_argument('--cuda_devices', '-c', default='0')
examples/applications/IXI_HH_sex_classification_resnet/deploy.py:    # GPU allocation options
examples/applications/IXI_HH_sex_classification_resnet/deploy.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
examples/applications/IXI_HH_sex_classification_resnet/train.py:    parser.add_argument('--cuda_devices', '-c', default='0')
examples/applications/IXI_HH_sex_classification_resnet/train.py:    # GPU allocation options
examples/applications/IXI_HH_sex_classification_resnet/train.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
examples/applications/IXI_HH_age_regression_resnet/README.md:                    [--cuda_devices CUDA_DEVICES] [--model_path MODEL_PATH]
examples/applications/IXI_HH_age_regression_resnet/deploy.py:    parser.add_argument('--cuda_devices', '-c', default='0')
examples/applications/IXI_HH_age_regression_resnet/deploy.py:    # GPU allocation options
examples/applications/IXI_HH_age_regression_resnet/deploy.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
examples/applications/IXI_HH_age_regression_resnet/train.py:    parser.add_argument('--cuda_devices', '-c', default='0')
examples/applications/IXI_HH_age_regression_resnet/train.py:    # GPU allocation options
examples/applications/IXI_HH_age_regression_resnet/train.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
examples/applications/IXI_HH_DCGAN/README.md:                    [--cuda_devices CUDA_DEVICES] [--model_path MODEL_PATH]
examples/applications/IXI_HH_DCGAN/train.py:    parser.add_argument('--cuda_devices', '-c', default='0')
examples/applications/IXI_HH_DCGAN/train.py:    # GPU allocation options
examples/applications/IXI_HH_DCGAN/train.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
examples/applications/IXI_HH_representation_learning_cae/README.md:                    [--cuda_devices CUDA_DEVICES] [--model_path MODEL_PATH]
examples/applications/IXI_HH_representation_learning_cae/train.py:    parser.add_argument('--cuda_devices', '-c', default='0')
examples/applications/IXI_HH_representation_learning_cae/train.py:    # GPU allocation options
examples/applications/IXI_HH_representation_learning_cae/train.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
examples/applications/IXI_HH_superresolution/README.md:                    [--cuda_devices CUDA_DEVICES] [--model_path MODEL_PATH]
examples/applications/IXI_HH_superresolution/train.py:    parser.add_argument('--cuda_devices', '-c', default='0')
examples/applications/IXI_HH_superresolution/train.py:    # GPU allocation options
examples/applications/IXI_HH_superresolution/train.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

```
