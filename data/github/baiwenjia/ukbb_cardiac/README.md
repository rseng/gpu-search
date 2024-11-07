# https://github.com/baiwenjia/ukbb_cardiac

```console
demo_pipeline.py:    # The GPU device id
demo_pipeline.py:    CUDA_VISIBLE_DEVICES = 0
demo_pipeline.py:    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name sa --data_dir demo_image '
demo_pipeline.py:              '--model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))
demo_pipeline.py:    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name la_2ch --data_dir demo_image '
demo_pipeline.py:              '--model_path trained_model/FCN_la_2ch'.format(CUDA_VISIBLE_DEVICES))
demo_pipeline.py:    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name la_4ch --data_dir demo_image '
demo_pipeline.py:              '--model_path trained_model/FCN_la_4ch'.format(CUDA_VISIBLE_DEVICES))
demo_pipeline.py:    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name la_4ch --data_dir demo_image '
demo_pipeline.py:              '--seg4 --model_path trained_model/FCN_la_4ch_seg4'.format(CUDA_VISIBLE_DEVICES))
demo_pipeline.py:    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network_ao.py --seq_name ao --data_dir demo_image '
demo_pipeline.py:              '--model_path trained_model/UNet-LSTM_ao'.format(CUDA_VISIBLE_DEVICES))
README.md:pip3 install tensorflow-gpu numpy scipy matplotlib seaborn pandas python-dateutil pydicom SimpleITK nibabel scikit-image opencv-python vtk
README.md:There is one parameter in the script, *CUDA_VISIBLE_DEVICES*, which controls which GPU device to use on your machine. Currently, I set it to 0, which means the first GPU on your machine.
README.md:**Speed** The speed of image segmentation depends several factors, such as whether to use GPU or CPU, the GPU hardware, the test image size etc. In my case, I use a Nvidia Titan K80 GPU and it takes about 10 seconds to segment a full time sequence (50 time frames), with the image size to be 192x208x10x50 (i.e. each 2D image slice to be 192x208 pixels, with 10 image slices and 50 time frames). Adding the time for short-axis image segmentation, long-axis image segmentation, aortic image segmentation together, it will take about 25 seconds per subject.
README.md:For strain evaluation, it will take several minutes per subject, since MIRTK runs on CPUs, instead of GPUs. 
common/deploy_network.py:        print('Including image I/O, CUDA resource allocation, '
common/network.py:    # dimension (e.g. 256 features), which may exhaust the GPU memory (e.g.
common/network.py:    # 12 GB for Nvidia Titan K80).
common/deploy_network_ao.py:        print('Including image I/O, CUDA resource allocation, '

```
