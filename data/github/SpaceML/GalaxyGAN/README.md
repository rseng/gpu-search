# https://github.com/SpaceML/GalaxyGAN

```console
README.md:* CUDA
README.md:. (Can be launched using p2.xlarge instance in GPU compute catagory)
README.md:note: If you get error like "nvidia-uvm 4.4.0-62 generic" was missing, this is because Amazon updated the kernal of the Ubuntu system, please re-install the cuda again.
README.md:	bash train.sh -input ~/fits_train -fwhm 1.4 -sigma 1.2 -figure figures -gpu 1 -model models
README.md:	bash test.sh -input ~/fits_test -fwhm 1.4 -sigma 1.2 -figure figures -gpu 1 -output result -model models -mode full
README.md:	bash test.sh -input ~/fits_test -fwhm 1.4 -sigma 1.2 -figure figures -gpu 1 -output result -model models -mode blank
README.md:NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)
README.md:	bash train.sh -input fitsdata/fits_train -fwhm 1.4 -sigma 1.2 -figure figures -gpu 1 -model models
README.md:	bash test.sh -input fitsdata/fits_test -fwhm 1.4 -sigma 1.2 -figure figures -gpu 1 -output result -model models -mode full
README.md:	bash test.sh -input fitsdata/fits_test -fwhm 1.4 -sigma 1.2 -figure figures -gpu 1 -output result -model models -mode blank
train.sh:        -gpu    )               shift
train.sh:                                user_gpu=$1
train.sh:DATA_ROOT=../$base_dir/$name name=$fwhm"_"$sig  which_direction=BtoA display=0 niter=20 batchSize=1 gpu=$user_gpu save_latest_freq=2000 checkpoints_dir=../$model_dir th train.lua
web_UI/gpu_machine/process.php:$th_command = "export CUDNN_PATH=/usr/local/cuda/lib64/libcudnn.so && export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/torch/install/lib:/usr/local/cuda/lib64 && DATA_ROOT=".$processDir." name=001002_2.5_5.0 which_direction=BtoA phase=test display=0 gpu=1 checkpoints_dir=/mnt/ds3lab/roou_v2_checkpoint/ results_dir=demo_response/ /usr/local/torch/install/bin/th /mnt/ds3lab/astronomy/pix2pix/test.lua > /tmp/th.log 2>&1";
web_UI/README:1. LAMP Stack installed on both the webserver and the GPU machine
web_UI/README:3. CUDA, cuDNN installed on the GPU Machine
web_UI/README:4. Torch installed in GPU Machine at /usr/local (chmod 777 so that everyone can access)
web_UI/README:3. upload.php stores the file on disk and sends it to the gpu machine via cURL, error raised if cURL is not installed.
web_UI/README:4. gpu_machine/process.php gets the images, stores it on disk, calls torch with the necessary arguments and sends back the result to webserver/upload.php.
test.sh:        -gpu    )               shift
test.sh:                                user_gpu=$1
test.sh:DATA_ROOT=../$base_dir/$name name=$fwhm"_"$sig which_direction=BtoA phase=test display=0 gpu=$user_gpu checkpoints_dir=../$model_dir results_dir=../$result_dir th test.lua

```
