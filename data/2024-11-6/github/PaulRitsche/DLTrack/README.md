# https://github.com/PaulRitsche/DL_Track_US

```console
docs/source/installation.rst:GPU setup
docs/source/installation.rst:The processing speed of a single image or video frame analyzed with DL_Track_US is highly dependent on computing power. While possible, model inference and model training using a CPU only will decrese processing speed and prolong the model training process. Therefore, we advise to use a GPU whenever possible. Prior to using a GPU it needs to be set up. Firstly the GPU drivers must be locally installed on your computer. You can find out which drivers are right for your GPU `here <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_. Subsequent to installing the drivers, you need to install the interdependant CUDA and cuDNN software packages. To use DL_Track_US with tensorflow version 2.10 you need to install CUDA version 11.2 from `here <https://developer.nvidia.com/cuda-11.2.0-download-archive>`_ and cuDNN version 8.5 for CUDA version 11.x from `here <https://developer.nvidia.com/rdp/cudnn-archive>`_ (you may need to create an nvidia account). As a next step, you need to be your own installation wizard. We refer to this `video <https://www.youtube.com/watch?v=OEFKlRSd8Ic>`_ (up to date, minute 9 to minute 13) or this `video <https://www.youtube.com/watch?v=IubEtS2JAiY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=2>`_ (older, entire video but replace CUDA and cuDNN versions). There are procedures at the end of each video testing whether a GPU is detected by tensorflow or not. If you run into problems with the GPU/CUDA setup, please open a discussion in the Q&A section of `DL_Track_US discussions <https://github.com/PaulRitsche/DL_Track_US/discussions/categories/q-a>`_ and assign the label "Problem".
docs/source/installation.rst:In case you want to make use of you M1 / M2 chips for model training and / or inference, we refer you to this `tutorial <https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706>`_. There you will find a detailed description of how to enable GPU support for tensorflow. It is not strictly necessary to do that for model training or inference, but will speed up the process. 
Paper/paper.bib:	abstract = {There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more eﬃciently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caﬀe) and the trained networks are available at http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net.},
README.md:For detailled information about installaion of the DL_Track_US python package we refer you to our [documentation](https://dltrack.readthedocs.io/en/latest/installation.html). There you will finde guidelines not only for the installation procedure of DL_Track_US, but also concerding conda and GPU setup.
DL_Track_US/DL_Track_US_GUI.py:        (no GPU or GPU with RAM lower than 8 gigabyte).
README_Pypi.md:For detailled information about installaion of the DL_Track_US python package we refer you to our [documentation](https://dltrack.readthedocs.io/en/latest/installation.html). There you will finde guidelines not only for the installation procedure of DL_Track_US, but also concerning conda and GPU setup.

```