# https://github.com/PengyiZhang/DRR4Covid

```console
FCNClsSegModel/train_no_mmd.py:    torch.cuda.manual_seed_all(seed)
FCNClsSegModel/train_no_mmd.py:def set_visible_gpu(gpu_idex):
FCNClsSegModel/train_no_mmd.py:    to control which gpu is visible for CUDA user
FCNClsSegModel/train_no_mmd.py:    set_visible_gpu(1)
FCNClsSegModel/train_no_mmd.py:    print(os.environ["CUDA_DEVICE_ORDER"])
FCNClsSegModel/train_no_mmd.py:    print(os.environ["CUDA_VISIBLE_DEVICES"])
FCNClsSegModel/train_no_mmd.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
FCNClsSegModel/train_no_mmd.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(gpu_idex)
FCNClsSegModel/train_no_mmd.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
FCNClsSegModel/train_no_mmd.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpuid)
FCNClsSegModel/train_no_mmd.py:    device_name = "cuda" if torch.cuda.is_available() else "cpu"
FCNClsSegModel/train_no_mmd.py:    # Send the model to GPU
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx5/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx5/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx5/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx5/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx5/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx5/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx5/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx5/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx5/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx1/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx1/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx1/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx1/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx1/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx1/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx1/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx1/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx1/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx0/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx0/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx0/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx0/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx0/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx0/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx0/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx0/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx0/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx6/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx6/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx6/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx6/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx6/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx6/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx6/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx6/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx6/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx2/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx2/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx2/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx2/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx2/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx2/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx2/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx2/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx2/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx4/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx4/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx4/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx4/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx4/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx4/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx4/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx4/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx4/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx3/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx3/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx3/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx3/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx3/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx3/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx3/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx3/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_mmd/clsx3/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx5/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx5/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx5/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx5/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx5/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx5/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx5/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx5/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx5/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx1/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx1/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx1/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx1/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx1/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx1/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx1/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx1/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx1/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx0/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx0/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx0/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx0/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx0/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx0/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx0/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx0/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx0/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx6/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx6/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx6/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx6/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx6/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx6/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx6/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx6/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx6/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx2/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx2/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx2/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx2/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx2/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx2/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx2/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx2/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx2/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx4/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx4/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx4/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx4/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx4/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx4/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx4/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx4/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx4/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx3/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx3/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx3/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx3/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx3/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx3/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx3/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx3/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R50/cls_nommd/clsx3/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx5/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx5/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx5/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx5/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx5/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx5/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx5/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx5/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx5/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx1/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx1/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx1/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx1/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx1/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx1/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx1/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx1/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx1/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx0/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx0/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx0/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx0/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx0/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx0/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx0/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx0/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx0/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx6/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx6/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx6/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx6/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx6/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx6/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx6/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx6/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx6/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx2/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx2/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx2/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx2/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx2/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx2/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx2/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx2/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx2/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx4/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx4/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx4/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx4/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx4/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx4/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx4/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx4/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx4/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx3/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx3/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx3/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx3/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx3/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx3/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx3/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx3/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_mmd/clsx3/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx5/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx5/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx5/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx5/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx5/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx5/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx5/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx5/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx5/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx1/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx1/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx1/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx1/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx1/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx1/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx1/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx1/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx1/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx6/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx6/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx6/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx6/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx6/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx6/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx6/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx6/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx6/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx2/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx2/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx2/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx2/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx2/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx2/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx2/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx2/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx2/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx4/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx4/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx4/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx4/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx4/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx4/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx4/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx4/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx4/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx3/experiment-FCNR18-drrxray-60.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx3/experiment-FCNR18-drrxray-80.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx3/experiment-FCNR18-drrxray-50.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx3/experiment-FCNR18-drrxray-20.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx3/experiment-FCNR18-drrxray-30.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx3/experiment-FCNR18-drrxray-10.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx3/experiment-FCNR18-drrxray-40.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx3/experiment-FCNR18-drrxray-70.yaml:    gpuid: 0
FCNClsSegModel/cfgs/experiment01/R18/cls_nommd/clsx3/experiment-FCNR18-drrxray-90.yaml:    gpuid: 0
FCNClsSegModel/mmd.py:    weight_ss = torch.from_numpy(weight_ss).cuda()
FCNClsSegModel/mmd.py:    weight_tt = torch.from_numpy(weight_tt).cuda()
FCNClsSegModel/mmd.py:    weight_st = torch.from_numpy(weight_st).cuda()
FCNClsSegModel/mmd.py:    loss = torch.Tensor([0]).cuda()
FCNClsSegModel/README.md:python3.6 train_mmd.py --config cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-50.yaml --fold 0 --setgpuid 0
FCNClsSegModel/README.md:python3.6 train_no_mmd.py --config cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-50.yaml --fold 0 --setgpuid 0
FCNClsSegModel/README.md:python3.6 eval_mmd.py --config cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-50.yaml --fold 0 --setgpuid 0
FCNClsSegModel/README.md:python3.6 eval_no_mmd.py --config cfgs/experiment01/R18/cls_nommd/clsx0/experiment-FCNR18-drrxray-50.yaml --fold 0 --setgpuid 0
FCNClsSegModel/eval_mmd.py:    torch.cuda.manual_seed_all(seed)
FCNClsSegModel/eval_mmd.py:def set_visible_gpu(gpu_idex):
FCNClsSegModel/eval_mmd.py:    to control which gpu is visible for CUDA user
FCNClsSegModel/eval_mmd.py:    set_visible_gpu(1)
FCNClsSegModel/eval_mmd.py:    print(os.environ["CUDA_DEVICE_ORDER"])
FCNClsSegModel/eval_mmd.py:    print(os.environ["CUDA_VISIBLE_DEVICES"])
FCNClsSegModel/eval_mmd.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
FCNClsSegModel/eval_mmd.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(gpu_idex)
FCNClsSegModel/eval_mmd.py:    parser.add_argument('--setgpuid', default=0, type=int)
FCNClsSegModel/eval_mmd.py:    opt.gpuid = opt.setgpuid
FCNClsSegModel/eval_mmd.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
FCNClsSegModel/eval_mmd.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpuid)
FCNClsSegModel/eval_mmd.py:    device_name = "cuda" if torch.cuda.is_available() else "cpu"
FCNClsSegModel/eval_mmd.py:    # Send the model to GPU
FCNClsSegModel/loss.py:            self.nll_weight = class_weights#Variable(class_weights.float()).cuda()
FCNClsSegModel/loss.py:def Weighted_Jaccard_loss(label, pred, class_weights=None, gpu=0, num_classes=3):
FCNClsSegModel/loss.py:    label = Variable(label.long()).cuda(gpu)
FCNClsSegModel/loss.py:        class_weights = Variable(class_weights).cuda(gpu)
FCNClsSegModel/loss.py:        criterion = LossMulti(jaccard_weight=0.5, class_weights=class_weights,num_classes=num_classes)#.cuda(gpu)
FCNClsSegModel/loss.py:        criterion = LossMulti(jaccard_weight=0.5, num_classes=num_classes)  # .cuda(gpu)
FCNClsSegModel/utils.py:    torch.cuda.manual_seed_all(seed)
FCNClsSegModel/eval_no_mmd.py:    torch.cuda.manual_seed_all(seed)
FCNClsSegModel/eval_no_mmd.py:def set_visible_gpu(gpu_idex):
FCNClsSegModel/eval_no_mmd.py:    to control which gpu is visible for CUDA user
FCNClsSegModel/eval_no_mmd.py:    set_visible_gpu(1)
FCNClsSegModel/eval_no_mmd.py:    print(os.environ["CUDA_DEVICE_ORDER"])
FCNClsSegModel/eval_no_mmd.py:    print(os.environ["CUDA_VISIBLE_DEVICES"])
FCNClsSegModel/eval_no_mmd.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
FCNClsSegModel/eval_no_mmd.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(gpu_idex)
FCNClsSegModel/eval_no_mmd.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
FCNClsSegModel/eval_no_mmd.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpuid)
FCNClsSegModel/eval_no_mmd.py:    device_name = "cuda" if torch.cuda.is_available() else "cpu"
FCNClsSegModel/eval_no_mmd.py:    # Send the model to GPU
FCNClsSegModel/train_mmd.py:    torch.cuda.manual_seed_all(seed)
FCNClsSegModel/train_mmd.py:def set_visible_gpu(gpu_idex):
FCNClsSegModel/train_mmd.py:    to control which gpu is visible for CUDA user
FCNClsSegModel/train_mmd.py:    set_visible_gpu(1)
FCNClsSegModel/train_mmd.py:    print(os.environ["CUDA_DEVICE_ORDER"])
FCNClsSegModel/train_mmd.py:    print(os.environ["CUDA_VISIBLE_DEVICES"])
FCNClsSegModel/train_mmd.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
FCNClsSegModel/train_mmd.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(gpu_idex)
FCNClsSegModel/train_mmd.py:    parser.add_argument('--setgpuid', default=0, type=int)
FCNClsSegModel/train_mmd.py:    opt.gpuid = opt.setgpuid
FCNClsSegModel/train_mmd.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
FCNClsSegModel/train_mmd.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpuid)
FCNClsSegModel/train_mmd.py:    device_name = "cuda" if torch.cuda.is_available() else "cpu"
FCNClsSegModel/train_mmd.py:    # Send the model to GPU
InfectionAwareDRRGenerator/test/ProjectorsModule.py:    SiddonGpu: GPU accelerated (CUDA) DRR generation from CT or MRI scan.  
InfectionAwareDRRGenerator/test/ProjectorsModule.py:    delete: eventually deletes the projector object (only needed to deallocate memory from GPU) 
InfectionAwareDRRGenerator/test/ProjectorsModule.py:from SiddonGpuPy import pySiddonGpu     # Python wrapped C library for GPU accelerated DRR generation
InfectionAwareDRRGenerator/test/ProjectorsModule.py:    p = SiddonGpu(projector_info,
InfectionAwareDRRGenerator/test/ProjectorsModule.py:class SiddonGpu():
InfectionAwareDRRGenerator/test/ProjectorsModule.py:    """GPU accelearated DRR generation from volumetric image (CT or MRI scan).
InfectionAwareDRRGenerator/test/ProjectorsModule.py:       This class renders a DRR from a volumetric image, with an accelerated GPU algorithm
InfectionAwareDRRGenerator/test/ProjectorsModule.py:       from a Python wrapped library (SiddonGpuPy), written in C++ and accelerated with Cuda.
InfectionAwareDRRGenerator/test/ProjectorsModule.py:            delete (function): deletes the projector object (needed to deallocate memory from GPU)
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        # Set parameters for GPU library SiddonGpuPy
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        DRRsize_forGpu = np.array([ projector_info['DRRsize_x'],  projector_info['DRRsize_y'], 1], dtype=np.int32)
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        MovSize_forGpu = np.array([ movImageInfo['Size'][0], movImageInfo['Size'][1], movImageInfo['Size'][2] ], dtype=np.int32)
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        MovSpacing_forGpu = np.array([ movImageInfo['Spacing'][0], movImageInfo['Spacing'][1], movImageInfo['Spacing'][2] ]).astype(np.float32)
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        tGpu1 = time.time()
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        self.projector = pySiddonGpu(NumThreadsPerBlock,
InfectionAwareDRRGenerator/test/ProjectorsModule.py:                                  MovSize_forGpu,
InfectionAwareDRRGenerator/test/ProjectorsModule.py:                                  MovSpacing_forGpu,
InfectionAwareDRRGenerator/test/ProjectorsModule.py:                                  DRRsize_forGpu)
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        tGpu2 = time.time()
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        print( '\nSiddon object initialized. Time elapsed for initialization: ', tGpu2 - tGpu1, '\n')
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        source_forGpu = np.array([ source_transformed[0], source_transformed[1], source_transformed[2] ], dtype=np.float32)
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        #tGpu3 = time.time()
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        output, output_mask, output_mask_lung, output_mask_value = self.projector.generateDRR(source_forGpu,DRRPhy_array)
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        #tGpu4 = time.time()
InfectionAwareDRRGenerator/test/ProjectorsModule.py:        """Deletes the projector object >>> GPU is freed <<<"""
InfectionAwareDRRGenerator/test/genDRRV2.py:projector_info = {'Name': 'SiddonGpu', 
InfectionAwareDRRGenerator/README.md:projector_info = {'Name': 'SiddonGpu', 
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/readme.txt:A CUDA-accelerated C++ library for DRR generation using an improved version of the Siddon algorithm.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/readme.txt:The library defines a class object SiddonGPu, that loads a CT scan onto the GPU memory when initialized.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cuh:* Implementation of a CUDA-based Cpp library for fast DRR generation with GPU acceleration
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cuh:* The class loads a CT scan onto the GPU memory. The function generateDRR can be called multiple times 
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cuh:class SiddonGpu {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cuh:	SiddonGpu(); // default constructor
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cuh:	SiddonGpu(int *NumThreadsPerBlock,
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cuh:	~SiddonGpu(); // destructor
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:* Implementation of a CUDA-based Cpp library for fast DRR generation with GPU acceleration
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:#include "cuda_runtime.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:__global__ void cuda_kernel(float *DRRarray, float *Maskarray, float *Lungarray, float *Valuearray,
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:SiddonGpu::SiddonGpu() { }
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:* Overloaded constructor loads the CT scan (together with size and spacing) onto GPU memory
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:SiddonGpu::SiddonGpu(int *NumThreadsPerBlock,
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&m_d_movImgArray, m_movImgMemSize);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&m_d_MovSize, 3 * sizeof(int));
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&m_d_MovSpacing, 3 * sizeof(float));
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&m_d_movMaskArray, m_movImgMemSize);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&m_d_Weights, 3 * sizeof(float)); // bk,lungs,infection
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(m_d_movImgArray, movImgArray, m_movImgMemSize, cudaMemcpyHostToDevice);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(m_d_MovSize, MovSize, 3 * sizeof(int), cudaMemcpyHostToDevice);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(m_d_MovSpacing, MovSpacing, 3 * sizeof(float), cudaMemcpyHostToDevice);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(m_d_movMaskArray, movMaskArray, m_movImgMemSize, cudaMemcpyHostToDevice);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(m_d_Weights, Weights, 3 * sizeof(float), cudaMemcpyHostToDevice);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	//std::cout << "Siddon object Initialization: GPU memory prepared \n" << std::endl;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:* Destructor clears everything left from the GPU memory
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:SiddonGpu::~SiddonGpu() {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaFree(m_d_movImgArray);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaFree(m_d_MovSize);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaFree(m_d_MovSpacing);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	std::cout << "Siddon object destruction: GPU memory cleared \n" << std::endl;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:void SiddonGpu::generateDRR(float *source,
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaError_t ierrAsync;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaError_t ierrSync;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&d_drr_array, m_DrrMemSize);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&d_mask_array, m_DrrMemSize); // 0: infection-shot mask, 1: lung-shot mask, 2: infection total mask
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&d_lung_array, m_DrrMemSize);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&d_value_array, m_DrrMemSize);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&d_source, 3 * sizeof(float));
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMalloc((void**)&d_DestArray, m_DestMemSize);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(d_DestArray, DestArray, m_DestMemSize, cudaMemcpyHostToDevice);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(d_source, source, 3 * sizeof(float), cudaMemcpyHostToDevice);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	//std::cout << "DRR generation: GPU memory prepared \n" << std::endl;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	//// Query GPU device
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	//cudaDeviceProp prop;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	//cudaGetDeviceProperties(&prop, 0);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	//cudaGetDeviceProperties(&prop, 0);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cuda_kernel << <number_of_blocks, threads_per_block >> >(d_drr_array,
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	ierrSync = cudaGetLastError();
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	if (ierrSync != cudaSuccess) { 
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:		//printf("Cuda Sync error: %s\n", cudaGetErrorString(ierrSync));
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:		std::cout << "Cuda Sync error: "<< cudaGetErrorString(ierrSync) << std::endl;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	if (ierrAsync != cudaSuccess) { 
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:		//printf("Cuda Async error: %s\n", cudaGetErrorString(ierrAsync)); 
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:		std::cout << "Cuda Sync error: "<< cudaGetErrorString(ierrSync) << std::endl;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(drrArray, d_drr_array, m_DrrMemSize, cudaMemcpyDeviceToHost);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(maskArray, d_mask_array, m_DrrMemSize, cudaMemcpyDeviceToHost);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(lungArray, d_lung_array, m_DrrMemSize, cudaMemcpyDeviceToHost);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaMemcpy(valueArray, d_value_array, m_DrrMemSize, cudaMemcpyDeviceToHost);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaFree(d_drr_array);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaFree(d_mask_array);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaFree(d_lung_array);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaFree(d_value_array);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaFree(d_source);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	cudaFree(d_DestArray);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu:	//std::cout << "DRR generation: GPU memory cleared \n" << std::endl;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/CMakeLists.txt:project(SiddonGpu)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/CMakeLists.txt:find_package(CUDA REQUIRED)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/CMakeLists.txt:set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler -fPIC; -O3;)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/CMakeLists.txt:cuda_add_library(SiddonGpu SiddonLib/siddon_class.cu SiddonLib/siddon_class.cuh)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/CMakeLists.txt:target_link_libraries(SiddonGpu cudart cudadevrt)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:CMAKE_SOURCE_DIR = /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:CMAKE_BINARY_DIR = /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:	$(CMAKE_COMMAND) -E cmake_progress_start /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/progress.marks
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:	$(CMAKE_COMMAND) -E cmake_progress_start /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles 0
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:# Target rules for targets named SiddonGpu
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:SiddonGpu: cmake_check_build_system
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:	$(MAKE) -f CMakeFiles/Makefile2 SiddonGpu
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:.PHONY : SiddonGpu
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:SiddonGpu/fast:
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:	$(MAKE) -f CMakeFiles/SiddonGpu.dir/build.make CMakeFiles/SiddonGpu.dir/build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:.PHONY : SiddonGpu/fast
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/Makefile:	@echo "... SiddonGpu"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/TargetDirectories.txt:/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/edit_cache.dir
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/TargetDirectories.txt:/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/rebuild_cache.dir
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/TargetDirectories.txt:/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:CMAKE_SOURCE_DIR = /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:CMAKE_BINARY_DIR = /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:all: CMakeFiles/SiddonGpu.dir/all
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:clean: CMakeFiles/SiddonGpu.dir/clean
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:# Target rules for target CMakeFiles/SiddonGpu.dir
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:CMakeFiles/SiddonGpu.dir/all:
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:	$(MAKE) -f CMakeFiles/SiddonGpu.dir/build.make CMakeFiles/SiddonGpu.dir/depend
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:	$(MAKE) -f CMakeFiles/SiddonGpu.dir/build.make CMakeFiles/SiddonGpu.dir/build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles --progress-num=1,2 "Built target SiddonGpu"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:.PHONY : CMakeFiles/SiddonGpu.dir/all
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:CMakeFiles/SiddonGpu.dir/rule: cmake_check_build_system
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:	$(CMAKE_COMMAND) -E cmake_progress_start /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles 2
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:	$(MAKE) -f CMakeFiles/Makefile2 CMakeFiles/SiddonGpu.dir/all
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:	$(CMAKE_COMMAND) -E cmake_progress_start /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles 0
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:.PHONY : CMakeFiles/SiddonGpu.dir/rule
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:SiddonGpu: CMakeFiles/SiddonGpu.dir/rule
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:.PHONY : SiddonGpu
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:CMakeFiles/SiddonGpu.dir/clean:
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:	$(MAKE) -f CMakeFiles/SiddonGpu.dir/build.make CMakeFiles/SiddonGpu.dir/clean
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile2:.PHONY : CMakeFiles/SiddonGpu.dir/clean
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/CMakeDirectoryInformation.cmake:set(CMAKE_RELATIVE_PATH_TOP_SOURCE "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/CMakeDirectoryInformation.cmake:set(CMAKE_RELATIVE_PATH_TOP_BINARY "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile.cmake:  "CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile.cmake:  "CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile.cmake:  "/usr/local/share/cmake-3.16/Modules/FindCUDA.cmake"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile.cmake:  "/usr/local/share/cmake-3.16/Modules/FindCUDA/run_nvcc.cmake"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile.cmake:  "/usr/local/share/cmake-3.16/Modules/FindCUDA/select_compute_arch.cmake"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile.cmake:  "CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile.cmake:  "CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/Makefile.cmake:  "CMakeFiles/SiddonGpu.dir/DependInfo.cmake"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/CMakeRuleHashes.txt:067d6e620079362fd2647455bca37f2c CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/cmake_clean_target.cmake:  "libSiddonGpu.a"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:#                               entries in CUDA_HOST_FLAGS. This is the build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(source_file "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(NVCC_generated_dependency_file "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.NVCC-depend") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(cmake_dependency_file "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_make2cmake "/usr/local/share/cmake-3.16/Modules/FindCUDA/make2cmake.cmake") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_parse_cubin "/usr/local/share/cmake-3.16/Modules/FindCUDA/parse_cubin.cmake") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_HOST_COMPILER "/usr/bin/cc") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(generated_file_path "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/.") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(generated_file_internal "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/./SiddonGpu_generated_siddon_class.cu.o") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(generated_cubin_file_internal "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/./SiddonGpu_generated_siddon_class.cu.o.cubin.txt") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda/bin/nvcc") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_NVCC_FLAGS -Xcompiler;-fPIC;-O3 ;; ) # list
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_NVCC_FLAGS_DEBUG  ; )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_NVCC_FLAGS_MINSIZEREL  ; )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_NVCC_FLAGS_RELEASE  ; )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_NVCC_FLAGS_RELWITHDEBINFO  ; )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_NVCC_INCLUDE_DIRS [==[/usr/local/cuda/include;$<TARGET_PROPERTY:SiddonGpu,INCLUDE_DIRECTORIES>]==]) # list (needs to be in lua quotes to address backslashes)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:string(REPLACE "\\" "/" CUDA_NVCC_INCLUDE_DIRS "${CUDA_NVCC_INCLUDE_DIRS}")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_NVCC_COMPILE_DEFINITIONS [==[$<TARGET_PROPERTY:SiddonGpu,COMPILE_DEFINITIONS>]==]) # list (needs to be in lua quotes see #16510 ).
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(cuda_language_flag ) # list
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:list(REMOVE_DUPLICATES CUDA_NVCC_INCLUDE_DIRS)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_NVCC_INCLUDE_ARGS)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:foreach(dir ${CUDA_NVCC_INCLUDE_DIRS})
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  list(APPEND CUDA_NVCC_INCLUDE_ARGS "-I${dir}")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:list(REMOVE_DUPLICATES CUDA_NVCC_COMPILE_DEFINITIONS)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:foreach(def ${CUDA_NVCC_COMPILE_DEFINITIONS})
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:# been chosen by FindCUDA.cmake.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:#message("CUDA_NVCC_HOST_COMPILER_FLAGS = ${CUDA_NVCC_HOST_COMPILER_FLAGS}")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_${build_configuration}})
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:# Any -ccbin existing in CUDA_NVCC_FLAGS gets highest priority
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:list( FIND CUDA_NVCC_FLAGS "-ccbin" ccbin_found0 )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:list( FIND CUDA_NVCC_FLAGS "--compiler-bindir" ccbin_found1 )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:if( ccbin_found0 LESS 0 AND ccbin_found1 LESS 0 AND CUDA_HOST_COMPILER )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  if (CUDA_HOST_COMPILER STREQUAL "" AND DEFINED CCBIN)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:    set(CCBIN -ccbin "${CUDA_HOST_COMPILER}")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:# cuda_execute_process - Executes a command with optional command echo and status message.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:#   CUDA_result - return value from running the command
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:macro(cuda_execute_process status command)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:    message(FATAL_ERROR "Malformed call to cuda_execute_process.  Missing COMMAND as second argument. (command = ${command})")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:    set(cuda_execute_process_string)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:        list(APPEND cuda_execute_process_string "\"${arg}\"")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:        list(APPEND cuda_execute_process_string ${arg})
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:    execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${cuda_execute_process_string})
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  execute_process(COMMAND ${ARGN} RESULT_VARIABLE CUDA_result )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:# For CUDA 2.3 and below, -G -M doesn't work, so remove the -G flag
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(depends_CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDA_VERSION 10.2)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:if(CUDA_VERSION VERSION_LESS "3.0")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  list(REMOVE_ITEM depends_CUDA_NVCC_FLAGS "-G")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:# nvcc doesn't define __CUDACC__ for some reason when generating dependency files.  This
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:set(CUDACC_DEFINE -D__CUDACC__)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  COMMAND "${CUDA_NVCC_EXECUTABLE}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  ${CUDACC_DEFINE}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  ${depends_CUDA_NVCC_FLAGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  ${CUDA_NVCC_INCLUDE_ARGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:if(CUDA_result)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  -P "${CUDA_make2cmake}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:if(CUDA_result)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:if(CUDA_result)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:if(CUDA_result)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  COMMAND "${CUDA_NVCC_EXECUTABLE}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  ${cuda_language_flag}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  ${CUDA_NVCC_FLAGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  ${CUDA_NVCC_INCLUDE_ARGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:if(CUDA_result)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:    COMMAND "${CUDA_NVCC_EXECUTABLE}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:    ${CUDA_NVCC_FLAGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:    ${CUDA_NVCC_INCLUDE_ARGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:  cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake.pre-gen:    -P "${CUDA_parse_cubin}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend:SET(CUDA_NVCC_DEPEND
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend:  "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cuh"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/builtin_types.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/channel_descriptor.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/common_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/device_double_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/device_double_functions.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/device_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/device_functions.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/host_config.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/host_defines.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/math_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/math_functions.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/sm_70_rt.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/crt/sm_70_rt.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/cuda_device_runtime_api.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/cuda_runtime.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/cuda_runtime_api.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/cuda_surface_types.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/cuda_texture_types.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/device_atomic_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/device_atomic_functions.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/device_launch_parameters.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/device_types.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/driver_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/driver_types.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/library_types.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_20_atomic_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_20_atomic_functions.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_20_intrinsics.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_20_intrinsics.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_30_intrinsics.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_30_intrinsics.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_32_atomic_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_32_atomic_functions.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_32_intrinsics.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_32_intrinsics.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_35_atomic_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_35_intrinsics.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_60_atomic_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_60_atomic_functions.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_61_intrinsics.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/sm_61_intrinsics.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/surface_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/surface_indirect_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/surface_types.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/texture_fetch_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/texture_indirect_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/texture_types.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/vector_functions.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/vector_functions.hpp"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend: "/usr/local/cuda/include/vector_types.h"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:#                               entries in CUDA_HOST_FLAGS. This is the build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(source_file "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/SiddonLib/siddon_class.cu") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(NVCC_generated_dependency_file "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.NVCC-depend") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(cmake_dependency_file "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_make2cmake "/usr/local/share/cmake-3.16/Modules/FindCUDA/make2cmake.cmake") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_parse_cubin "/usr/local/share/cmake-3.16/Modules/FindCUDA/parse_cubin.cmake") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_HOST_COMPILER "/usr/bin/cc") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(generated_file_path "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/.") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(generated_file_internal "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/./SiddonGpu_generated_siddon_class.cu.o") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(generated_cubin_file_internal "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/./SiddonGpu_generated_siddon_class.cu.o.cubin.txt") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda/bin/nvcc") # path
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_NVCC_FLAGS -Xcompiler;-fPIC;-O3 ;; ) # list
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_NVCC_FLAGS_DEBUG  ; )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_NVCC_FLAGS_MINSIZEREL  ; )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_NVCC_FLAGS_RELEASE  ; )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_NVCC_FLAGS_RELWITHDEBINFO  ; )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_NVCC_INCLUDE_DIRS [==[/usr/local/cuda/include;/usr/local/cuda/include]==]) # list (needs to be in lua quotes to address backslashes)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:string(REPLACE "\\" "/" CUDA_NVCC_INCLUDE_DIRS "${CUDA_NVCC_INCLUDE_DIRS}")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_NVCC_COMPILE_DEFINITIONS [==[]==]) # list (needs to be in lua quotes see #16510 ).
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(cuda_language_flag ) # list
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:list(REMOVE_DUPLICATES CUDA_NVCC_INCLUDE_DIRS)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_NVCC_INCLUDE_ARGS)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:foreach(dir ${CUDA_NVCC_INCLUDE_DIRS})
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  list(APPEND CUDA_NVCC_INCLUDE_ARGS "-I${dir}")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:list(REMOVE_DUPLICATES CUDA_NVCC_COMPILE_DEFINITIONS)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:foreach(def ${CUDA_NVCC_COMPILE_DEFINITIONS})
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:# been chosen by FindCUDA.cmake.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:#message("CUDA_NVCC_HOST_COMPILER_FLAGS = ${CUDA_NVCC_HOST_COMPILER_FLAGS}")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_${build_configuration}})
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:# Any -ccbin existing in CUDA_NVCC_FLAGS gets highest priority
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:list( FIND CUDA_NVCC_FLAGS "-ccbin" ccbin_found0 )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:list( FIND CUDA_NVCC_FLAGS "--compiler-bindir" ccbin_found1 )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:if( ccbin_found0 LESS 0 AND ccbin_found1 LESS 0 AND CUDA_HOST_COMPILER )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  if (CUDA_HOST_COMPILER STREQUAL "" AND DEFINED CCBIN)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:    set(CCBIN -ccbin "${CUDA_HOST_COMPILER}")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:# cuda_execute_process - Executes a command with optional command echo and status message.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:#   CUDA_result - return value from running the command
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:macro(cuda_execute_process status command)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:    message(FATAL_ERROR "Malformed call to cuda_execute_process.  Missing COMMAND as second argument. (command = ${command})")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:    set(cuda_execute_process_string)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:        list(APPEND cuda_execute_process_string "\"${arg}\"")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:        list(APPEND cuda_execute_process_string ${arg})
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:    execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${cuda_execute_process_string})
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  execute_process(COMMAND ${ARGN} RESULT_VARIABLE CUDA_result )
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:# For CUDA 2.3 and below, -G -M doesn't work, so remove the -G flag
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(depends_CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDA_VERSION 10.2)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:if(CUDA_VERSION VERSION_LESS "3.0")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  list(REMOVE_ITEM depends_CUDA_NVCC_FLAGS "-G")
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:# nvcc doesn't define __CUDACC__ for some reason when generating dependency files.  This
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:set(CUDACC_DEFINE -D__CUDACC__)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  COMMAND "${CUDA_NVCC_EXECUTABLE}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  ${CUDACC_DEFINE}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  ${depends_CUDA_NVCC_FLAGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  ${CUDA_NVCC_INCLUDE_ARGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:if(CUDA_result)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  -P "${CUDA_make2cmake}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:if(CUDA_result)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:if(CUDA_result)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:if(CUDA_result)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  COMMAND "${CUDA_NVCC_EXECUTABLE}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  ${cuda_language_flag}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  ${CUDA_NVCC_FLAGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  ${CUDA_NVCC_INCLUDE_ARGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:if(CUDA_result)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:    COMMAND "${CUDA_NVCC_EXECUTABLE}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:    ${CUDA_NVCC_FLAGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:    ${CUDA_NVCC_INCLUDE_ARGS}
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:  cuda_execute_process(
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake:    -P "${CUDA_parse_cubin}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMAKE_SOURCE_DIR = /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMAKE_BINARY_DIR = /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:include CMakeFiles/SiddonGpu.dir/depend.make
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:include CMakeFiles/SiddonGpu.dir/progress.make
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:include CMakeFiles/SiddonGpu.dir/flags.make
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: ../SiddonLib/siddon_class.cu
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: ../SiddonLib/siddon_class.cuh
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/_G_config.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/alloca.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/assert.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/backward/binders.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/allocator.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/atomic_lockfree_defines.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/basic_ios.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/basic_ios.tcc
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/basic_string.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/basic_string.tcc
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/char_traits.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/concept_check.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/cpp_type_traits.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/cxxabi_forced.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/exception_defines.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/functexcept.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/ios_base.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/istream.tcc
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/locale_classes.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/locale_classes.tcc
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/locale_facets.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/locale_facets.tcc
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/localefwd.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/memoryfwd.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/move.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/ostream.tcc
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/ostream_insert.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/postypes.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/predefined_ops.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/ptr_traits.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/range_access.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/stl_algobase.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/stl_function.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/stl_iterator.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/stl_iterator_base_funcs.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/stl_iterator_base_types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/stl_pair.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/streambuf.tcc
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/streambuf_iterator.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/bits/stringfwd.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/cctype
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/clocale
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/cmath
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/cstdlib
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/cwchar
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/cwctype
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/debug/debug.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/exception
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/ext/alloc_traits.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/ext/atomicity.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/ext/new_allocator.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/ext/numeric_traits.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/ext/type_traits.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/ios
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/iosfwd
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/iostream
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/istream
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/new
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/ostream
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/stdexcept
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/streambuf
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/c++/5/string
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/ctype.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/endian.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/features.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/libio.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/limits.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/linux/limits.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/locale.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/math.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/pthread.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/sched.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/stdc-predef.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/stdio.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/stdlib.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/string.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/time.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/wchar.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/wctype.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/byteswap-16.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/byteswap.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/endian.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/huge_val.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/huge_valf.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/huge_vall.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/inf.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/libm-simd-decl-stubs.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/local_lim.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/locale.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/math-vector.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/mathcalls.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/mathdef.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/mathinline.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/nan.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/posix1_lim.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/posix2_lim.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/pthreadtypes.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/sched.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/select.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/select2.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/setjmp.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/sigset.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/stdio.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/stdio2.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/stdio_lim.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/stdlib-bsearch.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/stdlib-float.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/stdlib.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/string3.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/sys_errlist.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/time.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/timex.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/typesizes.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/waitflags.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/waitstatus.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/wchar.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/wchar2.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/wordsize.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/bits/xopen_lim.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/atomic_word.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/c++allocator.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/c++config.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/c++locale.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/cpu_defines.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/ctype_base.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/ctype_inline.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/gthr-default.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/gthr.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/os_defines.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/gnu/stubs-64.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/gnu/stubs.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/sys/cdefs.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/sys/select.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/sys/sysmacros.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/x86_64-linux-gnu/sys/types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/include/xlocale.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/lib/gcc/x86_64-linux-gnu/5/include-fixed/limits.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/lib/gcc/x86_64-linux-gnu/5/include-fixed/syslimits.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/lib/gcc/x86_64-linux-gnu/5/include/stdarg.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/builtin_types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/channel_descriptor.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/common_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/device_double_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/device_double_functions.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/device_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/device_functions.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/host_config.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/host_defines.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/math_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/math_functions.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/sm_70_rt.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/crt/sm_70_rt.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/cuda_device_runtime_api.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/cuda_runtime.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/cuda_runtime_api.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/cuda_surface_types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/cuda_texture_types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/device_atomic_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/device_atomic_functions.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/device_launch_parameters.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/device_types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/driver_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/driver_types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/library_types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_20_atomic_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_20_atomic_functions.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_20_intrinsics.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_20_intrinsics.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_30_intrinsics.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_30_intrinsics.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_32_atomic_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_32_atomic_functions.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_32_intrinsics.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_32_intrinsics.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_35_atomic_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_35_intrinsics.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_60_atomic_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_60_atomic_functions.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_61_intrinsics.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/sm_61_intrinsics.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/surface_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/surface_indirect_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/surface_types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/texture_fetch_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/texture_indirect_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/texture_types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/vector_functions.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/vector_functions.hpp
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: /usr/local/cuda/include/vector_types.h
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o: ../SiddonLib/siddon_class.cu
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:	cd /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib && /usr/local/bin/cmake -E make_directory /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:	cd /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/./SiddonGpu_generated_siddon_class.cu.o -D generated_cubin_file:STRING=/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/./SiddonGpu_generated_siddon_class.cu.o.cubin.txt -P /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.cmake
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:# Object files for target SiddonGpu
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:SiddonGpu_OBJECTS =
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:# External object files for target SiddonGpu
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:SiddonGpu_EXTERNAL_OBJECTS = \
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:"/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:libSiddonGpu.a: CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:libSiddonGpu.a: CMakeFiles/SiddonGpu.dir/build.make
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:libSiddonGpu.a: CMakeFiles/SiddonGpu.dir/link.txt
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libSiddonGpu.a"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:	$(CMAKE_COMMAND) -P CMakeFiles/SiddonGpu.dir/cmake_clean_target.cmake
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SiddonGpu.dir/link.txt --verbose=$(VERBOSE)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/build: libSiddonGpu.a
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:.PHONY : CMakeFiles/SiddonGpu.dir/build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/clean:
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:	$(CMAKE_COMMAND) -P CMakeFiles/SiddonGpu.dir/cmake_clean.cmake
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:.PHONY : CMakeFiles/SiddonGpu.dir/clean
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:CMakeFiles/SiddonGpu.dir/depend: CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:	cd /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/DependInfo.cmake --color=$(COLOR)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/build.make:.PHONY : CMakeFiles/SiddonGpu.dir/depend
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/cmake_clean.cmake:  "CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/cmake_clean.cmake:  "libSiddonGpu.a"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/cmake_clean.cmake:  "libSiddonGpu.pdb"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/cmake_clean.cmake:  include(CMakeFiles/SiddonGpu.dir/cmake_clean_${lang}.cmake OPTIONAL)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/link.txt:/usr/bin/ar qc libSiddonGpu.a  CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/link.txt:/usr/bin/ranlib libSiddonGpu.a
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:# For build in directory: /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CMAKE_PROJECT_NAME:STATIC=SiddonGpu
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_64_BIT_DEVICE_CODE:BOOL=ON
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://Attach the build rule to the CUDA source file.  Enable only when
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:// the CUDA source file is added to at most one target.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE:BOOL=ON
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_BUILD_CUBIN:BOOL=OFF
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_BUILD_EMULATION:BOOL=OFF
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://"cudart" library
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_CUDART_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcudart.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://"cuda" library (older versions only).
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_CUDA_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libcuda.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_GENERATED_OUTPUT_DIR:PATH=
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_HOST_COMPILATION_CPP:BOOL=ON
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_HOST_COMPILER:FILEPATH=/usr/bin/cc
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_EXECUTABLE:FILEPATH=/usr/local/cuda/bin/nvcc
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_FLAGS:STRING=
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_FLAGS_DEBUG:STRING=
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_FLAGS_MINSIZEREL:STRING=
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_FLAGS_RELEASE:STRING=
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_FLAGS_RELWITHDEBINFO:STRING=
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://"OpenCL" library
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_OpenCL_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libOpenCL.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_PROPAGATE_HOST_FLAGS:BOOL=ON
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_SDK_ROOT_DIR:PATH=CUDA_SDK_ROOT_DIR-NOTFOUND
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://Compile CUDA objects with separable compilation enabled.  Requires
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:// CUDA 5.0+
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_SEPARABLE_COMPILATION:BOOL=OFF
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_TOOLKIT_INCLUDE:PATH=/usr/local/cuda/include
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_TOOLKIT_ROOT_DIR:PATH=/usr/local/cuda
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://Use the static version of the CUDA runtime library if available
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_USE_STATIC_CUDA_RUNTIME:BOOL=ON
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://Print out the commands run while compiling the CUDA source file.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_VERBOSE_BUILD:BOOL=OFF
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://Version of CUDA as computed from nvcc.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_VERSION:STRING=10.2
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cublas_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libcublas.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://"cudadevrt" library
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cudadevrt_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcudadevrt.a
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://static CUDA runtime library
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cudart_static_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcudart_static.a
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cufft_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcufft.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cupti_LIBRARY:FILEPATH=/usr/local/cuda/extras/CUPTI/lib64/libcupti.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_curand_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcurand.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cusolver_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcusolver.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cusparse_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcusparse.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppc_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppc.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppial_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppial.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppicc_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppicc.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppicom_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppicom.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppidei_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppidei.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppif_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppif.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppig_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppig.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppim_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppim.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppist_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppist.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppisu_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppisu.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppitc_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppitc.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_npps_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnpps.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nvToolsExt_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnvToolsExt.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_rt_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/librt.so
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:SiddonGpu_BINARY_DIR:STATIC=/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:SiddonGpu_LIB_DEPENDS:STATIC=general;/usr/local/cuda/lib64/libcudart_static.a;general;pthread;general;dl;general;/usr/lib/x86_64-linux-gnu/librt.so;general;cudart;general;cudadevrt;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:SiddonGpu_SOURCE_DIR:STATIC=/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CMAKE_CACHEFILE_DIR:INTERNAL=/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CMAKE_HOME_DIRECTORY:INTERNAL=/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_64_BIT_DEVICE_CODE
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_64_BIT_DEVICE_CODE-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://List of intermediate files that are part of the cuda dependency
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_ADDITIONAL_CLEAN_FILES:INTERNAL=/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeFiles/SiddonGpu.dir/SiddonLib/SiddonGpu_generated_siddon_class.cu.o.depend
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_BUILD_CUBIN
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_BUILD_CUBIN-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_BUILD_EMULATION
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_BUILD_EMULATION-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_CUDART_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_CUDART_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_CUDA_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_CUDA_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_GENERATED_OUTPUT_DIR
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_GENERATED_OUTPUT_DIR-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_HOST_COMPILATION_CPP
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_HOST_COMPILATION_CPP-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_NVCC_EXECUTABLE
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_EXECUTABLE-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_NVCC_FLAGS
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_FLAGS-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_NVCC_FLAGS_DEBUG
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_FLAGS_DEBUG-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_NVCC_FLAGS_MINSIZEREL
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_FLAGS_MINSIZEREL-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_NVCC_FLAGS_RELEASE
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_FLAGS_RELEASE-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_NVCC_FLAGS_RELWITHDEBINFO
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_NVCC_FLAGS_RELWITHDEBINFO-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_OpenCL_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_OpenCL_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_PROPAGATE_HOST_FLAGS
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_PROPAGATE_HOST_FLAGS-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://This is the value of the last time CUDA_SDK_ROOT_DIR was set
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_SDK_ROOT_DIR_INTERNAL:INTERNAL=CUDA_SDK_ROOT_DIR-NOTFOUND
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_SEPARABLE_COMPILATION
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_SEPARABLE_COMPILATION-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_TOOLKIT_INCLUDE
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_TOOLKIT_INCLUDE-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://This is the value of the last time CUDA_TOOLKIT_ROOT_DIR was
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_TOOLKIT_ROOT_DIR_INTERNAL:INTERNAL=/usr/local/cuda
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://This is the value of the last time CUDA_TOOLKIT_TARGET_DIR was
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_TOOLKIT_TARGET_DIR_INTERNAL:INTERNAL=/usr/local/cuda
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_VERBOSE_BUILD
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_VERBOSE_BUILD-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_VERSION
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_VERSION-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_cublas_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cublas_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_cudadevrt_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cudadevrt_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_cudart_static_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cudart_static_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_cufft_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cufft_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_cupti_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cupti_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_curand_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_curand_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_cusolver_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cusolver_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_cusparse_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_cusparse_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_make2cmake:INTERNAL=/usr/local/share/cmake-3.16/Modules/FindCUDA/make2cmake.cmake
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppc_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppc_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppial_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppial_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppicc_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppicc_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppicom_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppicom_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppidei_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppidei_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppif_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppif_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppig_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppig_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppim_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppim_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppist_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppist_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppisu_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppisu_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nppitc_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nppitc_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_npps_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_npps_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://ADVANCED property for variable: CUDA_nvToolsExt_LIBRARY
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_nvToolsExt_LIBRARY-ADVANCED:INTERNAL=1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_parse_cubin:INTERNAL=/usr/local/share/cmake-3.16/Modules/FindCUDA/parse_cubin.cmake
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:CUDA_run_nvcc:INTERNAL=/usr/local/share/cmake-3.16/Modules/FindCUDA/run_nvcc.cmake
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt://Details about finding CUDA
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/CMakeCache.txt:FIND_PACKAGE_MESSAGE_DETAILS_CUDA:INTERNAL=[/usr/local/cuda][/usr/local/cuda/bin/nvcc][/usr/local/cuda/include][/usr/local/cuda/lib64/libcudart_static.a][v10.2()]
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/cmake_install.cmake:# Install script for directory: /home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/cmake_install.cmake:file(WRITE "/home/zargus/Zhangpy/COVID19DRRV2.0/GenDRRV2.0/CUDA_DigitallyReconstructedRadiographs/SiddonClassLib/src/build/${CMAKE_INSTALL_MANIFEST}"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/setup.py:    ext_modules = [Extension("SiddonGpuPy",
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/setup.py:                             sources=["SiddonGpuPy.pyx"],
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/setup.py:                                             "/usr/local/cuda/lib64/"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/setup.py:                             libraries = ["SiddonGpu", "cudart_static", "rt"],
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/readme.txt:- Initialization of the class loads the CT scan onto the GPU device.
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/readme.txt:make sure the paths are correct in SiddonGpuPy.pyx file
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:#define __PYX_HAVE__SiddonGpuPy
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:#define __PYX_HAVE_API__SiddonGpuPy
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  "SiddonGpuPy.pyx",
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:struct __pyx_obj_11SiddonGpuPy_pySiddonGpu;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:/* "SiddonGpuPy.pyx":26
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp: * cdef class pySiddonGpu :             # <<<<<<<<<<<<<<
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp: *     cdef SiddonGpu* thisptr # hold a C++ instance
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:struct __pyx_obj_11SiddonGpuPy_pySiddonGpu {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  SiddonGpu *thisptr;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:/* Module declarations from 'SiddonGpuPy' */
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyTypeObject *__pyx_ptype_11SiddonGpuPy_pySiddonGpu = 0;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:#define __Pyx_MODULE_NAME "SiddonGpuPy"
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:extern int __pyx_module_is_main_SiddonGpuPy;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:int __pyx_module_is_main_SiddonGpuPy = 0;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:/* Implementation of 'SiddonGpuPy' */
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static const char __pyx_k_pySiddonGpu[] = "pySiddonGpu";
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_n_s_pySiddonGpu;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static int __pyx_pf_11SiddonGpuPy_11pySiddonGpu___cinit__(struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *__pyx_v_self, PyArrayObject *__pyx_v_NumThreadsPerBlock, PyArrayObject *__pyx_v_movImgArray, PyArrayObject *__pyx_v_movMaskArray, PyArrayObject *__pyx_v_Weights, PyArrayObject *__pyx_v_MovSize, PyArrayObject *__pyx_v_MovSpacing, PyObject *__pyx_v_X0, PyObject *__pyx_v_Y0, PyObject *__pyx_v_Z0, PyArrayObject *__pyx_v_DRRsize); /* proto */
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pf_11SiddonGpuPy_11pySiddonGpu_2generateDRR(struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *__pyx_v_self, PyArrayObject *__pyx_v_source, PyArrayObject *__pyx_v_DestArray); /* proto */
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pf_11SiddonGpuPy_11pySiddonGpu_4delete(struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *__pyx_v_self); /* proto */
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pf_11SiddonGpuPy_11pySiddonGpu_6__reduce_cython__(CYTHON_UNUSED struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *__pyx_v_self); /* proto */
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pf_11SiddonGpuPy_11pySiddonGpu_8__setstate_cython__(CYTHON_UNUSED struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *__pyx_v_self, CYTHON_UNUSED PyObject *__pyx_v___pyx_state); /* proto */
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_tp_new_11SiddonGpuPy_pySiddonGpu(PyTypeObject *t, PyObject *a, PyObject *k); /*proto*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:/* "SiddonGpuPy.pyx":29
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp: *     cdef SiddonGpu* thisptr # hold a C++ instance
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static int __pyx_pw_11SiddonGpuPy_11pySiddonGpu_1__cinit__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static int __pyx_pw_11SiddonGpuPy_11pySiddonGpu_1__cinit__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __Pyx_AddTraceback("SiddonGpuPy.pySiddonGpu.__cinit__", __pyx_clineno, __pyx_lineno, __pyx_filename);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_r = __pyx_pf_11SiddonGpuPy_11pySiddonGpu___cinit__(((struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *)__pyx_v_self), __pyx_v_NumThreadsPerBlock, __pyx_v_movImgArray, __pyx_v_movMaskArray, __pyx_v_Weights, __pyx_v_MovSize, __pyx_v_MovSpacing, __pyx_v_X0, __pyx_v_Y0, __pyx_v_Z0, __pyx_v_DRRsize);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static int __pyx_pf_11SiddonGpuPy_11pySiddonGpu___cinit__(struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *__pyx_v_self, PyArrayObject *__pyx_v_NumThreadsPerBlock, PyArrayObject *__pyx_v_movImgArray, PyArrayObject *__pyx_v_movMaskArray, PyArrayObject *__pyx_v_Weights, PyArrayObject *__pyx_v_MovSize, PyArrayObject *__pyx_v_MovSpacing, PyObject *__pyx_v_X0, PyObject *__pyx_v_Y0, PyObject *__pyx_v_Z0, PyArrayObject *__pyx_v_DRRsize) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":38
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp: *         self.thisptr = new SiddonGpu(&NumThreadsPerBlock[0],
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":39
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp: *         self.thisptr = new SiddonGpu(&NumThreadsPerBlock[0],             # <<<<<<<<<<<<<<
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":40
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp: *         self.thisptr = new SiddonGpu(&NumThreadsPerBlock[0],
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":41
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp: *         self.thisptr = new SiddonGpu(&NumThreadsPerBlock[0],
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":42
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":43
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":44
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":45
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":46
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":39
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp: *         self.thisptr = new SiddonGpu(&NumThreadsPerBlock[0],             # <<<<<<<<<<<<<<
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_v_self->thisptr = new SiddonGpu((&(*__Pyx_BufPtrCContig1d(int *, __pyx_pybuffernd_NumThreadsPerBlock.rcbuffer->pybuffer.buf, __pyx_t_1, __pyx_pybuffernd_NumThreadsPerBlock.diminfo[0].strides))), (&(*__Pyx_BufPtrCContig1d(float *, __pyx_pybuffernd_movImgArray.rcbuffer->pybuffer.buf, __pyx_t_3, __pyx_pybuffernd_movImgArray.diminfo[0].strides))), (&(*__Pyx_BufPtrCContig1d(float *, __pyx_pybuffernd_movMaskArray.rcbuffer->pybuffer.buf, __pyx_t_4, __pyx_pybuffernd_movMaskArray.diminfo[0].strides))), (&(*__Pyx_BufPtrCContig1d(float *, __pyx_pybuffernd_Weights.rcbuffer->pybuffer.buf, __pyx_t_5, __pyx_pybuffernd_Weights.diminfo[0].strides))), (&(*__Pyx_BufPtrCContig1d(int *, __pyx_pybuffernd_MovSize.rcbuffer->pybuffer.buf, __pyx_t_6, __pyx_pybuffernd_MovSize.diminfo[0].strides))), (&(*__Pyx_BufPtrCContig1d(float *, __pyx_pybuffernd_MovSpacing.rcbuffer->pybuffer.buf, __pyx_t_7, __pyx_pybuffernd_MovSpacing.diminfo[0].strides))), __pyx_t_8, __pyx_t_9, __pyx_t_10, (&(*__Pyx_BufPtrCContig1d(int *, __pyx_pybuffernd_DRRsize.rcbuffer->pybuffer.buf, __pyx_t_11, __pyx_pybuffernd_DRRsize.diminfo[0].strides))));
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":29
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp: *     cdef SiddonGpu* thisptr # hold a C++ instance
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __Pyx_AddTraceback("SiddonGpuPy.pySiddonGpu.__cinit__", __pyx_clineno, __pyx_lineno, __pyx_filename);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:/* "SiddonGpuPy.pyx":48
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pw_11SiddonGpuPy_11pySiddonGpu_3generateDRR(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pw_11SiddonGpuPy_11pySiddonGpu_3generateDRR(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __Pyx_AddTraceback("SiddonGpuPy.pySiddonGpu.generateDRR", __pyx_clineno, __pyx_lineno, __pyx_filename);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_r = __pyx_pf_11SiddonGpuPy_11pySiddonGpu_2generateDRR(((struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *)__pyx_v_self), __pyx_v_source, __pyx_v_DestArray);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pf_11SiddonGpuPy_11pySiddonGpu_2generateDRR(struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *__pyx_v_self, PyArrayObject *__pyx_v_source, PyArrayObject *__pyx_v_DestArray) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":52
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":53
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":54
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":56
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":57
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":59
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":60
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":62
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":63
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":65
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":67
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":48
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __Pyx_AddTraceback("SiddonGpuPy.pySiddonGpu.generateDRR", __pyx_clineno, __pyx_lineno, __pyx_filename);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:/* "SiddonGpuPy.pyx":69
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pw_11SiddonGpuPy_11pySiddonGpu_5delete(PyObject *__pyx_v_self, CYTHON_UNUSED PyObject *unused); /*proto*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pw_11SiddonGpuPy_11pySiddonGpu_5delete(PyObject *__pyx_v_self, CYTHON_UNUSED PyObject *unused) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_r = __pyx_pf_11SiddonGpuPy_11pySiddonGpu_4delete(((struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *)__pyx_v_self));
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pf_11SiddonGpuPy_11pySiddonGpu_4delete(struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *__pyx_v_self) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":70
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:    /* "SiddonGpuPy.pyx":72
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:    /* "SiddonGpuPy.pyx":70
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":69
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pw_11SiddonGpuPy_11pySiddonGpu_7__reduce_cython__(PyObject *__pyx_v_self, CYTHON_UNUSED PyObject *unused); /*proto*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pw_11SiddonGpuPy_11pySiddonGpu_7__reduce_cython__(PyObject *__pyx_v_self, CYTHON_UNUSED PyObject *unused) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_r = __pyx_pf_11SiddonGpuPy_11pySiddonGpu_6__reduce_cython__(((struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *)__pyx_v_self));
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pf_11SiddonGpuPy_11pySiddonGpu_6__reduce_cython__(CYTHON_UNUSED struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *__pyx_v_self) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __Pyx_AddTraceback("SiddonGpuPy.pySiddonGpu.__reduce_cython__", __pyx_clineno, __pyx_lineno, __pyx_filename);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pw_11SiddonGpuPy_11pySiddonGpu_9__setstate_cython__(PyObject *__pyx_v_self, PyObject *__pyx_v___pyx_state); /*proto*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pw_11SiddonGpuPy_11pySiddonGpu_9__setstate_cython__(PyObject *__pyx_v_self, PyObject *__pyx_v___pyx_state) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_r = __pyx_pf_11SiddonGpuPy_11pySiddonGpu_8__setstate_cython__(((struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *)__pyx_v_self), ((PyObject *)__pyx_v___pyx_state));
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_pf_11SiddonGpuPy_11pySiddonGpu_8__setstate_cython__(CYTHON_UNUSED struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *__pyx_v_self, CYTHON_UNUSED PyObject *__pyx_v___pyx_state) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __Pyx_AddTraceback("SiddonGpuPy.pySiddonGpu.__setstate_cython__", __pyx_clineno, __pyx_lineno, __pyx_filename);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyObject *__pyx_tp_new_11SiddonGpuPy_pySiddonGpu(PyTypeObject *t, PyObject *a, PyObject *k) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *p;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  p = ((struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *)o);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  if (unlikely(__pyx_pw_11SiddonGpuPy_11pySiddonGpu_1__cinit__(o, a, k) < 0)) goto bad;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static void __pyx_tp_dealloc_11SiddonGpuPy_pySiddonGpu(PyObject *o) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *p = (struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *)o;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static int __pyx_tp_traverse_11SiddonGpuPy_pySiddonGpu(PyObject *o, visitproc v, void *a) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *p = (struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *)o;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static int __pyx_tp_clear_11SiddonGpuPy_pySiddonGpu(PyObject *o) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *p = (struct __pyx_obj_11SiddonGpuPy_pySiddonGpu *)o;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyMethodDef __pyx_methods_11SiddonGpuPy_pySiddonGpu[] = {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  {"generateDRR", (PyCFunction)(void*)(PyCFunctionWithKeywords)__pyx_pw_11SiddonGpuPy_11pySiddonGpu_3generateDRR, METH_VARARGS|METH_KEYWORDS, 0},
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  {"delete", (PyCFunction)__pyx_pw_11SiddonGpuPy_11pySiddonGpu_5delete, METH_NOARGS, 0},
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  {"__reduce_cython__", (PyCFunction)__pyx_pw_11SiddonGpuPy_11pySiddonGpu_7__reduce_cython__, METH_NOARGS, 0},
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  {"__setstate_cython__", (PyCFunction)__pyx_pw_11SiddonGpuPy_11pySiddonGpu_9__setstate_cython__, METH_O, 0},
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static PyTypeObject __pyx_type_11SiddonGpuPy_pySiddonGpu = {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  "SiddonGpuPy.pySiddonGpu", /*tp_name*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  sizeof(struct __pyx_obj_11SiddonGpuPy_pySiddonGpu), /*tp_basicsize*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_tp_dealloc_11SiddonGpuPy_pySiddonGpu, /*tp_dealloc*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_tp_traverse_11SiddonGpuPy_pySiddonGpu, /*tp_traverse*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_tp_clear_11SiddonGpuPy_pySiddonGpu, /*tp_clear*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_methods_11SiddonGpuPy_pySiddonGpu, /*tp_methods*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_tp_new_11SiddonGpuPy_pySiddonGpu, /*tp_new*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  "SiddonGpuPy.array", /*tp_name*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  "SiddonGpuPy.Enum", /*tp_name*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  "SiddonGpuPy.memoryview", /*tp_name*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  "SiddonGpuPy._memoryviewslice", /*tp_name*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static int __pyx_pymod_exec_SiddonGpuPy(PyObject* module); /*proto*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  {Py_mod_exec, (void*)__pyx_pymod_exec_SiddonGpuPy},
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:    "SiddonGpuPy",
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  {&__pyx_n_s_pySiddonGpu, __pyx_k_pySiddonGpu, sizeof(__pyx_k_pySiddonGpu), 0, 0, 1, 1},
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  if (PyType_Ready(&__pyx_type_11SiddonGpuPy_pySiddonGpu) < 0) __PYX_ERR(1, 26, __pyx_L1_error)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_type_11SiddonGpuPy_pySiddonGpu.tp_print = 0;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  if ((CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP) && likely(!__pyx_type_11SiddonGpuPy_pySiddonGpu.tp_dictoffset && __pyx_type_11SiddonGpuPy_pySiddonGpu.tp_getattro == PyObject_GenericGetAttr)) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:    __pyx_type_11SiddonGpuPy_pySiddonGpu.tp_getattro = __Pyx_PyObject_GenericGetAttr;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  if (PyObject_SetAttr(__pyx_m, __pyx_n_s_pySiddonGpu, (PyObject *)&__pyx_type_11SiddonGpuPy_pySiddonGpu) < 0) __PYX_ERR(1, 26, __pyx_L1_error)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  if (__Pyx_setup_reduce((PyObject*)&__pyx_type_11SiddonGpuPy_pySiddonGpu) < 0) __PYX_ERR(1, 26, __pyx_L1_error)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_ptype_11SiddonGpuPy_pySiddonGpu = &__pyx_type_11SiddonGpuPy_pySiddonGpu;
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:__Pyx_PyMODINIT_FUNC initSiddonGpuPy(void) CYTHON_SMALL_CODE; /*proto*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:__Pyx_PyMODINIT_FUNC initSiddonGpuPy(void)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:__Pyx_PyMODINIT_FUNC PyInit_SiddonGpuPy(void) CYTHON_SMALL_CODE; /*proto*/
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:__Pyx_PyMODINIT_FUNC PyInit_SiddonGpuPy(void)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:static CYTHON_SMALL_CODE int __pyx_pymod_exec_SiddonGpuPy(PyObject *__pyx_pyinit_module)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:    PyErr_SetString(PyExc_RuntimeError, "Module 'SiddonGpuPy' has already been imported. Re-initialisation is not supported.");
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __Pyx_RefNannySetupContext("__Pyx_PyMODINIT_FUNC PyInit_SiddonGpuPy(void)", 0);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  __pyx_m = Py_InitModule4("SiddonGpuPy", __pyx_methods, 0, 0, PYTHON_API_VERSION); Py_XINCREF(__pyx_m);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  if (__pyx_module_is_main_SiddonGpuPy) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:    if (!PyDict_GetItemString(modules, "SiddonGpuPy")) {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:      if (unlikely(PyDict_SetItemString(modules, "SiddonGpuPy", __pyx_m) < 0)) __PYX_ERR(1, 1, __pyx_L1_error)
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":4
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:  /* "SiddonGpuPy.pyx":1
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:      __Pyx_AddTraceback("init SiddonGpuPy", __pyx_clineno, __pyx_lineno, __pyx_filename);
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.cpp:    PyErr_SetString(PyExc_ImportError, "init SiddonGpuPy");
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/include/siddon_class.cuh:* Implementation of a CUDA-based Cpp library for fast DRR generation with GPU acceleration
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/include/siddon_class.cuh:* The class loads a CT scan onto the GPU memory. The function generateDRR can be called multiple times 
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/include/siddon_class.cuh:class SiddonGpu {
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/include/siddon_class.cuh:	SiddonGpu(); // default constructor
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/include/siddon_class.cuh:	SiddonGpu(int *NumThreadsPerBlock,
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/include/siddon_class.cuh:	~SiddonGpu(); // destructor
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.pyx:    cdef cppclass SiddonGpu :
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.pyx:        SiddonGpu()
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.pyx:        SiddonGpu(int *NumThreadsPerBlock,
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.pyx:cdef class pySiddonGpu :
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.pyx:    cdef SiddonGpu* thisptr # hold a C++ instance
InfectionAwareDRRGenerator/CUDA_DigitallyReconstructedRadiographs/SiddonPythonModule/SiddonGpuPy.pyx:        self.thisptr = new SiddonGpu(&NumThreadsPerBlock[0],

```
