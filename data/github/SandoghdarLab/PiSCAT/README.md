# https://github.com/SandoghdarLab/PiSCAT

```console
docs/Tutorial_UAI_2/Tutorial_UAI_2.md:We use the FastDVDNet model to extract suitable features for anomaly detection [[1](https://openaccess.thecvf.com/content_CVPR_2020/html/Tassano_FastDVDnet_Towards_Real-Time_Deep_Video_Denoising_Without_Flow_Estimation_CVPR_2020_paper.html)]. The outcome
docs/Tutorial_UAI_2/Tutorial_UAI_2.md:12-30 GB of computer memory (RAM) to run. We also use tensor flow, which can utilize GPU for
docs/Tutorial_UAI_2/Tutorial_UAI_2.md:1. [Tassano, Matias, Julie Delon, and Thomas Veit. "Fastdvdnet: Towards real-time deep video denoising without flow estimation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.](https://openaccess.thecvf.com/content_CVPR_2020/html/Tassano_FastDVDnet_Towards_Real-Time_Deep_Video_Denoising_Without_Flow_Estimation_CVPR_2020_paper.html)
tests/Module/InputOutput/gpu_configurations_test.py:from piscat.InputOutput.gpu_configurations import GPUConfigurations
tests/Module/InputOutput/gpu_configurations_test.py:class TestGPUConfigurations(unittest.TestCase):
tests/Module/InputOutput/gpu_configurations_test.py:        self.file_name = os.path.join(self.dir_name, "gpu_configurations.json")
tests/Module/InputOutput/gpu_configurations_test.py:    def test_save_read_gpu_setting(self):
tests/Module/InputOutput/gpu_configurations_test.py:        test_obj_save = GPUConfigurations()
tests/Module/InputOutput/gpu_configurations_test.py:        test_obj_load = GPUConfigurations(flag_report=True)
tests/Module/InputOutput/gpu_configurations_test.py:            test_obj_save.gpu_active_flag == test_obj_load.gpu_active_flag,
tests/Module/InputOutput/gpu_configurations_test.py:    def test_print_all_available_gpu(self):
tests/Module/InputOutput/gpu_configurations_test.py:        test_obj = GPUConfigurations()
tests/Module/InputOutput/gpu_configurations_test.py:        test_obj.print_all_available_gpu()
piscat/InputOutput/gpu_configurations.py:class GPUConfigurations(PrintColors):
piscat/InputOutput/gpu_configurations.py:    def __init__(self, gpu_device=None, gpu_active_flag=True, flag_report=False):
piscat/InputOutput/gpu_configurations.py:        """This class generates a JSON file for setting on the GPU that the
piscat/InputOutput/gpu_configurations.py:        hyperparameters in a gpu usage.  For parallelization, PiSCAT used
piscat/InputOutput/gpu_configurations.py:        gpu_device: int
piscat/InputOutput/gpu_configurations.py:            Select the GPU device that will be used.
piscat/InputOutput/gpu_configurations.py:        gpu_active_flag: bool
piscat/InputOutput/gpu_configurations.py:            Turn on the GPU version of the code. Otherwise, code is executed on
piscat/InputOutput/gpu_configurations.py:            self.read_gpu_setting(flag_report)
piscat/InputOutput/gpu_configurations.py:            list_gpu_cpu = self.get_available_devices()
piscat/InputOutput/gpu_configurations.py:            for gpu_ in list_gpu_cpu:
piscat/InputOutput/gpu_configurations.py:                if "GPU" in gpu_[0]:
piscat/InputOutput/gpu_configurations.py:                    print(str(gpu_))
piscat/InputOutput/gpu_configurations.py:                self.gpu_active_flag = False
piscat/InputOutput/gpu_configurations.py:                self.gpu_device = None
piscat/InputOutput/gpu_configurations.py:                self.gpu_active_flag = True
piscat/InputOutput/gpu_configurations.py:                if gpu_device is None:
piscat/InputOutput/gpu_configurations.py:                    self.gpu_device = 0
piscat/InputOutput/gpu_configurations.py:                    self.gpu_device = gpu_device
piscat/InputOutput/gpu_configurations.py:            setting_dic = {"gpu_active": [self.gpu_active_flag], "gpu_device": [self.gpu_device]}
piscat/InputOutput/gpu_configurations.py:            self.save_gpu_setting(setting_dic)
piscat/InputOutput/gpu_configurations.py:    def save_gpu_setting(self, setting_dic):
piscat/InputOutput/gpu_configurations.py:        name = "gpu_configurations.json"
piscat/InputOutput/gpu_configurations.py:    def read_gpu_setting(self, flag_report=False):
piscat/InputOutput/gpu_configurations.py:        filepath = os.path.join(here, subdir, "gpu_configurations.json")
piscat/InputOutput/gpu_configurations.py:            gpu_setting = json.load(json_file)
piscat/InputOutput/gpu_configurations.py:        self.gpu_active_flag = gpu_setting["gpu_active"]["0"]
piscat/InputOutput/gpu_configurations.py:        self.gpu_device = gpu_setting["gpu_device"]["0"]
piscat/InputOutput/gpu_configurations.py:            print("PiSCAT's general parallel GPU flag is set to {}".format(self.gpu_active_flag))
piscat/InputOutput/gpu_configurations.py:            print("\nThe code is executed on the GPU device {}.".format(self.gpu_device))
piscat/InputOutput/gpu_configurations.py:            if x.device_type == "GPU" or x.device_type == "CPU"
piscat/InputOutput/gpu_configurations.py:    def print_all_available_gpu(self):
piscat/InputOutput/gpu_configurations.py:        list_gpu_cpu = self.get_available_devices()
piscat/InputOutput/gpu_configurations.py:        for gpu_ in list_gpu_cpu:
piscat/InputOutput/gpu_configurations.py:            if "GPU" in gpu_[0]:
piscat/InputOutput/gpu_configurations.py:                print(str(gpu_[0]) + ", memory:" + str(gpu_[1]))
piscat/InputOutput/gpu_configurations.py:            print(f"{self.WARNING}\nNo GPU found, TensorFlow will use CPU only!!{self.ENDC}")

```
