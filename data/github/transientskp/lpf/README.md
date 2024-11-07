# https://github.com/transientskp/lpf

```console
lpf/_nn/scripts/train.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpf/main.py:        if torch.cuda.is_available():  # type: ignore
lpf/main.py:            logger.info("Running on GPU.")
lpf/main.py:            self.cuda: bool = True
lpf/main.py:            self.cuda: bool = False
lpf/main.py:            "cuda" if self.cuda else "cpu",
lpf/main.py:        if self.cuda:
lpf/main.py:            self.nn.set_device("cuda")
lpf/main.py:        if self.cuda:
lpf/main.py:            images = images.cuda()
lpf/main.py:        if self.cuda:
lpf/main.py:            torch.cuda.synchronize()  # type: ignore

```
