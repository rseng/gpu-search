# https://github.com/nikhilroxtomar/FANet

```console
test.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.py:    x = torch.randn((2, 3, 256, 256)).cuda()
model.py:    m = torch.randn((2, 1, 256, 256)).cuda()
model.py:    model = FANet().cuda()
train.py:    device = torch.device('cuda')
utils.py:    torch.cuda.manual_seed(seed)
blocks.py:        fmask = (self.fmask(x) > 0.5).type(torch.cuda.FloatTensor)
blocks.py:        x1 = x * torch.logical_or(fmask, m).type(torch.cuda.FloatTensor)

```
