# https://github.com/Smith42/pix2prof

```console
train.py:# Check if GPU is available
train.py:if torch.cuda.is_available():
train.py:    print("Using a CUDA compatible device")
train.py:    cuda = torch.device("cuda:0")
train.py:    cuda = torch.device("cpu")
train.py:         galaxy = torch.Tensor(galaxy).to(cuda)
train.py:         profile = torch.Tensor(profile).to(cuda)
train.py:    decoder_input = SOS_TOKEN.to(cuda)
train.py:        decoder_input = SOS_TOKEN.to(cuda)
train.py:    encoder = ResNet18(num_classes=args.encoding_len).to(cuda)
train.py:    decoder = GRUNet(input_dim=1, hidden_dim=args.encoding_len, output_dim=1, n_layers=3).to(cuda)
train.py:                galaxy, profile = galaxy.to(cuda), profile.to(cuda) 
eval.py:# Check machine for GPU
eval.py:if torch.cuda.is_available():
eval.py:    print("Using a CUDA compatible device")
eval.py:    cuda = torch.device("cuda:0")
eval.py:    cuda = torch.device("cpu")
eval.py:    checkpoint = torch.load(args.checkpoint, map_location=cuda)
eval.py:    encoder = ResNet18(num_classes=args.encoding_len).to(cuda)
eval.py:    decoder = GRUNet(input_dim=1, hidden_dim=args.encoding_len, output_dim=1, n_layers=3).to(cuda)

```
