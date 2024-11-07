# https://github.com/xszhu-lab/AGImpute

```console
AGImpute.py:parser.add_argument('--GPU', action='store_true', default=False, help='Use GPU train')
AGImpute.py:if opt.GPU:
AGImpute.py:    cuda = True if torch.cuda.is_available() else False
AGImpute.py:    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
AGImpute.py:            if torch.cuda.is_available():
AGImpute.py:                automodel.cuda()
AGImpute.py:                print("Autoencoder runing on GPUs")
AGImpute.py:            data_tensor = torch.from_numpy(label_data_li.values).to(torch.float32).cuda()
AGImpute.py:                    data_tensor_noise = torch.from_numpy(add_noise).to(torch.float32).cuda()
AGImpute.py:                    data_tensor, data_tensor_noise = Variable(data_tensor).cuda(), Variable(data_tensor_noise).cuda()
AGImpute.py:                    loss = loss_fn(train_pre[1], data_tensor).cuda()
AGImpute.py:        torch.cuda.empty_cache()
AGImpute.py:        if torch.cuda.is_available():
AGImpute.py:            generator.cuda()
AGImpute.py:            discriminator.cuda()
AGImpute.py:            print("Gan runing on GPUs")
AGImpute.py:                if cuda == True:
AGImpute.py:                torch.cuda.empty_cache()
AGImpute.py:            if cuda == True:
AGImpute.py:            if cuda == True:
AGImpute.py:            de_result = automodel.decoder(tensor_pd_result).cuda()
AGImpute.py:        torch.cuda.empty_cache()
AGImpute.py:                torch.cuda.empty_cache()
AGImpute.py:                torch.cuda.empty_cache()
AGImpute.py:        torch.cuda.empty_cache()
README.md:AGImpute is implemented in `python`(>3.8) and `pytorch`(>10.1) or `cuda`(11.4),Please install `python`(>3.8) and `pytorch`(>10.1) or cuda dependencies before run AGImpute.Users can either use pre-configured conda environment(recommended)or build your own environmen manually.
README.md:    --GPU                       Use GPU for AGImpute.

```
