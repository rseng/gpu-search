# https://github.com/Rohit-Kundu/ET-NET_Covid-Detection

```console
utils/utils_cnn.py:    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
utils/utils_cnn.py:              labels=labels.cuda()
utils/utils_cnn.py:              outputs = model(images.cuda())
utils/utils_cnn.py:              correct += (predicted == labels.cuda()).sum().item()
utils/utils_cnn.py:    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
utils/utils_cnn.py:              labels=labels.cuda()
utils/utils_cnn.py:              outputs = model(images.cuda())
utils/utils_cnn.py:              correct += (predicted == labels.cuda()).sum().item()
utils/utils_cnn.py:    net = net.cuda()
utils/utils_cnn.py:            inputs,labels=inputs.cuda(),labels.cuda()
utils/utils_cnn.py:            inputs,labels=inputs.cuda(),labels.cuda()
utils/utils_cnn.py:              labels=labels.cuda()
utils/utils_cnn.py:              outputs = net(images.cuda())
utils/utils_cnn.py:              correct += (predicted == labels.cuda()).sum().item()

```
