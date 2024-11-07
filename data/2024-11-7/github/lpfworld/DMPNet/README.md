# https://github.com/lpfworld/DMPNet

```console
test_one_image.py:    if torch.cuda.is_available():
test_one_image.py:        img = img.cuda()
test_one_image.py:        pretrained_model = pretrained_model.cuda()
test_one_image.py:    if torch.cuda.is_available():
test_one_image.py:    parser.add_argument('--gpu', default='0', help='assign device')
README.md:python test_one_image.py --gpu 0 --model_path pretrained_model_path --test_img_path your_image_path
README.md:python test_dataset.py --gpu 0 --model_path pretrained_model_path --test_img_dir your_image_directory
main.py:            img = img.cuda()
main.py:    net.cuda()
main.py:            img = img.cuda()
main.py:            target = target.unsqueeze(1).cuda()

```
