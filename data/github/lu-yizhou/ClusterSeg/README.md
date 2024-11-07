# https://github.com/lu-yizhou/ClusterSeg

```console
ClusterSeg/test.py:        image = image.cuda()
ClusterSeg/test.py:    torch.cuda.manual_seed(seed)
ClusterSeg/test.py:    net = ClusterSeg(config, img_size=img_size, num_classes=num_classes, in_channels=3).cuda()
ClusterSeg/Dataset.py:            torch.cuda.manual_seed(seed)
ClusterSeg/Dataset.py:            torch.cuda.manual_seed(seed)
ClusterSeg/Dataset.py:            torch.cuda.manual_seed(seed)
ClusterSeg/main.py:parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
ClusterSeg/main.py:            image, label, bound = image_batch.cuda(), label_batch.cuda(), bound_batch.cuda()
ClusterSeg/main.py:            boundary_gt = (bound != 0).type(torch.int64).cuda()
ClusterSeg/main.py:            cluster_gt = (bound == 2).type(torch.int64).cuda()
ClusterSeg/main.py:            image, label, bound = image.cuda(), label.cuda(), bound.cuda()
ClusterSeg/main.py:            boundary_gt = (bound != 0).long().cuda()
ClusterSeg/main.py:            cluster_gt = (bound == 2).long().cuda()
ClusterSeg/main.py:    net = ClusterSeg(config, img_size=args.img_size, num_classes=args.num_classes, in_channels=args.in_channels).cuda()
PS-CLusterSeg/test.py:parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
PS-CLusterSeg/test.py:parser.add_argument('--labeled_bs', type=int, default=3, help='labeled_batch_size per gpu')
PS-CLusterSeg/test.py:parser.add_argument('--plabeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
PS-CLusterSeg/test.py:parser.add_argument('--unlabeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
PS-CLusterSeg/test.py:parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
PS-CLusterSeg/test.py:os.environ['CUDA_VISIBLE_DEVICES'] = "0"
PS-CLusterSeg/test.py:batch_size = args.batch_size * len(args.gpu.split(','))
PS-CLusterSeg/test.py:    model = ClusterSeg(config, img_size=args.scale, num_classes=num_classes, in_channels=3).cuda()
PS-CLusterSeg/test.py:            image_batch = image_batch.cuda()
PS-CLusterSeg/losses.py:    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(np.log(C)).cuda()
PS-CLusterSeg/losses.py:    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True) / torch.tensor(np.log(C)).cuda()
PS-CLusterSeg/main.py:parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
PS-CLusterSeg/main.py:parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
PS-CLusterSeg/main.py:parser.add_argument('--plabeled_bs', type=int, default=0, help='labeled_batch_size per gpu')
PS-CLusterSeg/main.py:parser.add_argument('--unlabeled_bs', type=int, default=0, help='labeled_batch_size per gpu')
PS-CLusterSeg/main.py:parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
PS-CLusterSeg/main.py:os.environ['CUDA_VISIBLE_DEVICES'] = "0"
PS-CLusterSeg/main.py:batch_size = args.batch_size * len(args.gpu.split(','))
PS-CLusterSeg/main.py:    net_cuda = net.cuda()
PS-CLusterSeg/main.py:        for param in net_cuda.parameters():
PS-CLusterSeg/main.py:    return net_cuda
PS-CLusterSeg/main.py:    RGB = torch.exp(image) / torch.sum(torch.exp(image), dim=1, keepdim=True).cuda()
PS-CLusterSeg/main.py:    mean = torch.zeros((B, C, H, W)).cuda()
PS-CLusterSeg/main.py:            image_batch, label_batch, bound_batch = image_batch.cuda(), label_batch.cuda(), bound_batch.cuda()
PS-CLusterSeg/main.py:            boundary_gt = (bound_batch != 0).cuda()
PS-CLusterSeg/main.py:            cluster_gt = torch.tensor(cluster_gt).cuda()
PS-CLusterSeg/dataset.py:            torch.cuda.manual_seed(seed)
PS-CLusterSeg/dataset.py:            torch.cuda.manual_seed(seed)
PS-CLusterSeg/dataset.py:            torch.cuda.manual_seed(seed)

```
