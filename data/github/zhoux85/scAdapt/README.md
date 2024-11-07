# https://github.com/zhoux85/scAdapt

```console
README.md:	       --gpu_id GPU id to run
scAdapt/loss_utility.py:torch.cuda.manual_seed_all(seed)
scAdapt/loss_utility.py:        semantic_loss = torch.FloatTensor([0.0]).cuda()
scAdapt/loss_utility.py:    zeros = torch.zeros(class_num).cuda()
scAdapt/loss_utility.py:    zeros = torch.zeros(class_num, d).cuda()
scAdapt/example.py:    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
scAdapt/example.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  #'0,1,2,3'
scAdapt/scAdapt.py:torch.cuda.manual_seed_all(seed)
scAdapt/scAdapt.py:    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
scAdapt/scAdapt.py:    base_network = FeatureExtractor(num_inputs=train_set['features'].shape[1], embed_size = embedding_size).cuda()
scAdapt/scAdapt.py:    label_predictor = LabelPredictor(base_network.output_num(), class_num).cuda()
scAdapt/scAdapt.py:    center_loss = CenterLoss(num_classes=class_num, feat_dim=embedding_size, use_gpu=True)
scAdapt/scAdapt.py:    ad_net = scAdversarialNetwork(base_network.output_num(), 1024).cuda()
scAdapt/scAdapt.py:    s_global_centroid = torch.zeros(class_num, embedding_size).cuda()
scAdapt/scAdapt.py:    t_global_centroid = torch.zeros(class_num, embedding_size).cuda()
scAdapt/scAdapt.py:            feature_target = base_network(torch.FloatTensor(test_set['features']).cuda())
scAdapt/scAdapt.py:                code_arr_s = base_network(Variable(torch.FloatTensor(train_set['features']).cuda()))
scAdapt/scAdapt.py:                code_arr_t = base_network(Variable(torch.FloatTensor(test_set_eval['features']).cuda()))
scAdapt/scAdapt.py:        inputs_source, inputs_target, labels_source, labels_target = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda(), labels_target.cuda()
scAdapt/scAdapt.py:            semantic_loss = torch.FloatTensor([0.0]).cuda()
scAdapt/scAdapt.py:            center_loss_src = torch.FloatTensor([0.0]).cuda()
scAdapt/scAdapt.py:            sum_dist_loss = torch.FloatTensor([0.0]).cuda()
scAdapt/scAdapt.py:            lds_loss = torch.FloatTensor([0.0]).cuda()
scAdapt/networks.py:torch.cuda.manual_seed_all(seed)
scAdapt/config.py:parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
scAdapt/config.py:os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # '0,1,2,3'
scAdapt/center_loss.py:torch.cuda.manual_seed_all(seed)
scAdapt/center_loss.py:    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
scAdapt/center_loss.py:        self.use_gpu = use_gpu
scAdapt/center_loss.py:        if self.use_gpu:
scAdapt/center_loss.py:            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
scAdapt/center_loss.py:        if self.use_gpu: classes = classes.cuda()
scAdapt/main.py:    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
scAdapt/main.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  #'0,1,2,3'

```
