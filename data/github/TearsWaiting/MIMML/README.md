# https://github.com/TearsWaiting/MIMML

```console
model_test/main_SL.py:    device = torch.device("cuda" if config.cuda else "cpu")
model_test/main_SL.py:        if config.cuda: model.cuda()
model_test/main_SL.py:    if config.cuda: model.cuda()
model_test/main_SL.py:    torch.cuda.set_device(config.device)
model/Transformer_Encoder.py:        device = torch.device("cuda" if config.cuda else "cpu")
model/Transformer_Encoder.py:    torch.cuda.set_device(config.device)  # 选择要使用的GPU
model/Transformer_Encoder.py:    torch.cuda.set_device(config.device)  # 选择要使用的GPU
model/Transformer_Encoder.py:    if config.cuda:
model/Transformer_Encoder.py:        device = torch.device('cuda')
model/ProtoNet.py:        if self.config.cuda:
model/ProtoNet.py:            reset_label = reset_label.cuda()
model/ProtoNet.py:        device = torch.device('cuda') if self.config.cuda else torch.device('cpu')
model/Focal_Loss.py:        if inputs.is_cuda and not self.alpha.is_cuda:
model/Focal_Loss.py:            self.alpha = self.alpha.cuda()
Framework/DataManager.py:        if self.config.cuda:
Framework/DataManager.py:            self.device = torch.device('cuda')
Framework/DataManager.py:            torch.cuda.set_device(self.config.device)
Framework/DataManager.py:                torch.cuda.manual_seed(self.config.seed)
Framework/DataManager.py:            if self.config.cuda:
Framework/DataManager.py:    def __construct_dataset(self, data, label, cuda=True):
Framework/DataManager.py:        if cuda:
Framework/DataManager.py:            input_ids, labels = torch.cuda.LongTensor(data), torch.cuda.LongTensor(label)
Framework/DataManager.py:        self.train_dataset = self.__construct_dataset(self.train_unified_ids, self.train_label, self.config.cuda)
Framework/DataManager.py:        self.test_dataset = self.__construct_dataset(self.test_unified_ids, self.test_label, self.config.cuda)
Framework/DataManager.py:            if self.config.cuda:
Framework/DataManager.py:                self.support_samples, self.support_labels, self.query_samples, self.query_labels = torch.cuda.LongTensor(
Framework/DataManager.py:                    self.train_unified_ids), torch.cuda.LongTensor(self.train_label), torch.cuda.LongTensor(
Framework/DataManager.py:                    self.test_unified_ids), torch.cuda.LongTensor(self.test_label)
Framework/DataManager.py:            train_dataset = self.__construct_dataset(support_samples, support_labels, self.config.cuda)
Framework/DataManager.py:            test_dataset = self.__construct_dataset(query_samples, query_labels, self.config.cuda)
Framework/DataManager.py:        dataset_train = self.__construct_dataset(train_seqs, train_labels, self.config.cuda)
Framework/DataManager.py:        dataset_test = self.__construct_dataset(test_seqs, test_labels, self.config.cuda)
Framework/DataManager.py:            self.train_dataset = self.__construct_dataset(train_unified_ids, train_label, self.config.cuda)
Framework/DataManager.py:            self.valid_dataset = self.__construct_dataset(valid_unified_ids, valid_label, self.config.cuda)
Framework/ModelManager.py:            if self.config.cuda:
Framework/ModelManager.py:                self.model.cuda()
Framework/ModelManager.py:            if self.config.cuda:
Framework/ModelManager.py:                self.model.cuda()
Framework/ModelManager.py:                    if self.config.cuda:
Framework/ModelManager.py:                        alpha.cuda()
Framework/ModelManager.py:        pretrained_dict = torch.load(param_path, map_location={'cuda:1': 'cuda:0'})
config/config_meta_miniImageNet.py:    parse.add_argument('-cuda', type=bool, default=True)
config/config_default.py:    parse.add_argument('-cuda', type=bool, default=None)
config/config_meta.py:    parse.add_argument('-cuda', type=bool, default=True)
config/config_SL.py:    parse.add_argument('-cuda', type=bool, default=True)
env_test/version_test.py:    if torch.cuda.is_available():
env_test/version_test.py:        torch.cuda.set_device(0)
env_test/version_test.py:        device = torch.device('cuda')
env_test/gpu_test.py:def gpu_test():
env_test/gpu_test.py:    print('GPU is_available', torch.cuda.is_available())
env_test/gpu_test.py:    gpu_num = pynvml.nvmlDeviceGetCount()
env_test/gpu_test.py:    print('gpu num:', gpu_num)
env_test/gpu_test.py:    for i in range(gpu_num):
env_test/gpu_test.py:        print('-' * 50, 'gpu[{}]'.format(str(i)), '-' * 50)
env_test/gpu_test.py:        gpu = pynvml.nvmlDeviceGetHandleByIndex(i)
env_test/gpu_test.py:        print('gpu object:', gpu)
env_test/gpu_test.py:        meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu)
env_test/gpu_test.py:    gpu_test()
data/task_data/Meta Dataset/BPD-DS/Toxin (Venom)/Toxin (Venom) (starPepDB).tsv:544	1	GPSFCKADEKPCEYHADCCNCCLSGICAPSTNWILPGCSTSSFFKI
data/task_data/Meta Dataset/BPD-DS/Cytotoxic/Cytotoxic (starPepDB).tsv:758	1	GKPCHEEGQLCDPFLQNCCLGWNCVFVCI
data/task_data/Meta Dataset/BPD-DS-RT/Toxin (Venom)/(Pos)Toxin (Venom) (starPepDB).tsv:25	1	GPSFCKADEKPCEYHADCCNCCLSGICAPSTNWILPGCSTSSFFKI
data/task_data/Meta Dataset/BPD-ALL-RT/Toxic/(Pos)Toxic (merge).tsv:4627	1	GKPCHEEGQLCDPFLQNCCLGWNCVFVCI
data/task_data/Meta Dataset/BPD-ALL-RT/Toxin (Venom)/(Pos)Toxin (Venom) (starPepDB).tsv:543	1	GPSFCKADEKPCEYHADCCNCCLSGICAPSTNWILPGCSTSSFFKI
data/task_data/Meta Dataset/BPD-ALL-RT/Cytotoxic/(Pos)Cytotoxic (starPepDB).tsv:1679	1	GKPCHEEGQLCDPFLQNCCLGWNCVFVCI
data/task_data/Meta Dataset/BPD-ALL-RT/Toxic to mammals/(Pos)Toxic to mammals (starPepDB).tsv:4620	1	GKPCHEEGQLCDPFLQNCCLGWNCVFVCI
data/task_data/Meta Dataset/BPD-ALL/Antiviral/Antiviral (merge).tsv:3721	1	QEFYGNCCLGHVKPMKIKGKRIESYRMQETDGDCHISAVVFLIKKKPSHVKQKTICANPQEAWVQELMAAVDSRNPKN
data/task_data/Meta Dataset/BPD-ALL/Toxic/Toxic (merge).tsv:6574	1	GKPCHEEGQLCDPFLQNCCLGWNCVFVCI
data/task_data/Meta Dataset/BPD-ALL/Toxin (Venom)/Toxin (Venom) (starPepDB).tsv:424	1	GPSFCKADEKPCEYHADCCNCCLSGICAPSTNWILPGCSTSSFFKI
data/task_data/Meta Dataset/BPD-ALL/Chemotaxis/Chemotaxis (starPepDB).tsv:52	1	QEFYGNCCLGHVKPMKIKGKRIESYRMQETDGDCHISAVVFLIKKKPSHVKQKTICANPQEAWVQELMAAVDSRNPKN
data/task_data/Meta Dataset/BPD-ALL/Antibacterial/Antibacterial (merge).tsv:2837	1	QEFYGNCCLGHVKPMKIKGKRIESYRMQETDGDCHISAVVFLIKKKPSHVKQKTICANPQEAWVQELMAAVDSRNPKN
data/task_data/Meta Dataset/BPD-ALL/Cytotoxic/Cytotoxic (starPepDB).tsv:796	1	GKPCHEEGQLCDPFLQNCCLGWNCVFVCI
data/task_data/Meta Dataset/BPD-ALL/Toxic to mammals/Toxic to mammals (starPepDB).tsv:6568	1	GKPCHEEGQLCDPFLQNCCLGWNCVFVCI
data/task_data/Meta Dataset/BPD-ALL/Antimicrobial/Antimicrobial (starPepDB).tsv:1113	1	QEFYGNCCLGHVKPMKIKGKRIESYRMQETDGDCHISAVVFLIKKKPSHVKQKTICANPQEAWVQELMAAVDSRNPKN
README.md:              [-save-best SAVE_BEST] [-cuda CUDA] [-device DEVICE]
result/default_meta_train/log/2021_10_18_09_00_31.txt:2021-10-18_09:00:43 INFO: Config: Namespace(adapt_iteration=10, adapt_lr=0.0005, alpha=0.1, backbone='TextCNN', cuda=True, dataset='Peptide Sequence', device=0, dim_cnn_out=128, dim_embedding=128, dropout=0.5, epoch=251, filter_sizes='1,2,4,8,16,24,32,64', if_MIM=True, if_transductive=True, lamb=0.1, learn_name='default_meta_train', loss_func='FL', lr=0.0002, max_len=207, meta_batch_size=10, metric='ACC', mode='meta learning', model='ProtoNet', model_save_name='MIMML', num_filter=128, num_meta_test=10, num_meta_train=24, num_meta_valid=10, num_workers=4, optimizer='Adam', path_meta_dataset='../data/task_data/Meta Dataset/BPD-ALL-RT', path_params=None, path_save='../result/', path_token2index='../data/meta_data/residue2idx.pkl', process_name='train (0)', reg=0.0, save_best=True, save_figure_type='png', seed=50, static=False, task_type_run='meta-train', temp=20, test_iteration=100, test_query=15, test_shot=5, test_way=5, threshold=0.6, train_iteration=1, train_query=15, train_shot=5, train_way=5, valid_draw=10, valid_interval=5, valid_iteration=5, valid_query=15, valid_shot=5, valid_start_epoch=300, valid_way=5, vocab_size=28)
result/default_meta_train/log/2021_10_18_09_00_31.txt:2021-10-18_09:15:59 INFO: Config: Namespace(adapt_iteration=10, adapt_lr=0.0005, alpha=0.1, backbone='TextCNN', cuda=True, dataset='Peptide Sequence', device=0, dim_cnn_out=128, dim_embedding=128, dropout=0.5, epoch=251, filter_sizes='1,2,4,8,16,24,32,64', if_MIM=True, if_transductive=True, lamb=0.1, learn_name='default_meta_train', loss_func='FL', lr=0.0002, max_len=207, meta_batch_size=10, metric='ACC', mode='meta learning', model='ProtoNet', model_save_name='MIMML', num_filter=128, num_meta_test=10, num_meta_train=24, num_meta_valid=10, num_workers=4, optimizer='Adam', path_meta_dataset='../data/task_data/Meta Dataset/BPD-ALL-RT', path_params=None, path_save='../result/', path_token2index='../data/meta_data/residue2idx.pkl', process_name='train (0)', reg=0.0, save_best=True, save_figure_type='png', seed=50, static=False, task_type_run='meta-train', temp=20, test_iteration=100, test_query=15, test_shot=5, test_way=5, threshold=0.6, title='MIMML Epoch[250]', train_iteration=1, train_query=15, train_shot=5, train_way=5, valid_draw=10, valid_interval=5, valid_iteration=5, valid_query=15, valid_shot=5, valid_start_epoch=300, valid_way=5, vocab_size=28)
result/default_meta_train/config.txt:cuda: True
result/default_meta_finetune/log/2021_10_18_09_21_03.txt:2021-10-18_09:21:10 INFO: Config: Namespace(alpha=0.01, batch_size=32, cuda=True, dataset='None', device=0, dim_cnn_out=128, dim_embedding=128, dropout=0.5, epoch=100, filter_sizes='1,2,4,8,16,24,32,64', gamma=2, interval_log=20, interval_test=1, interval_valid=1, learn_name='default_meta_finetune', loss_func='FL', lr=0.0005, max_len=207, metric='MCC', mode='train-test', model='TextCNN', model_save_name='CNN', num_class=2, num_filter=128, optimizer='AdamW', output_extend='finetune', path_dataset='../data/task_data/Meta Dataset/BPD-ALL-RT', path_params='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt', path_save='../result/', path_test_data='../data/task_data/Finetune Dataset/Anti-angiogenic Peptide/test/benchmarkdataset-pospos-test.tsv', path_token2index='../data/meta_data/residue2idx.pkl', path_train_data='../data/task_data/Finetune Dataset/Anti-angiogenic Peptide/train/benchmarkdataset-pospos-train.tsv', reg=0.0025, save_best=True, save_figure_type='png', seed=50, static=False, task_type_run='meta-finetune', threshold=0.48, vocab_size=28)
result/default_meta_finetune/log/2021_10_18_09_21_03.txt:2021-10-18_09:23:46 INFO: Config: Namespace(alpha=0.01, batch_size=32, cuda=True, dataset='None', device=0, dim_cnn_out=128, dim_embedding=128, dropout=0.5, epoch=100, filter_sizes='1,2,4,8,16,24,32,64', gamma=2, interval_log=20, interval_test=1, interval_valid=1, learn_name='default_meta_finetune', loss_func='FL', lr=0.0005, max_len=207, metric='MCC', mode='train-test', model='TextCNN', model_save_name='CNN', num_class=2, num_filter=128, optimizer='AdamW', output_extend='finetune', path_dataset='../data/task_data/Meta Dataset/BPD-ALL-RT', path_params='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt', path_save='../result/', path_test_data='../data/task_data/Finetune Dataset/Anti-angiogenic Peptide/test/benchmarkdataset-pospos-test.tsv', path_token2index='../data/meta_data/residue2idx.pkl', path_train_data='../data/task_data/Finetune Dataset/Anti-angiogenic Peptide/train/benchmarkdataset-pospos-train.tsv', reg=0.0025, save_best=True, save_figure_type='png', seed=50, static=False, task_type_run='meta-finetune', threshold=0.48, vocab_size=28)
result/default_meta_finetune/config.txt:cuda: True
result/pretrain_meta_train_BPD_ALL_RT_MIMML/config.txt:cuda: True
result/default_pretrain/log/2021_10_18_08_41_46.txt:2021-10-18_08:41:58 INFO: Config: Namespace(alpha=None, batch_size=320, cuda=True, dataset='None', device=0, dim_cnn_out=128, dim_embedding=128, dropout=0.5, epoch=50, filter_sizes='1,2,4,8,16,24,32,64', gamma=2, interval_log=20, interval_test=1, interval_valid=1, learn_name='default_pretrain', loss_func='FL', lr=0.0005, max_len=207, metric='ACC', mode='train-test', model='TextCNN', model_save_name='CNN', num_class=25, num_filter=128, num_meta_test=10, num_meta_train=24, num_meta_valid=10, optimizer='AdamW', output_extend='pretrain', path_dataset='../data/task_data/Meta Dataset/BPD-ALL-RT', path_params=None, path_save='../result/', path_test_data='../data/task_data/IL-6/Validate.tsv', path_token2index='../data/meta_data/residue2idx.pkl', path_train_data='../data/task_data/IL-6/Train.tsv', reg=0.0025, save_best=True, save_figure_type='png', seed=50, static=False, task_type_run='pretrain', threshold=0.33, vocab_size=28)
result/default_pretrain/log/2021_10_18_08_41_46.txt:2021-10-18_08:59:33 INFO: Config: Namespace(alpha=None, batch_size=320, cuda=True, dataset='None', device=0, dim_cnn_out=128, dim_embedding=128, dropout=0.5, epoch=50, filter_sizes='1,2,4,8,16,24,32,64', gamma=2, interval_log=20, interval_test=1, interval_valid=1, learn_name='default_pretrain', loss_func='FL', lr=0.0005, max_len=207, metric='ACC', mode='train-test', model='TextCNN', model_save_name='CNN', num_class=25, num_filter=128, num_meta_test=10, num_meta_train=24, num_meta_valid=10, optimizer='AdamW', output_extend='pretrain', path_dataset='../data/task_data/Meta Dataset/BPD-ALL-RT', path_params=None, path_save='../result/', path_test_data='../data/task_data/IL-6/Validate.tsv', path_token2index='../data/meta_data/residue2idx.pkl', path_train_data='../data/task_data/IL-6/Train.tsv', reg=0.0025, save_best=True, save_figure_type='png', seed=50, static=False, task_type_run='pretrain', threshold=0.33, vocab_size=28)
result/default_pretrain/config.txt:cuda: True
result/pretrain_BPD_ALL_RT/config.txt:cuda: True
result/default_meta_train_with_pretrain/log/2021_10_18_09_02_04.txt:2021-10-18_09:07:52 INFO: Config: Namespace(adapt_iteration=10, adapt_lr=0.0005, alpha=0.1, backbone='TextCNN', cuda=True, dataset='Peptide Sequence', device=0, dim_cnn_out=128, dim_embedding=128, dropout=0.5, epoch=251, filter_sizes='1,2,4,8,16,24,32,64', if_MIM=True, if_transductive=True, lamb=0.1, learn_name='default_meta_train_with_pretrain', loss_func='FL', lr=0.0002, max_len=207, meta_batch_size=10, metric='ACC', mode='meta learning', model='ProtoNet', model_save_name='MIMML', num_filter=128, num_meta_test=10, num_meta_train=24, num_meta_valid=10, num_workers=4, optimizer='Adam', path_meta_dataset='../data/task_data/Meta Dataset/BPD-ALL-RT', path_params='../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt', path_save='../result/', path_token2index='../data/meta_data/residue2idx.pkl', process_name='train (0)', reg=0.0, save_best=True, save_figure_type='png', seed=50, static=False, task_type_run='meta-test', temp=20, test_iteration=100, test_query=15, test_shot=5, test_way=5, threshold=0.6, train_iteration=1, train_query=15, train_shot=5, train_way=5, valid_draw=10, valid_interval=5, valid_iteration=5, valid_query=15, valid_shot=5, valid_start_epoch=300, valid_way=5, vocab_size=28)
result/default_meta_train_with_pretrain/log/2021_10_18_09_01_02.txt:2021-10-18_09:02:15 INFO: Config: Namespace(adapt_iteration=10, adapt_lr=0.0005, alpha=0.1, backbone='TextCNN', cuda=True, dataset='Peptide Sequence', device=0, dim_cnn_out=128, dim_embedding=128, dropout=0.5, epoch=251, filter_sizes='1,2,4,8,16,24,32,64', if_MIM=True, if_transductive=True, lamb=0.1, learn_name='default_meta_train_with_pretrain', loss_func='FL', lr=0.0002, max_len=207, meta_batch_size=10, metric='ACC', mode='meta learning', model='ProtoNet', model_save_name='MIMML', num_filter=128, num_meta_test=10, num_meta_train=24, num_meta_valid=10, num_workers=4, optimizer='Adam', path_meta_dataset='../data/task_data/Meta Dataset/BPD-ALL-RT', path_params='../result/default_pretrain/model/CNN, Epoch[20.000].pt', path_save='../result/', path_token2index='../data/meta_data/residue2idx.pkl', process_name='train (0)', reg=0.0, save_best=True, save_figure_type='png', seed=50, static=False, task_type_run='meta-train-with-pretrain', temp=20, test_iteration=100, test_query=15, test_shot=5, test_way=5, threshold=0.6, train_iteration=1, train_query=15, train_shot=5, train_way=5, valid_draw=10, valid_interval=5, valid_iteration=5, valid_query=15, valid_shot=5, valid_start_epoch=300, valid_way=5, vocab_size=28)
result/default_meta_train_with_pretrain/log/2021_10_18_09_01_02.txt:2021-10-18_09:18:45 INFO: Config: Namespace(adapt_iteration=10, adapt_lr=0.0005, alpha=0.1, backbone='TextCNN', cuda=True, dataset='Peptide Sequence', device=0, dim_cnn_out=128, dim_embedding=128, dropout=0.5, epoch=251, filter_sizes='1,2,4,8,16,24,32,64', if_MIM=True, if_transductive=True, lamb=0.1, learn_name='default_meta_train_with_pretrain', loss_func='FL', lr=0.0002, max_len=207, meta_batch_size=10, metric='ACC', mode='meta learning', model='ProtoNet', model_save_name='MIMML', num_filter=128, num_meta_test=10, num_meta_train=24, num_meta_valid=10, num_workers=4, optimizer='Adam', path_meta_dataset='../data/task_data/Meta Dataset/BPD-ALL-RT', path_params='../result/default_pretrain/model/CNN, Epoch[20.000].pt', path_save='../result/', path_token2index='../data/meta_data/residue2idx.pkl', process_name='train (0)', reg=0.0, save_best=True, save_figure_type='png', seed=50, static=False, task_type_run='meta-train-with-pretrain', temp=20, test_iteration=100, test_query=15, test_shot=5, test_way=5, threshold=0.6, title='MIMML Epoch[250]', train_iteration=1, train_query=15, train_shot=5, train_way=5, valid_draw=10, valid_interval=5, valid_iteration=5, valid_query=15, valid_shot=5, valid_start_epoch=300, valid_way=5, vocab_size=28)
result/default_meta_train_with_pretrain/config.txt:cuda: True
preprocess/meta_dataset.py:def construct_dataset(data, label, cuda=True):
preprocess/meta_dataset.py:    if cuda:
preprocess/meta_dataset.py:        input_ids, labels = torch.cuda.LongTensor(data), torch.cuda.LongTensor(label)
preprocess/meta_dataset.py:        cuda = False
preprocess/meta_dataset.py:        cuda = True
preprocess/meta_dataset.py:        torch.cuda.set_device(device)
preprocess/meta_dataset.py:    dataset_train = construct_dataset(train_seqs, train_labels, cuda)
preprocess/meta_dataset.py:    dataset_test = construct_dataset(test_seqs, test_labels, cuda)

```
