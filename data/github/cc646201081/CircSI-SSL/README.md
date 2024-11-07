# https://github.com/cc646201081/CircSI-SSL

```console
data_preprocessing/CRBP/getDataView.py:# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
utils.py:    torch.cuda.manual_seed(SEED)
main.py:os.environ['CUDA_VISIBLE_DEVICES'] = "0"
main.py:    parser.add_argument('--device', default='cuda', type=str,
main.py:                        help='cpu or cuda')

```
