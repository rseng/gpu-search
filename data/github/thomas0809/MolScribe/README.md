# https://github.com/thomas0809/MolScribe

```console
predict.py:    device = torch.device('cuda')
README.md:The script uses one GPU and batch size of 64 by default. If more GPUs are available, update `NUM_GPUS_PER_NODE` and 
README.md:The script uses four GPUs and batch size of 256 by default. It takes about one day to train the model with four A100 GPUs.
scripts/train_uspto_joint_chartok.sh:NUM_GPUS_PER_NODE=4
scripts/train_uspto_joint_chartok.sh:    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
scripts/train_uspto_joint_chartok.sh:    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
scripts/train_uspto_joint_chartok.sh:    --fp16 --backend nccl 2>&1
scripts/eval_uspto_joint_chartok_1m680k.sh:NUM_GPUS_PER_NODE=1
scripts/eval_uspto_joint_chartok_1m680k.sh:    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
scripts/eval_uspto_joint_chartok_1m680k.sh:    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE)) \
scripts/eval_uspto_joint_chartok.sh:NUM_GPUS_PER_NODE=1
scripts/eval_uspto_joint_chartok.sh:    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
scripts/eval_uspto_joint_chartok.sh:    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE)) \
scripts/eval_uspto_joint_chartok.sh:    --fp16 --backend nccl 2>&1
scripts/train_uspto_joint_chartok_1m680k.sh:NUM_GPUS_PER_NODE=4
scripts/train_uspto_joint_chartok_1m680k.sh:    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
scripts/train_uspto_joint_chartok_1m680k.sh:    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
train.py:    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'])
train.py:        with torch.cuda.amp.autocast(enabled=args.fp16):
train.py:    # Inference is distributed. The batch is divided and run independently on multiple GPUs, and the predictions
train.py:        with torch.cuda.amp.autocast(enabled=args.fp16):
train.py:    # gather predictions from different GPUs
train.py:    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
train.py:    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train.py:        torch.cuda.set_device(args.local_rank)
molscribe/utils.py:    torch.cuda.manual_seed(seed)

```
