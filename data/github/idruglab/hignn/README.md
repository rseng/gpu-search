# https://github.com/idrugLab/hignn

```console
source/cross_validate.py:    if torch.cuda.is_available():
source/cross_validate.py:        logger.info('GPU mode...')
source/train.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source/train.py:    if torch.cuda.is_available():
source/train.py:        logger.info('GPU mode...')
source/utils.py:# Set seed for random, numpy, torch, cuda.
source/utils.py:    torch.cuda.manual_seed(seed)
source/utils.py:    torch.cuda.manual_seed_all(seed)
source/utils.py:    torch.cuda.empty_cache()
test/logs/bbbp_seed2021_random_2022-02-23.log:[2022-02-23 15:08:33] (cross_validate.py 172): INFO GPU mode...
test/logs/bbbp_seed2021_random_2022-02-23.log:[2022-02-23 15:11:07] (cross_validate.py 172): INFO GPU mode...

```
