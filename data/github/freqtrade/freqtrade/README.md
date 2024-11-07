# https://github.com/freqtrade/freqtrade

```console
docs/edge.md:    Let's say that you think that the price of *stonecoin* today is 10.0\$. You believe that, because they will start mining stonecoin, it will go up to 15.0\$ tomorrow. There is the risk that the stone is too hard, and the GPUs can't mine it, so the price might go to 0\$ tomorrow. You are planning to invest 100\$, which will give you 10 shares (100 / 10).
docs/freqai-parameter-table.md:| `n_jobs`, `thread_count`, `task_type` | Set the number of threads for parallel processing and the `task_type` (`gpu` or `cpu`). Different model libraries use different parameter names. <br> **Datatype:** Float.
docs/freqai-running.md:During dry/live mode, FreqAI trains each coin pair sequentially (on separate threads/GPU from the main Freqtrade bot). This means that there is always an age discrepancy between models. If you are training on 50 pairs, and each pair requires 5 minutes to train, the oldest model will be over 4 hours old. This may be undesirable if the characteristic time scale (the trade duration target) for a strategy is less than 4 hours. You can decide to only make trade entries if the model is less than a certain number of hours old by setting the `expiration_hours` in the config file:
docs/faq.md:### Why does freqtrade not have GPU support?
docs/faq.md:First of all, most indicator libraries don't have GPU support - as such, there would be little benefit for indicator calculations.
docs/faq.md:The GPU improvements would only apply to pandas-native calculations - or ones written by yourself.
docs/faq.md:Their statement about GPU support is [pretty clear](https://scikit-learn.org/stable/faq.html#will-you-add-gpu-support).
docs/faq.md:GPU's also are only good at crunching numbers (floating point operations).
docs/faq.md:As such, GPU's are not too well suited for most parts of hyperopt.
docs/faq.md:The benefit of using GPU would therefore be pretty slim - and will not justify the complexity introduced by trying to add GPU support.
docs/faq.md:There is however nothing preventing you from using GPU-enabled indicators within your strategy if you think you must have this - you will however probably be disappointed by the slim gain that will give you (compared to the complexity).
docs/freqai-configuration.md:    This docker-compose file also contains a (disabled) section to enable GPU resources within docker containers. This obviously assumes the system has GPU resources available.
docs/freqai-configuration.md:Define our model, loss function, and optimizer, and then move them to the appropriate device (GPU or CPU). Inside the loop, we iterate through the batches in the dataloader, move the data to the device, compute the prediction and loss, backpropagate, and update the model parameters using the optimizer. 
docs/freqai-configuration.md:Torch provides a `torch.compile()` method that can be used to improve performance for specific GPU hardware. More details can be found [here](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). In brief, you simply wrap your `model` in `torch.compile()`:
docs/freqai.md:* **High performance** - Threading allows for adaptive model retraining on a separate thread (or on GPU if available) from model inferencing (prediction) and bot trade operations. Newest models and data are kept in RAM for rapid inferencing
docs/freqai.md:   We do provide an explicit docker-compose file for this in `docker/docker-compose-freqai.yml` - which can be used via `docker compose -f docker/docker-compose-freqai.yml run ...` - or can be copied to replace the original docker file. This docker-compose file also contains a (disabled) section to enable GPU resources within docker containers. This obviously assumes the system has GPU resources available.
docker-compose.yml:    # # Enable GPU Image and GPU Resources (only relevant for freqAI)
docker-compose.yml:    #         - driver: nvidia
docker-compose.yml:    #           capabilities: [gpu]
docker/docker-compose-freqai.yml:    # # Enable GPU Image and GPU Resources
docker/docker-compose-freqai.yml:    #         - driver: nvidia
docker/docker-compose-freqai.yml:    #           capabilities: [gpu]
freqtrade/freqai/torch/PyTorchDataConvertor.py:        :param device: The device to use for training (e.g. 'cpu', 'cuda').
freqtrade/freqai/torch/PyTorchDataConvertor.py:        :param device: The device to use for training (e.g. 'cpu', 'cuda').
freqtrade/freqai/torch/PyTorchModelTrainer.py:        :param device: The device to use for training (e.g. 'cpu', 'cuda').
freqtrade/freqai/base_models/BasePyTorchModel.py:            else ("cuda" if torch.cuda.is_available() else "cpu")

```
