# https://github.com/AshleySpindler/AstroVaDEr-Public

```console
s3_test.py:from keras.utils import multi_gpu_model
s3_test.py:    params['GPUs'] = int(config['training']['GPUs'])
s3_test.py:    if params['GPUs'] > 1:
s3_test.py:        VADE_gpu = multi_gpu_model(VADE, gpus=params['GPUs'])
s3_test.py:        VADE_gpu = VADE
s3_test.py:    x_pred = VADE_gpu.predict(x_test, batch_size=200, verbose=True)
s3_test.py:    x_pred = VADE_gpu.predict(x_test, batch_size=200, verbose=True)
requirements.txt:tensorflow-gpu==1.15.4
s3VDC.py:from keras.utils import multi_gpu_model
s3VDC.py:        GPUs: how many GPUs to use in training
s3VDC.py:    params['GPUs'] = int(config['training']['GPUs'])
s3VDC.py:        # on the GPU memory
s3VDC.py:    if params['GPUs'] > 1:
s3VDC.py:        VADE_gpu = multi_gpu_model(VADE, gpus=params['GPUs'])
s3VDC.py:        VADE_gpu = VADE
s3VDC.py:    VADE_gpu.compile(optimizers.Adam(lr=params['lr']),
s3VDC.py:    warm_up_hist = VADE_gpu.fit_generator(train_generator, epochs=params['warm_up_steps'],
s3VDC.py:    x_pred = VADE_gpu.predict(x_test[0:100])
s3VDC.py:    VADE.get_layer('latentGMM_Layer').trainable = True # affects VAE_gpu
s3VDC.py:    VADE_gpu.compile(optimizers.Adam(lr=params['lr'], clipvalue=1),
s3VDC.py:    annealing_hist = VADE_gpu.fit_generator(train_generator, epochs=total_epochs,
s3VDC.py:    x_pred = VADE_gpu.predict(x_test[0:100])
astrovader-config.txt:GPUs = 2

```
