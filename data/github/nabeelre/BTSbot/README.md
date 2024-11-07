# https://github.com/nabeelre/BTSbot

```console
README.md:I've experienced weird behavior when training and running inference on the GPU cores of my M1 Mac, so we'll disable them here.
README.md:    tf.config.set_visible_devices([], 'GPU')
alert_utils.py:        # Disable GPUs if running on macOS
alert_utils.py:        print("disabling GPUs")
alert_utils.py:        tf.config.set_visible_devices([], 'GPU')
bts_train.py:        # Disable GPUs if running on macOS
bts_train.py:        print("disabling GPUs")
bts_train.py:        tf.config.set_visible_devices([], 'GPU')
bts_val.py:        # Disable GPUs if on darwin (macOS)
bts_val.py:        print("disabling GPUs")
bts_val.py:        tf.config.set_visible_devices([], 'GPU')

```
