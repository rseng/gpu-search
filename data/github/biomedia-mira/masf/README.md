# https://github.com/biomedia-mira/masf

```console
requirements.txt:tensorflow-gpu==2.3.1
main.py:    print('No GPU given... setting to 0')
main.py:    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
main.py:    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])

```
