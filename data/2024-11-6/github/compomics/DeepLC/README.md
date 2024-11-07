# https://github.com/compomics/DeepLC

```console
CHANGELOG.md:- Activate garbage collection to clear GPU memory
README.md:**__Q: I have a graphics card, but DeepLC is not using the GPU. Why?__**
README.md:For now DeepLC defaults to the CPU instead of the GPU. Clearly, because you want
README.md:to use the GPU, you are a power user :-). If you want to make the most of that expensive
README.md:GPU, you need to change or remove the following line (at the top) in __deeplc.py__:
README.md:os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
README.md:os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
README.md:Either remove the line or change to (where the number indicates the number of GPUs):
README.md:os.environ['CUDA_VISIBLE_DEVICES'] = '1'
deeplc/deeplc.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
deeplc/deeplc.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
examples/datasets/dia.csv:ANPDPNCCLGVFGLSLYTTER,,5997.0
examples/datasets/dia.csv:ANPDPNCCLGVFGLSLYTTERDLREVFSK,,8340.0
examples/datasets/dia.csv:LEISTAPEIINCCLTPEMLPVKPFPENR,,7128.0
examples/datasets/dia.csv:LSLQNCCLTGAGCGVLSSTLR,,4482.0
examples/datasets/dia.csv:NCCLLEIQETEAK,,2388.0
examples/datasets/dia.csv:RNCCLLEIQETEAK,,1344.0
examples/datasets/dia.csv:TNDINCCLSIK,,1248.0
examples/datasets/LUNA_HILIC.csv:AEDPSNYENVIDIAEQAGK,,31.03
examples/datasets/SCX.csv:AEDPSNYENVIDIAEQAGK,,11
examples/datasets/Xbridge.csv:AEDPSNYENVIDIAEQAGK,,37.85
examples/datasets/unmod.csv:AEDPSNYENVIDIAEQAGK,,13525.32
examples/datasets/LUNA_SILICA.csv:AEDPSNYENVIDIAEQAGK,,29.05
examples/datasets/LUNA_SILICA.csv:GEINCCLK,,29

```
