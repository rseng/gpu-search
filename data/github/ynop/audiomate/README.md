# https://github.com/ynop/audiomate

```console
audiomate/corpus/io/nvidia_jasper.py:class NvidiaJasperWriter(base.CorpusWriter):
audiomate/corpus/io/nvidia_jasper.py:    Writes files to use for training with NVIDIA Jasper
audiomate/corpus/io/nvidia_jasper.py:    (https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper).
audiomate/corpus/io/nvidia_jasper.py:        return 'nvidia-jasper'
audiomate/corpus/io/__init__.py:from .nvidia_jasper import NvidiaJasperWriter  # noqa: F401
docs/reference/io.rst:NVIDIA Jasper
docs/reference/io.rst:.. autoclass:: NvidiaJasperWriter
docs/notes/changelog.rst:* Added writer (:class:`audiomate.corpus.io.NvidiaJasperWriter`) for
docs/notes/changelog.rst:  `NVIDIA Jasper <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper>`_.
tests/corpus/io/test_nvidia_jasper.py:class TestNvidiaJasperWriter:
tests/corpus/io/test_nvidia_jasper.py:        writer = io.NvidiaJasperWriter()
tests/corpus/io/test_nvidia_jasper.py:        writer = io.NvidiaJasperWriter(export_all_audio=True)
README.md:* [NVIDIA Jasper](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper)

```
