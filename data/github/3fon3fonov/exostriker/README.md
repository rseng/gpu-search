# https://github.com/3fon3fonov/exostriker

```console
exostriker/lib/pyqtgraph/util/cupy_helper.py:        if os.name == "nt" and cupy.cuda.runtime.runtimeGetVersion() < 11000:
exostriker/lib/pyqtgraph/util/cupy_helper.py:            warn("In Windows, CUDA toolkit should be version 11 or higher, or some functions may misbehave.")
exostriker/lib/pyqtgraph/functions.py:        consistently behaved correctly on windows with cuda toolkit version >= 11.1.
exostriker/lib/pyqtgraph/examples/VideoTemplate.ui:     <widget class="QCheckBox" name="cudaCheck">
exostriker/lib/pyqtgraph/examples/VideoTemplate.ui:       <string>Use CUDA (GPU) if available</string>
exostriker/lib/pyqtgraph/examples/VideoTemplate_generic.py:        self.cudaCheck = QtWidgets.QCheckBox(self.centralwidget)
exostriker/lib/pyqtgraph/examples/VideoTemplate_generic.py:        self.cudaCheck.setObjectName("cudaCheck")
exostriker/lib/pyqtgraph/examples/VideoTemplate_generic.py:        self.gridLayout_2.addWidget(self.cudaCheck, 9, 0, 1, 2)
exostriker/lib/pyqtgraph/examples/VideoTemplate_generic.py:        self.cudaCheck.setText(_translate("MainWindow", "Use CUDA (GPU) if available"))
exostriker/lib/pyqtgraph/examples/VideoSpeedTest.py:parser.add_argument('--cuda', default=False, action='store_true', help="Use CUDA to process on the GPU", dest="cuda")
exostriker/lib/pyqtgraph/examples/VideoSpeedTest.py:ui.cudaCheck.setChecked(args.cuda and _has_cupy)
exostriker/lib/pyqtgraph/examples/VideoSpeedTest.py:ui.cudaCheck.setEnabled(_has_cupy)
exostriker/lib/pyqtgraph/examples/VideoSpeedTest.py:if args.cuda and _has_cupy:
exostriker/lib/pyqtgraph/examples/VideoSpeedTest.py:def noticeCudaCheck():
exostriker/lib/pyqtgraph/examples/VideoSpeedTest.py:    if ui.cudaCheck.isChecked():
exostriker/lib/pyqtgraph/examples/VideoSpeedTest.py:            ui.cudaCheck.setChecked(False)
exostriker/lib/pyqtgraph/examples/VideoSpeedTest.py:ui.cudaCheck.toggled.connect(noticeCudaCheck)
exostriker/lib/pyqtgraph/widgets/RawImageWidget.py:                argb = argb.get()  # transfer GPU data back to the CPU

```
