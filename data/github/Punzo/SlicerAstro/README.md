# https://github.com/Punzo/SlicerAstro

```console
AstroVolume/MRML/vtkMRMLAstroSmoothingParametersNode.cxx:      os << indent << "Hardware: GPU\n";
AstroVolume/Widgets/qSlicerAstroScalarVolumeDisplayWidget.cxx:  QObject::connect(this->LogPushButton, SIGNAL(toggled(bool)),
AstroVolume/Widgets/qSlicerAstroScalarVolumeDisplayWidget.cxx:      d->LogPushButton->show();
AstroVolume/Widgets/qSlicerAstroScalarVolumeDisplayWidget.cxx:      d->LogPushButton->blockSignals(true);
AstroVolume/Widgets/qSlicerAstroScalarVolumeDisplayWidget.cxx:        d->LogPushButton->setChecked(true);
AstroVolume/Widgets/qSlicerAstroScalarVolumeDisplayWidget.cxx:        d->LogPushButton->setChecked(false);
AstroVolume/Widgets/qSlicerAstroScalarVolumeDisplayWidget.cxx:      d->LogPushButton->blockSignals(false);
AstroVolume/Widgets/qSlicerAstroScalarVolumeDisplayWidget.cxx:      d->LogPushButton->hide();
AstroVolume/Widgets/Resources/UI/qSlicerAstroScalarVolumeDisplayWidget.ui:      <widget class="QPushButton" name="LogPushButton">
vtkOpenGLFilters/vtkAstroOpenGLImageGaussian.cxx:  // for GPU we do not want threading
vtkOpenGLFilters/vtkAstroOpenGLImageGradient.h:// .NAME vtkAstroOpenGLImageGradient - Compute Box using the GPU
vtkOpenGLFilters/vtkAstroOpenGLImageBox.h:// .NAME vtkAstroOpenGLImageBox - Compute Box using the GPU
vtkOpenGLFilters/vtkAstroOpenGLImageGaussian.h:// .NAME vtkAstroOpenGLImageGaussian - Compute Box using the GPU
vtkOpenGLFilters/vtkAstroOpenGLImageGradient.cxx:  // for GPU we do not want threading
vtkOpenGLFilters/vtkAstroOpenGLImageAlgorithmHelper.h:// .NAME vtkAstroOpenGLImageAlgorithmHelper - Help image algorithms use the GPU
vtkOpenGLFilters/vtkAstroOpenGLImageAlgorithmHelper.h:// Designed to make it easier to accelerate an image algorithm on the GPU
vtkOpenGLFilters/vtkAstroOpenGLImageBox.cxx:  // for GPU we do not want threading
AstroSmoothing/qSlicerAstroSmoothingModule.cxx:         " regarding the GPU (OpenGL) implementation of the filters.";
AstroSmoothing/Resources/UI/qSlicerAstroSmoothingModuleWidget.ui:          <string>GPU</string>
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:/// CPU and GPU hardware, offering interactive performance when processing data-cubes
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:  /// \param vtkRenderWindow to init the GPU algorithm
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:  /// Run box filter algorithm on GPU
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:  /// \param vtkRenderWindow to init the GPU algorithm
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:  int BoxGPUFilter(vtkMRMLAstroSmoothingParametersNode *pnode, vtkRenderWindow* renderWindow);
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:  /// Run Gaussian filter algorithm on GPU
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:  /// \param vtkRenderWindow to init the GPU algorithm
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:  int GaussianGPUFilter(vtkMRMLAstroSmoothingParametersNode *pnode, vtkRenderWindow* renderWindow);
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:  /// Run intensity-driven gradient filter algorithm on GPU
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:  /// \param vtkRenderWindow to init the GPU algorithm
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.h:  int GradientGPUFilter(vtkMRMLAstroSmoothingParametersNode *pnode, vtkRenderWindow* renderWindow);
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:        success = this->BoxGPUFilter(pnode, renderWindow);
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:          success = this->GaussianGPUFilter(pnode, renderWindow);
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:        success = this->GradientGPUFilter(pnode, renderWindow);
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:int vtkSlicerAstroSmoothingLogic::BoxGPUFilter(vtkMRMLAstroSmoothingParametersNode *pnode,
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:  vtkWarningMacro("vtkSlicerAstroSmoothingLogic::BoxGPUFilter "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::BoxGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::BoxGPUFilter :"
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::BoxGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::BoxGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::BoxGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:  vtkDebugMacro(" Box Filter (GPU, OpenGL) Time : "<<mtime<<" ms.");
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:int vtkSlicerAstroSmoothingLogic::GaussianGPUFilter(vtkMRMLAstroSmoothingParametersNode *pnode,
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:  vtkWarningMacro("vtkSlicerAstroSmoothingLogic::GaussianGPUFilter "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::GaussianGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::GaussianGPUFilter :"
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::GaussianGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::GaussianGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::GaussianGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:  vtkDebugMacro("Gaussian Filter (GPU, OpenGL) Time : "<<mtime<<" ms.");
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:int vtkSlicerAstroSmoothingLogic::GradientGPUFilter(vtkMRMLAstroSmoothingParametersNode *pnode,
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:  vtkWarningMacro("vtkSlicerAstroSmoothingLogic::GradientGPUFilter "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::GradientGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::GradientGPUFilter :"
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::GradientGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::GradientGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:    vtkErrorMacro("vtkSlicerAstroSmoothingLogic::GradientGPUFilter : "
AstroSmoothing/Logic/vtkSlicerAstroSmoothingLogic.cxx:  vtkDebugMacro(" Intensity-Driven Gradient Filter (GPU, OpenGL) Time : "<<mtime<<" ms.");

```
