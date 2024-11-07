# https://github.com/UrbanRuralSystems/ur-scape

```console
ProjectSettings/ProjectSettings.asset:  gpuSkinning: 0
ProjectSettings/ProjectSettings.asset:  xboxOneDisableKinectGpuReservation: 0
ProjectSettings/ProjectSettings.asset:  ps4GPU800MHz: 1
ProjectSettings/ProjectSettings.asset:  XboxOneEnableGPUVariability: 0
Assets/GUI/Scripts/ColourSelector.cs:			openButton.onClick.AddListener(OnOpenClick);
Assets/GUI/Scripts/ColourSelector.cs:	private void OnOpenClick()
Assets/Map/Scripts/Grid/GridMapLayer.cs:	private bool gpuChangedValues = false;
Assets/Map/Scripts/Grid/GridMapLayer.cs:		if (valuesBuffer != null && gpuChangedValues)
Assets/Map/Scripts/Grid/GridMapLayer.cs:        gpuChangedValues = false;
Assets/Map/Scripts/Grid/GridMapLayer.cs:    public void SetGpuChangedValues()
Assets/Map/Scripts/Grid/GridMapLayer.cs:		gpuChangedValues = true;
Assets/Tools/Contours/Scripts/ContoursMapLayer.cs:			generator = new ContoursGenerator_GPU(this);
Assets/Tools/Contours/Scripts/ContoursGenerator.cs:public class ContoursGenerator_GPU : ContoursGenerator
Assets/Tools/Contours/Scripts/ContoursGenerator.cs:	public ContoursGenerator_GPU(ContoursMapLayer contoursMapLayer) : base(contoursMapLayer)
Assets/Tools/Contours/Scripts/ContoursGenerator.cs:		contoursMapLayer.SetGpuChangedValues();
Assets/Tools/Contours/Scripts/ContoursGenerator.cs:		contoursMapLayer.SetGpuChangedValues();
Assets/Tools/Planning/Models/crop.obj.meta:    optimizeMeshForGPU: 1
Assets/Tools/Planning/Models/High Rises.obj.meta:    optimizeMeshForGPU: 1
Assets/Tools/Planning/Models/Low Rises.obj.meta:    optimizeMeshForGPU: 1
Assets/Tools/Planning/Models/Detached house.obj.meta:    optimizeMeshForGPU: 1
README.md:* GPU with support for DirectX 11 (Windows) or Metal (macOS)

```
