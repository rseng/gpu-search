# https://github.com/R-Vessel-X/SlicerRVXLiverSegmentation

```console
README.md:* Reduce CUDA insufficient memory error by managing sliding window inference devices
RVXLiverSegmentationEffect/RVXLiverSegmentationEffectLib/SegmentEditorEffect.py:    # CPU / CUDA options
RVXLiverSegmentationEffect/RVXLiverSegmentationEffectLib/SegmentEditorEffect.py:    self.device.addItems(["cuda", "cpu"])
RVXLiverSegmentationEffect/RVXLiverSegmentationEffectLib/SegmentEditorEffect.py:      self.logic.launchLiverSegmentation(masterVolumeNode, use_cuda=self.device.currentText == "cuda",
RVXLiverSegmentationEffect/RVXLiverSegmentationEffectLib/SegmentEditorEffect.py:  def launchLiverSegmentation(cls, in_out_volume_node, use_cuda, modality):
RVXLiverSegmentationEffect/RVXLiverSegmentationEffectLib/SegmentEditorEffect.py:    device = torch.device("cpu") if not use_cuda or not torch.cuda.is_available() else torch.device("cuda:0")
RVXLiverSegmentationEffect/RVXLiverSegmentationEffectLib/SegmentEditorEffect.py:      torch.cuda.empty_cache()

```
