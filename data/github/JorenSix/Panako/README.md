# https://github.com/JorenSix/Panako

```console
src/main/java/be/panako/strategy/panako/PanakoGPUEventPointProcessor.java: * Use an external script to extract event points, e.g. using the GPU.
src/main/java/be/panako/strategy/panako/PanakoGPUEventPointProcessor.java:public class PanakoGPUEventPointProcessor {
src/main/java/be/panako/strategy/panako/PanakoGPUEventPointProcessor.java:    public PanakoGPUEventPointProcessor(){
src/main/java/be/panako/strategy/panako/PanakoStrategy.java:		if(!Config.getBoolean(Key.PANAKO_USE_GPU_EP_EXTRACTOR)) {
src/main/java/be/panako/strategy/panako/PanakoStrategy.java:		if(Config.getBoolean(Key.PANAKO_USE_GPU_EP_EXTRACTOR)){
src/main/java/be/panako/strategy/panako/PanakoStrategy.java:			List<PanakoFingerprint> prints = new PanakoGPUEventPointProcessor().extractFingerprints(resource);
src/main/java/be/panako/util/Key.java:	 * Use the default (CPU based JGaborator) Event point extractor or use CUDA/MPS for event point
src/main/java/be/panako/util/Key.java:	PANAKO_USE_GPU_EP_EXTRACTOR("FALSE");

```
